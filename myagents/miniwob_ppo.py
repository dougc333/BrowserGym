#!/usr/bin/env python3
import os
import math
import time
import argparse
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import gymnasium as gym
import browsergym.miniwob  # registers MiniWoB envs


# -----------------------------
# Utilities: hashing text -> vector (no extra deps)
# -----------------------------
def hash_text_to_vec(text: str, dim: int = 256) -> torch.Tensor:
    """
    Very simple hashing trick: map character 3-grams into a fixed dim vector.
    """
    text = (text or "").lower()
    v = torch.zeros(dim, dtype=torch.float32)
    if len(text) < 3:
        return v
    for i in range(len(text) - 2):
        tri = text[i : i + 3]
        h = 0
        for ch in tri:
            h = (h * 131 + ord(ch)) & 0xFFFFFFFF
        idx = h % dim
        v[idx] += 1.0
    # normalize
    v = v / (v.norm(p=2) + 1e-6)
    return v


def obs_to_candidates(obs: Dict[str, Any], max_candidates: int = 64) -> Tuple[List[str], torch.Tensor]:
    """
    Returns:
      - bids: list of clickable bid strings
      - feats: [N, F] tensor of per-bid features

    Uses extra_element_properties[bid] which includes bbox and clickable.
    Feature design:
      - bbox center (cx, cy) normalized by viewport-ish constants
      - bbox size (w, h) normalized
      - visibility
      - clickable flag (should be 1.0)
    """
    extra = obs.get("extra_element_properties")
    if not isinstance(extra, dict):
        return [], torch.zeros((0, 7), dtype=torch.float32)

    bids = []
    feats = []

    # Fallback normalization constants; MiniWoB pages are usually ~1000x700-ish.
    # This doesn't have to be perfect for the demo.
    WN, HN = 1000.0, 700.0

    for bid, props in extra.items():
        if not isinstance(props, dict):
            continue
        if props.get("clickable") is not True:
            continue

        bbox = props.get("bbox")
        if not (isinstance(bbox, (list, tuple)) and len(bbox) == 4):
            # [x0,y0,x1,y1] sometimes absent; still keep a placeholder
            x0 = y0 = x1 = y1 = 0.0
        else:
            x0, y0, x1, y1 = map(float, bbox)

        w = max(0.0, x1 - x0)
        h = max(0.0, y1 - y0)
        cx = x0 + 0.5 * w
        cy = y0 + 0.5 * h

        vis = float(props.get("visibility", 0.0))
        clickable = 1.0

        # Basic per-element feature vector
        f = torch.tensor(
            [
                cx / WN,
                cy / HN,
                w / WN,
                h / HN,
                vis,
                clickable,
                1.0,  # bias-ish constant feature
            ],
            dtype=torch.float32,
        )

        bids.append(str(bid))
        feats.append(f)

        if len(bids) >= max_candidates:
            break

    if not feats:
        return [], torch.zeros((0, 7), dtype=torch.float32)

    return bids, torch.stack(feats, dim=0)


# -----------------------------
# PPO model: scores each candidate bid + value
# -----------------------------
class BidPolicyValue(nn.Module):
    def __init__(self, goal_dim: int = 256, bid_feat_dim: int = 7, hidden: int = 128):
        super().__init__()
        # Combine (goal_embed || bid_feat) -> logit
        self.policy = nn.Sequential(
            nn.Linear(goal_dim + bid_feat_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),  # per-bid logit
        )
        # State value depends only on goal embedding (simple baseline)
        self.value = nn.Sequential(
            nn.Linear(goal_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, goal_vec: torch.Tensor, bid_feats: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        goal_vec: [G]
        bid_feats: [N, F]
        Returns:
          logits: [N]
          value: scalar tensor
        """
        N = bid_feats.shape[0]
        if N == 0:
            return torch.empty((0,), dtype=torch.float32), self.value(goal_vec).squeeze(-1)

        goal_rep = goal_vec.unsqueeze(0).expand(N, -1)  # [N,G]
        x = torch.cat([goal_rep, bid_feats], dim=-1)    # [N,G+F]
        logits = self.policy(x).squeeze(-1)             # [N]
        v = self.value(goal_vec).squeeze(-1)            # []
        return logits, v


# -----------------------------
# Rollout storage
# -----------------------------
@dataclass
class Step:
    goal_vec: torch.Tensor
    bid_feats: torch.Tensor
    action_idx: int
    logp: torch.Tensor
    value: torch.Tensor
    reward: float
    done: bool


def masked_categorical_sample(logits: torch.Tensor) -> Tuple[int, torch.Tensor, torch.Tensor]:
    """
    Sample action from logits (no mask needed if logits is only candidates).
    Returns: (action_idx, logp, entropy)
    """
    dist = torch.distributions.Categorical(logits=logits)
    a = dist.sample()
    logp = dist.log_prob(a)
    ent = dist.entropy()
    return int(a.item()), logp, ent


def compute_gae(steps: List[Step], gamma: float, lam: float, last_value: float = 0.0) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns:
      returns: [T]
      adv: [T]
    """
    T = len(steps)
    rewards = torch.tensor([s.reward for s in steps], dtype=torch.float32)
    dones = torch.tensor([1.0 if s.done else 0.0 for s in steps], dtype=torch.float32)
    values = torch.stack([s.value for s in steps]).detach().float()

    adv = torch.zeros(T, dtype=torch.float32)
    gae = 0.0
    for t in reversed(range(T)):
        next_v = last_value if t == T - 1 else values[t + 1].item()
        delta = rewards[t].item() + gamma * next_v * (1.0 - dones[t].item()) - values[t].item()
        gae = delta + gamma * lam * (1.0 - dones[t].item()) * gae
        adv[t] = gae

    ret = adv + values
    # normalize adv
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    return ret, adv


# -----------------------------
# Main training loop
# -----------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--env-id", default="browsergym/miniwob.click-button")
    p.add_argument("--total-iters", type=int, default=50)
    p.add_argument("--steps-per-iter", type=int, default=32)
    p.add_argument("--ppo-epochs", type=int, default=4)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae-lam", type=float, default=0.95)
    p.add_argument("--clip", type=float, default=0.2)
    p.add_argument("--ent-coef", type=float, default=0.01)
    p.add_argument("--vf-coef", type=float, default=0.5)
    p.add_argument("--max-candidates", type=int, default=64)
    p.add_argument("--headless", action="store_true")
    args = p.parse_args()

    # MiniWoB requires base URL
    if not os.environ.get("MINIWOB_URL"):
        raise RuntimeError(
            "MINIWOB_URL is not set.\n"
            "Example:\n"
            "  export MINIWOB_URL=http://127.0.0.1:8000/miniwob/\n"
            "And run the http.server from miniwob-plusplus/miniwob/html"
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    env = gym.make(args.env_id, headless=args.headless)

    model = BidPolicyValue(goal_dim=256, bid_feat_dim=7, hidden=128).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    obs, info = env.reset()
    ep_return = 0.0
    ep_len = 0
    completed = 0

    for it in range(args.total_iters):
        steps: List[Step] = []
        entropies = []

        # -------- collect rollout --------
        for t in range(args.steps_per_iter):
            goal = obs.get("goal", "")
            goal_vec = hash_text_to_vec(goal, dim=256).to(device)

            bids, bid_feats = obs_to_candidates(obs, max_candidates=args.max_candidates)
            bid_feats = bid_feats.to(device)

            if len(bids) == 0:
                # no clickable candidates, end episode
                obs, info = env.reset()
                ep_return = 0.0
                ep_len = 0
                continue

            logits, value = model(goal_vec, bid_feats)
            a_idx, logp, ent = masked_categorical_sample(logits)
            entropies.append(ent)

            action = {"action": "click", "bid": bids[a_idx]}
            obs2, reward, terminated, truncated, info = env.step(action)
            done = bool(terminated) or bool(truncated)

            steps.append(
                Step(
                    goal_vec=goal_vec.detach(),
                    bid_feats=bid_feats.detach(),
                    action_idx=a_idx,
                    logp=logp.detach(),
                    value=value.detach(),
                    reward=float(reward),
                    done=done,
                )
            )

            ep_return += float(reward)
            ep_len += 1

            obs = obs2

            if done:
                completed += 1
                obs, info = env.reset()
                ep_return = 0.0
                ep_len = 0

        # bootstrap value (simple: 0 if episode boundary; else estimate current)
        with torch.no_grad():
            goal_vec = hash_text_to_vec(obs.get("goal", ""), dim=256).to(device)
            bids, bid_feats = obs_to_candidates(obs, max_candidates=args.max_candidates)
            bid_feats = bid_feats.to(device)
            _, last_v = model(goal_vec, bid_feats)
            last_v = float(last_v.item()) if len(bids) > 0 else 0.0

        returns, adv = compute_gae(steps, gamma=args.gamma, lam=args.gae_lam, last_value=last_v)
        returns = returns.to(device)
        adv = adv.to(device)

        # -------- PPO update --------
        # Flatten into a per-step batch; we recompute logits for each step’s candidate set.
        for epoch in range(args.ppo_epochs):
            policy_loss = 0.0
            value_loss = 0.0
            entropy_loss = 0.0

            for i, s in enumerate(steps):
                goal_vec = s.goal_vec.to(device)
                bid_feats = s.bid_feats.to(device)
                logits, value = model(goal_vec, bid_feats)

                dist = torch.distributions.Categorical(logits=logits)
                new_logp = dist.log_prob(torch.tensor(s.action_idx, device=device))
                ratio = torch.exp(new_logp - s.logp.to(device))

                surr1 = ratio * adv[i]
                surr2 = torch.clamp(ratio, 1.0 - args.clip, 1.0 + args.clip) * adv[i]
                policy_loss = policy_loss + (-torch.min(surr1, surr2))

                v_err = (value - returns[i]).pow(2)
                value_loss = value_loss + v_err

                entropy_loss = entropy_loss + (-dist.entropy())

            policy_loss = policy_loss / len(steps)
            value_loss = value_loss / len(steps)
            entropy_loss = entropy_loss / len(steps)

            loss = policy_loss + args.vf_coef * value_loss + args.ent_coef * entropy_loss

            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        avg_ent = float(torch.stack(entropies).mean().item()) if entropies else 0.0
        avg_reward = float(np.mean([s.reward for s in steps])) if steps else 0.0
        success_rate = float(np.mean([1.0 if s.reward > 0 else 0.0 for s in steps])) if steps else 0.0

        print(
            f"[iter {it:03d}] "
            f"avg_step_reward={avg_reward:.3f} "
            f"pos_reward_frac={success_rate:.2f} "
            f"entropy={avg_ent:.2f} "
            f"episodes_completed={completed}"
        )

    env.close()


if __name__ == "__main__":
    main()