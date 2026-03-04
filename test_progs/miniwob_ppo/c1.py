#!/usr/bin/env python3
"""
MiniWoB PPO (click-by-bid) baseline for BrowserGym.

Key idea:
- Extract clickable element candidates by BID from obs["extra_element_properties"]
- Add BOTH geometry features (bbox center/size/visibility) AND per-element text features
  parsed from the DOM snapshot (flatten_dom_to_str) and hashed into a fixed vector.
- PPO learns to map goal text -> correct element text (e.g., "Click on the 'OK' button").

Deps:
  pip install gymnasium torch numpy browsergym-miniwob playwright
  playwright install  (or: python -m playwright install)

Optional (recommended for best DOM parsing):
  pip install beautifulsoup4 lxml

Runtime requirement:
  export MINIWOB_URL=http://127.0.0.1:8000/miniwob/
  and run an http server that serves MiniWoB HTML + core assets.
"""

import os
import math
import time
import argparse
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple

import numpy as np
import torch
import torch.nn as nn

import gymnasium as gym
import browsergym.miniwob  # registers MiniWoB envs

from browsergym.utils.obs import flatten_dom_to_str  # correct import path

# Optional DOM parsing
try:
    import bs4  # beautifulsoup4
except Exception:
    bs4 = None


# -----------------------------
# Hashing text -> vector (no extra deps)
# -----------------------------
def hash_text_to_vec(text: str, dim: int = 64) -> torch.Tensor:
    """Simple char 3-gram hashing trick -> [dim]."""
    text = (text or "").lower()
    v = torch.zeros(dim, dtype=torch.float32)
    if len(text) < 3:
        return v
    for i in range(len(text) - 2):
        tri = text[i : i + 3]
        h = 0
        for ch in tri:
            h = (h * 131 + ord(ch)) & 0xFFFFFFFF
        v[h % dim] += 1.0
    v = v / (v.norm(p=2) + 1e-6)
    return v


def hash_goal_vec(goal: str, dim: int = 256) -> torch.Tensor:
    """Same trick but bigger dim for goal text."""
    goal = (goal or "").lower()
    v = torch.zeros(dim, dtype=torch.float32)
    if len(goal) < 3:
        return v
    for i in range(len(goal) - 2):
        tri = goal[i : i + 3]
        h = 0
        for ch in tri:
            h = (h * 131 + ord(ch)) & 0xFFFFFFFF
        v[h % dim] += 1.0
    v = v / (v.norm(p=2) + 1e-6)
    return v


# -----------------------------
# DOM -> bid->text
# -----------------------------
def extract_bid_text_map(obs: Dict[str, Any]) -> Dict[str, str]:
    """
    Build bid -> text string from DOM snapshot.
    Requires bs4; if bs4 missing, returns {}.
    """
    if bs4 is None:
        return {}

    dom = obs.get("dom_object")
    extra = obs.get("extra_element_properties")
    if not isinstance(dom, dict) or not isinstance(extra, dict):
        return {}

    html = flatten_dom_to_str(dom, extra, with_center_coords=False)
    soup = bs4.BeautifulSoup(html, "lxml")

    bid2text: Dict[str, str] = {}

    for el in soup.find_all(attrs={"bid": True}):
        bid = str(el.get("bid"))

        parts: List[str] = []

        # Visible text
        txt = el.get_text(" ", strip=True)
        if txt:
            parts.append(txt)

        # Useful attributes
        for attr in ("aria-label", "title", "alt"):
            v = el.get(attr)
            if isinstance(v, str) and v.strip():
                parts.append(v.strip())

        # Critical for <input type="button|submit"> etc.
        if el.name == "input":
            v = el.get("value")
            if isinstance(v, str) and v.strip():
                parts.append(v.strip())

        # Sometimes type helps
        t = el.get("type")
        if isinstance(t, str) and t.strip():
            parts.append(f"type={t.strip()}")

        s = " | ".join(parts).strip()
        if s and bid not in bid2text:
            bid2text[bid] = s

    return bid2text


# -----------------------------
# Candidates: bids + features
# -----------------------------
def obs_to_candidates(
    obs: Dict[str, Any],
    max_candidates: int = 64,
    text_dim: int = 64,
) -> Tuple[List[str], torch.Tensor]:
    """
    Returns:
      - bids: list of clickable bid strings
      - feats: [N, 7 + text_dim] tensor of per-bid features

    Feature design:
      - bbox center (cx, cy) normalized
      - bbox size (w, h) normalized
      - visibility
      - clickable (1.0)
      - constant (1.0)
      - plus hashed text vector from DOM (if bs4 installed)
    """
    extra = obs.get("extra_element_properties")
    if not isinstance(extra, dict):
        return [], torch.zeros((0, 7 + text_dim), dtype=torch.float32)

    bid2text = extract_bid_text_map(obs)

    # Stable ordering: sort bids numerically when possible
    def bid_key(b):
        try:
            return (0, int(b))
        except Exception:
            return (1, str(b))

    items = []
    for bid, props in extra.items():
        if not isinstance(props, dict):
            continue
        if props.get("clickable") is not True:
            continue
        items.append((str(bid), props))

    items.sort(key=lambda x: bid_key(x[0]))
    items = items[:max_candidates]

    bids: List[str] = []
    feats: List[torch.Tensor] = []

    # Normalization constants (good-enough for MiniWoB)
    WN, HN = 1000.0, 700.0

    for bid, props in items:
        bbox = props.get("bbox")
        if not (isinstance(bbox, (list, tuple)) and len(bbox) == 4):
            x0 = y0 = x1 = y1 = 0.0
        else:
            x0, y0, x1, y1 = map(float, bbox)

        w = max(0.0, x1 - x0)
        h = max(0.0, y1 - y0)
        cx = x0 + 0.5 * w
        cy = y0 + 0.5 * h

        vis = float(props.get("visibility", 0.0))
        clickable = 1.0

        geom = torch.tensor(
            [
                cx / WN,
                cy / HN,
                w / WN,
                h / HN,
                vis,
                clickable,
                1.0,
            ],
            dtype=torch.float32,
        )

        text = bid2text.get(bid, "")
        text_vec = hash_text_to_vec(text, dim=text_dim)  # [text_dim]

        f = torch.cat([geom, text_vec], dim=0)  # [7 + text_dim]

        bids.append(bid)
        feats.append(f)

    if not feats:
        return [], torch.zeros((0, 7 + text_dim), dtype=torch.float32)

    return bids, torch.stack(feats, dim=0)


# -----------------------------
# PPO model: scores each candidate + value
# -----------------------------
class BidPolicyValue(nn.Module):
    def __init__(self, goal_dim: int, bid_feat_dim: int, hidden: int = 128):
        super().__init__()
        self.policy = nn.Sequential(
            nn.Linear(goal_dim + bid_feat_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),  # per-bid logit
        )
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
          value: scalar
        """
        N = bid_feats.shape[0]
        if N == 0:
            return torch.empty((0,), dtype=torch.float32, device=bid_feats.device), self.value(goal_vec).squeeze(-1)

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


def sample_action(logits: torch.Tensor) -> Tuple[int, torch.Tensor, torch.Tensor]:
    dist = torch.distributions.Categorical(logits=logits)
    a = dist.sample()
    return int(a.item()), dist.log_prob(a), dist.entropy()


def compute_gae(steps: List[Step], gamma: float, lam: float, last_value: float = 0.0) -> Tuple[torch.Tensor, torch.Tensor]:
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
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    return ret, adv


# -----------------------------
# Main
# -----------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--env-id", default="browsergym/miniwob.click-button")
    p.add_argument("--total-iters", type=int, default=50)
    p.add_argument("--steps-per-iter", type=int, default=256)
    p.add_argument("--ppo-epochs", type=int, default=4)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae-lam", type=float, default=0.95)
    p.add_argument("--clip", type=float, default=0.2)
    p.add_argument("--ent-coef", type=float, default=0.01)
    p.add_argument("--vf-coef", type=float, default=0.5)
    p.add_argument("--max-candidates", type=int, default=64)
    p.add_argument("--goal-dim", type=int, default=256)
    p.add_argument("--text-dim", type=int, default=64)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--headless", action="store_true")
    p.add_argument("--disable-checker", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    if not os.environ.get("MINIWOB_URL"):
        raise RuntimeError(
            "MINIWOB_URL is not set.\n"
            "Example:\n"
            "  export MINIWOB_URL=http://127.0.0.1:8000/miniwob/\n"
            "And run the http.server from miniwob-plusplus/miniwob/html"
        )

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    env_kwargs = dict(headless=args.headless)
    if args.disable_checker:
        env_kwargs["disable_env_checker"] = True

    env = gym.make(args.env_id, **env_kwargs)

    bid_feat_dim = 7 + args.text_dim
    model = BidPolicyValue(goal_dim=args.goal_dim, bid_feat_dim=bid_feat_dim, hidden=args.hidden).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    obs, info = env.reset()

    episodes_completed = 0

    for it in range(args.total_iters):
        steps: List[Step] = []
        entropies: List[torch.Tensor] = []
        successes_in_batch = 0

        # -------- collect rollout --------
        while len(steps) < args.steps_per_iter:
            goal = obs.get("goal", "")
            goal_vec = hash_goal_vec(goal, dim=args.goal_dim).to(device)

            bids, bid_feats = obs_to_candidates(
                obs,
                max_candidates=args.max_candidates,
                text_dim=args.text_dim,
            )
            bid_feats = bid_feats.to(device)

            if len(bids) == 0:
                obs, info = env.reset()
                continue

            logits, value = model(goal_vec, bid_feats)
            a_idx, logp, ent = sample_action(logits)
            entropies.append(ent)

            action = {"action": "click", "bid": bids[a_idx]}
            obs2, reward, terminated, truncated, info = env.step(action)
            done = bool(terminated) or bool(truncated)

            if float(reward) > 0:
                successes_in_batch += 1

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

            obs = obs2
            if done:
                episodes_completed += 1
                obs, info = env.reset()

        # -------- bootstrap value --------
        with torch.no_grad():
            goal_vec = hash_goal_vec(obs.get("goal", ""), dim=args.goal_dim).to(device)
            bids, bid_feats = obs_to_candidates(obs, max_candidates=args.max_candidates, text_dim=args.text_dim)
            bid_feats = bid_feats.to(device)
            _, last_v = model(goal_vec, bid_feats) if len(bids) > 0 else (None, torch.tensor(0.0, device=device))
            last_v = float(last_v.item())

        returns, adv = compute_gae(steps, gamma=args.gamma, lam=args.gae_lam, last_value=last_v)
        returns = returns.to(device)
        adv = adv.to(device)

        # -------- PPO update --------
        for _epoch in range(args.ppo_epochs):
            policy_loss = 0.0
            value_loss = 0.0
            entropy_loss = 0.0

            for i, s in enumerate(steps):
                goal_vec = s.goal_vec.to(device)
                bid_feats = s.bid_feats.to(device)

                logits, value = model(goal_vec, bid_feats)
                dist = torch.distributions.Categorical(logits=logits)

                a = torch.tensor(s.action_idx, device=device)
                new_logp = dist.log_prob(a)

                ratio = torch.exp(new_logp - s.logp.to(device))
                surr1 = ratio * adv[i]
                surr2 = torch.clamp(ratio, 1.0 - args.clip, 1.0 + args.clip) * adv[i]
                policy_loss = policy_loss + (-torch.min(surr1, surr2))

                value_loss = value_loss + (value - returns[i]).pow(2)
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
        pos_frac = float(np.mean([1.0 if s.reward > 0 else 0.0 for s in steps])) if steps else 0.0

        print(
            f"[iter {it:03d}] "
            f"avg_step_reward={avg_reward:.3f} "
            f"pos_reward_frac={pos_frac:.2f} "
            f"entropy={avg_ent:.2f} "
            f"episodes_completed={episodes_completed} "
            f"successes_in_batch={successes_in_batch}/{len(steps)}"
        )

    env.close()


if __name__ == "__main__":
    main()