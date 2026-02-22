#!/usr/bin/env python3
"""
Vectorized MiniWoB PPO (click-by-bid) for BrowserGym.

Key differences vs your single-env script:
- Runs N environments in parallel with gymnasium.vector (SyncVectorEnv by default).
- Pads candidates to max_candidates and uses an action mask.
- Batched PPO update with minibatches.

Install:
  pip install -U gymnasium torch numpy browsergym-miniwob playwright
  python -m playwright install
  pip install -U beautifulsoup4 lxml   # optional, improves text features

Runtime:
  export MINIWOB_URL=http://127.0.0.1:8000/miniwob/
"""

import os
import argparse
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn

import gymnasium as gym
import browsergym.miniwob  # registers envs
from browsergym.utils.obs import flatten_dom_to_str

try:
    import bs4
except Exception:
    bs4 = None


# -----------------------------
# Hashing text -> vector
# -----------------------------
def hash_text_to_vec(text: str, dim: int) -> torch.Tensor:
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


def hash_goal_vec(goal: str, dim: int) -> torch.Tensor:
    return hash_text_to_vec(goal, dim)


# -----------------------------
# DOM -> bid->text
# -----------------------------
def extract_bid_text_map(obs: Dict[str, Any], *, parser: str = "lxml") -> Dict[str, str]:
    if bs4 is None:
        return {}

    dom = obs.get("dom_object")
    extra = obs.get("extra_element_properties")
    if not isinstance(dom, dict) or not isinstance(extra, dict):
        return {}

    html = flatten_dom_to_str(dom, extra, with_center_coords=False)
    soup = bs4.BeautifulSoup(html, parser)

    bid2text: Dict[str, str] = {}

    def add(parts: List[str], s: Any):
        if isinstance(s, str):
            s = s.strip()
            if s:
                parts.append(s)

    for el in soup.find_all(attrs={"bid": True}):
        bid = str(el.get("bid"))
        parts: List[str] = []
        add(parts, el.get_text(" ", strip=True))
        for attr in ("aria-label", "title", "alt", "placeholder", "name", "id"):
            add(parts, el.get(attr))
        if el.name == "input":
            add(parts, el.get("value"))
        text = " | ".join(parts).strip()
        if text:
            bid2text.setdefault(bid, text)

    return bid2text


# -----------------------------
# Candidates -> bids + features (single obs)
# -----------------------------
def obs_to_candidates(
    obs: Dict[str, Any],
    max_candidates: int,
    text_dim: int,
    *,
    include_text: bool = True,
) -> Tuple[List[str], torch.Tensor, Dict[str, str]]:
    extra = obs.get("extra_element_properties")
    if not isinstance(extra, dict):
        return [], torch.zeros((0, 7 + text_dim), dtype=torch.float32), {}

    bid2text = extract_bid_text_map(obs) if include_text else {}

    def bid_key(b: str):
        try:
            return (0, int(b))
        except Exception:
            return (1, b)

    items: List[Tuple[str, Dict[str, Any]]] = []
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

    WN, HN = 1000.0, 700.0  # approx MiniWoB

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

        geom = torch.tensor(
            [cx / WN, cy / HN, w / WN, h / HN, vis, 1.0, 1.0],
            dtype=torch.float32,
        )

        txt = bid2text.get(bid, "")
        txt_vec = hash_text_to_vec(txt, text_dim)

        bids.append(bid)
        feats.append(torch.cat([geom, txt_vec], dim=0))

    if not feats:
        return [], torch.zeros((0, 7 + text_dim), dtype=torch.float32), bid2text

    return bids, torch.stack(feats, dim=0), bid2text


# -----------------------------
# Action formatting
# -----------------------------
def click_action(bid: str) -> str:
    bid = str(bid).replace("'", "\\'")
    return f"click('{bid}')"


# -----------------------------
# Batched model: (N,G) + (N,K,D) -> logits (N,K), value (N,)
# -----------------------------
class BatchedBidPolicyValue(nn.Module):
    def __init__(self, goal_dim: int, bid_feat_dim: int, hidden: int):
        super().__init__()
        self.goal_dim = goal_dim
        self.bid_feat_dim = bid_feat_dim

        self.policy = nn.Sequential(
            nn.Linear(goal_dim + bid_feat_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )
        self.value = nn.Sequential(
            nn.Linear(goal_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, goal_vecs: torch.Tensor, cand_feats: torch.Tensor):
        """
        goal_vecs: (N,G)
        cand_feats: (N,K,D)
        returns:
          logits: (N,K)
          values: (N,)
        """
        N, K, D = cand_feats.shape
        goal_rep = goal_vecs.unsqueeze(1).expand(N, K, self.goal_dim)  # (N,K,G)
        x = torch.cat([goal_rep, cand_feats], dim=-1)  # (N,K,G+D)
        logits = self.policy(x).squeeze(-1)  # (N,K)
        values = self.value(goal_vecs).squeeze(-1)  # (N,)
        return logits, values


# -----------------------------
# Vector helpers: batch obs -> padded tensors + bids list
# -----------------------------
@torch.no_grad()
def batch_obs_to_tensors(
    obs_list: List[Dict[str, Any]],
    max_candidates: int,
    goal_dim: int,
    text_dim: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[List[str]]]:
    """
    returns:
      goal_vecs: (N,G)
      cand_feats: (N,K,D)
      mask: (N,K) bool
      bids_list: list of list[str] length N
    """
    N = len(obs_list)
    K = max_candidates
    D = 7 + text_dim

    goal_vecs = torch.zeros((N, goal_dim), dtype=torch.float32, device=device)
    cand_feats = torch.zeros((N, K, D), dtype=torch.float32, device=device)
    mask = torch.zeros((N, K), dtype=torch.bool, device=device)
    bids_list: List[List[str]] = []

    include_text = (bs4 is not None)

    for i, obs in enumerate(obs_list):
        goal_vecs[i] = hash_goal_vec(obs.get("goal", ""), goal_dim).to(device)
        bids, feats, _ = obs_to_candidates(obs, max_candidates=max_candidates, text_dim=text_dim, include_text=include_text)
        bids_list.append(bids)

        n = min(len(bids), K)
        if n > 0:
            cand_feats[i, :n] = feats[:n].to(device)
            mask[i, :n] = True
        else:
            # no candidates -> keep mask false; we will force a dummy valid action at index 0
            mask[i, 0] = True

    return goal_vecs, cand_feats, mask, bids_list


def masked_categorical(logits: torch.Tensor, mask: torch.Tensor):
    """
    logits: (N,K), mask: (N,K) bool
    returns dist over valid actions only
    """
    masked_logits = logits.masked_fill(~mask, -1e9)
    return torch.distributions.Categorical(logits=masked_logits)


def compute_gae_vec(
    rewards: torch.Tensor,      # (T,N)
    dones: torch.Tensor,        # (T,N) float 0/1
    values: torch.Tensor,       # (T,N)
    last_values: torch.Tensor,  # (N,)
    gamma: float,
    lam: float,
):
    """
    Vectorized GAE over time for N envs.
    returns:
      returns: (T,N)
      adv: (T,N) normalized
    """
    T, N = rewards.shape
    adv = torch.zeros((T, N), dtype=torch.float32, device=rewards.device)
    gae = torch.zeros((N,), dtype=torch.float32, device=rewards.device)

    for t in reversed(range(T)):
        next_v = last_values if t == T - 1 else values[t + 1]
        not_done = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_v * not_done - values[t]
        gae = delta + gamma * lam * not_done * gae
        adv[t] = gae

    ret = adv + values

    # normalize advantages over the whole batch
    adv_flat = adv.reshape(-1)
    adv = (adv - adv_flat.mean()) / (adv_flat.std() + 1e-8)
    return ret, adv


# -----------------------------
# Env factory
# -----------------------------
def make_env(env_id: str, headless: bool, disable_checker: bool):
    def _init():
        kwargs = dict(headless=headless)
        if disable_checker:
            kwargs["disable_env_checker"] = True
        return gym.make(env_id, **kwargs)
    return _init


# -----------------------------
# Main
# -----------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--env-id", default="browsergym/miniwob.click-button")
    p.add_argument("--num-envs", type=int, default=8)
    p.add_argument("--steps-per-env", type=int, default=128, help="rollout steps PER ENV per iteration")
    p.add_argument("--total-iters", type=int, default=200)

    p.add_argument("--ppo-epochs", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=512, help="minibatch size for PPO update (after flatten)")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae-lam", type=float, default=0.95)
    p.add_argument("--clip", type=float, default=0.2)
    p.add_argument("--ent-coef", type=float, default=0.01)
    p.add_argument("--vf-coef", type=float, default=0.5)

    p.add_argument("--max-candidates", type=int, default=64)
    p.add_argument("--goal-dim", type=int, default=256)
    p.add_argument("--text-dim", type=int, default=128)
    p.add_argument("--hidden", type=int, default=256)

    p.add_argument("--headless", action="store_true")
    p.add_argument("--disable-checker", action="store_true")
    p.add_argument("--seed", type=int, default=0)

    args = p.parse_args()

    if not os.environ.get("MINIWOB_URL"):
        raise RuntimeError(
            "Missing MINIWOB_URL.\n"
            "Example:\n"
            "  export MINIWOB_URL=http://127.0.0.1:8000/miniwob/\n"
            "and run: python -m http.server 8000 (from miniwob-plusplus/miniwob/html)"
        )

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)
    if bs4 is None:
        print("[warn] beautifulsoup4/lxml not installed -> element text features DISABLED.")
        print("       Run: pip install -U beautifulsoup4 lxml")

    # Vector env
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, args.headless, args.disable_checker) for _ in range(args.num_envs)]
    )

    # model
    bid_feat_dim = 7 + args.text_dim
    model = BatchedBidPolicyValue(args.goal_dim, bid_feat_dim, args.hidden).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    obs, _ = envs.reset(seed=args.seed)
    # obs is usually an object array of dicts
    obs_list = list(obs)

    K = args.max_candidates
    T = args.steps_per_env
    N = args.num_envs

    for it in range(args.total_iters):
        # rollout storage (T,N,...) mostly tensors
        actions = torch.zeros((T, N), dtype=torch.long, device=device)
        old_logp = torch.zeros((T, N), dtype=torch.float32, device=device)
        values = torch.zeros((T, N), dtype=torch.float32, device=device)
        rewards = torch.zeros((T, N), dtype=torch.float32, device=device)
        dones = torch.zeros((T, N), dtype=torch.float32, device=device)
        entropies = torch.zeros((T, N), dtype=torch.float32, device=device)

        # also store inputs for PPO recompute
        goal_buf = torch.zeros((T, N, args.goal_dim), dtype=torch.float32, device=device)
        cand_buf = torch.zeros((T, N, K, bid_feat_dim), dtype=torch.float32, device=device)
        mask_buf = torch.zeros((T, N, K), dtype=torch.bool, device=device)
        bids_buf: List[List[List[str]]] = []

        successes = 0

        # ---- collect rollout ----
        for t in range(T):
            goal_vecs, cand_feats, mask, bids_list = batch_obs_to_tensors(
                obs_list, args.max_candidates, args.goal_dim, args.text_dim, device
            )
            bids_buf.append(bids_list)

            goal_buf[t] = goal_vecs
            cand_buf[t] = cand_feats
            mask_buf[t] = mask

            with torch.no_grad():
                logits, v = model(goal_vecs, cand_feats)
                dist = masked_categorical(logits, mask)
                a = dist.sample()
                lp = dist.log_prob(a)
                ent = dist.entropy()

            actions[t] = a
            old_logp[t] = lp
            values[t] = v
            entropies[t] = ent

            # map (env i, action index) -> bid string -> action string
            a_cpu = a.detach().cpu().numpy()
            act_strs = []
            for i in range(N):
                bids = bids_list[i]
                idx = int(a_cpu[i])
                if idx >= len(bids) or len(bids) == 0:
                    idx = 0
                bid = bids[idx] if bids else "0"
                act_strs.append(click_action(bid))

            obs2, rew, term, trunc, _ = envs.step(np.array(act_strs, dtype=object))
            done = np.logical_or(term, trunc)

            rewards[t] = torch.tensor(rew, dtype=torch.float32, device=device)
            dones[t] = torch.tensor(done.astype(np.float32), device=device)
            successes += int((np.array(rew) > 0).sum())

            # reset done envs (vector env does not auto-reset)
            # easiest: call reset on those indices
            if done.any():
                done_idx = np.nonzero(done)[0].tolist()
                for j in done_idx:
                    o_j, _ = envs.env_method("reset", indices=j, seed=args.seed + it * 100000 + j + t)
                    obs2[j] = o_j[0]

            obs_list = list(obs2)

        # bootstrap last values
        with torch.no_grad():
            goal_last, cand_last, mask_last, _ = batch_obs_to_tensors(
                obs_list, args.max_candidates, args.goal_dim, args.text_dim, device
            )
            _, last_v = model(goal_last, cand_last)  # (N,)

        # GAE
        ret, adv = compute_gae_vec(rewards, dones, values, last_v, args.gamma, args.gae_lam)

        # ---- PPO update (minibatches) ----
        B = T * N
        # flatten batch
        flat_goal = goal_buf.reshape(B, args.goal_dim)
        flat_cand = cand_buf.reshape(B, K, bid_feat_dim)
        flat_mask = mask_buf.reshape(B, K)
        flat_actions = actions.reshape(B)
        flat_old_logp = old_logp.reshape(B)
        flat_ret = ret.reshape(B)
        flat_adv = adv.reshape(B)

        idx = torch.randperm(B, device=device)

        for ep in range(args.ppo_epochs):
            for start in range(0, B, args.batch_size):
                mb = idx[start : start + args.batch_size]

                g = flat_goal[mb]
                c = flat_cand[mb]
                m = flat_mask[mb]
                a = flat_actions[mb]
                oldlp = flat_old_logp[mb]
                tgt_ret = flat_ret[mb]
                tgt_adv = flat_adv[mb]

                logits, v = model(g, c)
                dist = masked_categorical(logits, m)

                newlp = dist.log_prob(a)
                ratio = torch.exp(newlp - oldlp)

                surr1 = ratio * tgt_adv
                surr2 = torch.clamp(ratio, 1.0 - args.clip, 1.0 + args.clip) * tgt_adv
                pol_loss = -(torch.min(surr1, surr2)).mean()

                vf_loss = 0.5 * (v - tgt_ret).pow(2).mean()
                ent_loss = -dist.entropy().mean()

                loss = pol_loss + args.vf_coef * vf_loss + args.ent_coef * ent_loss

                opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()

        avg_reward = float(rewards.mean().item())
        pos_frac = float((rewards > 0).float().mean().item())
        avg_ent = float(entropies.mean().item())

        print(
            f"[iter {it:03d}] "
            f"avg_step_reward={avg_reward:.3f} pos_reward_frac={pos_frac:.2f} entropy={avg_ent:.2f} "
            f"successes_in_batch={successes}/{T*N}"
        )

    envs.close()


if __name__ == "__main__":
    main()