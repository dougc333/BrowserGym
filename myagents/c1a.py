#!/usr/bin/env python3
import os
import argparse
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple

import numpy as np
import torch
import torch.nn as nn

import gymnasium as gym
import browsergym.miniwob  # registers envs

# Optional DOM parsing (not required for baseline sanity)
try:
    import bs4
    from browsergym.utils.obs import flatten_dom_to_str
except Exception:
    bs4 = None
    flatten_dom_to_str = None


def hash_text_to_vec(text: str, dim: int) -> torch.Tensor:
    text = (text or "").lower()
    v = torch.zeros(dim, dtype=torch.float32)
    if len(text) < 3:
        return v
    for i in range(len(text) - 2):
        tri = text[i:i+3]
        h = 0
        for ch in tri:
            h = (h * 131 + ord(ch)) & 0xFFFFFFFF
        v[h % dim] += 1.0
    v = v / (v.norm(p=2) + 1e-6)
    return v


def extract_bid_text_map(obs: Dict[str, Any]) -> Dict[str, str]:
    if bs4 is None or flatten_dom_to_str is None:
        return {}

    dom = obs.get("dom_object")
    extra = obs.get("extra_element_properties")
    if not isinstance(dom, dict) or not isinstance(extra, dict):
        return {}

    html = flatten_dom_to_str(dom, extra, with_center_coords=False)
    soup = bs4.BeautifulSoup(html, "lxml")

    out: Dict[str, str] = {}

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
        txt = " | ".join(parts).strip()
        if txt:
            out.setdefault(bid, txt)

    return out


# def obs_to_candidates(obs: Dict[str, Any], max_candidates: int, text_dim: int) -> Tuple[List[str], torch.Tensor]:
#     extra = obs.get("extra_element_properties")
#     if not isinstance(extra, dict):
#         return [], torch.zeros((0, 7 + text_dim), dtype=torch.float32)

#     bid2text = extract_bid_text_map(obs)

#     def bid_key(b: str):
#         try:
#             return (0, int(b))
#         except Exception:
#             return (1, b)

#     items = []
#     for bid, props in extra.items():
#         if not isinstance(props, dict):
#             continue
#         if props.get("clickable") is not True:
#             continue
#         items.append((str(bid), props))

#     items.sort(key=lambda x: bid_key(x[0]))
#     items = items[:max_candidates]

#     bids: List[str] = []
#     feats: List[torch.Tensor] = []

#     WN, HN = 1000.0, 700.0
#     for bid, props in items:
#         bbox = props.get("bbox")
#         if not (isinstance(bbox, (list, tuple)) and len(bbox) == 4):
#             x0 = y0 = x1 = y1 = 0.0
#         else:
#             x0, y0, x1, y1 = map(float, bbox)

#         w = max(0.0, x1 - x0)
#         h = max(0.0, y1 - y0)
#         cx = x0 + 0.5 * w
#         cy = y0 + 0.5 * h
#         vis = float(props.get("visibility", 0.0))

#         geom = torch.tensor(
#             [cx / WN, cy / HN, w / WN, h / HN, vis, 1.0, 1.0],
#             dtype=torch.float32,
#         )
#         txt = bid2text.get(bid, "")
#         txt_vec = hash_text_to_vec(txt, text_dim)
#         bids.append(bid)
#         feats.append(torch.cat([geom, txt_vec], dim=0))

#     if not feats:
#         return [], torch.zeros((0, 7 + text_dim), dtype=torch.float32)
#     return bids, torch.stack(feats, dim=0)
def obs_to_candidates(obs: Dict[str, Any], max_candidates: int, text_dim: int):
    extra = obs.get("extra_element_properties")
    if not isinstance(extra, dict):
        return [], torch.zeros((0, 7 + text_dim), dtype=torch.float32), {}

    bid2text = extract_bid_text_map(obs)

    # Build items from ALL DOM bids we found, not only extra.clickable=True
    items = []
    for bid, txt in bid2text.items():
        props = extra.get(bid)
        if props is None and bid.isdigit():
            props = extra.get(int(bid))
        if not isinstance(props, dict):
            continue
        bbox = props.get("bbox")
        if not (isinstance(bbox, (list, tuple)) and len(bbox) == 4):
            continue
        vis = float(props.get("visibility", 0.0))
        # keep visible-ish elements; tweak threshold if needed
        if vis <= 0.0:
            continue
        items.append((bid, props, txt))

    # If DOM parsing failed (no bs4), fall back to clickable=True as before
    if not items:
        items = []
        for bid, props in extra.items():
            if not isinstance(props, dict):
                continue
            if props.get("clickable") is not True:
                continue
            bbox = props.get("bbox")
            if not (isinstance(bbox, (list, tuple)) and len(bbox) == 4):
                continue
            vis = float(props.get("visibility", 0.0))
            txt = bid2text.get(str(bid), "")
            items.append((str(bid), props, txt))

    def bid_key(b):
        try:
            return (0, int(b))
        except Exception:
            return (1, str(b))

    items.sort(key=lambda x: bid_key(x[0]))
    items = items[:max_candidates]

    bids = []
    feats = []

    WN, HN = 1000.0, 700.0
    for bid, props, txt in items:
        x0, y0, x1, y1 = map(float, props["bbox"])
        w = max(0.0, x1 - x0)
        h = max(0.0, y1 - y0)
        cx = x0 + 0.5 * w
        cy = y0 + 0.5 * h
        vis = float(props.get("visibility", 0.0))
        clickable = 1.0 if props.get("clickable") is True else 0.0

        geom = torch.tensor(
            [cx / WN, cy / HN, w / WN, h / HN, vis, clickable, 1.0],
            dtype=torch.float32,
        )
        txt_vec = hash_text_to_vec(txt, text_dim)
        bids.append(str(bid))
        feats.append(torch.cat([geom, txt_vec], dim=0))

    if not feats:
        return [], torch.zeros((0, 7 + text_dim), dtype=torch.float32), bid2text

    return bids, torch.stack(feats, dim=0), bid2text

def sanity_bruteforce_episode(env, max_steps=10, max_candidates=64, text_dim=128):
    obs, _ = env.reset()
    print("\n=== SANITY BRUTEFORCE ===")
    print("goal:", obs.get("goal"))

    for t in range(max_steps):
        bids, feats, bid2text = obs_to_candidates(obs, max_candidates, text_dim)
        print(f"[t={t}] num_candidates={len(bids)} sample={bids[:10]}")
        # print all candidate texts
        for bid in bids[:10]:
            print(" ", bid, "->", bid2text.get(bid, "")[:80])

        # try each bid once
        for bid in bids:
            obs2, reward, term, trunc, _ = env.step({"action": "click", "bid": bid})
            if float(reward) > 0:
                print(f"[SUCCESS] clicked bid={bid} reward={reward}")
                return True
            if term or trunc:
                break
            obs = obs2

        if term or trunc:
            break
    print("[FAIL] never got positive reward in bruteforce")
    return False

class BidPolicyValue(nn.Module):
    def __init__(self, goal_dim: int, bid_feat_dim: int, hidden: int):
        super().__init__()
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

    def forward(self, goal_vec: torch.Tensor, bid_feats: torch.Tensor):
        n = bid_feats.shape[0]
        if n == 0:
            return torch.empty((0,), device=bid_feats.device), self.value(goal_vec).squeeze(-1)
        goal_rep = goal_vec.unsqueeze(0).expand(n, -1)
        x = torch.cat([goal_rep, bid_feats], dim=-1)
        logits = self.policy(x).squeeze(-1)
        v = self.value(goal_vec).squeeze(-1)
        return logits, v


@dataclass
class Step:
    goal_vec: torch.Tensor
    bid_feats: torch.Tensor
    action_idx: int
    logp: torch.Tensor
    value: torch.Tensor
    reward: float
    done: bool


def sample_action(logits: torch.Tensor):
    dist = torch.distributions.Categorical(logits=logits)
    a = dist.sample()
    return int(a.item()), dist.log_prob(a), dist.entropy()


def compute_gae(steps: List[Step], gamma: float, lam: float, last_value: float):
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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--env-id", dest="env_id", default="browsergym/miniwob.click-button")
    p.add_argument("--sanity-bruteforce", action="store_true")
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
    p.add_argument("--text-dim", type=int, default=128)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--headless", action="store_true")
    p.add_argument("--disable-checker", action="store_true")
    p.add_argument("--debug", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    if not os.environ.get("MINIWOB_URL"):
        raise RuntimeError("Missing MINIWOB_URL (export MINIWOB_URL=http://127.0.0.1:8000/miniwob/)")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    if bs4 is None:
        print("[warn] bs4/lxml not installed -> DOM text features disabled (still should get random success).")

    env_kwargs = dict(headless=args.headless)
    if args.disable_checker:
        env_kwargs["disable_env_checker"] = True

    env = gym.make(args.env_id, **env_kwargs)
    if args.sanity_bruteforce:
      ok = sanity_bruteforce_episode(env, max_steps=5, max_candidates=args.max_candidates, text_dim=args.text_dim)
      env.close()
      raise SystemExit(0 if ok else 1)

    bid_feat_dim = 7 + args.text_dim
    model = BidPolicyValue(args.goal_dim, bid_feat_dim, args.hidden).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    obs, _ = env.reset()
    episodes_completed = 0

    for it in range(args.total_iters):
        steps: List[Step] = []
        entropies: List[torch.Tensor] = []
        successes = 0
        env_steps = 0

        while len(steps) < args.steps_per_iter:
            goal = obs.get("goal", "")
            goal_vec = hash_text_to_vec(goal, args.goal_dim).to(device)

            bids, bid_feats = obs_to_candidates(obs, args.max_candidates, args.text_dim)
            bid_feats = bid_feats.to(device)

            if args.debug and env_steps < 3:
                print(f"\n[debug] goal={goal!r}")
                print(f"[debug] num_clickable={len(bids)} sample_bids={bids[:10]}")

            if not bids:
                obs, _ = env.reset()
                continue

            logits, value = model(goal_vec, bid_feats)
            a_idx, logp, ent = sample_action(logits)
            entropies.append(ent)

            bid = bids[a_idx]
            action = {"action": "click", "bid": bid}
            assert isinstance(action["bid"], str) and action["bid"], "bid must be non-empty string"

            obs2, reward, terminated, truncated, _ = env.step(action)
            env_steps += 1
            done = bool(terminated) or bool(truncated)

            r = float(reward)
            if r > 0:
                successes += 1
                if args.debug:
                    print(f"[debug] SUCCESS click bid={bid} reward={r}")

            steps.append(
                Step(goal_vec.detach(), bid_feats.detach(), a_idx, logp.detach(), value.detach(), r, done)
            )

            obs = obs2
            if done:
                episodes_completed += 1
                obs, _ = env.reset()

        with torch.no_grad():
            gv = hash_text_to_vec(obs.get("goal", ""), args.goal_dim).to(device)
            bids, bf = obs_to_candidates(obs, args.max_candidates, args.text_dim)
            bf = bf.to(device)
            last_v = float(model(gv, bf)[1].item()) if bids else 0.0

        returns, adv = compute_gae(steps, args.gamma, args.gae_lam, last_v)
        returns, adv = returns.to(device), adv.to(device)

        for _ in range(args.ppo_epochs):
            pol = 0.0
            vf = 0.0
            entl = 0.0

            for i, s in enumerate(steps):
                logits, v = model(s.goal_vec.to(device), s.bid_feats.to(device))
                dist = torch.distributions.Categorical(logits=logits)
                a = torch.tensor(s.action_idx, device=device)
                new_logp = dist.log_prob(a)

                ratio = torch.exp(new_logp - s.logp.to(device))
                surr1 = ratio * adv[i]
                surr2 = torch.clamp(ratio, 1.0 - args.clip, 1.0 + args.clip) * adv[i]
                pol = pol + (-torch.min(surr1, surr2))

                vf = vf + (v - returns[i]).pow(2)
                entl = entl + (-dist.entropy())

            pol /= len(steps)
            vf /= len(steps)
            entl /= len(steps)

            loss = pol + args.vf_coef * vf + args.ent_coef * entl
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        avg_ent = float(torch.stack(entropies).mean().item()) if entropies else 0.0
        avg_r = float(np.mean([s.reward for s in steps])) if steps else 0.0
        pos = float(np.mean([1.0 if s.reward > 0 else 0.0 for s in steps])) if steps else 0.0

        print(
            f"[iter {it:03d}] avg_step_reward={avg_r:.3f} pos_reward_frac={pos:.2f} "
            f"entropy={avg_ent:.2f} episodes_completed={episodes_completed} successes_in_batch={successes}/{len(steps)}"
        )

    env.close()


if __name__ == "__main__":
    main()