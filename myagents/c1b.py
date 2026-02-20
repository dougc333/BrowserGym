#!/usr/bin/env python3
"""
MiniWoB PPO (click-by-bid) baseline for BrowserGym + sanity bruteforce.

What this gives you:
- obs_to_candidates() returns candidate bids + (geom + text-hash) features
- extract_bid_text_map() parses DOM to recover per-element labels (e.g. "Ok", "Submit")
- --sanity-bruteforce: runs ONE episode where we try clicking each candidate bid and
  prints bid->text/bbox/vis to confirm the *correct* bid is in the candidate set.

Install:
  pip install gymnasium torch numpy browsergym-miniwob playwright
  python -m playwright install

For DOM parsing (recommended):
  pip install beautifulsoup4 lxml

Runtime:
  export MINIWOB_URL=http://127.0.0.1:8000/miniwob/
  # run http server from miniwob-plusplus/miniwob/html (the directory containing core/, miniwob/, etc)
  python -m http.server 8000
"""

import os
import argparse
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple

import numpy as np
import torch
import torch.nn as nn

import gymnasium as gym
import browsergym.miniwob  # registers MiniWoB envs

from browsergym.utils.obs import flatten_dom_to_str

# Optional DOM parsing
try:
    import bs4  # beautifulsoup4
except Exception:
    bs4 = None


# -----------------------------
# Hashing text -> vector (no extra deps)
# -----------------------------
def hash_text_to_vec(text: str, dim: int = 128) -> torch.Tensor:
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
    return hash_text_to_vec(goal, dim=dim)


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

    # Any element that has a 'bid' attribute
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

        # Critical for <input>
        if el.name == "input":
            v = el.get("value")
            if isinstance(v, str) and v.strip():
                parts.append(v.strip())
            t = el.get("type")
            if isinstance(t, str) and t.strip():
                parts.append(f"type={t.strip()}")

        s = " | ".join(parts).strip()
        if s:
            # If duplicates, keep first
            bid2text.setdefault(bid, s)

    return bid2text


# -----------------------------
# Candidates: bids + features
# -----------------------------
def _parse_bbox(bbox):
    """
    Returns (x0,y0,x1,y1,w,h,cx,cy) or None.
    Supports both:
      - [x0,y0,x1,y1]
      - [x,y,w,h]
    """
    if not (isinstance(bbox, (list, tuple)) and len(bbox) == 4):
        return None
    a, b, c, d = map(float, bbox)

    # Heuristic: if (c,d) look like width/height (small-ish) rather than absolute coords
    # In MiniWoB you may see [x,y,w,h] for elements and [x0,y0,x1,y1] for containers.
    if c >= 0 and d >= 0 and (c < 1200 and d < 900) and (a + c > a) and (b + d > b):
        # treat as [x,y,w,h]
        x0, y0 = a, b
        w, h = c, d
        x1, y1 = x0 + w, y0 + h
    else:
        # treat as [x0,y0,x1,y1]
        x0, y0, x1, y1 = a, b, c, d
        w = max(0.0, x1 - x0)
        h = max(0.0, y1 - y0)

    if w <= 0 or h <= 0:
        return None
    cx = x0 + 0.5 * w
    cy = y0 + 0.5 * h
    return x0, y0, x1, y1, w, h, cx, cy


def obs_to_candidates(
    obs: Dict[str, Any],
    max_candidates: int = 128,
    text_dim: int = 128,
) -> Tuple[List[str], torch.Tensor, Dict[str, str]]:
    extra = obs.get("extra_element_properties")
    if not isinstance(extra, dict):
        return [], torch.zeros((0, 7 + text_dim), dtype=torch.float32), {}

    bid2text = extract_bid_text_map(obs)

    def get_props_for_bid(bid: str):
        props = extra.get(bid)
        if props is None and bid.isdigit():
            props = extra.get(int(bid))
        return props if isinstance(props, dict) else None

    def valid_clickable(props: Dict[str, Any]):
        if props.get("clickable") is not True:
            return None
        vis = float(props.get("visibility", 0.0))
        if vis <= 0.0:
            return None
        bb = _parse_bbox(props.get("bbox"))
        if bb is None:
            return None
        # Optional: drop massive full-page containers
        _, _, _, _, w, h, _, _ = bb
        if w > 980.0 and h > 600.0:
            return None
        return bb

    items: List[Tuple[str, Dict[str, Any], str, tuple]] = []

    # Prefer DOM-derived clickable elements (have better text)
    if bid2text:
        for bid, txt in bid2text.items():
            props = get_props_for_bid(str(bid))
            if not props:
                continue
            bb = valid_clickable(props)
            if bb is None:
                continue
            items.append((str(bid), props, txt, bb))

    # Fallback: any clickable element from extra
    if not items:
        for bid_raw, props in extra.items():
            if not isinstance(props, dict):
                continue
            bb = valid_clickable(props)
            if bb is None:
                continue
            bid = str(bid_raw)
            txt = bid2text.get(bid, "")
            items.append((bid, props, txt, bb))

    def bid_key(b: str):
        try:
            return (0, int(b))
        except Exception:
            return (1, b)

    items.sort(key=lambda x: bid_key(x[0]))
    items = items[:max_candidates]

    bids: List[str] = []
    feats: List[torch.Tensor] = []

    WN, HN = 1000.0, 700.0

    for bid, props, txt, bb in items:
        x0, y0, x1, y1, w, h, cx, cy = bb
        vis = float(props.get("visibility", 0.0))

        geom = torch.tensor(
            [cx / WN, cy / HN, w / WN, h / HN, vis, 1.0, 1.0],
            dtype=torch.float32,
        )
        txt_vec = hash_text_to_vec(txt, dim=text_dim)

        bids.append(bid)
        feats.append(torch.cat([geom, txt_vec], dim=0))

    if not feats:
        return [], torch.zeros((0, 7 + text_dim), dtype=torch.float32), bid2text

    return bids, torch.stack(feats, dim=0), bid2text


def debug_print_candidates(obs: Dict[str, Any], bids: List[str], bid2text: Dict[str, str], k: int = 25):
    extra = obs.get("extra_element_properties", {})
    print("[debug] candidates (first %d):" % min(k, len(bids)))
    for bid in bids[:k]:
        props = extra.get(bid) or (extra.get(int(bid)) if bid.isdigit() else None) or {}
        bbox = props.get("bbox")
        vis = props.get("visibility")
        clickable = props.get("clickable")
        txt = bid2text.get(bid, "")
        print(f"  bid={bid:>4} vis={vis} clickable={clickable} bbox={bbox} text={txt!r}")


# def sanity_bruteforce_episode(env, max_steps: int = 1, max_candidates: int = 128, text_dim: int = 128) -> bool:
#     """
#     Sanity check: on the current task, extract candidate BIDs and click each one once.
#     Returns True if any click yields reward > 0.
#     """

#     obs, info = env.reset()
#     print("\n=== SANITY BRUTEFORCE ===")
#     print("goal:", obs.get("goal"))
#     print("url :", obs.get("url"))

#     for t in range(max_steps):
#         term = False
#         trunc = False

#         bids, feats, bid2text = obs_to_candidates(obs, max_candidates=max_candidates, text_dim=text_dim)
#         print(f"\n[t={t}] num_candidates={len(bids)} sample={bids[:10]}")

#         if len(bids) == 0:
#             print("[FAIL] num_candidates=0 (check bbox parsing / candidate filters)")
#             return False

#         debug_print_candidates(obs, bids, bid2text, k=25)

#         # Try each bid once
#         for bid in bids:
#             action = {"action": "click", "bid": bid}
#             obs2, reward, term, trunc, info = env.step(action)
#             r = float(reward)

#             if r > 0:
#                 print(f"\n[SUCCESS] clicked bid={bid} reward={r}")
#                 return True

#             # If the environment ended, stop trying further bids
#             if term or trunc:
#                 obs = obs2
#                 break

#             # continue from updated obs
#             obs = obs2

#         if term or trunc:
#             print("[info] episode ended (term/trunc) without success")
#             break

#     print("\n[FAIL] never got positive reward in bruteforce")
#     return False
def sanity_bruteforce_episode(env, max_steps=10, max_candidates=128, text_dim=128):
    print("\n=== SANITY BRUTEFORCE ===")
    obs, info = env.reset()

    goal = obs.get("goal", "")
    url = obs.get("url", "")
    print("goal:", goal)
    print("url :", url)

    for t in range(max_steps):
        # Always define these so they can't be unbound
        term = False
        trunc = False

        bids, feats, bid2text = obs_to_candidates(
            obs, max_candidates=max_candidates, text_dim=text_dim
        )

        print(f"\n[t={t}] num_candidates={len(bids)} sample={bids[:min(10,len(bids))]}")
        debug_print_candidates(obs, bids, bid2text, k=min(25, len(bids)))

        if len(bids) == 0:
            print("[FAIL] no candidates")
            return False

        # Click each bid once
        for bid in bids:
            action = {"action": "click", "bid": str(bid)}
            print(f"[try] click bid={bid!r} text={bid2text.get(str(bid), '')!r}")

            obs2, reward, term, trunc, info = env.step(action)
            r = float(reward)
            print(f"      reward={r} term={term} trunc={trunc}")

            if r > 0.0:
                print(f"\n[SUCCESS] clicked bid={bid} reward={r}")
                return True

            obs = obs2
            if term or trunc:
                print("[info] episode ended (term/trunc) before success")
                return False

    print("\n[FAIL] never got positive reward in bruteforce")
    return False
# -----------------------------
# PPO model
# -----------------------------
class BidPolicyValue(nn.Module):
    def __init__(self, goal_dim: int, bid_feat_dim: int, hidden: int = 128):
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
        N = bid_feats.shape[0]
        if N == 0:
            return torch.empty((0,), dtype=torch.float32, device=bid_feats.device), self.value(goal_vec).squeeze(-1)
        goal_rep = goal_vec.unsqueeze(0).expand(N, -1)
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


def compute_gae(steps: List[Step], gamma: float, lam: float, last_value: float = 0.0):
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
    p.add_argument("--text-dim", type=int, default=128)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--headless", action="store_true")
    p.add_argument("--disable-checker", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--sanity-bruteforce", action="store_true")
    args = p.parse_args()

    if not os.environ.get("MINIWOB_URL"):
        raise RuntimeError(
            "MINIWOB_URL is not set.\n"
            "Example:\n"
            "  export MINIWOB_URL=http://127.0.0.1:8000/miniwob/\n"
            "And run: python -m http.server 8000  (from miniwob-plusplus/miniwob/html)"
        )

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    env_kwargs = dict(headless=args.headless)
    if args.disable_checker:
        env_kwargs["disable_env_checker"] = True

    env = gym.make(args.env_id, **env_kwargs)

    if args.sanity_bruteforce:
        ok = sanity_bruteforce_episode(
            env,
            max_steps=10,
            max_candidates=args.max_candidates,
            text_dim=args.text_dim,
        )
        print("sanity result:", ok)
        return

    # ---- PPO train ----
    bid_feat_dim = 7 + args.text_dim
    model = BidPolicyValue(goal_dim=args.goal_dim, bid_feat_dim=bid_feat_dim, hidden=args.hidden).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    obs, info = env.reset()
    episodes_completed = 0

    for it in range(args.total_iters):
        steps: List[Step] = []
        entropies: List[torch.Tensor] = []
        successes_in_batch = 0

        while len(steps) < args.steps_per_iter:
            goal = obs.get("goal", "")
            goal_vec = hash_goal_vec(goal, dim=args.goal_dim).to(device)

            bids, bid_feats, bid2text = obs_to_candidates(
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

        with torch.no_grad():
            goal_vec = hash_goal_vec(obs.get("goal", ""), dim=args.goal_dim).to(device)
            bids, bid_feats, _ = obs_to_candidates(obs, max_candidates=args.max_candidates, text_dim=args.text_dim)
            bid_feats = bid_feats.to(device)
            _, last_v = model(goal_vec, bid_feats) if len(bids) > 0 else (None, torch.tensor(0.0, device=device))
            last_v = float(last_v.item())

        returns, adv = compute_gae(steps, gamma=args.gamma, lam=args.gae_lam, last_value=last_v)
        returns = returns.to(device)
        adv = adv.to(device)

        for _ in range(args.ppo_epochs):
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
            f"[iter {it:03d}] avg_step_reward={avg_reward:.3f} "
            f"pos_reward_frac={pos_frac:.2f} entropy={avg_ent:.2f} "
            f"episodes_completed={episodes_completed} successes_in_batch={successes_in_batch}/{len(steps)}"
        )

    env.close()


if __name__ == "__main__":
    main()