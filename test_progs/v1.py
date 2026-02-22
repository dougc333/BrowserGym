#!/usr/bin/env python3
"""
MiniWoB PPO (click-by-bid) baseline for BrowserGym + MP4 recording.

Adds:
  - --record-mp4 PATH (records an eval episode after training)
  - draws a red bbox around the clicked BID and overlays text (step/bid/reward)
Deps:
  pip install -U imageio imageio-ffmpeg opencv-python
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

# ---- video deps (optional) ----
try:
    import cv2
    import imageio.v2 as imageio
except Exception:
    cv2 = None
    imageio = None


# -----------------------------
# MP4 recorder with bbox highlight
# -----------------------------
def _get_bbox_for_bid(obs: Dict[str, Any], bid: str) -> Optional[Tuple[int, int, int, int]]:
    extra = obs.get("extra_element_properties")
    if not isinstance(extra, dict):
        return None
    props = extra.get(str(bid))
    if not isinstance(props, dict):
        return None
    bbox = props.get("bbox")
    if not (isinstance(bbox, (list, tuple)) and len(bbox) == 4):
        return None
    try:
        x0, y0, x1, y1 = map(float, bbox)
        x0i, y0i, x1i, y1i = int(round(x0)), int(round(y0)), int(round(x1)), int(round(y1))
        if x1i <= x0i or y1i <= y0i:
            return None
        return x0i, y0i, x1i, y1i
    except Exception:
        return None


class MiniWoBVideoRecorder:
    def __init__(self, path: str, fps: int = 5):
        self.path = path
        self.fps = fps
        self.frames: List[np.ndarray] = []

    def add_frame(self, obs: Dict[str, Any], action_bid: str, reward: float, step: int):
        if imageio is None or cv2 is None:
            return
        img = obs.get("screenshot")
        if img is None:
            return

        frame = np.array(img, copy=True)
        if frame.dtype != np.uint8:
            frame = frame.astype(np.uint8)

        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # bbox highlight
        bbox = _get_bbox_for_bid(obs, action_bid)
        if bbox is not None:
            x0, y0, x1, y1 = bbox
            h, w = frame_bgr.shape[:2]
            x0 = max(0, min(w - 1, x0)); x1 = max(0, min(w - 1, x1))
            y0 = max(0, min(h - 1, y0)); y1 = max(0, min(h - 1, y1))
            if x1 > x0 and y1 > y0:
                cv2.rectangle(frame_bgr, (x0, y0), (x1, y1), (0, 0, 255), 3)  # red

        # overlay text
        text = f"step={step} bid={action_bid} reward={reward:.2f}"
        cv2.putText(
            frame_bgr, text, (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7,
            (0, 255, 0), 2, cv2.LINE_AA
        )

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        self.frames.append(frame_rgb)

    def save(self):
        if imageio is None or cv2 is None:
            print("[warn] video deps missing; install: pip install imageio imageio-ffmpeg opencv-python")
            return
        if not self.frames:
            print("[warn] no frames recorded; not saving video.")
            return
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        print(f"[info] saving mp4: {self.path} ({len(self.frames)} frames @ {self.fps} fps)")
        imageio.mimsave(self.path, self.frames, fps=self.fps)


# -----------------------------
# Hashing text -> vector
# -----------------------------
def hash_text_to_vec(text: str, dim: int) -> torch.Tensor:
    """Char 3-gram hashing trick -> [dim]."""
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
# Candidates -> bids + features
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
# Model
# -----------------------------
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
    t = len(steps)
    rewards = torch.tensor([s.reward for s in steps], dtype=torch.float32)
    dones = torch.tensor([1.0 if s.done else 0.0 for s in steps], dtype=torch.float32)
    values = torch.stack([s.value for s in steps]).detach().float()

    adv = torch.zeros(t, dtype=torch.float32)
    gae = 0.0
    for i in reversed(range(t)):
        next_v = last_value if i == t - 1 else values[i + 1].item()
        delta = rewards[i].item() + gamma * next_v * (1.0 - dones[i].item()) - values[i].item()
        gae = delta + gamma * lam * (1.0 - dones[i].item()) * gae
        adv[i] = gae

    ret = adv + values
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    return ret, adv


# -----------------------------
# Main
# -----------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--env-id", default="browsergym/miniwob.click-button")
    p.add_argument("--total-iters", type=int, default=30)
    p.add_argument("--steps-per-iter", type=int, default=512)
    p.add_argument("--ppo-epochs", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae-lam", type=float, default=0.95)
    p.add_argument("--clip", type=float, default=0.2)
    p.add_argument("--ent-coef", type=float, default=0.02)
    p.add_argument("--vf-coef", type=float, default=0.5)
    p.add_argument("--max-candidates", type=int, default=64)
    p.add_argument("--goal-dim", type=int, default=256)
    p.add_argument("--text-dim", type=int, default=128)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--headless", action="store_true")
    p.add_argument("--disable-checker", action="store_true")
    p.add_argument("--debug-dom", action="store_true")
    p.add_argument("--max-steps", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)

    # NEW: record an eval episode after training
    p.add_argument("--record-mp4", type=str, default="", help="If set, save an eval rollout mp4 here.")
    p.add_argument("--record-fps", type=int, default=5)
    p.add_argument("--record-max-steps", type=int, default=200)

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

    if args.record_mp4 and (cv2 is None or imageio is None):
        print("[warn] video deps missing; install: pip install imageio imageio-ffmpeg opencv-python")
        print("       will continue without recording.")
        args.record_mp4 = ""

    env_kwargs = dict(headless=args.headless)
    if args.disable_checker:
        env_kwargs["disable_env_checker"] = True

    env = gym.make(args.env_id, **env_kwargs)

    bid_feat_dim = 7 + args.text_dim
    model = BidPolicyValue(args.goal_dim, bid_feat_dim, args.hidden).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    obs, _ = env.reset(seed=args.seed)
    episodes_completed = 0
    steps_since_reset = 0

    for it in range(args.total_iters):
        steps: List[Step] = []
        entropies: List[torch.Tensor] = []
        successes = 0

        while len(steps) < args.steps_per_iter:
            goal = obs.get("goal", "")
            goal_vec = hash_goal_vec(goal, args.goal_dim).to(device)

            bids, bid_feats, bid2text = obs_to_candidates(
                obs,
                max_candidates=args.max_candidates,
                text_dim=args.text_dim,
                include_text=(bs4 is not None),
            )
            bid_feats = bid_feats.to(device)

            if args.debug_dom and len(steps) == 0:
                sample = list(bid2text.items())[:8]
                if sample:
                    print(f"[iter {it:03d}] goal={goal!r}")
                    print(f"[iter {it:03d}] sample bid->text:", sample)

            if not bids:
                obs, _ = env.reset(seed=args.seed + episodes_completed + 1)
                steps_since_reset = 0
                continue

            logits, value = model(goal_vec, bid_feats)
            a_idx, logp, ent = sample_action(logits)
            entropies.append(ent)

            action_str = click_action(bids[a_idx])

            obs2, reward, terminated, truncated, _ = env.step(action_str)
            done = bool(terminated) or bool(truncated)

            err = (obs2.get("last_action_error") or "").strip()
            if err and err != "None":
                print("[warn] last_action_error:", err, " last_action:", obs2.get("last_action"))

            r = float(reward)
            if r > 0:
                successes += 1

            steps.append(
                Step(
                    goal_vec=goal_vec.detach(),
                    bid_feats=bid_feats.detach(),
                    action_idx=a_idx,
                    logp=logp.detach(),
                    value=value.detach(),
                    reward=r,
                    done=done,
                )
            )

            obs = obs2
            steps_since_reset += 1

            if args.max_steps > 0 and steps_since_reset >= args.max_steps:
                done = True

            if done:
                episodes_completed += 1
                obs, _ = env.reset(seed=args.seed + episodes_completed + 1)
                steps_since_reset = 0

        # bootstrap
        with torch.no_grad():
            g = hash_goal_vec(obs.get("goal", ""), args.goal_dim).to(device)
            bids, bf, _ = obs_to_candidates(
                obs, args.max_candidates, args.text_dim, include_text=(bs4 is not None)
            )
            bf = bf.to(device)
            if bids:
                _, last_v = model(g, bf)
                last_v = float(last_v.item())
            else:
                last_v = 0.0

        returns, adv = compute_gae(steps, args.gamma, args.gae_lam, last_v)
        returns = returns.to(device)
        adv = adv.to(device)

        # PPO update
        for _ in range(args.ppo_epochs):
            pol = 0.0
            vf = 0.0
            entl = 0.0

            for i, s in enumerate(steps):
                gv = s.goal_vec.to(device)
                bf = s.bid_feats.to(device)
                logits, v = model(gv, bf)
                dist = torch.distributions.Categorical(logits=logits)

                a = torch.tensor(s.action_idx, device=device)
                new_logp = dist.log_prob(a)
                ratio = torch.exp(new_logp - s.logp.to(device))

                surr1 = ratio * adv[i]
                surr2 = torch.clamp(ratio, 1.0 - args.clip, 1.0 + args.clip) * adv[i]
                pol = pol + (-torch.min(surr1, surr2))

                vf = vf + (v - returns[i]).pow(2)
                entl = entl + (-dist.entropy())

            pol = pol / len(steps)
            vf = vf / len(steps)
            entl = entl / len(steps)

            loss = pol + args.vf_coef * vf + args.ent_coef * entl

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
            f"episodes_completed={episodes_completed} successes_in_batch={successes}/{len(steps)}"
        )

    # ---- record an eval episode mp4 (optional) ----
    if args.record_mp4:
        model.eval()
        rec = MiniWoBVideoRecorder(args.record_mp4, fps=args.record_fps)

        obs, _ = env.reset(seed=args.seed + 12345)
        for t in range(args.record_max_steps):
            goal_vec = hash_goal_vec(obs.get("goal", ""), args.goal_dim).to(device)
            bids, bid_feats, _ = obs_to_candidates(
                obs,
                max_candidates=args.max_candidates,
                text_dim=args.text_dim,
                include_text=(bs4 is not None),
            )
            bid_feats = bid_feats.to(device)

            if not bids:
                obs, _ = env.reset(seed=args.seed + 12345 + t)
                continue

            with torch.no_grad():
                logits, _ = model(goal_vec, bid_feats)
                a_idx = int(torch.argmax(logits).item())

            a_idx = min(a_idx, len(bids) - 1)
            bid = bids[a_idx]
            action_str = click_action(bid)

            obs2, reward, terminated, truncated, _ = env.step(action_str)
            rec.add_frame(obs, action_bid=bid, reward=float(reward), step=t)

            obs = obs2
            if terminated or truncated:
                break

        rec.save()
        model.train()

    env.close()


if __name__ == "__main__":
    main()