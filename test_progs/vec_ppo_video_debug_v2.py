#!/usr/bin/env python3
"""
vec_ppo_video.py — Vectorized MiniWoB PPO (click-by-bid) for BrowserGym
+ MP4 recording DURING TRAINING (records ONE env index, saves every K iters)
+ Overlay ALL candidate boxes (top-K clickable BIDs), highlight chosen BID, show goal text
+ Optionally draw each candidate's (truncated) text label near its box.
+ Trajectory printing + JSONL dumping for ONE chosen env index during training

Notes:
- Gymnasium VectorEnv may return observations either as:
  (a) np.ndarray(dtype=object) of per-env dicts, OR
  (b) a dict-of-arrays (batched by env) depending on wrapper.
  This script normalizes both into a list[dict] of length num_envs.

Deps:
  pip install -U gymnasium torch numpy browsergym-miniwob playwright
  python -m playwright install
Optional (better text features + labels):
  pip install -U beautifulsoup4 lxml
Video deps:
  pip install -U imageio imageio-ffmpeg opencv-python

Runtime:
  export MINIWOB_URL=http://127.0.0.1:8000/miniwob/
"""

import os
import json
import time
import argparse
from typing import Dict, Any, List, Tuple, Optional
from collections import deque

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
# Observation normalization (VectorEnv output -> list[dict])
# -----------------------------
def obs_to_list(obs: Any, num_envs: int) -> List[Dict[str, Any]]:
    """
    Normalize VectorEnv observation into list of per-env dicts.
    Handles:
      - np.ndarray(dtype=object) length N, each element is dict
      - dict-of-arrays: {k: np.array([v0..vN-1])}
    """
    # Case 1: already a list-like of dicts
    if isinstance(obs, (list, tuple)):
        if len(obs) == num_envs and all(isinstance(x, dict) for x in obs):
            return list(obs)

    # Case 2: object array
    if isinstance(obs, np.ndarray):
        if obs.dtype == object and len(obs) == num_envs:
            out = list(obs)
            if all(isinstance(x, dict) for x in out):
                return out

    # Case 3: dict-of-batched values
    if isinstance(obs, dict):
        out: List[Dict[str, Any]] = []
        for i in range(num_envs):
            d = {}
            for k, v in obs.items():
                try:
                    d[k] = v[i]
                except Exception:
                    d[k] = v
            out.append(d)
        return out

    # Fallback: best-effort wrap
    return [obs for _ in range(num_envs)]


# -----------------------------
# Trajectory helpers
# -----------------------------
def _safe_str(x: Any, max_len: int = 120) -> str:
    s = "" if x is None else str(x)
    s = s.replace("\n", " ").replace("\r", " ")
    return s if len(s) <= max_len else (s[: max_len - 3] + "...")


def format_step_line(*, it: int, t: int, env_idx: int, goal: str, chosen_bid: str, reward: float, done: bool, n_cands: int):
    return (
        f"[traj it={it:03d} env={env_idx}] t={t:04d} "
        f"r={reward:+.2f} done={int(done)} "
        f"chosen={chosen_bid} #cands={n_cands} goal='{_safe_str(goal, 80)}'"
    )


def topk_candidates_line(bids: List[str], probs_1d: torch.Tensor, k: int) -> str:
    k = min(int(k), len(bids))
    if k <= 0 or len(bids) == 0:
        return ""
    probs_1d = probs_1d[: len(bids)]
    vals, idxs = torch.topk(probs_1d, k=k)
    parts = []
    for p, j in zip(vals.tolist(), idxs.tolist()):
        parts.append(f"{bids[j]}:{p:.3f}")
    return "topk: " + ", ".join(parts)


# -----------------------------
# Video helpers
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


def _clamp_box(box: Tuple[int, int, int, int], w: int, h: int) -> Tuple[int, int, int, int]:
    x0, y0, x1, y1 = box
    x0 = max(0, min(w - 1, x0))
    x1 = max(0, min(w - 1, x1))
    y0 = max(0, min(h - 1, y0))
    y1 = max(0, min(h - 1, y1))
    return x0, y0, x1, y1


class RollingVideoRecorder:
    """
    Ring-buffer MP4 recorder: stores last max_frames, saves at end of rollout.

    Overlay:
      - Draw all candidate boxes (thin)
      - Optionally draw candidate text labels (tiny)
      - Highlight chosen action box (thick)
      - HUD: t, reward, chosen bid, #cands, goal snippet
    """

    def __init__(
        self,
        path: str,
        fps: int = 5,
        max_frames: int = 300,
        draw_max_cands: int = 64,
        draw_labels: bool = True,
        label_max_chars: int = 18,
    ):
        self.path = path
        self.fps = fps
        self.draw_max_cands = draw_max_cands
        self.draw_labels = draw_labels
        self.label_max_chars = label_max_chars
        self._buf = deque(maxlen=max_frames)

    def add_frame(
        self,
        obs: Dict[str, Any],
        *,
        candidate_bids: List[str],
        bid2text: Dict[str, str],
        chosen_bid: str,
        reward: float,
        step: int,
    ):
        if cv2 is None or imageio is None:
            return
        img = obs.get("screenshot")
        if img is None:
            return

        frame = np.array(img, copy=True)
        if frame.dtype != np.uint8:
            frame = frame.astype(np.uint8)

        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        h, w = frame_bgr.shape[:2]

        # --- draw candidate boxes (thin, cyan-ish) ---
        cand_to_draw = candidate_bids[: self.draw_max_cands]
        for bid in cand_to_draw:
            box = _get_bbox_for_bid(obs, bid)
            if box is None:
                continue
            x0, y0, x1, y1 = _clamp_box(box, w, h)
            if x1 <= x0 or y1 <= y0:
                continue
            cv2.rectangle(frame_bgr, (x0, y0), (x1, y1), (255, 255, 0), 1)

            if self.draw_labels:
                txt = (bid2text.get(bid, "") or "").replace("\n", " ").strip()
                if not txt:
                    txt = f"bid:{bid}"
                if len(txt) > self.label_max_chars:
                    txt = txt[: self.label_max_chars - 3] + "..."
                # small label near top-left
                cv2.putText(
                    frame_bgr,
                    txt,
                    (x0 + 2, max(12, y0 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.35,
                    (0, 0, 0),
                    3,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    frame_bgr,
                    txt,
                    (x0 + 2, max(12, y0 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.35,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

        # --- highlight chosen bid (thick, red) ---
        chosen_box = _get_bbox_for_bid(obs, chosen_bid)
        if chosen_box is not None:
            x0, y0, x1, y1 = _clamp_box(chosen_box, w, h)
            if x1 > x0 and y1 > y0:
                cv2.rectangle(frame_bgr, (x0, y0), (x1, y1), (0, 0, 255), 3)

        # --- HUD text ---
        goal = (obs.get("goal", "") or "").replace("\n", " ")
        if len(goal) > 80:
            goal = goal[:77] + "..."
        hud1 = f"t={step} r={reward:.2f} chosen_bid={chosen_bid}  #cands={len(candidate_bids)}"
        hud2 = f"goal: {goal}"

        def put_line(y, txt):
            cv2.putText(frame_bgr, txt, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(frame_bgr, txt, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2, cv2.LINE_AA)

        put_line(28, hud1)
        put_line(54, hud2)

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        self._buf.append(frame_rgb)

    def save(self):
        if cv2 is None or imageio is None:
            print("[warn] video deps missing; install: pip install imageio imageio-ffmpeg opencv-python")
            return
        if not self._buf:
            print("[warn] no frames recorded; not saving video.")
            return
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        frames = list(self._buf)
        print(f"[info] saving mp4: {self.path} ({len(frames)} frames @ {self.fps} fps)")
        imageio.mimsave(self.path, frames, fps=self.fps)


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
        N, K, _ = cand_feats.shape
        goal_rep = goal_vecs.unsqueeze(1).expand(N, K, self.goal_dim)
        x = torch.cat([goal_rep, cand_feats], dim=-1)
        logits = self.policy(x).squeeze(-1)  # (N,K)
        values = self.value(goal_vecs).squeeze(-1)  # (N,)
        return logits, values


@torch.no_grad()
def batch_obs_to_tensors(
    obs_list: List[Dict[str, Any]],
    max_candidates: int,
    goal_dim: int,
    text_dim: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[List[str]], List[Dict[str, str]]]:
    """
    Returns:
      goal_vecs: (N,G)
      cand_feats: (N,K,D)
      mask: (N,K) bool
      bids_list: list[list[str]] length N
      bid2text_list: list[dict] length N (for labels/debug)
    """
    N = len(obs_list)
    K = max_candidates
    D = 7 + text_dim

    goal_vecs = torch.zeros((N, goal_dim), dtype=torch.float32, device=device)
    cand_feats = torch.zeros((N, K, D), dtype=torch.float32, device=device)
    mask = torch.zeros((N, K), dtype=torch.bool, device=device)
    bids_list: List[List[str]] = []
    bid2text_list: List[Dict[str, str]] = []

    include_text = (bs4 is not None)

    for i, obs in enumerate(obs_list):
        goal_vecs[i] = hash_goal_vec(obs.get("goal", ""), goal_dim).to(device)
        bids, feats, bid2text = obs_to_candidates(
            obs, max_candidates=max_candidates, text_dim=text_dim, include_text=include_text
        )
        bids_list.append(bids)
        bid2text_list.append(bid2text)

        n = min(len(bids), K)
        if n > 0:
            cand_feats[i, :n] = feats[:n].to(device)
            mask[i, :n] = True
        else:
            # allow a dummy slot so categorical doesn't crash
            mask[i, 0] = True

    return goal_vecs, cand_feats, mask, bids_list, bid2text_list


def masked_categorical(logits: torch.Tensor, mask: torch.Tensor):
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
    adv_flat = adv.reshape(-1)
    adv = (adv - adv_flat.mean()) / (adv_flat.std() + 1e-8)
    return ret, adv


def make_env(env_id: str, headless: bool, disable_checker: bool, idx: int):
    def _init():
        print(f"[init] creating env[{idx}] ...", flush=True)
        kwargs = dict(headless=headless)
        if disable_checker:
            kwargs["disable_env_checker"] = True
        env = gym.make(env_id, **kwargs)
        print(f"[init] created env[{idx}]", flush=True)
        return env
    return _init


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

    p.add_argument("--init-only", action="store_true", help="Create envs + reset once, then exit (debug startup hangs).")

    # ---- training-time MP4 recording ----
    p.add_argument("--train-mp4-dir", type=str, default="", help="If set, save training rollout videos here.")
    p.add_argument("--train-mp4-every", type=int, default=5, help="Save a video every K iters.")
    p.add_argument("--train-mp4-env-idx", type=int, default=0, help="Which env index to record (0..num_envs-1).")
    p.add_argument("--train-mp4-fps", type=int, default=5)
    p.add_argument("--train-mp4-max-frames", type=int, default=300)
    p.add_argument("--train-mp4-draw-max-cands", type=int, default=64, help="Max candidate boxes to draw per frame.")
    p.add_argument("--train-mp4-draw-labels", action="store_true", help="Draw candidate text labels near boxes.")
    p.add_argument("--train-mp4-label-max-chars", type=int, default=18)

    # ---- trajectory printing / dumping (for one env index) ----
    p.add_argument("--traj-env-idx", type=int, default=-1,
                   help="Which env index to print/dump trajectories for. -1 -> use train-mp4-env-idx.")
    p.add_argument("--traj-print", action="store_true",
                   help="Print per-step trajectory lines for traj-env-idx.")
    p.add_argument("--traj-print-topk", type=int, default=0,
                   help="If >0, also print top-k candidate bids w/ probs each step (can be noisy).")
    p.add_argument("--traj-print-at-end", action="store_true",
                   help="Print full episode trajectory when that env ends.")
    p.add_argument("--traj-jsonl", type=str, default="",
                   help="If set, append one JSON line per finished episode trajectory.")
    p.add_argument("--traj-max-steps", type=int, default=5000,
                   help="Safety cap: keep at most this many steps in a single episode trajectory buffer.")

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

    if args.train_mp4_dir and (cv2 is None or imageio is None):
        print("[warn] video deps missing; install: pip install imageio imageio-ffmpeg opencv-python")
        print("       disabling training videos.")
        args.train_mp4_dir = ""

    print("[startup] building vector envs...", flush=True)
    # Vector env (use Gymnasium's autoreset to avoid per-env reset APIs)
    has_autoreset_mode = False
    try:
        # Gymnasium >=0.29
        envs = gym.vector.SyncVectorEnv(
            [make_env(args.env_id, args.headless, args.disable_checker, i) for i in range(args.num_envs)],
            autoreset_mode=gym.vector.AutoresetMode.SAME_STEP,
        )
        has_autoreset_mode = True
    except Exception:
        # Older Gymnasium: no autoreset_mode; fall back to basic SyncVectorEnv and reset-all on done.
        envs = gym.vector.SyncVectorEnv(
            [make_env(args.env_id, args.headless, args.disable_checker, i) for i in range(args.num_envs)]
        )
        has_autoreset_mode = False

    print("[startup] vector envs ready", flush=True)

    bid_feat_dim = 7 + args.text_dim
    model = BatchedBidPolicyValue(args.goal_dim, bid_feat_dim, args.hidden).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    print("[startup] resetting envs...", flush=True)
    obs, _ = envs.reset(seed=args.seed)
    obs_list = obs_to_list(obs, args.num_envs)
    print("[startup] reset complete", flush=True)
    if args.init_only:
        print("[startup] --init-only set; exiting after reset.", flush=True)
        envs.close()
        return

    K = args.max_candidates
    T = args.steps_per_env
    N = args.num_envs

    rec_env = max(0, min(N - 1, int(args.train_mp4_env_idx)))
    traj_env = rec_env if args.traj_env_idx < 0 else max(0, min(N - 1, int(args.traj_env_idx)))

    traj_buf: List[Dict[str, Any]] = []
    traj_ep_id = 0
    if args.traj_jsonl:
        os.makedirs(os.path.dirname(args.traj_jsonl) or ".", exist_ok=True)

    for it in range(args.total_iters):
        actions = torch.zeros((T, N), dtype=torch.long, device=device)
        old_logp = torch.zeros((T, N), dtype=torch.float32, device=device)
        values = torch.zeros((T, N), dtype=torch.float32, device=device)
        rewards = torch.zeros((T, N), dtype=torch.float32, device=device)
        dones = torch.zeros((T, N), dtype=torch.float32, device=device)
        entropies = torch.zeros((T, N), dtype=torch.float32, device=device)

        goal_buf = torch.zeros((T, N, args.goal_dim), dtype=torch.float32, device=device)
        cand_buf = torch.zeros((T, N, K, bid_feat_dim), dtype=torch.float32, device=device)
        mask_buf = torch.zeros((T, N, K), dtype=torch.bool, device=device)

        successes = 0

        # ---- per-iter recorder ----
        rec: Optional[RollingVideoRecorder] = None
        record_this_iter = bool(args.train_mp4_dir) and (it % args.train_mp4_every == 0)
        if record_this_iter:
            os.makedirs(args.train_mp4_dir, exist_ok=True)
            rec_path = os.path.join(args.train_mp4_dir, f"train_iter_{it:03d}.mp4")
            rec = RollingVideoRecorder(
                rec_path,
                fps=args.train_mp4_fps,
                max_frames=args.train_mp4_max_frames,
                draw_max_cands=args.train_mp4_draw_max_cands,
                draw_labels=args.train_mp4_draw_labels,
                label_max_chars=args.train_mp4_label_max_chars,
            )

        # ---- rollout ----
        for t in range(T):
            goal_vecs, cand_feats, mask, bids_list, bid2text_list = batch_obs_to_tensors(
                obs_list, args.max_candidates, args.goal_dim, args.text_dim, device
            )

            goal_buf[t] = goal_vecs
            cand_buf[t] = cand_feats
            mask_buf[t] = mask

            with torch.no_grad():
                logits, v = model(goal_vecs, cand_feats)
                dist = masked_categorical(logits, mask)
                a = dist.sample()
                lp = dist.log_prob(a)
                ent = dist.entropy()

                masked_logits = logits.masked_fill(~mask, -1e9)
                probs = torch.softmax(masked_logits, dim=-1)  # (N,K)

            actions[t] = a
            old_logp[t] = lp
            values[t] = v
            entropies[t] = ent

            a_cpu = a.detach().cpu().numpy()
            act_strs = []
            for i in range(N):
                bids = bids_list[i]
                idx = int(a_cpu[i])
                if idx >= len(bids) or len(bids) == 0:
                    idx = 0
                bid = bids[idx] if bids else "0"
                act_strs.append(click_action(bid))

            # ---- record pre-step obs for one env (video) ----
            if rec is not None:
                bids_r = bids_list[rec_env]
                idx_r = int(a_cpu[rec_env])
                if len(bids_r) == 0:
                    chosen_bid = "0"
                else:
                    idx_r = min(idx_r, len(bids_r) - 1)
                    chosen_bid = bids_r[idx_r]
                rec_obs = obs_list[rec_env]
                rec_cands = bids_r[:]
                rec_bid2text = bid2text_list[rec_env]
            else:
                rec_obs = None
                chosen_bid = None
                rec_cands = None
                rec_bid2text = None

            # ---- trajectory capture pre-step (for traj_env) ----
            do_traj = bool(args.traj_print or args.traj_print_at_end or args.traj_jsonl)
            if do_traj:
                obs_tr = obs_list[traj_env]
                bids_tr = bids_list[traj_env]
                a_tr = int(a_cpu[traj_env]) if N > 0 else 0
                if len(bids_tr) == 0:
                    chosen_tr_bid = "0"
                    a_tr = 0
                else:
                    a_tr = min(max(a_tr, 0), len(bids_tr) - 1)
                    chosen_tr_bid = bids_tr[a_tr]

                if len(traj_buf) < int(args.traj_max_steps):
                    traj_buf.append({
                        "it": int(it),
                        "t": int(t),
                        "goal": obs_tr.get("goal", ""),
                        "chosen_bid": chosen_tr_bid,
                        "action_str": act_strs[traj_env],
                        "num_candidates": int(len(bids_tr)),
                        "candidates_head": bids_tr[: min(10, len(bids_tr))],
                        "reward": None,
                        "done": None,
                    })

                if args.traj_print:
                    print(
                        format_step_line(
                            it=it, t=t, env_idx=traj_env,
                            goal=obs_tr.get("goal", ""),
                            chosen_bid=chosen_tr_bid,
                            reward=0.0,
                            done=False,
                            n_cands=len(bids_tr),
                        ),
                        flush=True,
                    )
                    if args.traj_print_topk and args.traj_print_topk > 0:
                        try:
                            topk = topk_candidates_line(
                                bids_tr,
                                probs[traj_env].detach().cpu(),
                                args.traj_print_topk,
                            )
                            if topk:
                                print("   " + topk, flush=True)
                        except Exception:
                            pass

            # ---- step envs ----
            obs2, rew, term, trunc, _ = envs.step(np.array(act_strs, dtype=object))
            done = np.logical_or(term, trunc)

            rewards[t] = torch.tensor(rew, dtype=torch.float32, device=device)
            dones[t] = torch.tensor(done.astype(np.float32), device=device)
            successes += int((np.array(rew) > 0).sum())

            # ---- add video frame (pre-step obs) ----
            if rec is not None and rec_obs is not None and chosen_bid is not None and rec_cands is not None:
                rec.add_frame(
                    rec_obs,
                    candidate_bids=rec_cands,
                    bid2text=(rec_bid2text or {}),
                    chosen_bid=chosen_bid,
                    reward=float(rew[rec_env]),
                    step=t,
                )

            # ---- trajectory: fill post-step reward/done, maybe dump episode ----
            if do_traj:
                r_tr = float(rew[traj_env])
                d_tr = bool(done[traj_env])

                if traj_buf:
                    traj_buf[-1]["reward"] = r_tr
                    traj_buf[-1]["done"] = d_tr

                if args.traj_print:
                    print(f"   -> r={r_tr:+.2f} done={int(d_tr)}", flush=True)

                if (args.traj_print_at_end or args.traj_jsonl) and d_tr:
                    traj_ep_id += 1

                    if args.traj_print_at_end:
                        print("\n=== TRAJECTORY END ===", flush=True)
                        for s in traj_buf:
                            print(
                                f"it={int(s.get('it', 0)):03d} t={int(s.get('t', 0)):04d} "
                                f"r={float(s.get('reward') or 0.0):+0.2f} done={int(bool(s.get('done')))} "
                                f"chosen={s.get('chosen_bid')} goal='{_safe_str(s.get('goal',''), 80)}'",
                                flush=True
                            )
                        print("=== /TRAJECTORY END ===\n", flush=True)

                    if args.traj_jsonl:
                        recj = {
                            "timestamp": time.time(),
                            "env_id": args.env_id,
                            "traj_env_idx": int(traj_env),
                            "episode_id": int(traj_ep_id),
                            "steps": traj_buf,
                        }
                        with open(args.traj_jsonl, "a", encoding="utf-8") as f:
                            f.write(json.dumps(recj) + "\n")

                    traj_buf.clear()

            # If autoreset_mode isn't supported (older Gymnasium), simplest safe fallback:
            # reset ALL envs when ANY env is done.
            if done.any() and not has_autoreset_mode:
                obs2, _ = envs.reset(seed=args.seed + it * 100000 + t)

            obs_list = obs_to_list(obs2, args.num_envs)

        # ---- on_rollout_end ----
        if rec is not None:
            rec.save()

        # bootstrap last values
        with torch.no_grad():
            goal_last, cand_last, mask_last, _, _ = batch_obs_to_tensors(
                obs_list, args.max_candidates, args.goal_dim, args.text_dim, device
            )
            _, last_v = model(goal_last, cand_last)

        ret, adv = compute_gae_vec(rewards, dones, values, last_v, args.gamma, args.gae_lam)

        # ---- PPO update ----
        B = T * N
        flat_goal = goal_buf.reshape(B, args.goal_dim)
        flat_cand = cand_buf.reshape(B, K, bid_feat_dim)
        flat_mask = mask_buf.reshape(B, K)
        flat_actions = actions.reshape(B)
        flat_old_logp = old_logp.reshape(B)
        flat_ret = ret.reshape(B)
        flat_adv = adv.reshape(B)

        idx = torch.randperm(B, device=device)

        for _ in range(args.ppo_epochs):
            for start in range(0, B, args.batch_size):
                mb = idx[start : start + args.batch_size]

                g = flat_goal[mb]
                c = flat_cand[mb]
                m = flat_mask[mb]
                a_mb = flat_actions[mb]
                oldlp = flat_old_logp[mb]
                tgt_ret = flat_ret[mb]
                tgt_adv = flat_adv[mb]

                logits, v = model(g, c)
                dist = masked_categorical(logits, m)

                newlp = dist.log_prob(a_mb)
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