#!/usr/bin/env python3
import os
import re
import json
import argparse
from typing import Dict, Any, List, Tuple

import numpy as np
import gymnasium as gym
import browsergym.miniwob  # registers envs

from PIL import Image

# If you use transformers VLMs:
import torch
from transformers import AutoProcessor, AutoModelForVision2Seq


SYSTEM_PROMPT = """You are a browser agent.
You will be given a screenshot and a goal.
You MUST output exactly one action in this format:

click('<bid>')

Rules:
- Output ONLY the action string. No other text.
- <bid> must be one of the clickable element ids in the page.
"""

CLICK_RE = re.compile(r"click\('([^']+)'\)")


def obs_screenshot_to_pil(obs: Dict[str, Any]) -> Image.Image:
    img = obs["screenshot"]  # (H,W,3) uint8
    if not isinstance(img, np.ndarray):
        img = np.array(img)
    return Image.fromarray(img.astype(np.uint8))


def get_clickable_bids(obs: Dict[str, Any], max_bids: int = 64) -> List[str]:
    extra = obs.get("extra_element_properties", {})
    if not isinstance(extra, dict):
        return []
    bids = []
    for bid, props in extra.items():
        if isinstance(props, dict) and props.get("clickable") is True:
            bids.append(str(bid))
    # stable ordering helps debugging
    bids.sort(key=lambda b: int(b) if b.isdigit() else 10**9)
    return bids[:max_bids]


def build_user_text(goal: str, clickable_bids: List[str]) -> str:
    # You can omit the list if you want the model to infer from vision only,
    # but including the BID list dramatically improves reliability.
    return (
        f"Goal: {goal}\n\n"
        f"Clickable BIDs (valid choices): {', '.join(clickable_bids) if clickable_bids else '(none found)'}\n\n"
        "Pick the correct bid and output exactly: click('<bid>')"
    )


@torch.no_grad()
def vlm_pick_action(
    model,
    processor,
    image: Image.Image,
    goal: str,
    clickable_bids: List[str],
    device: torch.device,
    max_new_tokens: int = 24,
) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": build_user_text(goal, clickable_bids)},
            ],
        },
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    inputs = processor(
        text=[text],
        images=[image],
        return_tensors="pt",
    ).to(device)

    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,  # greedy for pass@1 style eval
    )

    decoded = processor.batch_decode(out, skip_special_tokens=True)[0].strip()

    # Some models echo prompt; extract the first click(...) occurrence.
    m = CLICK_RE.search(decoded)
    if m:
        bid = m.group(1)
        return f"click('{bid}')"

    # fallback: if model returned just a bid number etc.
    # try to pick first integer token and wrap it
    toks = re.findall(r"\b\d+\b", decoded)
    if toks:
        return f"click('{toks[0]}')"

    # last resort: click first candidate if any
    if clickable_bids:
        return f"click('{clickable_bids[0]}')"
    return "click('0')"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env-id", default="browsergym/miniwob.click-button")
    ap.add_argument("--model", default="Qwen/Qwen2-VL-2B-Instruct")  # example
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--max-steps", type=int, default=50)
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = torch.device(args.device)

    print("[load] model:", args.model)
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(
        args.model,
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
        device_map="auto" if device.type == "cuda" else None,
        trust_remote_code=True,
    ).to(device if device.type != "cuda" else model.device)

    env = gym.make(
        args.env_id,
        headless=args.headless,
        disable_env_checker=True,
    )

    rng = np.random.RandomState(args.seed)

    successes = 0
    for ep in range(args.episodes):
        obs, info = env.reset(seed=int(rng.randint(0, 2**31 - 1)))
        goal = obs.get("goal", "")

        for t in range(args.max_steps):
            image = obs_screenshot_to_pil(obs)
            clickable = get_clickable_bids(obs, max_bids=64)

            action = vlm_pick_action(
                model=model,
                processor=processor,
                image=image,
                goal=goal,
                clickable_bids=clickable,
                device=device,
            )

            obs2, r, term, trunc, _ = env.step(action)
            done = bool(term or trunc)

            # trajectory print
            print(
                f"ep={ep:03d} t={t:03d} r={r:+.2f} done={int(done)} "
                f"action={action} goal={json.dumps(goal)} err={json.dumps(obs2.get('last_action_error'))}"
            )

            obs = obs2
            if r > 0:
                successes += 1
            if done:
                break

    env.close()
    print(f"[summary] successes={successes} / episodes={args.episodes}")


if __name__ == "__main__":
    main()
