#!/usr/bin/env python3
"""
miniwob_hyperbolic_agent.py

One-episode MiniWoB (BrowserGym) runner that:
  - extracts grounded page state (flattened DOM with bids + center coords + bbox)
  - sends ONE request per step to Hyperbolic chat/completions
  - prints: when sending, the exact JSON payload, and the raw response body
  - robustly extracts a JSON action even if the model returns extra text
  - validates bid; if invalid, falls back to a DOM-based heuristic (e.g. find "OK")

Prereqs (in your venv):
  pip install gymnasium browsergym-miniwob playwright requests

Playwright browser install (once):
  python -m playwright install chromium

MiniWoB server (separate terminal):
  cd /path/to/miniwob-plusplus/miniwob/html
  python -m http.server 8000

And:
  export MINIWOB_URL=http://127.0.0.1:8000/miniwob/
  export HYPERBOLIC_API_KEY="..."

Run:
  python miniwob_hyperbolic_agent.py --env-id browsergym/miniwob.click-button --model openai/gpt-oss-120b
"""

import argparse
import json
import os
import re
import time
from typing import Any, Dict, List, Optional, Tuple

import requests
import gymnasium as gym
import browsergym.miniwob  # registers envs
from browsergym.utils.obs import flatten_dom_to_str


HYPERBOLIC_URL = "https://api.hyperbolic.xyz/v1/chat/completions"


# ----------------------------
# Keys / env
# ----------------------------
def require_env(name: str) -> str:
    v = os.environ.get(name)
    if not v:
        raise RuntimeError(f"Missing {name}. Example:\nexport {name}=...")
    return v


def normalize_miniwob_url(url: str) -> str:
    # BrowserGym expects base like http://127.0.0.1:8000/miniwob/
    url = url.strip()
    if not url.endswith("/"):
        url += "/"
    return url


# ----------------------------
# Hyperbolic request + debug
# ----------------------------
def hyperbolic_chat_raw(
    api_key: str,
    model: str,
    messages: List[Dict[str, str]],
    max_tokens: int = 128,
    temperature: float = 0.2,
    top_p: float = 0.9,
    response_format_json: bool = True,
    timeout_s: int = 120,
) -> Tuple[int, str, Dict[str, Any]]:
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    payload: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
    }

    # Some providers honor OpenAI-style response_format; if ignored, we still parse robustly.
    if response_format_json:
        payload["response_format"] = {"type": "json_object"}

    print("\n========== LLM REQUEST ==========")
    print(f"[send] POST {HYPERBOLIC_URL}")
    print("[send] payload JSON:")
    print(json.dumps(payload, indent=2, ensure_ascii=False))

    r = requests.post(HYPERBOLIC_URL, headers=headers, json=payload, timeout=timeout_s)
    status = r.status_code
    body = r.text

    print("\n========== LLM RESPONSE ==========")
    print(f"[recv] status: {status}")
    # print full body (can be large). truncate only if truly huge.
    if len(body) > 12000:
        print("[recv] raw body (truncated):")
        print(body[:12000] + "\n...<truncated>...")
    else:
        print("[recv] raw body:")
        print(body)

    try:
        j = r.json()
    except Exception:
        j = {}

    return status, body, j


# ----------------------------
# Robust JSON extraction
# ----------------------------
def extract_first_json_object(text: str) -> Optional[Dict[str, Any]]:
    """
    Extract the first {...} JSON object from arbitrary text.
    Handles cases where the model wraps content in extra tokens or prose.

    Returns dict or None.
    """
    if not isinstance(text, str) or "{" not in text:
        return None

    # Fast path: exact JSON
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    # Balanced-brace scan (safer than naive regex)
    start = text.find("{")
    while start != -1:
        depth = 0
        for i in range(start, len(text)):
            c = text[i]
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    chunk = text[start : i + 1]
                    try:
                        obj = json.loads(chunk)
                        if isinstance(obj, dict):
                            return obj
                    except Exception:
                        break  # try next '{'
        start = text.find("{", start + 1)

    return None


def get_message_content(resp_json: Dict[str, Any]) -> str:
    try:
        return resp_json["choices"][0]["message"]["content"]
    except Exception:
        return ""


# ----------------------------
# Grounded page state
# ----------------------------
def obs_brief(obs: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "url": obs.get("url"),
        "goal": obs.get("goal"),
        "focused_element_bid": obs.get("focused_element_bid"),
    }


def get_clickable_bids(obs: Dict[str, Any], limit: int = 80) -> List[str]:
    bids: List[str] = []
    extra = obs.get("extra_element_properties")
    if isinstance(extra, dict):
        for bid, props in extra.items():
            if isinstance(props, dict) and props.get("clickable") is True:
                bids.append(str(bid))
    return bids[:limit]


def flatten_page(obs: Dict[str, Any], max_chars: int = 20000) -> str:
    """
    Flatten DOM with center coords (BrowserGym already uses bbox to compute center).
    This string includes bid="..." attributes, tags, and often text content.
    """
    dom = obs.get("dom_object")
    extra = obs.get("extra_element_properties")
    if not isinstance(dom, dict) or not isinstance(extra, dict):
        return ""

    s = flatten_dom_to_str(dom, extra, with_center_coords=True)
    if len(s) > max_chars:
        return s[:max_chars] + "\n...<truncated dom>..."
    return s


# ----------------------------
# Heuristic fallback: find likely bid from DOM text
# ----------------------------
def find_bid_by_text(dom_flat: str, wanted: str) -> Optional[str]:
    """
    Search flattened DOM for an element containing wanted text and extract bid="...".
    Very simple heuristic; good for click-button / ok / submit tasks.
    """
    if not dom_flat or not wanted:
        return None

    wanted_norm = wanted.strip().lower()

    # Find lines containing the text; extract bid="NN"
    for line in dom_flat.splitlines():
        if wanted_norm in line.lower():
            m = re.search(r'\bbid="([^"]+)"', line)
            if m:
                return m.group(1)

    return None


def infer_target_from_goal(goal: str) -> Optional[str]:
    """
    MiniWoB goals often look like: Click on the "Ok" button.
    We'll extract the quoted text if present.
    """
    if not isinstance(goal, str):
        return None
    m = re.search(r'"([^"]+)"', goal)
    if m:
        return m.group(1)
    return None


# ----------------------------
# Choose action
# ----------------------------
def choose_action(
    api_key: str,
    model: str,
    obs: Dict[str, Any],
    last_action: Optional[Dict[str, Any]],
    last_reward: Optional[float],
    last_done: Optional[bool],
) -> Dict[str, Any]:
    goal = obs.get("goal") or ""
    url = obs.get("url") or ""
    clickable = get_clickable_bids(obs, limit=80)
    dom_flat = flatten_page(obs)

    # Build a *grounded* prompt: DOM includes bid=".." + text + center="(x,y)"
    system = (
        "You control a browser UI.\n"
        "Return ONLY a JSON object with one of these forms:\n"
        '  {"action":"click","bid":"<BID>"}\n'
        '  {"action":"stop"}\n'
        "Rules:\n"
        "- bid must be one of CLICKABLE_BIDS.\n"
        "- No extra keys.\n"
        "- No extra text, no markdown, no code fences.\n"
    )

    user_parts = [
        f"URL: {url}",
        f"GOAL: {goal}",
    ]
    if last_action is not None:
        user_parts += [
            f"LAST_ACTION: {last_action}",
            f"LAST_REWARD: {last_reward}",
            f"LAST_DONE: {last_done}",
        ]

    user_parts += [
        f"CLICKABLE_BIDS: {clickable}",
        "PAGE_DOM (flattened, includes bid + text + center coords):",
        dom_flat,
        "Pick the best bid to click to satisfy the GOAL.",
    ]
    user = "\n".join(user_parts)

    status, raw_body, resp_json = hyperbolic_chat_raw(
        api_key=api_key,
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        max_tokens=256,
        temperature=0.2,
        top_p=0.9,
        response_format_json=True,
    )

    if status < 200 or status >= 300:
        # Hard stop on provider error (no retries per your request)
        return {"action": "stop"}

    content = get_message_content(resp_json)
    print("\n[parsed] choices[0].message.content:")
    print(content)

    # 1) Try to parse content as JSON (robust extraction)
    action = extract_first_json_object(content)

    # Some providers put action JSON directly in resp_json when response_format works
    if action is None:
        action = extract_first_json_object(raw_body)

    if not isinstance(action, dict):
        print("[parsed] FAILED to extract JSON action; will try heuristic fallback.")
        action = None

    # 2) Validate action
    if isinstance(action, dict):
        if action.get("action") == "stop":
            return {"action": "stop"}
        if action.get("action") == "click":
            bid = str(action.get("bid", ""))
            if bid in clickable:
                return {"action": "click", "bid": bid}
            print(f"[parsed] bid '{bid}' is invalid (not in clickable); will try fallback.")

    # 3) Fallback: infer target text from goal, find matching bid in DOM, ensure clickable
    target = infer_target_from_goal(str(goal)) or ""
    if target:
        bid_guess = find_bid_by_text(dom_flat, target)
        if bid_guess and bid_guess in clickable:
            print(f"[fallback] found bid by text '{target}': {bid_guess}")
            return {"action": "click", "bid": str(bid_guess)}

    # 4) Last resort: click the first clickable bid (sometimes works on simple tasks)
    if clickable:
        print(f"[fallback] clicking first clickable bid: {clickable[0]}")
        return {"action": "click", "bid": clickable[0]}

    return {"action": "stop"}


# ----------------------------
# Main episode runner (1 episode)
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env-id", default="browsergym/miniwob.click-button")
    ap.add_argument("--model", default="openai/gpt-oss-120b")
    ap.add_argument("--max-steps", type=int, default=10)
    ap.add_argument("--headless", action="store_true", default=True)
    ap.add_argument("--out", default="traj.json")
    args = ap.parse_args()

    api_key = require_env("HYPERBOLIC_API_KEY")
    miniwob_url = normalize_miniwob_url(require_env("MINIWOB_URL"))

    # Sanity checks for server routing (helps prevent "core is not defined" issues)
    # We expect:
    #   MINIWOB_URL = http://127.0.0.1:8000/miniwob/
    # so core assets live at:
    #   http://127.0.0.1:8000/core/core.js
    base = miniwob_url[:-len("/miniwob/")] if miniwob_url.endswith("/miniwob/") else miniwob_url.rstrip("/")
    core_js = base + "/core/core.js"
    try:
        r1 = requests.head(core_js, timeout=2)
        print(f"[check] HEAD {core_js} -> {r1.status_code}")
    except Exception as e:
        print(f"[check] WARNING: could not reach {core_js}: {e}")

    env = gym.make(args.env_id, headless=args.headless)

    obs, info = env.reset()
    print("\n==============================")
    print(f"[step 0] obs: {obs_brief(obs)}")
    print("==============================\n")

    traj: Dict[str, Any] = {
        "env_id": args.env_id,
        "model": args.model,
        "miniwob_url": miniwob_url,
        "initial": obs_brief(obs),
        "steps": [],
        "final": {},
    }

    last_action = None
    last_reward = None
    last_done = None

    for t in range(args.max_steps):
        action = choose_action(
            api_key=api_key,
            model=args.model,
            obs=obs,
            last_action=last_action,
            last_reward=last_reward,
            last_done=last_done,
        )

        if action.get("action") == "stop":
            print(f"[step {t}] action=STOP -> ending episode")
            traj["final"] = {"reason": "stop", "t": t}
            break

        print(f"[step {t}] env.step({action})")
        obs2, reward, terminated, truncated, info = env.step(action)
        done = bool(terminated) or bool(truncated)

        step_rec = {
            "t": t,
            "action": action,
            "reward": float(reward) if reward is not None else None,
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "obs": obs_brief(obs2),
        }
        traj["steps"].append(step_rec)

        print(f"[step {t}] reward={reward} done={done} obs={obs_brief(obs2)}")

        last_action = action
        last_reward = reward
        last_done = done
        obs = obs2

        if done:
            traj["final"] = {"reason": "done", "t": t}
            break

        print("\n==============================")
        print(f"[step {t+1}] obs: {obs_brief(obs)}")
        print("==============================\n")

    env.close()

    with open(args.out, "w") as f:
        json.dump(traj, f, indent=2, ensure_ascii=False)
    print(f"\nFinished one episode.\nfinal: {traj['final']}\nwrote: {args.out}")


if __name__ == "__main__":
    main()