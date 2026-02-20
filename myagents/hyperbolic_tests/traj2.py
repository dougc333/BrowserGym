#!/usr/bin/env python3
"""
miniwob_hyperbolic_agent.py

A minimal, debuggable MiniWoB agent that:
- runs 1 episode
- extracts clickable elements + their visible text from obs["extra_element_properties"]
- sends that to Hyperbolic chat/completions
- prints the exact JSON payload and raw response for every LLM call
- robustly extracts a {"action":"click","bid":"..."} JSON object even if the model adds extra text
- steps the env until done or max_steps

Prereqs:
  pip install gymnasium browsergym-miniwob playwright requests
  python -m playwright install

You MUST also run a MiniWoB web server separately, e.g.:
  cd /Users/dc/browsergym/miniwob-plusplus/miniwob/html
  python -m http.server 8000

And set:
  export MINIWOB_URL=http://127.0.0.1:8000/miniwob/
  export HYPERBOLIC_API_KEY=...

Run:
  python miniwob_hyperbolic_agent.py --env-id browsergym/miniwob.click-button --model openai/gpt-oss-120b
"""

import argparse
import json
import os
import re
import sys
from typing import Any, Dict, List, Optional, Tuple

import requests

import gymnasium as gym
import browsergym.miniwob  # registers envs


HYPERBOLIC_URL = "https://api.hyperbolic.xyz/v1/chat/completions"
DEFAULT_MODEL = "openai/gpt-oss-120b"


def resolve_api_key(cli_key: Optional[str]) -> str:
    if cli_key:
        return cli_key
    env_key = os.environ.get("HYPERBOLIC_API_KEY")
    if env_key:
        return env_key
    raise RuntimeError(
        "Missing Hyperbolic API key.\n"
        "Set HYPERBOLIC_API_KEY or pass --api-key.\n"
    )


def require_miniwob_url() -> str:
    miniwob_url = os.environ.get("MINIWOB_URL")
    if not miniwob_url:
        raise RuntimeError(
            "Missing MINIWOB_URL.\n"
            "Example:\n"
            "  export MINIWOB_URL=http://127.0.0.1:8000/miniwob/\n"
            "(and run python -m http.server 8000 from miniwob-plusplus/miniwob/html)\n"
        )
    return miniwob_url


def obs_brief(obs: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "url": obs.get("url"),
        "goal": obs.get("goal"),
        "focused_element_bid": obs.get("focused_element_bid"),
    }


def clean_text(x: Any) -> str:
    if not isinstance(x, str):
        return ""
    s = x.strip()
    s = re.sub(r"\s+", " ", s)
    return s


def extract_clickables_with_labels(obs: Dict[str, Any], limit: int = 60) -> List[Dict[str, Any]]:
    """
    Extract clickable elements from obs["extra_element_properties"] and keep
    fields that help identify the right thing (text, aria-label, tag, role, bbox).

    This is the key fix: the LLM needs the *labels*, not just the bid list.
    """
    extra = obs.get("extra_element_properties")
    out: List[Dict[str, Any]] = []

    if not isinstance(extra, dict):
        return out

    for bid, props in extra.items():
        if not isinstance(props, dict):
            continue
        if props.get("clickable") is not True:
            continue

        # Different BrowserGym versions may expose slightly different keys.
        # We'll try a few common ones and keep what exists.
        text = (
            clean_text(props.get("text"))
            or clean_text(props.get("innerText"))
            or clean_text(props.get("value"))
            or clean_text(props.get("name"))
        )
        aria = clean_text(props.get("aria_label")) or clean_text(props.get("aria-label"))
        title = clean_text(props.get("title"))
        tag = clean_text(props.get("tag")) or clean_text(props.get("tagName"))
        role = clean_text(props.get("role"))
        bbox = props.get("bbox")

        item = {
            "bid": str(bid),
            "text": text,
            "aria_label": aria,
            "title": title,
            "tag": tag,
            "role": role,
            "bbox": bbox,
        }

        # Drop empty keys to keep the prompt short
        item = {k: v for k, v in item.items() if v not in ("", None, [])}
        out.append(item)

        if len(out) >= limit:
            break

    return out


def hyperbolic_chat_raw(
    api_key: str,
    payload: Dict[str, Any],
    timeout_s: int = 120,
) -> Tuple[int, str, Dict[str, Any]]:
    """
    Sends request. Returns (status_code, raw_text, json_or_empty).
    """
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    r = requests.post(HYPERBOLIC_URL, headers=headers, json=payload, timeout=timeout_s)
    raw = r.text
    try:
        j = r.json()
    except Exception:
        j = {}
    return r.status_code, raw, j


def extract_first_json_object(s: str) -> Optional[Dict[str, Any]]:
    """
    Robustly extract the first JSON object from a model response, even if the
    model adds extra text (or those <|channel|> tokens).
    """
    if not s:
        return None

    # Quick path: whole string is JSON
    try:
        obj = json.loads(s)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    # Remove common chat template tokens that may appear
    s2 = s.replace("<|channel|>", " ").replace("<|message|>", " ").replace("<|end|>", " ")
    # Find first {...} block (non-greedy) and try parsing progressively
    candidates = re.findall(r"\{.*?\}", s2, flags=re.DOTALL)
    for c in candidates:
        try:
            obj = json.loads(c)
            if isinstance(obj, dict):
                return obj
        except Exception:
            continue
    return None


def validate_action(action: Dict[str, Any], clickable_bids: List[str]) -> Dict[str, Any]:
    """
    Ensure action is one of:
      {"action":"stop"}
      {"action":"click","bid":"<bid>"} where bid is in clickable_bids
    """
    if not isinstance(action, dict):
        return {"action": "stop"}

    a = action.get("action")
    if a == "stop":
        return {"action": "stop"}

    if a == "click":
        bid = str(action.get("bid", ""))
        if bid in clickable_bids:
            return {"action": "click", "bid": bid}
        # If model returns something like "<BID>", reject.
        return {"action": "stop"}

    return {"action": "stop"}


def choose_action_with_llm(
    api_key: str,
    model: str,
    obs: Dict[str, Any],
    last: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    goal = clean_text(obs.get("goal"))
    url = clean_text(obs.get("url"))

    clickables = extract_clickables_with_labels(obs, limit=60)
    clickable_bids = [c["bid"] for c in clickables if "bid" in c]

    system = (
        "You control a browser UI.\n"
        "Return ONLY a JSON object with one of these forms:\n"
        '  {"action":"click","bid":"<BID>"}\n'
        '  {"action":"stop"}\n'
        "Rules:\n"
        "- Choose a BID from the provided elements.\n"
        "- Prefer elements whose text/aria_label/title matches the GOAL.\n"
        "- No extra text, no markdown, no code fences.\n"
    )

    user_lines = [
        f"URL: {url}",
        f"GOAL: {goal}",
    ]
    if last is not None:
        user_lines += [
            f"LAST_ACTION: {last.get('action')}",
            f"LAST_REWARD: {last.get('reward')}",
            f"LAST_DONE: {last.get('done')}",
        ]
    user_lines += [
        "CLICKABLE_ELEMENTS (each has bid + optional labels):",
        json.dumps(clickables, indent=2),
        "Pick the best bid to click to satisfy the goal.",
    ]
    user = "\n".join(user_lines)

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "max_tokens": 128,
        "temperature": 0.2,
        "top_p": 0.9,
        # Ask for JSON object formatting if supported.
        "response_format": {"type": "json_object"},
    }

    print("\n========== LLM REQUEST ==========")
    print(f"[send] POST {HYPERBOLIC_URL}")
    print("[send] payload JSON:")
    print(json.dumps(payload, indent=2))

    status, raw, j = hyperbolic_chat_raw(api_key=api_key, payload=payload, timeout_s=120)

    print("\n========== LLM RESPONSE ==========")
    print(f"[recv] status: {status}")
    # print full raw for debugging (can be large); truncate a bit but keep enough
    if len(raw) > 8000:
        print("[recv] raw body (truncated to 8000 chars):")
        print(raw[:8000])
    else:
        print("[recv] raw body:")
        print(raw)

    if status != 200:
        print("[recv] non-200 -> stop")
        return {"action": "stop"}

    content = ""
    try:
        content = j["choices"][0]["message"]["content"]
    except Exception:
        # Some providers may format differently; fallback to raw parsing
        content = ""

    print("\n[parsed] choices[0].message.content:")
    print(content)

    action_obj = extract_first_json_object(content) or extract_first_json_object(raw)
    if action_obj is None:
        print("\n[parsed] FAILED to extract JSON object -> stop")
        return {"action": "stop"}

    print("\n[parsed] extracted JSON object:")
    print(action_obj)

    action = validate_action(action_obj, clickable_bids)
    if action["action"] == "stop" and action_obj.get("action") == "click":
        print(f"[parsed] invalid bid={action_obj.get('bid')} (not in clickable_bids) -> stop")

    return action


def run_one_episode(env, api_key: str, model: str, max_steps: int) -> Dict[str, Any]:
    obs, info = env.reset()
    traj: Dict[str, Any] = {
        "env_id": env.spec.id if env.spec else None,
        "initial_obs": obs_brief(obs),
        "steps": [],
        "final": {},
    }

    last = None

    for t in range(max_steps):
        print("\n==============================")
        print(f"[step {t}] obs: {obs_brief(obs)}")
        print("==============================")

        action = choose_action_with_llm(api_key=api_key, model=model, obs=obs, last=last)

        if action.get("action") == "stop":
            print(f"[step {t}] action=STOP -> ending episode")
            traj["final"] = {"reason": "stop", "t": t}
            break

        print(f"[step {t}] env.step({action})")
        obs2, reward, terminated, truncated, info = env.step(action)
        done = bool(terminated) or bool(truncated)

        step = {
            "t": t,
            "action": action,
            "reward": float(reward) if isinstance(reward, (int, float)) else reward,
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "done": done,
            "obs": obs_brief(obs2),
        }
        traj["steps"].append(step)

        print(f"[step {t}] reward={reward} done={done} obs={obs_brief(obs2)}")

        last = {"action": action, "reward": reward, "done": done}
        obs = obs2

        if done:
            traj["final"] = {"reason": "done", "t": t, "reward": reward}
            break

    if not traj.get("final"):
        traj["final"] = {"reason": "max_steps", "t": max_steps}

    return traj


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-id", default="browsergym/miniwob.click-button")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--max-steps", type=int, default=10)
    parser.add_argument("--out", default="traj.json")
    parser.add_argument("--headless", action="store_true", help="Run Playwright headless (default true).")
    parser.add_argument("--show", action="store_true", help="Show browser (headless=false).")
    args = parser.parse_args()

    api_key = resolve_api_key(args.api_key)
    require_miniwob_url()

    headless = True
    if args.show:
        headless = False
    elif args.headless:
        headless = True

    env = gym.make(args.env_id, headless=headless)

    traj = run_one_episode(env, api_key=api_key, model=args.model, max_steps=args.max_steps)

    with open(args.out, "w") as f:
        json.dump(traj, f, indent=2)

    env.close()

    print("\nFinished one episode.")
    print("final:", traj["final"])
    print("wrote:", args.out)


if __name__ == "__main__":
    main()