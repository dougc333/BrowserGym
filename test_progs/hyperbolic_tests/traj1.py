#!/usr/bin/env python3
import os
import json
import time
import argparse
import requests
from typing import Any, Dict, List, Optional

import gymnasium as gym
import browsergym.miniwob  # registers envs


HYPERBOLIC_URL = "https://api.hyperbolic.xyz/v1/chat/completions"
DEFAULT_MODEL = "openai/gpt-oss-120b"


# ----------------------------
# API KEY
# ----------------------------
def resolve_api_key(cli_key: Optional[str]) -> str:
    if cli_key and cli_key.strip():
        return cli_key.strip()
    env_key = os.environ.get("HYPERBOLIC_API_KEY")
    if env_key and env_key.strip():
        return env_key.strip()
    raise RuntimeError(
        "Missing Hyperbolic API key.\n"
        "Set:\n  export HYPERBOLIC_API_KEY='...'\n"
        "Or pass:\n  --api-key '...'\n"
    )


# ----------------------------
# OBS HELPERS
# ----------------------------
def extract_goal(obs: Dict[str, Any]) -> str:
    g = obs.get("goal")
    return g.strip() if isinstance(g, str) else ""


def obs_brief(obs: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "url": obs.get("url"),
        "goal": extract_goal(obs),
        "focused_element_bid": obs.get("focused_element_bid"),
    }


def list_clickable_bids(obs: Dict[str, Any], limit: int = 60) -> List[str]:
    extra = obs.get("extra_element_properties")
    bids: List[str] = []
    if isinstance(extra, dict):
        for bid, props in extra.items():
            if isinstance(props, dict) and props.get("clickable") is True:
                bids.append(str(bid))
    return bids[:limit]


# ----------------------------
# JSON EXTRACTION (robust)
# ----------------------------
def _strip_code_fences(s: str) -> str:
    s = s.strip()
    if s.startswith("```"):
        s = s.split("\n", 1)[-1]
        if s.endswith("```"):
            s = s.rsplit("```", 1)[0]
    return s.strip()


def extract_first_json_object(text: str) -> Optional[dict]:
    t = _strip_code_fences(text)

    # whole-string JSON?
    try:
        obj = json.loads(t)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    # find first balanced {...}
    start = t.find("{")
    if start == -1:
        return None

    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(t)):
        ch = t[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        else:
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    blob = t[start : i + 1]
                    try:
                        obj = json.loads(blob)
                        if isinstance(obj, dict):
                            return obj
                    except Exception:
                        return None
    return None


def validate_action(a: Any) -> Dict[str, Any]:
    if not isinstance(a, dict):
        return {"action": "stop"}

    act = a.get("action")
    if act == "stop":
        return {"action": "stop"}

    if act == "click":
        bid = a.get("bid")
        if isinstance(bid, (str, int)) and str(bid).strip():
            return {"action": "click", "bid": str(bid).strip()}

    return {"action": "stop"}


# ----------------------------
# LLM CALL (NO RETRIES, FULL LOGGING)
# ----------------------------
def hyperbolic_chat_once(
    api_key: str,
    messages: List[Dict[str, str]],
    model: str,
    max_tokens: int = 256,
    temperature: float = 0.2,
    top_p: float = 0.9,
    timeout_s: int = 120,
) -> Dict[str, Any]:
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
        # Attempt JSON-mode (harmless if ignored)
        "response_format": {"type": "json_object"},
    }

    print("\n========== LLM REQUEST ==========")
    print("[send] POST", HYPERBOLIC_URL)
    print("[send] payload JSON:")
    print(json.dumps(payload, indent=2)[:8000])

    r = requests.post(HYPERBOLIC_URL, headers=headers, json=payload, timeout=timeout_s)

    print("\n========== LLM RESPONSE ==========")
    print("[recv] status:", r.status_code)
    # Print raw body (truncate to avoid terminal floods)
    body = r.text
    print("[recv] raw body (truncated):")
    print(body[:8000])

    # Try to parse JSON; if it fails, return a synthetic error container
    try:
        return r.json()
    except Exception as e:
        return {"_parse_error": str(e), "_raw_text": body, "_status": r.status_code}


# ----------------------------
# ACTION SELECTION
# ----------------------------
def choose_action_with_llm(
    api_key: str,
    model: str,
    obs: Dict[str, Any],
    prev_steps: List[Dict[str, Any]],
) -> Dict[str, Any]:
    goal = extract_goal(obs)
    url = obs.get("url")
    clickable_bids = list_clickable_bids(obs, limit=60)

    last = prev_steps[-1] if prev_steps else None
    last_summary = ""
    if last:
        last_summary = (
            f"LAST_ACTION: {last['action']}\n"
            f"LAST_REWARD: {last['reward']}\n"
            f"LAST_DONE: {bool(last['terminated'] or last['truncated'])}\n"
        )

    system = (
        "You control a browser UI.\n"
        "Return ONLY a JSON object with one of these forms:\n"
        "  {\"action\":\"click\",\"bid\":\"<BID>\"}\n"
        "  {\"action\":\"stop\"}\n"
        "No extra text, no markdown, no code fences."
    )

    user = (
        f"URL: {url}\n"
        f"GOAL: {goal}\n"
        f"{last_summary}"
        f"CLICKABLE_BIDS: {clickable_bids}\n"
        "Pick the best bid to click to satisfy the goal.\n"
    )

    resp = hyperbolic_chat_once(
        api_key=api_key,
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        max_tokens=128,
        temperature=0.2,
        top_p=0.9,
    )

    # Extract "content" if it looks like an OpenAI-style response
    content = ""
    try:
        content = resp["choices"][0]["message"]["content"]
        print("\n[parsed] choices[0].message.content:")
        print(content[:8000])
    except Exception:
        print("\n[parsed] Could not find resp['choices'][0]['message']['content'] in JSON:")
        print(json.dumps(resp, indent=2)[:8000])
        return {"action": "stop"}

    obj = extract_first_json_object(content)
    if obj is None:
        print("\n[parsed] FAILED to extract JSON object from model content.")
        return {"action": "stop"}

    action = validate_action(obj)
    print("\n[parsed] extracted action JSON:", action)
    return action


# ----------------------------
# EPISODE LOOP (ONE EPISODE)
# ----------------------------
def run_one_episode(env, api_key: str, model: str, max_steps: int) -> Dict[str, Any]:
    obs, info = env.reset()

    traj = {
        "env_id": env.spec.id if env.spec else None,
        "initial": obs_brief(obs),
        "steps": [],
        "final": {},
    }

    prev_steps: List[Dict[str, Any]] = []

    for t in range(max_steps):
        print("\n==============================")
        print(f"[step {t}] obs:", obs_brief(obs))
        print("==============================")

        action = choose_action_with_llm(api_key, model, obs, prev_steps)

        if action.get("action") == "stop":
            traj["final"] = {"reason": "stop", "t": t}
            print(f"[step {t}] action=STOP -> ending episode")
            break

        print(f"[step {t}] env.step({action})")
        obs2, reward, terminated, truncated, info = env.step(action)

        step = {
            "t": t,
            "action": action,
            "reward": float(reward) if isinstance(reward, (int, float)) else reward,
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "obs": obs_brief(obs2),
        }
        traj["steps"].append(step)
        prev_steps.append(step)
        obs = obs2

        done = bool(terminated) or bool(truncated)
        print(f"[step {t}] reward={reward} done={done} obs={obs_brief(obs)}")

        if done:
            traj["final"] = {"reason": "done", "t": t}
            break

    if not traj["final"]:
        traj["final"] = {"reason": "max_steps", "t": max_steps}

    return traj


# ----------------------------
# MAIN
# ----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-id", default="browsergym/miniwob.click-button")
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--out", default="traj.json")

    args = parser.parse_args()

    api_key = resolve_api_key(args.api_key)

    miniwob_url = os.environ.get("MINIWOB_URL")
    if not miniwob_url:
        raise RuntimeError(
            "Missing MINIWOB_URL.\n"
            "Example:\n"
            "  export MINIWOB_URL=http://127.0.0.1:8000/miniwob/\n"
            "(and run the MiniWoB http.server from miniwob-plusplus/miniwob/html)\n"
        )

    env = gym.make(args.env_id, headless=True)

    traj = run_one_episode(env, api_key, args.model, args.max_steps)

    with open(args.out, "w") as f:
        json.dump(traj, f, indent=2)

    env.close()
    print("\nFinished one episode.")
    print("final:", traj["final"])
    print(f"wrote: {args.out}")


if __name__ == "__main__":
    main()