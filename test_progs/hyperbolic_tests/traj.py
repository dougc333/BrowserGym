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
DEFAULT_MODEL = "openai/gpt-oss-120b"  # you confirmed this one works


# ------------------------------------------------------------
# API KEY HANDLING
# ------------------------------------------------------------
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


# ------------------------------------------------------------
# HYPERBOLIC CALL (robust)
# ------------------------------------------------------------
def hyperbolic_chat(
    api_key: str,
    messages: List[Dict[str, str]],
    model: str = DEFAULT_MODEL,
    max_tokens: int = 256,
    temperature: float = 0.2,
    top_p: float = 0.9,
    timeout_s: int = 120,
    retries: int = 5,
) -> str:
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    payload = {
        "messages": messages,
        "model": model,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
    }

    backoff = 1.0
    last_err: Optional[str] = None

    for attempt in range(retries):
        try:
            r = requests.post(HYPERBOLIC_URL, headers=headers, json=payload, timeout=timeout_s)

            # Try parse JSON even on non-200
            try:
                j = r.json()
            except Exception:
                j = {"raw_text": r.text[:2000]}

            if r.status_code != 200:
                # transient? retry on 429 / 5xx
                last_err = f"HTTP {r.status_code}: {j}"
                if r.status_code == 429 or (500 <= r.status_code < 600):
                    time.sleep(backoff)
                    backoff = min(backoff * 2, 16)
                    continue
                raise RuntimeError(last_err)

            if "choices" not in j:
                # This is the exact thing you hit (50001 error object or unexpected payload)
                raise RuntimeError(f"Missing choices in response: {j}")

            return j["choices"][0]["message"]["content"]

        except Exception as e:
            last_err = str(e)
            # Retry for “internal error” style issues too
            time.sleep(backoff)
            backoff = min(backoff * 2, 16)

    raise RuntimeError(f"Hyperbolic call failed after {retries} tries. Last error: {last_err}")


# ------------------------------------------------------------
# MINI OBS UTILS
# ------------------------------------------------------------
def extract_goal(obs: Dict[str, Any]) -> str:
    g = obs.get("goal")
    return g.strip() if isinstance(g, str) else ""


def obs_brief(obs: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "url": obs.get("url"),
        "goal": extract_goal(obs),
        "focused_element_bid": obs.get("focused_element_bid"),
    }


# ------------------------------------------------------------
# JSON EXTRACTION (LLMs often add text)
# ------------------------------------------------------------
def extract_first_json_obj(text: str) -> Optional[Dict[str, Any]]:
    # Fast heuristic: find first {...} block and try parsing progressively.
    start = text.find("{")
    if start == -1:
        return None
    for end in range(len(text) - 1, start, -1):
        if text[end] == "}":
            snippet = text[start : end + 1]
            try:
                obj = json.loads(snippet)
                if isinstance(obj, dict):
                    return obj
            except Exception:
                continue
    return None


# ------------------------------------------------------------
# ACTION SELECTION
# ------------------------------------------------------------
def choose_action_with_llm(
    api_key: str,
    model: str,
    obs: Dict[str, Any],
    prev_steps: List[Dict[str, Any]],
) -> Dict[str, Any]:
    goal = extract_goal(obs)
    url = obs.get("url")

    clickable_bids: List[str] = []
    extra = obs.get("extra_element_properties")
    if isinstance(extra, dict):
        for bid, props in extra.items():
            if isinstance(props, dict) and props.get("clickable") is True:
                clickable_bids.append(str(bid))
    clickable_bids = clickable_bids[:40]

    # (Optional) provide tiny history
    last_actions = [{"action": s["action"], "reward": s["reward"]} for s in prev_steps[-3:]]

    system = (
        "You control a browser UI environment.\n"
        "Return ONLY a JSON object (no markdown, no extra text).\n"
        "Valid actions:\n"
        '  {"action":"click","bid":"<BID>"}\n'
        '  {"action":"stop"}\n'
        "Pick a bid from CLICKABLE_BIDS.\n"
    )

    user = (
        f"URL: {url}\n"
        f"GOAL: {goal}\n"
        f"CLICKABLE_BIDS: {clickable_bids}\n"
        f"RECENT: {json.dumps(last_actions)}\n"
        "Return JSON now."
    )

    try:
        text = hyperbolic_chat(
            api_key=api_key,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            model=model,
            max_tokens=120,
            temperature=0.2,
            top_p=0.9,
        )
    except Exception as e:
        # If provider is erroring, stop gracefully
        print(f"[warn] LLM call failed: {e}")
        return {"action": "stop"}

    obj = extract_first_json_obj(text)
    if not obj:
        return {"action": "stop"}

    # Basic validation
    if obj.get("action") == "click":
        bid = str(obj.get("bid", ""))
        if bid in clickable_bids:
            return {"action": "click", "bid": bid}
        return {"action": "stop"}

    if obj.get("action") == "stop":
        return {"action": "stop"}

    return {"action": "stop"}


# ------------------------------------------------------------
# EPISODE LOOP
# ------------------------------------------------------------
def run_episode(env, api_key: str, model: str, max_steps: int) -> Dict[str, Any]:
    obs, info = env.reset()

    traj: Dict[str, Any] = {
        "env_id": env.spec.id if env.spec else None,
        "initial": obs_brief(obs),
        "steps": [],
        "final": {},
    }

    prev_steps: List[Dict[str, Any]] = []

    for t in range(max_steps):
        action = choose_action_with_llm(api_key, model, obs, prev_steps)

        if action.get("action") == "stop":
            traj["final"] = {"reason": "stop", "t": t}
            break

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

        if terminated or truncated:
            traj["final"] = {"reason": "done", "t": t}
            break

    return traj


# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-id", default="browsergym/miniwob.click-button")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--out", default="trajs.jsonl")
    parser.add_argument("--headless", action="store_true", default=True)
    args = parser.parse_args()

    api_key = resolve_api_key(args.api_key)

    miniwob_url = os.environ.get("MINIWOB_URL")
    if not miniwob_url:
        raise RuntimeError(
            "Missing MINIWOB_URL.\n"
            "Example:\n"
            "export MINIWOB_URL=http://127.0.0.1:8000/miniwob/\n"
            "(and run the MiniWoB http.server from the miniwob-plusplus/miniwob/html directory)\n"
        )

    env = gym.make(args.env_id, headless=args.headless)

    with open(args.out, "w") as f:
        for ep in range(args.episodes):
            traj = run_episode(env, api_key, args.model, args.max_steps)
            f.write(json.dumps(traj) + "\n")
            print(f"episode {ep+1}/{args.episodes} done: final={traj.get('final')}")

    env.close()
    print("Finished.")


if __name__ == "__main__":
    main()