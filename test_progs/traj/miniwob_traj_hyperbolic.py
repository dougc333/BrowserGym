#!/usr/bin/env python3
import os
import json
import time
import argparse
import requests
from typing import Any, Dict, List, Optional

import gymnasium as gym
import browsergym.miniwob  # registers MiniWoB env ids


HYPERBOLIC_URL = "https://api.hyperbolic.xyz/v1/chat/completions"
DEFAULT_MODEL = "openai/gpt-oss-20b"


def hyperbolic_chat(
    api_key: str,
    messages: List[Dict[str, str]],
    model: str = DEFAULT_MODEL,
    max_tokens: int = 512,
    temperature: float = 0.2,
    top_p: float = 0.8,
    timeout_s: int = 120,
) -> str:
    """Call Hyperbolic OpenAI-compatible chat completions endpoint and return assistant text."""
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    data = {
        "messages": messages,
        "model": model,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
    }
    r = requests.post(HYPERBOLIC_URL, headers=headers, json=data, timeout=timeout_s)
    r.raise_for_status()
    j = r.json()
    # OpenAI-style: choices[0].message.content
    return j["choices"][0]["message"]["content"]


def extract_goal(obs: Dict[str, Any]) -> str:
    g = obs.get("goal")
    if isinstance(g, str) and g.strip():
        return g.strip()
    return ""


def obs_brief(obs: Dict[str, Any]) -> Dict[str, Any]:
    """Small observation summary to store in the trajectory."""
    url = obs.get("url")
    goal = extract_goal(obs)
    focused = obs.get("focused_element_bid")
    return {
        "url": url,
        "goal": goal,
        "focused_element_bid": focused,
    }


def choose_action_with_llm(
    api_key: str,
    model: str,
    obs: Dict[str, Any],
    info: Dict[str, Any],
    prev_steps: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Produce a BrowserGym/MiniWoB action dict.
    We keep it simple: ask the model for a JSON action:
      {"action":"click","bid":"..."} or {"action":"type","bid":"...","text":"..."} etc.
    """
    goal = extract_goal(obs) or "(no goal provided)"
    url = obs.get("url") or "(no url)"

    # Try to include a small set of “clickable candidates” if available.
    # Some BrowserGym setups include extra_element_properties keyed by bid.
    # We’ll extract a handful of clickable bids to help the model.
    clickable_bids = []
    extra = obs.get("extra_element_properties")
    if isinstance(extra, dict):
        for bid, props in extra.items():
            try:
                if isinstance(props, dict) and props.get("clickable") is True:
                    clickable_bids.append(str(bid))
            except Exception:
                pass
    clickable_bids = clickable_bids[:50]

    history_lines = []
    for s in prev_steps[-6:]:
        a = s.get("action", {})
        r = s.get("reward")
        history_lines.append(f"- action={a} reward={r}")

    system = (
        "You control a web UI environment. "
        "Return ONLY valid JSON for the next action. No markdown. No explanation.\n\n"
        "Valid actions:\n"
        '- Click: {"action":"click","bid":"<bid>"}\n'
        '- Type:  {"action":"type","bid":"<bid>","text":"..."}\n'
        '- Press: {"action":"press","key":"Enter"}  (or other key)\n'
        '- Scroll: {"action":"scroll","dx":0,"dy":400}\n'
        '- Stop if impossible: {"action":"stop"}\n\n'
        "Rules:\n"
        "- Prefer clicking a clickable bid when goal says click.\n"
        "- If goal says click an 'x', choose a likely close button bid (often small, top-right).\n"
        "- Keep actions minimal.\n"
    )

    user = (
        f"URL: {url}\n"
        f"GOAL: {goal}\n"
        f"CLICKABLE_BIDS(sample up to 50): {clickable_bids}\n"
        f"RECENT_STEPS:\n" + ("\n".join(history_lines) if history_lines else "(none)") + "\n\n"
        "Return the next action JSON now."
    )

    text = hyperbolic_chat(
        api_key=api_key,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        model=model,
        max_tokens=200,
        temperature=0.2,
        top_p=0.8,
    ).strip()

    # Parse JSON strictly
    try:
        action = json.loads(text)
        if not isinstance(action, dict) or "action" not in action:
            raise ValueError("not a dict action")
        return action
    except Exception:
        # Fallback: stop
        return {"action": "stop"}


def run_episode(env, api_key: str, model: str, max_steps: int) -> Dict[str, Any]:
    obs, info = env.reset()
    episode = {
        "env_id": env.spec.id if env.spec else None,
        "start_time_unix": time.time(),
        "initial": obs_brief(obs),
        "steps": [],
        "final": {},
    }

    prev_steps: List[Dict[str, Any]] = []

    for t in range(1, max_steps + 1):
        action = choose_action_with_llm(api_key, model, obs, info, prev_steps)

        if action.get("action") == "stop":
            episode["final"] = {
                "reason": "model_stop",
                "t": t,
                "last_obs": obs_brief(obs),
            }
            break

        try:
            obs2, reward, terminated, truncated, info2 = env.step(action)
        except Exception as e:
            episode["steps"].append({
                "t": t,
                "action": action,
                "error": str(e),
                "obs": obs_brief(obs),
            })
            episode["final"] = {
                "reason": "env_step_exception",
                "t": t,
                "error": str(e),
                "last_obs": obs_brief(obs),
            }
            break

        step_rec = {
            "t": t,
            "action": action,
            "reward": float(reward) if isinstance(reward, (int, float)) else reward,
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "obs": obs_brief(obs2),
            "last_action_error": obs2.get("last_action_error", ""),
        }
        episode["steps"].append(step_rec)
        prev_steps.append(step_rec)

        obs, info = obs2, info2

        if terminated or truncated:
            episode["final"] = {
                "reason": "done",
                "t": t,
                "last_obs": obs_brief(obs),
                "reward_sum": sum(s.get("reward", 0.0) or 0.0 for s in episode["steps"]),
            }
            break
    else:
        episode["final"] = {
            "reason": "max_steps",
            "t": max_steps,
            "last_obs": obs_brief(obs),
        }

    episode["end_time_unix"] = time.time()
    return episode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env-id", default="browsergym/miniwob.click-button")
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--max-steps", type=int, default=25)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--out", default="miniwob_trajs.jsonl")
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--slow-mo", type=int, default=0)
    args = ap.parse_args()

    api_key = os.environ.get("HYPERBOLIC_API_KEY")
    if not api_key:
        raise SystemExit("Missing HYPERBOLIC_API_KEY env var")

    # IMPORTANT: MiniWoB base URL must be set
    base_url = os.environ.get("MINIWOB_URL")
    if not base_url:
        raise SystemExit('Missing MINIWOB_URL env var, e.g. export MINIWOB_URL="http://127.0.0.1:8000/miniwob/"')

    # Create env
    env = gym.make(
        args.env_id,
        headless=args.headless,
        slow_mo=args.slow_mo if args.slow_mo > 0 else None,
    )

    print(f"[env] {args.env_id}")
    print(f"MINIWOB_URL: {base_url}")
    print(f"[llm] {args.model}")
    print(f"[out] {args.out}")

    with open(args.out, "w", encoding="utf-8") as f:
        for ep in range(1, args.episodes + 1):
            traj = run_episode(env, api_key=api_key, model=args.model, max_steps=args.max_steps)
            f.write(json.dumps(traj, ensure_ascii=False) + "\n")
            f.flush()
            final = traj.get("final", {})
            print(f"episode {ep}/{args.episodes}: {final.get('reason')} t={final.get('t')}")

    env.close()
    print("OK")


if __name__ == "__main__":
    main()