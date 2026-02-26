#!/usr/bin/env python3
from __future__ import annotations

import os
import sys

import gymnasium as gym
import browsergym.miniwob  # noqa: F401

ENV_ID = 'browsergym/miniwob.click-checkboxes-soft'
KNOWN_GOAL = 'Select words similar to joyful, mad, rabbit, courageous and click Submit.'
ACTIONS = [
        "click('30')"
] if True else []

def main() -> int:
    env_id = sys.argv[1] if len(sys.argv) > 1 else ENV_ID
    if not os.environ.get("MINIWOB_URL"):
        print('MINIWOB_URL is not set. Example: export MINIWOB_URL="http://127.0.0.1:8000/miniwob/"', file=sys.stderr)
        return 2
    env = gym.make(env_id)
    try:
        obs, info = env.reset()
        if isinstance(obs, dict) and obs.get("goal"):
            print(f"Goal: {obs['goal']}")
        elif KNOWN_GOAL:
            print(f"Goal (cached): {KNOWN_GOAL}")

        if not ACTIONS:
            print("No cached successful actions for this environment.")
            return 1

        reward = None
        terminated = False
        truncated = False
        for action in ACTIONS:
            print(f"Action: {action}")
            obs, reward, terminated, truncated, info = env.step(action)
            if isinstance(obs, dict) and obs.get("last_action_error"):
                print(f"last_action_error: {obs['last_action_error']}")
            print(f"After action: reward={reward} terminated={terminated} truncated={truncated}")
            if terminated or truncated:
                break

        return 0 if (reward is not None and reward > 0) else 1
    finally:
        env.close()

if __name__ == "__main__":
    raise SystemExit(main())
