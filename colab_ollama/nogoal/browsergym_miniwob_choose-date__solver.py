#!/usr/bin/env python3
from __future__ import annotations
import os, sys
import gymnasium as gym
import browsergym.miniwob  # noqa: F401
ENV_ID = 'browsergym/miniwob.choose-date'
KNOWN_GOAL = 'Select 03/17/2016 as the date and hit submit.'
ACTIONS = []
def main() -> int:
    env_id = sys.argv[1] if len(sys.argv) > 1 else ENV_ID
    if not os.environ.get("MINIWOB_URL"):
        print("MINIWOB_URL is not set", file=sys.stderr); return 2
    env = gym.make(env_id)
    try:
        obs, info = env.reset()
        print(f"Goal: {(obs.get('goal') if isinstance(obs, dict) else KNOWN_GOAL)}")
        reward = None
        for a in ACTIONS:
            print(f"Action: {a}")
            obs, reward, term, trunc, info = env.step(a)
            print(f"After action: reward={reward} terminated={term} truncated={trunc}")
            if isinstance(obs, dict) and obs.get('last_action_error'):
                print(f"last_action_error: {obs['last_action_error']}")
            if term or trunc:
                break
        return 0 if (reward is not None and reward > 0) else 1
    finally:
        env.close()
if __name__ == '__main__':
    raise SystemExit(main())
