#!/usr/bin/env python3
"""
iterate_all_miniwob_envs.py

Iterate through ALL BrowserGym MiniWoB environments,
reset each, run random actions, and print trajectory info.

Requirements:
  pip install browsergym-miniwob gymnasium playwright
  python -m playwright install

Make sure MiniWoB server is running:
  export MINIWOB_URL=http://127.0.0.1:8000/miniwob/
  cd miniwob-plusplus/miniwob/html
  python -m http.server 8000
"""

import os
import gymnasium as gym
import browsergym.miniwob
import numpy as np
import ollama
from PIL import Image
import io

# -----------------------------
# Find all MiniWoB env IDs
# -----------------------------
def get_all_miniwob_env_ids():
    ids = []

    for spec in gym.envs.registry.values():
        env_id = spec.id
        if env_id.startswith("browsergym/miniwob."):
            ids.append(env_id)

    ids.sort()
    return ids


# -----------------------------
# Run one environment
# -----------------------------
def run_env(env_id, max_steps=2):
    print("\n==============================")
    print("ENV:", env_id)

    env = gym.make(
        env_id,
        headless=True,
        disable_env_checker=True,
    )

    obs, info = env.reset()
    print(f"goal: {obs.get('goal')}")
    print(f"chat_messages: {obs['chat_messages']}")
    print(f"screenshot shape: {obs['screenshot'].shape}")
    total_reward = 0

    for t in range(max_steps):

        # find clickable bids
        extra = obs.get("extra_element_properties", {})
        clickable = [
            bid for bid, props in extra.items()
            if props.get("clickable", False)
        ]

        if len(clickable) == 0:
            action = "click('0')"
        else:
            bid = np.random.choice(clickable)

            action = f"click('{bid}')"

        obs, reward, terminated, truncated, info = env.step(action)

        total_reward += reward

        print(
            f"t={t:02d} "
            f"reward={reward:.2f} "
            f"done={terminated or truncated} "
            f"action={action}"
        )

        if terminated or truncated:
            break

    print("TOTAL REWARD:", total_reward)

    env.close()

    return total_reward


# -----------------------------
# Iterate all envs
# -----------------------------
def main():

    if not os.environ.get("MINIWOB_URL"):
        raise RuntimeError(
            "Set MINIWOB_URL first:\n"
            "export MINIWOB_URL=http://127.0.0.1:8000/miniwob/"
        )

    env_ids = get_all_miniwob_env_ids()

    print("\nFound", len(env_ids), "MiniWoB envs\n")

    results = {}

    for env_id in env_ids:
        try:
            reward = run_env(env_id)
            results[env_id] = reward

        except Exception as e:
            print("FAILED:", env_id, str(e))
            results[env_id] = None

    print("\n==============================")
    print("SUMMARY")
    print("==============================")

    success = 0
    total = 0

    for env_id, reward in results.items():

        if reward is not None:
            total += 1
            if reward > 0:
                success += 1

        print(f"{env_id:45s} {reward}")

    print("\nPASS@1:", success / total if total else 0)


# -----------------------------
if __name__ == "__main__":
    main()