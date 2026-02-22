#!/usr/bin/env python3
import os, random
import gymnasium as gym
import browsergym.miniwob  # registers envs

def get_clickable_bids(obs, limit=50):
    extra = obs.get("extra_element_properties", {})
    bids = []
    if isinstance(extra, dict):
        for bid, p in extra.items():
            if isinstance(p, dict) and p.get("clickable") is True:
                bids.append(str(bid))
    return bids[:limit]

def main():
    assert os.environ.get("MINIWOB_URL"), "Set MINIWOB_URL=http://127.0.0.1:8000/miniwob/"

    # Key: disable_env_checker avoids Gym warnings; action mapping uses string actions
    env = gym.make(
        "browsergym/miniwob.click-button",
        headless=True,
        disable_env_checker=True,
    )

    obs, info = env.reset()
    print("goal:", obs.get("goal"))
    bids = get_clickable_bids(obs)
    print("num clickable:", len(bids), "sample:", bids)

    succ = 0
    for t in range(50):
        if not bids:
            print("no clickable bids, reset")
            obs, info = env.reset()
            bids = get_clickable_bids(obs)
            continue

        bid = random.choice(bids)

        # IMPORTANT: step() with an action string
        action = f"click('{bid}')"
        obs, r, term, trunc, info = env.step(action)

        lae = obs.get("last_action_error")
        if lae:
            print("last_action_error:", lae)

        if r > 0:
            succ += 1
            print("SUCCESS at t", t, "action", action, "reward", r)
            break

        if term or trunc:
            obs, info = env.reset()
            bids = get_clickable_bids(obs)

    print("succ:", succ)
    env.close()

if __name__ == "__main__":
    main()