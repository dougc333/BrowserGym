import gymnasium as gym
import browsergym.miniwob  # registers envs

env = gym.make("browsergym/miniwob.click-button")
obs, info = env.reset()
print("goal:", obs["goal"])
print("terminated after reset? no")

# pick any clickable bid and click it once
ax = obs["axtree_object"]
bids = [str(n["browsergym_id"]) for n in ax["nodes"]
        if not n.get("ignored", False)
        and str((n.get("role") or {}).get("value","")).lower() == "button"
        and n.get("browsergym_id") is not None]
print("buttons:", bids)

obs, r, term, trunc, info = env.step(f"click('{bids[0]}')")
print("after one click:", "reward=", r, "terminated=", term, "last_err=", obs.get("last_action_error"))
env.close()
