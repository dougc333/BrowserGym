import gymnasium as gym
import browsergym.miniwob

env = gym.make("browsergym/miniwob.click-button")
for i in range(10):
    obs, info = env.reset()
    ax = obs["axtree_object"]
    bids = []
    for n in ax.get("nodes", []):
        if n.get("ignored", False): 
            continue
        role = str((n.get("role") or {}).get("value","")).lower()
        bid = n.get("browsergym_id")
        name = str(((n.get("name") or {}).get("value")) or "")
        if bid is not None and role == "button":
            bids.append((str(bid), name))
    print(i, "buttons:", bids)
env.close()
