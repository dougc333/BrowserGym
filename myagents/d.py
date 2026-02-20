import os, json
import gymnasium as gym
import browsergym.miniwob

assert os.environ.get("MINIWOB_URL")

env = gym.make("browsergym/miniwob.click-button", headless=True, disable_env_checker=True)
obs, info = env.reset()

extra = obs["extra_element_properties"]
clickable = [str(b) for b,p in extra.items() if isinstance(p,dict) and p.get("clickable") is True]
print("GOAL:", obs["goal"])
print("CLICKABLE:", clickable)

for bid in clickable:
    action = {"action": "click", "bid": bid}          # ✅ correct
    # action = json.dumps(action)                     # also works in many setups
    obs, r, term, trunc, info = env.step(action)
    print("sent:", action, "reward:", r, "term:", term, "trunc:", trunc)
    print("env saw last_action:", obs.get("last_action"))
    print("env saw last_action_error:", obs.get("last_action_error"))
    if r > 0:
        print("✅ SUCCESS")
        break

env.close()