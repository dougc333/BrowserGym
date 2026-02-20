import os
import gymnasium as gym
import browsergym.miniwob  # registers env ids

# 1) Make sure MiniWoB server URL is set
# Example:
# export MINIWOB_URL=http://127.0.0.1:8000/miniwob/
print("MINIWOB_URL =", os.environ.get("MINIWOB_URL"))

# 2) Create env (debug-friendly settings)
env = gym.make(
    "browsergym/miniwob.click-button",
    headless=True,              # set False if you have an X server / local desktop
    disable_env_checker=True,   # avoids Gym warning spam while debugging
)
obs, info = env.reset()
# 3) Inspect spaces (super useful to know action dict schema)
print("OBS space:", env.observation_space)
#print("ACT space:", env.action_space)
print("action_space:", env.action_space)

# pick an input and submit bid from your debug
input_bid = "13"
submit_bid = "15"

obs, r, term, trunc, info = env.step({"action": "click", "bid": input_bid})
print("click input", r, term, trunc)

obs, r, term, trunc, info = env.step({"action": "type", "text": "hello"})
print("type", r, term, trunc)

obs, r, term, trunc, info = env.step({"action": "click", "bid": submit_bid})
print("click submit", r, term, trunc)


# 4) Reset and inspect observation
obs, info = env.reset()
print("\n--- RESET ---")
print("url :", obs.get("url"))
print("goal:", obs.get("goal"))
print("focused_element_bid:", obs.get("focused_element_bid"))
print("keys:", list(obs.keys()))

# 5) Print clickable bids
extra = obs.get("extra_element_properties", {})
clickable = []
if isinstance(extra, dict):
    for bid, props in extra.items():
        if isinstance(props, dict) and props.get("clickable") is True:
            clickable.append(str(bid))
print("num_clickable:", len(clickable))
print("sample_clickable:", clickable[:20])

# 6) Try one manual click (pick a bid from sample_clickable)
# Example:
# action = {"action": "click", "bid": clickable[0]}
# obs, r, term, trunc, info = env.step(action)
# print("step reward:", r, "term:", term, "trunc:", trunc)

# Keep env open for interactive debugging
print("\nEnv ready. You can now call env.step({...}) interactively.")