import os
import gymnasium as gym
import browsergym.miniwob

assert os.environ.get("MINIWOB_URL"), "export MINIWOB_URL=http://127.0.0.1:8000/miniwob/"

env = gym.make(
    "browsergym/miniwob.click-button",
    headless=True,
    disable_env_checker=True,
)

obs, info = env.reset()
print("GOAL:", obs["goal"])
extra = obs["extra_element_properties"]

clickable = []
for bid, props in extra.items():
    if isinstance(props, dict) and props.get("clickable") is True:
        clickable.append(str(bid))

print("CLICKABLE:", clickable)

# Try each clickable bid once
for bid in clickable:
    act = f"click {bid}"
    obs, r, term, trunc, info = env.step(act)
    print("action:", act, "reward:", r, "term:", term, "trunc:", trunc, "last_action_error:", obs.get("last_action_error"))
    if r > 0:
        print("✅ SUCCESS with", act)
        break

env.close()