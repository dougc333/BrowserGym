import gymnasium as gym
import browsergym.miniwob

env = gym.make(
    "browsergym/miniwob.click-button",
    headless=True,
    disable_env_checker=True,
)

obs, info = env.reset(seed=0)

print("goal:", obs.get("goal"))
print("keys:", sorted(obs.keys()))
print("screenshot shape:", None if obs.get("screenshot") is None else obs["screenshot"].shape)

env.close()