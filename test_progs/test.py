import gymnasium as gym
import browsergym.miniwob  # registers envs

env = gym.make(
    "browsergym/miniwob.click-button",
    headless=True,              # IMPORTANT
    disable_env_checker=True,
)

obs, info = env.reset(seed=0)
print("goal:", obs["goal"])
print("screenshot:", obs["screenshot"].shape)
print("keys:", sorted(obs.keys()))
env.close()
