# run as colab cli python program
import gymnasium as gym
import browsergym.miniwob
import numpy as np
import ollama
from PIL import Image
import io


env = gym.make(
    "browsergym/miniwob.click-button",
    headless=True,
    disable_env_checker=True,
)

obs, info = env.reset(seed=0)


print(f"obs: {type(obs)}")
print(f"obs keys: {obs.keys()}")
print("--------------------------------")
print(f"info: {type(info)}")
print(f"info keys: {info.keys()}")
print("--------------------------------")
goal = obs.get("goal", "")
print("GOAL:", goal)
print("--------------------------------")
axtree = obs.get('axtree_object')
print(f"axtree: {type(axtree)}")
print(f"axtree keys: {axtree.keys()}")
print(f"axtree: {axtree}")
print("--------------------------------")
print(f"screenshot shape: {obs.get('screenshot').shape}")
img = Image.fromarray(obs['screenshot'])
img_byte_arr = io.BytesIO()
img.save(img_byte_arr, format='PNG')


print("goal:", obs.get("goal"))
print("keys:", sorted(obs.keys()))
print("screenshot shape:", None if obs.get("screenshot") is None else obs["screenshot"].shape)

env.close()