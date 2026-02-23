import gymnasium as gym
import browsergym.miniwob
import numpy as np
from PIL import Image
import os

env = gym.make(
    "browsergym/miniwob.click-button",
    headless=True,
    disable_env_checker=True,
)

obs, info = env.reset(seed=0)

print("goal:", obs.get("goal"))
print("keys:", sorted(obs.keys()))

screenshot = obs.get("screenshot")

if screenshot is not None:
    print("screenshot shape:", screenshot.shape)

    # ensure uint8
    img = np.asarray(screenshot).astype(np.uint8)

    # create output dir
    os.makedirs("screenshots", exist_ok=True)

    path = "screenshots/miniwob_click_button_seed0.png"

    # save
    Image.fromarray(img).save(path)

    print("saved screenshot to:", path)


env.close()