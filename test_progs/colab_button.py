import gymnasium as gym
import numpy as np
import torch as th
import os
import shutil
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback, CallbackList
from gymnasium.wrappers import RecordVideo

# 1. NEW CALLBACK: Moves files from local /logs/ to Google Drive
class DriveMoveCallback(BaseCallback):
    def __init__(self, local_path, drive_path, verbose=0):
        super().__init__(verbose)
        self.local_path = local_path
        self.drive_path = drive_path
        # Create drive directory if it doesn't exist
        os.makedirs(self.drive_path, exist_ok=True)

    def _on_step(self) -> bool:
        # Check for new checkpoint files in the local logs folder
        if os.path.exists(self.local_path):
            for file_name in os.listdir(self.local_path):
                if file_name.endswith(".zip"):
                    local_file = os.path.join(self.local_path, file_name)
                    drive_file = os.path.join(self.drive_path, file_name)
                    
                    # Move to Drive (shutil.move handles the renaming/transfer)
                    print(f"Moving {file_name} to Google Drive...")
                    shutil.move(local_file, drive_file)
        return True

# --- Existing Wrappers ---
class MiniWobActionWrapper(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.action_space = gym.spaces.Discrete(20)

    def action(self, action_index):
        return f"click('{action_index}')"

class MiniWobObsWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(40,), dtype=np.float32)

    def observation(self, obs):
        flat_obs = np.zeros(40, dtype=np.float32)
        nodes = obs.get('axtree_object', {}).get('nodes', []) or obs.get('dom_object', {}).get('nodes', [])
        clickable_elements = [n for n in nodes if n.get('bid') is not None]
        
        for i, el in enumerate(clickable_elements[:10]):
            bbox = el.get('bbox', {})
            idx = i * 4
            flat_obs[idx:idx+4] = [
                bbox.get('x', 0) / 500.0, bbox.get('y', 0) / 500.0,
                bbox.get('width', 0) / 500.0, bbox.get('height', 0) / 500.0
            ]
        return flat_obs

class MiniWobRewardWrapper(gym.RewardWrapper):
    def reward(self, reward):
        if reward <= 0:
            last_action = getattr(self.env.unwrapped, 'last_action', None)
            if last_action and "click" in str(last_action).lower():
                return 0.1
        return reward

class BrowserRenderWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.latest_obs = None
    @property
    def render_mode(self): return "rgb_array"
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.latest_obs = obs
        return obs, info
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.latest_obs = obs
        return obs, reward, terminated, truncated, info
    def render(self):
        if self.latest_obs and "screenshot" in self.latest_obs:
            return self.latest_obs["screenshot"]
        return np.zeros((720, 1280, 3), dtype=np.uint8)

# --- Training Function ---
def train():
    import browsergym.miniwob 

    env = gym.make("browsergym/miniwob.click-button", headless=True)
    env = BrowserRenderWrapper(env)
    env = RecordVideo(env, "./videos", episode_trigger=lambda ep: ep % 100 == 0)
    env = MiniWobActionWrapper(env)
    env = MiniWobObsWrapper(env)
    env = MiniWobRewardWrapper(env)

    # 1. Standard Checkpoint Callback (Saves locally every 10k)
    checkpoint_callback = CheckpointCallback(
        save_freq=10000, 
        save_path="./logs/",
        name_prefix="ppo_miniwob_checkpoint"
    )

    # 2. Drive Move Callback (Moves from local to Drive)
    drive_path = "/content/drive/MyDrive/Colab Notebooks/miniwob_checkpoints"
    move_to_drive = DriveMoveCallback(local_path="./logs/", drive_path=drive_path)

    # 3. Combine Callbacks
    callback_list = CallbackList([checkpoint_callback, move_to_drive])

    model = PPO(
        "MlpPolicy",
        env,
        policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256])),
        learning_rate=3e-4,
        n_steps=128,
        verbose=1
    )

    print(f"--- Starting Training. Checkpoints will be moved to: {drive_path} ---")
    model.learn(total_timesteps=150_000, callback=callback_list)
    
    # Save final model directly to Drive
    model.save(os.path.join(drive_path, "ppo_miniwob_final"))

if __name__ == "__main__":
    train()