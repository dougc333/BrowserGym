import gymnasium as gym
import numpy as np
import torch as th
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from gymnasium.wrappers import RecordVideo


checkpoint_callback = CheckpointCallback(
        save_freq=10000, 
        save_path="./logs/",
        name_prefix="ppo_miniwob_checkpoint"
    )

# [Paste your MiniWobActionWrapper, MiniWobObsWrapper, and MiniWobRewardWrapper here]
class MiniWobActionWrapper(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.action_space = gym.spaces.Discrete(20)

    def action(self, action_index):
        # We assume the first 20 BIDs are the clickable buttons
        return f"click('{action_index}')"

# 2. OBSERVATION WRAPPER: Extract AXTree nodes into a flat (40,) vector
class MiniWobObsWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(
            low=0, high=1, shape=(40,), dtype=np.float32
        )

    def observation(self, obs):
        if not hasattr(self, 'count'):
            self.count = 0
            
        if self.count % 100 == 0:
            print(f"Step {self.count}: Available keys in obs: {obs.keys()}")
            self.count += 1
        
        flat_obs = np.zeros(40, dtype=np.float32)
        
        # Try to get nodes from axtree_object first (cleaner than dom_object)
        axtree = obs.get('axtree_object', {})
        nodes = axtree.get('nodes', [])
        
        # If axtree is empty, fallback to dom_object
        if not nodes:
            nodes = obs.get('dom_object', {}).get('nodes', [])
        
        # Filter for elements that have a Browser ID (BID)
        clickable_elements = [n for n in nodes if n.get('bid') is not None]
        
        for i, el in enumerate(clickable_elements[:10]):
            bbox = el.get('bbox', {})
            # Normalize coordinates (MiniWoB is small, 500 is a safe scale)
            x = bbox.get('x', 0) / 500.0 
            y = bbox.get('y', 0) / 500.0
            w = bbox.get('width', 0) / 500.0
            h = bbox.get('height', 0) / 500.0
            
            idx = i * 4
            flat_obs[idx:idx+4] = [x, y, w, h]
            
        return flat_obs

# 3. REWARD WRAPPER: Partial credit for clicking (The HER alternative)
class MiniWobRewardWrapper(gym.RewardWrapper):
    def reward(self, reward):
        if reward <= 0:
            # Check the underlying env for the last action string
            last_action = getattr(self.env.unwrapped, 'last_action', None)
            if last_action and "click" in str(last_action).lower():
                return 0.1  # Bonus for attempting to interact
        return reward

class BrowserRenderWrapper(gym.Wrapper):
    def __init__(self, env):
        # We don't pass render_mode to super() because BrowserEnv doesn't support it
        super().__init__(env)
        self.latest_obs = None

    # We define render_mode as a property to satisfy RecordVideo
    @property
    def render_mode(self):
        return "rgb_array"

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

def train():
    import browsergym.miniwob 

    # Colab MUST be headless=True
    # We also specify the window size to ensure coordinates are consistent
    env = gym.make(
        "browsergym/miniwob.click-button", 
        headless=True,
    )
    env = BrowserRenderWrapper(env)
    env = RecordVideo(
        env, 
        video_folder="./videos", 
        episode_trigger=lambda episode_id: episode_id % 100 == 0,
        name_prefix="miniwob_training"
    )
    # Apply Wrapper Stack
    env = MiniWobActionWrapper(env)
    env = MiniWobObsWrapper(env)
    env = MiniWobRewardWrapper(env)

    model = PPO(
        "MlpPolicy",
        env,
        policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256])),
        learning_rate=3e-4,
        n_steps=128,  # Optimization for Colab's virtual CPU
        batch_size=64,
        verbose=1
    )

    print("--- Starting Training on Colab ---")
    model.learn(total_timesteps=150_000, callback=checkpoint_callback)
    model.save("ppo_miniwob_colab")

if __name__ == "__main__":
    train()