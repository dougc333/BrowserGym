import gymnasium as gym
import numpy as np
import torch as th
from stable_baselines3 import PPO

# 1. ACTION WRAPPER: Map Discrete(20) -> f"click('{id}')"
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

# 4. TRAINING FUNCTION
def train():
    import browsergym.miniwob 

    # headless=False allows you to watch the agent learn!
    try:
        env = gym.make("browsergym/miniwob.click-button", headless=False)
    except:
        env = gym.make("browsergym/miniwob.click-button")
    
    # Apply Wrapper Stack
    env = MiniWobActionWrapper(env)
    env = MiniWobObsWrapper(env)
    env = MiniWobRewardWrapper(env)

    # Policy Configuration
    # We use 256 hidden units to ensure spatial reasoning capacity
    model = PPO(
        "MlpPolicy",
        env,
        policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256])),
        learning_rate=3e-4,
        ent_coef=0.02, # Higher entropy helps find buttons faster
        n_steps=1024,  # Faster updates for monitoring
        verbose=1
    )

    print("--- Starting Training (Targeting >90% Success) ---")
    model.learn(total_timesteps=150_000)
    model.save("ppo_miniwob_click_button")

if __name__ == "__main__":
    train()