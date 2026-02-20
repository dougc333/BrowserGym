import gymnasium as gym
import numpy as np
import torch as th
import torch.nn as nn
import re, os, shutil
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback, CallbackList
import time

# --- 1. ACTION WRAPPER (Fixes the Unicode Error) ---
class MiniWobActionWrapper(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)
        # PPO now sees a numeric space. 50 is usually enough for all calendar buttons.
        self.action_space = gym.spaces.Discrete(50) 

    def action(self, action_index):
        # Maps the number PPO chooses back to a BrowserGym text command
        return f"click('{action_index}')"

# --- 2. OBSERVATION WRAPPER (Extracts Goal + Elements) ---
class DateObsWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gym.spaces.Dict({
            "goal_id": gym.spaces.Discrete(32), # 0-31 for the day
            "element_ids": gym.spaces.Box(low=0, high=100, shape=(10,), dtype=np.int32),
            "coords": gym.spaces.Box(low=0, high=1, shape=(40,), dtype=np.float32)
        })

    def observation(self, obs):
        # Parse the day from the goal string (e.g., "Select 05/14/2023" -> 14)
        goal_text = obs.get('goal', "")
        # Look for a 1-2 digit number that represents the day
        day_match = re.findall(r'\b([1-9]|[12][0-9]|3[01])\b', goal_text)
        goal_day = int(day_match[0]) if day_match else 0
        
        element_ids = np.zeros(10, dtype=np.int32)
        coords = np.zeros(40, dtype=np.float32)
        
        nodes = obs.get('axtree_object', {}).get('nodes', [])
        clickable = [n for n in nodes if n.get('bid')]
        
        for i, node in enumerate(clickable[:10]):
            element_ids[i] = int(node.get('bid', 0)) % 100
            bbox = node.get('bbox', {})
            idx = i * 4
            coords[idx:idx+4] = [
                bbox.get('x', 0)/500, bbox.get('y', 0)/500,
                bbox.get('width', 0)/500, bbox.get('height', 0)/500
            ]
        return {"goal_id": goal_day, "element_ids": element_ids, "coords": coords}

# --- 3. REWARD WRAPPER ---
class MiniWobRewardWrapper(gym.RewardWrapper):
    def reward(self, reward):
        # Give a small 0.01 bonus for any click to encourage exploration
        if reward <= 0:
            return 0.01
        return reward

# --- 4. FEATURE EXTRACTOR (Using nn.Embedding) ---
class DateExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict):
        super().__init__(observation_space, features_dim=512)
        
        self.goal_embedding = nn.Embedding(32, 16)   # Output: (batch, 32, 16)
        self.id_embedding = nn.Embedding(101, 32)    # Output: (batch, 10, 32)
        
        self.spatial_net = nn.Sequential(
            nn.Linear(40, 128), 
            nn.ReLU()
        )
        
        # FIX: The input dimension here MUST be 960 (512 + 320 + 128)
        self.combined_net = nn.Sequential(
            nn.Linear(960, 512), 
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU()
        )

    def forward(self, observations):
        # 1. Goal: [batch, 32, 16] -> flatten to [batch, 512]
        g_emb = self.goal_embedding(observations["goal_id"].long())
        g_emb_flat = g_emb.flatten(1)
        
        # 2. IDs: [batch, 10, 32] -> flatten to [batch, 320]
        ids = observations["element_ids"].long()
        id_embs = self.id_embedding(ids)
        id_embs_flat = id_embs.flatten(1)
        
        # 3. Spatial: [batch, 128]
        spatial = self.spatial_net(observations["coords"])
        
        # 4. Combined: [batch, 960]
        combined = th.cat((g_emb_flat, id_embs_flat, spatial), dim=1)
        
        return self.combined_net(combined)
  
# --- 5. DRIVE SYNC CALLBACK ---
class DriveMoveCallback(BaseCallback):
    def __init__(self, local_path, drive_path):
        super().__init__()
        self.local_path = local_path
        self.drive_path = drive_path
        if not os.path.exists(self.drive_path): os.makedirs(self.drive_path)

    def _on_step(self) -> bool:
        if os.path.exists(self.local_path):
            for f in os.listdir(self.local_path):
                if f.endswith(".zip"):
                    print(f"Moving {f} to persistent storage...")
                    shutil.move(os.path.join(self.local_path, f), os.path.join(self.drive_path, f))
        return True

def train_date_picker():
    import browsergym.miniwob
    save_path = "./miniwob_date_checkpoints" 
    
    # Clean up local logs before starting
    if os.path.exists("./logs/"):
        shutil.rmtree("./logs/")
    os.makedirs("./logs/")

    # 1. Create Environment with standard kwargs
    def make_env():
        # Removed 'wait_for_halt' as it's not a valid kwarg here
        env = gym.make(
            "browsergym/miniwob.choose-date", 
            headless=True
        )
        # dont need sleep for macos
        env = MiniWobActionWrapper(env)
        env = DateObsWrapper(env)
        env = MiniWobRewardWrapper(env)
        return env

    # Initialize environment
    try:
        env = make_env()
    except Exception as e:
        print(f"Initial env creation failed: {e}. Retrying...")
        time.sleep(2)
        env = make_env()

    # 2. Setup Model
    model = PPO(
        "MultiInputPolicy",
        env,
        policy_kwargs=dict(features_extractor_class=DateExtractor),
        n_steps=512,
        verbose=1
    )

    # 3. Setup Callbacks
    callbacks = CallbackList([
        CheckpointCallback(save_freq=10000, save_path="./logs/", name_prefix="date_task"),
        DriveMoveCallback("./logs/", save_path)
    ])

    print("--- Training Started (Date Task) ---")
    
    # 4. Training Loop with Retry for the internal MiniWoB 'null' error
    attempts = 0
    while attempts < 5:
        try:
            model.learn(total_timesteps=300000, callback=callbacks)
            break # Success!
        except Exception as e:
            if "setAttribute" in str(e) or "null" in str(e):
                attempts += 1
                print(f"MiniWoB JS Core failed (Attempt {attempts}/5). Resetting env...")
                time.sleep(1)
                model.set_env(make_env()) # Refresh the browser instance
            else:
                raise e # Real error, don't ignore

if __name__ == "__main__":
    train_date_picker()