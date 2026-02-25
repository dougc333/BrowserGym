import gymnasium as gym
import browsergym.miniwob  # registers MiniWoB env IDs

def make_env(env_id: str, **kwargs):
    return gym.make(env_id, **kwargs)