
import ollama
# run as colab cli python program
import gymnasium as gym
import browsergym.miniwob
import numpy as np
import ollama
from PIL import Image
import io

# Use the "Colab Server is LIVE at" URL from your Colab output
COLAB_NGROK_URL = 'https://labrador-fair-trivially.ngrok-free.app'


class TestColabQwenAxtree:
    def __init__(self):
      self.client = ollama.Client(
        host=COLAB_NGROK_URL,
        headers={'ngrok-skip-browser-warning': 'true'}
      )
 

    def test_colabqwen_axtree(self):
        env = gym.make(
            "browsergym/miniwob.click-button",
            headless=True,
            disable_env_checker=True,
        )
        obs, info = env.reset(seed=0)
        axtree = obs.get('axtree_object')

        self.messages.append({
            'role': 'user',
            'content': axtree
        })

        response = self.client.chat(
            model='qwen3-vl',
            messages=self.messages
        )

        print("Response from Colab test_colabqwen_axtree:")
        print("response keys:", response.keys())
        print("response['message'] keys:", response['message'].keys())
        print(response['message']['content'])

    def test_image(self):
        response = self.client.chat(
        model='qwen3-vl',
        messages=[{
            'role': 'user',
            'content': 'What is written in this image?',
            'images': ['buttonclick.png'] # Use the local path on your Mac
          }]
        )
        print("Response from Colab:")
        print(response['message']['content'])
    

t =  TestColabQwenAxtree()
t.test_colabqwen_axtree()
