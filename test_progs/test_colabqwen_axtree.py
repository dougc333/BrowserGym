
import ollama
# run as colab cli python program
import gymnasium as gym
import browsergym.miniwob
import numpy as np
import ollama
from PIL import Image
import io
import json

# Use the "Colab Server is LIVE at" URL from your Colab output
COLAB_NGROK_URL = 'https://labrador-fair-trivially.ngrok-free.app'


class TestColabQwenAxtree:
    def __init__(self):
      self.client = ollama.Client(
        host=COLAB_NGROK_URL,
        headers={'ngrok-skip-browser-warning': 'true'},
        timeout=ollama.Timeout(120.0, connect=60.0) 
      )
      self.messages = []
 
    def test_colabqwen_axtree(self):
        env = gym.make(
            "browsergym/miniwob.click-button",
            headless=True,
            disable_env_checker=True,
        )
        obs, info = env.reset(seed=0)
        
        # 1. Prepare data
        #axtree_str = json.dumps(obs.get('axtree_object'))
        goal = obs.get('goal')
        
        # 2. Convert NumPy screenshot to Bytes
        screenshot_np = obs.get('screenshot')
        img = Image.fromarray(screenshot_np)
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format='PNG')
        img_bytes = img_byte_arr.getvalue()

        # 3. Create a clean prompt string
        #f"AXTree: {axtree_str}\n\n"
        prompt = (
            f"Goal: {goal}\n\n"
            "Task: Extract the buttons from thescreenshot and draw a red 1px border around each button and return response as svg."
        )

        # 4. Clear and rebuild messages correctly
        self.messages = [{
            'role': 'user',
            'content': prompt,
            'images': [img_bytes]  # Ollama accepts bytes directly
        }]

        response = self.client.chat(
            model='qwen3-vl',
            messages=self.messages,
            stream=True
        )

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
