
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
      self.prompt = """
      You are a helpful assistant that can answer questions and help with tasks.
      You are given an axtree and a goal.
      You need to return the next action to take.
      """
      self.messages = [
        {
            'role': 'user',
            'content': self.prompt
        }
      ]

    def test_colabqwen_axtree(self):
        pass

    def request(self):
        response = self.client.chat(
        model='qwen3-vl',
        messages=[{
            'role': 'user',
            'content': 'What is written in this image?',
            'images': ['/Users/dc/BrowserGym/test_progs/buttonclick.png'] # Use the local path on your Mac
          }]
        )
        print("Response from Colab:")
        print(response['message']['content'])
    