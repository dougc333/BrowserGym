import ollama
import gymnasium as gym
import browsergym.miniwob
from PIL import Image
import io
import httpx

class TestColabQwenAxtree:
    def __init__(self):
        self.client = ollama.Client(
            host='http://localhost:11434',
            # We increase this to 300 (5 mins) just in case, 
            # but streaming will mostly solve the timeout issue.
            timeout=300.0 
        )
 
    def test_colabqwen_axtree(self):
        env = gym.make(
            "browsergym/miniwob.click-button",
            headless=True,
            disable_env_checker=True,
        )
        obs, info = env.reset(seed=0)
        
        # 1. Prepare Screenshot
        img = Image.fromarray(obs.get('screenshot'))
        img_byte_arr = io.BytesIO()
        # Using JPEG with slightly lower quality reduces the data 
        # the model has to "read," making it faster.
        img.save(img_byte_arr, format='JPEG', quality=80)
        img_bytes = img_byte_arr.getvalue()

        # 2. Optimized Prompt
        prompt = (
            f"Goal: {obs.get('goal')}\n"
            "Find every button in this image. "
            "Return an SVG string containing only the <rect> elements with red 10px borders "
            "that perfectly overlay each button found."
        )

        print("--- Sending to Qwen3-VL (Streaming) ---")
        
        try:
            # 3. Use Streaming to prevent ReadTimeout
            stream = self.client.chat(
                model='qwen3-vl',
                messages=[{'role': 'user', 'content': prompt, 'images': [img_bytes]}],
                stream=True
            )

            for chunk in stream:
                content = chunk['message']['content']
                print(content, end='', flush=True)
            print("\n--- Done ---")

        except httpx.ReadTimeout:
            print("\nError: The model took too long to start responding. Check Colab GPU usage.")
        except Exception as e:
            print(f"\nError: {e}")

if __name__ == "__main__":
    t = TestColabQwenAxtree()
    t.test_colabqwen_axtree()