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
            timeout=300.0 
        )
 
    def test_colabqwen_axtree(self):
        env = gym.make(
            "browsergym/miniwob.click-button",
            headless=True,
            disable_env_checker=True,
        )
        obs, info = env.reset(seed=0)
        
        # 1. Prepare and Save Screenshot
        img = Image.fromarray(obs.get('screenshot'))
        # Save locally so you can inspect what the model is looking at
        img.save("test.jpg", format='JPEG', quality=90)
        print("--- Screenshot saved as test.jpg ---")

        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format='JPEG', quality=80)
        img_bytes = img_byte_arr.getvalue()

        # 2. Optimized Prompt
        prompt = (
            f"Goal: {obs.get('goal')}\n"
            "Find every button in this image. "
            "Return an SVG string containing only the <rect> elements with red 10px borders "
            "that perfectly overlay each button found. Output the SVG code only."
        )

        print("--- Sending to Qwen3-VL (Streaming) ---")
        
        # We will collect the full response string to save to a file
        full_svg_output = ""
        
        try:
            stream = self.client.chat(
                model='qwen3-vl',
                messages=[{'role': 'user', 'content': prompt, 'images': [img_bytes]}],
                stream=True
            )

            for chunk in stream:
                content = chunk['message']['content']
                print(content, end='', flush=True)
                full_svg_output += content
            
            # 3. Save the SVG output to a file
            with open("overlay.svg", "w") as f:
                f.write(full_svg_output)
            
            print("\n--- Done. Output saved to overlay.svg ---")

        except httpx.ReadTimeout:
            print("\nError: ReadTimeout. The model is likely overwhelmed or the GPU is throttled.")
        except Exception as e:
            print(f"\nError: {e}")

if __name__ == "__main__":
    t = TestColabQwenAxtree()
    t.test_colabqwen_axtree()
