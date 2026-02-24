import ollama
import gymnasium as gym
import browsergym.miniwob
from PIL import Image
import io
import httpx
import re

class TestColabQwenAxtree:
    def __init__(self):
        # Increased timeout to 300s to allow VLM time to think
        self.client = ollama.Client(
            host='http://localhost:11434',
            timeout=300.0 
        )
 
    def test_colabqwen_axtree(self):
        # Initialize the environment
        env = gym.make(
            "browsergym/miniwob.click-button",
            headless=True,
            disable_env_checker=True,
        )
        obs, info = env.reset(seed=0)
        
        # 1. Prepare and Save the Base Screenshot
        # We save as test.jpg so you can compare the SVG against it
        img = Image.fromarray(obs.get('screenshot'))
        img.save("test.jpg", format='JPEG', quality=90)
        print("--- Screenshot saved as test.jpg ---")

        # Prepare bytes for Ollama
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format='JPEG', quality=80)
        img_bytes = img_byte_arr.getvalue()

        # 2. Optimized Prompt for "Code-Only" Output
        prompt = (
            f"Goal: {obs.get('goal')}\n"
            "Find every button in this image. "
            "Return an SVG string containing only the <rect> elements with red 10px borders "
            "that perfectly overlay each button found. Output ONLY the SVG code."
        )

        print("--- Sending to Qwen3-VL (Streaming) ---")
        full_output = ""
        
        try:
            # 3. Stream the response to keep the connection alive
            stream = self.client.chat(
                model='qwen3-vl',
                messages=[{'role': 'user', 'content': prompt, 'images': [img_bytes]}],
                stream=True
            )

            for chunk in stream:
                content = chunk['message']['content']
                print(content, end='', flush=True)
                full_output += content
            
            # 4. The Regex Cleaner: Fixes the "Extra content" error
            # This looks for the start <svg and end </svg> regardless of other text
            svg_match = re.search(r'(<svg.*?</svg>)', full_output, re.DOTALL | re.IGNORECASE)
            
            if svg_match:
                clean_svg = svg_match.group(1)
                with open("overlay.svg", "w") as f:
                    f.write(clean_svg)
                print("\n\n--- Success: Valid SVG saved to overlay.svg ---")
            else:
                print("\n\n--- Warning: No valid SVG tags found. Saving raw text to raw_output.txt ---")
                with open("raw_output.txt", "w") as f:
                    f.write(full_output)

        except httpx.ReadTimeout:
            print("\nError: ReadTimeout. The model is likely working but taking too long.")
        except Exception as e:
            print(f"\nError: {e}")

if __name__ == "__main__":
    t = TestColabQwenAxtree()
    t.test_colabqwen_axtree()