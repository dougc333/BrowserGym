import requests
import os

url = "https://api.hyperbolic.xyz/v1/chat/completions"

api_key = os.environ.get("HYPERBOLIC_API_KEY")
if not api_key:
    raise RuntimeError("Please set HYPERBOLIC_API_KEY")

headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {api_key}",
}

data = {
    "messages": [
        {
            "role": "user",
            "content": "What can I do in SF?"
        }
    ],
    "model": "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "max_tokens": 512,
    "temperature": 0.7,
    "top_p": 0.9
}

response = requests.post(url, headers=headers, json=data, timeout=120)

response.raise_for_status()

print(response.json())