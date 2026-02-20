import os
import requests

API_KEY = os.environ["HYPERBOLIC_API_KEY"]

url = "https://api.hyperbolic.xyz/v1/chat/completions"

headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {API_KEY}",
}

data = {
    "model": "openai/gpt-oss-120b",
    "messages": [
        {"role": "user", "content": "What are three tips for learning to code?"}
    ],
    "max_tokens": 512,
    "temperature": 0.7,
}

resp = requests.post(url, headers=headers, json=data)
resp.raise_for_status()

result = resp.json()

content = result["choices"][0]["message"]["content"]

# Strip reasoning channel if present
if "<|channel|>final<|message|>" in content:
    content = content.split("<|channel|>final<|message|>")[-1]

print("Final answer:\n")
print(content.strip())