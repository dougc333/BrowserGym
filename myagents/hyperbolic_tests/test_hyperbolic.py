import os
import sys
import json
import requests

URL = "https://api.hyperbolic.xyz/v1/chat/completions"

API_KEY = os.environ.get("HYPERBOLIC_API_KEY")
if not API_KEY:
    raise RuntimeError("HYPERBOLIC_API_KEY environment variable not set")

HEADERS = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {API_KEY}",
}

def hyperbolic_chat(
    prompt: str,
    model: str = "openai/gpt-oss-120b",  # known-good from your curl
    max_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float | None = None,
    timeout_s: int = 120,
):
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if top_p is not None:
        payload["top_p"] = top_p

    r = requests.post(URL, headers=HEADERS, json=payload, timeout=timeout_s)

    # Always parse JSON (or show raw text if not JSON)
    try:
        data = r.json()
    except Exception:
        raise RuntimeError(f"Non-JSON response (status={r.status_code}):\n{r.text[:2000]}")

    # If not OK, print full error and raise
    if r.status_code != 200:
        print(f"HTTP {r.status_code}")
        print(json.dumps(data, indent=2)[:4000])
        raise SystemExit(1)

    # Hyperbolic returns OpenAI-style `choices` on success
    if "choices" not in data:
        print("Unexpected success payload shape:")
        print(json.dumps(data, indent=2)[:4000])
        raise SystemExit(2)

    return data["choices"][0]["message"]["content"], data

if __name__ == "__main__":
    prompt = "Explain quantum computing in simple terms."
    model = sys.argv[1] if len(sys.argv) > 1 else "openai/gpt-oss-120b"

    text, raw = hyperbolic_chat(prompt, model=model, max_tokens=256, temperature=0.7)
    print(text)