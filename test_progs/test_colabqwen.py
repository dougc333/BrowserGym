
import ollama

# Use the "Colab Server is LIVE at" URL from your Colab output
COLAB_NGROK_URL = 'https://labrador-fair-trivially.ngrok-free.app'

client = ollama.Client(
    host=COLAB_NGROK_URL,
    headers={'ngrok-skip-browser-warning': 'true'}
)

try:
    response = client.chat(
        model='qwen3-vl',
        messages=[{
            'role': 'user',
            'content': 'What is written in this image?',
            'images': ['/Users/dc/BrowserGym/test_progs/buttonclick.png'] # Use the local path on your Mac
        }]
    )
    print("Response from Colab:")
    print(response['message']['content'])
except Exception as e:
    print(f"Connection Error: {e}")