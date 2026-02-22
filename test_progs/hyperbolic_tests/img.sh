

curl -X POST "https://api.hyperbolic.xyz/v1/image/generation" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $HYPERBOLIC_API_KEY" \
  -d '{
    "model_name": "FLUX.1-dev",
    "prompt": "A futuristic city skyline at sunset, cyberpunk style, neon lights",
    "height": 1024,
    "width": 1024,
    "steps": 30,
    "cfg_scale": 5
  }' | jq -r ".images[0].image" | base64 -d > generated_image.png