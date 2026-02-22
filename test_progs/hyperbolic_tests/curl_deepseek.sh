


curl -X POST "https://api.hyperbolic.xyz/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer ${HYPERBOLIC_API_KEY}" \
  -d '{
    "model": "deepseek-ai/DeepSeek-V3",
    "messages": [
      {"role": "user", "content": "Explain quantum computing in simple terms."}
    ],
    "max_tokens": 512,
    "temperature": 0.7
  }'