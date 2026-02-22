curl -X POST "https://api.hyperbolic.xyz/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer ${HYPERBOLIC_API_KEY}" \
  -d '{
    "messages": [
      {"role": "user", "content": "What are three tips for learning to code?"}
    ],
    "model": "openai/gpt-oss-120b",
    "max_tokens": 512,
    "temperature": 0.7
  }'