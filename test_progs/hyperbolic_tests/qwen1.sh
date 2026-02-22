curl -X POST "https://api.hyperbolic.xyz/v1/chat/completions" \
      -H "Content-Type: application/json" \
      -H "Authorization: Bearer $HYPERBOLIC_API_KEY" \
      --data-raw '{
          "messages": [
              {
                "role": "user",
                "content": "What can I do in SF?"
              }
          ],
          "model": "Qwen/Qwen3-Next-80B-A3B-Instruct",
          "max_tokens": 507,
          "temperature": 0.7,
          "top_p": 0.8,
          "stream": false
      }'