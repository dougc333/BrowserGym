curl -X POST "https://api.hyperbolic.xyz/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer $HYPERBOLIC_API_KEY" \
    --data-raw '{
        "messages": [{
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "What is this image?"
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "data:image/png;base64,"
                        }
                    }
                ]
            }
        ],
        "model": "Qwen/Qwen2.5-VL-7B-Instruct",
        "max_tokens": 512,
        "temperature": 0.1,
        "top_p": 0.001,
        "stream": false
    }'
