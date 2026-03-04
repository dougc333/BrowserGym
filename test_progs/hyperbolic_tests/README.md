Hyperbolic not reliable 
- many errors at the FE in their website despite 50$ deposit
- limited context window at 32k. Even with paid account for model inference
- many error messages, service unavailable. 
- cant get API to reliably work Easier to run ollama on colab with localhost access or NGROK. 


the problem is the succint description of bid:12

A button is represented by a bid which is unique to wob. 
And there is a bounding box which indicates where it is on the page but there is no specificaiton of what and how the page looks. Very vague. Not enough for person or LLM to make decision

If there is a standardized formant then can SFT adn FT. 


Pointless. not worth running miniwob. Too mcuh work to reformat the data


Dont forget the ${} syntax for sh scripts. run as source .sh vs. ./program.sh. Source .sh will set environment vars in current shell while ./.sh will create a new shell. 

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