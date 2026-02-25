# src/obg/llm/client.py
from __future__ import annotations

import base64
import json
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import requests


@dataclass
class LLMResponse:
    text: str
    raw: Dict[str, Any]
    latency_ms: int
    # Ollama often returns token counts/timings in some endpoints; keep optional.
    prompt_eval_count: Optional[int] = None
    eval_count: Optional[int] = None
    prompt_eval_duration_ns: Optional[int] = None
    eval_duration_ns: Optional[int] = None


class OllamaClient:
    """
    Minimal Ollama client for BrowserGym agents.

    Supports:
      - /api/chat (recommended for multi-turn + optional images)
      - /api/generate (single prompt)

    Notes:
      - For BrowserGym, you typically pass compact affordances + goal text.
      - For Qwen3-VL (vision), include images when needed (e.g. screenshot bytes).
      - This client is intentionally simple; add retries/backoff if desired.
    """

    def __init__(
        self,
        host: str = "http://127.0.0.1:11434",
        model: str = "qwen3-vl:latest",
        timeout_s: int = 120,
    ):
        self.host = host.rstrip("/")
        self.model = model
        self.timeout_s = timeout_s
        self._sess = requests.Session()

    # -------------------------
    # Public API
    # -------------------------

    def chat(
        self,
        messages: List[Dict[str, Any]],
        *,
        temperature: float = 0.2,
        top_p: float = 0.95,
        max_tokens: int = 256,
        stop: Optional[List[str]] = None,
        stream: bool = False,
        logprobs: bool = False,
        top_logprobs: Optional[int] = None,
        extra_options: Optional[Dict[str, Any]] = None,
    ) -> LLMResponse:
        """
        messages: list like OpenAI-ish:
          [{"role":"system","content":"..."}, {"role":"user","content":"..."}]

        For images (Qwen3-VL), include:
          {"role":"user", "content":"...", "images":[<base64str>, ...]}
        """
        url = f"{self.host}/api/chat"
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": stream,
            "options": {
                "temperature": temperature,
                "top_p": top_p,
                # Ollama naming: num_predict controls max generated tokens
                "num_predict": max_tokens,
            },
        }
        if stop:
            payload["options"]["stop"] = stop

        # logprobs support depends on Ollama build and endpoint;
        # safe to include if your Ollama supports it.
        if logprobs:
            payload["logprobs"] = True
            if top_logprobs is not None:
                payload["top_logprobs"] = int(top_logprobs)

        if extra_options:
            payload["options"].update(extra_options)

        t0 = time.time()
        r = self._sess.post(url, json=payload, timeout=self.timeout_s)
        latency_ms = int((time.time() - t0) * 1000)

        # For stream=True you would parse chunked JSON lines; keep stream=False for BrowserGym.
        r.raise_for_status()
        data = r.json()

        # Ollama chat response shape:
        # { message: { role, content }, done: bool, ...timings... }
        text = (data.get("message") or {}).get("content", "") or ""

        return LLMResponse(
            text=text,
            raw=data,
            latency_ms=latency_ms,
            prompt_eval_count=data.get("prompt_eval_count"),
            eval_count=data.get("eval_count"),
            prompt_eval_duration_ns=data.get("prompt_eval_duration"),
            eval_duration_ns=data.get("eval_duration"),
        )

    def generate(
        self,
        prompt: str,
        *,
        temperature: float = 0.2,
        top_p: float = 0.95,
        max_tokens: int = 256,
        stop: Optional[List[str]] = None,
        stream: bool = False,
        extra_options: Optional[Dict[str, Any]] = None,
    ) -> LLMResponse:
        """
        /api/generate is convenient for one-shot prompts, but /api/chat is better for agents.
        """
        url = f"{self.host}/api/generate"
        payload: Dict[str, Any] = {
            "model": self.model,
            "prompt": prompt,
            "stream": stream,
            "options": {
                "temperature": temperature,
                "top_p": top_p,
                "num_predict": max_tokens,
            },
        }
        if stop:
            payload["options"]["stop"] = stop
        if extra_options:
            payload["options"].update(extra_options)

        t0 = time.time()
        r = self._sess.post(url, json=payload, timeout=self.timeout_s)
        latency_ms = int((time.time() - t0) * 1000)
        r.raise_for_status()
        data = r.json()

        # generate response shape: { response: "...", ...timings... }
        text = data.get("response", "") or ""

        return LLMResponse(
            text=text,
            raw=data,
            latency_ms=latency_ms,
            prompt_eval_count=data.get("prompt_eval_count"),
            eval_count=data.get("eval_count"),
            prompt_eval_duration_ns=data.get("prompt_eval_duration"),
            eval_duration_ns=data.get("eval_duration"),
        )

    # -------------------------
    # Helpers for BrowserGym/VL
    # -------------------------

    @staticmethod
    def image_bytes_to_b64(img_bytes: bytes) -> str:
        return base64.b64encode(img_bytes).decode("utf-8")

    def browsergym_action(
        self,
        *,
        goal: str,
        affordances_text: str,
        system_rules: str,
        images_b64: Optional[List[str]] = None,
        temperature: float = 0.0,
        max_tokens: int = 128,
    ) -> LLMResponse:
        """
        Convenience wrapper for the common BrowserGym pattern:
          - system prompt: grammar + constraints
          - user prompt: GOAL + affordances list (+ optional images)
          - return: model text (you parse into CLICK/TYPE/etc)
        """
        user_content = f"GOAL: {goal}\n\nUI:\n{affordances_text}\n\nReturn exactly one action."
        msg: Dict[str, Any] = {"role": "user", "content": user_content}
        if images_b64:
            msg["images"] = images_b64

        return self.chat(
            messages=[
                {"role": "system", "content": system_rules},
                msg,
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )


# -------------------------
# Example usage (BrowserGym)
# -------------------------
if __name__ == "__main__":
    # Minimal smoke test
    client = OllamaClient(
        host="http://127.0.0.1:11434",
        model="qwen3-vl:latest",
    )

    system_rules = (
        "You are controlling a web UI with actions:\n"
        "CLICK(id)\nTYPE(id, \"text\")\nSCROLL(dir, amount)\nWAIT(ms)\n"
        "Only use ids shown in the UI list.\n"
    )

    affordances = "13 | button | \"no\"\n15 | button | \"Ok\"\n17 | button | \"submit\""
    r = client.browsergym_action(
        goal='Click on the "no" button.',
        affordances_text=affordances,
        system_rules=system_rules,
        images_b64=None,  # optionally pass screenshot b64(s)
        temperature=0.0,
    )
    print("LLM:", r.text)