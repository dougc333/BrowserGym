"""
LLM backends. Every backend exposes the same call:

    text, usage = llm.complete(system, user)

so the agent code never cares which model is behind it.

  anthropic   Claude via the Anthropic API (needs ANTHROPIC_API_KEY)
  openai      Any OpenAI-compatible server: Ollama, vLLM, LM Studio, OpenAI itself
  scripted    A tiny rule-based stand-in for dry runs; NOT a real result
"""

from __future__ import annotations

import json
import os
import random
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    calls: int = 0

    def add(self, other: "Usage") -> None:
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        self.calls += other.calls


def _retry(fn, attempts: int = 6):
    delay = 2.0
    for i in range(attempts):
        try:
            return fn()
        except Exception as e:  # rate limits, overloaded, transient network errors
            msg = str(e).lower()
            transient = any(
                s in msg for s in ("429", "rate", "overload", "529", "500", "502", "503", "timeout", "connection")
            )
            if not transient or i == attempts - 1:
                raise
            time.sleep(delay + random.random())
            delay = min(delay * 2, 60)


class AnthropicLLM:
    def __init__(self, model: str, temperature: float | None = 0.0, max_tokens: int = 600):
        import inspect

        from anthropic import Anthropic
        from anthropic.resources.messages import Messages

        self.client = Anthropic()
        self.model, self.max_tokens = model, max_tokens
        # Anthropic SDK 2.x no longer takes sampling parameters; older SDKs do.
        accepts = "temperature" in inspect.signature(Messages.create).parameters
        self.temperature = temperature if accepts else None

    def complete(self, system: str, user: str) -> tuple[str, Usage]:
        kwargs = dict(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        resp = _retry(lambda: self.client.messages.create(**kwargs))
        text = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
        return text, Usage(resp.usage.input_tokens, resp.usage.output_tokens, 1)


class OpenAICompatLLM:
    """POST /v1/chat/completions, standard library only (works with Ollama and vLLM)."""

    def __init__(self, model: str, base_url: str, api_key: str = "", temperature: float | None = 0.0,
                 max_tokens: int = 600):
        self.model, self.base_url = model, base_url.rstrip("/")
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "not-needed")
        self.temperature, self.max_tokens = temperature, max_tokens

    def complete(self, system: str, user: str) -> tuple[str, Usage]:
        body = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        }
        if self.temperature is not None:
            body["temperature"] = self.temperature

        def call():
            req = urllib.request.Request(
                f"{self.base_url}/chat/completions",
                data=json.dumps(body).encode(),
                headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"},
            )
            try:
                with urllib.request.urlopen(req, timeout=300) as r:
                    return json.loads(r.read())
            except urllib.error.HTTPError as e:
                raise RuntimeError(f"{e.code} {e.read()[:300]!r}") from e

        data = _retry(call)
        text = data["choices"][0]["message"].get("content") or ""
        u = data.get("usage") or {}
        return text, Usage(u.get("prompt_tokens", 0), u.get("completion_tokens", 0), 1)


class ScriptedLLM:
    """
    Deterministic stand-in used only to exercise the pipeline (prompting, parsing,
    memory writes, reports) without a model. It clicks an element whose name appears
    in the goal, else a random clickable element. Its numbers mean nothing.
    """

    def __init__(self, seed: int = 0):
        self.rng = random.Random(seed)

    def complete(self, system: str, user: str) -> tuple[str, Usage]:
        usage = Usage(len(system + user) // 4, 40, 1)
        if "Write the workflow" in user:
            return "1. Find the element named in the goal.\n2. Click it.\n3. Click Submit if present.", usage
        if "Write the lesson" in user:
            return "Click the element whose label matches the goal text exactly.", usage
        goal = (re.search(r"# Goal\n(.*)", user) or [None, ""])[1].lower()
        elems = re.findall(r"\[(\w+)\] (\w+) '([^']*)'", user.split("# Current page")[-1])
        clickable = [(b, r, n) for b, r, n in elems if r in ("button", "link", "checkbox", "radio", "tab", "option")]
        named = [b for b, _, n in clickable if n and n.lower() in goal]
        bid = named[0] if named else (self.rng.choice(clickable)[0] if clickable else None)
        action = f'click("{bid}")' if bid else "noop()"
        return f"<thinking>scripted</thinking>\n<action>{action}</action>", usage


def make_llm(spec: str, temperature: float | None = 0.0):
    """
    spec examples:
      anthropic:claude-haiku-5-5
      openai:qwen2.5:7b-instruct@http://localhost:11434/v1     (Ollama)
      openai:Qwen/Qwen2.5-7B-Instruct@http://localhost:8000/v1 (vLLM)
      scripted
    """
    if spec == "scripted":
        return ScriptedLLM()
    kind, _, rest = spec.partition(":")
    if kind == "anthropic":
        return AnthropicLLM(rest, temperature)
    if kind == "openai":
        model, _, url = rest.rpartition("@") if "@" in rest else (rest, "", "https://api.openai.com/v1")
        return OpenAICompatLLM(model, url, temperature=temperature)
    raise ValueError(f"Unknown LLM spec {spec!r}")
