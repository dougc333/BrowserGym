"""
Multimodal LLM backends. A prompt is a list of parts:

    {"type": "text", "text": "..."}
    {"type": "image", "png_b64": "..."}        # screenshots for the vision modes

    text, usage = llm.complete(system, parts)

  anthropic:<model>              Claude (text + images)
  openai:<model>@<base_url>      any OpenAI-compatible server (Ollama, vLLM); images
                                 need a vision model, e.g. qwen2.5vl:7b
  scripted                       rule-based stand-in for dry runs; NOT a real result
"""

from __future__ import annotations

import inspect
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

    def add(self, o: "Usage") -> None:
        self.input_tokens += o.input_tokens
        self.output_tokens += o.output_tokens
        self.calls += o.calls


def _retry(fn, attempts: int = 6):
    delay = 2.0
    for i in range(attempts):
        try:
            return fn()
        except Exception as e:
            msg = str(e).lower()
            transient = any(s in msg for s in ("429", "rate", "overload", "529", "500", "502", "503", "timeout", "connection"))
            if not transient or i == attempts - 1:
                raise
            time.sleep(delay + random.random())
            delay = min(delay * 2, 60)


class AnthropicLLM:
    def __init__(self, model: str, max_tokens: int = 600, temperature: float | None = None):
        from anthropic import Anthropic
        from anthropic.resources.messages import Messages

        self.client = Anthropic()
        self.model, self.max_tokens = model, max_tokens
        # SDK 2.x dropped sampling parameters; default sampling is stochastic,
        # which is what k-rollout difficulty estimation needs anyway.
        self.temperature = temperature if "temperature" in inspect.signature(Messages.create).parameters else None

    def complete(self, system: str, parts: list[dict]) -> tuple[str, Usage]:
        content = []
        for p in parts:
            if p["type"] == "text":
                content.append({"type": "text", "text": p["text"]})
            else:
                content.append({"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": p["png_b64"]}})
        kw = dict(model=self.model, max_tokens=self.max_tokens, system=system, messages=[{"role": "user", "content": content}])
        if self.temperature is not None:
            kw["temperature"] = self.temperature
        r = _retry(lambda: self.client.messages.create(**kw))
        text = "".join(b.text for b in r.content if getattr(b, "type", "") == "text")
        return text, Usage(r.usage.input_tokens, r.usage.output_tokens, 1)


class OpenAICompatLLM:
    def __init__(self, model: str, base_url: str, max_tokens: int = 600, temperature: float | None = 1.0):
        self.model, self.base_url = model, base_url.rstrip("/")
        self.api_key = os.environ.get("OPENAI_API_KEY", "not-needed")
        self.max_tokens, self.temperature = max_tokens, temperature

    def complete(self, system: str, parts: list[dict]) -> tuple[str, Usage]:
        content = [
            {"type": "text", "text": p["text"]} if p["type"] == "text"
            else {"type": "image_url", "image_url": {"url": "data:image/png;base64," + p["png_b64"]}}
            for p in parts
        ]
        body = {"model": self.model, "max_tokens": self.max_tokens,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": content}]}
        if self.temperature is not None:
            body["temperature"] = self.temperature

        def call():
            req = urllib.request.Request(
                f"{self.base_url}/chat/completions", data=json.dumps(body).encode(),
                headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"},
            )
            try:
                with urllib.request.urlopen(req, timeout=300) as r:
                    return json.loads(r.read())
            except urllib.error.HTTPError as e:
                raise RuntimeError(f"{e.code} {e.read()[:300]!r}") from e

        d = _retry(call)
        u = d.get("usage") or {}
        return d["choices"][0]["message"].get("content") or "", Usage(u.get("prompt_tokens", 0), u.get("completion_tokens", 0), 1)


class ScriptedLLM:
    """Plumbing-only stand-in: clicks a named element (axtree/som) or a random point (pixels)."""

    def __init__(self, seed: int = 0):
        self.rng = random.Random(seed)

    def complete(self, system: str, parts: list[dict]) -> tuple[str, Usage]:
        text = "\n".join(p["text"] for p in parts if p["type"] == "text")
        usage = Usage(len(text) // 4 + 300 * sum(p["type"] == "image" for p in parts), 30, 1)
        if "mouse_click(x, y)" in system:
            w, h = (int(v) for v in re.search(r"(\d+) x (\d+) pixels", text).groups())
            return f"<thinking>scripted</thinking><action>mouse_click({self.rng.randint(0, w - 1)}, {self.rng.randint(0, h - 1)})</action>", usage
        goal = (re.search(r"# Goal\n(.*)", text) or [None, ""])[1].lower()
        elems = re.findall(r"\[(\w+)\] (\w+) '([^']*)'", text)
        named = [b for b, _, n in elems if n and n.lower() in goal]
        bids = named or [b for b, *_ in elems] or [str(self.rng.randint(10, 40))]
        return f'<thinking>scripted</thinking><action>click("{self.rng.choice(bids)}")</action>', usage


def make_llm(spec: str):
    if spec == "scripted":
        return ScriptedLLM()
    kind, _, rest = spec.partition(":")
    if kind == "anthropic":
        return AnthropicLLM(rest)
    if kind == "openai":
        model, _, url = rest.rpartition("@") if "@" in rest else (rest, "", "https://api.openai.com/v1")
        return OpenAICompatLLM(model, url)
    raise ValueError(f"unknown LLM spec {spec!r}")
