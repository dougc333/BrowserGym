"""
Policies: functions from an observation (plus history) to an action.

    pi(a_t | o_t, h_t)

* ClaudePolicy  - the LLM is the controller. It reads the AXTree text, reasons,
                  and emits one action string from BrowserGym's action space.
* RandomPolicy  - a baseline that clicks a random clickable element. Useful to
                  show what "no prior knowledge" looks like, and why sparse
                  rewards make pure RL exploration on the web so hard.
* (human)       - not a class: the front end sends `act` commands directly. When a
                  human overrides Claude, we keep both actions, which gives you
                  DAgger-style corrections and preference pairs for free.
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import os
import random
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field, replace
from typing import Awaitable, Callable, Optional

from PIL import Image, ImageDraw

from env_worker import ACTION_SET
from protocol import Observation, Prompt, Proposal, Usage

DEFAULT_MODEL = os.environ.get("CLAUDE_MODEL", "claude-sonnet-5-5")
AVAILABLE_MODELS = ["claude-haiku-5-5", "claude-sonnet-5-5", "claude-opus-5-5"]
NOUS_BASE_URL = os.environ.get("NOUS_BASE_URL", "https://inference-api.nousresearch.com/v1")
NOUS_DEFAULT_MODEL = os.environ.get("NOUS_MODEL", "deepseek/deepseek-v4.1-flash")
NOUS_MODELS = list(
    dict.fromkeys(
        [
            NOUS_DEFAULT_MODEL,
            "deepseek/deepseek-v4.1-flash",
            "deepseek/deepseek-v4-flash-vision-exp",
            "tencent/hy4-flash:free",
            "qwen/qwen3.7-flash",
        ]
    )
)
NOUS_REASONING_EFFORT = os.environ.get("NOUS_REASONING_EFFORT", "low")  # max|xhigh|high|low; "" = provider default
NOUS_TIMEOUT_SECONDS = float(os.environ.get("NOUS_TIMEOUT_SECONDS", "120"))

SYSTEM_PROMPT = """\
You are the controller of a web browser, acting as a policy in a reinforcement-learning \
environment (BrowserGym). At every step you receive an observation of the current page \
as an accessibility tree (AXTree). Interactive elements are prefixed with a browser id \
in square brackets, e.g. [13] button 'Submit'. You act by emitting exactly ONE action \
from the action space, referring to elements by their bid.

The episode ends when the task's success condition is met or the step budget runs out. \
Reward is sparse: 1.0 for success, 0 otherwise, so be precise and efficient.

Reply in exactly this format and nothing else:
<thinking>one to four short sentences: what the goal needs, what you see, what to do next</thinking>
<action>a single action call, e.g. click("13")</action>"""

ACTION_RE = re.compile(r"<action>\s*(.*?)\s*</action>", re.S)
THINK_RE = re.compile(r"<thinking>\s*(.*?)\s*</thinking>", re.S)
FENCE_RE = re.compile(r"```(?:python)?\s*(.*?)```", re.S)


@dataclass
class HistoryItem:
    action: str
    thought: str
    error: str = ""


@dataclass
class PolicyContext:
    obs: Observation
    history: list[HistoryItem] = field(default_factory=list)
    use_screenshot: bool = False


Emit = Callable[[str], Awaitable[None]]  # streams text deltas to the UI


def annotate_screenshot(data_url: str, step: int = 50) -> tuple[str, str]:
    """Overlay a labelled pixel grid so a vision model can read off exact coordinates.

    Returns (annotated data URL, a one-line note describing the image's coordinate system).
    Pixel (0, 0) is the top-left of the page and matches what mouse_* actions expect.
    """
    im = Image.open(io.BytesIO(base64.b64decode(data_url.split(",", 1)[1]))).convert("RGB")
    w, h = im.size
    overlay = Image.new("RGBA", im.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(overlay)
    for x in range(0, w, step):
        d.line([(x, 0), (x, h)], fill=(255, 0, 0, 90))
        d.text((x + 2, 1), str(x), fill=(255, 0, 0, 255))
    for y in range(step, h, step):
        d.line([(0, y), (w, y)], fill=(255, 0, 0, 90))
        d.text((2, y + 1), str(y), fill=(255, 0, 0, 255))
    im = Image.alpha_composite(im.convert("RGBA"), overlay).convert("RGB")
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=85)
    note = (
        f"The attached screenshot is {w}x{h} pixels. Red grid lines every {step}px are labelled "
        "with their pixel coordinates (x along the top, y down the left). Coordinates in "
        "mouse_* actions are in these same pixels; read positions off the grid, do not guess."
    )
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode(), note


def build_prompt(ctx: PolicyContext) -> Prompt:
    o = ctx.obs
    parts = [
        f"# Goal\n{o.goal}",
        "# Open tabs\n" + ("\n".join(f"- {t}" for t in o.open_tabs) or "- (none)"),
        f"# Current page accessibility tree (step {o.t})\n{o.axtree}",
        "# Action space\n"
        + ACTION_SET.describe(with_long_description=False, with_examples=True),
    ]
    if ctx.history:
        lines = []
        for i, h in enumerate(ctx.history):
            line = f"{i}. {h.action}"
            if h.error:
                line += f"   -> ERROR: {h.error}"
            lines.append(line)
        parts.append("# Actions taken so far\n" + "\n".join(lines))
    if o.last_action_error:
        parts.append(f"# Error from the last action\n{o.last_action_error}")
    image = None
    if ctx.use_screenshot and o.screenshot:
        # The exact (grid-annotated) image the model sees is stored on the prompt so that
        # SFT / preference exports can reproduce the multimodal input.
        image, note = annotate_screenshot(o.screenshot)
        parts.append("# Screenshot\n" + note)
    parts.append("# Next action\nThink briefly, then give one action.")
    return Prompt(system=SYSTEM_PROMPT, user="\n\n".join(parts), image=image)


def parse_response(text: str) -> tuple[str, str]:
    """Pull (thought, action) out of the model's reply. This is the policy's 'decoder'."""
    thought_m = THINK_RE.search(text)
    action_m = ACTION_RE.search(text) or FENCE_RE.search(text)
    thought = thought_m.group(1).strip() if thought_m else text.split("<action>")[0].strip()
    action = action_m.group(1).strip() if action_m else ""
    if not action:
        # Fall back to the last line that looks like a function call.
        calls = re.findall(r"^\s*([a-z_]+\(.*\))\s*$", text, re.M)
        action = calls[-1] if calls else 'noop()'
    return thought, action


class ClaudePolicy:
    name = "claude"

    def __init__(self) -> None:
        from anthropic import AsyncAnthropic  # imported lazily so the demo runs without it

        self.client = AsyncAnthropic()

    async def __call__(
        self,
        ctx: PolicyContext,
        emit: Emit,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        reasoning_effort: Optional[str] = None,  # ignored: Nous-only
    ) -> Proposal:
        model = model or DEFAULT_MODEL
        prompt = build_prompt(ctx)
        content: list[dict] = [{"type": "text", "text": prompt.user}]
        if prompt.image:
            content.insert(
                0,
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": prompt.image.split(",", 1)[1],
                    },
                },
            )
        kwargs: dict = dict(
            model=model,
            max_tokens=1024,
            system=prompt.system,
            messages=[{"role": "user", "content": content}],
        )
        if temperature is not None:
            kwargs["temperature"] = temperature

        t0 = time.time()
        chunks: list[str] = []
        async with self.client.messages.stream(**kwargs) as stream:
            async for delta in stream.text_stream:
                chunks.append(delta)
                await emit(delta)
            final = await stream.get_final_message()
        raw = "".join(chunks)
        thought, action = parse_response(raw)
        return Proposal(
            t=ctx.obs.t,
            policy="claude",
            model=model,
            thought=thought,
            action=action,
            raw=raw,
            prompt=prompt,
            latency_ms=int((time.time() - t0) * 1000),
            usage=Usage(
                input_tokens=final.usage.input_tokens, output_tokens=final.usage.output_tokens
            ),
        )


_NOUS_VISION_MODELS = {"tencent/hy4-flash:free", "qwen/qwen3.7-flash"}  # accept images per Nous /models


def _nous_is_vision(model: str) -> bool:
    return model in _NOUS_VISION_MODELS or "v4.1" in model or "vision" in model


class NousPolicy:
    """Nous Portal policy using its OpenAI-compatible chat-completions API."""

    name = "nous"

    def __init__(self) -> None:
        self.api_key = os.environ.get("NOUS_API_KEY", "")
        self.base_url = NOUS_BASE_URL.rstrip("/")

    def _complete(self, body: dict) -> dict:
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(body).encode(),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=NOUS_TIMEOUT_SECONDS) as response:
                return json.loads(response.read())
        except urllib.error.HTTPError as error:
            detail = error.read().decode(errors="replace")[:1000]
            raise RuntimeError(f"Nous API returned HTTP {error.code}: {detail}") from error

    async def __call__(
        self,
        ctx: PolicyContext,
        emit: Emit,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        reasoning_effort: Optional[str] = None,
    ) -> Proposal:
        model = model or NOUS_DEFAULT_MODEL
        if ctx.use_screenshot and not _nous_is_vision(model):
            # Text-only models (e.g. deepseek-v4-pro) reject images with HTTP 400, so run
            # them without the screenshot (and record the prompt without it).
            ctx = replace(ctx, use_screenshot=False)
        prompt = build_prompt(ctx)
        user_content: object = prompt.user
        if prompt.image:
            user_content = [
                {"type": "image_url", "image_url": {"url": prompt.image}},
                {"type": "text", "text": prompt.user},
            ]
        body: dict = {
            "model": model,
            "max_tokens": 4096,
            "messages": [
                {"role": "system", "content": prompt.system},
                {"role": "user", "content": user_content},
            ],
        }
        effort = reasoning_effort or NOUS_REASONING_EFFORT  # UI choice wins over the env default
        if effort:
            body["reasoning_effort"] = effort
        if temperature is not None:
            body["temperature"] = temperature

        t0 = time.time()
        try:
            result = await asyncio.wait_for(
                asyncio.to_thread(self._complete, body), timeout=NOUS_TIMEOUT_SECONDS + 1
            )
        except asyncio.TimeoutError as error:
            raise RuntimeError(
                f"Nous request exceeded {NOUS_TIMEOUT_SECONDS:g} seconds"
            ) from error
        choice = result["choices"][0]
        raw = choice["message"].get("content") or ""
        if not raw.strip() and choice.get("finish_reason") == "length":
            # Reasoning ate the whole output budget; don't silently turn that into noop().
            raise RuntimeError(
                "Nous model used all max_tokens on reasoning and returned no answer; "
                "lower NOUS_REASONING_EFFORT or raise max_tokens"
            )
        await emit(raw)
        thought, action = parse_response(raw)
        usage = result.get("usage") or {}
        return Proposal(
            t=ctx.obs.t,
            policy="nous",
            model=model,
            thought=thought,
            action=action,
            raw=raw,
            prompt=prompt,
            latency_ms=int((time.time() - t0) * 1000),
            usage=Usage(
                input_tokens=usage.get("prompt_tokens", 0),
                output_tokens=usage.get("completion_tokens", 0),
            ),
        )


class RandomPolicy:
    name = "random"

    def __init__(self, clickable: Callable[[], list[str]]) -> None:
        self.clickable = clickable

    async def __call__(self, ctx: PolicyContext, emit: Emit, **_) -> Proposal:
        bids = self.clickable()
        if bids:
            bid = random.choice(bids)
            thought = f"Random baseline: picked one of {len(bids)} clickable elements uniformly."
            action = f'click("{bid}")'
        else:
            thought, action = "Random baseline: nothing clickable, waiting.", "noop()"
        raw = f"<thinking>{thought}</thinking>\n<action>{action}</action>"
        for word in raw.split(" "):  # stream it so the UI behaves like the LLM case
            await emit(word + " ")
        return Proposal(
            t=ctx.obs.t,
            policy="random",
            thought=thought,
            action=action,
            raw=raw,
            prompt=build_prompt(ctx),
        )


def has_api_key() -> bool:
    return bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"))


_PRICE_CACHE: dict = {"at": 0.0, "prices": {}}


def nous_prices() -> dict[str, dict[str, float]]:
    """USD per token for our Nous models, from /v1/models. Cached for an hour."""
    now = time.time()
    if _PRICE_CACHE["prices"] and now - _PRICE_CACHE["at"] < 3600:
        return _PRICE_CACHE["prices"]
    try:
        req = urllib.request.Request(
            f"{NOUS_BASE_URL.rstrip('/')}/models",
            headers={"Authorization": f"Bearer {os.environ.get('NOUS_API_KEY', '')}"},
        )
        with urllib.request.urlopen(req, timeout=5) as r:
            data = json.loads(r.read())["data"]
        wanted = set(NOUS_MODELS)
        prices = {
            m["id"]: {
                "prompt": float(m["pricing"]["prompt"]),
                "completion": float(m["pricing"]["completion"]),
            }
            for m in data
            if m["id"] in wanted and m.get("pricing")
        }
    except Exception:
        return _PRICE_CACHE["prices"]  # stale or empty: the UI just hides the cost
    _PRICE_CACHE.update(at=now, prices=prices)
    return prices


def has_nous_key() -> bool:
    return bool(os.environ.get("NOUS_API_KEY"))
