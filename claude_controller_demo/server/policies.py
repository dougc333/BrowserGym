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

import os
import random
import re
import time
from dataclasses import dataclass, field
from typing import Awaitable, Callable, Optional

from env_worker import ACTION_SET
from protocol import Observation, Prompt, Proposal, Usage

DEFAULT_MODEL = os.environ.get("CLAUDE_MODEL", "claude-sonnet-5-5")
AVAILABLE_MODELS = ["claude-haiku-5-5", "claude-sonnet-5-5", "claude-opus-5-5"]

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
    parts.append("# Next action\nThink briefly, then give one action.")
    return Prompt(system=SYSTEM_PROMPT, user="\n\n".join(parts))


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
    ) -> Proposal:
        model = model or DEFAULT_MODEL
        prompt = build_prompt(ctx)
        content: list[dict] = [{"type": "text", "text": prompt.user}]
        if ctx.use_screenshot and ctx.obs.screenshot:
            b64 = ctx.obs.screenshot.split(",", 1)[1]
            content.insert(
                0,
                {
                    "type": "image",
                    "source": {"type": "base64", "media_type": "image/jpeg", "data": b64},
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
