"""
One web agent, three ways of seeing the page (the computer-vision axis of the curriculum):

  axtree   TEXT    Flattened accessibility tree. Acts on element ids: click("13").
  som      VISION  Screenshot with Set-of-Marks overlays (a numbered box drawn on
                   every interactive element). Acts on the same ids. No tree.
  pixels   VISION  Raw screenshot only. Acts on coordinates: mouse_click(120, 85).
                   This is the "computer use" setting: grounding is the model's job.

Some MiniWoB tasks are only solvable from pixels (click-color has an EMPTY
accessibility tree; visual-addition's blocks are not in the tree), others are
easiest from text. Profiling each task in each mode shows where vision is
necessary, where it helps, and where it costs accuracy.
"""

from __future__ import annotations

import base64
import io
import re
import time
from dataclasses import dataclass, field

import numpy as np
from PIL import Image

try:
    from browsergym.core.action.highlevel import HighLevelActionSet
    from browsergym.utils.obs import flatten_axtree_to_str, overlay_som
except ImportError as e:
    raise SystemExit(
        f"{e}. Install BrowserGym into this Python: "
        "pip install -e ../browsergym/core -e ../browsergym/miniwob && playwright install chromium"
    ) from e

from llm import Usage

MODALITIES = ["axtree", "som", "pixels"]

ACTION_SETS = {
    "axtree": HighLevelActionSet(subsets=["bid"], strict=False, multiaction=False),
    "som": HighLevelActionSet(subsets=["bid"], strict=False, multiaction=False),
    "pixels": HighLevelActionSet(subsets=["coord"], strict=False, multiaction=False),
}

SYSTEM = {
    "axtree": (
        "You control a web browser. You see the page as an accessibility tree; interactive "
        "elements are prefixed by an id in brackets, e.g. [13] button 'Submit'. Act on elements by id."
    ),
    "som": (
        "You control a web browser. You see a screenshot in which interactive elements are "
        "outlined with dashed boxes and labeled with a numeric id in a small black tag at the "
        "top-left of each box. Act on elements by that id, e.g. click(\"13\")."
    ),
    "pixels": (
        "You control a web browser through mouse and keyboard. You see a screenshot; act with "
        "pixel coordinates measured from the top-left corner of the screenshot, e.g. "
        "mouse_click(x, y) on the center of the target. To type, click the field first, then "
        "use keyboard_type(\"text\")."
    ),
}
FORMAT = (
    "\n\nReply in exactly this format and nothing else:\n"
    "<thinking>one to three short sentences</thinking>\n"
    "<action>one action call</action>"
)

ACTION_RE = re.compile(r"<action>\s*(.*?)\s*</action>", re.S)
THINK_RE = re.compile(r"<thinking>\s*(.*?)\s*</thinking>", re.S)
CALL_RE = re.compile(r"^\s*([a-z_]+\(.*\))\s*$", re.M)


def parse(text: str) -> tuple[str, str]:
    a, t = ACTION_RE.search(text), THINK_RE.search(text)
    action = a.group(1).strip() if a else (CALL_RE.findall(text) or ["noop()"])[-1]
    return (t.group(1).strip() if t else ""), action.strip("`").strip()


def png_b64(img: np.ndarray) -> str:
    buf = io.BytesIO()
    Image.fromarray(img).convert("RGB").save(buf, format="PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode()


def goal_of(obs: dict) -> str:
    return obs.get("goal") or " ".join(g.get("text", "") for g in obs.get("goal_object", []) if g.get("type") == "text")


def build_observation(obs: dict, modality: str) -> list[dict]:
    """The parts of o_t the policy sees, for one modality."""
    parts: list[dict] = [{"type": "text", "text": f"# Goal\n{goal_of(obs)}"}]
    if modality == "axtree":
        tree = flatten_axtree_to_str(
            obs["axtree_object"], extra_properties=obs["extra_element_properties"],
            with_clickable=True, filter_visible_only=True,
        )
        parts.append({"type": "text", "text": f"# Current page accessibility tree\n{tree}"})
    else:
        shot = obs["screenshot"]
        if modality == "som":
            shot = overlay_som(shot, extra_properties=obs["extra_element_properties"])
        h, w = shot.shape[:2]
        label = "with element ids" if modality == "som" else "raw"
        parts.append({"type": "text", "text": f"# Current screenshot ({label}), {w} x {h} pixels"})
        parts.append({"type": "image", "png_b64": png_b64(shot)})
    return parts


@dataclass
class Rollout:
    task: str
    seed: int
    modality: str
    rollout: int
    success: bool
    reward: float
    n_steps: int
    truncated: bool
    errors: int
    usage: Usage
    wall_s: float
    actions: list[str] = field(default_factory=list)
    goal: str = ""


def run_episode(env, llm, task: str, seed: int, modality: str, rollout: int, max_steps: int) -> Rollout:
    """One episode with working memory (the action history), in the given modality."""
    t0 = time.time()
    obs, _ = env.reset(seed=seed)
    goal = goal_of(obs)
    system = SYSTEM[modality] + FORMAT
    space = ACTION_SETS[modality].describe(with_long_description=False, with_examples=True)
    usage, history = Usage(), []
    reward, terminated, truncated, errors = 0.0, False, False, 0
    for _ in range(max_steps):
        parts = build_observation(obs, modality)
        if history:
            parts.append({"type": "text", "text": "# Actions you already took\n" + "\n".join(
                f"{i + 1}. {a}" + (f"  ERROR: {e[:160]}" if e else "") for i, (a, e) in enumerate(history))})
        parts.append({"type": "text", "text": f"# Action space\n{space}\n\n# Next action"})
        text, u = llm.complete(system, parts)
        usage.add(u)
        _, action = parse(text)
        obs, reward, terminated, truncated, _ = env.step(action)
        err = obs.get("last_action_error") or ""
        errors += bool(err)
        history.append((action, err))
        if terminated or truncated:
            break
    return Rollout(
        task=task, seed=seed, modality=modality, rollout=rollout,
        success=bool(terminated and reward > 0), reward=float(reward), n_steps=len(history),
        truncated=not terminated, errors=errors, usage=usage, wall_s=time.time() - t0,
        actions=[a for a, _ in history], goal=goal,
    )
