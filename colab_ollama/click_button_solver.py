#!/usr/bin/env python3
from __future__ import annotations

import os
import re
import sys
from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import browsergym.miniwob  # noqa: F401  # registers envs


ENV_ID = "browsergym/miniwob.click-button"


@dataclass
class Button:
    browsergym_id: str
    name: str
    node_id: str = ""

    @property
    def norm_name(self) -> str:
        return normalize(self.name)


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip()).casefold()


def get_axtree(obs: Any, info: Any) -> dict[str, Any]:
    ax = None
    if isinstance(obs, dict):
        ax = (
            obs.get("axtree_object")
            or obs.get("axtree")
            or obs.get("ax_tree")
            or obs.get("accessibility_tree")
        )
    if ax is None and isinstance(info, dict):
        ax = (
            info.get("axtree_object")
            or info.get("axtree")
            or info.get("ax_tree")
            or info.get("accessibility_tree")
        )
    if not isinstance(ax, dict) or "nodes" not in ax:
        raise RuntimeError(
            f"No AXTree found. obs keys={list(obs.keys()) if isinstance(obs, dict) else type(obs)}"
        )
    return ax


def extract_buttons(axtree: dict[str, Any]) -> list[Button]:
    out: list[Button] = []
    for n in axtree.get("nodes", []) or []:
        if n.get("ignored", False):
            continue
        role = str(((n.get("role") or {}).get("value")) or "").lower()
        if role != "button":
            continue
        bid = n.get("browsergym_id")
        if bid is None:
            continue
        name = str(((n.get("name") or {}).get("value")) or "")
        out.append(
            Button(
                browsergym_id=str(bid),
                name=name,
                node_id=str(n.get("nodeId", "")),
            )
        )
    out.sort(key=lambda b: (b.norm_name, int(b.browsergym_id) if b.browsergym_id.isdigit() else 10**9, b.browsergym_id))
    return out


def parse_goal_target(goal: str) -> str | None:
    g = (goal or "").strip()

    # Prefer quoted text if present.
    m = re.search(r"['\"]([^'\"]+)['\"]", g)
    if m:
        return m.group(1).strip()

    # Common MiniWoB click-button phrasings.
    patterns = [
        r"click\s+the\s+button\s+with\s+text\s+(.+)$",
        r"click\s+the\s+button\s+labeled\s+(.+)$",
        r"click\s+button\s+(.+)$",
        r"click\s+the\s+(.+?)\s+button$",
    ]
    for p in patterns:
        m = re.search(p, g, flags=re.IGNORECASE)
        if m:
            candidate = m.group(1).strip().strip(".")
            if candidate:
                return candidate

    return None


def choose_button(buttons: list[Button], goal: str) -> Button:
    if not buttons:
        raise RuntimeError("No buttons found in AXTree.")

    target = parse_goal_target(goal)
    if target:
        n_target = normalize(target)

        exact = [b for b in buttons if b.norm_name == n_target]
        if len(exact) == 1:
            return exact[0]
        if exact:
            return exact[0]

        contains = [b for b in buttons if n_target in b.norm_name or b.norm_name in n_target]
        if len(contains) == 1:
            return contains[0]
        if contains:
            return contains[0]

    if len(buttons) == 1:
        return buttons[0]

    # Fallback heuristic: if goal mentions one of the button names, use it.
    norm_goal = normalize(goal)
    for b in buttons:
        if b.norm_name and b.norm_name in norm_goal:
            return b

    names = ", ".join(repr(b.name) for b in buttons)
    raise RuntimeError(f"Could not identify target button from goal={goal!r}. Buttons: {names}")


def main() -> int:
    env_id = sys.argv[1] if len(sys.argv) > 1 else ENV_ID

    if not os.environ.get("MINIWOB_URL"):
        print(
            'MINIWOB_URL is not set. Example:\n  export MINIWOB_URL="http://127.0.0.1:8000/miniwob/"',
            file=sys.stderr,
        )
        return 2

    env = gym.make(env_id)
    try:
        obs, info = env.reset()
        if not isinstance(obs, dict):
            raise RuntimeError(f"Unexpected obs type: {type(obs)}")

        goal = str(obs.get("goal") or "")
        if not goal:
            raise RuntimeError(f"No goal found in obs keys={list(obs.keys())}")

        axtree = get_axtree(obs, info)
        buttons = extract_buttons(axtree)

        print(f"Goal: {goal}")
        print("Buttons:")
        for b in buttons:
            print(f"  bid={b.browsergym_id:>4} nodeId={b.node_id:>4} name={b.name!r}")

        chosen = choose_button(buttons, goal)
        print(f"Chosen button: bid={chosen.browsergym_id} name={chosen.name!r}")

        action = f"click('{chosen.browsergym_id}')"
        print(f"Action: {action}")
        obs, reward, terminated, truncated, info = env.step(action)
        print(
            f"After click: reward={reward} terminated={terminated} truncated={truncated}"
        )

        if isinstance(obs, dict) and obs.get("last_action_error"):
            print(f"last_action_error: {obs['last_action_error']}")

        return 0
    finally:
        env.close()


if __name__ == "__main__":
    raise SystemExit(main())
