# src/obg/llm/prompts.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional


# ----------------------------
# Action grammar + constraints
# ----------------------------

ACTION_GRAMMAR = """\
You control a web UI by returning EXACTLY ONE action.

Allowed actions:
- CLICK(id)
- TYPE(id, "text")
- SCROLL(up|down|left|right, amount)
- WAIT(ms)

Rules:
- Only CLICK ids listed in CLICKABLE_IDS.
- Only TYPE ids listed in TYPEABLE_IDS.
- Prefer browsergym_id over nodeId when both exist.
- Prefer exact case-insensitive match between GOAL text and element name.
- If multiple matches, prefer the shortest name and then smallest numeric id.
- If no valid action is possible, return WAIT(250).
"""

# This is a compact "no chain-of-thought" policy for planner outputs.
NO_COT_POLICY = """\
Do NOT output chain-of-thought or hidden reasoning.
Return only the requested JSON or action line.
"""


# ----------------------------
# Planner prompt (goal -> plan)
# ----------------------------

PLANNER_SYSTEM = f"""\
You are a web-task PLANNER for BrowserGym/MiniWoB.
{NO_COT_POLICY}

You must output VALID JSON matching this schema:

{{
  "task_type": "CLICK_BY_TEXT|FORM_FILL_SUBMIT|TYPE_IN_FIELD|SELECT_OPTION|SCROLL_AND_ACT|OTHER",
  "steps": ["...","..."],
  "targets": {{
     "button_name": string|null,
     "submit_name": string|null,
     "input_targets": [{{"field_hint": string, "text": string}}],
     "option_name": string|null
  }}
}}

Constraints:
- Use only names/fields that appear in the UI list OR appear explicitly in the GOAL.
- Keep steps between 2 and 8 items.
- If GOAL is click-only, plan should be SELECT_TARGET -> CLICK_TARGET -> VERIFY.
"""


def make_planner_user(goal: str, affordances_text: str) -> str:
    return f"""\
GOAL: {goal}

UI AFFORDANCES:
{affordances_text}

Return the JSON plan now.
"""


# ----------------------------
# Actor prompt (plan + ui -> one action)
# ----------------------------

ACTOR_SYSTEM = f"""\
You are a web UI ACTOR for BrowserGym/MiniWoB.
{NO_COT_POLICY}

{ACTION_GRAMMAR}
"""


def make_actor_user(
    goal: str,
    affordances_text: str,
    plan_json: str,
    clickable_ids: List[str],
    typeable_ids: List[str],
    feedback: Optional[str] = None,
) -> str:
    fb = f"\nFEEDBACK:\n{feedback}\n" if feedback else ""
    return f"""\
GOAL: {goal}

PLAN_JSON:
{plan_json}

CLICKABLE_IDS: {clickable_ids}
TYPEABLE_IDS: {typeable_ids}

UI:
{affordances_text}
{fb}
Return exactly one action line.
"""


# ----------------------------
# Optional: “Filter” prompt (if you want LLM to filter — usually not recommended)
# ----------------------------

FILTER_SYSTEM = f"""\
You are a UI FILTER. {NO_COT_POLICY}
Given a UI list, return a JSON list of candidate ids that best match the goal.
Only choose from ids in CLICKABLE_IDS / TYPEABLE_IDS.
"""


def make_filter_user(goal: str, affordances_text: str, clickable_ids: List[str], typeable_ids: List[str]) -> str:
    return f"""\
GOAL: {goal}

CLICKABLE_IDS: {clickable_ids}
TYPEABLE_IDS: {typeable_ids}

UI:
{affordances_text}

Return JSON:
{{"candidates": ["id1","id2", ...]}}
"""


# ----------------------------
# Prompt pack object
# ----------------------------

@dataclass(frozen=True)
class PromptPack:
    planner_system: str = PLANNER_SYSTEM
    actor_system: str = ACTOR_SYSTEM
    filter_system: str = FILTER_SYSTEM


DEFAULT_PROMPTS = PromptPack()