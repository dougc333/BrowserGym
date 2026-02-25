# src/obg/agent/planner_actor.py
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from obg.browser.axtree import (
    Affordances,
    extract_affordances_from_axtree,
    format_affordances_for_prompt,
)
from obg.llm.client import OllamaClient, LLMResponse
from obg.llm.prompts import DEFAULT_PROMPTS, make_actor_user, make_planner_user


@dataclass
class Plan:
    task_type: str
    steps: List[str]
    targets: Dict[str, Any]


class PlannerActorAgent:
    """
    Two-stage agent for BrowserGym/MiniWoB:
      1) Planner: GOAL + UI -> JSON plan (short, no chain-of-thought)
      2) Actor: GOAL + PLAN + UI -> ONE action (CLICK/TYPE/SCROLL/WAIT)

    You typically run:
      aff = agent.extract_affordances(axtree)
      plan = agent.plan(goal, aff)
      action = agent.act(goal, plan, aff)
    """

    def __init__(
        self,
        llm: OllamaClient,
        *,
        prompts=DEFAULT_PROMPTS,
        prefer_browsergym_id: bool = True,
    ):
        self.llm = llm
        self.prompts = prompts
        self.prefer_browsergym_id = prefer_browsergym_id

    # -------------------------
    # Stage 0: affordances
    # -------------------------

    def extract_affordances(self, axtree: Dict[str, Any]) -> Affordances:
        return extract_affordances_from_axtree(
            axtree,
            prefer_browsergym_id=self.prefer_browsergym_id,
            keep_empty_names=False,
            max_per_section=50,
        )

    # -------------------------
    # Stage 1: plan
    # -------------------------

    def plan(self, goal: str, aff: Affordances) -> Plan:
        ui = format_affordances_for_prompt(aff)
        user = make_planner_user(goal, ui)

        resp = self.llm.chat(
            messages=[
                {"role": "system", "content": self.prompts.planner_system},
                {"role": "user", "content": user},
            ],
            temperature=0.0,
            max_tokens=512,
        )

        data = self._parse_planner_json(resp.text)
        return Plan(
            task_type=str(data.get("task_type", "OTHER")),
            steps=[str(s) for s in (data.get("steps") or [])][:12],
            targets=dict(data.get("targets") or {}),
        )

    # -------------------------
    # Stage 2: act
    # -------------------------

    def act(
        self,
        goal: str,
        plan: Plan,
        aff: Affordances,
        *,
        feedback: Optional[str] = None,
    ) -> str:
        ui = format_affordances_for_prompt(aff)
        plan_json = json.dumps(
            {"task_type": plan.task_type, "steps": plan.steps, "targets": plan.targets},
            ensure_ascii=False,
        )

        user = make_actor_user(
            goal=goal,
            affordances_text=ui,
            plan_json=plan_json,
            clickable_ids=aff.clickable_ids(),
            typeable_ids=aff.typeable_ids(),
            feedback=feedback,
        )

        resp = self.llm.chat(
            messages=[
                {"role": "system", "content": self.prompts.actor_system},
                {"role": "user", "content": user},
            ],
            temperature=0.0,
            max_tokens=128,
        )

        return self._parse_action(resp.text, aff)

    # -------------------------
    # Parsing helpers
    # -------------------------

    def _parse_planner_json(self, text: str) -> Dict[str, Any]:
        """
        Planner must output valid JSON. We tolerate ```json fences.
        """
        t = text.strip()
        t = re.sub(r"^\s*```(?:json)?\s*", "", t)
        t = re.sub(r"\s*```\s*$", "", t).strip()

        try:
            obj = json.loads(t)
            if not isinstance(obj, dict):
                raise ValueError("planner JSON was not an object")
            return obj
        except Exception:
            # last resort: extract first {...}
            m = re.search(r"\{.*\}", t, flags=re.DOTALL)
            if not m:
                raise ValueError(f"Planner returned non-JSON: {text[:4000]}")
            obj = json.loads(m.group(0))
            if not isinstance(obj, dict):
                raise ValueError("planner JSON was not an object")
            return obj

    def _parse_action(self, text: str, aff: Affordances) -> str:
        """
        Enforce: return exactly one action. If invalid -> WAIT(250)
        """
        t = (text.strip().splitlines() or [""])[0].strip()

        clickable = set(aff.clickable_ids())
        typeable = set(aff.typeable_ids())

        # CLICK(id)
        m = re.fullmatch(r"CLICK$begin:math:text$\(\[\^\)\]\+\)$end:math:text$", t, flags=re.IGNORECASE)
        if m:
            _id = m.group(1).strip().strip('"').strip("'")
            return f"CLICK({_id})" if _id in clickable else "WAIT(250)"

        # TYPE(id, "text")
        m = re.fullmatch(r"TYPE$begin:math:text$\\s\*\(\[\^\,\]\+\)\\s\*\,\\s\*\(\.\+\)\\s\*$end:math:text$", t, flags=re.IGNORECASE)
        if m:
            _id = m.group(1).strip().strip('"').strip("'")
            raw_txt = m.group(2).strip()
            txt_json = self._ensure_json_string(raw_txt)
            return f"TYPE({_id}, {txt_json})" if _id in typeable else "WAIT(250)"

        # SCROLL(dir, amount)
        m = re.fullmatch(r"SCROLL$begin:math:text$\\s\*\(up\|down\|left\|right\)\\s\*\,\\s\*\(\\d\+\)\\s\*$end:math:text$", t, flags=re.IGNORECASE)
        if m:
            direction = m.group(1).lower()
            amount = max(1, min(int(m.group(2)), 5000))
            return f"SCROLL({direction}, {amount})"

        # WAIT(ms)
        m = re.fullmatch(r"WAIT$begin:math:text$\\s\*\(\\d\+\)\\s\*$end:math:text$", t, flags=re.IGNORECASE)
        if m:
            ms = max(1, min(int(m.group(1)), 10000))
            return f"WAIT({ms})"

        return "WAIT(250)"

    def _ensure_json_string(self, s: str) -> str:
        """
        Ensure s is a JSON string literal (double-quoted, escapes OK).
        """
        if s.startswith('"') and s.endswith('"'):
            try:
                json.loads(s)
                return s
            except Exception:
                pass
        return json.dumps(s)


# -------------------------
# Minimal demo (no env)
# -------------------------
if __name__ == "__main__":
    llm = OllamaClient(model="qwen3-vl:latest")
    agent = PlannerActorAgent(llm)

    # Fake affordances (normally from AXTree)
    from obg.browser.axtree import Affordances, UIElement

    aff = Affordances(
        buttons=[UIElement(id="13", id_kind="browsergym_id", role="button", name="no", focusable=True)],
        links=[],
        inputs=[],
        checkboxes=[],
        radios=[],
        other_clickables=[],
    )

    plan = agent.plan('Click on the "no" button.', aff)
    act = agent.act('Click on the "no" button.', plan, aff)
    print(plan)
    print(act)