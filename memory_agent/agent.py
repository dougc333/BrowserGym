"""
One agent, four memory configurations (CoALA terms from the lecture):

  none        Nothing but the goal and the current AXTree. Every step starts from
              scratch: the agent cannot see what it already did.
  working     + working memory: this episode's actions, their targets and errors.
  episodic    + episodic memory: the most similar past SUCCESSFUL episodes of
              this task, retrieved from disk and shown as worked examples.
  procedural  + procedural memory: a workflow induced from past successes (AWM)
              and lessons written after past failures (Reflexion).

Everything else is identical: same model, same prompt skeleton, same action
space, same step budget. So any difference in success rate is due to memory.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field

from browsergym.core.action.highlevel import HighLevelActionSet
from browsergym.utils.obs import flatten_axtree_to_str

from llm import Usage
from memory import EpisodeRecord, StepRecord, TaskMemory

CONDITIONS = ["none", "working", "episodic", "procedural"]
USES_LONG_TERM = {"episodic", "procedural"}

ACTION_SET = HighLevelActionSet(subsets=["bid", "nav"], strict=False, multiaction=False)

SYSTEM = """\
You control a web browser to complete a task. You observe the page as an
accessibility tree: interactive elements are prefixed by an id in square
brackets, e.g. [13] button 'Submit'. Respond with exactly one action from the
action space, referring to elements by id.

Reply in exactly this format and nothing else:
<thinking>one to three short sentences</thinking>
<action>one action call, e.g. click("13")</action>"""

ACTION_RE = re.compile(r"<action>\s*(.*?)\s*</action>", re.S)
THINK_RE = re.compile(r"<thinking>\s*(.*?)\s*</thinking>", re.S)
CALL_RE = re.compile(r"^\s*([a-z_]+\(.*\))\s*$", re.M)


def parse(text: str) -> tuple[str, str]:
    a = ACTION_RE.search(text)
    t = THINK_RE.search(text)
    action = a.group(1).strip() if a else (CALL_RE.findall(text) or ["noop()"])[-1]
    return (t.group(1).strip() if t else ""), action.strip("`").strip()


def observe(obs: dict) -> tuple[str, str]:
    axtree = flatten_axtree_to_str(
        obs["axtree_object"],
        extra_properties=obs["extra_element_properties"],
        with_clickable=True,
        filter_visible_only=True,
    )
    goal = obs.get("goal") or " ".join(
        g.get("text", "") for g in obs.get("goal_object", []) if g.get("type") == "text"
    )
    return goal, axtree


def describe_target(action: str, axtree: str) -> str:
    """click("13") -> "button 'Submit'": a description that survives across seeds."""
    m = re.match(r"\s*\w+\(\s*['\"]([^'\"]+)['\"]", action)
    if not m:
        return ""
    line = re.search(rf"\[{re.escape(m.group(1))}\] ([^\n]*)", axtree)
    return line.group(1).split(", clickable")[0][:80] if line else "(element not on page)"


@dataclass
class EpisodeResult:
    success: bool
    reward: float
    n_steps: int
    truncated: bool
    usage: Usage
    record: EpisodeRecord
    wall_s: float
    memory_chars: int = 0  # how much long-term memory was injected into the prompt
    errors: int = 0
    repeated_actions: int = 0  # consecutive identical actions (a symptom of no working memory)
    prompts: list[str] = field(default_factory=list)


class MemoryAgent:
    def __init__(self, condition: str, llm, memory: TaskMemory | None, k_episodes: int = 2):
        assert condition in CONDITIONS, condition
        self.condition, self.llm, self.memory, self.k = condition, llm, memory, k_episodes

    # ------------------------------------------------------------- prompt
    def long_term_context(self, goal: str) -> str:
        if self.condition not in USES_LONG_TERM or self.memory is None:
            return ""
        parts = []
        eps = self.memory.retrieve_episodes(goal, self.k)
        if eps:
            parts.append(
                "# Memory: past successful episodes of this task (ids differ now; match by label)\n"
                + "\n\n".join(e.render() for e in eps)
            )
        if self.condition == "procedural":
            wf = self.memory.workflow()
            if wf:
                parts.append(f"# Memory: workflow learned from past successes\n{wf}")
            lessons = self.memory.reflections()
            if lessons:
                parts.append("# Memory: lessons from past failures\n" + "\n".join(f"- {l}" for l in lessons))
        return "\n\n".join(parts)

    def prompt(self, goal: str, axtree: str, history: list[StepRecord], ltm: str) -> str:
        parts = [f"# Goal\n{goal}"]
        if ltm:
            parts.append(ltm)
        parts.append(f"# Current page accessibility tree\n{axtree}")
        if self.condition != "none" and history:
            lines = []
            for i, s in enumerate(history):
                line = f"{i + 1}. {s.action} -> {s.target}"
                if s.error:
                    line += f"  ERROR: {s.error[:200]}"
                lines.append(line)
            parts.append("# Actions you already took in this episode\n" + "\n".join(lines))
        parts.append("# Action space\n" + ACTION_SET.describe(with_long_description=False, with_examples=True))
        parts.append("# Next action\nThink briefly, then give one action.")
        return "\n\n".join(parts)

    # ------------------------------------------------------------- episode
    def run_episode(self, env, task: str, seed: int, max_steps: int, keep_prompts: bool = False) -> EpisodeResult:
        t0 = time.time()
        obs, _ = env.reset(seed=seed)
        goal, axtree = observe(obs)
        ltm = self.long_term_context(goal)  # retrieved once, at episode start
        usage, history, prompts = Usage(), [], []
        reward, terminated, truncated, errors, repeats = 0.0, False, False, 0, 0

        for _ in range(max_steps):
            user = self.prompt(goal, axtree, history, ltm)
            if keep_prompts:
                prompts.append(user)
            text, u = self.llm.complete(SYSTEM, user)
            usage.add(u)
            thought, action = parse(text)
            target = describe_target(action, axtree)
            if history and history[-1].action == action:
                repeats += 1
            obs, reward, terminated, truncated, _ = env.step(action)
            err = obs.get("last_action_error") or ""
            errors += bool(err)
            history.append(StepRecord(action=action, target=target, thought=thought, error=err))
            if terminated or truncated:
                break
            goal_now, axtree = observe(obs)
            goal = goal_now or goal

        success = bool(terminated and reward > 0)
        record = EpisodeRecord(task=task, seed=seed, goal=goal, success=success, steps=history)
        return EpisodeResult(
            success=success,
            reward=float(reward),
            n_steps=len(history),
            truncated=not terminated,
            usage=usage,
            record=record,
            wall_s=time.time() - t0,
            memory_chars=len(ltm),
            errors=errors,
            repeated_actions=repeats,
            prompts=prompts,
        )
