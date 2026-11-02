"""
Long-term memory, stored on the filesystem (lecture: "Memory Systems: Filesystem").

One directory per (condition, task type):

    memory/<condition>/<task>/
        episodes.jsonl      EPISODIC  - every finished episode, success or not
        workflow.md         PROCEDURAL - AWM-style workflow induced from successes
        reflections.jsonl   PROCEDURAL - Reflexion-style lessons written after failures

Three operations, matching the lecture's "what / when / how":
  what  -> trajectories (episodic), workflows and lessons (procedural)
  when  -> written at the end of every episode (event-driven); read at the start
  how   -> files; episodic retrieval ranks by goal-word overlap, then recency
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

from llm import Usage

WORD = re.compile(r"[a-z0-9]+")


@dataclass
class StepRecord:
    action: str
    target: str  # human-readable element the action touched, e.g. "button 'Submit'"
    thought: str = ""
    error: str = ""


@dataclass
class EpisodeRecord:
    task: str
    seed: int
    goal: str
    success: bool
    steps: list[StepRecord] = field(default_factory=list)

    def render(self, with_thoughts: bool = False) -> str:
        lines = [f"Goal: {self.goal}"]
        for i, s in enumerate(self.steps):
            line = f"  {i + 1}. {s.action}  -> {s.target or '?'}"
            if s.error:
                line += f"  [error: {s.error[:120]}]"
            if with_thoughts and s.thought:
                line += f"\n     ({s.thought[:200]})"
            lines.append(line)
        lines.append(f"Outcome: {'SUCCESS' if self.success else 'FAILURE'}")
        return "\n".join(lines)


def _words(s: str) -> set[str]:
    return set(WORD.findall(s.lower()))


INDUCE_SYSTEM = (
    "You distill reusable procedures from an agent's successful web-task episodes "
    "(Agent Workflow Memory). Be concrete and short."
)
INDUCE_PROMPT = """\
Below are successful episodes of the same kind of web task. The specific values
(names, numbers, dates) differ between episodes, and element ids like "13" change
every time, so never refer to ids.

{episodes}

Write the workflow: a numbered list of at most 8 steps that solves ANY instance of
this task. Use placeholders such as {{target}} for values taken from the goal, and
describe elements by role and label (e.g. "the 'Submit' button"). Mention any
non-obvious mechanic you noticed. Output only the list."""

REFLECT_SYSTEM = "You help a web agent learn from its failures (Reflexion). Be specific and brief."
REFLECT_PROMPT = """\
The agent failed this episode:

{episode}

Write the lesson: one to three sentences on what went wrong and what to do
differently next time on this kind of task. Do not refer to element ids."""


class TaskMemory:
    def __init__(self, root: Path, llm, max_examples: int = 5, max_reflections: int = 3):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.llm = llm
        self.max_examples = max_examples
        self.max_reflections = max_reflections
        self.usage = Usage()  # tokens spent maintaining memory (not acting)

    # ---------------------------------------------------------------- read
    def episodes(self) -> list[EpisodeRecord]:
        p = self.root / "episodes.jsonl"
        if not p.exists():
            return []
        out = []
        for line in p.read_text().splitlines():
            d = json.loads(line)
            d["steps"] = [StepRecord(**s) for s in d["steps"]]
            out.append(EpisodeRecord(**d))
        return out

    def retrieve_episodes(self, goal: str, k: int = 2) -> list[EpisodeRecord]:
        """Episodic recall: successful past episodes most similar to this goal."""
        wins = [e for e in self.episodes() if e.success]
        g = _words(goal)
        ranked = sorted(
            enumerate(wins),
            key=lambda ie: (len(g & _words(ie[1].goal)) / (len(g | _words(ie[1].goal)) or 1), ie[0]),
            reverse=True,
        )
        return [e for _, e in ranked[:k]]

    def workflow(self) -> str:
        p = self.root / "workflow.md"
        return p.read_text().strip() if p.exists() else ""

    def reflections(self) -> list[str]:
        p = self.root / "reflections.jsonl"
        if not p.exists():
            return []
        lessons = [json.loads(l)["lesson"] for l in p.read_text().splitlines()]
        unique = list(dict.fromkeys(reversed(lessons)))  # newest first, duplicates dropped
        return list(reversed(unique[: self.max_reflections]))

    # ---------------------------------------------------------------- write
    def add_episode(self, ep: EpisodeRecord, consolidate: bool) -> None:
        with (self.root / "episodes.jsonl").open("a") as f:
            f.write(json.dumps(asdict(ep)) + "\n")
        if not consolidate:
            return
        if ep.success:
            self._induce_workflow()
        else:
            self._reflect(ep)

    def _induce_workflow(self) -> None:
        wins = [e for e in self.episodes() if e.success][-self.max_examples:]
        if not wins:
            return
        body = "\n\n".join(f"Episode {i + 1}\n{e.render()}" for i, e in enumerate(wins))
        text, u = self.llm.complete(INDUCE_SYSTEM, INDUCE_PROMPT.format(episodes=body))
        self.usage.add(u)
        if text.strip():
            (self.root / "workflow.md").write_text(text.strip() + "\n")

    def _reflect(self, ep: EpisodeRecord) -> None:
        text, u = self.llm.complete(REFLECT_SYSTEM, REFLECT_PROMPT.format(episode=ep.render(with_thoughts=True)))
        self.usage.add(u)
        if text.strip():
            with (self.root / "reflections.jsonl").open("a") as f:
                f.write(json.dumps({"seed": ep.seed, "lesson": text.strip()}) + "\n")
