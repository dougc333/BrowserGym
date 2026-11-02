"""Unit tests that need no browser and no model:  python -m pytest -q test_memory_agent.py"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from agent import describe_target, parse  # noqa: E402
from llm import ScriptedLLM  # noqa: E402
from memory import EpisodeRecord, StepRecord, TaskMemory  # noqa: E402
from report import mcnemar_exact, paired, wilson  # noqa: E402

AXTREE = "RootWebArea 'Task'\n\t[12] button 'ONE', clickable\n\t[13] textbox 'Username', clickable"


def test_parse_tagged_and_fallback():
    assert parse("<thinking>go</thinking>\n<action>click(\"12\")</action>") == ("go", 'click("12")')
    assert parse('I will do this\nclick("13")')[1] == 'click("13")'
    assert parse("no idea")[1] == "noop()"


def test_describe_target_survives_new_ids():
    assert describe_target('click("12")', AXTREE) == "button 'ONE'"
    assert describe_target('fill("13", "bob")', AXTREE) == "textbox 'Username'"
    assert describe_target('click("99")', AXTREE) == "(element not on page)"
    assert describe_target("scroll(0, 100)", AXTREE) == ""


def test_memory_roundtrip_and_retrieval(tmp_path):
    mem = TaskMemory(tmp_path, ScriptedLLM())
    win = EpisodeRecord("t", 0, "Click button ONE", True, [StepRecord('click("12")', "button 'ONE'")])
    other = EpisodeRecord("t", 1, "Enter the username alice", True, [StepRecord('fill("13","alice")', "textbox")])
    loss = EpisodeRecord("t", 2, "Click button TWO", False, [StepRecord('click("12")', "button 'ONE'")])
    for ep in (win, other):
        mem.add_episode(ep, consolidate=True)
    mem.add_episode(loss, consolidate=True)
    mem.add_episode(loss, consolidate=True)  # duplicate lesson is deduplicated
    assert [e.seed for e in mem.episodes()] == [0, 1, 2, 2]
    assert mem.retrieve_episodes("Click button TWO", k=1)[0].seed == 0  # most similar success
    assert all(e.success for e in mem.retrieve_episodes("anything", k=5))
    assert mem.workflow().startswith("1.")
    assert len(mem.reflections()) == 1
    assert mem.usage.calls == 4


def test_stats():
    lo, hi = wilson(5, 10)
    assert 0.23 < lo < 0.24 and 0.76 < hi < 0.77
    assert wilson(0, 0) == (0.0, 0.0)
    assert mcnemar_exact(0, 0) == 1.0
    assert abs(mcnemar_exact(10, 0) - 2 / 1024) < 1e-12
    a = {("t1", s): True for s in range(10)} | {("t2", s): s < 5 for s in range(10)}
    b = {("t1", s): s < 2 for s in range(10)} | {("t2", s): s < 5 for s in range(10)}
    r = paired(a, b, iters=500)
    assert r["n_pairs"] == 20 and abs(r["diff"] - 0.4) < 1e-9
    assert r["wins"] == 8 and r["losses"] == 0
    assert r["ci"][0] <= r["diff"] <= r["ci"][1]
