"""No browser, no model:  python -m pytest -q test_curriculum.py"""

import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from curriculum import Config, Curriculum, Item, bucket_of, dapo_fill_batch  # noqa: E402
from llm import ScriptedLLM  # noqa: E402


def items(ps):
    return [Item(key=f"m/t{i}/0", modality="m", task=f"t{i}", seed=0, p=p) for i, p in enumerate(ps)]


def test_buckets():
    assert [bucket_of(p) for p in (0, 0.1, 0.5, 0.9, 1)] == ["unsolvable", "hard", "zpd", "easy", "solved"]


def test_initial_parking_and_zpd_weights():
    cur = Curriculum(items([0.0, 0.5, 0.875, 1.0]))
    st = {it.p: it.status for it in cur.items.values()}
    assert st == {0.0: "frozen", 0.5: "active", 0.875: "active", 1.0: "retired"}
    w = {it.p: cur.weight(it) for it in cur.items.values()}
    assert w[0.5] > w[0.875] > 0 and w[0.0] == w[1.0] == 0


def test_sampling_prefers_zpd():
    cur = Curriculum(items([0.5, 0.97]), Config(floor=0.0), rng_seed=1)
    counts = {k: 0 for k in cur.items}
    for _ in range(2000):
        counts[cur.sample(1)[0][0].key] += 1
    assert counts["m/t0/0"] > 5 * counts["m/t1/0"]


def test_retire_freeze_requeue():
    cur = Curriculum(items([0.5, 0.5]), Config(retire_after=2, requeue_every=3, ema=0.5))
    for _ in range(2):
        cur.update("m/t0/0", [1] * 8)
        cur.update("m/t1/0", [0] * 8)
    assert cur.items["m/t0/0"].status == "retired" and cur.items["m/t1/0"].status == "frozen"
    assert cur.sample(5) == []
    for _ in range(3):
        cur.end_step()
    assert all(it.status == "active" for it in cur.items.values())
    assert all(0.05 <= it.p <= 0.95 for it in cur.items.values())


def test_staged_unlocks():
    cur = Curriculum(items([0.9, 0.5, 0.1]), Config(strategy="staged", stage_threshold=0.9, requeue_every=0))
    assert {it.task for it in cur.items.values() if cur.weight(it) > 0} == {"t0"}
    cur.update("m/t0/0", [1] * 7 + [0])
    cur.items["m/t0/0"].p = 0.95
    cur.end_step()
    assert cur.unlocked_stage == 1
    assert {it.task for it in cur.items.values() if cur.weight(it) > 0} == {"t0", "t1"}


def test_dapo_fill_batch_filters_all_same_groups():
    cur = Curriculum(items([0.5] * 6), Config(retire_after=99), rng_seed=0)
    rng = random.Random(0)
    outcome = {"m/t0/0": 1.0, "m/t1/0": 0.0}  # always pass / always fail

    def roll(it, seed, g):
        p = outcome.get(it.key, 0.5)
        return [1.0 if rng.random() < p else 0.0 for _ in range(g)]

    kept, st = dapo_fill_batch(cur, roll, batch_size=3, group_size=8)
    assert st["kept"] == 3 and all(0 < sum(r) < 8 for _, _, r in kept)
    assert all(it.key not in outcome for it, _, _ in kept)
    assert cur.step == 1


def test_profile_roundtrip(tmp_path):
    prof = tmp_path / "profile"
    prof.mkdir()
    rows = [dict(modality="pixels", task="click-color", seed=s, success=ok) for s in (0, 1) for ok in (True, False)]
    (prof / "pixels__click-color.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    inst = Curriculum.from_profile(prof, "instance")
    task = Curriculum.from_profile(prof, "task")
    assert sorted(inst.items) == ["pixels/click-color/0", "pixels/click-color/1"]
    assert list(task.items) == ["pixels/click-color"] and task.items["pixels/click-color"].p == 0.5
    _, seed = task.sample(1)[0]
    assert isinstance(seed, int)
    inst.save(tmp_path / "c.json")
    again = Curriculum.load(tmp_path / "c.json")
    assert again.items.keys() == inst.items.keys()


def test_scripted_llm_modes():
    llm = ScriptedLLM()
    t, _ = llm.complete("... mouse_click(x, y) ...", [{"type": "text", "text": "# Current screenshot (raw), 100 x 50 pixels"}])
    assert "mouse_click(" in t
    t, _ = llm.complete("ids", [{"type": "text", "text": "# Goal\nclick ok\n[13] button 'ok'"}])
    assert 'click("13")' in t
