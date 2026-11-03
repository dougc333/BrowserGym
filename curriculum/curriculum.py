"""
Step 2: turn measured difficulty into an RL curriculum.

Each item is (modality, task, seed) or, with --level task, (modality, task) with a
fresh seed drawn whenever it is sampled. Each item carries a pass-rate estimate p
for the current policy, which puts it in a bucket:

    unsolvable  p = 0          no learning signal: GRPO/DAPO advantages are all zero
    hard        0 < p < 0.25   signal is rare and lands on one lucky success
    zpd         0.25..0.75     "zone of proximal development": most signal per rollout
    easy        0.75 < p < 1   signal lands on one unlucky failure
    solved      p = 1          no learning signal

Strategies
  zpd     sample active items with weight p(1-p) + floor (peaks at p = 0.5)
  staged  unlock easy -> zpd -> hard stages as the policy improves, zpd-weighted inside
  uniform baseline: every active item equally likely

Online maintenance, driven by the rewards your trainer reports back:
  * p is updated with an exponential moving average of each group's pass rate
  * items solved in `retire_after` consecutive groups are RETIRED
  * items failed in `retire_after` consecutive groups are FROZEN
  * every `requeue_every` steps, retired and frozen items are requeued once to be
    re-probed, since a policy can regress (solved items) or improve (frozen items)
  * `dapo_fill_batch` implements DAPO dynamic sampling on top of the curriculum

CLI
  python curriculum.py build  curricula/<name> [--level instance|task] [--strategy zpd]
  python curriculum.py sample curricula/<name> -n 8
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Iterable

BUCKETS = ["unsolvable", "hard", "zpd", "easy", "solved"]


def bucket_of(p: float, lo: float = 0.25, hi: float = 0.75) -> str:
    if p <= 0:
        return "unsolvable"
    if p >= 1:
        return "solved"
    return "hard" if p < lo else "zpd" if p <= hi else "easy"


@dataclass
class Item:
    key: str
    modality: str
    task: str
    seed: int | None  # None: draw a fresh seed each time (task-level curriculum)
    p: float  # current pass-rate estimate
    n_obs: int = 0  # rollouts observed (profile + training)
    status: str = "active"  # active | retired | frozen
    streak_solved: int = 0
    streak_failed: int = 0
    stage: int = 0
    times_sampled: int = 0
    history: list[float] = field(default_factory=list)  # recent group pass rates

    @property
    def bucket(self) -> str:
        return bucket_of(self.p)


@dataclass
class Config:
    strategy: str = "zpd"  # zpd | staged | uniform
    floor: float = 0.02  # minimum weight so no active item is starved
    ema: float = 0.3  # weight of the newest group in p
    retire_after: int = 3
    requeue_every: int = 50
    stage_threshold: float = 0.7  # unlock next stage when mean p of unlocked items reaches this
    seed_space: int = 100_000  # seeds drawn for task-level items


class Curriculum:
    def __init__(self, items: Iterable[Item], config: Config | None = None, rng_seed: int = 0):
        self.items: dict[str, Item] = {i.key: i for i in items}
        self.cfg = config or Config()
        self.rng = random.Random(rng_seed)
        self.step = 0
        self.unlocked_stage = 0
        if self.cfg.strategy == "staged":
            self._assign_stages()
        # items with no signal at the start are parked, not deleted
        for it in self.items.values():
            if it.p >= 1:
                it.status = "retired"
            elif it.p <= 0:
                it.status = "frozen"

    # -------------------------------------------------------------- construction
    @classmethod
    def from_profile(cls, profile_dir: Path, level: str = "instance", modalities: list[str] | None = None,
                     config: Config | None = None) -> "Curriculum":
        groups: dict[tuple, list[bool]] = defaultdict(list)
        for f in sorted(Path(profile_dir).glob("*.jsonl")):
            for line in f.read_text().splitlines():
                r = json.loads(line)
                if modalities and r["modality"] not in modalities:
                    continue
                key = (r["modality"], r["task"], r["seed"] if level == "instance" else None)
                groups[key].append(bool(r["success"]))
        items = []
        for (mod, task, seed), succ in sorted(groups.items(), key=lambda kv: str(kv[0])):
            key = f"{mod}/{task}" + (f"/{seed}" if seed is not None else "")
            items.append(Item(key=key, modality=mod, task=task, seed=seed, p=sum(succ) / len(succ), n_obs=len(succ)))
        return cls(items, config)

    def _assign_stages(self) -> None:
        for it in self.items.values():
            it.stage = 0 if it.p > 0.75 else 1 if it.p >= 0.25 else 2

    # -------------------------------------------------------------- sampling
    def weight(self, it: Item) -> float:
        if it.status != "active":
            return 0.0
        if self.cfg.strategy == "uniform":
            return 1.0
        if self.cfg.strategy == "staged" and it.stage > self.unlocked_stage:
            return 0.0
        return it.p * (1 - it.p) + self.cfg.floor

    def sample(self, n: int) -> list[tuple[Item, int]]:
        """Up to n distinct items (with a seed to run), weighted, without replacement."""
        pool = [(it, self.weight(it)) for it in self.items.values()]
        pool = [(it, w) for it, w in pool if w > 0]
        out = []
        for _ in range(min(n, len(pool))):
            total = sum(w for _, w in pool)
            x, acc = self.rng.random() * total, 0.0
            for j, (it, w) in enumerate(pool):
                acc += w
                if acc >= x:
                    out.append((it, it.seed if it.seed is not None else self.rng.randrange(self.cfg.seed_space)))
                    it.times_sampled += 1
                    pool.pop(j)
                    break
        return out

    # -------------------------------------------------------------- feedback
    def update(self, key: str, rewards: list[float]) -> None:
        """Report one group of rollouts (rewards in [0, 1]) for an item."""
        it = self.items[key]
        g = sum(1.0 if r > 0 else 0.0 for r in rewards) / len(rewards)
        it.p = (1 - self.cfg.ema) * it.p + self.cfg.ema * g
        it.n_obs += len(rewards)
        it.history = (it.history + [g])[-10:]
        if g >= 1:
            it.streak_solved, it.streak_failed = it.streak_solved + 1, 0
            if it.streak_solved >= self.cfg.retire_after:
                it.status = "retired"
        elif g <= 0:
            it.streak_failed, it.streak_solved = it.streak_failed + 1, 0
            if it.streak_failed >= self.cfg.retire_after:
                it.status = "frozen"
        else:
            it.streak_solved = it.streak_failed = 0

    def end_step(self) -> None:
        """Call once per optimizer step: requeue parked items periodically, unlock stages."""
        self.step += 1
        if self.cfg.requeue_every and self.step % self.cfg.requeue_every == 0:
            for it in self.items.values():
                if it.status in ("retired", "frozen"):
                    it.status = "active"
                    it.streak_solved = it.streak_failed = 0
                    # pull the estimate off the boundary so it can be sampled and re-measured
                    it.p = min(max(it.p, 0.05), 0.95)
        if self.cfg.strategy == "staged":
            unlocked = [it for it in self.items.values() if it.stage <= self.unlocked_stage]
            if unlocked and self.unlocked_stage < 2:
                if sum(it.p for it in unlocked) / len(unlocked) >= self.cfg.stage_threshold:
                    self.unlocked_stage += 1

    # -------------------------------------------------------------- reporting
    def stats(self) -> dict:
        its = list(self.items.values())
        return {
            "step": self.step,
            "items": len(its),
            "mean_p": round(sum(i.p for i in its) / max(1, len(its)), 4),
            "status": dict(Counter(i.status for i in its)),
            "buckets": {b: sum(i.bucket == b for i in its) for b in BUCKETS},
            "unlocked_stage": self.unlocked_stage if self.cfg.strategy == "staged" else None,
        }

    def save(self, path: Path) -> None:
        Path(path).write_text(json.dumps({
            "config": asdict(self.cfg), "step": self.step, "unlocked_stage": self.unlocked_stage,
            "items": [asdict(i) for i in self.items.values()],
        }, indent=1))

    @classmethod
    def load(cls, path: Path) -> "Curriculum":
        d = json.loads(Path(path).read_text())
        c = cls.__new__(cls)
        c.cfg, c.rng = Config(**d["config"]), random.Random(d["step"])
        c.items = {i["key"]: Item(**i) for i in d["items"]}
        c.step, c.unlocked_stage = d["step"], d["unlocked_stage"]
        return c


def dapo_fill_batch(cur: Curriculum, rollout: Callable[[Item, int, int], list[float]], batch_size: int,
                    group_size: int, oversample: float = 1.5, max_rounds: int = 8) -> tuple[list, dict]:
    """
    DAPO dynamic sampling on top of the curriculum. Draw items, roll out a group of
    `group_size` for each, report every group back to the curriculum, keep only
    groups with mixed rewards, and repeat until `batch_size` groups are kept.

    rollout(item, seed, group_size) -> list of rewards
    """
    kept, drawn, all_pass, all_fail = [], 0, 0, 0
    for _ in range(max_rounds):
        need = batch_size - len(kept)
        if need <= 0:
            break
        picks = cur.sample(max(1, math.ceil(need * oversample)))
        if not picks:
            break
        for it, seed in picks:
            rewards = rollout(it, seed, group_size)
            drawn += 1
            cur.update(it.key, rewards)
            mean = sum(r > 0 for r in rewards) / len(rewards)
            if mean == 1:
                all_pass += 1
            elif mean == 0:
                all_fail += 1
            elif len(kept) < batch_size:
                kept.append((it, seed, rewards))
    cur.end_step()
    return kept, {"drawn": drawn, "kept": len(kept), "filtered_all_pass": all_pass,
                  "filtered_all_fail": all_fail, "keep_rate": round(len(kept) / max(1, drawn), 3)}


# ------------------------------------------------------------------ build report
def build_report(cur: Curriculum, profile_dir: Path) -> str:
    rows = [json.loads(l) for f in sorted(Path(profile_dir).glob("*.jsonl")) for l in f.read_text().splitlines()]
    llm = rows[0]["llm"] if rows else "?"
    mods = [m for m in ("axtree", "som", "pixels") if any(r["modality"] == m for r in rows)]
    tasks = sorted({r["task"] for r in rows})
    by = defaultdict(list)
    for r in rows:
        by[(r["modality"], r["task"])].append(r)
    pr = lambda rs: sum(r["success"] for r in rs) / len(rs) if rs else None

    out = [f"# Curriculum: `{profile_dir.parent.name}`", "",
           f"Profiled with `{llm}`: {len(rows)} rollouts over {len(tasks)} tasks and {len(mods)} modalities. "
           f"Strategy `{cur.cfg.strategy}`, {len(cur.items)} items.", ""]
    if llm == "scripted":
        out += ["> **Dry run with the scripted stand-in policy: these pass rates are meaningless.**", ""]

    out += ["## Pass rate by task and modality", "",
            "| Task | " + " | ".join(mods) + " | Note |", "|---|" + "---:|" * len(mods) + "---|"]
    for t in tasks:
        ps = {m: pr(by[(m, t)]) for m in mods}
        cells = [f"{100 * ps[m]:.0f}% ({bucket_of(ps[m])})" if ps[m] is not None else "–" for m in mods]
        note = ""
        txt, vis = ps.get("axtree"), max([v for m, v in ps.items() if m != "axtree" and v is not None], default=None)
        if txt is not None and vis is not None:
            if txt == 0 and vis > 0:
                note = "vision necessary"
            elif vis - txt >= 0.25:
                note = "vision helps"
            elif txt - vis >= 0.25:
                note = "text easier"
        out.append(f"| {t} | " + " | ".join(cells) + f" | {note} |")

    out += ["", "## Modality summary", "",
            "| Modality | Pass rate | Unsolvable | Hard | ZPD | Easy | Solved | Tokens / episode | Steps / episode |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for m in mods:
        rs = [r for r in rows if r["modality"] == m]
        its = [i for i in cur.items.values() if i.modality == m]
        b = Counter(i.bucket for i in its)
        out.append(
            f"| {m} | {100 * pr(rs):.1f}% | " + " | ".join(str(b.get(k, 0)) for k in BUCKETS)
            + f" | {sum(r['input_tokens'] + r['output_tokens'] for r in rs) / len(rs):,.0f}"
            + f" | {sum(r['n_steps'] for r in rs) / len(rs):.1f} |")

    s = cur.stats()
    active = sorted((i for i in cur.items.values() if cur.weight(i) > 0), key=cur.weight, reverse=True)
    out += ["", "## Initial curriculum", "",
            f"- Active: {s['status'].get('active', 0)}, parked as solved (retired): {s['status'].get('retired', 0)}, "
            f"parked as unsolvable (frozen): {s['status'].get('frozen', 0)}",
            f"- Bucket counts: " + ", ".join(f"{k} {v}" for k, v in s["buckets"].items()), "",
            "Highest-weight items (sampled most often first):", "",
            "| Item | p | Bucket | Weight |", "|---|---:|---|---:|"]
    for it in active[:15]:
        out.append(f"| {it.key} | {it.p:.2f} | {it.bucket} | {cur.weight(it):.3f} |")
    if not active:
        out.append("| (none: every item is solved or unsolvable for this policy) | | | |")
    return "\n".join(out) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="build curriculum.json and report.md from a profile")
    b.add_argument("dir", type=Path)
    b.add_argument("--level", choices=["instance", "task"], default="instance")
    b.add_argument("--strategy", choices=["zpd", "staged", "uniform"], default="zpd")
    b.add_argument("--modalities", default="", help="comma list; default: all profiled")
    s = sub.add_parser("sample", help="draw a batch from a built curriculum")
    s.add_argument("dir", type=Path)
    s.add_argument("-n", type=int, default=8)
    a = ap.parse_args(sys.argv[1:] if len(sys.argv) > 1 else ["-h"])

    if a.cmd == "build":
        prof = a.dir / "profile"
        if not any(prof.glob("*.jsonl")):
            sys.exit(f"no profile in {prof}; run profile.py first")
        cur = Curriculum.from_profile(prof, a.level, [m for m in a.modalities.split(",") if m] or None,
                                      Config(strategy=a.strategy))
        cur.save(a.dir / "curriculum.json")
        md = build_report(cur, prof)
        (a.dir / "report.md").write_text(md)
        print(md)
        print(f"wrote {a.dir / 'curriculum.json'} and {a.dir / 'report.md'}")
    else:
        cur = Curriculum.load(a.dir / "curriculum.json")
        for it, seed in cur.sample(a.n):
            print(f"{it.key:<40} seed={seed:<6} p={it.p:.2f} {it.bucket}")


if __name__ == "__main__":
    main()
