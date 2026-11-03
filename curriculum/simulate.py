"""
Why use a curriculum? A small simulation you can run in seconds, no GPU or API.

A synthetic learner has a skill per task; an item (task, seed) has a difficulty;
P(success) = sigmoid(skill - difficulty). Like GRPO/DAPO, the learner only improves
from groups with MIXED rewards: an update moves the task's skill by
lr * std(group rewards), so all-pass and all-fail groups teach nothing.

Every strategy gets the same rollout budget. We compare:
  uniform          sample items uniformly, train on every group (no filtering)
  uniform+dapo     uniform sampling + DAPO dynamic sampling (drop all-same groups)
  zpd+dapo         curriculum weighted by p(1-p) + DAPO
  staged+dapo      easy -> zpd -> hard stages + DAPO

This illustrates the mechanism under stated assumptions; it is not evidence about
real RL training. Real runs: see README "Plugging into a trainer".

  python simulate.py                      # 5 repeats, default budget
  python simulate.py --budget 40000 --repeats 10
"""

from __future__ import annotations

import argparse
import math
import random
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from curriculum import Config, Curriculum, Item, dapo_fill_batch  # noqa: E402


def sigmoid(x: float) -> float:
    return 1 / (1 + math.exp(-x))


class World:
    def __init__(self, n_tasks: int, n_seeds: int, rng: random.Random, lr: float, offset: float, spread: float):
        self.rng, self.lr = rng, lr
        self.skill = {f"t{t}": rng.gauss(0.0, 1.2) for t in range(n_tasks)}
        self.diff = {(f"t{t}", s): rng.gauss(offset, spread) for t in range(n_tasks) for s in range(n_seeds)}
        self.rollouts = 0

    def p_true(self, task: str, seed: int) -> float:
        return sigmoid(self.skill[task] - self.diff[(task, seed)])

    def roll(self, task: str, seed: int, g: int) -> list[float]:
        self.rollouts += g
        p = self.p_true(task, seed)
        return [1.0 if self.rng.random() < p else 0.0 for _ in range(g)]

    def learn(self, task: str, rewards: list[float]) -> None:
        if len(set(rewards)) > 1:  # GRPO-style: zero advantage when all rewards are equal
            self.skill[task] += self.lr * statistics.pstdev(rewards)

    def mean_success(self) -> float:
        return sum(self.p_true(t, s) for t, s in self.diff) / len(self.diff)


def profile(world: World, k: int) -> list[Item]:
    """Estimate each item's pass rate from k rollouts, as profile.py does for real."""
    items = []
    for (t, s) in world.diff:
        succ = sum(world.roll(t, s, k))
        items.append(Item(key=f"sim/{t}/{s}", modality="sim", task=t, seed=s, p=succ / k, n_obs=k))
    return items


def run(strategy: str, seed: int, a) -> dict:
    rng = random.Random(seed)
    world = World(a.tasks, a.seeds, rng, a.lr, a.offset, a.spread)
    start = world.mean_success()
    items = profile(world, a.k)
    curve, wasted, used = [], 0, 0

    if strategy == "uniform":
        keys = list(world.diff)
        while world.rollouts < a.budget:
            for _ in range(a.batch):
                t, s = rng.choice(keys)
                r = world.roll(t, s, a.group)
                used += 1
                wasted += len(set(r)) == 1
                world.learn(t, r)
            curve.append((world.rollouts, world.mean_success()))
    else:
        name = strategy.split("+")[0]
        cur = Curriculum(items, Config(strategy=name, requeue_every=a.requeue), rng_seed=seed)
        while world.rollouts < a.budget:
            kept, st = dapo_fill_batch(cur, lambda it, sd, g: world.roll(it.task, sd, g), a.batch, a.group)
            used += st["drawn"]
            wasted += st["filtered_all_pass"] + st["filtered_all_fail"]
            for it, _, r in kept:
                world.learn(it.task, r)
            curve.append((world.rollouts, world.mean_success()))
            if st["drawn"] == 0:  # nothing active: everything parked until the next requeue
                cur.end_step()

    to80 = next((r for r, m in curve if m >= a.target), None)
    return {"start": start, "final": world.mean_success(), "to_target": to80,
            "wasted_frac": wasted / max(1, used)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tasks", type=int, default=12)
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--k", type=int, default=4, help="profiling rollouts per item")
    ap.add_argument("--group", type=int, default=8, help="rollouts per group (G)")
    ap.add_argument("--batch", type=int, default=16, help="groups per optimizer step")
    ap.add_argument("--budget", type=int, default=30000, help="total rollouts per strategy, incl. profiling")
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--offset", type=float, default=1.2, help="mean item difficulty (higher = harder pool)")
    ap.add_argument("--spread", type=float, default=1.5, help="spread of item difficulty (higher = more lopsided)")
    ap.add_argument("--requeue", type=int, default=20)
    ap.add_argument("--target", type=float, default=0.8)
    ap.add_argument("--repeats", type=int, default=5)
    a = ap.parse_args()

    strategies = ["uniform", "uniform+dapo", "zpd+dapo", "staged+dapo"]
    print(f"{a.tasks} tasks x {a.seeds} seeds, difficulty ~ N({a.offset}, {a.spread}), G={a.group}, batch={a.batch}, budget={a.budget} rollouts, "
          f"{a.repeats} repeats\n")
    print(f"| Strategy | Start | Final mean success | Rollouts to {int(a.target * 100)}% | Zero-signal groups |")
    print("|---|---:|---:|---:|---:|")
    for s in strategies:
        res = [run(s, seed, a) for seed in range(a.repeats)]
        hit = [r["to_target"] for r in res if r["to_target"]]
        reach = f"{statistics.mean(hit):,.0f} ({len(hit)}/{len(res)} runs)" if hit else f"not reached (0/{len(res)})"
        print(f"| {s} | {statistics.mean(r['start'] for r in res):.1%} "
              f"| {statistics.mean(r['final'] for r in res):.1%} ± {statistics.pstdev([r['final'] for r in res]):.1%} "
              f"| {reach} | {statistics.mean(r['wasted_frac'] for r in res):.0%} |")
    print("\nZero-signal groups: rollouts spent on all-pass or all-fail groups. Uniform trains on them "
          "(zero gradient); DAPO strategies discard them after paying for the rollouts.")


if __name__ == "__main__":
    main()
