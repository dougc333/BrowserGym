"""
Step 1 of the curriculum: measure difficulty.

For every (modality, task, seed) run k independent rollouts with the current policy
and record success. A task instance's pass rate p = successes / k is its difficulty
*for this policy in this modality*, which is what an RL curriculum needs.

Results go to curricula/<name>/profile/<modality>__<task>.jsonl (resumable).

  python profile.py --name smoke --llm scripted --tasks quick --seeds 2 --k 2
  python profile.py --name haiku --llm anthropic:claude-haiku-5-5 --tasks mixed --workers 4
  python curriculum.py build curricula/haiku
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

TASK_PRESETS = {
    "quick": ["click-button", "login-user", "click-color", "identify-shape"],
    # text-friendly: everything needed is in the accessibility tree
    "text": [
        "click-button", "click-checkboxes", "login-user", "enter-text", "choose-date",
        "use-autocomplete", "click-tab-2", "search-engine", "book-flight", "email-inbox",
    ],
    # vision-heavy: the answer is (partly or wholly) only in the pixels
    "visual": [
        "click-color",       # accessibility tree is empty: pixels only
        "click-shape",       # SVG shapes, colors and sizes
        "count-shape",       # count shapes of a kind
        "identify-shape",    # the figure is an unlabeled image
        "visual-addition",   # count blocks that are not in the tree
        "grid-coordinate",   # map (x, y) grid coordinates to a click
        "click-pie",         # radial menu
        "circle-center",     # click the center of a circle
    ],
}
TASK_PRESETS["mixed"] = TASK_PRESETS["text"][:5] + TASK_PRESETS["visual"][:6]


def preflight(llm_spec: str) -> None:
    problems = []
    try:
        import browsergym.core  # noqa: F401
        import browsergym.miniwob  # noqa: F401
    except ImportError as e:
        problems.append(f"{e}. Install into {sys.executable}: pip install -e ../browsergym/core "
                        "-e ../browsergym/miniwob && playwright install chromium")
    url = os.environ.get("MINIWOB_URL", "")
    if not url:
        problems.append("MINIWOB_URL is not set.")
    elif url.startswith("file://") and not Path(url[7:], "click-test.html").exists():
        problems.append(f"MINIWOB_URL={url!r} does not contain click-test.html.")
    if llm_spec.startswith("anthropic:") and not os.environ.get("ANTHROPIC_API_KEY", "").startswith("sk-ant-"):
        problems.append("ANTHROPIC_API_KEY is missing or not a real key.")
    if problems:
        sys.exit("Cannot start:\n  - " + "\n  - ".join(problems))


def job(a: dict) -> dict:
    os.environ.setdefault("MINIWOB_URL", a["miniwob_url"])
    import gymnasium as gym

    import browsergym.miniwob  # noqa: F401
    from llm import make_llm
    from policy import ACTION_SETS, run_episode

    shard = Path(a["out"]) / "profile" / f"{a['modality']}__{a['task']}.jsonl"
    shard.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if shard.exists():
        for line in shard.read_text().splitlines():
            r = json.loads(line)
            done.add((r["seed"], r["rollout"]))
    llm, env = make_llm(a["llm"]), None
    try:
        for seed in a["seeds"]:
            for k in range(a["k"]):
                if (seed, k) in done:
                    continue
                if env is None:
                    env = gym.make(f"browsergym/miniwob.{a['task']}", headless=True,
                                   action_mapping=ACTION_SETS[a["modality"]].to_python_code)
                r = run_episode(env, llm, a["task"], seed, a["modality"], k, a["max_steps"])
                row = {
                    "llm": a["llm"], "modality": r.modality, "task": r.task, "seed": r.seed,
                    "rollout": r.rollout, "success": r.success, "reward": r.reward, "n_steps": r.n_steps,
                    "truncated": r.truncated, "errors": r.errors, "input_tokens": r.usage.input_tokens,
                    "output_tokens": r.usage.output_tokens, "wall_s": round(r.wall_s, 2),
                    "goal": r.goal, "actions": r.actions,
                }
                with shard.open("a") as f:
                    f.write(json.dumps(row) + "\n")
                print(f"[{r.modality:>6} {r.task:<18}] seed {seed:<4} #{k} {'PASS' if r.success else 'fail'} "
                      f"steps={r.n_steps}", flush=True)
    except Exception:
        return {"ok": False, "job": f"{a['modality']}/{a['task']}", "error": traceback.format_exc()}
    finally:
        if env is not None:
            env.close()
    return {"ok": True}


def main() -> None:
    from policy import MODALITIES

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True, help="curriculum name (curricula/<name>)")
    ap.add_argument("--llm", required=True, help="anthropic:<model> | openai:<model>@<url> | scripted")
    ap.add_argument("--tasks", default="mixed", help=f"preset {sorted(TASK_PRESETS)} or comma list")
    ap.add_argument("--modalities", default=",".join(MODALITIES))
    ap.add_argument("--seeds", type=int, default=4, help="task instances per task (seeds 0..N-1)")
    ap.add_argument("--k", type=int, default=4, help="rollouts per instance (pass rate = successes / k)")
    ap.add_argument("--max-steps", type=int, default=12)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--out-dir", default=str(HERE / "curricula"))
    args = ap.parse_args()

    preflight(args.llm)
    tasks = TASK_PRESETS.get(args.tasks) or [t.strip() for t in args.tasks.split(",") if t.strip()]
    mods = [m.strip() for m in args.modalities.split(",")]
    bad = [m for m in mods if m not in MODALITIES]
    if bad:
        sys.exit(f"unknown modality {bad}; choose from {MODALITIES}")
    out = Path(args.out_dir) / args.name
    out.mkdir(parents=True, exist_ok=True)
    (out / "profile_config.json").write_text(json.dumps(dict(vars(args), tasks=tasks, modalities=mods), indent=2))

    jobs = [dict(out=str(out), llm=args.llm, task=t, modality=m, seeds=list(range(args.seeds)), k=args.k,
                 max_steps=args.max_steps, miniwob_url=os.environ["MINIWOB_URL"]) for t in tasks for m in mods]
    print(f"{len(jobs)} jobs, {len(jobs) * args.seeds * args.k} episodes, llm={args.llm}", flush=True)
    t0, fails = time.time(), []
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context("spawn")) as pool:
        for f in as_completed([pool.submit(job, j) for j in jobs]):
            r = f.result()
            if not r["ok"]:
                fails.append(r)
                print(f"!! {r['job']} failed:\n{r['error']}", flush=True)
    print(f"done in {time.time() - t0:.0f}s; {len(fails)} failed jobs (re-run to resume)")
    print(f"next: python {HERE / 'curriculum.py'} build {out}")


if __name__ == "__main__":
    main()
