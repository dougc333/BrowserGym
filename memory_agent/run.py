"""
Memory ablation on MiniWoB.

For every (condition, task) pair:
  1. TRAIN phase: run seeds 0..T-1 in order. After each episode, agents with
     long-term memory write it to disk (and induce workflows / reflect).
  2. TEST phase: run held-out seeds 1000..1000+N-1 with memory FROZEN.

All conditions see the same seeds, so results are paired. Results are appended to
results/<exp>/runs/<condition>__<task>.jsonl and the run is resumable.

Examples
  python run.py --exp smoke --llm scripted --tasks quick --train 2 --test 2
  python run.py --exp haiku --llm anthropic:claude-haiku-5-5 --tasks memory12 --workers 4
  python run.py --exp qwen7b --llm "openai:qwen2.5:7b-instruct@http://localhost:11434/v1"
Then: python report.py results/<exp>
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
from dataclasses import asdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

TASK_PRESETS = {
    # Quick smoke test: one single-step task and three multi-step tasks.
    "quick": ["click-button", "click-checkboxes", "login-user", "choose-date"],
    # Main set: tasks where an agent must track progress across several steps
    # (working memory) or where a non-obvious mechanic can be learned (long-term memory).
    "memory12": [
        "click-checkboxes",       # several boxes, then Submit: must remember which are done
        "click-button-sequence",  # two buttons in order
        "login-user",             # fill two fields, then log in
        "enter-date",             # date input with a specific format
        "choose-date",            # datepicker month navigation
        "use-autocomplete",       # type, then pick from a suggestion list
        "book-flight",            # long form with autocomplete and a date picker
        "search-engine",          # search, then paginate to the n-th result
        "email-inbox",            # find an email and act on it
        "click-tab-2",            # find a link hidden behind tabs
        "navigate-tree",          # expand a folder tree
        "guess-number",           # needs feedback from earlier guesses
    ],
}
TEST_SEED_OFFSET = 1000


def job(args: dict) -> dict:
    """Runs one (condition, task) pair in its own process with its own Chromium."""
    os.environ.setdefault("MINIWOB_URL", args["miniwob_url"])
    import gymnasium as gym

    import browsergym.miniwob  # noqa: F401
    from agent import ACTION_SET, USES_LONG_TERM, MemoryAgent
    from llm import make_llm
    from memory import TaskMemory

    exp, cond, task = Path(args["exp_dir"]), args["condition"], args["task"]
    shard = exp / "runs" / f"{cond}__{task}.jsonl"
    shard.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if shard.exists():
        for line in shard.read_text().splitlines():
            r = json.loads(line)
            done.add((r["phase"], r["seed"]))

    llm = make_llm(args["llm"], args["temperature"])
    mem = TaskMemory(exp / "memory" / cond / task, llm) if cond in USES_LONG_TERM else None
    agent = MemoryAgent(cond, llm, mem, k_episodes=args["k_episodes"])
    plan = [("train", s) for s in range(args["train"])] + [
        ("test", TEST_SEED_OFFSET + s) for s in range(args["test"])
    ]
    env = None
    n_new = 0
    try:
        for idx, (phase, seed) in enumerate(plan):
            if (phase, seed) in done:
                continue
            if env is None:
                env = gym.make(
                    f"browsergym/miniwob.{task}",
                    headless=True,
                    action_mapping=ACTION_SET.to_python_code,
                )
            mem_tok_before = (mem.usage.input_tokens, mem.usage.output_tokens) if mem else (0, 0)
            res = agent.run_episode(env, task, seed, args["max_steps"], keep_prompts=args["save_prompts"])
            if mem is not None and phase == "train":
                if seed not in {e.seed for e in mem.episodes()}:
                    # episodic stores raw episodes only; procedural also induces a
                    # workflow after successes and writes a lesson after failures
                    mem.add_episode(res.record, consolidate=(cond == "procedural"))
            mem_in = (mem.usage.input_tokens - mem_tok_before[0]) if mem else 0
            mem_out = (mem.usage.output_tokens - mem_tok_before[1]) if mem else 0
            row = {
                "exp": exp.name,
                "llm": args["llm"],
                "condition": cond,
                "task": task,
                "phase": phase,
                "seed": seed,
                "idx": idx,
                "success": res.success,
                "reward": res.reward,
                "n_steps": res.n_steps,
                "truncated": res.truncated,
                "errors": res.errors,
                "repeated_actions": res.repeated_actions,
                "input_tokens": res.usage.input_tokens,
                "output_tokens": res.usage.output_tokens,
                "llm_calls": res.usage.calls,
                "memory_input_tokens": mem_in,
                "memory_output_tokens": mem_out,
                "memory_chars": res.memory_chars,
                "wall_s": round(res.wall_s, 2),
                "goal": res.record.goal,
                "steps": [asdict(s) for s in res.record.steps],
            }
            if args["save_prompts"]:
                row["prompts"] = res.prompts
            with shard.open("a") as f:
                f.write(json.dumps(row) + "\n")
            n_new += 1
            print(
                f"[{cond:>10} {task:<22}] {phase:<5} seed {seed:<5} "
                f"{'PASS' if res.success else 'fail'}  steps={res.n_steps:<2} "
                f"tok={res.usage.input_tokens}",
                flush=True,
            )
    except Exception:
        return {"condition": cond, "task": task, "ok": False, "error": traceback.format_exc()}
    finally:
        if env is not None:
            env.close()
    return {"condition": cond, "task": task, "ok": True, "episodes": n_new}


def preflight(llm_spec: str) -> None:
    """Fail fast, with a fix, before spawning dozens of browser jobs."""
    problems = []
    try:
        import browsergym.core  # noqa: F401
        import browsergym.miniwob  # noqa: F401
    except ImportError as e:
        problems.append(
            f"{e}. Install BrowserGym into THIS Python ({sys.executable}):\n"
            "      pip install -e ../browsergym/core -e ../browsergym/miniwob && playwright install chromium"
        )
    url = os.environ.get("MINIWOB_URL", "")
    if not url:
        problems.append("MINIWOB_URL is not set (see ../browsergym/miniwob/README.md or `make setup-miniwob`).")
    elif url.startswith("file://") and not Path(url[len("file://"):], "click-test.html").exists():
        problems.append(f"MINIWOB_URL points to {url!r}, but click-test.html is not there.")
    if llm_spec.startswith("anthropic:"):
        key = os.environ.get("ANTHROPIC_API_KEY", "")
        if not key.startswith("sk-ant-"):
            problems.append("ANTHROPIC_API_KEY is missing or not a real key (it should start with 'sk-ant-').")
        try:
            import anthropic  # noqa: F401
        except ImportError:
            problems.append("The anthropic package is missing: pip install anthropic")
    if problems:
        sys.exit("Cannot start:\n  - " + "\n  - ".join(problems))


def main() -> None:
    from agent import CONDITIONS

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exp", required=True, help="experiment name (results/<exp>)")
    ap.add_argument("--llm", required=True, help="anthropic:<model> | openai:<model>@<base_url> | scripted")
    ap.add_argument("--tasks", default="memory12", help="preset name or comma-separated MiniWoB task names")
    ap.add_argument("--conditions", default=",".join(CONDITIONS))
    ap.add_argument("--train", type=int, default=5, help="training seeds per task (memory is built here)")
    ap.add_argument("--test", type=int, default=10, help="held-out test seeds per task (memory frozen)")
    ap.add_argument("--max-steps", type=int, default=15)
    ap.add_argument("--k-episodes", type=int, default=2, help="episodes retrieved for episodic/procedural")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--workers", type=int, default=2, help="parallel (condition, task) jobs, one Chromium each")
    ap.add_argument("--save-prompts", action="store_true", help="store every prompt (large files)")
    ap.add_argument("--results-dir", default=str(HERE / "results"))
    args = ap.parse_args()

    preflight(args.llm)
    miniwob_url = os.environ["MINIWOB_URL"]
    tasks = TASK_PRESETS.get(args.tasks) or [t.strip() for t in args.tasks.split(",") if t.strip()]
    conditions = [c.strip() for c in args.conditions.split(",")]
    for c in conditions:
        if c not in CONDITIONS:
            sys.exit(f"unknown condition {c!r}; choose from {CONDITIONS}")

    exp_dir = Path(args.results_dir) / args.exp
    exp_dir.mkdir(parents=True, exist_ok=True)
    config = dict(vars(args), tasks=tasks, conditions=conditions, test_seed_offset=TEST_SEED_OFFSET)
    cfg_path = exp_dir / "config.json"
    if cfg_path.exists():
        old = json.loads(cfg_path.read_text())
        for key in ("llm", "max_steps", "k_episodes", "temperature"):
            if old.get(key) != config.get(key):
                sys.exit(f"{exp_dir} was run with {key}={old.get(key)!r}; use a new --exp name.")
    cfg_path.write_text(json.dumps(config, indent=2))

    jobs = [
        dict(
            exp_dir=str(exp_dir), condition=c, task=t, llm=args.llm, temperature=args.temperature,
            train=args.train, test=args.test, max_steps=args.max_steps, k_episodes=args.k_episodes,
            save_prompts=args.save_prompts, miniwob_url=miniwob_url,
        )
        for t in tasks
        for c in conditions
    ]
    n_eps = len(jobs) * (args.train + args.test)
    print(f"{len(jobs)} jobs, {n_eps} episodes, llm={args.llm}, workers={args.workers}", flush=True)
    t0 = time.time()
    failures = []
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context("spawn")) as pool:
        futs = [pool.submit(job, j) for j in jobs]
        for f in as_completed(futs):
            r = f.result()
            if not r["ok"]:
                failures.append(r)
                print(f"!! job {r['condition']}/{r['task']} failed:\n{r['error']}", flush=True)
    print(f"done in {time.time() - t0:.0f}s; {len(failures)} failed jobs (re-run the same command to resume)")
    print(f"report: python {HERE / 'report.py'} {exp_dir}")


if __name__ == "__main__":
    main()
