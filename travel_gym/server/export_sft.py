"""
Export recorded travel-gym trajectories as supervised fine-tuning data for a student model.

Each browser step becomes one chat example: the exact prompt the policy saw at that step
(rebuilt with policies.build_prompt from the goal, the notes and errors carried over from the
previous step, the recent actions and the accessibility tree) and the target reply in the
<thinking>/<notes>/<action> format the LLM policies use. Training on this teaches a student to
act from the same observation it will get at inference time.

Filters (behavior cloning keeps only good demonstrations):
  --only-success      runs that booked the optimum (default on)
  --policies          which teachers to include (default: all browser policies)
  --drop-errors       drop steps whose action raised an error (default on)
  --max-axtree-chars  truncate very long pages so examples fit the student's context

Splits are by episode seed, never by step, so no test episode leaks into training.

  python export_sft.py --out ../data/sft --test-frac 0.2
  python export_sft.py --policies scripted claude --keep-failures   # failures too, for DPO/KTO pairing
"""
import argparse
import hashlib
import json
import random
from pathlib import Path

from db import connect
from policies import SYSTEM, StepContext, build_prompt


def reply(thought: str, notes: str, action: str) -> str:
    return f"<thinking>{thought or 'Next step.'}</thinking>\n<notes>{notes}</notes>\n<action>{action}</action>"


def split_of(seed: int, test_frac: float) -> str:
    h = int(hashlib.sha1(str(seed).encode()).hexdigest(), 16) % 1000
    return "test" if h < test_frac * 1000 else "train"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="../data/sft")
    ap.add_argument("--policies", nargs="*", default=None)
    ap.add_argument("--keep-failures", action="store_true")
    ap.add_argument("--keep-errors", action="store_true")
    ap.add_argument("--max-axtree-chars", type=int, default=24000)
    ap.add_argument("--test-frac", type=float, default=0.2)
    ap.add_argument("--recent", type=int, default=12)
    a = ap.parse_args()

    conn = connect()
    runs = conn.execute(
        "SELECT r.id, r.policy, r.model, r.success, r.seed, r.difficulty, e.task "
        "FROM runs r JOIN episodes e ON e.id = r.episode_id WHERE r.status = 'done'").fetchall()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    files = {s: open(out / f"{s}.jsonl", "w") for s in ("train", "test")}
    stats = {"runs": 0, "steps": 0, "skipped_runs": 0, "dropped_error_steps": 0, "train": 0, "test": 0}

    for run_id, policy, model, success, seed, difficulty, task in runs:
        if policy.startswith("mcp"):          # MCP runs have no accessibility tree; export them separately
            continue
        if a.policies and policy not in a.policies:
            continue
        if not success and not a.keep_failures:
            stats["skipped_runs"] += 1
            continue
        goal = json.loads(task)["goal"]
        steps = conn.execute("SELECT t, url, thought, notes, action, error, axtree FROM steps "
                             "WHERE run_id = ? ORDER BY t", (run_id,)).fetchall()
        split = split_of(seed, a.test_frac)
        notes, history, last_error = "", [], ""
        stats["runs"] += 1
        for t, url, thought, new_notes, action, error, axtree in steps:
            ax = axtree if len(axtree) <= a.max_axtree_chars else axtree[: a.max_axtree_chars] + "\n...(truncated)"
            ctx = StepContext(goal=goal, axtree=ax, url=url, t=t, history=list(history),
                              notes=notes, task={}, last_error=last_error)
            if a.keep_errors or not error:
                ex = {
                    "messages": [
                        {"role": "system", "content": SYSTEM},
                        {"role": "user", "content": build_prompt(ctx, recent=a.recent)},
                        {"role": "assistant", "content": reply(thought, new_notes, action)},
                    ],
                    "meta": {"run_id": run_id, "t": t, "policy": policy, "model": model, "seed": seed,
                             "success": bool(success), "action_error": error or None,
                             "difficulty": json.loads(difficulty) if isinstance(difficulty, str) else difficulty},
                }
                files[split].write(json.dumps(ex) + "\n")
                stats[split] += 1
                stats["steps"] += 1
            else:
                stats["dropped_error_steps"] += 1
            history.append((action, error or ""))
            notes, last_error = new_notes or notes, error or ""

    for f in files.values():
        f.close()
    print(json.dumps(stats, indent=2))
    print(f"wrote {out}/train.jsonl and {out}/test.jsonl")


if __name__ == "__main__":
    main()
