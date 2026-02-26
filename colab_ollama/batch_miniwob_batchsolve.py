#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import textwrap
import traceback
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Iterable

import gymnasium as gym
import browsergym.miniwob  # noqa: F401


ROOT = Path(__file__).resolve().parent
DEFAULT_GOAL_DIR = ROOT / "goal"
DEFAULT_NOGOAL_DIR = ROOT / "nogoal"
DEFAULT_AXTREE_DIR = ROOT / "axtrees"
DEFAULT_GOALTEXT_DIR = ROOT / "goal_texts"
DEFAULT_REPORT = ROOT / "batch_miniwob_report.json"


@dataclass
class UIEl:
    bid: str
    role: str
    name: str
    node_id: str = ""
    editable: bool = False
    disabled: bool = False
    readonly: bool = False

    @property
    def norm_name(self) -> str:
        return norm(self.name)


@dataclass
class SolveResult:
    env_id: str
    safe_name: str
    goal_text: str | None
    goal_cmd_ok: bool
    axtree_cmd_ok: bool
    solved: bool
    reward: float | None
    terminated: bool | None
    truncated: bool | None
    last_action_error: str | None
    actions: list[str]
    failure_reason: str | None
    exception: str | None
    generated_script: str
    axtree_path: str
    goal_path: str


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip()).casefold()


def sanitize_env_id(env_id: str) -> str:
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", env_id)
    return s.replace("/", "_").replace(".", "_")


def run_cmd(args: list[str], *, env: dict[str, str] | None = None, timeout: int = 120) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args,
        text=True,
        capture_output=True,
        env=env,
        timeout=timeout,
    )


def list_miniwob_envs() -> list[str]:
    cp = run_cmd(["obg", "list-miniwob"], timeout=60)
    if cp.returncode != 0:
        raise RuntimeError(f"obg list-miniwob failed: {cp.stderr.strip() or cp.stdout.strip()}")
    envs = [
        line.strip()
        for line in cp.stdout.splitlines()
        if line.strip() and line.strip().startswith("browsergym/miniwob.")
    ]
    # Keep stable order and drop accidental duplicates.
    return list(dict.fromkeys(envs))


def get_goal_via_obg(env_id: str, out_path: Path, *, base_env: dict[str, str]) -> tuple[bool, str | None, str | None]:
    cp = run_cmd(["obg", "goal", "--env", env_id], env=base_env, timeout=180)
    if cp.returncode == 0:
        txt = cp.stdout.strip()
        out_path.write_text(txt + ("\n" if txt and not txt.endswith("\n") else ""), encoding="utf-8")
        return True, txt, None
    err = (cp.stderr or cp.stdout).strip() or f"exit={cp.returncode}"
    out_path.write_text(err + "\n", encoding="utf-8")
    return False, None, err


def get_axtree_via_obg(env_id: str, out_path: Path, *, base_env: dict[str, str]) -> tuple[bool, str | None]:
    cp = run_cmd(["obg", "print-axtree", "--env", env_id, "--out", str(out_path)], env=base_env, timeout=240)
    if cp.returncode == 0:
        return True, None
    err = (cp.stderr or cp.stdout).strip() or f"exit={cp.returncode}"
    if not out_path.exists():
        out_path.write_text(err + "\n", encoding="utf-8")
    return False, err


def get_axtree_from_obs(obs: Any, info: Any) -> dict[str, Any]:
    ax = None
    if isinstance(obs, dict):
        ax = (
            obs.get("axtree_object")
            or obs.get("axtree")
            or obs.get("ax_tree")
            or obs.get("accessibility_tree")
        )
    if ax is None and isinstance(info, dict):
        ax = (
            info.get("axtree_object")
            or info.get("axtree")
            or info.get("ax_tree")
            or info.get("accessibility_tree")
        )
    if not isinstance(ax, dict) or "nodes" not in ax:
        raise RuntimeError("No AXTree in observation/info")
    return ax


def props_to_dict(props: list[dict[str, Any]] | None) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for p in props or []:
        k = p.get("name")
        v = (p.get("value") or {}).get("value") if isinstance(p, dict) else None
        if k is not None:
            out[str(k)] = v
    return out


def extract_ui(axtree: dict[str, Any]) -> list[UIEl]:
    out: list[UIEl] = []
    for n in axtree.get("nodes", []) or []:
        if not isinstance(n, dict) or n.get("ignored", False):
            continue
        role = str(((n.get("role") or {}).get("value")) or "").lower()
        if not role:
            continue
        bid = n.get("browsergym_id")
        if bid is None:
            continue
        props = props_to_dict(n.get("properties"))
        editable_token = props.get("editable")
        editable = editable_token == "plaintext" or bool(props.get("settable", False))
        out.append(
            UIEl(
                bid=str(bid),
                role=role,
                name=str(((n.get("name") or {}).get("value")) or ""),
                node_id=str(n.get("nodeId", "")),
                editable=editable,
                disabled=bool(props.get("disabled", False)),
                readonly=bool(props.get("readonly", False)),
            )
        )
    return out


def quoted_strings(goal: str) -> list[str]:
    return [m.group(1).strip() for m in re.finditer(r"['\"]([^'\"]+)['\"]", goal or "") if m.group(1).strip()]


def choose_click_target(goal: str, ui: list[UIEl]) -> UIEl | None:
    click_roles = {"button", "link", "checkbox", "radio", "menuitem", "tab", "option", "listitem"}
    candidates = [e for e in ui if e.role in click_roles and not e.disabled]
    if not candidates:
        return None

    targets = quoted_strings(goal)
    for t in targets:
        nt = norm(t)
        exact = [e for e in candidates if e.norm_name == nt]
        if exact:
            return exact[0]
        contains = [e for e in candidates if nt in e.norm_name or e.norm_name in nt]
        if contains:
            return contains[0]

    goal_n = norm(goal)
    # common explicit label mentions without quotes
    for e in candidates:
        if e.norm_name and e.norm_name in goal_n:
            return e

    # simple click-button fallback: unique submit/ok/yes/cancel style mentions
    keywords = ["submit", "ok", "okay", "yes", "no", "next", "done", "continue", "search"]
    for kw in keywords:
        if kw in goal_n:
            matches = [e for e in candidates if kw in e.norm_name]
            if matches:
                return matches[0]

    if len(candidates) == 1 and re.search(r"\b(click|press|select|choose)\b", goal_n):
        return candidates[0]

    return None


def choose_fill_target(goal: str, ui: list[UIEl]) -> tuple[UIEl, str] | None:
    inputs = [e for e in ui if e.role in {"textbox", "searchbox", "combobox"} and e.editable and not e.disabled and not e.readonly]
    if not inputs:
        return None

    quoted = quoted_strings(goal)
    if not quoted:
        return None

    goal_n = norm(goal)
    # Prefer non-control strings (not button labels) for text entry.
    common_button_words = {"ok", "okay", "submit", "cancel", "yes", "no", "next", "done"}
    vals = [q for q in quoted if norm(q) not in common_button_words]
    if not vals:
        vals = quoted

    # If the goal looks like typing/entering text and there is one input, use first quoted string.
    if len(inputs) == 1 and re.search(r"\b(type|enter|input|fill|search|write)\b", goal_n):
        return inputs[0], vals[0]

    # Otherwise use first input and first quoted string as a conservative guess.
    if len(inputs) == 1:
        return inputs[0], vals[0]

    return None


def propose_actions(goal: str, ui: list[UIEl]) -> list[str]:
    actions: list[str] = []

    fill_choice = choose_fill_target(goal, ui)
    if fill_choice is not None:
        inp, val = fill_choice
        actions.append(f"fill('{inp.bid}', {val!r})")
        # Often a follow-up click is required.
        click_after = choose_click_target(goal, ui)
        if click_after and click_after.bid != inp.bid:
            actions.append(f"click('{click_after.bid}')")
        else:
            # common submit-ish button fallback after typing
            for kw in ("submit", "ok", "okay", "search", "go", "done", "next"):
                for e in ui:
                    if e.role == "button" and kw in e.norm_name and not e.disabled:
                        actions.append(f"click('{e.bid}')")
                        return dedupe(actions)
        return dedupe(actions)

    click_target = choose_click_target(goal, ui)
    if click_target is not None:
        actions.append(f"click('{click_target.bid}')")
        return actions

    return actions


def dedupe(xs: Iterable[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for x in xs:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def attempt_solve(env_id: str, *, headless: bool | None = None) -> tuple[bool, float | None, bool | None, bool | None, str | None, list[str], str | None, str | None]:
    env = None
    try:
        env = gym.make(env_id) if headless is None else gym.make(env_id, headless=headless)
        obs, info = env.reset()
        if not isinstance(obs, dict):
            return False, None, None, None, None, [], f"Unexpected obs type: {type(obs)}", None
        goal = str(obs.get("goal") or "")
        axtree = get_axtree_from_obs(obs, info)
        ui = extract_ui(axtree)
        actions = propose_actions(goal, ui)
        if not actions:
            return False, None, None, None, None, [], "No heuristic action proposal", None

        reward = None
        terminated = False
        truncated = False
        last_action_error = None
        cur_obs: Any = obs
        cur_info: Any = info
        for action in actions:
            cur_obs, reward, terminated, truncated, cur_info = env.step(action)
            if isinstance(cur_obs, dict):
                err = cur_obs.get("last_action_error")
                if err:
                    last_action_error = str(err)
            if terminated or truncated:
                break

        solved = bool(reward is not None and reward > 0)
        return solved, reward, terminated, truncated, last_action_error, actions, None, goal
    except Exception:
        return False, None, None, None, None, [], "Exception during solve", traceback.format_exc()
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass


def render_solver_program(env_id: str, goal_text: str | None, actions: list[str]) -> str:
    actions_list = ",\n        ".join(repr(a) for a in actions) if actions else ""
    goal_literal = repr(goal_text) if goal_text is not None else "None"
    return textwrap.dedent(
        f'''\
        #!/usr/bin/env python3
        from __future__ import annotations

        import os
        import sys

        import gymnasium as gym
        import browsergym.miniwob  # noqa: F401

        ENV_ID = {env_id!r}
        KNOWN_GOAL = {goal_literal}
        ACTIONS = [
                {actions_list}
        ] if {bool(actions)!r} else []

        def main() -> int:
            env_id = sys.argv[1] if len(sys.argv) > 1 else ENV_ID
            if not os.environ.get("MINIWOB_URL"):
                print('MINIWOB_URL is not set. Example: export MINIWOB_URL="http://127.0.0.1:8000/miniwob/"', file=sys.stderr)
                return 2
            env = gym.make(env_id)
            try:
                obs, info = env.reset()
                if isinstance(obs, dict) and obs.get("goal"):
                    print(f"Goal: {{obs['goal']}}")
                elif KNOWN_GOAL:
                    print(f"Goal (cached): {{KNOWN_GOAL}}")

                if not ACTIONS:
                    print("No cached successful actions for this environment.")
                    return 1

                reward = None
                terminated = False
                truncated = False
                for action in ACTIONS:
                    print(f"Action: {{action}}")
                    obs, reward, terminated, truncated, info = env.step(action)
                    if isinstance(obs, dict) and obs.get("last_action_error"):
                        print(f"last_action_error: {{obs['last_action_error']}}")
                    print(f"After action: reward={{reward}} terminated={{terminated}} truncated={{truncated}}")
                    if terminated or truncated:
                        break

                return 0 if (reward is not None and reward > 0) else 1
            finally:
                env.close()

        if __name__ == "__main__":
            raise SystemExit(main())
        '''
    )


def ensure_dirs(*dirs: Path) -> None:
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)


def main() -> int:
    ap = argparse.ArgumentParser(description="Iterate MiniWoB envs, save goals/axtrees, attempt heuristic solve, and generate per-env solver files.")
    ap.add_argument("--limit", type=int, default=0, help="Optional cap on number of envs to process (0 = all discovered).")
    ap.add_argument("--start", type=int, default=0, help="Start index into the env list.")
    ap.add_argument("--headless", choices=["inherit", "0", "1"], default="inherit", help="Optional PLAYWRIGHT_HEADLESS hint for subprocess calls.")
    ap.add_argument("--only", nargs="*", default=None, help="Optional explicit env ids to process.")
    args = ap.parse_args()

    ensure_dirs(DEFAULT_GOAL_DIR, DEFAULT_NOGOAL_DIR, DEFAULT_AXTREE_DIR, DEFAULT_GOALTEXT_DIR)

    base_env = os.environ.copy()
    solve_headless: bool | None = None
    if args.headless != "inherit":
        solve_headless = (args.headless == "1")
        base_env["PLAYWRIGHT_HEADLESS"] = "true" if solve_headless else "false"

    if not base_env.get("MINIWOB_URL"):
        print('Warning: MINIWOB_URL is not set; obg goal/print-axtree and solves will fail.', file=sys.stderr)

    envs = args.only if args.only else list_miniwob_envs()
    if args.start:
        envs = envs[args.start :]
    if args.limit and args.limit > 0:
        envs = envs[: args.limit]

    print(f"Discovered {len(envs)} environments to process.")

    results: list[SolveResult] = []
    for idx, env_id in enumerate(envs, start=1):
        safe_name = sanitize_env_id(env_id)
        print(f"[{idx}/{len(envs)}] {env_id}")

        goal_path = DEFAULT_GOALTEXT_DIR / f"{safe_name}.txt"
        axtree_path = DEFAULT_AXTREE_DIR / f"{safe_name}.json"

        goal_ok, goal_text, goal_err = get_goal_via_obg(env_id, goal_path, base_env=base_env)
        axtree_ok, axtree_err = get_axtree_via_obg(env_id, axtree_path, base_env=base_env)

        solved, reward, terminated, truncated, last_action_error, actions, failure_reason, exc_text_or_goal = attempt_solve(env_id, headless=solve_headless)

        exception_text = None
        if failure_reason == "Exception during solve":
            exception_text = exc_text_or_goal
            if not goal_text:
                goal_text = None
        elif goal_text is None and isinstance(exc_text_or_goal, str):
            # attempt_solve returns goal in this slot for non-exception path
            goal_text = exc_text_or_goal

        script_text = render_solver_program(env_id, goal_text, actions if solved else [])
        target_dir = DEFAULT_GOAL_DIR if solved else DEFAULT_NOGOAL_DIR
        stale_dir = DEFAULT_NOGOAL_DIR if solved else DEFAULT_GOAL_DIR
        script_path = target_dir / f"{safe_name}__solver.py"
        stale_path = stale_dir / f"{safe_name}__solver.py"
        if stale_path.exists():
            stale_path.unlink()
        script_path.write_text(script_text, encoding="utf-8")
        try:
            script_path.chmod(0o755)
        except Exception:
            pass

        if not solved and failure_reason is None:
            if last_action_error:
                failure_reason = f"last_action_error: {last_action_error}"
            elif not actions:
                failure_reason = "No actions executed"
            elif reward is not None:
                failure_reason = f"reward={reward}"
            elif goal_err:
                failure_reason = f"obg goal failed: {goal_err}"
            elif axtree_err:
                failure_reason = f"obg print-axtree failed: {axtree_err}"

        res = SolveResult(
            env_id=env_id,
            safe_name=safe_name,
            goal_text=goal_text,
            goal_cmd_ok=goal_ok,
            axtree_cmd_ok=axtree_ok,
            solved=solved,
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            last_action_error=last_action_error,
            actions=actions,
            failure_reason=failure_reason,
            exception=exception_text,
            generated_script=str(script_path),
            axtree_path=str(axtree_path),
            goal_path=str(goal_path),
        )
        results.append(res)

    report = {
        "discovered_count": len(results),
        "solved_count": sum(1 for r in results if r.solved),
        "nogoal_count": sum(1 for r in results if not r.solved),
        "results": [asdict(r) for r in results],
    }
    DEFAULT_REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Wrote report: {DEFAULT_REPORT}")
    print(f"Solved: {report['solved_count']} | Not solved: {report['nogoal_count']}")
    print(f"Goal scripts: {DEFAULT_GOAL_DIR}")
    print(f"No-goal scripts: {DEFAULT_NOGOAL_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
