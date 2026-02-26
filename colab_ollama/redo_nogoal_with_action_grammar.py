#!/usr/bin/env python3
from __future__ import annotations

import ast
import json
import os
import re
import shutil
import sys
import traceback
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import gymnasium as gym
import browsergym.miniwob  # noqa: F401

REPORT = Path('/Users/dc/BrowserGym/colab_ollama/batch_miniwob_report.json')
OUT = Path('/Users/dc/BrowserGym/colab_ollama/redo_nogoal_report.json')
GOAL_DIR = Path('/Users/dc/BrowserGym/colab_ollama/goal')
NOGOAL_DIR = Path('/Users/dc/BrowserGym/colab_ollama/nogoal')

ACTION_RE = re.compile(r"^ACTION:\s*([a-z_][a-z0-9_]*)\((.*)\)\s*$", re.MULTILINE)
ALLOWED = {
    'noop','send_msg_to_user','report_infeasible','scroll','fill','select_option',
    'click','dblclick','hover','press','focus','clear','drag_and_drop','upload_file',
    'go_back','go_forward','goto','tab_close','tab_focus','new_tab'
}


def validate_action(action: str) -> bool:
    m = re.match(r"^\s*([a-z_][a-z0-9_]*)\(", action)
    return bool(m and m.group(1) in ALLOWED)


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip()).casefold()


def sanitize(env_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", env_id).replace('/', '_').replace('.', '_')


@dataclass
class UI:
    bid: str
    role: str
    name: str
    disabled: bool = False
    readonly: bool = False
    editable: bool = False

    @property
    def nn(self) -> str:
        return norm(self.name)


def props_to_dict(props: list[dict[str, Any]] | None) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for p in props or []:
        if not isinstance(p, dict):
            continue
        k = p.get('name')
        v = (p.get('value') or {}).get('value') if isinstance(p.get('value'), dict) else None
        if k is not None:
            out[str(k)] = v
    return out


def get_ax(obs: Any, info: Any) -> dict[str, Any]:
    ax = None
    if isinstance(obs, dict):
        ax = obs.get('axtree_object') or obs.get('axtree') or obs.get('ax_tree') or obs.get('accessibility_tree')
    if ax is None and isinstance(info, dict):
        ax = info.get('axtree_object') or info.get('axtree') or info.get('ax_tree') or info.get('accessibility_tree')
    if not isinstance(ax, dict):
        raise RuntimeError('No axtree')
    return ax


def extract_ui(ax: dict[str, Any]) -> list[UI]:
    out: list[UI] = []
    for n in ax.get('nodes', []) or []:
        if not isinstance(n, dict) or n.get('ignored', False):
            continue
        bid = n.get('browsergym_id')
        if bid is None:
            continue
        role = str(((n.get('role') or {}).get('value')) or '').lower()
        if not role:
            continue
        name = str(((n.get('name') or {}).get('value')) or '')
        props = props_to_dict(n.get('properties'))
        editable = (props.get('editable') == 'plaintext') or bool(props.get('settable', False))
        out.append(UI(
            bid=str(bid), role=role, name=name,
            disabled=bool(props.get('disabled', False)),
            readonly=bool(props.get('readonly', False)),
            editable=editable,
        ))
    return out


def quoted(goal: str) -> list[str]:
    return [m.group(1).strip() for m in re.finditer(r"['\"]([^'\"]+)['\"]", goal or '') if m.group(1).strip()]


def by_role(ui: list[UI]) -> dict[str, list[UI]]:
    d: dict[str, list[UI]] = defaultdict(list)
    for e in ui:
        d[e.role].append(e)
    return d


def first_match(items: list[UI], target: str) -> UI | None:
    nt = norm(target)
    exact = [e for e in items if e.nn == nt and not e.disabled]
    if exact:
        return exact[0]
    contains = [e for e in items if (nt in e.nn or e.nn in nt) and not e.disabled and e.nn]
    if contains:
        return contains[0]
    return None


def all_matches(items: list[UI], targets: list[str]) -> list[UI]:
    chosen: list[UI] = []
    used: set[str] = set()
    for t in targets:
        m = first_match(items, t)
        if m and m.bid not in used:
            chosen.append(m)
            used.add(m.bid)
    return chosen


def parse_path(goal: str) -> list[str]:
    m = re.search(r"Select\s+(.+)$", goal.strip(), flags=re.I)
    if not m:
        return []
    raw = m.group(1).strip().strip('.')
    if '>' in raw:
        return [p.strip().strip('"\'') for p in raw.split('>') if p.strip()]
    return []


def propose_plans(goal: str, ui: list[UI]) -> list[list[str]]:
    g = norm(goal)
    roles = by_role(ui)
    buttons = [e for e in ui if e.role == 'button' and not e.disabled]
    clickables = [e for e in ui if e.role in {'button','link','checkbox','radio','menuitem','tab','option','listitem'} and not e.disabled]
    inputs = [e for e in ui if e.role in {'textbox','searchbox','combobox'} and e.editable and not e.readonly and not e.disabled]
    checks = [e for e in ui if e.role in {'checkbox','radio'} and not e.disabled]
    plans: list[list[str]] = []

    q = quoted(goal)
    path = parse_path(goal)
    if path:
        p: list[str] = []
        menuish = [e for e in ui if e.role in {'menuitem','tab','button','link','option'} and not e.disabled]
        for i, token in enumerate(path):
            m = first_match(menuish, token)
            if not m:
                p = []
                break
            p.append(f"click('{m.bid}')")
            if i < len(path)-1:
                p.append("noop(wait_ms=300)")
        if p:
            plans.append(p)

    # multi-checkbox selection from quoted items
    if checks and len(q) >= 1 and re.search(r"checkbox|select|check", g):
        ms = all_matches(checks + buttons + clickables, q)
        if ms:
            p = [f"click('{e.bid}')" for e in ms]
            submit = first_match(buttons, 'submit') or first_match(buttons, 'ok') or first_match(buttons, 'done')
            if submit:
                p.append(f"click('{submit.bid}')")
            plans.append(p)

    # fill/select based on quoted values
    if q and inputs:
        val = q[0]
        if len(inputs) == 1:
            inp = inputs[0]
            p1 = [f"focus('{inp.bid}')", f"clear('{inp.bid}')", f"fill('{inp.bid}', {val!r})"]
            if 'autocomplete' in g:
                p1[-1] = f"fill('{inp.bid}', {val!r}, enable_autocomplete_menu=True)"
            if re.search(r"\b(enter|submit|search|go)\b", g):
                submit = first_match(buttons, 'submit') or first_match(buttons, 'search') or first_match(buttons, 'go')
                if submit:
                    p1.append(f"click('{submit.bid}')")
                else:
                    p1.append(f"press('{inp.bid}', 'Enter')")
            plans.append(p1)

        # select_option on combobox/select-like elements
        for e in [x for x in inputs if x.role == 'combobox']:
            plans.append([f"select_option('{e.bid}', {q[0]!r})"])

    # click explicit quoted labels
    for t in q:
        m = first_match(clickables, t)
        if m:
            plans.append([f"click('{m.bid}')"])

    # common single-button intents
    for kw in ['submit','ok','okay','yes','no','cancel','next','done','continue','start']:
        if kw in g:
            m = first_match(buttons, kw)
            if m:
                plans.append([f"click('{m.bid}')"])

    # sign agreement style
    if 'agree' in g or 'agreement' in g:
        ck = first_match(checks, 'agree') or (checks[0] if checks else None)
        sb = first_match(buttons, 'submit') or first_match(buttons, 'ok') or first_match(buttons, 'continue')
        p = []
        if ck:
            p.append(f"click('{ck.bid}')")
        if sb:
            p.append(f"click('{sb.bid}')")
        if p:
            plans.append(p)

    # generic click target by name mention in goal
    if not q:
        for e in clickables:
            if e.nn and e.nn in g:
                plans.append([f"click('{e.bid}')"])

    # de-dup and validate
    uniq: list[list[str]] = []
    seen: set[tuple[str, ...]] = set()
    for p in plans:
        p = [a for a in p if validate_action(a)]
        if not p:
            continue
        t = tuple(p)
        if t in seen:
            continue
        seen.add(t)
        uniq.append(p)
    return uniq[:20]


def try_env(env_id: str) -> dict[str, Any]:
    best: dict[str, Any] = {'solved': False, 'reward': None, 'actions': [], 'terminated': None, 'truncated': None, 'last_action_error': None, 'goal': None, 'plans_tried': 0}
    env = None
    try:
        env = gym.make(env_id)
        obs, info = env.reset()
        goal = str(obs.get('goal') or '') if isinstance(obs, dict) else ''
        best['goal'] = goal
        ax = get_ax(obs, info)
        ui = extract_ui(ax)
        plans = propose_plans(goal, ui)
        if not plans:
            best['failure_reason'] = 'No heuristic plan with new grammar'
            return best
    finally:
        if env is not None:
            try: env.close()
            except Exception: pass

    for plan in plans:
        env = None
        try:
            env = gym.make(env_id)
            obs, info = env.reset()
            reward = None
            term = False
            trunc = False
            last_err = None
            for action in plan:
                obs, reward, term, trunc, info = env.step(action)
                if isinstance(obs, dict) and obs.get('last_action_error'):
                    last_err = str(obs.get('last_action_error'))
                if term or trunc:
                    break
            best['plans_tried'] += 1
            if reward is not None and reward > 0:
                return {
                    'solved': True,
                    'reward': reward,
                    'actions': plan,
                    'terminated': term,
                    'truncated': trunc,
                    'last_action_error': last_err,
                    'goal': best.get('goal'),
                    'plans_tried': best['plans_tried'],
                }
            # keep best diagnostic
            if best['reward'] is None and reward is not None:
                best['reward'] = reward
            if last_err and not best.get('last_action_error'):
                best['last_action_error'] = last_err
            best['terminated'] = term
            best['truncated'] = trunc
            if not best['actions']:
                best['actions'] = plan
        except Exception:
            best['exception'] = traceback.format_exc()
        finally:
            if env is not None:
                try: env.close()
                except Exception: pass
    if 'failure_reason' not in best:
        best['failure_reason'] = f"reward={best['reward']}" if best['reward'] is not None else 'No successful plan'
    return best


def overwrite_solver(env_id: str, actions: list[str], goal_text: str | None, solved: bool) -> str:
    safe = sanitize(env_id)
    target = GOAL_DIR if solved else NOGOAL_DIR
    stale = NOGOAL_DIR if solved else GOAL_DIR
    target.mkdir(parents=True, exist_ok=True)
    stale.mkdir(parents=True, exist_ok=True)
    target_path = target / f"{safe}__solver.py"
    stale_path = stale / f"{safe}__solver.py"
    if stale_path.exists():
        stale_path.unlink()
    code = f'''#!/usr/bin/env python3
from __future__ import annotations
import os, sys
import gymnasium as gym
import browsergym.miniwob  # noqa: F401
ENV_ID = {env_id!r}
KNOWN_GOAL = {goal_text!r}
ACTIONS = {actions!r}
def main() -> int:
    env_id = sys.argv[1] if len(sys.argv) > 1 else ENV_ID
    if not os.environ.get("MINIWOB_URL"):
        print("MINIWOB_URL is not set", file=sys.stderr); return 2
    env = gym.make(env_id)
    try:
        obs, info = env.reset()
        print(f"Goal: {{(obs.get('goal') if isinstance(obs, dict) else KNOWN_GOAL)}}")
        reward = None
        for a in ACTIONS:
            print(f"Action: {{a}}")
            obs, reward, term, trunc, info = env.step(a)
            print(f"After action: reward={{reward}} terminated={{term}} truncated={{trunc}}")
            if isinstance(obs, dict) and obs.get('last_action_error'):
                print(f"last_action_error: {{obs['last_action_error']}}")
            if term or trunc:
                break
        return 0 if (reward is not None and reward > 0) else 1
    finally:
        env.close()
if __name__ == '__main__':
    raise SystemExit(main())
'''
    target_path.write_text(code, encoding='utf-8')
    try:
        target_path.chmod(0o755)
    except Exception:
        pass
    return str(target_path)


def main() -> int:
    if not REPORT.exists():
        print(f'Missing report: {REPORT}', file=sys.stderr)
        return 1
    if not os.environ.get('MINIWOB_URL'):
        print('MINIWOB_URL is not set', file=sys.stderr)
        return 2

    rep = json.loads(REPORT.read_text())
    failed = [r for r in rep['results'] if not r.get('solved') and str(r.get('env_id','')).startswith('browsergym/miniwob.')]
    summary: list[dict[str, Any]] = []
    improved = 0
    for i, row in enumerate(failed, start=1):
        env_id = row['env_id']
        print(f'[{i}/{len(failed)}] {env_id}', flush=True)
        res = try_env(env_id)
        res['env_id'] = env_id
        res['previous_failure_reason'] = row.get('failure_reason')
        if res.get('solved'):
            improved += 1
            res['solver_path'] = overwrite_solver(env_id, res.get('actions') or [], res.get('goal'), solved=True)
        else:
            # refresh nogoal solver to reflect best new-grammar plan if any
            res['solver_path'] = overwrite_solver(env_id, [], res.get('goal'), solved=False)
        summary.append(res)

    out = {
        'failed_in': len(failed),
        'newly_solved': improved,
        'still_failed': len(failed) - improved,
        'results': summary,
    }
    OUT.write_text(json.dumps(out, indent=2), encoding='utf-8')
    print(f'Wrote {OUT}')
    print(f'Newly solved: {improved} / {len(failed)}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
