# src/obg/cli.py
import json
import shutil
from pathlib import Path
import typer
import cv2
import subprocess
import base64
import browsergym
import browsergym.miniwob  # MUST happen before gym.make
from collections import deque
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

import os, re, sys, subprocess
import gymnasium as gym


app = typer.Typer()



@app.command("goal")
def print_goal(
    env_id: str = typer.Option(
        "browsergym/miniwob.click-menu",
        "--env",
        help="Gymnasium env id (default: click-menu)",
    ),
    show_object: bool = typer.Option(
        False,
        "--show-object/--no-show-object",
        help="Also print obs['goal_object'] if present",
    ),
    show_task_info: bool = typer.Option(
        False,
        "--show-task-info/--no-show-task-info",
        help="Also print info['task_info'] if present",
    ),
):
    """Reset an env and print its goal text."""
    import gymnasium as gym
    import browsergym.miniwob  # registers envs
    import json

    if env_id.endswith("-v0"):
        env_id = env_id[:-3]

    env = gym.make(env_id)
    try:
        obs, info = env.reset()
        if not isinstance(obs, dict):
            raise typer.BadParameter(f"Unexpected obs type: {type(obs)}")

        goal = obs.get("goal")
        if goal is None:
            raise typer.BadParameter(f"No 'goal' in obs. keys={list(obs.keys())}")

        typer.echo(goal)

        if show_object and "goal_object" in obs:
            typer.echo("\n--- goal_object ---")
            typer.echo(json.dumps(obs["goal_object"], indent=2, ensure_ascii=False))

        if show_task_info and isinstance(info, dict) and "task_info" in info:
            typer.echo("\n--- task_info ---")
            typer.echo(json.dumps(info["task_info"], indent=2, ensure_ascii=False))

    finally:
        env.close()

@app.command("list-actions")
def list_actions(
    env_id: str = typer.Option("browsergym/miniwob.click-menu", "--env"),
    debug: bool = typer.Option(False, "--debug/--no-debug"),
    only_clickables: bool = typer.Option(
        True,
        "--only-clickables/--all-bids",
        help="If true, list only likely-clickable nodes (button/link/menuitem/etc).",
    ),
    max_bids: int = typer.Option(300, "--max-bids", help="Cap BID output"),
):
    """
    1) Print whether we can access env.page (Playwright Page) from the env.
    2) List BIDs from AXTree (optionally only clickables).
    3) Probe which high-level action verbs are accepted by the env parser.
    """
    import re
    import gymnasium as gym
    import browsergym.miniwob  # registers envs

    # --- candidates for action probing (lowercase high-level actions) ---
    CANDIDATES = {
        "click": ["click('0')", "click('1')", "click('999999')"],
        "dblclick": ["dblclick('0')"],
        "hover": ["hover('0')"],
        "fill": ["fill('0','x')"],
        "clear": ["clear('0')"],
        "focus": ["focus('0')"],
        "press": ["press('0','Enter')"],
        "scroll": ["scroll(0, 200)", "scroll(0, -200)"],
        "select_option": ["select_option('0','x')"],
        "drag_and_drop": ["drag_and_drop('0','1')"],
        "goto": ["goto('https://example.com')"],
        "go_back": ["go_back()"],
        "go_forward": ["go_forward()"],
        "new_tab": ["new_tab()"],
        "tab_close": ["tab_close()"],
        "tab_focus": ["tab_focus(0)"],
        "noop": ["noop()", "noop(10)"],
        "send_msg_to_user": ["send_msg_to_user('hi')"],
        "report_infeasible": ["report_infeasible('nope')"],
        "upload_file": ["upload_file('0','/tmp/x')"],
    }

    UNKNOWN_PAT = re.compile(
        r"(unknown action|unrecognized|invalid action|cannot parse|parse error|unsupported|invalid action type)",
        re.IGNORECASE,
    )

    def looks_unknown(msg: str) -> bool:
        return bool(UNKNOWN_PAT.search((msg or "").strip()))

    def get_axtree(obs, info):
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
        return ax if isinstance(ax, dict) else None

    def role_str(n) -> str:
        return str((n.get("role") or {}).get("value", "")).strip().lower()

    CLICKLIKE_ROLES = {"button", "link", "menuitem", "checkbox", "radio", "option", "tab", "textbox", "combobox"}

    def extract_bids_from_axtree(ax) -> list[tuple[str, str, str]]:
        """
        Returns list of (bid, role, name). If only_clickables=True, filter by role.
        """
        out = []
        for n in (ax.get("nodes", []) or []):
            if n.get("ignored", False):
                continue
            bid = n.get("browsergym_id")
            if bid is None:
                continue
            r = role_str(n)
            if only_clickables and r not in CLICKLIKE_ROLES:
                continue
            name = str(((n.get("name") or {}).get("value")) or "")
            out.append((str(bid), r or "", name))
        # stable sort for readability
        out.sort(key=lambda t: (t[1], t[2], t[0]))
        return out

    def get_page_accessible(env) -> bool:
        """
        True if we can reach a Playwright Page object from common locations.
        (Does not guarantee it's alive, but usually enough for hover/evaluate.)
        """
        candidates = [
            getattr(env, "page", None),
            getattr(env, "_page", None),
            getattr(getattr(env, "unwrapped", env), "page", None),
            getattr(getattr(env, "unwrapped", env), "_page", None),
            getattr(getattr(env, "task", None), "page", None),
        ]
        for p in candidates:
            if p is None:
                continue
            # best-effort sanity check
            if hasattr(p, "url") and (hasattr(p, "evaluate") or hasattr(p, "eval_on_selector")):
                return True
        return False

    if env_id.endswith("-v0"):
        env_id = env_id[:-3]

    env = gym.make(env_id)
    try:
        obs, info = env.reset()
        if not isinstance(obs, dict):
            raise typer.BadParameter(f"Unexpected obs type: {type(obs)}")

        page_ok = get_page_accessible(env)
        typer.echo(f"Env: {env_id}")
        typer.echo(f"Playwright page accessible from env: {page_ok}\n")

        ax = get_axtree(obs, info)
        if ax is None:
            typer.echo("⚠️ No AXTree found in obs/info; cannot list BIDs.\n")
        else:
            bids = extract_bids_from_axtree(ax)
            typer.echo(f"BIDs in AXTree ({'clickables' if only_clickables else 'all'}): {len(bids)}")
            typer.echo(f"{'bid':>6}  {'role':<10}  name")
            typer.echo("-" * 90)
            for i, (bid, r, name) in enumerate(bids[:max_bids]):
                typer.echo(f"{bid:>6}  {r:<10}  {name}")
            if len(bids) > max_bids:
                typer.echo(f"... ({len(bids) - max_bids} more)")
            typer.echo("")

        # --- probe verbs ---
        recognized, rejected = [], []

        for verb, examples in CANDIDATES.items():
            ok = False
            for a in examples:
                threw = False
                exc_msg = ""
                try:
                    obs2, reward, term, trunc, info2 = env.step(a)
                except Exception as e:
                    threw = True
                    exc_msg = str(e)
                    obs2 = obs

                last_action = obs2.get("last_action") if isinstance(obs2, dict) else None
                last_err = obs2.get("last_action_error") if isinstance(obs2, dict) else None

                # recognition heuristic:
                # - if step didn't throw -> accepted
                # - if it threw but not "unknown/parse" -> likely accepted but args wrong
                # - if last_action echoes our string -> accepted
                if (not threw) or (threw and not looks_unknown(exc_msg)) or (last_action == a):
                    ok = True

                if debug:
                    typer.echo(
                        f"[{verb}] try={a!r} threw={threw} exc={exc_msg[:120]!r} "
                        f"last_action={last_action!r} last_err={last_err!r}"
                    )

                if ok:
                    break

            (recognized if ok else rejected).append(verb)

        typer.echo("Recognized verbs (probe):")
        for v in recognized:
            typer.echo(f" - {v}")

        typer.echo("\nRejected / likely unsupported:")
        for v in rejected:
            typer.echo(f" - {v}")

    finally:
        env.close()
@app.command("click-all-buttons")
def click_all_buttons(
    env_id: str = typer.Option("browsergym/miniwob.buy-ticket", "--env"),
    out_dir: Path = typer.Option(Path("screenshots"), "--out-dir"),
    wait_ms: int = typer.Option(200, "--wait-ms", help="noop() wait after each click (ms)"),
    max_buttons: int = typer.Option(50, "--max-buttons", help="Safety cap"),
):
    """
    For every AXTree node that is a button:
      - save screenshot BEFORE env.step():  before_<env>_click_<bid>.png
      - env.step("click('<bid>')")  (BrowserGym HighLevelActionSet)
      - optionally env.step("noop(<wait_ms>)")
      - save screenshot AFTER:       after_<env>_click_<bid>.png
    """
    import re
    import gymnasium as gym
    import browsergym.miniwob  # registers envs
    import numpy as np
    import cv2

    def safe_name(s: str) -> str:
        return re.sub(r"[^A-Za-z0-9._-]+", "_", s).strip("_")

    def get_axtree(obs, info):
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
        return ax if isinstance(ax, dict) else None

    def role_str(n) -> str:
        return str((n.get("role") or {}).get("value", "")).strip().lower()

    def chrome_role_int(n):
        v = (n.get("chromeRole") or {}).get("value", None)
        try:
            return int(v) if v is not None else None
        except Exception:
            return None

    def is_button_node(n) -> bool:
        # robust: role contains "button" OR chromeRole == 9 (what you saw earlier)
        return ("button" in role_str(n)) or (chrome_role_int(n) == 9)

    def save_obs_screenshot(obs, path: Path) -> bool:
        shot = obs.get("screenshot", None) if isinstance(obs, dict) else None
        if shot is None or not isinstance(shot, np.ndarray):
            return False
        img = shot
        if img.dtype != np.uint8:
            img = np.clip(img, 0, 255).astype(np.uint8)
        if img.ndim == 3 and img.shape[2] == 3:
            bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        elif img.ndim == 3 and img.shape[2] == 4:
            bgr = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
        else:
            bgr = img
        path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(path), bgr)
        return True

    # Friendly: drop -v0 if present
    if env_id.endswith("-v0"):
        env_id = env_id[:-3]

    safe_env = safe_name(env_id)
    out_dir.mkdir(parents=True, exist_ok=True)

    env = gym.make(env_id)
    try:
        obs, info = env.reset()
        if not isinstance(obs, dict):
            raise typer.BadParameter(f"Unexpected obs type: {type(obs)}")

        ax = get_axtree(obs, info)
        if ax is None:
            raise typer.BadParameter(f"No AXTree found. obs keys={list(obs.keys())}")

        # collect unique button bids
        bids = []
        seen = set()
        for n in (ax.get("nodes", []) or []):
            if n.get("ignored", False):
                continue
            if not is_button_node(n):
                continue
            bid = n.get("browsergym_id")
            if bid is None:
                continue
            bid = str(bid)
            if bid in seen:
                continue
            seen.add(bid)
            name = str(((n.get("name") or {}).get("value")) or "")
            bids.append((bid, name))

        if not bids:
            typer.echo("No buttons found in AXTree.")
            return

        if len(bids) > max_buttons:
            bids = bids[:max_buttons]
            typer.echo(f"⚠️ Capped buttons to first {max_buttons}.")

        typer.echo(f"Found {len(bids)} button(s). Clicking each...\n")

        for bid, name in bids:
            before_path = out_dir / f"before_{safe_env}_click_{bid}.png"
            after_path = out_dir / f"after_{safe_env}_click_{bid}.png"

            ok_before = save_obs_screenshot(obs, before_path)
            if not ok_before:
                typer.echo(f"⚠️ Could not save BEFORE screenshot (missing obs['screenshot'] ndarray). bid={bid}")

            action = f"click('{bid}')"
            typer.echo(f"CLICK {bid}  name={name!r}  action={action}")

            obs2, reward, terminated, truncated, info2 = env.step(action)

            # optional settle time for UI updates to appear in next screenshot
            if wait_ms and (not terminated) and (not truncated):
                obs2, _r2, terminated, truncated, info2 = env.step(f"noop({float(wait_ms)})")

            ok_after = save_obs_screenshot(obs2, after_path)
            if not ok_after:
                typer.echo(f"⚠️ Could not save AFTER screenshot. bid={bid}")

            last_err = obs2.get("last_action_error") if isinstance(obs2, dict) else None
            if last_err:
                typer.echo(f"  last_action_error: {last_err}")

            typer.echo(f"  reward={reward} terminated={terminated} truncated={truncated}")
            typer.echo(f"  saved: {before_path}  {after_path}\n")

            obs, info = obs2, info2

            if terminated or truncated:
                typer.echo("Episode ended; stopping further clicks.")
                break

    finally:
        env.close()
#obg list-roles
#obg list-roles --env browsergym/miniwob.buy-ticket
#obg list-roles --per-role 25
#obg list-roles --per-role -1 --top 999   # dump everything (be careful)
@app.command("list-roles")
def list_roles(
    env_id: str = typer.Option(
        "browsergym/miniwob.click-menu",
        "--env",
        help="Gymnasium env id",
    ),
    top: int = typer.Option(30, "--top", help="How many top roles to show"),
    per_role: int = typer.Option(
        10,
        "--per-role",
        help="How many nodes to print per role (0 = none, -1 = all)",
    ),
    include_ignored: bool = typer.Option(
        False,
        "--include-ignored/--no-include-ignored",
        help="Include ignored AX nodes",
    ),
):
    """Print role/chromeRole frequency + grouped node tables from AXTree."""
    import gymnasium as gym
    import browsergym.miniwob  # registers envs
    from collections import Counter, defaultdict

    def get_axtree(obs, info):
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
            raise typer.BadParameter(
                f"No AXTree found for {env_id}. "
                f"obs keys={list(obs.keys()) if isinstance(obs, dict) else type(obs)}"
            )
        return ax

    def props_to_dict(props):
        out = {}
        for p in props or []:
            k = p.get("name")
            v = (p.get("value") or {}).get("value")
            if k is not None:
                out[str(k)] = v
        return out

    def role_str(n) -> str:
        return str((n.get("role") or {}).get("value", "")).strip().lower() or "<empty>"

    def chrome_role_int(n):
        v = (n.get("chromeRole") or {}).get("value", None)
        try:
            return int(v) if v is not None else None
        except Exception:
            return None

    if env_id.endswith("-v0"):
        env_id = env_id[:-3]

    env = gym.make(env_id)
    try:
        obs, info = env.reset()
        if not isinstance(obs, dict):
            raise typer.BadParameter(f"Unexpected obs type: {type(obs)}")

        ax = get_axtree(obs, info)
        nodes = ax.get("nodes", []) or []

        rc = Counter()
        cc = Counter()
        by_role = defaultdict(list)

        for n in nodes:
            if (not include_ignored) and n.get("ignored", False):
                continue

            r = role_str(n)
            rc[r] += 1

            cr = chrome_role_int(n)
            if cr is not None:
                cc[cr] += 1

            bid = n.get("browsergym_id")
            node_id = n.get("nodeId")
            name = str(((n.get("name") or {}).get("value")) or "")
            props = props_to_dict(n.get("properties", []))

            by_role[r].append(
                {
                    "nodeId": "" if node_id is None else str(node_id),
                    "bid": "" if bid is None else str(bid),
                    "focusable": props.get("focusable", None),
                    "invalid": props.get("invalid", None),
                    "chromeRole": "" if cr is None else str(cr),
                    "name": name,
                }
            )

        typer.echo(f"Env: {env_id}")
        typer.echo(f"Nodes: {len(nodes)} (include_ignored={include_ignored})\n")

        typer.echo("Top role.value strings:")
        for k, v in rc.most_common(top):
            typer.echo(f"  {k:<24} {v}")
        typer.echo("")

        typer.echo("Top chromeRole numeric values:")
        for k, v in cc.most_common(top):
            typer.echo(f"  {k:<24} {v}")
        typer.echo("")

        if per_role == 0:
            return

        # Print grouped tables in order of most common roles
        for role_name, _count in rc.most_common(top):
            rows = by_role.get(role_name) or []
            if not rows:
                continue

            typer.echo(f"=== role: {role_name} (n={len(rows)}) ===")
            typer.echo(
                f"{'nodeId':>6}  {'bid':>6}  {'focusable':>9}  {'invalid':>7}  {'chrome':>6}  name"
            )
            typer.echo("-" * 120)

            limit = len(rows) if per_role < 0 else min(per_role, len(rows))
            for i in range(limit):
                s = rows[i]
                typer.echo(
                    f"{s['nodeId']:>6}  {s['bid']:>6}  "
                    f"{str(s['focusable']):>9}  {str(s['invalid']):>7}  "
                    f"{s['chromeRole']:>6}  {s['name']}"
                )

            if limit < len(rows):
                typer.echo(f"... ({len(rows) - limit} more)")
            typer.echo("")

    finally:
        env.close()

@app.command("open-webkit")
def open_webkit(
    env_id: str = typer.Option(
        "browsergym/miniwob.click-menu",
        "--env",
        help="BrowserGym env id (e.g. browsergym/miniwob.click-menu)",
    ),
    url: str | None = typer.Option(
        None,
        "--url",
        help="Optional explicit URL. If set, ignores MINIWOB_URL/env mapping.",
    ),
):
    """
    Open a headed WebKit browser at MINIWOB_URL/<task>.html and detach.
    The command exits, but the browser window stays open.
    """
    import os
    import re
    import sys
    import subprocess

    def normalize_task(env_id: str) -> str:
        # env_id examples:
        # - browsergym/miniwob.click-menu
        # - miniwob.click-menu
        # - click-menu
        s = env_id.strip()
        s = s[:-3] if s.endswith("-v0") else s
        # keep only the "click-menu" part if it contains "miniwob."
        m = re.search(r"miniwob\.([A-Za-z0-9_-]+)$", s)
        if m:
            return m.group(1)
        # otherwise if it contains '/', take last chunk
        if "/" in s:
            s = s.split("/")[-1]
        # if it still contains '.', take last part
        if "." in s:
            s = s.split(".")[-1]
        return s

    if url is None:
        base = os.environ.get("MINIWOB_URL", "").strip()
        if not base:
            raise typer.BadParameter(
                'MINIWOB_URL is not set. Example:\n'
                '  export MINIWOB_URL="http://127.0.0.1:8000/miniwob/"'
            )
        if not base.endswith("/"):
            base += "/"
        task = normalize_task(env_id)
        if task.endswith(".html"):
            full_url = base + task
        else:
            full_url = base + task + ".html"
    else:
        full_url = url

    # Spawn a detached python process that keeps the browser alive.
    # It will run until you close the browser window manually.
    child_code = r"""
from playwright.sync_api import sync_playwright
import sys, time

url = sys.argv[1]
with sync_playwright() as p:
    browser = p.webkit.launch(headless=False)
    page = browser.new_page(viewport={"width": 1400, "height": 900})
    page.goto(url)
    # Keep alive forever; user closes window to end.
    while True:
        time.sleep(3600)
"""

    p = subprocess.Popen(
        [sys.executable, "-c", child_code, full_url],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,   # detach from this CLI session
    )

    typer.echo(f"✅ Opened WebKit (detached pid={p.pid}) -> {full_url}")
    typer.echo("Close the browser window to stop it (or kill the pid).")

@app.command("list-miniwob")
def list_miniwob():
    """List all registered BrowserGym MiniWoB Gymnasium environment IDs."""
    import gymnasium as gym
    import browsergym.miniwob  # registers envs

    ids = sorted(
        spec.id for spec in gym.envs.registry.values()
        if "miniwob" in spec.id.lower()
    )
    for eid in ids:
        print(eid)
    print(f"\nTotal: {len(ids)}")


@app.command("print-axtree")
def print_axtree(
    env_id: str = typer.Option(
        "browsergym/miniwob.click-menu",
        "--env",
        help="Gymnasium env id (default: click-menu)",
    ),
    out: Path | None = typer.Option(
        None,
        "--out",
        help="Optional path to write AXTree JSON",
    ),
    head: int = typer.Option(
        0,
        "--head",
        help="If >0, print only first N nodes (still valid JSON)",
    ),
):

    env = gym.make(env_id)
    try:
        obs, info = env.reset()

        axtree = None
        if isinstance(obs, dict):
            axtree = (
                obs.get("axtree_object")          # BrowserGym key (your case)
                or obs.get("axtree")
                or obs.get("ax_tree")
                or obs.get("accessibility_tree")
            )
        if axtree is None and isinstance(info, dict):
            axtree = (
                info.get("axtree_object")
                or info.get("axtree")
                or info.get("ax_tree")
                or info.get("accessibility_tree")
            )

        if axtree is None:
            raise typer.BadParameter(
                f"No axtree found for {env_id}. "
                f"obs keys={list(obs.keys()) if isinstance(obs, dict) else type(obs)}; "
                f"info keys={list(info.keys()) if isinstance(info, dict) else type(info)}"
            )

        if head and isinstance(axtree, dict) and isinstance(axtree.get("nodes"), list):
            axtree = dict(axtree)
            axtree["nodes"] = axtree["nodes"][:head]

        text = json.dumps(axtree, indent=2, ensure_ascii=False)

        if out is not None:
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(text, encoding="utf-8")
            typer.echo(f"✅ Wrote AXTree to {out}")
        else:
            typer.echo(text)
    finally:
        env.close()


@app.command("save-screenshot")
def save_screenshot(
    env_id: str = typer.Option(
        "browsergym/miniwob.click-menu",
        "--env",
        help="Gymnasium env id (default: click-menu)",
    ),
    out: Path | None = typer.Option(
        None,
        "--out",
        help="Optional output PNG path. If omitted: screenshots/<env>.png",
    ),
    open_file: bool = typer.Option(
        False,
        "--open/--no-open",
        help="Open the image after saving (macOS: open)",
    ),
):

    # Friendly: drop -v0 if user passes it
    if env_id.endswith("-v0"):
        env_id = env_id[:-3]

    # Default output path: screenshots/<env_id>.png (sanitized)
    if out is None:
        safe = re.sub(r"[^A-Za-z0-9._-]+", "_", env_id).strip("_")
        out = Path("screenshots") / f"{safe}.png"

    env = gym.make(env_id)
    try:
        obs, info = env.reset()

        if not isinstance(obs, dict) or "screenshot" not in obs:
            raise typer.BadParameter(
                f"No screenshot found in obs for {env_id}. "
                f"obs keys={list(obs.keys()) if isinstance(obs, dict) else type(obs)}"
            )

        shot = obs["screenshot"]
        png_bytes: bytes

        if isinstance(shot, np.ndarray):
            img = shot
            if img.dtype != np.uint8:
                img = np.clip(img, 0, 255).astype(np.uint8)

            if img.ndim == 3 and img.shape[2] == 3:
                bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)   # assume RGB
            elif img.ndim == 3 and img.shape[2] == 4:
                bgr = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
            elif img.ndim == 2:
                bgr = img
            else:
                raise typer.BadParameter(f"Unexpected screenshot ndarray shape: {img.shape}")

            ok, buf = cv2.imencode(".png", bgr)
            if not ok:
                raise typer.BadParameter("cv2.imencode failed for screenshot ndarray")
            png_bytes = buf.tobytes()

        elif isinstance(shot, (bytes, bytearray)):
            png_bytes = bytes(shot)

        elif isinstance(shot, str):
            png_bytes = base64.b64decode(shot)

        else:
            raise typer.BadParameter(f"Unrecognized screenshot type: {type(shot)}")

        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(png_bytes)
        typer.echo(f"✅ Saved screenshot: {out.resolve()}")

        if open_file:
            subprocess.run(["open", str(out.resolve())], check=False)

    finally:
        env.close()




@app.command("list-buttons")
def list_buttons(
    env_id: str = typer.Option(
        "browsergym/miniwob.buy-ticket" ,
        "--env",
        help="Gymnasium env id (default: click-menu)",
    ),
    debug_roles: bool = typer.Option(
        False,
        "--debug-roles/--no-debug-roles",
        help="Print a summary of roles seen in the AXTree.",
    ),
):
    """Extract button nodes from AXTree and print nodeId, browsergym_id, text, focusable, invalid."""
    import gymnasium as gym
    import browsergym.miniwob  # registers envs
    from collections import Counter

    if env_id.endswith("-v0"):
        env_id = env_id[:-3]

    def props_to_dict(props):
        out = {}
        for p in props or []:
            k = p.get("name")
            v = (p.get("value") or {}).get("value")
            if k is not None:
                out[str(k)] = v
        return out

    def get_role_str(n):
        role = n.get("role") or {}
        v = role.get("value", "")
        return str(v).strip().lower()

    def get_chrome_role(n):
        chrome_role = n.get("chromeRole") or {}
        v = chrome_role.get("value", None)
        try:
            return int(v) if v is not None else None
        except Exception:
            return None

    def is_button_node(n) -> bool:
        role_s = get_role_str(n)
        chrome_role = get_chrome_role(n)
        # robust: string role contains "button" OR Chrome internal role id == 9
        return ("button" in role_s) or (chrome_role == 9)

    env = gym.make(env_id)
    try:
        obs, info = env.reset()

        if not isinstance(obs, dict):
            raise typer.BadParameter(f"Unexpected obs type: {type(obs)}")

        axtree = (
            obs.get("axtree_object")
            or obs.get("axtree")
            or obs.get("ax_tree")
            or obs.get("accessibility_tree")
        )
        if axtree is None and isinstance(info, dict):
            axtree = (
                info.get("axtree_object")
                or info.get("axtree")
                or info.get("ax_tree")
                or info.get("accessibility_tree")
            )

        if not isinstance(axtree, dict) or "nodes" not in axtree:
            raise typer.BadParameter(
                f"No AXTree found for {env_id}. obs keys={list(obs.keys())}"
            )

        nodes = axtree.get("nodes", []) or []

        if debug_roles:
            rc = Counter()
            cc = Counter()
            for n in nodes:
                if n.get("ignored", False):
                    continue
                rc[get_role_str(n) or "<empty>"] += 1
                cr = get_chrome_role(n)
                if cr is not None:
                    cc[cr] += 1
            typer.echo("Top role.value strings:")
            for k, v in rc.most_common(20):
                typer.echo(f"  {k}: {v}")
            typer.echo("\nTop chromeRole numeric values:")
            for k, v in cc.most_common(20):
                typer.echo(f"  {k}: {v}")
            typer.echo("")

        buttons = []
        for n in nodes:
            if n.get("ignored", False):
                continue
            if not is_button_node(n):
                continue

            node_id = n.get("nodeId")
            bid = n.get("browsergym_id")
            name = ((n.get("name") or {}).get("value")) or ""

            props = props_to_dict(n.get("properties", []))
            focusable = props.get("focusable", False)
            invalid = props.get("invalid", None)

            buttons.append(
                {
                    "nodeId": str(node_id) if node_id is not None else "",
                    "browsergym_id": str(bid) if bid is not None else "",
                    "text": str(name),
                    "focusable": focusable,
                    "invalid": invalid,
                    "role": get_role_str(n),
                    "chromeRole": get_chrome_role(n),
                }
            )

        if not buttons:
            typer.echo("No button nodes found. Try --debug-roles to see role/chromeRole values.")
            return

        typer.echo(f"{'nodeId':>6}  {'bid':>6}  {'focusable':>9}  {'invalid':>7}  {'chrome':>6}  role  text")
        typer.echo("-" * 110)
        for b in buttons:
            typer.echo(
                f"{b['nodeId']:>6}  {b['browsergym_id']:>6}  "
                f"{str(b['focusable']):>9}  {str(b['invalid']):>7}  "
                f"{str(b['chromeRole']):>6}  {b['role']:<10}  {b['text']}"
            )

        typer.echo(f"\nTotal buttons: {len(buttons)}")

    finally:
        env.close()


@app.command("expand-menu")
def expand_menu(
    env_id: str = typer.Option(
        "browsergym/miniwob.click-menu",
        "--env",
        help="BrowserGym env id (default: click-menu)",
    ),
    max_wait_steps: int = typer.Option(
        8,
        "--max-wait-steps",
        help="How many WAIT steps to refresh UI after expand/click",
    ),
):
    """
    Expand MiniWoB click-menu submenus using ONLY BrowserGym observations:
    - uses AXTree to find menuitems/hasPopup
    - uses extra_element_properties bboxes to click near the right edge (submenu arrow)
    - saves screenshots and prints newly revealed submenu items
    - solves 'Select A>B>...' goal paths
    """
    # ----------------- small helpers -----------------
    def safe_name(s: str) -> str:
        return re.sub(r"[^A-Za-z0-9._-]+", "_", s).strip("_")

    def normalize_task_from_env_id(eid: str) -> str:
        s = eid.strip()
        if s.endswith("-v0"):
            s = s[:-3]
        m = re.search(r"miniwob\.([A-Za-z0-9_-]+)$", s)
        if m:
            return m.group(1)
        if "/" in s:
            s = s.split("/")[-1]
        if "." in s:
            s = s.split(".")[-1]
        return s

    def miniwob_url_for_env(eid: str) -> str:
        base = os.environ.get("MINIWOB_URL", "").strip()
        if not base:
            raise typer.BadParameter(
                'MINIWOB_URL is not set. Example:\n'
                '  export MINIWOB_URL="http://127.0.0.1:8000/miniwob/"'
            )
        if not base.endswith("/"):
            base += "/"
        task = normalize_task_from_env_id(eid)
        return base + (task if task.endswith(".html") else f"{task}.html")

    def spawn_detached_webkit(url: str, width: int = 1400, height: int = 900) -> int:
        child_code = r"""
from playwright.sync_api import sync_playwright
import sys, time
url = sys.argv[1]
w = int(sys.argv[2]); h = int(sys.argv[3])
with sync_playwright() as p:
    browser = p.webkit.launch(headless=False)
    page = browser.new_page(viewport={"width": w, "height": h})
    page.goto(url)
    while True:
        time.sleep(3600)
"""
        p = subprocess.Popen(
            [sys.executable, "-c", child_code, url, str(width), str(height)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        return p.pid

    def props_to_dict(props: List[Dict[str, Any]]) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for p in props or []:
            k = p.get("name")
            v = (p.get("value") or {}).get("value")
            if k is not None:
                out[str(k)] = v
        return out

    def get_axtree(obs: Dict[str, Any], info: Dict[str, Any]) -> Dict[str, Any]:
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
            raise typer.BadParameter(
                f"No AXTree found. obs keys={list(obs.keys()) if isinstance(obs, dict) else type(obs)}"
            )
        return ax

    def list_menuitems(ax: Dict[str, Any]) -> List[Tuple[str, str, str, Dict[str, Any]]]:
        rows: List[Tuple[str, str, str, Dict[str, Any]]] = []
        for n in ax.get("nodes", []) or []:
            if n.get("ignored", False):
                continue
            role = str((n.get("role") or {}).get("value", "")).lower()
            if role != "menuitem":
                continue
            bid = n.get("browsergym_id")
            if bid is None:
                continue
            name = str(((n.get("name") or {}).get("value")) or "")
            props = props_to_dict(n.get("properties", []))
            rows.append((str(bid), str(n.get("nodeId", "")), name, props))
        return rows

    CLICKLIKE_ROLES = {"menuitem", "button", "link", "listitem", "option", "tab"}

    def list_clickables(ax: Dict[str, Any]) -> List[Tuple[str, str, str]]:
        out: List[Tuple[str, str, str]] = []
        for n in ax.get("nodes", []) or []:
            if n.get("ignored", False):
                continue
            role = str((n.get("role") or {}).get("value", "")).lower()
            if role not in CLICKLIKE_ROLES:
                continue
            bid = n.get("browsergym_id")
            if bid is None:
                continue
            name = str(((n.get("name") or {}).get("value")) or "")
            out.append((str(bid), role, name))
        return out

    def clickables_set(ax: Dict[str, Any]) -> set[tuple[str, str, str]]:
        return {(bid, role, name) for (bid, role, name) in list_clickables(ax)}

    def save_obs_screenshot(obs: Dict[str, Any], path: Path) -> bool:
        shot = obs.get("screenshot", None)
        if shot is None or not isinstance(shot, np.ndarray):
            return False
        img = shot
        if img.dtype != np.uint8:
            img = np.clip(img, 0, 255).astype(np.uint8)
        if img.ndim == 3 and img.shape[2] == 3:
            bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        elif img.ndim == 3 and img.shape[2] == 4:
            bgr = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
        else:
            bgr = img
        path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(path), bgr)
        return True

    def safe_step(env, action: str):
        try:
            return env.step(action)
        except Exception:
            return None

    def do_wait_refresh(env, obs, info, ax, steps: int = 6, wait_ms: int = 120):
        """Call WAIT repeatedly to force obs/axtree refresh."""
        cur_obs, cur_info, cur_ax = obs, info, ax
        for _ in range(steps):
            r = safe_step(env, f"WAIT({wait_ms})")
            if r is None:
                break
            cur_obs, _rew, term, trunc, cur_info = r
            if not isinstance(cur_obs, dict):
                break
            cur_ax = get_axtree(cur_obs, cur_info)
            if term or trunc:
                break
        return cur_obs, cur_info, cur_ax

    def bbox_for_bid(extra: Any, bid: str):
        """Return (x,y,w,h) in CSS pixels if available."""
        bid = str(bid)
        if not isinstance(extra, list):
            return None
        for e in extra:
            ebid = str(e.get("browsergym_id") or e.get("bid") or e.get("id") or "")
            if ebid != bid:
                continue
            if isinstance(e.get("bbox"), dict):
                bb = e["bbox"]
                x, y = float(bb.get("x", 0)), float(bb.get("y", 0))
                w = float(bb.get("width", bb.get("w", 0)))
                h = float(bb.get("height", bb.get("h", 0)))
                return (x, y, w, h)
            if isinstance(e.get("bounding_box"), dict):
                bb = e["bounding_box"]
                x, y = float(bb.get("x", 0)), float(bb.get("y", 0))
                w = float(bb.get("width", bb.get("w", 0)))
                h = float(bb.get("height", bb.get("h", 0)))
                return (x, y, w, h)
            if all(k in e for k in ("left", "top", "right", "bottom")):
                x, y = float(e["left"]), float(e["top"])
                w, h = float(e["right"] - e["left"]), float(e["bottom"] - e["top"])
                return (x, y, w, h)
        return None

    def click_bid(env, bid: str):
        return safe_step(env, f"CLICK({bid})")

    def click_xy(env, x: float, y: float):
        # Assumes BrowserGym supports coordinate click in this format.
        return safe_step(env, f"CLICK({int(x)},{int(y)})")

    def parse_goal_path(goal: str) -> List[str]:
        m = re.search(r"Select\s+(.+)$", goal.strip(), flags=re.IGNORECASE)
        if not m:
            return []
        return [p.strip() for p in m.group(1).split(">") if p.strip()]

    def find_bid_by_name(ax: Dict[str, Any], target: str) -> str | None:
        t = target.strip().lower()
        # prefer menuitems first
        for bid, _nid, name, _props in list_menuitems(ax):
            if name.strip().lower() == t:
                return bid
        # fallback to any clickable role
        for bid, role, name in list_clickables(ax):
            if name.strip().lower() == t:
                return bid
        return None

    # ----------------- run -----------------
    if env_id.endswith("-v0"):
        env_id = env_id[:-3]
    safe_env = safe_name(env_id)

    env = gym.make(env_id)
    try:
        obs, info = env.reset()
        if not isinstance(obs, dict):
            raise typer.BadParameter(f"Unexpected obs type: {type(obs)}")

        typer.echo(f"debug obs keys: {sorted(list(obs.keys()))}")

        goal = obs.get("goal", "") or ""
        typer.echo("=== GOAL ===")
        typer.echo(goal if goal else "(no obs['goal'])")
        typer.echo("")

        # Always open detached viewer (for human watching only)
        url = miniwob_url_for_env(env_id)
        pid = spawn_detached_webkit(url, width=1400, height=900)
        typer.echo(f"✅ Opened detached WebKit pid={pid} -> {url}\n")

        ax = get_axtree(obs, info)

        typer.echo("=== INITIAL MENUITEMS ===")
        for bid, node_id, name, props in list_menuitems(ax):
            hp = props.get("hasPopup", None)
            extra = f" hasPopup={hp}" if hp is not None else ""
            typer.echo(f"menuitem  bid={bid:<6} nodeId={node_id:<6} name={name}{extra}")
        typer.echo("")

        # Expand all submenu triggers at top level (and repeat until stable)
        typer.echo("=== EXPANSION TRACE ===")
        expanded = set()
        for _round in range(3):  # a couple passes is enough in this task
            popup_items = [(bid, node_id, name) for (bid, node_id, name, props) in list_menuitems(ax)
                           if str(props.get("hasPopup", "")).lower() == "menu"]
            if not popup_items:
                break

            progressed = False
            for bid, node_id, name in popup_items:
                if bid in expanded:
                    continue

                before = clickables_set(ax)
                safe_bid = safe_name(bid)

                save_obs_screenshot(obs, Path("screenshots") / f"{safe_env}_bid{safe_bid}_before.png")

                typer.echo(f"[expand] parent={name!r} bid={bid} nodeId={node_id}")

                # 1) click parent to focus/select
                r = click_bid(env, bid)
                if r is None:
                    typer.echo("  (CLICK(bid) failed; skipping)")
                    continue
                obs, reward, term, trunc, info = r
                ax = get_axtree(obs, info)

                # 2) click near right edge (arrow area) using bbox
                extra = obs.get("extra_element_properties", [])
                bb = bbox_for_bid(extra, bid)
                if bb is not None:
                    x, y, w, h = bb
                    x_arrow = x + w - 3
                    y_mid = y + h / 2
                    safe_step(env, "WAIT(60)")
                    click_xy(env, x_arrow, y_mid)

                # 3) refresh a bit
                obs, info, ax = do_wait_refresh(env, obs, info, ax, steps=max_wait_steps, wait_ms=120)

                save_obs_screenshot(obs, Path("screenshots") / f"{safe_env}_bid{safe_bid}_after_expand.png")

                after = clickables_set(ax)
                new_items = sorted(after - before, key=lambda x: (x[1], x[2], x[0]))

                typer.echo(f"  clickables_now={len(after)} reward={reward}")
                if new_items:
                    typer.echo("  NEW items revealed:")
                    for nbid, nrole, nname in new_items:
                        typer.echo(f"    {nrole:<8} bid={nbid:<6} name={nname}")
                else:
                    typer.echo("  (no new items revealed)")

                expanded.add(bid)
                progressed = True

                if term or trunc:
                    typer.echo("  (episode ended during expansion)")
                    break

            if not progressed:
                break

        typer.echo("")
        typer.echo(f"Done expanding. expanded={len(expanded)}")

        # Solve goal path: Select A>B>C
        path = parse_goal_path(goal)
        typer.echo("\n=== SOLVE GOAL PATH ===")
        if not path:
            typer.echo("(no recognizable 'Select A>B>...' path)")
            return
        typer.echo(" -> ".join(path))

        # For each step: ensure its submenu is visible by expanding the parent, then click the child
        for i, name in enumerate(path):
            # refresh state
            obs, info, ax = do_wait_refresh(env, obs, info, ax, steps=2, wait_ms=80)

            bid = find_bid_by_name(ax, name)
            if bid is None:
                typer.echo(f"❌ Could not find {name!r} in current AXTree.")
                break

            # If this is not the last step, expand it (submenu parent)
            is_last = (i == len(path) - 1)

            typer.echo(f"[step] {name!r} -> bid={bid}")
            r = click_bid(env, bid)
            if r is None:
                typer.echo("  (click failed; stopping)")
                break
            obs, reward, term, trunc, info = r
            ax = get_axtree(obs, info)

            # If not last, attempt arrow click to reveal submenu
            if not is_last:
                extra = obs.get("extra_element_properties", [])
                bb = bbox_for_bid(extra, bid)
                if bb is not None:
                    x, y, w, h = bb
                    click_xy(env, x + w - 3, y + h / 2)
                obs, info, ax = do_wait_refresh(env, obs, info, ax, steps=max_wait_steps, wait_ms=120)

            if term or trunc:
                typer.echo(f"Episode ended. reward={reward}")
                break
    finally:
        env.close()

@app.command("list-button-bboxes")
def list_button_bboxes(
    env_id: str = typer.Option("browsergym/miniwob.buy-ticket", "--env"),
    annotate: bool = typer.Option(False, "--annotate/--no-annotate", help="Save annotated PNG"),
):

    def props_to_dict(props):
        out = {}
        for p in props or []:
            k = p.get("name")
            v = (p.get("value") or {}).get("value")
            if k is not None:
                out[str(k)] = v
        return out

    def is_button_node(n) -> bool:
        role_s = str((n.get("role") or {}).get("value", "")).lower()
        chrome_role = (n.get("chromeRole") or {}).get("value", None)
        try:
            chrome_role = int(chrome_role) if chrome_role is not None else None
        except Exception:
            chrome_role = None
        return ("button" in role_s) or (chrome_role == 9)

    def bbox_from_extra(extra, bid: str):
        print("bbox from extra bid:{bid}")
        bid = str(bid)
        for e in extra or []:
            ebid = str(e.get("browsergym_id") or e.get("bid") or e.get("id") or "")
            if ebid != bid:
                continue
            if isinstance(e.get("bbox"), dict):
                bb = e["bbox"]
                x, y = float(bb.get("x", 0)), float(bb.get("y", 0))
                w = float(bb.get("width", bb.get("w", 0)))
                h = float(bb.get("height", bb.get("h", 0)))
                return (x, y, x + w, y + h)
            if isinstance(e.get("bounding_box"), dict):
                bb = e["bounding_box"]
                x, y = float(bb.get("x", 0)), float(bb.get("y", 0))
                w = float(bb.get("width", bb.get("w", 0)))
                h = float(bb.get("height", bb.get("h", 0)))
                return (x, y, x + w, y + h)
            if all(k in e for k in ("left", "top", "right", "bottom")):
                return (float(e["left"]), float(e["top"]), float(e["right"]), float(e["bottom"]))
        return None

    def get_page(env):
        # Try common locations (BrowserGym versions differ)
        for attr in ("page", "_page"):
            if hasattr(env, attr):
                return getattr(env, attr)
        if hasattr(env, "unwrapped"):
            for attr in ("page", "_page"):
                if hasattr(env.unwrapped, attr):
                    return getattr(env.unwrapped, attr)
        return None

    def bbox_from_dom(page, bid: str):
        # MiniWoB usually uses data-bid attributes
        sel = f'[data-bid="{bid}"]'
        try:
            box = page.eval_on_selector(
                sel,
                """el => {
                  const r = el.getBoundingClientRect();
                  return {x1:r.left, y1:r.top, x2:r.right, y2:r.bottom};
                }""",
            )
            if not box:
                return None
            return (float(box["x1"]), float(box["y1"]), float(box["x2"]), float(box["y2"]))
        except Exception:
            return None

    env = gym.make(env_id)
    try:
        obs, info = env.reset()

        if not isinstance(obs, dict):
            raise typer.BadParameter(f"Unexpected obs type: {type(obs)}")

        axtree = obs.get("axtree_object") or obs.get("axtree") or obs.get("ax_tree") or obs.get("accessibility_tree")
        if not isinstance(axtree, dict):
            raise typer.BadParameter(f"No AXTree found. obs keys={list(obs.keys())}")

        # Collect button nodes from AXTree
        buttons = []
        print(f"len buttons from axtree:{len(buttons)} ")
        for n in axtree.get("nodes", []) or []:
            if n.get("ignored", False):
                continue
            if not is_button_node(n):
                continue
            bid = n.get("browsergym_id")
            name = ((n.get("name") or {}).get("value")) or ""
            props = props_to_dict(n.get("properties", []))
            buttons.append((str(n.get("nodeId","")), str(bid or ""), str(name), props.get("focusable", False)))

        if not buttons:
            typer.echo("No buttons found in AXTree.")
            return

        # Choose bbox source
        extra = obs.get("extra_element_properties", None)
        use_extra = isinstance(extra, list) and len(extra) > 0
        print(f"use_extra:{use_extra}")
        page = None
        if not use_extra:
            page = get_page(env)
            print("found page")
            if page is None:
                raise typer.BadParameter(
                    "No extra_element_properties list, and could not access Playwright page from env."
                )

        # Optional image for annotation
        img_bgr = None
        if annotate and isinstance(obs.get("screenshot", None), np.ndarray):
            img = obs["screenshot"]
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) if img.ndim == 3 and img.shape[2] == 3 else img.copy()

        typer.echo(f"{'nodeId':>6}  {'bid':>6}  bbox(x1,y1,x2,y2)               source  text")
        typer.echo("-" * 140)

        for node_id, bid, text, focusable in buttons:
            bb = bbox_from_extra(extra, bid) if use_extra else bbox_from_dom(page, bid)
            src = "extra" if use_extra else "dom"
            typer.echo(f"{node_id:>6}  {bid:>6}  {str(bb):<32}  {src:<6}  {text}")

            if annotate and bb and img_bgr is not None:
                x1, y1, x2, y2 = map(int, map(round, bb))
                cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img_bgr, bid, (x1, max(0, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

        if annotate and img_bgr is not None:
            safe = env_id.replace("/", "_").replace(".", "_")
            out = Path("screenshots") / f"{safe}_bboxes.png"
            out.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out), img_bgr)
            typer.echo(f"\n✅ Wrote annotated image: {out.resolve()}")

    finally:
        env.close()





if __name__ == "__main__":
    app()