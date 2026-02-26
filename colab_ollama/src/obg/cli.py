# src/obg/cli.py
import json
import shutil
from pathlib import Path
import typer
import cv2
import subprocess
import base64
import browsergym 
from collections import deque
from typing import Any, Dict, List, Optional, Tuple

import os, re, sys, subprocess
import gymnasium as gym

   
app = typer.Typer()


@app.command("sync-ssh")
def sync_ssh(
    src: Path = typer.Option(
        Path("/content/drive/MyDrive/Colab Notebooks/.ssh"),
        "--src",
        help="Source .ssh directory (Google Drive mounted path)",
    ),
    dst: Path = typer.Option(
        Path("/content/.ssh"),
        "--dst",
        help="Destination .ssh directory on Colab VM",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        help="Overwrite existing destination by deleting it first",
    ),
):
    """Copy .ssh from Drive into /content and set safe permissions."""
    if not src.exists() or not src.is_dir():
        raise typer.BadParameter(f"Source does not exist or is not a directory: {src}")

    if dst.exists():
        if not force:
            raise typer.BadParameter(f"Destination already exists: {dst}. Use --force.")
        shutil.rmtree(dst)

    shutil.copytree(src, dst)

    os.chmod(dst, 0o700)
    for p in dst.rglob("*"):
        os.chmod(p, 0o700 if p.is_dir() else 0o600)

    typer.echo(f"✅ Copied {src} -> {dst} and set permissions.")

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
    """Reset a BrowserGym MiniWoB env and print the accessibility tree (AXTree)."""
    import gymnasium as gym
    import browsergym.miniwob  # registers envs

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

@app.command("print-goal")
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




@app.command("list-button-bboxes")
def list_button_bboxes(
    env_id: str = typer.Option("browsergym/miniwob.buy-ticket", "--env"),
    annotate: bool = typer.Option(False, "--annotate/--no-annotate", help="Save annotated PNG"),
):
    """Print bboxes for all button nodes. Uses extra_element_properties if available, else DOM rects via Playwright page."""
    import gymnasium as gym
    import browsergym.miniwob  # registers envs
    import numpy as np
    import cv2
    from pathlib import Path

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

    def is_button_node(n) -> bool:
        role_s = str((n.get("role") or {}).get("value", "")).lower()
        chrome_role = (n.get("chromeRole") or {}).get("value", None)
        try:
            chrome_role = int(chrome_role) if chrome_role is not None else None
        except Exception:
            chrome_role = None
        return ("button" in role_s) or (chrome_role == 9)

    def bbox_from_extra(extra, bid: str):
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

        page = None
        if not use_extra:
            page = get_page(env)
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

@app.command("list-actions")
def list_actions(
    env_id: str = typer.Option("browsergym/miniwob.click-menu", "--env"),
    debug: bool = typer.Option(False, "--debug/--no-debug"),
):
    """
    Probe which string-action verbs are accepted by the env parser.

    Robust signals:
      - If env.step(action) does NOT raise -> verb is recognized (even if args invalid)
      - If it raises, check exception text for parse/unknown
      - Also check obs['last_action'] / obs['last_action_error'] after the call
    """
    import re
    import gymnasium as gym
    import browsergym.miniwob  # registers envs

    if env_id.endswith("-v0"):
        env_id = env_id[:-3]

    CANDIDATES = {
        "CLICK": ["CLICK(0)", "CLICK(1)", "CLICK(10,10)"],
        "TYPE": ['TYPE(0,"x")', 'TYPE(1,"x")'],
        "SCROLL": ["SCROLL(down,100)", "SCROLL(up,100)"],
        "WAIT": ["WAIT(10)", "WAIT(100)"],
        "HOVER": ["HOVER(0)", "HOVER(1)", "HOVER(10,10)"],
        "PRESS": ['PRESS("Enter")', 'PRESS("Tab")'],
        "KEYPRESS": ['KEYPRESS("Enter")', 'KEYPRESS("Tab")'],
    }

    UNKNOWN_PAT = re.compile(
        r"(unknown action|unrecognized|invalid action|cannot parse|parse error|unsupported)",
        re.IGNORECASE,
    )

    def looks_unknown(msg: str) -> bool:
        return bool(UNKNOWN_PAT.search((msg or "").strip()))

    env = gym.make(env_id)
    try:
        obs, info = env.reset()
        if not isinstance(obs, dict):
            raise typer.BadParameter(f"Unexpected obs type: {type(obs)}")

        recognized, rejected = [], []

        for verb, examples in CANDIDATES.items():
            ok = False

            for a in examples:
                before_last_action = obs.get("last_action") if isinstance(obs, dict) else None

                threw = False
                exc_msg = ""
                try:
                    obs2, reward, term, trunc, info2 = env.step(a)
                except Exception as e:
                    threw = True
                    exc_msg = str(e)
                    obs2 = obs  # keep previous

                last_action = obs2.get("last_action") if isinstance(obs2, dict) else None
                last_err = obs2.get("last_action_error") if isinstance(obs2, dict) else None

                # Determine recognition:
                # - If step didn't throw: recognized.
                # - If it threw but NOT with "unknown/parse": still likely recognized (bad args downstream).
                # - If obs.last_action updated to our string: recognized.
                if (not threw) or (threw and not looks_unknown(exc_msg)) or (last_action == a):
                    ok = True

                if debug:
                    print(
                        f"[{verb}] try={a!r} threw={threw} exc={exc_msg[:120]!r} "
                        f"last_action={last_action!r} last_err={last_err!r}"
                    )

                if ok:
                    break

            (recognized if ok else rejected).append(verb)

        print(f"Env: {env_id}\n")
        print("Recognized verbs (probe):")
        for v in recognized:
            print(" -", v)

        print("\nRejected / likely unsupported:")
        for v in rejected:
            print(" -", v)

        print(
            "\nTip: For MiniWoB you should see at least CLICK / TYPE / SCROLL / WAIT "
            "(even if particular signatures differ). Run with --debug to see why a verb was rejected."
        )

    finally:
        env.close()

@app.command("find-action-parser")
def find_action_parser(
    needle: str = typer.Option("parse", "--needle", help="Search needle (e.g. parse, action, grammar)"),
    max_hits: int = typer.Option(40, "--max-hits", help="Max hits to print"),
):
    """Search installed browsergym sources for likely action parsing code."""
    import re
    from pathlib import Path
    import browsergym
    import typer

    roots = list(getattr(browsergym, "__path__", []) or [])
    if not roots:
        raise typer.BadParameter("browsergym.__path__ is empty")
    root = Path(roots[0]).resolve()
    typer.echo(f"browsergym root: {root}")

    # patterns that tend to show up in action parsing / action space
    pats = [
        re.compile(r"\baction_space\b"),
        re.compile(r"\bparse\b", re.IGNORECASE),
        re.compile(r"\bCLICK\b|\bTYPE\b|\bSCROLL\b|\bWAIT\b"),
        re.compile(r"Unicode\(", re.IGNORECASE),
        re.compile(r"gym\.spaces\.(Text|Sequence|Box|Discrete)"),
        re.compile(r"re\.compile\("),
    ]
    needle_re = re.compile(re.escape(needle), re.IGNORECASE) if needle else None

    hits = []
    for p in root.rglob("*.py"):
        txt = p.read_text(encoding="utf-8", errors="ignore")
        score = sum(1 for pat in pats if pat.search(txt))
        if needle_re and not needle_re.search(txt):
            continue
        if score > 0:
            hits.append((score, p))

    hits.sort(reverse=True, key=lambda x: x[0])
    if not hits:
        typer.echo("No candidate files found.")
        return

    typer.echo("\nTop candidates:")
    for score, p in hits[:max_hits]:
        typer.echo(f"  score={score:>2}  {p}")

    # show a small snippet from the best file(s)
    best = hits[:3]
    for score, p in best:
        typer.echo(f"\n--- snippet: {p} ---")
        txt = p.read_text(encoding="utf-8", errors="ignore")
        # show lines near "action_space" or "parse"
        for key in ["action_space", "parse", "CLICK", "TYPE", "SCROLL", "WAIT"]:
            i = txt.find(key)
            if i != -1:
                start = max(0, i - 400)
                end = min(len(txt), i + 400)
                typer.echo(txt[start:end])
                break



if __name__ == "__main__":
    app()