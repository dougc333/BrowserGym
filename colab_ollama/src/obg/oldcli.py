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
