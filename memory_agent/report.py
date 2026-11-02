"""
Quantify the effect of memory.

    python report.py results/<exp>            -> report.md, report.html, summary.json

Primary metric: success rate on the held-out TEST seeds (memory frozen).
Comparisons are paired: every condition ran the same (task, seed) pairs.

  * Wilson 95% interval for each success rate
  * Paired difference with a 95% cluster-bootstrap interval (tasks resampled,
    since episodes of the same task are correlated)
  * Exact McNemar test on the discordant pairs
  * Learning curve over TRAIN episodes: does success rise as memory accumulates?
  * Cost: acting tokens per episode, plus tokens spent maintaining memory
Standard library only.
"""

from __future__ import annotations

import html
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path

ORDER = ["none", "working", "episodic", "procedural"]
COLORS = {"none": "#9aa3ad", "working": "#4c8bf5", "episodic": "#b07cf0", "procedural": "#f0a020"}
LABEL = {
    "none": "No memory",
    "working": "Working memory",
    "episodic": "Working + episodic",
    "procedural": "Working + procedural",
}


# ------------------------------------------------------------------ stats
def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact binomial test on discordant pairs (b: A wins, c: B wins)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2**n
    return min(1.0, 2 * tail)


def paired(rows_a: dict, rows_b: dict, iters: int = 5000, seed: int = 0) -> dict:
    """A - B on shared (task, seed) keys, cluster bootstrap over tasks."""
    keys = sorted(set(rows_a) & set(rows_b))
    if not keys:
        return {}
    a = [int(rows_a[k]) for k in keys]
    b_ = [int(rows_b[k]) for k in keys]
    diff = (sum(a) - sum(b_)) / len(keys)
    wins = sum(1 for x, y in zip(a, b_) if x and not y)
    losses = sum(1 for x, y in zip(a, b_) if y and not x)
    by_task = defaultdict(list)
    for k, x, y in zip(keys, a, b_):
        by_task[k[0]].append(x - y)
    tasks = list(by_task)
    rng = random.Random(seed)
    boots = []
    for _ in range(iters):
        sample = [d for t in (rng.choice(tasks) for _ in tasks) for d in by_task[t]]
        boots.append(sum(sample) / len(sample))
    boots.sort()
    return {
        "n_pairs": len(keys),
        "diff": diff,
        "ci": (boots[int(0.025 * iters)], boots[int(0.975 * iters) - 1]),
        "wins": wins,
        "losses": losses,
        "p_mcnemar": mcnemar_exact(wins, losses),
    }


# ------------------------------------------------------------------ load + aggregate
def load(exp: Path) -> list[dict]:
    rows = []
    for p in sorted((exp / "runs").glob("*.jsonl")):
        for line in p.read_text().splitlines():
            if line.strip():
                rows.append(json.loads(line))
    return rows


def summarize(rows: list[dict]) -> dict:
    conds = [c for c in ORDER if any(r["condition"] == c for r in rows)]
    tasks = sorted({r["task"] for r in rows})
    test = [r for r in rows if r["phase"] == "test"]
    train = [r for r in rows if r["phase"] == "train"]

    overall = {}
    for c in conds:
        rs = [r for r in test if r["condition"] == c]
        k, n = sum(r["success"] for r in rs), len(rs)
        wins = [r for r in rs if r["success"]]
        mem_rows = [r for r in rows if r["condition"] == c]
        overall[c] = {
            "n": n,
            "successes": k,
            "rate": k / n if n else 0.0,
            "ci": wilson(k, n),
            "steps_success": sum(r["n_steps"] for r in wins) / len(wins) if wins else None,
            "steps_all": sum(r["n_steps"] for r in rs) / n if n else None,
            "repeats": sum(r["repeated_actions"] for r in rs) / n if n else None,
            "errors": sum(r["errors"] for r in rs) / n if n else None,
            "tok_per_episode": sum(r["input_tokens"] + r["output_tokens"] for r in rs) / n if n else None,
            "memory_tokens_total": sum(r["memory_input_tokens"] + r["memory_output_tokens"] for r in mem_rows),
            "memory_chars": sum(r["memory_chars"] for r in rs) / n if n else None,
        }

    per_task = {t: {} for t in tasks}
    for t in tasks:
        for c in conds:
            rs = [r for r in test if r["condition"] == c and r["task"] == t]
            if rs:
                per_task[t][c] = (sum(r["success"] for r in rs), len(rs))

    keyed = {c: {(r["task"], r["seed"]): r["success"] for r in test if r["condition"] == c} for c in conds}
    comparisons = []
    pairs = [(c, "none") for c in conds if c != "none"] + [
        (conds[i + 1], conds[i]) for i in range(len(conds) - 1) if conds[i] != "none"
    ]
    for a, b in pairs:
        if a in keyed and b in keyed:
            res = paired(keyed[a], keyed[b])
            if res:
                comparisons.append({"a": a, "b": b, **res})

    curve = {}
    for c in conds:
        by_idx = defaultdict(list)
        for r in train:
            if r["condition"] == c:
                by_idx[r["idx"]].append(r["success"])
        curve[c] = [sum(v) / len(v) for _, v in sorted(by_idx.items())]

    return {
        "conditions": conds,
        "tasks": tasks,
        "llm": rows[0]["llm"] if rows else "",
        "n_rows": len(rows),
        "overall": overall,
        "per_task": per_task,
        "comparisons": comparisons,
        "train_curve": curve,
    }


# ------------------------------------------------------------------ render
def pct(x: float | None) -> str:
    return "–" if x is None else f"{100 * x:.1f}%"


def num(x: float | None, d: int = 1) -> str:
    return "–" if x is None else f"{x:,.{d}f}"


def sig(p: float) -> str:
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."


def to_markdown(s: dict, exp: str) -> str:
    o, conds = s["overall"], s["conditions"]
    out = [f"# Memory ablation on MiniWoB: `{exp}`", "", f"Model: `{s['llm']}` · tasks: {len(s['tasks'])} · episodes: {s['n_rows']}", ""]
    if s["llm"] == "scripted":
        out += ["> **Dry run with the scripted stand-in policy. These numbers say nothing about memory.**", ""]
    out += [
        "## Held-out success rate (memory frozen)",
        "",
        "| Condition | Success | 95% CI | Steps (successes) | Repeated actions / ep | Errors / ep | Tokens / ep | Memory upkeep tokens |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for c in conds:
        r = o[c]
        out.append(
            f"| {LABEL[c]} | **{pct(r['rate'])}** ({r['successes']}/{r['n']}) | {pct(r['ci'][0])}–{pct(r['ci'][1])} "
            f"| {num(r['steps_success'])} | {num(r['repeats'], 2)} | {num(r['errors'], 2)} "
            f"| {num(r['tok_per_episode'], 0)} | {num(r['memory_tokens_total'], 0)} |"
        )
    out += ["", "## Paired differences (same task and seed)", "",
            "| Comparison | Δ success | 95% CI (task bootstrap) | A wins / B wins | McNemar p |",
            "|---|---:|---:|---:|---:|"]
    for cmp in s["comparisons"]:
        out.append(
            f"| {LABEL[cmp['a']]} vs {LABEL[cmp['b']]} | **{100 * cmp['diff']:+.1f} pts** "
            f"| {100 * cmp['ci'][0]:+.1f} to {100 * cmp['ci'][1]:+.1f} | {cmp['wins']} / {cmp['losses']} "
            f"| {cmp['p_mcnemar']:.3g} {sig(cmp['p_mcnemar'])} |"
        )
    out += ["", "## Per task (held-out successes / episodes)", "",
            "| Task | " + " | ".join(LABEL[c] for c in conds) + " |",
            "|---|" + "---:|" * len(conds)]
    for t, cells in s["per_task"].items():
        out.append(f"| {t} | " + " | ".join(f"{cells[c][0]}/{cells[c][1]}" if c in cells else "–" for c in conds) + " |")
    out += ["", "## Learning curve over training episodes (mean success across tasks)", "",
            "| Train episode | " + " | ".join(LABEL[c] for c in conds) + " |", "|---|" + "---:|" * len(conds)]
    n_idx = max((len(v) for v in s["train_curve"].values()), default=0)
    for i in range(n_idx):
        out.append(f"| {i + 1} | " + " | ".join(pct(s["train_curve"][c][i]) if i < len(s["train_curve"][c]) else "–" for c in conds) + " |")
    return "\n".join(out) + "\n"


def bar_svg(s: dict) -> str:
    conds, o = s["conditions"], s["overall"]
    W, H, L, B = 560, 260, 48, 40
    bw = (W - L - 20) / max(1, len(conds))
    y = lambda v: H - B - v * (H - B - 20)
    parts = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="Held-out success by condition">']
    for v in (0, 0.25, 0.5, 0.75, 1):
        parts.append(f'<line x1="{L}" x2="{W - 10}" y1="{y(v):.1f}" y2="{y(v):.1f}" class="grid"/>'
                     f'<text x="{L - 8}" y="{y(v) + 4:.1f}" class="axis" text-anchor="end">{int(v * 100)}%</text>')
    for i, c in enumerate(conds):
        r = o[c]
        x = L + i * bw + bw * 0.18
        w = bw * 0.64
        parts.append(f'<rect x="{x:.1f}" y="{y(r["rate"]):.1f}" width="{w:.1f}" height="{y(0) - y(r["rate"]):.1f}" rx="3" fill="{COLORS[c]}"/>')
        cx = x + w / 2
        lo, hi = r["ci"]
        parts.append(f'<line x1="{cx:.1f}" x2="{cx:.1f}" y1="{y(lo):.1f}" y2="{y(hi):.1f}" class="ci"/>'
                     f'<line x1="{cx - 6:.1f}" x2="{cx + 6:.1f}" y1="{y(lo):.1f}" y2="{y(lo):.1f}" class="ci"/>'
                     f'<line x1="{cx - 6:.1f}" x2="{cx + 6:.1f}" y1="{y(hi):.1f}" y2="{y(hi):.1f}" class="ci"/>')
        parts.append(f'<text x="{cx:.1f}" y="{y(r["rate"]) - 8:.1f}" class="val" text-anchor="middle" dx="18">{100 * r["rate"]:.0f}%</text>')
        parts.append(f'<text x="{cx:.1f}" y="{H - B + 18}" class="axis" text-anchor="middle">{html.escape(LABEL[c])}</text>')
    parts.append("</svg>")
    return "".join(parts)


def curve_svg(s: dict) -> str:
    curve = s["train_curve"]
    n = max((len(v) for v in curve.values()), default=0)
    if n < 2:
        return ""
    W, H, L, B = 560, 220, 48, 34
    x = lambda i: L + i * (W - L - 20) / (n - 1)
    y = lambda v: H - B - v * (H - B - 16)
    parts = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="Success over training episodes">']
    for v in (0, 0.5, 1):
        parts.append(f'<line x1="{L}" x2="{W - 10}" y1="{y(v):.1f}" y2="{y(v):.1f}" class="grid"/>'
                     f'<text x="{L - 8}" y="{y(v) + 4:.1f}" class="axis" text-anchor="end">{int(v * 100)}%</text>')
    for i in range(n):
        parts.append(f'<text x="{x(i):.1f}" y="{H - B + 18}" class="axis" text-anchor="middle">{i + 1}</text>')
    for c in s["conditions"]:
        pts = " ".join(f"{x(i):.1f},{y(v):.1f}" for i, v in enumerate(curve[c]))
        parts.append(f'<polyline points="{pts}" fill="none" stroke="{COLORS[c]}" stroke-width="2.5"/>')
        if curve[c]:
            parts.append(f'<text x="{x(len(curve[c]) - 1) + 6:.1f}" y="{y(curve[c][-1]) + 4:.1f}" class="axis" fill="{COLORS[c]}">{html.escape(c)}</text>')
    parts.append("</svg>")
    return "".join(parts)


def to_html(s: dict, md: str, exp: str) -> str:
    rows = []
    for line in md.splitlines():
        if line.startswith("|") and not line.startswith("|---"):
            cells = [c.strip() for c in line.strip("|").split("|")]
            tag = "th" if not rows or rows[-1] == "</table>" else "td"
            rows.append("<tr>" + "".join(f"<{tag}>{html.escape(c).replace('**', '')}</{tag}>" for c in cells) + "</tr>")
        elif line.startswith("## "):
            if rows and rows[-1] != "</table>" and rows[-1].startswith("<tr>"):
                rows.append("</table>")
            rows.append(f"<h2>{html.escape(line[3:])}</h2><table>")
    if rows and rows[-1] != "</table>":
        rows.append("</table>")
    body = "\n".join(rows).replace("<table>\n</table>", "")
    warn = '<p class="warn">Dry run with the scripted stand-in policy. These numbers say nothing about memory.</p>' if s["llm"] == "scripted" else ""
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Memory Ablation Report</title>
<style>
:root{{--bg:#fbfbfa;--fg:#1d2228;--muted:#6b7480;--line:#e3e6ea;--card:#fff}}
@media (prefers-color-scheme:dark){{:root:not([data-theme=light]){{--bg:#111418;--fg:#e6eaef;--muted:#8a94a0;--line:#2a3038;--card:#161a1f}}}}
:root[data-theme=dark]{{--bg:#111418;--fg:#e6eaef;--muted:#8a94a0;--line:#2a3038;--card:#161a1f}}
body{{background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,sans-serif;margin:0;padding:24px 16px}}
main{{max-width:980px;margin:0 auto}} h1{{font-size:22px;margin:0 0 4px}} h2{{font-size:16px;margin:28px 0 8px}}
.sub{{color:var(--muted);margin:0 0 16px}} .warn{{background:#f0a02022;border:1px solid #f0a020;padding:8px 12px;border-radius:6px}}
.charts{{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:16px}}
.card{{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:12px}}
.card h3{{margin:0 0 6px;font-size:13px;color:var(--muted);font-weight:600}}
svg{{width:100%;height:auto}} .grid{{stroke:var(--line)}} .axis{{fill:var(--muted);font-size:11px}}
.val{{fill:var(--fg);font-size:12px;font-weight:600}} .ci{{stroke:var(--fg);stroke-width:1.5}}
table{{border-collapse:collapse;width:100%;font-size:13px;display:block;overflow-x:auto}}
th,td{{border-bottom:1px solid var(--line);padding:6px 10px;text-align:right;white-space:nowrap}}
th:first-child,td:first-child{{text-align:left}} th{{color:var(--muted);font-weight:600}}
</style></head><body><main>
<h1>Memory ablation on MiniWoB</h1>
<p class="sub">{html.escape(exp)} · model <code>{html.escape(s['llm'])}</code> · {len(s['tasks'])} tasks · {s['n_rows']} episodes</p>
{warn}
<div class="charts">
<div class="card"><h3>Held-out success rate, 95% CI</h3>{bar_svg(s)}</div>
<div class="card"><h3>Success over training episodes (memory accumulating)</h3>{curve_svg(s) or '<p class="sub">Needs at least 2 training seeds.</p>'}</div>
</div>
{body}
</main></body></html>"""


def main() -> None:
    if len(sys.argv) < 2:
        sys.exit("usage: python report.py results/<exp>")
    exp = Path(sys.argv[1])
    rows = load(exp)
    if not rows:
        sys.exit(f"no results in {exp}/runs")
    s = summarize(rows)
    md = to_markdown(s, exp.name)
    (exp / "report.md").write_text(md)
    (exp / "report.html").write_text(to_html(s, md, exp.name))
    (exp / "summary.json").write_text(json.dumps(s, indent=2, default=list))
    print(md)
    print(f"wrote {exp / 'report.md'}, {exp / 'report.html'}, {exp / 'summary.json'}")


if __name__ == "__main__":
    main()
