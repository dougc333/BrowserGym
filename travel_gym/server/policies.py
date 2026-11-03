"""
Policies for the agentic search loop.  Each step:  (observation, state) -> (thought, notes, action)

  claude    Claude reads the goal, its own notes, its recent actions and the page's
            accessibility tree, and returns one action. It keeps a running NOTES block
            (working memory): long searches outgrow the context window, so the agent
            records candidate fares there instead of relying on old pages.
  openai    Same prompt through any OpenAI-compatible server (Ollama, vLLM).
  scripted  A systematic searcher with no model: it enumerates every allowed
            (airport, date), pages through results, opens candidates that could still
            beat the best total found so far, reads the true total on the review page,
            and books the best. It uses only what the site shows (never the database),
            so its runs are honest, long, and usually optimal: a free teacher for
            behavior cloning and a baseline for the LLM.
"""

from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass, field

from browsergym.core.action.highlevel import HighLevelActionSet

from generator import AIRPORT_NAMES

ACTION_SET = HighLevelActionSet(subsets=["bid", "nav", "chat", "infeas"], strict=False, multiaction=False)
DEFAULT_MODEL = os.environ.get("CLAUDE_MODEL", "claude-sonnet-5-5")
MODELS = ["claude-haiku-5-5", "claude-sonnet-5-5", "claude-opus-5-5"]

# Nous Research (Hermes models) through the Nous Portal's OpenAI-compatible API.
# Check the base URL and model ids in your Nous Portal account; both can be overridden.
NOUS_BASE_URL = os.environ.get("NOUS_BASE_URL", "https://inference-api.nousresearch.com/v1")
NOUS_DEFAULT_MODEL = os.environ.get("NOUS_MODEL", "Hermes-4-70B")
NOUS_MODELS = ["Hermes-4-70B", "Hermes-4-405B"]


def openai_compatible(policy: str, model: str | None) -> tuple[str, str, str]:
    """(model, base_url, api_key) for 'nous' / 'mcp-nous' and 'openai' / 'mcp-openai' (model@url)."""
    if policy.endswith("nous"):
        return model or NOUS_DEFAULT_MODEL, NOUS_BASE_URL, os.environ.get("NOUS_API_KEY", "")
    m, _, url = (model or "").rpartition("@")
    return m, url or "http://localhost:11434/v1", os.environ.get("OPENAI_API_KEY", "not-needed")


@dataclass
class Decision:
    thought: str
    action: str
    notes: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: int = 0
    prompt: str = ""


@dataclass
class StepContext:
    goal: str
    axtree: str
    url: str
    t: int
    history: list[tuple[str, str]]  # (action, error)
    notes: str
    task: dict  # parsed constraints (used by the scripted policy only)
    last_error: str = ""


# ------------------------------------------------------------------ LLM policy
SYSTEM = """\
You are a web agent searching a travel site for the cheapest flight that meets ALL of the user's
requirements, then booking it. You see the page as an accessibility tree; interactive elements
have ids in brackets, e.g. [13] button 'Search flights'. Act with exactly one action per turn.

Strategy that works:
- Search every allowed airport and date. Look at every results page ("Next page").
- Listed fares exclude taxes and bag fees when the site says so: the true total is only on the
  review page ("Continue to review"). Compare true totals for the right number of travelers and bags.
- Never book a flight that breaks a requirement, even if it is cheaper.
- Booking is final and allowed once. Book only when you are confident it is the cheapest valid option.
- Select options in dropdowns with select_option(id, "label or value").

Keep NOTES as your working memory: searches done, candidates and their true totals, the best so far.
Rewrite the full notes each turn (keep them short).

Reply in exactly this format:
<thinking>one to three sentences</thinking>
<notes>your updated notes</notes>
<action>one action call</action>"""

ACTION_RE = re.compile(r"<action>\s*(.*?)\s*</action>", re.S)
THINK_RE = re.compile(r"<thinking>\s*(.*?)\s*</thinking>", re.S)
NOTES_RE = re.compile(r"<notes>\s*(.*?)\s*</notes>", re.S)


def build_prompt(ctx: StepContext, recent: int = 12) -> str:
    hist = ctx.history[-recent:]
    offset = len(ctx.history) - len(hist)
    lines = [f"{offset + i + 1}. {a}" + (f"  ERROR: {e[:160]}" if e else "") for i, (a, e) in enumerate(hist)]
    parts = [
        f"# Goal\n{ctx.goal}",
        f"# Your notes (working memory)\n{ctx.notes or '(empty)'}",
        f"# Recent actions (step {ctx.t}; {offset} earlier actions omitted)\n" + ("\n".join(lines) or "(none)"),
        f"# Current page: {ctx.url}\n{ctx.axtree}",
        "# Action space\n" + ACTION_SET.describe(with_long_description=False, with_examples=True),
        "# Next action",
    ]
    if ctx.last_error:
        parts.insert(3, f"# Error from your last action\n{ctx.last_error}")
    return "\n\n".join(parts)


def parse(text: str, old_notes: str) -> tuple[str, str, str]:
    a, t, n = ACTION_RE.search(text), THINK_RE.search(text), NOTES_RE.search(text)
    action = a.group(1).strip().strip("`") if a else "noop()"
    return (t.group(1).strip() if t else ""), action, (n.group(1).strip() if n else old_notes)


class ClaudePolicy:
    name = "claude"

    def __init__(self, model: str | None = None):
        from anthropic import Anthropic

        self.client, self.model = Anthropic(), model or DEFAULT_MODEL

    def __call__(self, ctx: StepContext) -> Decision:
        prompt = build_prompt(ctx)
        t0 = time.time()
        for attempt in range(6):
            try:
                r = self.client.messages.create(model=self.model, max_tokens=900, system=SYSTEM,
                                                messages=[{"role": "user", "content": prompt}])
                break
            except Exception as e:  # transient API errors
                if attempt == 5 or not any(s in str(e).lower() for s in ("429", "overload", "529", "500", "timeout")):
                    raise
                time.sleep(2 ** attempt)
        text = "".join(b.text for b in r.content if getattr(b, "type", "") == "text")
        thought, action, notes = parse(text, ctx.notes)
        return Decision(thought, action, notes, r.usage.input_tokens, r.usage.output_tokens,
                        int((time.time() - t0) * 1000), prompt)


class OpenAIPolicy:
    name = "openai"

    def __init__(self, model: str, base_url: str, api_key: str = ""):
        import json
        import urllib.request

        self.model, self.base_url, self._json, self._req = model, base_url.rstrip("/"), json, urllib.request
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "not-needed")

    def __call__(self, ctx: StepContext) -> Decision:
        prompt = build_prompt(ctx)
        body = {"model": self.model, "max_tokens": 900,
                "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}]}
        t0 = time.time()
        req = self._req.Request(f"{self.base_url}/chat/completions", data=self._json.dumps(body).encode(),
                                headers={"Content-Type": "application/json",
                                         "Authorization": f"Bearer {self.api_key}"})
        for attempt in range(6):
            try:
                with self._req.urlopen(req, timeout=300) as resp:
                    d = self._json.loads(resp.read())
                break
            except Exception as e:
                if attempt == 5 or not any(s in str(e) for s in ("429", "500", "502", "503", "timed out")):
                    raise
                time.sleep(2 ** attempt)
        text = d["choices"][0]["message"].get("content") or ""
        u = d.get("usage") or {}
        thought, action, notes = parse(text, ctx.notes)
        return Decision(thought, action, notes, u.get("prompt_tokens", 0), u.get("completion_tokens", 0),
                        int((time.time() - t0) * 1000), prompt)


# ------------------------------------------------------------------ scripted searcher
ITEM_RE = re.compile(
    r"listitem '(?:Sponsored: )?Flight (\w+), [^,]+, (\d\d:\d\d) to [^,]+, (Nonstop|\d stops?), ([^,]+), "
    r"(Refundable|Non-refundable), fare \$([\d,]+\.\d\d) per person(?:, bag \$([\d,]+\.\d\d), taxes ([\d.]+)%)?'")
LINK_RE = r"\[(\w+)\] link '{label}'"
COMBO_RE = r"\[(\w+)\] combobox '{label}'"
BUTTON_RE = r"\[(\w+)\] button '{label}'"
TOTAL_RE = re.compile(r"Total price \$([\d,]+\.\d\d)")
MIN_TAX = 0.055  # lowest tax rate the scripted agent assumes when pruning (conservative)


def _bid(pattern: str, axtree: str) -> str | None:
    m = re.search(pattern, axtree)
    return m.group(1) if m else None


@dataclass
class ScriptedState:
    searches: list = field(default_factory=list)  # remaining (origin, date)
    phase: str = "init"
    form_step: int = 0
    current: tuple | None = None
    inspect: list = field(default_factory=list)  # (lower bound, flight_no) to open on this results page
    inspected: set = field(default_factory=set)
    current_fn: str = ""
    bags_set: bool = False
    best: dict | None = None  # {"flight_no", "total", "review_url"}
    back: int = 0


class ScriptedPolicy:
    """Finite-state searcher. Thought strings explain each move so its trajectories read well."""

    name = "scripted"

    def __init__(self):
        self.s = ScriptedState()

    def _notes(self) -> str:
        b = self.s.best
        best = f"best so far {b['flight_no']} ${b['total']:.2f}" if b else "no valid candidate yet"
        return f"searches left: {len(self.s.searches)}; inspected {len(self.s.inspected)} flights; {best}"

    def _d(self, thought: str, action: str) -> Decision:
        return Decision(thought, action, self._notes())

    def __call__(self, ctx: StepContext) -> Decision:
        s, task, ax = self.s, ctx.task, ctx.axtree
        if s.phase == "init":
            s.searches = [(o, d) for o in task["origins"] for d in task["dates"]]
            s.phase = "next_search"

        if s.phase == "back":
            if s.back > 0:
                s.back -= 1
                if s.back == 0:
                    s.phase = "results"
                return self._d("Return to the results list to continue comparing.", "go_back()")
            s.phase = "results"

        if s.phase == "next_search":
            if not s.searches:
                if not s.best:
                    return self._d("No flight satisfies every requirement.",
                                   'report_infeasible("No flight meets all requirements.")')
                s.phase = "book"
                return self._d(f"All searches done. The cheapest valid total is {s.best['flight_no']} at "
                               f"${s.best['total']:.2f}; return to its review page.", f'goto("{s.best["review_url"]}")')
            s.current = s.searches.pop(0)
            s.phase, s.form_step = "form", 0
            if "/results" in ctx.url or "/flight/" in ctx.url or "/review" in ctx.url:
                return self._d("Start the next search.", f'click("{_bid(LINK_RE.format(label="New search"), ax)}")')

        if s.phase == "form":
            origin, day = s.current
            steps = [
                ("From", AIRPORT_NAMES[origin]), ("To", AIRPORT_NAMES[task["dest"]]),
                ("Departure date", day), ("Travelers", f"{task['passengers']} adult{'s' if task['passengers'] > 1 else ''}"),
            ]
            if s.form_step < len(steps):
                label, value = steps[s.form_step]
                s.form_step += 1
                return self._d(f"Set {label} to {value}.", f'select_option("{_bid(COMBO_RE.format(label=label), ax)}", "{value}")')
            s.phase, s.inspect = "results_first", []
            return self._d(f"Search {origin} on {day}.", f'click("{_bid(BUTTON_RE.format(label="Search flights"), ax)}")')

        if s.phase == "book":
            if "/review" in ctx.url:
                s.phase = "done"
                return self._d("Confirm the booking of the cheapest valid flight.",
                               f'click("{_bid(BUTTON_RE.format(label="Confirm booking"), ax)}")')
            return self._d("Open the review page of the best flight.", f'goto("{s.best["review_url"]}")')

        if s.phase in ("results", "results_first"):
            if "No flights found" in ax:
                s.phase = "next_search"
                return self(ctx)
            if s.phase == "results_first":
                s.phase = "results"
                # build this page's inspection list from what is listed
                s.inspect = self._candidates(ax, task)
            best = s.best["total"] if s.best else float("inf")
            pending = [fn for bound, fn in s.inspect if fn not in s.inspected and bound < best]
            if pending:
                fn = pending[0]
                s.inspected.add(fn)
                s.current_fn, s.bags_set, s.phase = fn, False, "details"
                return self._d(f"{fn} meets the visible requirements and could beat the best total; open its details.",
                               f'click("{_bid(LINK_RE.format(label=f"View details for {fn}"), ax)}")')
            nxt = _bid(LINK_RE.format(label="Next page"), ax)
            if nxt:
                s.phase = "results_first"
                return self._d("Check the next page of results.", f'click("{nxt}")')
            s.phase = "next_search"
            return self(ctx)

        if s.phase == "details":
            bags = _bid(COMBO_RE.format(label="Checked bags per traveler"), ax)
            if bags and task["bags"] and not s.bags_set:
                s.bags_set = True
                return self._d(f"Choose {task['bags']} checked bag per traveler.",
                               f'select_option("{bags}", "{task["bags"]}")')
            s.phase = "review"
            return self._d("Continue to the review page to see the true total.",
                           f'click("{_bid(BUTTON_RE.format(label="Continue to review"), ax)}")')

        if s.phase == "review":
            m = TOTAL_RE.search(ax)
            total = float(m.group(1).replace(",", "")) if m else None
            fno = s.current_fn
            if total is not None and (s.best is None or total < s.best["total"]):
                s.best = {"flight_no": fno, "total": total, "review_url": ctx.url}
                thought = f"True total for {fno} is ${total:.2f}: the new best."
            elif total is None:
                thought = f"Could not read the total for {fno}."
            else:
                thought = f"True total for {fno} is ${total:.2f}: not better than ${s.best['total']:.2f}."
            s.phase, s.back = "back", 1  # review -> details now, details -> results next step
            return self._d(thought + " Go back.", "go_back()")

        return self._d("Done.", "noop()")

    def _candidates(self, ax: str, task: dict) -> list[str]:
        out = []
        for fn, dep, stops, cabin, refund, fare, bag, tax in ITEM_RE.findall(ax):
            if task["nonstop"] and stops != "Nonstop":
                continue
            if task["depart_after"] and dep < task["depart_after"]:
                continue
            if cabin != task["cabin"]:
                continue
            if task["refundable"] and refund != "Refundable":
                continue
            fare_v = float(fare.replace(",", ""))
            if bag:  # fees are listed: compute the exact total
                bound = task["passengers"] * (fare_v + task["bags"] * float(bag.replace(",", ""))) * (1 + float(tax) / 100)
            else:  # fees hidden: optimistic lower bound
                bound = task["passengers"] * fare_v * (1 + MIN_TAX)
            if self.s.best is None or bound < self.s.best["total"]:
                out.append((bound, fn))
        return sorted(out)


def make_policy(name: str, model: str | None = None):
    if name == "scripted":
        return ScriptedPolicy()
    if name == "claude":
        return ClaudePolicy(model)
    if name in ("openai", "nous"):
        return OpenAIPolicy(*openai_compatible(name, model))
    raise ValueError(name)
