"""
Agentic travel search over MCP: the same task as the browser version, but the agent
calls the travel-gym MCP server's tools instead of clicking a website.

    start mcp_server.py for the episode (stdio) -> list its tools ->
    loop { policy picks tool call(s) -> call over MCP -> record the step } -> check

Policies
  mcp-claude    Claude with the MCP tools exposed as Anthropic tool definitions, in an
                ordinary multi-turn tool-use conversation. Long searches produce many
                tool results, so results older than the last KEEP are cleared from the
                context (client-side context editing); the agent is told to keep a
                running "BEST so far" line in its own messages, which are never cleared.
  mcp-nous      Nous Research Hermes models (Nous Portal API, NOUS_API_KEY), and
  mcp-openai    any OpenAI-compatible server (model@url, e.g. Hermes in Ollama). Both use
                Hermes' native function-calling format: tools listed in <tools>, calls as
                <tool_call>{json}</tool_call>, results as <tool_response>. That works whether
                or not the server supports the OpenAI `tools` parameter.
  mcp-scripted  The systematic searcher from the browser version, speaking MCP: search
                every allowed airport, date and page, quote candidates that could still
                win, book the best. A model-free baseline and teacher.

Each tool call is recorded as one step, with a synthetic mcp:// URL shaped like the
website's (".../results?from=...&page=2", ".../review?flight=..."), so the dashboard's
coverage analysis treats browser and MCP runs identically.
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable
from urllib.parse import urlencode

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from policies import DEFAULT_MODEL

HERE = Path(__file__).resolve().parent
KEEP = 12  # most recent tool results kept verbatim in Claude's context
MAX_OBS = 20_000

SYSTEM = """\
You are a travel-booking agent with tools for a flight booking system. Complete the traveler's
request exactly: book the cheapest flight that satisfies EVERY requirement, where cheapest means
the TRUE total including taxes and bag fees for the right number of travelers and bags.

How to work:
1. Call get_trip_request first. If you need airport codes, call list_airports.
2. Search every allowed origin airport and every allowed date. Results are paginated: request
   every page (page=2, 3, ... until page == pages).
3. Ignore flights that break a requirement (stops, departure time, cabin, refundability...).
4. Listed fares may exclude taxes and bag fees. For candidates that could be cheapest, call
   price_quote with the exact travelers and bags to get the true total.
5. Older tool results are cleared from your context to save space. After every quote, write one
   line: "BEST so far: flight_id=<id> <flight_no> $<total>". Keep track of what you've searched.
6. book_flight is final and allowed once. Book only when every search is done, then stop and
   summarize what you booked and why."""


@dataclass
class StepRecord:
    url: str
    thought: str
    action: str
    observation: str
    error: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: int = 0


Record = Callable[[StepRecord], None]


def synthetic_url(ep: str, name: str, args: dict) -> str:
    base = f"mcp://travel-gym/e/{ep}"
    if name == "search_flights":
        q = {"from": str(args.get("origin", "")).upper(), "to": str(args.get("destination", "")).upper(),
             "date": args.get("date", ""), "travelers": args.get("travelers", 1),
             "page": args.get("page", 1), "sort": args.get("sort", "recommended")}
        return f"{base}/results?{urlencode(q)}"
    if name == "get_flight_details":
        return f"{base}/flight/{args.get('flight_id')}?{urlencode({'travelers': args.get('travelers', 1)})}"
    if name == "price_quote":
        return f"{base}/review?" + urlencode({"flight": args.get("flight_id"), "travelers": args.get("travelers"),
                                             "bags": args.get("bags_per_traveler")})
    if name == "book_flight":
        return f"{base}/confirmation/" + str(args.get("flight_id"))
    return f"{base}/{name}"


def _result_text(res) -> tuple[str, bool]:
    text = "\n".join(getattr(c, "text", "") for c in (res.content or []))
    return text[:MAX_OBS], bool(getattr(res, "is_error", False))


def server_params(ep: str) -> StdioServerParameters:
    env = {k: v for k, v in os.environ.items() if k in ("PATH", "HOME", "TRAVEL_GYM_DATA", "PYTHONPATH")}
    return StdioServerParameters(command=sys.executable, args=[str(HERE / "mcp_server.py"), "--episode", ep],
                                 env=env, cwd=str(HERE))


async def run_mcp(policy: str, model: str | None, ep: dict, max_steps: int, record: Record,
                  should_stop: Callable[[], bool]) -> str:
    """Runs one episode; returns 'done' or 'stopped'."""
    async with stdio_client(server_params(ep["id"])) as (r, w):
        async with ClientSession(r, w) as session:
            await session.initialize()
            if policy == "mcp-scripted":
                return await _scripted(session, ep, max_steps, record, should_stop)
            if policy in ("mcp-nous", "mcp-openai"):
                return await _hermes(session, ep, policy, model, max_steps, record, should_stop)
            return await _claude(session, ep, model or DEFAULT_MODEL, max_steps, record, should_stop)


# ------------------------------------------------------------------ Claude tool-use loop
def _clear_old_results(messages: list[dict]) -> None:
    """Replace all but the last KEEP tool results with a stub (client-side context editing)."""
    seen = 0
    for m in reversed(messages):
        if m["role"] != "user" or not isinstance(m["content"], list):
            continue
        for block in reversed(m["content"]):
            if block.get("type") == "tool_result":
                seen += 1
                if seen > KEEP and block.get("content") != "[older result cleared to save context]":
                    block["content"] = "[older result cleared to save context]"


async def _claude(session, ep: dict, model: str, max_steps: int, record: Record, should_stop) -> str:
    from anthropic import Anthropic

    client = Anthropic()
    tools = [{"name": t.name, "description": t.description or "", "input_schema": t.input_schema}
             for t in (await session.list_tools()).tools]
    messages: list[dict] = [{"role": "user", "content": "Please handle my trip request (call get_trip_request)."}]
    steps = 0
    while steps < max_steps:
        if should_stop():
            return "stopped"
        _clear_old_results(messages)
        t0 = time.time()
        for attempt in range(6):
            try:
                resp = client.messages.create(model=model, max_tokens=2000, system=SYSTEM, tools=tools,
                                              messages=messages)
                break
            except Exception as e:
                if attempt == 5 or not any(s in str(e).lower() for s in ("429", "overload", "529", "500", "timeout")):
                    raise
                time.sleep(2 ** attempt)
        latency = int((time.time() - t0) * 1000)
        messages.append({"role": "assistant", "content": [b.model_dump(exclude_none=True) for b in resp.content]})
        thought = " ".join(b.text for b in resp.content if b.type == "text").strip()
        calls = [b for b in resp.content if b.type == "tool_use"]
        if not calls:  # the agent is finished (or gave up)
            record(StepRecord(url=f"mcp://travel-gym/e/{ep['id']}/final", thought=thought, action="(end of turn)",
                              observation="", input_tokens=resp.usage.input_tokens,
                              output_tokens=resp.usage.output_tokens, latency_ms=latency))
            return "done"
        results = []
        for i, call in enumerate(calls):
            res = await session.call_tool(call.name, call.input)
            text, is_err = _result_text(res)
            record(StepRecord(
                url=synthetic_url(ep["id"], call.name, call.input), thought=thought if i == 0 else "",
                action=f"{call.name}({json.dumps(call.input)})", observation=text, error=text if is_err else "",
                input_tokens=resp.usage.input_tokens if i == 0 else 0,
                output_tokens=resp.usage.output_tokens if i == 0 else 0, latency_ms=latency if i == 0 else 0))
            results.append({"type": "tool_result", "tool_use_id": call.id, "content": text, "is_error": is_err})
            steps += 1
        messages.append({"role": "user", "content": results})
    return "done"


# ------------------------------------------------------------------ Hermes / OpenAI-compatible loop
HERMES_SYSTEM = SYSTEM + """

You are a function-calling AI model. You are provided with function signatures within <tools></tools>
XML tags. You may call one or more functions to assist with the user query. Don't make assumptions
about what values to plug into functions. Here are the available tools:
<tools>
{tools}
</tools>
For each function call return a json object with function name and arguments within
<tool_call></tool_call> XML tags as follows:
<tool_call>
{{"name": <function-name>, "arguments": <args-dict>}}
</tool_call>
Function results come back inside <tool_response></tool_response> tags. When you are completely
finished, reply without any <tool_call>."""

TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.S)
CLEARED = "[older result cleared to save context]"


def _parse_tool_calls(text: str) -> tuple[str, list[dict], list[str]]:
    """Hermes format: thought text plus zero or more <tool_call>{json}</tool_call> blocks."""
    calls, bad = [], []
    for raw in TOOL_CALL_RE.findall(text):
        try:
            c = json.loads(raw)
            calls.append({"name": c["name"], "arguments": c.get("arguments") or {}})
        except (json.JSONDecodeError, KeyError, TypeError):
            bad.append(raw[:200])
    thought = TOOL_CALL_RE.sub("", text)
    thought = re.sub(r"</?think>", "", thought).strip()
    return thought, calls, bad


def _chat(base_url: str, api_key: str, model: str, messages: list[dict]) -> tuple[str, int, int]:
    import urllib.error
    import urllib.request

    body = json.dumps({"model": model, "max_tokens": 2000, "messages": messages}).encode()
    for attempt in range(6):
        req = urllib.request.Request(f"{base_url.rstrip('/')}/chat/completions", data=body,
                                     headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"})
        try:
            with urllib.request.urlopen(req, timeout=300) as r:
                d = json.loads(r.read())
            u = d.get("usage") or {}
            return d["choices"][0]["message"].get("content") or "", u.get("prompt_tokens", 0), u.get("completion_tokens", 0)
        except urllib.error.HTTPError as e:
            detail = e.read()[:300]
            if attempt == 5 or e.code not in (429, 500, 502, 503, 529):
                raise RuntimeError(f"{e.code} from {base_url}: {detail!r}") from e
        except (urllib.error.URLError, TimeoutError):
            if attempt == 5:
                raise
        time.sleep(2 ** attempt)
    raise RuntimeError("unreachable")


async def _hermes(session, ep: dict, policy: str, model: str | None, max_steps: int, record: Record,
                  should_stop) -> str:
    from policies import openai_compatible

    model, base_url, api_key = openai_compatible(policy, model)
    tools = [{"type": "function", "function": {"name": t.name, "description": t.description or "",
                                               "parameters": t.input_schema}}
             for t in (await session.list_tools()).tools]
    messages: list[dict] = [
        {"role": "system", "content": HERMES_SYSTEM.format(tools="\n".join(json.dumps(t) for t in tools))},
        {"role": "user", "content": "Please handle my trip request (call get_trip_request)."},
    ]
    steps = 0
    while steps < max_steps:
        if should_stop():
            return "stopped"
        # client-side context editing: keep only the last KEEP tool responses verbatim
        resp_idx = [i for i, m in enumerate(messages) if m["role"] == "user" and m["content"].startswith("<tool_response>")]
        for i in resp_idx[:-KEEP]:
            messages[i]["content"] = f"<tool_response>{CLEARED}</tool_response>"
        t0 = time.time()
        text, tin, tout = _chat(base_url, api_key, model, messages)
        latency = int((time.time() - t0) * 1000)
        messages.append({"role": "assistant", "content": text})
        thought, calls, bad = _parse_tool_calls(text)
        if bad:  # malformed call: tell the model and let it retry (counts as a step)
            msg = "Error: could not parse tool call JSON. Use exactly <tool_call>{\"name\": ..., \"arguments\": {...}}</tool_call>."
            record(StepRecord(url=f"mcp://travel-gym/e/{ep['id']}/parse-error", thought=thought, action=bad[0],
                              observation=msg, error=msg, input_tokens=tin, output_tokens=tout, latency_ms=latency))
            messages.append({"role": "user", "content": f"<tool_response>{msg}</tool_response>"})
            steps += 1
            continue
        if not calls:
            record(StepRecord(url=f"mcp://travel-gym/e/{ep['id']}/final", thought=thought, action="(end of turn)",
                              observation="", input_tokens=tin, output_tokens=tout, latency_ms=latency))
            return "done"
        outputs = []
        for i, c in enumerate(calls):
            res = await session.call_tool(c["name"], c["arguments"])
            out, is_err = _result_text(res)
            record(StepRecord(url=synthetic_url(ep["id"], c["name"], c["arguments"]), thought=thought if i == 0 else "",
                              action=f"{c['name']}({json.dumps(c['arguments'])})", observation=out,
                              error=out if is_err else "", input_tokens=tin if i == 0 else 0,
                              output_tokens=tout if i == 0 else 0, latency_ms=latency if i == 0 else 0))
            outputs.append(json.dumps({"name": c["name"], "content": out}))
            steps += 1
        messages.append({"role": "user", "content": "<tool_response>\n" + "\n".join(outputs) + "\n</tool_response>"})
    return "done"


# ------------------------------------------------------------------ scripted MCP searcher
async def _scripted(session, ep: dict, max_steps: int, record: Record, should_stop) -> str:
    task, steps = ep["task"], 0
    best: dict | None = None

    async def call(name: str, args: dict, thought: str) -> dict | None:
        nonlocal steps
        res = await session.call_tool(name, args)
        text, is_err = _result_text(res)
        record(StepRecord(url=synthetic_url(ep["id"], name, args), thought=thought,
                          action=f"{name}({json.dumps(args)})", observation=text, error=text if is_err else ""))
        steps += 1
        return None if is_err else json.loads(text) if text.startswith(("{", "[")) else {"text": text}

    await call("get_trip_request", {}, "Read the traveler's requirements.")
    pax, bags = task["passengers"], task["bags"]
    for origin in task["origins"]:
        for day in task["dates"]:
            page, pages = 1, 1
            while page <= pages:
                if should_stop() or steps >= max_steps:
                    return "stopped" if should_stop() else "done"
                res = await call("search_flights", {"origin": origin, "destination": task["dest"], "date": day,
                                                    "travelers": pax, "page": page},
                                 f"Search {origin} on {day}, page {page}.")
                if not res:
                    break
                pages = res.get("pages", 0)
                for f in sorted(res.get("flights", []), key=lambda f: f["fare_per_person"]):
                    if task["nonstop"] and f["stops"] != "nonstop":
                        continue
                    if task["depart_after"] and f["depart"] < task["depart_after"]:
                        continue
                    if f["cabin"] != task["cabin"] or (task["refundable"] and not f["refundable"]):
                        continue
                    if "bag_fee_per_bag" in f:
                        bound = pax * (f["fare_per_person"] + bags * f["bag_fee_per_bag"]) * (1 + f["tax_rate_percent"] / 100)
                    else:
                        bound = pax * f["fare_per_person"] * 1.055  # optimistic: lowest plausible tax, no bag fee
                    if best and bound >= best["total"]:
                        continue
                    q = await call("price_quote", {"flight_id": f["flight_id"], "travelers": pax,
                                                   "bags_per_traveler": bags},
                                   f"{f['flight_no']} meets every visible requirement and could beat the best; get its true total.")
                    if q and (best is None or q["total_price"] < best["total"]):
                        best = {"flight_id": f["flight_id"], "flight_no": f["flight_no"], "total": q["total_price"]}
                page += 1
    if best is None:
        return "done"
    await call("book_flight", {"flight_id": best["flight_id"], "travelers": pax, "bags_per_traveler": bags},
               f"Every search is done; book the cheapest valid option, {best['flight_no']} at ${best['total']:.2f}.")
    return "done"
