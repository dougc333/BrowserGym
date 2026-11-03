# Travel Search Gym

A sandbox flight-booking website, a BrowserGym task on top of it, an agentic search
loop, and a dashboard that shows every trajectory step by step against the ground truth.

The task: *book the cheapest flight that meets every requirement* (airports, dates,
nonstop, departure time, cabin, travelers, bags, refundability), where "cheapest"
means the full price including taxes and bag fees. Each episode's flights are
generated from a seed into SQLite, so the correct answer is known exactly and the
reward is computed from the database, never from what the page or the agent says.

```
            data/travel.db (episodes, flights, bookings, runs, steps)
                 ▲ reads/writes                         ▲ reads ground truth
  website.py :8100  ◀── browser (Chromium) ──┐
                                             ├─ agent loop ── app.py :8000 ──▶ React dashboard
  mcp_server.py (stdio) ◀── MCP tool calls ──┘   observe → decide → act → record
```

The website and the dashboard API run on different ports on purpose: the agent's
browser only visits the website, which has no endpoint that reveals the answer.

## Is the search a loop?

Yes. Each run is the same loop until the agent books, reports the task infeasible,
or hits the step limit:

1. **Observe:** the page's accessibility tree (every result has a descriptive label).
2. **Decide:** the policy returns a thought, updated notes, and one action.
3. **Act:** `env.step(action)` in Chromium via BrowserGym.
4. **Record:** URL, page type, screenshot, thought, notes, action, error, and tokens go to SQLite and stream to the dashboard.

Long searches outgrow the context window, so the LLM agent keeps **notes** (its working
memory: searches done, candidates, true totals, best so far) that it rewrites every
step. The prompt carries only the notes, the last 12 actions, and the current page.

## Difficulty knobs

| Knob | Effect |
|---|---|
| `n_dates`, `n_origins` | allowed departure days × nearby airports = number of searches |
| `pages`, `page_size` | result pages per search |
| `hidden_fees` | listed fares exclude bags and taxes; the true total is only on the review page |
| `sort_available` | whether "Lowest fare" sorting exists (it sorts by listed fare, which misleads when fees are hidden) |
| `passengers`, `bags` | total scales with travelers and checked bags |
| `require_nonstop`, `depart_after`, `require_refundable` | constraints |
| `decoys` | cheaper flights that each break exactly one requirement; plus a "fee trap": the lowest listed fare among valid flights with a higher true total |

Presets: `easy` (1 search, about 11 steps for the scripted agent), `medium` (3 searches,
about 35), `hard` (15 searches over 45 pages, about 190), `extreme` (21 searches, refundable required).

## Policies

| Policy | What it is |
|---|---|
| `scripted` | Systematic searcher with no model. Enumerates every allowed airport and date, pages through results, opens only candidates that could still beat the best total, reads the true total on the review page, books the best. Uses only what the site shows. Optimal on every preset tested; a free teacher for behavior cloning. |
| `claude` | Claude with the prompt in `server/policies.py` (needs `ANTHROPIC_API_KEY`). |
| `openai` | Same prompt via any OpenAI-compatible server, e.g. `qwen2.5:7b-instruct@http://localhost:11434/v1`. |
| `mcp-scripted` | The scripted searcher, calling MCP tools instead of clicking. |
| `mcp-claude` | Claude with the MCP server's tools, in a normal tool-use conversation. |

## The same task over MCP

`server/mcp_server.py` exposes the travel site as an MCP server (stdio) with six tools:
`get_trip_request`, `list_airports`, `search_flights` (paginated, optional sort),
`get_flight_details`, `price_quote` (the true total), and `book_flight` (first booking is
final). They share one implementation (`travel_api.py`) with exactly the website's rules,
so hidden fees stay hidden and the same decoys and fee trap apply. The checker is unchanged.

For each MCP run the runner starts the server for that episode, lists its tools and
loops: the policy chooses tool calls, each call goes over MCP, and each call is recorded
as one step. Steps are stored with `mcp://` URLs shaped like the website's
(`.../results?from=DCA&date=...&page=2`, `.../review?flight=...`), so the coverage grid,
fares table and diagnosis work the same for browser and MCP runs. In the dashboard, MCP
steps show the tool call and the JSON result instead of a screenshot.

`mcp-claude` keeps the full conversation, but tool results older than the last 12 are
replaced with a stub to bound the context (client-side context editing), and the agent
is told to write a running "BEST so far" line after every quote, since its own messages
are never cleared.

**Browser vs MCP on identical episodes (scripted agents, same seeds):**

| Episode | Browser steps | MCP steps | Browser time | MCP time | Pages and quotes | Result |
|---|---:|---:|---:|---:|---|---|
| medium, seed 2 | 32 | 10 | 26 s | 1 s | same | both optimal |
| hard, seed 4 | 201 | 63 | 157 s | 1 s | same (45 pages, 16 quotes) | both optimal |
| hard, seed 11 | 186 | 60 | 145 s | 1 s | same (45 pages, 13 quotes) | both optimal |
| extreme, seed 5 | 225 | 93 | 176 s | 1 s | same (84 pages, 7 quotes) | both optimal |

Same search, same decisions; the browser version spends its extra steps on form filling,
clicking into details and going back. That gap (plus visual grounding) is what a browser
agent must learn and an API agent doesn't. Comparing `claude` with `mcp-claude` on the same
seeds shows how much of a model's difficulty is the web interface versus the search itself.

## Nous Research (Hermes) models

Policies `nous` (browser) and `mcp-nous` (MCP) run Hermes models through the Nous Portal's
OpenAI-compatible API; `mcp-openai` runs the same MCP loop against any OpenAI-compatible
server given as `model@url` (for example Hermes in Ollama: `hermes3:8b@http://localhost:11434/v1`).

```sh
export NOUS_API_KEY=...                       # from your Nous Portal account
export NOUS_MODEL=Hermes-4-70B                # optional; or pick per run in the dashboard
export NOUS_BASE_URL=https://inference-api.nousresearch.com/v1   # optional; check your portal's API docs
```

Restart `uvicorn app:app` after setting them; the top bar shows "Nous Hermes" when the key is
found. The MCP loop uses Hermes' native function-calling format: tool schemas inside `<tools>`
in the system prompt, calls as `<tool_call>{"name": ..., "arguments": ...}</tool_call>`, and
results returned in `<tool_response>`. That works whether or not a server supports the OpenAI
`tools` parameter. Malformed calls get a correction message instead of crashing the run, and
`<think>` blocks are kept as the step's thought.

The model ids and base URL above are defaults I could not verify against the live API; if a
call fails, list your account's models with
`curl -H "Authorization: Bearer $NOUS_API_KEY" $NOUS_BASE_URL/models` and set `NOUS_MODEL`.

**Use the server from Claude Code** (or any MCP client) to try the task by hand:

```sh
claude mcp add travel-gym -- python ~/BrowserGym/travel_gym/server/mcp_server.py --new hard --seed 7
```

Then ask Claude: "Call get_trip_request and complete the request." The server prints the
episode id to stderr; check the result with
`python ~/BrowserGym/travel_gym/server/mcp_server.py --check <episode id>`.

## The dashboard

- **New run:** presets, every knob, seed, policy and model, max steps, optional visible Chromium.
- **Run header:** the goal, status, booked vs optimal total, overpay, tokens, a per-constraint checklist, and a **trajectory strip** with one tick per step colored by page type (search, results, details, review), clickable.
- **Trajectory:** every step's action and thought, the screenshot the agent saw, its notes at that point, errors, and the exact accessibility tree it read.
- **Search coverage:** a grid of allowed airports × dates showing which result pages were explored, the best valid total in each cell, and a ★ on the cell holding the optimum. A diagnosis says whether the agent never saw the optimum, saw it but didn't open it, opened it but never checked its true total, or checked it and booked something else.
- **Fares vs truth:** the ten cheapest valid flights and every decoy, with listed fare vs true total, the constraint each decoy breaks, and whether the agent saw, opened, checked, or booked it (click "seen tN" to jump to that step).

## Run it

From `~/BrowserGym`, in the same environment as `memory_agent` and `curriculum`:

Install once:

```sh
cd ~/BrowserGym
source .venv/bin/activate
uv pip install -r travel_gym/server/requirements.txt
cd travel_gym/web
npm install
```

Build the dashboard and start everything:

```sh
cd ~/BrowserGym/travel_gym/web
npm run build
cd ../server
uvicorn app:app --port 8000
```

Open http://localhost:8000. The API starts the sandbox website on :8100 itself. For
front-end hot reload, run `npm run dev` in `web/` in a second terminal while the API
runs, and open http://localhost:5174. Claude needs `ANTHROPIC_API_KEY` in the shell
(your `.zshrc` works). The optional `Makefile` wraps these commands (`make install`,
`make serve`, `make dev`, `make test`).

Use the task from code:

```python
import gymnasium as gym, sys
sys.path.insert(0, "travel_gym/server")
import task  # registers browsergym/travel.search; the website must be running on :8100
env = gym.make("browsergym/travel.search", task_kwargs={"difficulty": "hard"})
obs, info = env.reset(seed=7)
```

## Tests

```sh
cd travel_gym/server && python -m pytest -q test_travel.py
```

Covers reproducible generation, website/MCP parity (same results, same order, fees hidden), an end-to-end MCP episode, a unique optimum, decoys that are cheaper but invalid,
the fee trap, exact scoring, and reward-hacking probes: booking a cheaper decoy,
changing travelers or bags to lower the price, booking twice, a fake confirmation page,
a flight from another episode, and checking that no page reveals the answer.

## Using the trajectories

Every step is in `data/travel.db` (`steps` table) with the exact accessibility tree the
policy read and the action it took, plus screenshots in `data/screens/`. Scripted runs
are optimal long-horizon demonstrations; Claude runs give natural variation and failures.
Filter by `runs.success` for behavior cloning, or pair a failed step with the scripted
agent's action in the same state for preference data. The `curriculum/` sampler can
choose difficulty settings near a 50% pass rate for RL.
