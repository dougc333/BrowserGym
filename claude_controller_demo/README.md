# Claude as an RL controller for BrowserGym

An interactive, educational demo of an LLM acting as the **policy** in a reinforcement-learning loop over real web tasks, and of how the trajectories it produces become **behavior-cloning** data.

- **Environment:** BrowserGym driving Chromium through Playwright (MiniWoB++ out of the box; any installed BrowserGym benchmark appears in the task list).
- **Observation:** the page's accessibility tree (AXTree), flattened to text, with every interactive node tagged by a `[bid]`.
- **Policy:** Claude, prompted with the observation and BrowserGym's action space, replies with a short thought and one action such as `click("13")`. A random baseline and a human policy are included for comparison.
- **Recorder:** every transition is saved; the front end turns episodes into SFT, (o, a, r, G) or preference datasets.
- **Front end:** React 19, Vite 8, TypeScript 7, Tailwind 4. One Zustand store, no prop drilling, and Zod validating every message, command and setting.

```
            ┌──────────── observation oₜ  (goal + AXTree text) ─────────────┐
            │                                                              ▼
   BrowserGym env  ◀── env.step(aₜ) ── action aₜ, e.g. click("13") ◀── policy π = Claude
   (Chromium)          → rₜ, done                    ▲
            │                                        │ optional human approval / correction
            └──── every (oₜ, aₜ, rₜ) ──▶ trajectory τ ──▶ runs/<id>.json ──▶ BC dataset (.jsonl)
```

## Quick start

Requires Python 3.10–3.12 (BrowserGym pins Playwright 1.44) and Node 20+.

```sh
cd claude_controller_demo
python -m venv .venv && source .venv/bin/activate
make install                    # BrowserGym core+miniwob, bridge deps, Chromium, npm packages
make miniwob                    # clones MiniWoB++ to ../miniwob-plusplus
export ANTHROPIC_API_KEY=sk-ant-...   # optional: without it, use the Random / Human policies
make dev                        # bridge on :8000, UI on http://localhost:5173
```

`make serve` builds the UI and serves everything from http://localhost:8000 instead.

Environment variables:

| Variable | Default | Purpose |
|---|---|---|
| `ANTHROPIC_API_KEY` | none | Enables the Claude policy |
| `CLAUDE_MODEL` | `claude-sonnet-5-5` | Default model (the UI can switch between Haiku, Sonnet, Opus) |
| `MINIWOB_URL` | set by the Makefile | `file://…/miniwob-plusplus/miniwob/html/miniwob/` |
| `CHROMIUM_EXECUTABLE` | Playwright's Chromium | Point at a different Chromium build |

On Python 3.13, Playwright 1.44 does not build. Install BrowserGym with `pip install --no-deps -e ../browsergym/core -e ../browsergym/miniwob`, then `pip install playwright gymnasium numpy pyparsing Pillow beautifulsoup4 "lxml<6"` and `playwright install chromium`.

## A guided tour

**Live Episode** — one episode, one step at a time.
1. Pick `miniwob.click-button`, press **env.reset()**. Chromium loads the task and the AXTree appears on the left.
2. Press **Propose**. Claude's tokens stream in; the reply is parsed into a thought and an action, and the action's target is highlighted in the tree. The *prompt* tab shows the exact text Claude received.
3. **Approve & step** sends the action to `env.step`. The loop diagram follows the phases: observe → think → act, and the transition lands in the trajectory.
4. Edit the action (type it, or click any `[bid]` in the tree) before stepping. That step is recorded as a **human correction**, with Claude's rejected proposal kept alongside it.
5. **Autopilot** runs the loop. With *Approve each action* on, it pauses for you every step (DAgger-style supervision).
6. *Show Chromium window* launches a visible browser on the bridge machine so you can watch the agent click. *Send screenshot to Claude* makes the observation multimodal.

**Benchmark Runner** — pick tasks and seeds, run every (task, seed) episode unsupervised, and see success rate per task. Run the same set with Claude and with Random to see why sparse rewards make learning from scratch hard. Every episode is saved, so a benchmark run is also a data-collection run.

**Trajectories → BC** — all saved episodes, a dataset builder and a live preview of one training row. Click any episode to replay it step by step in the Live view.

**Concepts** — the RL vocabulary mapped onto this system, and short notes on AXTree observations, in-context policies, filtered BC, distribution shift and DAgger.

## Dataset formats

| Format | One row per | Use it for |
|---|---|---|
| `sft_chat` | step | Fine-tuning a chat model: `system` + `user` = the exact prompt the policy saw, `assistant` = `<thinking>…</thinking><action>…</action>` |
| `obs_action` | step | Classic BC or a Decision Transformer: goal, AXTree, action, reward, return-to-go |
| `preference` | human correction | DPO: `chosen` = human action, `rejected` = Claude's proposal |

Filters: successful episodes only (filtered BC), drop steps whose action raised an error, include or exclude human labels; γ sets the discount for return-to-go. Raw episodes are in `runs/*.json`.

## How the code is laid out

```
server/
  protocol.py     Pydantic models for the wire protocol and records (mirror of web/src/shared/schemas.ts)
  env_worker.py   BrowserGym on one dedicated thread; AXTree flattening; the HighLevelActionSet
  policies.py     ClaudePolicy (streaming Messages API + action parsing), RandomPolicy, prompt builder
  app.py          FastAPI: WebSocket session state machine (reset/think/act/run/batch), REST for episodes
web/src/
  shared/schemas.ts  Zod: server events, client commands, settings, dataset rows
  shared/socket.ts   WebSocket client, Zod-validated in both directions
  shared/bc.ts       episodes → dataset rows, return-to-go
  store/lab.ts    the single Zustand store; components select slices, nothing is prop-drilled
  components/     LoopDiagram, ObservationPanel, PolicyPanel, Timeline, BenchmarkPanel, DatasetPanel, LearnPanel
```

Notes on a few design choices:

- **One env thread.** Playwright's sync API cannot run inside asyncio, and BrowserGym keeps a single global Playwright instance tied to the thread that created it. All env calls go through one process-wide worker thread; the server stays async around it.
- **Think and act are separate commands.** That split is what makes approval, correction and step-by-step teaching possible; autopilot just chains them.
- **The prompt is stored with each step**, including human-only steps, so SFT rows reproduce exactly what the policy conditioned on.
- **Zod as the state gate.** Settings changes pass `Settings.safeParse` before reaching the store, inbound events pass `ServerEvent.safeParse`, outbound commands pass `ClientCommand.safeParse`. Invalid data is reported in the status bar instead of corrupting state.

## Other benchmarks

The task list shows every registered `browsergym/*` environment. WebArena, WorkArena and the others need their own servers and credentials; follow their READMEs in `../browsergym/`, install the package, restart the bridge, and they appear in the list. Raise *Max steps* for long-horizon tasks.
