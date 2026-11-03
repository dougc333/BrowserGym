# Memory ablation on MiniWoB

Does memory make a web agent better, and by how much? This runs **one agent with four
memory configurations** on the same MiniWoB tasks and seeds, then measures the
difference with paired statistics.

The four configurations follow the memory types from the CoALA framework (CS329Z lecture 7):

| Condition | What the agent sees at each step | Memory type added |
|---|---|---|
| `none` | goal + current AXTree | nothing; every step starts from scratch |
| `working` | + this episode's actions, their targets, and errors | **working** memory (context) |
| `episodic` | + the 2 most similar past *successful* episodes of this task | **episodic** memory, retrieved from disk |
| `procedural` | + a workflow induced from past successes (AWM) and lessons from past failures (Reflexion) | **procedural** memory |

Model, prompt skeleton, action space and step budget are identical across conditions,
so differences come from memory alone.

```
TRAIN seeds 0..T-1 (in order)              TEST seeds 1000..1000+N-1
episode → write to memory/<cond>/<task>/   memory frozen, read only
          episodes.jsonl   (episodic)      → success rate, steps, tokens
          workflow.md      (AWM, after success)
          reflections.jsonl (Reflexion, after failure)
```

Long-term memory is stored as files (the lecture's filesystem pattern), one folder per
condition and task. Past episodes are stored with **element descriptions**
(`click("13") -> button 'Submit'`) instead of bare ids, because ids change with every seed.

## Run it

One-time setup, in its own environment (Python 3.10–3.12; BrowserGym pins Playwright 1.44):

```sh
cd ~/BrowserGym
uv venv .venv --python 3.12
source .venv/bin/activate
uv pip install -e browsergym/core -e browsergym/miniwob anthropic
playwright install chromium
```

Each session (the MiniWoB files are in `miniwob-plusplus/`; `make setup-miniwob` clones them):

```sh
cd ~/BrowserGym/memory_agent
source ../.venv/bin/activate
export MINIWOB_URL="file://$HOME/BrowserGym/miniwob-plusplus/miniwob/html/miniwob/"
```

Plumbing check with no model or API key (its numbers are meaningless):

```sh
python run.py --exp smoke --llm scripted --tasks quick --train 2 --test 2
```

With Claude (put your real key in place of `sk-ant-...`):

```sh
export ANTHROPIC_API_KEY=sk-ant-...
python run.py --exp haiku-quick --llm anthropic:claude-haiku-5-5 --tasks quick --train 3 --test 5 --workers 4
python run.py --exp haiku --llm anthropic:claude-haiku-5-5 --tasks memory12 --workers 4
```

Or with a local model through Ollama or vLLM (any OpenAI-compatible endpoint):

```sh
ollama pull qwen2.5:7b-instruct
python run.py --exp qwen7b --llm "openai:qwen2.5:7b-instruct@http://localhost:11434/v1" --workers 2
```

Then build the report (writes `report.md`, `report.html` and `summary.json`):

```sh
python report.py results/haiku
```

`run.py` checks the setup before starting (packages, `MINIWOB_URL`, API key) and stops
with the fix if something is missing.

Runs are resumable: re-run the same command and finished episodes are skipped.

| Option | Default | Meaning |
|---|---|---|
| `--tasks` | `memory12` | `quick` (4 tasks), `memory12`, or a comma-separated list of MiniWoB tasks |
| `--train` / `--test` | 5 / 10 | seeds per task for building memory / for measuring |
| `--conditions` | all four | e.g. `none,working` |
| `--max-steps` | 15 | step budget per episode |
| `--k-episodes` | 2 | past episodes retrieved for episodic/procedural |
| `--workers` | 2 | parallel (condition, task) jobs, one Chromium each |
| `--save-prompts` | off | store every prompt in the results (large) |

**Size and cost.** `memory12` with the defaults is 12 tasks × 15 seeds × 4 conditions =
720 episodes. At a few steps per episode and roughly 1.5–3k input tokens per step,
expect several million input tokens; the report prints exact token counts per condition.
Start with `--tasks quick --train 3 --test 5` to gauge it.

## What the report measures

- **Held-out success rate** per condition, with a Wilson 95% interval. This is the
  headline number.
- **Paired differences.** Every condition ran the same (task, seed) pairs, so each
  comparison counts the pairs where only A succeeded vs only B succeeded. It reports the
  difference in percentage points, a 95% bootstrap interval that resamples *tasks*
  (episodes of one task are correlated), and an exact McNemar p-value.
- **Mechanism metrics:** repeated identical actions per episode (the signature of
  missing working memory), action errors, steps to success.
- **Learning curve** over training episodes: with long-term memory, success should rise
  as memory accumulates; without it, the curve should be flat.
- **Cost:** acting tokens per episode, plus tokens spent maintaining memory (workflow
  induction and reflections).

## What to expect, and how to read it

- `none` vs `working` isolates **working memory**. Expect the biggest gap on tasks whose
  progress is *not visible on the page*: `guess-number` (earlier guesses), `search-engine`
  (which page you are on), and any task where the agent's earlier mistakes aren't
  reflected in the AXTree. On tasks that show their state (checked boxes, filled fields),
  a stateless agent can partly recover from the page itself, which is a finding too.
- `working` vs `episodic` / `procedural` isolates **long-term memory**. Expect gains on
  tasks with a non-obvious mechanic: date pickers (`choose-date`), autocomplete
  (`use-autocomplete`, `book-flight`), hidden tabs (`click-tab-2`), date formats
  (`enter-date`).
- **Stronger models leave less headroom.** A frontier model may already solve most
  tasks with working memory alone; the effect of long-term memory is usually clearer
  with a smaller model (Haiku, or a 7B local model).
- Claude runs are not exactly repeatable (the current Anthropic SDK takes no
  temperature). The paired design and the intervals account for that noise; for tighter
  intervals, add test seeds rather than re-running.
- Memory can also **hurt** (lecture: "using memory correctly"). A misleading workflow or
  a wrong lesson appears as losses in the paired comparison; look at
  `results/<exp>/memory/procedural/<task>/` to see what the agent learned.

## Files

| File | Purpose |
|---|---|
| `agent.py` | the agent, the four conditions, prompt construction, one-episode loop |
| `memory.py` | filesystem memory: episode store, retrieval, AWM workflow induction, Reflexion |
| `llm.py` | backends: `anthropic:<model>`, `openai:<model>@<url>`, `scripted` |
| `run.py` | experiment runner (train → test, parallel, resumable) |
| `report.py` | statistics and the Markdown/HTML report (standard library only) |
| `test_memory_agent.py` | unit tests that need no browser or model |
