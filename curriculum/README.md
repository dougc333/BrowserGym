# RL task curriculum for MiniWoB, in text and vision

Builds a difficulty-aware curriculum for RL on BrowserGym's MiniWoB tasks
(GRPO, DAPO, GSPO and friends), and does it for three ways of seeing the page, so
you can see where computer vision is necessary, where it helps, and what it costs.

| Modality | Agent sees | Agent acts with |
|---|---|---|
| `axtree` | text accessibility tree | element ids: `click("13")` |
| `som` | screenshot with Set-of-Marks boxes and ids drawn on interactive elements | element ids |
| `pixels` | raw screenshot only | coordinates: `mouse_click(120, 85)`, `keyboard_type(...)` |

Why the modality matters: on `click-color` the accessibility tree is **empty** and
Set-of-Marks labels nothing (the colored boxes aren't interactive elements), so only
`pixels` can solve it. On `visual-addition` the blocks to count exist only in the
pixels. On form tasks like `login-user`, text is the easy route and pixel-level
grounding is the hard part.

## The pipeline

```
profile.py     k rollouts per (modality, task, seed)  ->  pass rate p per item
curriculum.py  bucket by p, weight by p(1-p), retire / freeze / requeue, DAPO filling
simulate.py    synthetic learner: does the curriculum beat uniform sampling, and when?
```

Buckets, from the policy's own pass rate on each item:

| Bucket | p | What RL gets from it |
|---|---|---|
| unsolvable | 0 | nothing: every rollout fails, all advantages are zero |
| hard | (0, 0.25) | rare signal, concentrated on one lucky success |
| **zpd** | [0.25, 0.75] | the most signal per rollout (variance p(1-p) peaks at 0.5) |
| easy | (0.75, 1) | signal concentrated on one unlucky failure |
| solved | 1 | nothing |

Unsolvable and solved items are **parked**, not deleted: training reports every group's
rewards back, items that keep passing are retired, items that keep failing are frozen,
and parked items are requeued periodically, since policies both regress and improve.

## Run it

Setup (once), from `~/BrowserGym`, in the same environment as `memory_agent`:

```sh
uv venv .venv --python 3.12
source .venv/bin/activate
uv pip install -e browsergym/core -e browsergym/miniwob anthropic
playwright install chromium
```

Each session:

```sh
cd ~/BrowserGym/curriculum
source ../.venv/bin/activate
export MINIWOB_URL="file://$HOME/BrowserGym/miniwob-plusplus/miniwob/html/miniwob/"
```

Simulation (seconds, no browser or API):

```sh
python simulate.py
python simulate.py --offset 3 --spread 3 --budget 60000
```

Profile a policy and build the curriculum (uses `ANTHROPIC_API_KEY` from your shell):

```sh
python profile.py --name smoke --llm scripted --tasks quick --seeds 2 --k 2
python profile.py --name haiku --llm anthropic:claude-haiku-5-5 --tasks mixed --workers 4
python curriculum.py build curricula/haiku
python curriculum.py sample curricula/haiku -n 8
```

`build` writes `curriculum.json` (the sampler state) and `report.md`: pass rate per task
and modality with notes such as *vision necessary*, *vision helps*, or *text easier*, a
per-modality summary with bucket counts and tokens per episode, and the items the
sampler will draw most.

| `profile.py` option | Default | Meaning |
|---|---|---|
| `--tasks` | `mixed` | `quick`, `text`, `visual`, `mixed`, or a comma list |
| `--modalities` | all three | e.g. `axtree,pixels` |
| `--seeds` / `--k` | 4 / 4 | instances per task / rollouts per instance |
| `--max-steps` | 12 | step budget per episode |
| `--llm` | — | `anthropic:<model>`, `openai:<model>@<url>` (vision needs a VLM, e.g. `qwen2.5vl:7b` in Ollama), `scripted` |

`--tasks mixed` is 11 tasks x 3 modalities x 4 seeds x 4 rollouts = 528 episodes.
Start with `--tasks quick` to gauge time and cost. Screenshots are 498 x 321 PNGs, so
vision steps cost a few hundred extra input tokens each.

`build` options: `--level instance` (one item per task and seed; good when your
trainer uses fixed seeds) or `--level task` (one item per task, fresh seed on every
draw; good for open-ended training), and `--strategy zpd | staged | uniform`.

## Plugging into a trainer

```python
from curriculum import Curriculum, dapo_fill_batch

cur = Curriculum.load("curricula/haiku/curriculum.json")

def rollout(item, seed, G):
    # run G episodes of browsergym/miniwob.<item.task> with this seed and item.modality,
    # using your current policy; return their rewards (0/1)
    ...

for step in range(num_steps):
    groups, stats = dapo_fill_batch(cur, rollout, batch_size=64, group_size=8)
    trainer.update(groups)            # GRPO/DAPO/GSPO update on mixed-reward groups only
    log(stats | cur.stats())          # keep_rate, filtered_all_pass/fail, buckets, mean p
    if step % 50 == 0:
        cur.save("curricula/haiku/curriculum.json")
```

`dapo_fill_batch` reports *every* group back to the curriculum, including the ones it
filters out, so pass-rate estimates stay current. Watch `keep_rate`: when it drops,
most draws are already solved, which is the cue to add harder tasks.

## What the simulation shows

`simulate.py` gives a synthetic learner a skill per task; it improves only from
groups with mixed rewards, as in GRPO/DAPO. Same rollout budget for every strategy.
Results from one run (5 repeats each):

- **Balanced pool** (defaults): all strategies end near 88%. Only about a third of
  groups carry no signal, so a curriculum has little to fix.
- **Lopsided pool** (`--offset 3 --spread 3 --budget 60000`): uniform sampling wastes
  54% of groups and reaches 80% success in 2 of 5 runs (final ~78%). The `zpd` and
  `staged` curricula waste ~35%, reach 80% in 5 of 5 runs, and finish ~91%.

So a curriculum pays off when most tasks are too easy or too hard for the current
policy, which is the usual situation once training gets going. This is an
illustration under the simulation's assumptions, not a result about real RL.

## Files

| File | Purpose |
|---|---|
| `policy.py` | observation modes (axtree / som / pixels), action spaces, one-episode loop |
| `llm.py` | multimodal backends: Anthropic, OpenAI-compatible, scripted |
| `profile.py` | k-rollout difficulty profiling, parallel and resumable |
| `curriculum.py` | buckets, sampler, retire/freeze/requeue, staged unlocking, `dapo_fill_batch`, report |
| `simulate.py` | synthetic learner comparing uniform sampling with curricula |
| `test_curriculum.py` | unit tests (no browser or model) |
