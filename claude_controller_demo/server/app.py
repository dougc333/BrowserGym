"""
Claude-as-controller bridge for BrowserGym.

Runs the RL interaction loop and streams every phase to the React front end
over a WebSocket, while recording trajectories for behavior cloning.

           +---------------- observation o_t (AXTree) ----------------+
           |                                                          v
    BrowserGym env (Chromium)                               policy pi = Claude
           ^                                                          |
           +------ action a_t  (e.g. click("13")), reward r_t <-------+

Run:  uvicorn app:app --port 8000   (from this folder; see ../README.md)
"""

from __future__ import annotations

import asyncio
import json
import time
import traceback
import uuid
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import ValidationError

import env_worker
import policies
from env_worker import EnvWorker
from policies import ClaudePolicy, HistoryItem, PolicyContext, RandomPolicy, build_prompt
from protocol import (
    ActCmd,
    BatchCmd,
    Envelope,
    Episode,
    EpisodeSummary,
    Observation,
    Phase,
    Proposal,
    ResetCmd,
    RunCmd,
    Step,
    ThinkCmd,
)

ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = ROOT / "runs"
RUNS_DIR.mkdir(exist_ok=True)
DIST_DIR = ROOT / "web" / "dist"

app = FastAPI(title="BrowserGym x Claude controller")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


# ------------------------------------------------------------------ REST


@app.get("/api/config")
def get_config() -> dict[str, Any]:
    return {
        "has_api_key": policies.has_api_key(),
        "default_model": policies.DEFAULT_MODEL,
        "models": policies.AVAILABLE_MODELS,
        "benchmarks": env_worker.INSTALLED_BENCHMARKS,
        "action_space": env_worker.ACTION_SET.describe(
            with_long_description=False, with_examples=True
        ),
        "system_prompt": policies.SYSTEM_PROMPT,
    }


@app.get("/api/tasks")
def get_tasks() -> dict[str, list[str]]:
    return env_worker.list_tasks()


@app.get("/api/episodes")
def list_episodes() -> list[dict]:
    out = []
    for p in sorted(RUNS_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            data = json.loads(p.read_text())
            out.append(EpisodeSummary.model_validate(data).model_dump())
        except Exception:
            continue
    return out


@app.get("/api/episodes/{episode_id}")
def get_episode(episode_id: str) -> dict:
    p = RUNS_DIR / f"{Path(episode_id).name}.json"
    if not p.exists():
        raise HTTPException(404, "episode not found")
    return json.loads(p.read_text())


# ------------------------------------------------------------------ session


class Session:
    """One WebSocket connection = one browser = one agent loop."""

    def __init__(self, ws: WebSocket) -> None:
        self.ws = ws
        self.worker = EnvWorker()
        self.phase: Phase = "idle"
        self.episode: Optional[Episode] = None
        self.obs: Optional[Observation] = None
        self.history: list[HistoryItem] = []
        self.pending: Optional[Proposal] = None
        self.reset_cmd: Optional[ResetCmd] = None
        self.busy = asyncio.Lock()
        self.run_cfg: Optional[RunCmd] = None  # set while autopilot is on
        self.cancel_batch = False
        self._claude: Optional[ClaudePolicy] = None
        self._tasks: set[asyncio.Task] = set()

    # ---- plumbing
    async def send(self, type_: str, **payload: Any) -> None:
        try:
            await self.ws.send_text(json.dumps({"type": type_, **payload}, default=_jsonable))
        except Exception:
            pass

    async def set_phase(self, phase: Phase) -> None:
        self.phase = phase
        await self.send("phase", phase=phase)

    def spawn(self, coro) -> None:
        task = asyncio.create_task(self._guard(coro))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def _guard(self, coro) -> None:
        try:
            await coro
        except Exception as e:  # surface every failure in the UI instead of dying silently
            traceback.print_exc()
            self.run_cfg = None
            await self.send("error", message=f"{type(e).__name__}: {e}")
            if self.obs is not None and self.phase not in ("done", "idle"):
                await self.set_phase("observing")

    def policy(self, name: str):
        if name == "random":
            return RandomPolicy(self.worker.clickable_bids)
        if name == "claude":
            if not policies.has_api_key():
                raise RuntimeError(
                    "ANTHROPIC_API_KEY is not set on the server. Set it and restart, "
                    "or use the random / human policy."
                )
            if self._claude is None:
                self._claude = ClaudePolicy()
            return self._claude
        raise RuntimeError(f"'{name}' is not an automatic policy; send an act command instead.")

    # ---- the RL loop, one phase at a time
    async def reset(self, cmd: ResetCmd) -> None:
        self.run_cfg = None
        await self.set_phase("resetting")
        obs, _ = await self.worker.reset(cmd.task, cmd.seed, cmd.headless)
        self.reset_cmd, self.obs, self.history, self.pending = cmd, obs, [], None
        self.episode = Episode(
            id=f"{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}",
            task=cmd.task,
            seed=cmd.seed,
            policy="human",
            started_at=time.time(),
            goal=obs.goal,
        )
        await self.send("episode_start", episode=self.episode.model_dump(exclude={"steps"}))
        await self.send("observation", obs=obs.model_dump())
        await self.set_phase("observing")

    async def think(self, policy: str, model: Optional[str], temperature: Optional[float]) -> Proposal:
        if self.obs is None or self.phase == "done":
            raise RuntimeError("No active episode. Reset an environment first.")
        await self.set_phase("thinking")
        pi = self.policy(policy)
        await self.send("policy_start", t=self.obs.t, policy=policy, model=model)
        ctx = PolicyContext(
            obs=self.obs,
            history=list(self.history),
            use_screenshot=bool(self.reset_cmd and self.reset_cmd.use_screenshot),
        )

        async def emit(delta: str) -> None:
            await self.send("policy_delta", text=delta)

        proposal = await pi(ctx, emit, model=model, temperature=temperature)
        self.pending = proposal
        await self.send("proposal", proposal=proposal.model_dump())
        await self.set_phase("awaiting_approval")
        return proposal

    async def act(self, action: str, source: str, thought: str = "") -> Step:
        if self.obs is None or self.episode is None or self.phase == "done":
            raise RuntimeError("No active episode. Reset an environment first.")
        await self.set_phase("acting")
        o_t = self.obs
        proposal = self.pending if self.pending and self.pending.t == o_t.t else None
        result = await self.worker.step(action, o_t.t + 1)
        next_obs: Observation = result["obs"]
        max_steps = self.reset_cmd.max_steps if self.reset_cmd else 10
        truncated = result["truncated"] or (
            not result["terminated"] and len(self.episode.steps) + 1 >= max_steps
        )
        step = Step(
            t=o_t.t,
            obs=o_t,
            proposal=proposal,
            action=action,
            action_source=source,  # type: ignore[arg-type]
            prompt=proposal.prompt
            if proposal and proposal.prompt
            else build_prompt(PolicyContext(obs=o_t, history=list(self.history))),
            thought=thought or (proposal.thought if proposal else ""),
            reward=result["reward"],
            terminated=result["terminated"],
            truncated=truncated,
            action_error=next_obs.last_action_error,
            env_ms=result["env_ms"],
        )
        ep = self.episode
        ep.steps.append(step)
        ep.n_steps = len(ep.steps)
        ep.total_reward += step.reward
        if source != "human":
            ep.policy = source  # type: ignore[assignment]
            ep.model = proposal.model if proposal else ep.model
        elif proposal is not None and proposal.action != action:
            ep.human_interventions += 1
        self.history.append(HistoryItem(action=action, thought=step.thought, error=step.action_error))
        self.pending = None
        self.obs = next_obs
        await self.send("step", step=step.model_dump())

        if step.terminated or step.truncated:
            await self.finish(step)
        else:
            await self.send("observation", obs=next_obs.model_dump())
            await self.set_phase("observing")
        return step

    async def finish(self, last: Step) -> None:
        ep = self.episode
        assert ep is not None
        ep.ended_at = time.time()
        ep.terminated, ep.truncated = last.terminated, last.truncated
        ep.success = last.terminated and ep.total_reward > 0
        (RUNS_DIR / f"{ep.id}.json").write_text(ep.model_dump_json())
        self.run_cfg = None
        await self.send("observation", obs=self.obs.model_dump() if self.obs else None)
        await self.send("episode_end", episode=ep.model_dump(exclude={"steps"}))
        await self.set_phase("done")

    # ---- higher-level drivers
    async def autopilot(self) -> None:
        """Keep calling think -> act until the episode ends or the user pauses."""
        while self.run_cfg and self.phase == "observing":
            cfg = self.run_cfg
            proposal = await self.think(cfg.policy, cfg.model, cfg.temperature)
            if cfg.supervised or self.run_cfg is None:
                return  # wait for the human to approve / edit; `act` resumes the loop
            await self.act(proposal.action, source=cfg.policy, thought=proposal.thought)

    async def batch(self, cmd: BatchCmd) -> None:
        """Benchmark runner: many episodes, no supervision, one result row each."""
        self.cancel_batch = False
        jobs = [(t, s) for t in cmd.tasks for s in cmd.seeds]
        results: list[dict] = []
        await self.send("batch_progress", done=0, total=len(jobs), results=results, running=True)
        for i, (task, seed) in enumerate(jobs):
            if self.cancel_batch:
                break
            await self.reset(
                ResetCmd(type="reset", task=task, seed=seed, max_steps=cmd.max_steps,
                         use_screenshot=cmd.use_screenshot)
            )
            self.run_cfg = RunCmd(type="run", policy=cmd.policy, model=cmd.model,
                                  temperature=cmd.temperature)
            await self.autopilot()
            if self.episode and self.episode.ended_at:
                results.append(self.episode.model_dump(exclude={"steps"}))
            await self.send("batch_progress", done=i + 1, total=len(jobs), results=results,
                            running=True)
        await self.send("batch_progress", done=len(results), total=len(jobs), results=results,
                        running=False)

    # ---- command dispatch
    async def handle(self, data: dict) -> None:
        try:
            cmd = Envelope(cmd=data).cmd
        except ValidationError as e:
            await self.send("error", message=f"Invalid command: {e.errors()[0]['msg']}")
            return

        if cmd.type == "pause":
            self.run_cfg = None
            await self.send("info", message="Autopilot paused after the current step.")
            return
        if cmd.type == "cancel":
            self.cancel_batch, self.run_cfg = True, None
            await self.send("info", message="Batch will stop after the current episode.")
            return
        if self.busy.locked():
            # A human `act` during a supervised run is how approval works, so allow it then.
            await self.send("error", message="Busy: wait for the current step to finish.")
            return

        async def locked(coro):
            async with self.busy:
                await coro

        if cmd.type == "reset":
            self.spawn(locked(self.reset(cmd)))
        elif cmd.type == "think":
            self.spawn(locked(self.think(cmd.policy, cmd.model, cmd.temperature)))
        elif cmd.type == "act":
            async def act_then_resume(c: ActCmd = cmd):
                await self.act(c.action, c.source, c.thought)
                if self.run_cfg and self.phase == "observing":
                    await self.autopilot()
            self.spawn(locked(act_then_resume()))
        elif cmd.type == "run":
            async def run(c: RunCmd = cmd):
                self.run_cfg = c
                if self.phase == "awaiting_approval" and self.pending and not c.supervised:
                    await self.act(self.pending.action, c.policy, self.pending.thought)
                await self.autopilot()
            self.spawn(locked(run()))
        elif cmd.type == "batch":
            self.spawn(locked(self.batch(cmd)))

    async def close(self) -> None:
        self.run_cfg, self.cancel_batch = None, True
        for t in list(self._tasks):
            t.cancel()
        await self.worker.close()


def _jsonable(o: Any):
    if hasattr(o, "model_dump"):
        return o.model_dump()
    raise TypeError(type(o))


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket) -> None:
    await ws.accept()
    session = Session(ws)
    await session.send("hello", config=get_config())
    await session.set_phase("idle")
    try:
        while True:
            data = json.loads(await ws.receive_text())
            await session.handle(data)
    except WebSocketDisconnect:
        pass
    finally:
        await session.close()


# Serve the built front end (web/dist) so `make serve` is a single process.
if DIST_DIR.exists():
    app.mount("/assets", StaticFiles(directory=DIST_DIR / "assets"), name="assets")

    @app.get("/{path:path}")
    def spa(path: str):
        f = DIST_DIR / path
        return FileResponse(f if f.is_file() else DIST_DIR / "index.html")
