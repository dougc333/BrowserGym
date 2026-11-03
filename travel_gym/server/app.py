"""
Dashboard API for the travel-search gym (port 8000). It also starts the sandbox
website (port 8100) if it isn't already running.

The website and this API are separate on purpose: the agent's browser only ever
visits the website, which has no endpoint that reveals the answer. This API serves
the ground truth (optimum, decoys, coverage) to the dashboard only.

  uvicorn app:app --port 8000
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import urllib.request
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Literal, Optional

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from db import SCREENS
from generator import PRESETS, Difficulty
from policies import DEFAULT_MODEL, MODELS, NOUS_BASE_URL, NOUS_DEFAULT_MODEL, NOUS_MODELS
from runner import RunConfig, Runner, coverage, get_run, get_step, list_runs, list_steps
from task import SITE_URL

HERE = Path(__file__).resolve().parent
DIST = HERE.parent / "web" / "dist"


class Hub:
    """Fan-out of runner events (from the env thread) to every connected WebSocket."""

    def __init__(self) -> None:
        self.clients: set[WebSocket] = set()
        self.loop: asyncio.AbstractEventLoop | None = None

    def emit(self, event: dict) -> None:  # called from the runner thread
        if self.loop:
            asyncio.run_coroutine_threadsafe(self._broadcast(event), self.loop)

    async def _broadcast(self, event: dict) -> None:
        msg = json.dumps(event, default=str)
        for ws in list(self.clients):
            try:
                await ws.send_text(msg)
            except Exception:
                self.clients.discard(ws)


hub = Hub()
runner = Runner(hub.emit)
site_proc: subprocess.Popen | None = None


def site_up() -> bool:
    try:
        urllib.request.urlopen(SITE_URL + "/e/none/", timeout=1)
    except urllib.error.HTTPError:
        return True  # 404 for an unknown episode means the server answered
    except Exception:
        return False
    return True


@asynccontextmanager
async def lifespan(_: FastAPI):
    global site_proc
    hub.loop = asyncio.get_running_loop()
    if not site_up() and "localhost" in SITE_URL:
        port = SITE_URL.rsplit(":", 1)[-1].split("/")[0]
        site_proc = subprocess.Popen([sys.executable, "-m", "uvicorn", "website:app", "--port", port,
                                      "--log-level", "warning"], cwd=HERE)
        for _ in range(50):
            if site_up():
                break
            await asyncio.sleep(0.2)
    yield
    if site_proc:
        site_proc.terminate()


app = FastAPI(title="Travel search gym", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class DifficultyIn(BaseModel):
    n_dates: int = Field(3, ge=1, le=14)
    n_origins: int = Field(1, ge=1, le=3)
    pages: int = Field(2, ge=1, le=8)
    page_size: int = Field(8, ge=3, le=20)
    hidden_fees: bool = True
    sort_available: bool = True
    passengers: int = Field(2, ge=1, le=4)
    bags: bool = True
    require_nonstop: bool = True
    depart_after: bool = True
    require_refundable: bool = False
    decoys: int = Field(4, ge=0, le=20)


class RunIn(BaseModel):
    difficulty: DifficultyIn
    seed: int = Field(0, ge=0)
    policy: Literal["scripted", "claude", "openai", "nous", "mcp-scripted", "mcp-claude", "mcp-nous", "mcp-openai"] = "scripted"
    model: Optional[str] = None
    max_steps: int = Field(150, ge=1, le=1000)
    headless: bool = True


@app.get("/api/config")
def config():
    return {
        "presets": {k: asdict(v) for k, v in PRESETS.items()},
        "models": MODELS, "default_model": DEFAULT_MODEL,
        "has_api_key": bool(os.environ.get("ANTHROPIC_API_KEY")),
        "has_nous_key": bool(os.environ.get("NOUS_API_KEY")),
        "nous_models": NOUS_MODELS, "nous_default_model": NOUS_DEFAULT_MODEL, "nous_base_url": NOUS_BASE_URL,
        "site_url": SITE_URL, "site_up": site_up(),
    }


@app.post("/api/runs")
def create_run(body: RunIn):
    if body.policy in ("claude", "mcp-claude") and not os.environ.get("ANTHROPIC_API_KEY"):
        raise HTTPException(400, "ANTHROPIC_API_KEY is not set on the server.")
    if body.policy in ("nous", "mcp-nous") and not os.environ.get("NOUS_API_KEY"):
        raise HTTPException(400, "NOUS_API_KEY is not set on the server.")
    cfg = RunConfig(difficulty=body.difficulty.model_dump(), seed=body.seed, policy=body.policy,
                    model=body.model, max_steps=body.max_steps, headless=body.headless)
    Difficulty(**cfg.difficulty)  # validates field names
    return {"id": runner.submit(cfg)}


@app.get("/api/runs")
def runs():
    return list_runs()


@app.get("/api/runs/{run_id}")
def run_detail(run_id: str):
    r = get_run(run_id)
    if not r:
        raise HTTPException(404)
    return {"run": r, "steps": list_steps(run_id)}


@app.get("/api/runs/{run_id}/steps/{t}")
def step_detail(run_id: str, t: int):
    s = get_step(run_id, t)
    if not s:
        raise HTTPException(404)
    return s


@app.get("/api/runs/{run_id}/coverage")
def run_coverage(run_id: str):
    return coverage(run_id)


@app.post("/api/runs/{run_id}/stop")
def stop(run_id: str):
    runner.stop(run_id)
    return {"ok": True}


@app.get("/api/screens/{run_id}/{t}.jpg")
def screen(run_id: str, t: int):
    p = SCREENS / Path(run_id).name / f"{t}.jpg"
    if not p.exists():
        raise HTTPException(404)
    return FileResponse(p, media_type="image/jpeg")


@app.websocket("/ws")
async def ws(ws: WebSocket):
    await ws.accept()
    hub.clients.add(ws)
    try:
        while True:
            await ws.receive_text()  # keepalive; the client never needs to send anything
    except WebSocketDisconnect:
        hub.clients.discard(ws)


if DIST.exists():
    app.mount("/assets", StaticFiles(directory=DIST / "assets"), name="assets")

    @app.get("/{path:path}")
    def spa(path: str):
        f = DIST / path
        return FileResponse(f if f.is_file() else DIST / "index.html")
