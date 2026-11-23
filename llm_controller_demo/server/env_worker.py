"""
The environment side of the loop.

BrowserGym drives Chromium through Playwright's *sync* API, which must stay on
one OS thread and cannot run inside an asyncio event loop. So every env call
goes through a single-thread executor, and the async server awaits it.

    server (asyncio) --run_in_executor--> EnvWorker thread --Playwright--> Chromium
"""

from __future__ import annotations

import asyncio
import base64
import io
import os
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Optional

import gymnasium as gym
import numpy as np
from PIL import Image

import browsergym.core  # noqa: F401  registers browsergym/openended
from browsergym.core.action.highlevel import HighLevelActionSet
from browsergym.utils.obs import flatten_axtree_to_str

from protocol import Observation

# Register whichever benchmarks are installed. MiniWoB only needs MINIWOB_URL;
# the others need their own servers / credentials (see the BrowserGym READMEs).
BENCHMARK_MODULES = [
    "browsergym.miniwob",
    "browsergym.webarena",
    "browsergym.webarena_verified",
    "browsergym.visualwebarena",
    "browsergym.workarena",
    "browsergym.assistantbench",
]
INSTALLED_BENCHMARKS: list[str] = []
for _mod in BENCHMARK_MODULES:
    try:
        __import__(_mod)
        INSTALLED_BENCHMARKS.append(_mod.split(".")[-1])
    except Exception:  # not installed, or missing config
        pass


# The action space the policy is allowed to use. "bid" actions address elements
# by the [bid] shown in the AXTree, which is why AXTree observations pair so
# well with an LLM: the model can name exactly what it wants to touch.
ACTION_SET = HighLevelActionSet(
    subsets=["chat", "tab", "nav", "bid", "infeas"],
    strict=False,
    multiaction=False,
    demo_mode="off",
)


def list_tasks() -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for env_id in gym.envs.registry.keys():
        if not env_id.startswith("browsergym/"):
            continue
        bench = env_id.split("/", 1)[1].split(".", 1)[0]
        out.setdefault(bench, []).append(env_id)
    return {k: sorted(v) for k, v in sorted(out.items())}


def _to_data_url(img: np.ndarray, max_width: int = 800) -> str:
    im = Image.fromarray(img)
    if im.mode in ("RGBA", "LA"):
        im = im.convert("RGB")
    if im.width > max_width:
        im = im.resize((max_width, int(im.height * max_width / im.width)))
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=70)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def package_obs(raw: dict[str, Any], t: int) -> Observation:
    """Turn BrowserGym's raw observation dict into the compact o_t we show the policy."""
    axtree = flatten_axtree_to_str(
        raw["axtree_object"],
        extra_properties=raw["extra_element_properties"],
        with_clickable=True,
        filter_visible_only=True,
    )
    goal = raw.get("goal") or " ".join(
        g.get("text", "") for g in raw.get("goal_object", []) if g.get("type") == "text"
    )
    tabs = [
        f"{title} ({url})" for url, title in zip(raw["open_pages_urls"], raw["open_pages_titles"])
    ]
    return Observation(
        t=t,
        goal=goal,
        url=raw.get("url", ""),
        open_tabs=tabs,
        axtree=axtree,
        screenshot=_to_data_url(raw["screenshot"]) if raw.get("screenshot") is not None else None,
        last_action=raw.get("last_action", "") or "",
        last_action_error=raw.get("last_action_error", "") or "",
    )


# BrowserGym keeps ONE global Playwright instance, bound to the thread that first
# created it. So all sessions share this single, never-closed env thread.
_ENV_THREAD = ThreadPoolExecutor(max_workers=1, thread_name_prefix="browsergym")


class EnvWorker:
    def __init__(self) -> None:
        self._pool = _ENV_THREAD
        self.env: Optional[gym.Env] = None
        self.raw_obs: Optional[dict] = None
        self.task: Optional[str] = None
        self._headless: Optional[bool] = None

    async def _call(self, fn, *args):
        return await asyncio.get_running_loop().run_in_executor(self._pool, fn, *args)

    # ---- sync, executed on the env thread ----
    def _reset(self, task: str, seed: int, headless: bool) -> tuple[Observation, float]:
        if self.env is not None and (self.task != task or self._headless != headless):
            self.env.close()
            self.env = None
        if self.env is None:
            kwargs: dict[str, Any] = dict(
                headless=headless,
                action_mapping=ACTION_SET.to_python_code,
            )
            exe = os.environ.get("CHROMIUM_EXECUTABLE")
            if exe:
                kwargs["pw_chromium_kwargs"] = {"executable_path": exe}
            self.env = gym.make(task, **kwargs)
            self.task, self._headless = task, headless
        t0 = time.time()
        self.raw_obs, _ = self.env.reset(seed=seed)
        return package_obs(self.raw_obs, 0), time.time() - t0

    def _step(self, action: str, t: int) -> dict[str, Any]:
        assert self.env is not None, "call reset first"
        t0 = time.time()
        self.raw_obs, reward, terminated, truncated, info = self.env.step(action)
        return dict(
            obs=package_obs(self.raw_obs, t),
            reward=float(reward),
            terminated=bool(terminated),
            truncated=bool(truncated),
            env_ms=int((time.time() - t0) * 1000),
        )

    def _close(self) -> None:
        if self.env is not None:
            self.env.close()
            self.env = None

    # ---- async API ----
    async def reset(self, task: str, seed: int, headless: bool = True):
        return await self._call(self._reset, task, seed, headless)

    async def step(self, action: str, t: int):
        return await self._call(self._step, action, t)

    async def close(self):
        await self._call(self._close)

    def clickable_bids(self) -> list[str]:
        """Bids that are visible and clickable, used by the random baseline policy."""
        props = (self.raw_obs or {}).get("extra_element_properties", {})
        return [
            bid
            for bid, p in props.items()
            if p.get("clickable") and (p.get("visibility") or 0) > 0.5
        ]
