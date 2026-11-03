"""
The sandbox travel site as a BrowserGym task.

    env = gym.make("browsergym/travel.search", task_kwargs={"difficulty": "hard"})
    obs, info = env.reset(seed=7)       # generates a fresh episode, opens the search page
    ...                                 # the agent searches and books through the browser
    reward                              # 1.0 only if every constraint holds AND it is the cheapest valid itinerary

The episode ends when the agent books (the first booking is final) or reports the
task infeasible. Reward comes from the database, never from what the page says.
"""

from __future__ import annotations

import os
from dataclasses import asdict
from typing import Tuple

import playwright.sync_api
from browsergym.core.registration import register_task
from browsergym.core.task import AbstractBrowserTask

from generator import PRESETS, Difficulty, check, generate_episode

SITE_URL = os.environ.get("TRAVEL_SITE_URL", "http://localhost:8100")


class TravelSearchTask(AbstractBrowserTask):
    @classmethod
    def get_task_id(cls):
        return "travel.search"

    def __init__(self, seed: int, difficulty: str | dict = "medium", site_url: str = SITE_URL) -> None:
        super().__init__(seed)
        self.seed = seed
        self.difficulty = PRESETS[difficulty] if isinstance(difficulty, str) else Difficulty(**difficulty)
        self.site_url = site_url.rstrip("/")
        self.viewport = {"width": 1100, "height": 900}
        self.slow_mo = 50
        self.timeout = 8000
        self.episode_id: str | None = None

    def setup(self, page: playwright.sync_api.Page) -> tuple[str, dict]:
        ep = generate_episode(self.seed, self.difficulty)
        self.episode_id = ep["id"]
        page.goto(f"{self.site_url}/e/{ep['id']}/", timeout=15000)
        return ep["task"]["goal"], {"episode_id": ep["id"], "difficulty": asdict(self.difficulty)}

    def validate(self, page: playwright.sync_api.Page, chat_messages: list) -> Tuple[float, bool, str, dict]:
        outcome = check(self.episode_id)
        if outcome["booked"]:
            return outcome["reward"], True, "", {"outcome": outcome}
        if any(m.get("role") == "infeasible" for m in chat_messages):
            return 0.0, True, "", {"outcome": {**outcome, "reason": "reported infeasible"}}
        return 0.0, False, "", {"outcome": outcome}


try:
    register_task(TravelSearchTask.get_task_id(), TravelSearchTask, default_task_kwargs={"difficulty": "medium"})
except Exception:  # already registered (module imported twice)
    pass
