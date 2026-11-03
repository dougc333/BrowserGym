"""
The agentic search loop, and the analysis of what a run explored.

    reset -> loop { observe page -> policy decides -> env.step(action) -> record } -> check

BrowserGym drives Chromium through Playwright's sync API, which must stay on one
thread, so every run executes on a single dedicated worker thread, one at a time.
"""

from __future__ import annotations

import io
import json
import re
import threading
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Callable
from urllib.parse import parse_qs, urlparse

import gymnasium as gym
from PIL import Image

import task as _task  # noqa: F401  registers browsergym/travel.search
from browsergym.utils.obs import flatten_axtree_to_str
from db import SCREENS, connect, row, rows
from generator import PRESETS, check, episode_flights, get_episode, total_price, violations
from policies import ACTION_SET, StepContext, make_policy

Emit = Callable[[dict], None]


@dataclass
class RunConfig:
    difficulty: dict
    seed: int
    policy: str = "scripted"
    model: str | None = None
    max_steps: int = 150
    headless: bool = True


def page_type(url: str) -> str:
    p = urlparse(url).path
    for key, name in (("/results", "results"), ("/flight/", "details"), ("/review", "review"),
                      ("/confirmation/", "confirmation")):
        if key in p:
            return name
    return "search" if re.search(r"/e/\w+/?$", p) else "other"


def _save_screenshot(run_id: str, t: int, img) -> None:
    d = SCREENS / run_id
    d.mkdir(parents=True, exist_ok=True)
    im = Image.fromarray(img).convert("RGB")
    if im.width > 900:
        im = im.resize((900, int(im.height * 900 / im.width)))
    im.save(d / f"{t}.jpg", quality=70)


class Runner:
    def __init__(self, emit: Emit):
        self.emit = emit
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="travel-env")
        self.stop_flags: dict[str, threading.Event] = {}

    def submit(self, cfg: RunConfig) -> str:
        run_id = time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:4]
        conn = connect()
        with conn:
            conn.execute("INSERT INTO runs (id, difficulty, seed, policy, model, max_steps, status) VALUES (?,?,?,?,?,?,?)",
                         (run_id, json.dumps(cfg.difficulty), cfg.seed, cfg.policy, cfg.model, cfg.max_steps, "queued"))
        self.stop_flags[run_id] = threading.Event()
        self.emit({"type": "run", "run": get_run(run_id)})
        self.pool.submit(self._run, run_id, cfg)
        return run_id

    def stop(self, run_id: str) -> None:
        if run_id in self.stop_flags:
            self.stop_flags[run_id].set()

    def _update(self, run_id: str, **fields) -> None:
        conn = connect()
        with conn:
            sets = ", ".join(f"{k}=?" for k in fields)
            conn.execute(f"UPDATE runs SET {sets} WHERE id=?",
                         [json.dumps(v) if isinstance(v, (dict, list)) else v for v in fields.values()] + [run_id])
        self.emit({"type": "run", "run": get_run(run_id)})

    def _run(self, run_id: str, cfg: RunConfig) -> None:
        env = None
        try:
            self._update(run_id, status="running", started_at=time.time())
            policy = make_policy(cfg.policy, cfg.model)
            env = gym.make("browsergym/travel.search", task_kwargs={"difficulty": cfg.difficulty},
                           headless=cfg.headless, action_mapping=ACTION_SET.to_python_code)
            obs, info = env.reset(seed=cfg.seed)
            ep_id = info["task_info"]["episode_id"]
            self._update(run_id, episode_id=ep_id)
            ep = get_episode(ep_id)
            notes, history, tok_in, tok_out = "", [], 0, 0
            conn = connect()
            status = "done"
            for t in range(cfg.max_steps):
                if self.stop_flags[run_id].is_set():
                    status = "stopped"
                    break
                axtree = flatten_axtree_to_str(obs["axtree_object"], extra_properties=obs["extra_element_properties"],
                                               filter_visible_only=False)
                ctx = StepContext(goal=obs["goal"], axtree=axtree, url=obs["url"], t=t, history=history,
                                  notes=notes, task=ep["task"], last_error=obs.get("last_action_error") or "")
                dec = policy(ctx)
                _save_screenshot(run_id, t, obs["screenshot"])
                url = obs["url"]
                obs, reward, terminated, truncated, info = env.step(dec.action)
                err = obs.get("last_action_error") or ""
                tok_in, tok_out = tok_in + dec.input_tokens, tok_out + dec.output_tokens
                with conn:
                    conn.execute("INSERT INTO steps VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                                 (run_id, t, url, page_type(url), dec.thought, dec.notes, dec.action, err, axtree,
                                  dec.input_tokens, dec.output_tokens, dec.latency_ms))
                    conn.execute("UPDATE runs SET n_steps=?, input_tokens=?, output_tokens=? WHERE id=?",
                                 (t + 1, tok_in, tok_out, run_id))
                self.emit({"type": "step", "run_id": run_id, "step": get_step(run_id, t, with_axtree=False)})
                history.append((dec.action, err))
                notes = dec.notes
                if terminated or truncated:
                    break
            outcome = check(ep_id)
            if not outcome["booked"] and history and history[-1][0].startswith("report_infeasible"):
                outcome["reason"] = "agent reported the task infeasible (a valid flight always exists)"
            elif not outcome["booked"] and status == "done":
                outcome["reason"] = "no booking within the step budget"
            self._update(run_id, status=status, outcome=outcome, reward=outcome["reward"],
                         success=int(outcome["success"]), ended_at=time.time())
        except Exception as e:
            traceback.print_exc()
            self._update(run_id, status="error", error=f"{type(e).__name__}: {e}", ended_at=time.time())
        finally:
            if env is not None:
                env.close()
            self.stop_flags.pop(run_id, None)


# ------------------------------------------------------------------ queries
def get_run(run_id: str) -> dict | None:
    r = row(connect().execute("SELECT * FROM runs WHERE id=?", (run_id,)).fetchone())
    if r and isinstance(r.get("difficulty"), str):
        r["difficulty"] = json.loads(r["difficulty"])
    return r


def list_runs() -> list[dict]:
    out = rows(connect().execute("SELECT * FROM runs ORDER BY rowid DESC LIMIT 200"))
    for r in out:
        if isinstance(r.get("difficulty"), str):
            r["difficulty"] = json.loads(r["difficulty"])
    return out


def get_step(run_id: str, t: int, with_axtree: bool = True) -> dict | None:
    r = row(connect().execute("SELECT * FROM steps WHERE run_id=? AND t=?", (run_id, t)).fetchone())
    if r and not with_axtree:
        r.pop("axtree", None)
    return r


def list_steps(run_id: str) -> list[dict]:
    return rows(connect().execute(
        "SELECT run_id, t, url, page, thought, notes, action, error, prompt_tokens, output_tokens, latency_ms "
        "FROM steps WHERE run_id=? ORDER BY t", (run_id,)))


# ------------------------------------------------------------------ coverage analysis
def _ordered(flights: list[dict], sort: str, sort_available: bool) -> list[dict]:
    if sort == "price" and sort_available:
        return sorted(flights, key=lambda f: (f["base_fare"], f["rank"]))
    return sorted(flights, key=lambda f: f["rank"])


def coverage(run_id: str) -> dict:
    """What the agent looked at, compared with where the answer was."""
    run = get_run(run_id)
    if not run or not run.get("episode_id"):
        return {}
    ep = get_episode(run["episode_id"])
    task, diff, opt = ep["task"], ep["difficulty"], ep["optimum"]
    decoys = {int(k): v for k, v in diff.get("decoy_ids", {}).items()}
    flights = episode_flights(ep["id"])
    by_id = {f["id"]: f for f in flights}
    pax, bags, size = task["passengers"], task["bags"], diff["page_size"]

    seen_ids, pages_seen, details_ids, review_ids = set(), set(), set(), set()
    first_seen: dict[int, int] = {}
    for s in list_steps(run_id):
        u = urlparse(s["url"])
        q = {k: v[0] for k, v in parse_qs(u.query).items()}
        if s["page"] == "results":
            o, d, pg, srt = q.get("from"), q.get("date"), int(q.get("page", 1)), q.get("sort", "recommended")
            group = [f for f in flights if f["origin"] == o and f["date"] == d and f["dest"] == q.get("to")]
            shown = _ordered(group, srt, diff["sort_available"])[(pg - 1) * size: pg * size]
            for f in shown:
                seen_ids.add(f["id"])
                first_seen.setdefault(f["id"], s["t"])
            pages_seen.add((o, d, srt, pg))
        elif s["page"] == "details":
            m = re.search(r"/flight/(\d+)", u.path)
            if m:
                details_ids.add(int(m.group(1)))
        elif s["page"] == "review" and "flight" in q:
            review_ids.add(int(q["flight"]))
    booking = row(connect().execute("SELECT * FROM bookings WHERE episode_id=? ORDER BY id LIMIT 1", (ep["id"],)).fetchone())
    booked_id = booking["flight_id"] if booking else None

    cells = []
    for o in task["origins"]:
        for d in task["dates"]:
            group = [f for f in flights if f["origin"] == o and f["date"] == d]
            n_pages = (len(group) + size - 1) // size
            seen_pages = sorted({p for (oo, dd, _, p) in pages_seen if oo == o and dd == d})
            valid = [f for f in group if not violations(f, task)]
            cells.append({
                "origin": o, "date": d, "n_pages": n_pages, "pages_seen": seen_pages,
                "n_flights": len(group), "n_valid": len(valid),
                "best_valid_total": min((total_price(f, pax, bags) for f in valid), default=None),
                "has_optimum": any(f["id"] == opt["flight_id"] for f in group),
                "flights_seen": sum(f["id"] in seen_ids for f in group),
            })

    def entry(f, kind):
        return {
            "id": f["id"], "flight_no": f["flight_no"], "airline": f["airline"], "origin": f["origin"],
            "date": f["date"], "depart": f["depart"], "stops": f["stops"], "cabin": f["cabin"],
            "refundable": bool(f["refundable"]), "listed_fare": f["base_fare"],
            "true_total": total_price(f, pax, bags), "kind": kind,
            "violations": violations(f, task), "page": f["rank"] // size + 1,
            "seen": f["id"] in seen_ids, "opened": f["id"] in details_ids, "reviewed": f["id"] in review_ids,
            "booked": f["id"] == booked_id, "first_seen_step": first_seen.get(f["id"]),
        }

    valid_sorted = sorted((f for f in flights if not violations(f, task)), key=lambda f: total_price(f, pax, bags))
    ledger = [entry(f, "optimum" if f["id"] == opt["flight_id"] else ("fee trap" if decoys.get(f["id"]) == "fee trap" else "valid"))
              for f in valid_sorted[:10]]
    ledger += [entry(by_id[i], f"decoy: {k}") for i, k in decoys.items() if k != "fee trap" and i in by_id]
    if booked_id and all(e["id"] != booked_id for e in ledger):
        ledger.append(entry(by_id[booked_id], "booked"))

    o_id = opt["flight_id"]
    if booked_id == o_id:
        diagnosis = "Booked the optimum."
    elif o_id in review_ids:
        diagnosis = "Reviewed the optimum's true total but booked something else."
    elif o_id in details_ids:
        diagnosis = "Opened the optimum but never checked its true total."
    elif o_id in seen_ids:
        diagnosis = "The optimum was on a page it saw, but it never opened it."
    else:
        diagnosis = "Never saw the optimum: its search or results page was not explored."
    total_pages = sum(c["n_pages"] for c in cells)
    return {
        "episode_id": ep["id"], "task": task, "difficulty": diff, "optimum": opt,
        "cells": cells, "ledger": ledger, "diagnosis": diagnosis,
        "summary": {
            "searches_needed": len(cells), "searches_done": sum(1 for c in cells if c["pages_seen"]),
            "pages_total": total_pages, "pages_seen": sum(len(c["pages_seen"]) for c in cells),
            "flights_seen": len(seen_ids), "flights_total": len(flights), "details_opened": len(details_ids),
            "reviews": len(review_ids), "valid_seen": sum(1 for f in valid_sorted if f["id"] in seen_ids),
            "valid_total": len(valid_sorted),
        },
    }
