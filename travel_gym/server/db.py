"""
One SQLite database for everything (data/travel.db):

  episodes   one generated task instance: seed, difficulty, the task, the true optimum
  flights    that episode's flight inventory (what the sandbox website serves)
  bookings   what the agent booked (written by the website, read by the checker)
  runs       one agent run on an episode: policy, outcome, cost
  steps      every step of a run: observation summary, thought, notes, action, error

The website only ever reads `flights` and writes `bookings`; it has no endpoint that
exposes `episodes.optimum`. Only the dashboard API (a different port) reads it.
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = Path(os.environ.get("TRAVEL_GYM_DATA", ROOT / "data"))
DB_PATH = DATA / "travel.db"
SCREENS = DATA / "screens"

SCHEMA = """
CREATE TABLE IF NOT EXISTS episodes (
  id TEXT PRIMARY KEY,
  seed INTEGER NOT NULL,
  difficulty TEXT NOT NULL,          -- JSON
  task TEXT NOT NULL,                -- JSON: constraints + goal text
  optimum TEXT NOT NULL,             -- JSON: flight_id, total, rank info
  created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS flights (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  episode_id TEXT NOT NULL REFERENCES episodes(id),
  airline TEXT NOT NULL,
  flight_no TEXT NOT NULL,
  origin TEXT NOT NULL,
  dest TEXT NOT NULL,
  date TEXT NOT NULL,                -- YYYY-MM-DD
  depart TEXT NOT NULL,              -- HH:MM
  arrive TEXT NOT NULL,              -- HH:MM (may be +1 day)
  duration_min INTEGER NOT NULL,
  stops INTEGER NOT NULL,
  cabin TEXT NOT NULL,
  base_fare REAL NOT NULL,           -- per person, before taxes and bags
  bag_fee REAL NOT NULL,             -- per checked bag per person
  tax_rate REAL NOT NULL,
  refundable INTEGER NOT NULL,
  sponsored INTEGER NOT NULL,
  rank INTEGER NOT NULL              -- position in "recommended" order within its search
);
CREATE INDEX IF NOT EXISTS flights_search ON flights(episode_id, origin, dest, date);
CREATE TABLE IF NOT EXISTS bookings (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  episode_id TEXT NOT NULL REFERENCES episodes(id),
  flight_id INTEGER NOT NULL REFERENCES flights(id),
  passengers INTEGER NOT NULL,
  bags INTEGER NOT NULL,             -- checked bags per person
  total REAL NOT NULL,               -- as charged by the site
  created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS runs (
  id TEXT PRIMARY KEY,
  episode_id TEXT REFERENCES episodes(id),   -- set once the environment has generated it
  difficulty TEXT NOT NULL,          -- JSON: what was requested
  seed INTEGER NOT NULL,
  policy TEXT NOT NULL,
  model TEXT,
  max_steps INTEGER NOT NULL,
  status TEXT NOT NULL,              -- queued | running | done | error | stopped
  reward REAL,
  success INTEGER,
  n_steps INTEGER NOT NULL DEFAULT 0,
  input_tokens INTEGER NOT NULL DEFAULT 0,
  output_tokens INTEGER NOT NULL DEFAULT 0,
  outcome TEXT,                      -- JSON from the checker
  error TEXT,
  started_at REAL,
  ended_at REAL
);
CREATE TABLE IF NOT EXISTS steps (
  run_id TEXT NOT NULL REFERENCES runs(id),
  t INTEGER NOT NULL,
  url TEXT NOT NULL,
  page TEXT NOT NULL,                -- search | results | details | review | confirmation | other
  thought TEXT,
  notes TEXT,
  action TEXT NOT NULL,
  error TEXT,
  axtree TEXT,
  prompt_tokens INTEGER,
  output_tokens INTEGER,
  latency_ms INTEGER,
  PRIMARY KEY (run_id, t)
);
"""

_local = threading.local()


def connect() -> sqlite3.Connection:
    """One connection per thread; WAL so the website and dashboard can share the file."""
    conn = getattr(_local, "conn", None)
    if conn is None:
        DATA.mkdir(parents=True, exist_ok=True)
        SCREENS.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(DB_PATH, timeout=30, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.executescript(SCHEMA)
        _local.conn = conn
    return conn


def row(r: sqlite3.Row | None) -> dict | None:
    if r is None:
        return None
    d = dict(r)
    for k in ("difficulty", "task", "optimum", "outcome"):
        if k in d and isinstance(d[k], str):
            try:
                d[k] = json.loads(d[k])
            except json.JSONDecodeError:
                pass
    return d


def rows(rs) -> list[dict]:
    return [row(r) for r in rs]
