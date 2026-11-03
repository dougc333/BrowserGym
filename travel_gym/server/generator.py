"""
Task generation and checking for the sandbox travel site.

Every episode is generated from (seed, difficulty), so it is reproducible, and the
true answer is computed from the generated database, so checking is exact.

Difficulty knobs (they multiply the search the agent must do):
  n_dates           departure days in the allowed window        -> more searches
  n_origins         nearby airports allowed (1 = one airport)   -> more searches
  pages             result pages per search                     -> more paging
  page_size         results per page
  hidden_fees       bag fees and taxes only on details / review -> must open candidates
  sort_available    whether results can be sorted by price
  passengers        max travelers (total scales with this)
  bags              require one checked bag per traveler
  require_nonstop, depart_after, require_refundable          -> constraints to respect
  decoys            cheaper flights that each break exactly one constraint
"""

from __future__ import annotations

import json
import random
import time
import uuid
from dataclasses import asdict, dataclass
from datetime import date, timedelta

from db import connect, row, rows

METROS = {
    "San Francisco Bay Area": ["SFO", "OAK", "SJC"],
    "New York City": ["JFK", "EWR", "LGA"],
    "Los Angeles": ["LAX", "BUR", "LGB"],
    "Chicago": ["ORD", "MDW"],
    "Washington DC": ["DCA", "IAD", "BWI"],
}
AIRPORT_NAMES = {
    "SFO": "San Francisco (SFO)", "OAK": "Oakland (OAK)", "SJC": "San Jose (SJC)",
    "JFK": "New York JFK", "EWR": "Newark (EWR)", "LGA": "New York LaGuardia (LGA)",
    "LAX": "Los Angeles (LAX)", "BUR": "Burbank (BUR)", "LGB": "Long Beach (LGB)",
    "ORD": "Chicago O'Hare (ORD)", "MDW": "Chicago Midway (MDW)",
    "DCA": "Washington Reagan (DCA)", "IAD": "Washington Dulles (IAD)", "BWI": "Baltimore (BWI)",
}
# airline: (code, bag fee per bag, tax rate)
AIRLINES = {
    "Pacific Air": ("PA", 35.0, 0.075),
    "Coastline": ("CL", 40.0, 0.080),
    "Summit Airways": ("SM", 30.0, 0.090),
    "Bluejet": ("BJ", 45.0, 0.070),
    "Meridian": ("MD", 0.0, 0.110),
    "Northstar": ("NS", 50.0, 0.065),
}
CABINS = [("Economy", 0.78), ("Basic Economy", 0.10), ("Premium Economy", 0.08), ("Business", 0.04)]


@dataclass
class Difficulty:
    n_dates: int = 3
    n_origins: int = 1
    pages: int = 2
    page_size: int = 8
    hidden_fees: bool = True
    sort_available: bool = True
    passengers: int = 2
    bags: bool = True
    require_nonstop: bool = True
    depart_after: bool = True
    require_refundable: bool = False
    decoys: int = 4


PRESETS: dict[str, Difficulty] = {
    "easy": Difficulty(n_dates=1, n_origins=1, pages=1, page_size=10, hidden_fees=False, sort_available=True,
                       passengers=1, bags=False, require_nonstop=True, depart_after=False, decoys=2),
    "medium": Difficulty(n_dates=3, n_origins=1, pages=2, page_size=8, hidden_fees=True, sort_available=True,
                         passengers=2, bags=True, decoys=4),
    "hard": Difficulty(n_dates=5, n_origins=3, pages=3, page_size=8, hidden_fees=True, sort_available=False,
                       passengers=3, bags=True, decoys=6),
    "extreme": Difficulty(n_dates=7, n_origins=3, pages=4, page_size=6, hidden_fees=True, sort_available=False,
                          passengers=3, bags=True, require_refundable=True, decoys=8),
}


def total_price(f: dict, pax: int, bags: int) -> float:
    return round(pax * (f["base_fare"] + bags * f["bag_fee"]) * (1 + f["tax_rate"]), 2)


def violations(f: dict, task: dict) -> list[str]:
    v = []
    if f["origin"] not in task["origins"]:
        v.append("origin")
    if f["dest"] != task["dest"]:
        v.append("destination")
    if f["date"] not in task["dates"]:
        v.append("date")
    if task["nonstop"] and f["stops"] != 0:
        v.append("nonstop")
    if task["depart_after"] and f["depart"] < task["depart_after"]:
        v.append("departure time")
    if f["cabin"] != task["cabin"]:
        v.append("cabin")
    if task["refundable"] and not f["refundable"]:
        v.append("refundable")
    return v


def _fmt_time(hhmm: str) -> str:
    h, m = map(int, hhmm.split(":"))
    return f"{(h % 12) or 12}:{m:02d} {'AM' if h < 12 else 'PM'}"


def _goal(task: dict) -> str:
    o = task["origins"]
    if len(o) == 1:
        frm = AIRPORT_NAMES[o[0]]
    else:
        frm = f"any {task['origin_metro']} airport ({', '.join(o[:-1])} or {o[-1]})"
    d0, d1 = date.fromisoformat(task["dates"][0]), date.fromisoformat(task["dates"][-1])
    when = (f"on {d0:%b} {d0.day}, {d0.year}" if d0 == d1
            else f"departing any day from {d0:%b} {d0.day} to {d1:%b} {d1.day}, {d1.year}")
    parts = ["nonstop" if task["nonstop"] else None, task["cabin"].lower(), "refundable" if task["refundable"] else None]
    kind = " ".join(p for p in parts if p)
    after = f", leaving at or after {_fmt_time(task['depart_after'])}" if task["depart_after"] else ""
    who = f"{task['passengers']} adult{'s' if task['passengers'] > 1 else ''}"
    each = " each" if task["passengers"] > 1 else ""
    bags = f" with {task['bags']} checked bag{each}" if task["bags"] else " with no checked bags"
    return (f"Book the cheapest {kind} flight from {frm} to {AIRPORT_NAMES[task['dest']]}, {when}{after}, "
            f"for {who}{bags}. \"Cheapest\" means the full price including taxes and bag fees. "
            f"Do not book a flight that breaks any of these requirements.")


def generate_episode(seed: int, difficulty: Difficulty) -> dict:
    rng = random.Random(seed)
    d = difficulty
    metros = [m for m, a in METROS.items() if len(a) >= d.n_origins]
    origin_metro = rng.choice(metros)
    origins = rng.sample(METROS[origin_metro], d.n_origins)
    dest_metro = rng.choice([m for m in METROS if m != origin_metro])
    dest = rng.choice(METROS[dest_metro])
    start = date(2026, 12, 1) + timedelta(days=rng.randint(0, 20))
    dates = [(start + timedelta(days=i)).isoformat() for i in range(d.n_dates)]
    task = {
        "origins": origins, "origin_metro": origin_metro, "dest": dest, "dates": dates,
        "passengers": rng.randint(1, max(1, d.passengers)), "bags": 1 if d.bags else 0,
        "nonstop": d.require_nonstop, "cabin": "Economy", "refundable": d.require_refundable,
        "depart_after": rng.choice(["08:00", "09:00", "10:00", "12:00"]) if d.depart_after else None,
        "all_airports": sorted(AIRPORT_NAMES),
    }
    task["goal"] = _goal(task)
    pax, bags = task["passengers"], task["bags"]

    flights: list[dict] = []
    for o in origins:
        for dt in dates:
            n = d.pages * d.page_size - (rng.randint(0, d.page_size // 2) if d.pages > 1 else 0)
            batch = []
            for _ in range(n):
                airline = rng.choice(list(AIRLINES))
                code, bag_fee, tax = AIRLINES[airline]
                stops = rng.choices([0, 1, 2], [0.45, 0.4, 0.15])[0]
                cabin = rng.choices([c for c, _ in CABINS], [w for _, w in CABINS])[0]
                mult = {"Economy": 1.0, "Basic Economy": 0.8, "Premium Economy": 1.6, "Business": 2.8}[cabin]
                refundable = rng.random() < 0.3
                base = round(rng.uniform(110, 480) * mult * (1.15 if refundable else 1.0) * (0.85 if stops else 1.0))
                dep = rng.randrange(5 * 60, 22 * 60 + 30, 5)
                dur = 300 + stops * 95 + rng.randrange(0, 75, 5)
                arr = dep + dur
                batch.append(dict(
                    airline=airline, flight_no=f"{code}{rng.randint(100, 2999)}", origin=o, dest=dest, date=dt,
                    depart=f"{dep // 60:02d}:{dep % 60:02d}",
                    arrive=f"{(arr // 60) % 24:02d}:{arr % 60:02d}" + (" +1" if arr >= 1440 else ""),
                    duration_min=dur, stops=stops, cabin=cabin, base_fare=float(base),
                    bag_fee=bag_fee + rng.choice([0, 0, 5, 10]), tax_rate=round(tax + rng.uniform(-0.01, 0.02), 3),
                    refundable=int(refundable), sponsored=int(rng.random() < 0.12),
                ))
            flights.extend(batch)

    def valid(f):
        return not violations(f, task)

    # guarantee at least one valid flight
    if not any(valid(f) for f in flights):
        f = rng.choice(flights)
        f.update(stops=0, cabin="Economy", refundable=1,
                 depart=max(f["depart"], task["depart_after"] or "00:00"))

    # unique optimum: separate near-ties
    def ranked_valid():
        return sorted((f for f in flights if valid(f)), key=lambda f: total_price(f, pax, bags))

    vs = ranked_valid()
    for i in range(1, len(vs)):
        while total_price(vs[i], pax, bags) - total_price(vs[0], pax, bags) < 2.0:
            vs[i]["base_fare"] += 9.0
    best = ranked_valid()[0]
    best_total = total_price(best, pax, bags)

    # decoys: cheaper than the optimum but each breaks exactly one requirement
    kinds = (["nonstop"] if task["nonstop"] else []) + (["departure time"] if task["depart_after"] else []) \
        + ["cabin"] + (["refundable"] if task["refundable"] else [])
    pool = [f for f in flights if f is not best]
    rng.shuffle(pool)
    for i, f in enumerate(pool[: d.decoys]):
        kind = kinds[i % len(kinds)]
        f.update(stops=0, cabin="Economy", refundable=1 if task["refundable"] else f["refundable"])
        if task["depart_after"] and f["depart"] < task["depart_after"]:
            f["depart"] = task["depart_after"]
        if kind == "nonstop":
            f["stops"] = 1
        elif kind == "departure time":
            h = int(task["depart_after"][:2]) - rng.randint(1, 3)
            f["depart"] = f"{max(h, 5):02d}:{rng.randrange(0, 60, 5):02d}"
        elif kind == "cabin":
            f["cabin"] = "Basic Economy"
        elif kind == "refundable":
            f["refundable"] = 0
        target = best_total * rng.uniform(0.75, 0.93)
        f["base_fare"] = max(60.0, round(target / (pax * (1 + f["tax_rate"])) - bags * f["bag_fee"]))
        f["decoy"] = kind

    # fee trap: a valid flight with the lowest listed fare but a higher true total
    if d.hidden_fees:
        cand = [f for f in flights if valid(f) and f is not best]
        if cand:
            trap = rng.choice(cand)
            trap.update(bag_fee=85.0 if bags else trap["bag_fee"], tax_rate=0.16)
            trap["base_fare"] = best["base_fare"] - rng.randint(8, 25)
            while total_price(trap, pax, bags) <= best_total + 2:
                trap["base_fare"] += 3
                if not bags:
                    trap["tax_rate"] += 0.01
            trap["decoy"] = "fee trap"

    # recommended order within each search: sponsored first, then shuffled
    for o in origins:
        for dt in dates:
            group = [f for f in flights if f["origin"] == o and f["date"] == dt]
            rng.shuffle(group)
            group.sort(key=lambda f: -f["sponsored"])
            for r, f in enumerate(group):
                f["rank"] = r

    vs = ranked_valid()
    best = vs[0]
    assert all(total_price(f, pax, bags) > total_price(best, pax, bags) for f in vs[1:])
    cheaper_listed = sum(1 for f in vs if f["base_fare"] < best["base_fare"])

    ep_id = uuid.uuid4().hex[:12]
    conn = connect()
    with conn:
        conn.execute("INSERT INTO episodes VALUES (?,?,?,?,?,?)",
                     (ep_id, seed, json.dumps(asdict(d)), json.dumps(task), "{}", time.time()))
        for f in flights:
            cols = ["episode_id", "airline", "flight_no", "origin", "dest", "date", "depart", "arrive", "duration_min",
                    "stops", "cabin", "base_fare", "bag_fee", "tax_rate", "refundable", "sponsored", "rank"]
            cur = conn.execute(f"INSERT INTO flights ({','.join(cols)}) VALUES ({','.join('?' * len(cols))})",
                               [ep_id] + [f[c] for c in cols[1:]])
            f["id"] = cur.lastrowid
        optimum = {
            "flight_id": best["id"], "total": total_price(best, pax, bags), "origin": best["origin"],
            "date": best["date"], "page": best["rank"] // d.page_size + 1, "rank": best["rank"],
            "n_valid": len(vs), "n_flights": len(flights), "valid_with_lower_listed_fare": cheaper_listed,
            "searches_needed": len(origins) * len(dates),
        }
        conn.execute("UPDATE episodes SET optimum=? WHERE id=?", (json.dumps(optimum), ep_id))
        decoy_ids = {f["id"]: f["decoy"] for f in flights if "decoy" in f}
        conn.execute("UPDATE episodes SET difficulty=? WHERE id=?",
                     (json.dumps({**asdict(d), "decoy_ids": decoy_ids}), ep_id))
    return get_episode(ep_id)


def get_episode(ep_id: str) -> dict | None:
    return row(connect().execute("SELECT * FROM episodes WHERE id=?", (ep_id,)).fetchone())


def episode_flights(ep_id: str) -> list[dict]:
    return rows(connect().execute("SELECT * FROM flights WHERE episode_id=? ORDER BY origin, date, rank", (ep_id,)))


def check(ep_id: str) -> dict:
    """Exact outcome of an episode, from the database alone."""
    ep = get_episode(ep_id)
    task, opt = ep["task"], ep["optimum"]
    conn = connect()
    b = row(conn.execute("SELECT * FROM bookings WHERE episode_id=? ORDER BY id LIMIT 1", (ep_id,)).fetchone())
    if b is None:
        return {"booked": False, "reward": 0.0, "success": False, "optimal_total": opt["total"],
                "optimal_flight_id": opt["flight_id"], "constraints": [], "reason": "no booking"}
    f = row(conn.execute("SELECT * FROM flights WHERE id=?", (b["flight_id"],)).fetchone())
    v = violations(f, task)
    checks = [
        {"name": "origin", "ok": "origin" not in v, "expected": " / ".join(task["origins"]), "actual": f["origin"]},
        {"name": "destination", "ok": "destination" not in v, "expected": task["dest"], "actual": f["dest"]},
        {"name": "date", "ok": "date" not in v, "expected": f"{task['dates'][0]}..{task['dates'][-1]}", "actual": f["date"]},
        {"name": "nonstop", "ok": "nonstop" not in v, "expected": "nonstop" if task["nonstop"] else "any",
         "actual": "nonstop" if f["stops"] == 0 else f"{f['stops']} stop(s)"},
        {"name": "departure time", "ok": "departure time" not in v,
         "expected": f">= {task['depart_after']}" if task["depart_after"] else "any", "actual": f["depart"]},
        {"name": "cabin", "ok": "cabin" not in v, "expected": task["cabin"], "actual": f["cabin"]},
        {"name": "refundable", "ok": "refundable" not in v, "expected": "yes" if task["refundable"] else "any",
         "actual": "yes" if f["refundable"] else "no"},
        {"name": "passengers", "ok": b["passengers"] == task["passengers"], "expected": task["passengers"],
         "actual": b["passengers"]},
        {"name": "checked bags", "ok": b["bags"] == task["bags"], "expected": task["bags"], "actual": b["bags"]},
    ]
    all_ok = all(c["ok"] for c in checks)
    true_total = total_price(f, task["passengers"], task["bags"])
    optimal = all_ok and b["flight_id"] == opt["flight_id"]
    return {
        "booked": True, "success": optimal, "reward": 1.0 if optimal else 0.0,
        "constraints_ok": all_ok, "constraints": checks,
        "booked_flight_id": b["flight_id"], "booked_total": b["total"], "true_total_for_task": true_total,
        "optimal_flight_id": opt["flight_id"], "optimal_total": opt["total"],
        "regret": round(true_total - opt["total"], 2) if all_ok else None,
        "reason": "optimal" if optimal else ("constraint violated: " + ", ".join(c["name"] for c in checks if not c["ok"])
                                             if not all_ok else "valid but not the cheapest"),
    }
