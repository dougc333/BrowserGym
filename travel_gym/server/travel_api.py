"""
The travel site's operations as plain functions, with exactly the website's rules:
same result ordering, same page size, same hidden-fee behavior, same "first booking
is final". The MCP server exposes these as tools; tests check parity with website.py.
"""

from __future__ import annotations

import time

from db import connect, row, rows
from generator import AIRPORT_NAMES, METROS, total_price


class TravelError(ValueError):
    pass


def _episode(ep: str) -> dict:
    r = row(connect().execute("SELECT id, difficulty, task FROM episodes WHERE id=?", (ep,)).fetchone())
    if r is None:
        raise TravelError("Unknown trip session.")
    return r


def _flight(ep: str, flight_id: int) -> dict:
    f = row(connect().execute("SELECT * FROM flights WHERE id=? AND episode_id=?", (flight_id, ep)).fetchone())
    if f is None:
        raise TravelError(f"No flight with id {flight_id}.")
    return f


def _stops(n: int) -> str:
    return "nonstop" if n == 0 else f"{n} stop{'s' if n > 1 else ''}"


def trip_request(ep: str) -> str:
    return _episode(ep)["task"]["goal"]


def airports() -> list[dict]:
    metro = {code: m for m, codes in METROS.items() for code in codes}
    return [{"code": c, "name": n, "metro_area": metro.get(c)} for c, n in sorted(AIRPORT_NAMES.items())]


def search(ep: str, origin: str, destination: str, date: str, travelers: int = 1, page: int = 1,
           sort: str = "recommended") -> dict:
    d = _episode(ep)["difficulty"]
    if origin not in AIRPORT_NAMES or destination not in AIRPORT_NAMES:
        raise TravelError("Unknown airport code. Use list_airports.")
    order = "base_fare, rank" if sort == "price" and d["sort_available"] else "rank"
    fl = rows(connect().execute(
        f"SELECT * FROM flights WHERE episode_id=? AND origin=? AND dest=? AND date=? ORDER BY {order}",
        (ep, origin, destination, date)))
    if not fl:
        return {"results": 0, "page": 1, "pages": 0, "flights": [], "message": f"No flights found on {date}."}
    size = d["page_size"]
    pages = (len(fl) + size - 1) // size
    page = min(max(1, page), pages)
    out = []
    for f in fl[(page - 1) * size: page * size]:
        item = {
            "flight_id": f["id"], "flight_no": f["flight_no"], "airline": f["airline"],
            "depart": f["depart"], "arrive": f["arrive"], "duration_min": f["duration_min"],
            "stops": _stops(f["stops"]), "cabin": f["cabin"], "refundable": bool(f["refundable"]),
            "sponsored": bool(f["sponsored"]), "fare_per_person": f["base_fare"],
        }
        if d["hidden_fees"]:
            item["price_note"] = "fare only; taxes and bag fees are added in price_quote"
        else:
            item["bag_fee_per_bag"] = f["bag_fee"]
            item["tax_rate_percent"] = round(f["tax_rate"] * 100, 1)
        out.append(item)
    res = {"results": len(fl), "page": page, "pages": pages, "travelers": travelers,
           "sort": "price" if order != "rank" else "recommended", "flights": out}
    if not d["sort_available"] and sort == "price":
        res["message"] = "Sorting by price is not available; showing recommended order."
    return res


def details(ep: str, flight_id: int, travelers: int = 1) -> dict:
    d = _episode(ep)["difficulty"]
    f = _flight(ep, flight_id)
    return {
        "flight_id": f["id"], "flight_no": f["flight_no"], "airline": f["airline"],
        "route": f"{f['origin']} -> {f['dest']}", "date": f["date"], "depart": f["depart"], "arrive": f["arrive"],
        "stops": _stops(f["stops"]), "cabin": f["cabin"], "fare_per_person": f["base_fare"],
        "checked_bag_fee_per_bag": f["bag_fee"], "refundable": bool(f["refundable"]),
        "taxes": "calculated in price_quote" if d["hidden_fees"] else f"{f['tax_rate'] * 100:.1f}% of fare and bag fees",
        "travelers": travelers,
    }


def quote(ep: str, flight_id: int, travelers: int, bags: int) -> dict:
    _episode(ep)
    f = _flight(ep, flight_id)
    if not 1 <= travelers <= 4 or not 0 <= bags <= 2:
        raise TravelError("travelers must be 1-4 and bags 0-2 per traveler.")
    fare = round(travelers * f["base_fare"], 2)
    bag_total = round(travelers * bags * f["bag_fee"], 2)
    total = total_price(f, travelers, bags)
    return {"flight_id": f["id"], "flight_no": f["flight_no"], "travelers": travelers, "bags_per_traveler": bags,
            "fare": fare, "checked_bags": bag_total, "taxes_and_fees": round(total - fare - bag_total, 2),
            "total_price": total}


def book(ep: str, flight_id: int, travelers: int, bags: int) -> dict:
    q = quote(ep, flight_id, travelers, bags)
    conn = connect()
    existing = row(conn.execute("SELECT * FROM bookings WHERE episode_id=? ORDER BY id LIMIT 1", (ep,)).fetchone())
    if existing:
        return {"status": "already_booked", "message": "This trip was already booked; only the first booking counts.",
                "confirmation": f"SC{existing['id']:06d}"}
    with conn:
        cur = conn.execute("INSERT INTO bookings (episode_id, flight_id, passengers, bags, total, created_at) "
                           "VALUES (?,?,?,?,?,?)", (ep, flight_id, travelers, bags, q["total_price"], time.time()))
    return {"status": "booked", "confirmation": f"SC{cur.lastrowid:06d}", **q}
