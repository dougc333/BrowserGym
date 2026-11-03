"""
The sandbox travel website the agent browses ("SkyCompare", a made-up brand).

Plain server-rendered HTML with proper labels, so the accessibility tree is clean.
Runs on its own port (default 8100). It reads an episode's flights and difficulty
settings and writes bookings. It never exposes the optimum, the decoy list, or any
answer: the agent can only learn prices by searching like a person would.

  /e/<ep>/                      search form
  /e/<ep>/results?...           one search: paginated, sponsored first; optional sort
  /e/<ep>/flight/<id>?...       fare details: bag fee, refundability
  /e/<ep>/review?...            itemized total including taxes and bags
  POST /e/<ep>/book             books (first booking per episode is final)
  /e/<ep>/confirmation/<id>     confirmation
"""

from __future__ import annotations

import html
import time
from datetime import date, timedelta
from urllib.parse import urlencode

from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse

from db import connect, row, rows
from generator import AIRPORT_NAMES, total_price

app = FastAPI(title="SkyCompare sandbox", docs_url=None, redoc_url=None, openapi_url=None)
e = html.escape

CSS = """
*{box-sizing:border-box} body{font:15px/1.45 system-ui,sans-serif;margin:0;background:#f5f6f8;color:#1c2330}
header{background:#123a6b;color:#fff;padding:12px 24px;display:flex;gap:24px;align-items:center}
header a{color:#fff} main{max-width:880px;margin:24px auto;padding:0 16px}
.card{background:#fff;border:1px solid #dde2ea;border-radius:8px;padding:16px;margin:0 0 12px}
label{display:block;font-weight:600;margin:8px 0 4px} select,button{font:inherit;padding:6px 10px}
button{background:#1d6fd8;color:#fff;border:0;border-radius:6px;cursor:pointer}
.flight{display:flex;justify-content:space-between;gap:16px;align-items:center}
.muted{color:#5b6573;font-size:13px} .price{font-weight:700;font-size:18px} .badge{background:#ffe9a8;padding:1px 6px;border-radius:4px;font-size:12px}
nav.pager{display:flex;gap:16px;align-items:center} table{border-collapse:collapse} td{padding:4px 12px 4px 0}
"""


def page(title: str, body: str, ep: str) -> HTMLResponse:
    return HTMLResponse(f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><title>{e(title)} · SkyCompare</title>
<style>{CSS}</style></head><body><header><strong>SkyCompare</strong><a href="/e/{ep}/">New search</a></header>
<main><h1>{e(title)}</h1>{body}</main></body></html>""")


def episode(ep: str) -> dict:
    r = row(connect().execute("SELECT id, difficulty, task FROM episodes WHERE id=?", (ep,)).fetchone())
    if r is None:
        raise HTTPException(404, "Unknown trip session")
    return r


def money(x: float) -> str:
    return f"${x:,.2f}"


def hm(minutes: int) -> str:
    return f"{minutes // 60}h {minutes % 60:02d}m"


def stops_label(n: int) -> str:
    return "Nonstop" if n == 0 else f"{n} stop{'s' if n > 1 else ''}"


@app.get("/e/{ep}/", response_class=HTMLResponse)
def search_form(ep: str):
    t = episode(ep)["task"]
    first = date.fromisoformat(t["dates"][0])
    days = [date(first.year, first.month, 1) + timedelta(days=i) for i in range(45)]
    opts = lambda codes: "".join(f'<option value="{c}">{e(AIRPORT_NAMES[c])}</option>' for c in codes)
    body = f"""<form class="card" method="get" action="/e/{ep}/results">
<label for="from">From</label><select id="from" name="from">{opts(t['all_airports'])}</select>
<label for="to">To</label><select id="to" name="to">{opts(t['all_airports'])}</select>
<label for="date">Departure date</label><select id="date" name="date">
{''.join(f'<option value="{d.isoformat()}">{d:%a %b} {d.day}, {d.year}</option>' for d in days)}</select>
<label for="travelers">Travelers</label><select id="travelers" name="travelers">
{''.join(f'<option value="{n}">{n} adult{"s" if n > 1 else ""}</option>' for n in range(1, 5))}</select>
<p><button type="submit">Search flights</button></p></form>"""
    return page("Search flights", body, ep)


@app.get("/e/{ep}/results", response_class=HTMLResponse)
def results_page(ep: str, request: Request):
    # read query params directly: "from" is a Python keyword, so it can't be a parameter name
    q = request.query_params
    origin, dest, day = q.get("from", ""), q.get("to", ""), q.get("date", "")
    travelers = int(q.get("travelers", "1") or 1)
    pg = max(1, int(q.get("page", "1") or 1))
    sort = q.get("sort", "recommended")
    ep_row = episode(ep)
    d = ep_row["difficulty"]
    order = "rank" if sort != "price" or not d["sort_available"] else "base_fare, rank"
    fl = rows(connect().execute(
        f"SELECT * FROM flights WHERE episode_id=? AND origin=? AND dest=? AND date=? ORDER BY {order}",
        (ep, origin, dest, day)))
    title = f"{AIRPORT_NAMES.get(origin, origin)} to {AIRPORT_NAMES.get(dest, dest)}"
    if not fl:
        return page(title, f'<div class="card">No flights found on {e(day)}. <a href="/e/{ep}/">Change search</a></div>', ep)
    size = d["page_size"]
    n_pages = (len(fl) + size - 1) // size
    pg = min(pg, n_pages)
    shown = fl[(pg - 1) * size: pg * size]
    base = {"from": origin, "to": dest, "date": day, "travelers": travelers, "sort": sort}
    link = lambda **o: f"/e/{ep}/results?" + urlencode({**base, **o})

    sort_ui = ""
    if d["sort_available"]:
        cur = "Lowest fare" if sort == "price" else "Recommended"
        sort_ui = (f'<p>Sorted by: <strong>{cur}</strong> · Sort by '
                   f'<a href="{link(sort="recommended", page=1)}">Recommended</a> | '
                   f'<a href="{link(sort="price", page=1)}">Lowest fare</a></p>')
    items = []
    for f in shown:
        if d["hidden_fees"]:
            price = (f'<div class="price">{money(f["base_fare"])}</div>'
                     f'<div class="muted">fare per person · taxes and bag fees added at checkout</div>')
        else:
            price = (f'<div class="price">{money(f["base_fare"])}</div><div class="muted">fare per person + '
                     f'{money(f["bag_fee"])} per checked bag + {f["tax_rate"] * 100:.1f}% taxes</div>')
        badge = '<span class="badge">Sponsored</span> ' if f["sponsored"] else ""
        fees = "" if d["hidden_fees"] else f", bag {money(f['bag_fee'])}, taxes {f['tax_rate'] * 100:.1f}%"
        name = (f"{'Sponsored: ' if f['sponsored'] else ''}Flight {f['flight_no']}, {f['airline']}, {f['depart']} to {f['arrive']}, "
                f"{stops_label(f['stops'])}, {f['cabin']}, {'Refundable' if f['refundable'] else 'Non-refundable'}, "
                f"fare {money(f['base_fare'])} per person{fees}")
        items.append(f"""<li class="card flight" aria-label="{e(name)}"><div>{badge}<strong>{e(f['airline'])} {e(f['flight_no'])}</strong>
<div>{f['depart']} → {e(f['arrive'])} · {stops_label(f['stops'])} · {hm(f['duration_min'])}</div>
<div class="muted">{e(f['cabin'])} · {'Refundable' if f['refundable'] else 'Non-refundable'}</div></div>
<div>{price}<a href="/e/{ep}/flight/{f['id']}?{urlencode({'travelers': travelers})}">View details for {e(f['flight_no'])}</a></div></li>""")
    pager = []
    if pg > 1:
        pager.append(f'<a href="{link(page=pg - 1)}">Previous page</a>')
    pager.append(f"<span>Page {pg} of {n_pages}</span>")
    if pg < n_pages:
        pager.append(f'<a href="{link(page=pg + 1)}">Next page</a>')
    body = (f'<p>{len(fl)} flights on {e(day)} for {travelers} traveler{"s" if travelers > 1 else ""}.</p>'
            f'{sort_ui}<ul style="list-style:none;padding:0">{"".join(items)}</ul><nav class="pager">{" ".join(pager)}</nav>')
    return page(title, body, ep)


def flight(ep: str, fid: int) -> dict:
    f = row(connect().execute("SELECT * FROM flights WHERE id=? AND episode_id=?", (fid, ep)).fetchone())
    if f is None:
        raise HTTPException(404, "Unknown flight")
    return f


@app.get("/e/{ep}/flight/{fid}", response_class=HTMLResponse)
def details(ep: str, fid: int, travelers: int = 1):
    d = episode(ep)["difficulty"]
    f = flight(ep, fid)
    taxes = "calculated on the review page" if d["hidden_fees"] else f"{f['tax_rate'] * 100:.1f}% of fare and bag fees"
    body = f"""<div class="card"><table>
<tr><td>Route</td><td>{e(AIRPORT_NAMES[f['origin']])} → {e(AIRPORT_NAMES[f['dest']])}</td></tr>
<tr><td>Date</td><td>{e(f['date'])}</td></tr><tr><td>Times</td><td>{f['depart']} → {e(f['arrive'])} ({hm(f['duration_min'])})</td></tr>
<tr><td>Stops</td><td>{stops_label(f['stops'])}</td></tr><tr><td>Cabin</td><td>{e(f['cabin'])}</td></tr>
<tr><td>Fare</td><td>{money(f['base_fare'])} per person</td></tr>
<tr><td>Checked bag fee</td><td>{money(f['bag_fee'])} per bag per person</td></tr>
<tr><td>Taxes</td><td>{taxes}</td></tr>
<tr><td>Changes and refunds</td><td>{'Refundable' if f['refundable'] else 'Non-refundable'}</td></tr></table></div>
<form class="card" method="get" action="/e/{ep}/review">
<input type="hidden" name="flight" value="{f['id']}"><input type="hidden" name="travelers" value="{travelers}">
<p>Travelers: {travelers}</p>
<label for="bags">Checked bags per traveler</label><select id="bags" name="bags">
<option value="0">No checked bags</option><option value="1">1 checked bag</option><option value="2">2 checked bags</option></select>
<p><button type="submit">Continue to review</button></p></form>"""
    return page(f"{f['airline']} {f['flight_no']}", body, ep)


@app.get("/e/{ep}/review", response_class=HTMLResponse)
def review(ep: str, flight: int, travelers: int = 1, bags: int = 0):
    episode(ep)
    f = row(connect().execute("SELECT * FROM flights WHERE id=? AND episode_id=?", (flight, ep)).fetchone())
    if f is None:
        raise HTTPException(404, "Unknown flight")
    fare = travelers * f["base_fare"]
    bag_total = travelers * bags * f["bag_fee"]
    total = total_price(f, travelers, bags)
    body = f"""<div class="card"><p><strong>{e(f['airline'])} {e(f['flight_no'])}</strong> · {e(f['origin'])} → {e(f['dest'])}
 · {e(f['date'])} · {f['depart']} → {e(f['arrive'])} · {stops_label(f['stops'])} · {e(f['cabin'])}</p><table>
<tr><td>Fare ({travelers} × {money(f['base_fare'])})</td><td>{money(fare)}</td></tr>
<tr><td>Checked bags ({travelers} × {bags} × {money(f['bag_fee'])})</td><td>{money(bag_total)}</td></tr>
<tr><td>Taxes and fees</td><td>{money(total - fare - bag_total)}</td></tr>
<tr><td><strong>Total price</strong></td><td aria-label="Total price {money(total)}"><strong>{money(total)}</strong></td></tr></table></div>
<form class="card" method="post" action="/e/{ep}/book">
<input type="hidden" name="flight" value="{f['id']}"><input type="hidden" name="travelers" value="{travelers}">
<input type="hidden" name="bags" value="{bags}">
<p>Booking is final and can only be made once for this trip.</p><button type="submit">Confirm booking</button></form>"""
    return page("Review your trip", body, ep)


@app.post("/e/{ep}/book")
def book(ep: str, flight: int = Form(...), travelers: int = Form(...), bags: int = Form(...)):
    episode(ep)
    conn = connect()
    f = row(conn.execute("SELECT * FROM flights WHERE id=? AND episode_id=?", (flight, ep)).fetchone())
    if f is None:
        raise HTTPException(404, "Unknown flight")
    existing = conn.execute("SELECT id FROM bookings WHERE episode_id=?", (ep,)).fetchone()
    if existing:
        return RedirectResponse(f"/e/{ep}/confirmation/{existing['id']}?already=1", status_code=303)
    with conn:
        cur = conn.execute("INSERT INTO bookings (episode_id, flight_id, passengers, bags, total, created_at) "
                           "VALUES (?,?,?,?,?,?)", (ep, flight, travelers, bags, total_price(f, travelers, bags), time.time()))
    return RedirectResponse(f"/e/{ep}/confirmation/{cur.lastrowid}", status_code=303)


@app.get("/e/{ep}/confirmation/{bid}", response_class=HTMLResponse)
def confirmation(ep: str, bid: int, already: int = 0):
    episode(ep)
    b = row(connect().execute("SELECT * FROM bookings WHERE id=? AND episode_id=?", (bid, ep)).fetchone())
    if b is None:
        raise HTTPException(404)
    f = flight(ep, b["flight_id"])
    note = "<p>This trip was already booked. Only the first booking counts.</p>" if already else ""
    body = (f'<div class="card">{note}<p>Confirmation number <strong>SC{b["id"]:06d}</strong></p>'
            f'<p>{e(f["airline"])} {e(f["flight_no"])} · {e(f["origin"])} → {e(f["dest"])} · {e(f["date"])} '
            f'{f["depart"]} · {b["passengers"]} traveler(s) · {b["bags"]} bag(s) each</p>'
            f'<p>Total charged: <strong>{money(b["total"])}</strong></p></div>')
    return page("Booking confirmed", body, ep)
