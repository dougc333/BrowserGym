"""
Tests for the generator, the exact checker, and reward-hacking probes against the site.
No browser or model needed:   python -m pytest -q test_travel.py
"""

import os
import tempfile

os.environ["TRAVEL_GYM_DATA"] = tempfile.mkdtemp(prefix="travel_gym_test_")

from fastapi.testclient import TestClient  # noqa: E402

import generator as g  # noqa: E402
from db import connect  # noqa: E402
from website import app  # noqa: E402

client = TestClient(app)


def book(ep, flight_id, travelers, bags):
    return client.post(f"/e/{ep}/book", data={"flight": flight_id, "travelers": travelers, "bags": bags},
                       follow_redirects=False)


def test_generation_is_reproducible_and_optimum_unique():
    a = g.generate_episode(42, g.PRESETS["hard"])
    b = g.generate_episode(42, g.PRESETS["hard"])
    assert a["task"]["goal"] == b["task"]["goal"]
    assert a["optimum"]["total"] == b["optimum"]["total"]
    t = a["task"]
    valid = [f for f in g.episode_flights(a["id"]) if not g.violations(f, t)]
    totals = sorted(g.total_price(f, t["passengers"], t["bags"]) for f in valid)
    assert totals[0] == a["optimum"]["total"] and totals[1] > totals[0]


def test_decoys_are_cheaper_but_invalid():
    ep = g.generate_episode(7, g.PRESETS["medium"])
    t, decoys = ep["task"], ep["difficulty"]["decoy_ids"]
    flights = {f["id"]: f for f in g.episode_flights(ep["id"])}
    for fid, kind in decoys.items():
        f = flights[int(fid)]
        if kind == "fee trap":  # valid, lowest listed fare, but more expensive in truth
            assert not g.violations(f, t)
            assert g.total_price(f, t["passengers"], t["bags"]) > ep["optimum"]["total"]
        else:
            assert g.violations(f, t) == [kind]
            assert g.total_price(f, t["passengers"], t["bags"]) < ep["optimum"]["total"]


def test_booking_the_optimum_scores_1():
    ep = g.generate_episode(3, g.PRESETS["medium"])
    t = ep["task"]
    assert book(ep["id"], ep["optimum"]["flight_id"], t["passengers"], t["bags"]).status_code == 303
    out = g.check(ep["id"])
    assert out["success"] and out["reward"] == 1.0 and out["regret"] == 0


def test_hack_book_a_cheaper_decoy():
    ep = g.generate_episode(4, g.PRESETS["medium"])
    t = ep["task"]
    decoy = next(int(i) for i, k in ep["difficulty"]["decoy_ids"].items() if k != "fee trap")
    book(ep["id"], decoy, t["passengers"], t["bags"])
    out = g.check(ep["id"])
    assert out["reward"] == 0 and not out["constraints_ok"]


def test_hack_wrong_travelers_or_bags_to_lower_the_price():
    for travelers_delta, bags in ((-1, None), (0, 0)):
        ep = g.generate_episode(5 + travelers_delta, g.PRESETS["medium"])
        t = ep["task"]
        tr = max(1, t["passengers"] + travelers_delta) if t["passengers"] > 1 else t["passengers"]
        bg = t["bags"] if bags is None else bags
        if (tr, bg) == (t["passengers"], t["bags"]):
            continue
        book(ep["id"], ep["optimum"]["flight_id"], tr, bg)
        assert g.check(ep["id"])["reward"] == 0


def test_hack_second_booking_does_not_count():
    ep = g.generate_episode(6, g.PRESETS["medium"])
    t = ep["task"]
    other = next(f["id"] for f in g.episode_flights(ep["id"]) if f["id"] != ep["optimum"]["flight_id"])
    book(ep["id"], other, t["passengers"], t["bags"])
    book(ep["id"], ep["optimum"]["flight_id"], t["passengers"], t["bags"])  # retry after seeing a total
    assert g.check(ep["id"])["booked_flight_id"] == other
    assert connect().execute("SELECT COUNT(*) FROM bookings WHERE episode_id=?", (ep["id"],)).fetchone()[0] == 1


def test_hack_fake_confirmation_page_or_cross_episode_flight():
    ep = g.generate_episode(8, g.PRESETS["easy"])
    other = g.generate_episode(9, g.PRESETS["easy"])
    assert client.get(f"/e/{ep['id']}/confirmation/999999").status_code == 404
    assert book(ep["id"], other["optimum"]["flight_id"], 1, 0).status_code == 404  # flight from another episode
    assert g.check(ep["id"])["booked"] is False


def test_site_never_reveals_the_answer():
    ep = g.generate_episode(10, g.PRESETS["medium"])
    t = ep["task"]
    pages = [client.get(f"/e/{ep['id']}/").text]
    for o in t["origins"]:
        for d in t["dates"]:
            pages.append(client.get(f"/e/{ep['id']}/results", params={"from": o, "to": t["dest"], "date": d,
                                                                    "travelers": t["passengers"]}).text)
    blob = "\n".join(pages).lower()
    assert "optimum" not in blob and "decoy" not in blob and "fee trap" not in blob
    assert client.get("/openapi.json").status_code == 404 and client.get("/docs").status_code == 404


def test_hidden_fees_only_on_review():
    ep = g.generate_episode(11, g.PRESETS["medium"])
    t, opt = ep["task"], ep["optimum"]
    f = next(x for x in g.episode_flights(ep["id"]) if x["id"] == opt["flight_id"])
    res = client.get(f"/e/{ep['id']}/results", params={"from": f["origin"], "to": t["dest"], "date": f["date"],
                                                       "travelers": t["passengers"], "page": opt["page"]}).text
    total = f"${opt['total']:,.2f}"
    assert total not in res
    rev = client.get(f"/e/{ep['id']}/review", params={"flight": f["id"], "travelers": t["passengers"], "bags": t["bags"]}).text
    assert f"Total price {total}" in rev
