"""
The sandbox travel site as an MCP server (stdio). Same flights, same rules, same
exact checker as the browser version, but the agent calls tools instead of clicking.

  python mcp_server.py --episode <id>            serve an existing episode (the runner does this)
  python mcp_server.py --new hard --seed 7       generate a fresh episode and serve it
  python mcp_server.py --check <id>              print the checker's verdict for an episode

Use it from Claude Code (or any MCP client):
  claude mcp add travel-gym -- python /path/to/travel_gym/server/mcp_server.py --new hard --seed 7
  then: "Call get_trip_request and complete the request."

Like the website, nothing here reveals the optimum: prices come only from searching
and quoting.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import travel_api as api  # noqa: E402
from generator import PRESETS, check, generate_episode, get_episode  # noqa: E402

EPISODE: str = ""


def build_server():
    from mcp.server.mcpserver import MCPServer
    from mcp.server.mcpserver.exceptions import ToolError

    def call(fn, *args):
        """Expected failures (bad id, bad airport) go back to the agent as readable tool errors."""
        try:
            return fn(*args)
        except api.TravelError as e:
            raise ToolError(str(e)) from None

    server = MCPServer(
        "travel-gym",
        instructions=(
            "Sandbox flight booking. Call get_trip_request first. Search every allowed airport and date "
            "(and every results page), compare TRUE totals from price_quote, and book once: the first booking is final."
        ),
    )

    @server.tool(description="The traveler's request: requirements and what 'cheapest' means. Call this first.")
    def get_trip_request() -> str:
        return call(api.trip_request, EPISODE)

    @server.tool(description="All airport codes with names and metro areas.")
    def list_airports() -> list[dict]:
        return api.airports()

    @server.tool(description=(
        "Search one-way flights for one origin airport, destination airport and date (YYYY-MM-DD). "
        "Results are paginated: request page=2, 3, ... until page == pages. sort='price' sorts by listed fare "
        "when the site supports it. Listed fares may exclude taxes and bag fees."))
    def search_flights(origin: str, destination: str, date: str, travelers: int = 1, page: int = 1,
                       sort: str = "recommended") -> dict:
        return call(api.search, EPISODE, origin.upper(), destination.upper(), date, travelers, page, sort)

    @server.tool(description="Fare details for one flight: bag fee, refundability, and how taxes are charged.")
    def get_flight_details(flight_id: int, travelers: int = 1) -> dict:
        return call(api.details, EPISODE, flight_id, travelers)

    @server.tool(description="Itemized TRUE total for a flight with the given travelers and checked bags per traveler.")
    def price_quote(flight_id: int, travelers: int, bags_per_traveler: int) -> dict:
        return call(api.quote, EPISODE, flight_id, travelers, bags_per_traveler)

    @server.tool(description="Book a flight. FINAL: only the first booking for this trip counts.")
    def book_flight(flight_id: int, travelers: int, bags_per_traveler: int) -> dict:
        return call(api.book, EPISODE, flight_id, travelers, bags_per_traveler)

    return server


def main() -> None:
    global EPISODE
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--episode")
    g.add_argument("--new", choices=sorted(PRESETS))
    g.add_argument("--check")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    if a.check:
        print(json.dumps(check(a.check), indent=2))
        return
    if a.new:
        EPISODE = generate_episode(a.seed, PRESETS[a.new])["id"]
        print(f"travel-gym: serving new {a.new} episode {EPISODE} (check with --check {EPISODE})", file=sys.stderr)
    else:
        if get_episode(a.episode) is None:
            sys.exit(f"unknown episode {a.episode}")
        EPISODE = a.episode
    build_server().run("stdio")


if __name__ == "__main__":
    main()
