# src/obg/cli_db_list.py
# Minimal standalone CLI: list experiments from MySQL using SQLAlchemy.
#
# Run:
#   python -m obg.cli_db_list --db "mysql+pymysql://user:pass@host:3306/obg" --limit 20
#
from __future__ import annotations

import argparse
from datetime import datetime
from typing import Optional

from sqlalchemy import create_engine, text


def list_experiments(db_url: str, limit: int = 50, contains: Optional[str] = None) -> None:
    engine = create_engine(db_url, pool_pre_ping=True)

    where = ""
    params = {"limit": limit}
    if contains:
        where = "WHERE name LIKE :q"
        params["q"] = f"%{contains}%"

    q = text(
        f"""
        SELECT id, name, created_at, git_commit
        FROM experiments
        {where}
        ORDER BY created_at DESC
        LIMIT :limit
        """
    )

    with engine.connect() as conn:
        rows = conn.execute(q, params).fetchall()

    if not rows:
        print("No experiments found.")
        return

    # Pretty print
    print(f"{'id':<8} {'created_at':<20} {'git':<10} name")
    print("-" * 80)
    for r in rows:
        created = r.created_at
        if isinstance(created, datetime):
            created_s = created.strftime("%Y-%m-%d %H:%M:%S")
        else:
            created_s = str(created)
        git = (r.git_commit or "")[:10]
        print(f"{r.id:<8} {created_s:<20} {git:<10} {r.name}")


def main() -> None:
    ap = argparse.ArgumentParser(description="List experiments in the OBG database.")
    ap.add_argument("--db", required=True, help="SQLAlchemy DB URL (mysql+pymysql://...)")
    ap.add_argument("--limit", type=int, default=50, help="Max rows to show")
    ap.add_argument("--contains", type=str, default=None, help="Filter by substring in name")
    args = ap.parse_args()

    list_experiments(args.db, limit=args.limit, contains=args.contains)


if __name__ == "__main__":
    main()