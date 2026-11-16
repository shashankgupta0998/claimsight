"""Run every widget in dashboards/claimsight.json against your Langfuse project.

    make dashboards     # runs this with --check, keys from .env
    uv run --extra llm --env-file .env python scripts/langfuse_dashboards.py --days 1

Needs LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY and LANGFUSE_HOST in the environment.
Fill the project first, e.g. `make trace-eval` (the 257-case eval in replay, traced).

Prints each widget's rows. With --check it exits non-zero if any widget comes back empty,
which catches a renamed score or observation before a dashboard silently goes blank.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / "dashboards" / "claimsight.json"


def widgets(spec: dict):
    for board in spec["dashboards"]:
        for w in board["widgets"]:
            yield board["name"], w


def query_for(widget: dict, start: datetime, end: datetime) -> dict:
    return {
        **widget["query"],
        "fromTimestamp": start.isoformat(),
        "toTimestamp": end.isoformat(),
    }


def has_data(row: dict) -> bool:
    measures = [v for k, v in row.items() if "_" in k and k != "time_dimension"]
    return any(v not in (None, 0, [], "") for v in measures)


def fmt(row: dict) -> str:
    parts = []
    for k, v in row.items():
        if isinstance(v, float):
            v = f"{v:.4g}"
        parts.append(f"{k}={v}")
    return ", ".join(parts)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--days", type=float, default=7)
    ap.add_argument("--check", action="store_true", help="fail if a widget returns no rows")
    ap.add_argument("--rows", type=int, default=6, help="rows to print per widget")
    args = ap.parse_args(argv)

    from langfuse import get_client

    lf = get_client()
    if not lf.auth_check():
        print("Langfuse rejected the keys (check LANGFUSE_* and LANGFUSE_HOST).")
        return 2
    spec = json.loads(SPEC.read_text())
    end = datetime.now(UTC)
    start = end - timedelta(days=args.days)
    empty = []
    for board, w in widgets(spec):
        rows = lf.api.metrics.metrics(query=json.dumps(query_for(w, start, end))).data
        rows = [r for r in rows if has_data(r)]  # time series pad empty buckets with 0 / None
        print(f"\n[{board}] {w['title']} ({w['chart']}): {len(rows)} rows")
        for row in rows[: args.rows]:
            print(f"  {fmt(row)}")
        if not rows:
            empty.append(w["title"])
    if empty:
        print(f"\nEmpty widgets: {', '.join(empty)}")
    return 1 if args.check and empty else 0


if __name__ == "__main__":
    sys.exit(main())
