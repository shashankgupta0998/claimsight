"""Run every widget in dashboards/claimsight.json against your Langfuse project.

    make dashboards     # runs this with --check, keys from .env
    uv run --extra llm --env-file .env python scripts/langfuse_dashboards.py --days 1

Needs LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY and LANGFUSE_HOST in the environment.
Fill the project first, e.g. `make trace-eval` (the 257-case eval in replay, traced).

Prints each widget's rows. With --check it exits non-zero if any widget comes back empty,
which catches a renamed score or observation before a dashboard silently goes blank.

    uv run --extra llm --env-file .env python scripts/langfuse_dashboards.py --build
    uv run --extra llm --env-file .env python scripts/langfuse_dashboards.py --build --apply

--build turns the spec into Langfuse dashboards through the dashboard API (unstable since
Langfuse v3.216): one widget per spec entry, placed two to a row. Without --apply it only
prints what it would do. Rerunning updates the same widgets and dashboards in place: they
are found by name, and widgets also by the spec note in their description, so a widget of
yours that happens to share a title is left alone. Widgets it made that are no longer in the
spec are deleted. One metric per widget: Langfuse charts draw only the first.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / "dashboards" / "claimsight.json"


def widgets(spec: dict):
    for board in spec["dashboards"]:
        for w in board["widgets"]:
            yield board["name"], w


COUNT = {"measure": "count", "aggregation": "count"}


def query_for(widget: dict, start: datetime, end: datetime) -> dict:
    q = widget["query"]
    metrics = q["metrics"]
    # Time series pad empty buckets, so a bucket's own value can't say whether it had data
    # (an escalation rate of 0 is real). A row count can; the widget doesn't draw it.
    if "timeDimension" in q and COUNT not in metrics:
        metrics = [*metrics, COUNT]
    return {
        **q,
        "metrics": metrics,
        "fromTimestamp": start.isoformat(),
        "toTimestamp": end.isoformat(),
    }


def has_data(row: dict) -> bool:
    if "count_count" in row:
        return bool(row["count_count"])
    measures = [v for k, v in row.items() if "_" in k and k != "time_dimension"]
    return any(v not in (None, [], "") for v in measures)


def fmt(row: dict) -> str:
    parts = []
    for k, v in row.items():
        if isinstance(v, float):
            v = f"{v:.4g}"
        parts.append(f"{k}={v}")
    return ", ".join(parts)


# Spec chart -> Langfuse chart type. Time-series charts take their range and buckets from
# the dashboard, so the spec's `timeDimension` (metrics API only) isn't sent.
CHART_TYPES = {
    "stacked bar": "BAR_TIME_SERIES",
    "line": "LINE_TIME_SERIES",
    "histogram": "HISTOGRAM",
    "bar": "HORIZONTAL_BAR",
    "pie": "PIE",
    "number": "NUMBER",
}
NOTE = "(dashboards/claimsight.json)"
API = "/api/public/unstable"
TILE = 6  # the UI's default tile: half the 12-column grid, 6 rows high


def widget_body(widget: dict) -> dict:
    q = widget["query"]
    return {
        "name": widget["title"],
        "description": f"{widget['why']} {NOTE}",
        "view": q["view"],
        "dimensions": q["dimensions"],
        "metrics": [{"measure": m["measure"], "agg": m["aggregation"]} for m in q["metrics"]],
        "filters": q["filters"],
        "chartType": CHART_TYPES[widget["chart"]],
    }


def placements(widget_ids: list[str]) -> list[dict]:
    return [
        {
            "type": "widget",
            "id": f"claimsight-{i}",
            "widgetId": wid,
            "x": (i % 2) * TILE,
            "y": (i // 2) * TILE,
            "width": TILE,
            "height": TILE,
        }
        for i, wid in enumerate(widget_ids)
    ]


def list_all(client, path: str) -> list[dict]:
    items, page = [], 1
    while True:
        r = client.get(path, params={"page": page, "limit": 100})
        r.raise_for_status()
        body = r.json()
        items += body["data"]
        if page >= body["meta"].get("totalPages", 1):
            return items
        page += 1


def build(client, spec: dict, apply: bool) -> None:
    """Create or update every widget, then each dashboard with its widgets laid out."""
    ours = {
        w["name"]: w["id"]
        for w in list_all(client, f"{API}/dashboard-widgets")
        if w.get("description", "").endswith(NOTE)
    }
    boards = {d["name"]: d["id"] for d in list_all(client, f"{API}/dashboards")}
    for board in spec["dashboards"]:
        ids = []
        for w in board["widgets"]:
            body = widget_body(w)
            wid = ours.get(w["title"])
            print(f"{'update' if wid else 'create'} widget: {w['title']} ({body['chartType']})")
            if apply:
                r = (
                    client.patch(f"{API}/dashboard-widgets/{wid}", json=body)
                    if wid
                    else client.post(f"{API}/dashboard-widgets", json=body)
                )
                if r.is_error:
                    raise SystemExit(f"{w['title']}: {r.status_code} {r.text[:500]}")
                wid = r.json()["id"]
            ids.append(wid or f"<new {w['title']}>")
        did = boards.get(board["name"])
        print(f"{'update' if did else 'create'} dashboard: {board['name']} ({len(ids)} tiles)")
        if not apply:
            continue
        if not did:
            r = client.post(f"{API}/dashboards", json={"name": board["name"], "description": NOTE})
            r.raise_for_status()
            did = r.json()["id"]
        r = client.patch(
            f"{API}/dashboards/{did}", json={"definition": {"widgets": placements(ids)}}
        )
        if r.is_error:
            raise SystemExit(f"{board['name']}: {r.status_code} {r.text[:500]}")
        print(f"  {dashboard_url(client, did)}")
    # Ours but no longer in the spec (renamed or split): the dashboards above no longer
    # place them, so they can go. Widgets without the spec note are never touched.
    wanted = {w["title"] for _, w in widgets(spec)}
    for name, wid in ours.items():
        if name not in wanted:
            print(f"delete widget: {name} (no longer in the spec)")
            if apply:
                client.delete(f"{API}/dashboard-widgets/{wid}").raise_for_status()


def dashboard_url(client, dashboard_id: str) -> str:
    project = client.get("/api/public/projects").json()["data"][0]["id"]
    return f"{str(client.base_url).rstrip('/')}/project/{project}/dashboards/{dashboard_id}"


def retry_after(response) -> float:
    """Seconds Langfuse asks us to wait after a 429 (body first, then the header)."""
    try:
        return float(response.json()["details"]["retryAfterSeconds"])
    except (ValueError, KeyError, TypeError):
        return float(response.headers.get("retry-after", 30))


def api_client(tries: int = 5):
    """The unstable API allows 30 requests a minute: a rerun updating 13 widgets and two
    dashboards hits that, so wait as long as Langfuse says and try again."""
    import httpx

    class WaitOn429(httpx.HTTPTransport):
        def handle_request(self, request):
            for _ in range(tries - 1):
                response = super().handle_request(request)
                if response.status_code != 429:
                    return response
                response.read()
                wait = retry_after(response) + 1
                response.close()
                print(f"  rate limited; waiting {wait:.0f} s", flush=True)
                time.sleep(wait)
            return super().handle_request(request)

    host = os.environ.get("LANGFUSE_HOST") or os.environ.get("LANGFUSE_BASE_URL")
    keys = os.environ.get("LANGFUSE_PUBLIC_KEY"), os.environ.get("LANGFUSE_SECRET_KEY")
    if not host or not all(keys):
        raise SystemExit("Set LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY and LANGFUSE_HOST.")
    return httpx.Client(base_url=host, auth=keys, timeout=30, transport=WaitOn429())


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--days", type=float, default=7)
    ap.add_argument("--check", action="store_true", help="fail if a widget returns no rows")
    ap.add_argument("--rows", type=int, default=6, help="rows to print per widget")
    ap.add_argument("--build", action="store_true", help="create the dashboards in Langfuse")
    ap.add_argument("--apply", action="store_true", help="with --build: make the changes")
    args = ap.parse_args(argv)
    if args.build:
        with api_client() as client:
            build(client, json.loads(SPEC.read_text()), args.apply)
        if not args.apply:
            print("\nDry run: add --apply to make these changes.")
        return 0

    from langfuse import get_client
    from langfuse.api.core.api_error import ApiError

    lf = get_client()
    if not lf.auth_check():
        print("Langfuse rejected the keys (check LANGFUSE_* and LANGFUSE_HOST).")
        return 2
    spec = json.loads(SPEC.read_text())
    end = datetime.now(UTC)
    start = end - timedelta(days=args.days)
    empty = []
    for board, w in widgets(spec):
        try:
            rows = lf.api.metrics.metrics(
                query=json.dumps(query_for(w, start, end)),
                # The daily limit's retry-after is hours; fail now and say so instead.
                request_options={"max_retries": 0},
            ).data
        except ApiError as e:
            if e.status_code != 429:
                raise
            details = (e.body or {}).get("details", {}) if isinstance(e.body, dict) else {}
            print(
                f"\nLangfuse's metrics API limit is used up ({details.get('limit', '?')} "
                f"queries; resets {details.get('resetAt', 'later')}). Try again then."
            )
            return 3
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
