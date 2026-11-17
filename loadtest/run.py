"""Run the load-test matrix: start the API (and workers), drive it with Locust, collect.

    uv run --extra load --extra pg --extra queue python loadtest/run.py \\
        --db postgresql://user@localhost:5432/db --redis redis://localhost:6379/0

Scenarios (replay mode, so no model calls):

    inline-memory    one API process, in-memory checkpointer (the default config)
    inline-postgres  API processes share state through Postgres
    queued           API enqueues to Redis, N arq workers run the graph (Postgres state)

Each run gets a fresh Postgres schema and Redis database, dropped afterwards. Locust runs on
the same machine as the service, so it competes for CPU: compare scenarios with each other,
not with a production box. Results: evals/reports/loadtest.json and a table on stdout.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
import urllib.request
import uuid
from pathlib import Path
from urllib.parse import quote

ROOT = Path(__file__).resolve().parents[1]
PORT = 8790

# name, api processes, workers, users
MATRIX = [
    ("inline-memory", 1, 0, 1),
    ("inline-memory", 1, 0, 10),
    ("inline-memory", 1, 0, 40),
    ("inline-postgres", 1, 0, 10),
    ("inline-postgres", 1, 0, 40),
    ("inline-postgres", 4, 0, 40),
    ("queued", 1, 1, 40),
    ("queued", 1, 4, 40),
    ("queued", 4, 1, 40),
    ("queued", 4, 4, 40),
]


def wait_up(url: str, timeout: float = 60) -> None:
    end = time.time() + timeout
    while time.time() < end:
        try:
            urllib.request.urlopen(url, timeout=1)
            return
        except OSError:
            time.sleep(0.5)
    raise RuntimeError(f"{url} never came up")


def stats(prefix: Path) -> dict:
    rows = {r["Name"]: r for r in csv.DictReader(open(f"{prefix}_stats.csv"))}

    def row(name: str) -> dict | None:
        r = rows.get(name)
        if not r or int(r["Request Count"]) == 0:
            return None
        return {
            "count": int(r["Request Count"]),
            "failures": int(r["Failure Count"]),
            "rps": round(float(r["Requests/s"]), 1),
            "p50": float(r["50%"]),
            "p95": float(r["95%"]),
            "p99": float(r["99%"]),
        }

    return {
        "submit": row("/claims"),
        "decided": row("claim decided"),
        "review": row("/claims/[id]/review"),
        "all": row("Aggregated"),
    }


def run(name: str, api_procs: int, workers: int, users: int, args, out: Path) -> dict:
    import psycopg
    import redis

    env = {**os.environ, "CLAIMSIGHT_MODE": "replay"}
    for k in ("LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY"):  # don't trace load tests
        env.pop(k, None)
    schema = None
    if name != "inline-memory":
        schema = f"load_{uuid.uuid4().hex[:8]}"
        with psycopg.connect(args.db, autocommit=True) as conn:
            conn.execute(f"CREATE SCHEMA {schema}")
        sep = "&" if "?" in args.db else "?"
        env["CLAIMSIGHT_DATABASE_URL"] = f"{args.db}{sep}options={quote(f'-csearch_path={schema}')}"
        env["CLAIMSIGHT_CHECKPOINTER"] = "postgres"
    if name == "queued":
        redis_db = f"{args.redis.rstrip('/').rsplit('/', 1)[0]}/7"
        redis.Redis.from_url(redis_db).flushdb()
        env["CLAIMSIGHT_QUEUE"] = "redis"
        env["CLAIMSIGHT_REDIS_URL"] = redis_db

    logs = out.parent / "loadtest_logs"
    logs.mkdir(exist_ok=True)
    procs = []
    try:
        api = [sys.executable, "-m", "uvicorn", "claimsight.api.main:app", "--port", str(PORT)]
        api += ["--workers", str(api_procs), "--log-level", "warning"]
        procs.append(
            subprocess.Popen(
                api, env=env, cwd=ROOT, stdout=open(logs / "api.log", "w"), stderr=subprocess.STDOUT
            )
        )
        for i in range(workers):
            w = [sys.executable, "-m", "claimsight.queue"]
            procs.append(
                subprocess.Popen(
                    w,
                    env=env,
                    cwd=ROOT,
                    stdout=open(logs / f"worker{i}.log", "w"),
                    stderr=subprocess.STDOUT,
                )
            )
        wait_up(f"http://localhost:{PORT}/healthz")
        prefix = logs / f"{name}-a{api_procs}-w{workers}-u{users}"
        locust = [
            sys.executable,
            "-m",
            "locust",
            "-f",
            str(ROOT / "loadtest" / "locustfile.py"),
            "--headless",
            "-u",
            str(users),
            "-r",
            str(users),
            "-t",
            args.duration,
            "--host",
            f"http://localhost:{PORT}",
            "--csv",
            str(prefix),
            "--only-summary",
        ]
        subprocess.run(locust, cwd=ROOT, check=False, capture_output=True)
        result = {"scenario": name, "api_procs": api_procs, "workers": workers, "users": users}
        return result | stats(prefix)
    finally:
        for p in procs:
            p.terminate()
        for p in procs:
            p.wait(timeout=30)
        if schema:
            with psycopg.connect(args.db, autocommit=True) as conn:
                conn.execute(f"DROP SCHEMA {schema} CASCADE")
        if name == "queued":
            redis.Redis.from_url(redis_db).flushdb()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--db", required=True, help="Postgres URL (a schema per run is created)")
    ap.add_argument("--redis", required=True, help="Redis URL (db 7 is used and flushed)")
    ap.add_argument("--duration", default="30s")
    ap.add_argument("--only", help="run only scenarios whose name contains this")
    ap.add_argument("--out", type=Path, default=ROOT / "evals" / "reports" / "loadtest.json")
    args = ap.parse_args(argv)

    results = []
    for name, api_procs, workers, users in MATRIX:
        if args.only and args.only not in name:
            continue
        r = run(name, api_procs, workers, users, args, args.out)
        results.append(r)
        m = r["decided"] or r["submit"] or {}
        fails = (r["all"] or {}).get("failures", "-")
        print(
            f"{name:<16} api={api_procs} workers={workers} users={users:<3} "
            f"claims/s={m.get('rps', 0):<7} p50={m.get('p50', '-')} p95={m.get('p95', '-')} "
            f"p99={m.get('p99', '-')} fails={fails}",
            flush=True,
        )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
