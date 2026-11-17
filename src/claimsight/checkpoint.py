"""Where the claims graph keeps its state between steps.

An escalated claim pauses at `human_review` until an adjuster answers, which can be days.
With the in-memory saver that pause lives only as long as the process: a deploy or a crash
loses every claim waiting for review. The Postgres saver (langgraph-checkpoint-postgres)
writes each step to the `checkpoints*` tables, so any API process can pick the thread up
again by its claim id.

    memory    MemorySaver: tests, replay evals, a single dev process
    postgres  PostgresSaver on a psycopg connection pool (needs the `pg` extra)

State holds our Pydantic models and enums. LangGraph rebuilds them from msgpack on resume
and is moving to refuse types that aren't allowlisted, so the serializer names every type
in claimsight.schemas explicitly (and nothing else).
"""

from __future__ import annotations

import enum
import inspect
import time

from pydantic import BaseModel

from claimsight import schemas
from claimsight.config import Settings


def state_types() -> list[tuple[str, str]]:
    """Every model and enum in claimsight.schemas, as (module, name) for the allowlist."""
    return sorted(
        (schemas.__name__, name)
        for name, obj in vars(schemas).items()
        if inspect.isclass(obj)
        and obj.__module__ == schemas.__name__
        and issubclass(obj, BaseModel | enum.Enum)
    )


def serializer():
    from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

    return JsonPlusSerializer(allowed_msgpack_modules=state_types())


SETUP_LOCK = 7_301_100  # pg_advisory_lock key for checkpoint-table setup
SETUP_LOCK_TIMEOUT_S = 120


def make_checkpointer(s: Settings, database_url: str | None = None):
    if s.checkpointer == "memory":
        from langgraph.checkpoint.memory import MemorySaver

        return MemorySaver(serde=serializer())

    from langgraph.checkpoint.postgres import PostgresSaver
    from psycopg.rows import dict_row
    from psycopg_pool import ConnectionPool

    # The saver needs autocommit and dict rows; prepare_threshold=0 keeps it working
    # behind a transaction-mode pooler (PgBouncer, Supabase).
    pool = ConnectionPool(
        database_url or s.database_url,
        min_size=1,
        max_size=s.checkpoint_pool_size,
        kwargs={"autocommit": True, "row_factory": dict_row, "prepare_threshold": 0},
        open=True,
    )
    saver = PostgresSaver(pool, serde=serializer())
    # setup() creates or migrates the checkpoint tables. It's idempotent but not safe to run
    # concurrently: replicas starting together on a fresh database race on CREATE TABLE and
    # all but one crash (found by the load test). An advisory lock runs them one at a time.
    # Try-and-sleep, not a blocking pg_advisory_lock: setup runs CREATE INDEX CONCURRENTLY,
    # which waits for every running statement, including a blocked lock call, and that
    # deadlocks (undetected) with the replicas waiting for the lock.
    with pool.connection() as conn:
        deadline = time.monotonic() + SETUP_LOCK_TIMEOUT_S
        while not conn.execute("SELECT pg_try_advisory_lock(%s)", (SETUP_LOCK,)).fetchone()[
            "pg_try_advisory_lock"
        ]:
            if time.monotonic() > deadline:
                raise TimeoutError("another process has held the checkpoint setup lock too long")
            time.sleep(0.2)
        try:
            saver.setup()
        finally:
            conn.execute("SELECT pg_advisory_unlock(%s)", (SETUP_LOCK,))
    return saver
