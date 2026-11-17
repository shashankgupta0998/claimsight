"""Checkpointing: a claim waiting for an adjuster must survive a restart.

The Postgres tests need CLAIMSIGHT_TEST_DATABASE_URL (CI's pgvector job sets it) and run in
a throwaway schema. "Restart" means a new connection pool and a new graph or API app: no
Python object is shared with the run that paused the claim.
"""

from __future__ import annotations

import logging
import os
import uuid
from urllib.parse import quote

import pytest
from fastapi.testclient import TestClient
from langgraph.types import Command

from claimsight.api.main import create_app
from claimsight.checkpoint import make_checkpointer, state_types
from claimsight.config import Settings
from claimsight.graph import build_graph
from claimsight.schemas import Decision

DB_URL = os.environ.get("CLAIMSIGHT_TEST_DATABASE_URL")
PG = Settings(checkpointer="postgres")


def test_allowlist_covers_every_type_in_graph_state():
    names = {name for _, name in state_types()}
    assert {"ClaimInput", "CoverageFinding", "DecisionPacket", "Decision", "LossType"} <= names


def test_memory_saver_round_trips_without_unregistered_types(deps, claim, caplog):
    graph = build_graph(deps, checkpointer=make_checkpointer(Settings()), hitl=True)
    config = {"configurable": {"thread_id": "CLM-0004"}}
    with caplog.at_level(logging.WARNING):
        graph.invoke({"claim": claim("CLM-0004")}, config)
        graph.invoke(Command(resume={"decision": "approve", "reviewer": "adj-1"}), config)
        packet = graph.get_state(config).values["packet"]
    assert packet.decided_by == "adj-1"
    assert "unregistered type" not in caplog.text


@pytest.fixture
def pg_url():
    import psycopg

    schema = f"ckpt_{uuid.uuid4().hex[:8]}"
    with psycopg.connect(DB_URL, autocommit=True) as conn:
        conn.execute(f"CREATE SCHEMA {schema}")
    sep = "&" if "?" in DB_URL else "?"
    yield f"{DB_URL}{sep}options={quote(f'-csearch_path={schema}')}"
    with psycopg.connect(DB_URL, autocommit=True) as conn:
        conn.execute(f"DROP SCHEMA {schema} CASCADE")


@pytest.mark.pg
@pytest.mark.skipif(not DB_URL, reason="CLAIMSIGHT_TEST_DATABASE_URL not set")
def test_paused_claim_resumes_after_a_restart(deps, claim, pg_url, caplog):
    config = {"configurable": {"thread_id": "CLM-0004"}}
    first = make_checkpointer(PG, pg_url)
    build_graph(deps, checkpointer=first, hitl=True).invoke({"claim": claim("CLM-0004")}, config)
    first.conn.close()  # the process that paused the claim is gone

    second = make_checkpointer(PG, pg_url)
    graph = build_graph(deps, checkpointer=second, hitl=True)
    with caplog.at_level(logging.WARNING):
        snap = graph.get_state(config)
        assert snap.next == ("human_review",)
        assert snap.tasks[0].interrupts[0].value["proposed_decision"] == "escalate"
        graph.invoke(Command(resume={"decision": "deny", "reviewer": "adj-17"}), config)
        packet = graph.get_state(config).values["packet"]
    second.conn.close()
    assert packet.decision == Decision.DENY and packet.decided_by == "adj-17"
    assert "unregistered type" not in caplog.text


@pytest.mark.pg
@pytest.mark.skipif(not DB_URL, reason="CLAIMSIGHT_TEST_DATABASE_URL not set")
def test_review_through_a_second_api_process(deps, golden_cases, pg_url):
    first = make_checkpointer(PG, pg_url)
    resp = TestClient(create_app(deps, first)).post(
        "/claims", json=golden_cases["CLM-0004"]["claim"]
    )
    assert resp.json()["status"] == "awaiting_review"
    first.conn.close()

    second = make_checkpointer(PG, pg_url)
    other = TestClient(create_app(deps, second))
    assert other.get("/claims/CLM-0004").json()["status"] == "awaiting_review"
    resp = other.post("/claims/CLM-0004/review", json={"decision": "approve", "reviewer": "adj-3"})
    assert resp.json()["status"] == "decided"
    assert resp.json()["packet"]["decided_by"] == "adj-3"
    # A third process sees the decision, and can't submit the claim again.
    third = make_checkpointer(PG, pg_url)
    client = TestClient(create_app(deps, third))
    assert client.get("/claims/CLM-0004").json()["packet"]["decision"] == "approve"
    assert client.post("/claims", json=golden_cases["CLM-0004"]["claim"]).status_code == 409
    second.conn.close()
    third.conn.close()


@pytest.mark.pg
@pytest.mark.skipif(not DB_URL, reason="CLAIMSIGHT_TEST_DATABASE_URL not set")
def test_replicas_starting_together_on_a_fresh_database_all_come_up(pg_url):
    # Found by the load test: concurrent setup() raced on CREATE TABLE and crashed workers.
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(8) as pool:
        savers = list(pool.map(lambda _: make_checkpointer(PG, pg_url), range(8)))
    for saver in savers:
        saver.conn.close()
