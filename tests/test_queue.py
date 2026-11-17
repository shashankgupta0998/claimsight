"""Redis queue end to end: the API enqueues, an arq worker runs the graph, Postgres holds state.

Needs CLAIMSIGHT_TEST_DATABASE_URL and CLAIMSIGHT_TEST_REDIS_URL (CI's pgvector job sets
both). Each test gets its own Postgres schema and Redis database, flushed afterwards. The
worker runs in burst mode: it drains the queue, including retries, then stops.
"""

from __future__ import annotations

import os
import uuid
from urllib.parse import quote

import pytest
from fastapi.testclient import TestClient

from claimsight.api.main import create_app
from claimsight.checkpoint import make_checkpointer
from claimsight.graph import build_graph
from claimsight.queue import worker
from claimsight.service import ClaimService

DB_URL = os.environ.get("CLAIMSIGHT_TEST_DATABASE_URL")
REDIS_URL = os.environ.get("CLAIMSIGHT_TEST_REDIS_URL")
pytestmark = [
    pytest.mark.pg,
    pytest.mark.skipif(
        not (DB_URL and REDIS_URL), reason="needs CLAIMSIGHT_TEST_DATABASE_URL and _REDIS_URL"
    ),
]


@pytest.fixture
def env(deps):
    import psycopg
    import redis

    schema = f"queue_{uuid.uuid4().hex[:8]}"
    with psycopg.connect(DB_URL, autocommit=True) as conn:
        conn.execute(f"CREATE SCHEMA {schema}")
    sep = "&" if "?" in DB_URL else "?"
    db = f"{DB_URL}{sep}options={quote(f'-csearch_path={schema}')}"
    redis_db = f"{REDIS_URL.rstrip('/').rsplit('/', 1)[0]}/{9}"  # db 9: never a real one
    redis.Redis.from_url(redis_db).flushdb()
    settings = deps.settings.model_copy(
        update={
            "queue": "redis",
            "checkpointer": "postgres",
            "database_url": db,
            "redis_url": redis_db,
            "queue_retry_delay_s": 0,
        }
    )
    yield deps, settings
    redis.Redis.from_url(redis_db).flushdb()
    with psycopg.connect(DB_URL, autocommit=True) as conn:
        conn.execute(f"DROP SCHEMA {schema} CASCADE")


def _api(deps, settings):
    d = type(deps)(**{**deps.__dict__, "settings": settings})
    return TestClient(create_app(d))


def _drain(deps, settings, service=None):
    """Run a burst worker in its own process-like setup: its own pool and graph."""
    if service is None:
        d = type(deps)(**{**deps.__dict__, "settings": settings})
        graph = build_graph(d, checkpointer=make_checkpointer(settings), hitl=True)
        service = ClaimService(graph, settings.mode)
    w = worker(settings, service=service, burst=True, handle_signals=False)
    w.run()
    return w


def test_submit_returns_202_then_the_worker_decides(env, golden_cases):
    deps, s = env
    with _api(deps, s) as api:
        resp = api.post("/claims", json=golden_cases["CLM-0003"]["claim"])
        assert resp.status_code == 202 and resp.json()["status"] == "queued"
        assert api.get("/claims/CLM-0003").json()["status"] == "queued"
        w = _drain(deps, s)
        body = api.get("/claims/CLM-0003").json()
    assert w.jobs_complete == 1 and w.jobs_failed == 0
    assert body["status"] == "decided" and body["packet"]["decision"] == "approve"


def test_escalated_claim_waits_for_an_adjuster_through_the_queue(env, golden_cases):
    deps, s = env
    with _api(deps, s) as api:
        api.post("/claims", json=golden_cases["CLM-0004"]["claim"])
        _drain(deps, s)
        assert api.get("/claims/CLM-0004").json()["status"] == "awaiting_review"
        resp = api.post("/claims/CLM-0004/review", json={"decision": "deny", "reviewer": "adj-9"})
    assert resp.json()["status"] == "decided"
    assert resp.json()["packet"]["decided_by"] == "adj-9"


def test_a_claim_cannot_be_queued_twice(env, golden_cases):
    deps, s = env
    with _api(deps, s) as api:
        claim = golden_cases["CLM-0003"]["claim"]
        assert api.post("/claims", json=claim).status_code == 202
        assert api.post("/claims", json=claim).status_code == 409  # still queued
        w = _drain(deps, s)
        assert api.post("/claims", json=claim).status_code == 409  # decided
    assert w.jobs_complete == 1


class FlakyDetector:
    """Fails the first `fails` calls, then answers like the replay detector."""

    def __init__(self, inner, fails: int):
        self.inner, self.fails, self.calls, self.name = inner, fails, 0, inner.name

    def detect(self, photo):
        self.calls += 1
        if self.calls <= self.fails:
            raise ConnectionError("detector timed out")
        return self.inner.detect(photo)


class CountingReviewer:
    def __init__(self, inner):
        self.inner, self.calls, self.name = inner, 0, inner.name

    def review(self, request):
        self.calls += 1
        return self.inner.review(request)


def _service(deps, s, fails):
    flaky = FlakyDetector(deps.detector, fails)
    reviewer = CountingReviewer(deps.coverage_reviewer)
    d = type(deps)(
        **{**deps.__dict__, "settings": s, "detector": flaky, "coverage_reviewer": reviewer}
    )
    graph = build_graph(d, checkpointer=make_checkpointer(s), hitl=True)
    return ClaimService(graph, s.mode), flaky, reviewer


def test_a_failed_run_retries_from_its_checkpoint(env, golden_cases):
    deps, s = env
    service, flaky, reviewer = _service(deps, s, fails=1)
    with _api(deps, s) as api:
        api.post("/claims", json=golden_cases["CLM-0003"]["claim"])
        w = _drain(deps, s, service)
        body = api.get("/claims/CLM-0003").json()
    assert body["status"] == "decided" and body["packet"]["decision"] == "approve"
    assert w.jobs_retried == 1 and w.jobs_failed == 0
    # Coverage ran next to the vision node that failed; the retry didn't run it again.
    assert reviewer.calls == 1


def test_out_of_retries_reads_as_failed_with_the_error(env, golden_cases):
    deps, s = env
    service, flaky, _ = _service(deps, s, fails=99)
    with _api(deps, s) as api:
        api.post("/claims", json=golden_cases["CLM-0003"]["claim"])
        w = _drain(deps, s, service)
        body = api.get("/claims/CLM-0003").json()
    assert w.jobs_failed == 1 and flaky.calls == s.queue_max_tries
    assert body["status"] == "failed" and "detector timed out" in body["error"]


def test_queue_mode_refuses_the_in_memory_checkpointer(deps):
    s = deps.settings.model_copy(update={"queue": "redis", "checkpointer": "memory"})
    with pytest.raises(ValueError, match="CHECKPOINTER=postgres"):
        create_app(type(deps)(**{**deps.__dict__, "settings": s}))
