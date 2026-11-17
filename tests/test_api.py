import pytest
from fastapi.testclient import TestClient

from claimsight.api.main import create_app


@pytest.fixture
def client(deps):
    return TestClient(create_app(deps))


def test_clean_claim_is_decided(client, golden_cases):
    resp = client.post("/claims", json=golden_cases["CLM-0003"]["claim"])
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "decided"
    assert body["packet"]["decision"] == "approve"
    assert body["packet"]["fraud_flags"] == []


def test_escalated_claim_waits_for_review_then_resolves(client, golden_cases):
    resp = client.post("/claims", json=golden_cases["CLM-0004"]["claim"])
    assert resp.json()["status"] == "awaiting_review"
    assert resp.json()["review_request"]["proposed_decision"] == "escalate"

    resp = client.post(
        "/claims/CLM-0004/review",
        json={
            "decision": "approve",
            "reviewer": "adj-3",
            "note": "Front damage from a second incident.",
        },
    )
    body = resp.json()
    assert body["status"] == "decided"
    assert body["packet"]["decided_by"] == "adj-3"


def test_duplicate_submission_is_rejected(client, golden_cases):
    claim = golden_cases["CLM-0006"]["claim"]
    assert client.post("/claims", json=claim).status_code == 200
    assert client.post("/claims", json=claim).status_code == 409


def test_unknown_claim_is_404(client):
    assert client.get("/claims/NOPE").status_code == 404


def test_queued_status_reads_the_job_before_the_graph(deps, monkeypatch):
    """Load-test race: a worker finishing between the two reads made a decided claim 404."""
    from langgraph.checkpoint.memory import MemorySaver

    import claimsight.queue as q
    from claimsight.service import ClaimService, ClaimStatus

    finished = {"done": False}

    async def job_state(pool, claim_id):
        finished["done"] = True  # the worker completes right as Redis is asked
        return "done", None

    def status(self, claim_id):
        return ClaimStatus(claim_id=claim_id, status="decided") if finished["done"] else None

    monkeypatch.setattr(q, "job_state", job_state)
    monkeypatch.setattr(ClaimService, "status", status)
    s = deps.settings.model_copy(update={"queue": "redis"})
    app = create_app(type(deps)(**{**deps.__dict__, "settings": s}), MemorySaver(), object())
    resp = TestClient(app).get("/claims/LT-1")
    assert resp.status_code == 200 and resp.json()["status"] == "decided"


def test_api_key_guards_every_claims_route_but_not_health(deps, golden_cases):
    s = deps.settings.model_copy(update={"api_key": "s3cret"})
    client = TestClient(create_app(type(deps)(**{**deps.__dict__, "settings": s})))
    claim = golden_cases["CLM-0003"]["claim"]
    assert client.get("/healthz").status_code == 200
    assert client.post("/claims", json=claim).status_code == 401
    assert client.post("/claims", json=claim, headers={"X-API-Key": "wrong"}).status_code == 401
    assert client.get("/claims/CLM-0003").status_code == 401
    assert client.post("/claims/CLM-0003/review", json={}).status_code == 401
    ok = {"X-API-Key": "s3cret"}
    assert client.post("/claims", json=claim, headers=ok).status_code == 200
    assert client.get("/claims/CLM-0003", headers=ok).json()["status"] == "decided"


def test_database_url_also_reads_the_plain_name(monkeypatch):
    from claimsight.config import Settings

    monkeypatch.delenv("CLAIMSIGHT_DATABASE_URL", raising=False)
    monkeypatch.setenv("DATABASE_URL", "postgres://fly:pw@db.internal:5432/app")
    assert Settings(_env_file=None).database_url == "postgres://fly:pw@db.internal:5432/app"
    monkeypatch.setenv("CLAIMSIGHT_DATABASE_URL", "postgresql://mine/db")
    assert Settings(_env_file=None).database_url == "postgresql://mine/db"
