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
