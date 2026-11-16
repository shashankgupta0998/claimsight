"""Tracing against a recording fake: what a claim sends to Langfuse, and what it doesn't."""

from __future__ import annotations

import contextlib
import contextvars
import json

import pytest
from fastapi.testclient import TestClient

from claimsight import tracing
from claimsight.api.main import create_app
from claimsight.graph import build_graph

_current: contextvars.ContextVar = contextvars.ContextVar("obs", default=None)


class FakeLangfuse:
    def __init__(self):
        self.observations, self.scores, self.traces, self.flushed = [], [], [], 0

    @contextlib.contextmanager
    def propagate_attributes(self, **attrs):
        self.traces.append(attrs)
        yield

    @contextlib.contextmanager
    def start_as_current_observation(self, *, name, as_type="span", **fields):
        parent = _current.get()
        obs = {"name": name, "type": as_type, "parent": parent and parent["name"], **fields}
        obs["trace"] = parent["trace"] if parent else len(self.traces)
        self.observations.append(obs)
        token = _current.set(obs)
        try:
            yield obs
        finally:
            _current.reset(token)

    def get_current_trace_id(self):
        current = _current.get()
        return current and current["trace"]

    def score_current_trace(self, *, name, value, data_type):
        self.scores.append((self.get_current_trace_id(), name, value, data_type))

    def flush(self):
        self.flushed += 1


@pytest.fixture
def lf():
    fake = FakeLangfuse()
    tracing.set_backend(fake)
    yield fake
    tracing.set_backend(None)


def _run(deps, claim, claim_id):
    graph = build_graph(deps)
    c = claim(claim_id)
    with tracing.claim_trace(c.claim_id, c.loss_type.value, "replay", ["eval"]):
        packet = graph.invoke({"claim": c})["packet"]
        tracing.score_claim(packet, expected="approve")
    return packet


def test_a_claim_is_one_trace_with_a_span_per_agent(deps, claim, lf):
    _run(deps, claim, "CLM-0003")
    (attrs,) = lf.traces
    assert attrs["trace_name"] == "claim" and attrs["session_id"] == "CLM-0003"
    assert {"loss:collision", "mode:replay", "eval"} <= set(attrs["tags"])
    by_name = {o["name"]: o for o in lf.observations}
    assert by_name["claim"]["parent"] is None
    for agent in ("intake", "vision", "coverage", "verification", "estimator", "critic"):
        assert by_name[agent]["parent"] == "claim", agent  # parallel nodes nest too
    assert {o["trace"] for o in lf.observations} == {1}


def test_review_calls_are_generations_with_model_usage_and_cost(deps, claim, lf):
    packet = _run(deps, claim, "CLM-0003")
    gens = [o for o in lf.observations if o["type"] == "generation"]
    assert gens and all(g["name"] == "coverage_review" and g["parent"] == "coverage" for g in gens)
    assert all(g["model"].startswith("claude-") for g in gens)
    assert all(g["usage_details"]["input"] > 0 for g in gens)
    total = sum(g["cost_details"]["total"] for g in gens)
    assert total == pytest.approx(packet.coverage.review.cost_usd, abs=1e-6)


def test_claim_scores_feed_the_dashboards(deps, claim, lf):
    _run(deps, claim, "CLM-0003")
    scores = {name: (value, kind) for _, name, value, kind in lf.scores}
    assert scores["decision"] == ("approve", "CATEGORICAL")
    assert scores["escalated"] == (0.0, "BOOLEAN")
    assert scores["decision_correct"] == (1.0, "BOOLEAN")
    assert scores["review_route"][0] in ("cheap", "strong")
    assert scores["review_cost_usd"][1] == "NUMERIC" and scores["review_cost_usd"][0] > 0
    assert {t for t, *_ in lf.scores} == {1}  # all on the claim's trace


def test_claim_text_stays_out_unless_asked_for(deps, claim, lf, monkeypatch):
    _run(deps, claim, "CLM-0003")
    statement = claim("CLM-0003").statement
    assert statement not in json.dumps(lf.observations, default=str)
    assert all(g.get("output") is None for g in lf.observations if g["type"] == "generation")

    monkeypatch.setenv("CLAIMSIGHT_TRACE_CONTENT", "true")
    lf.observations.clear()
    _run(deps, claim, "CLM-0003")
    assert any(g.get("output") for g in lf.observations if g["type"] == "generation")


def test_review_outside_a_claim_makes_no_stray_trace(deps, lf):
    from evals.coverage_eval import load_cases, score_reviews

    score_reviews(deps, load_cases()[:3])
    assert lf.observations == [] and lf.traces == []


def test_api_traces_submit_and_review_in_one_session(deps, golden_cases, lf):
    client = TestClient(create_app(deps))
    client.post("/claims", json=golden_cases["CLM-0004"]["claim"])
    client.post("/claims/CLM-0004/review", json={"decision": "deny", "reviewer": "adj-1"})
    assert [t["session_id"] for t in lf.traces] == ["CLM-0004", "CLM-0004"]
    assert ("decision", "awaiting_review") in [(n, v) for _, n, v, _ in lf.scores]
    assert ("reviewed_by_adjuster", 1.0) in [(n, v) for _, n, v, _ in lf.scores]
    assert lf.flushed == 2


def test_tracing_off_is_a_no_op(deps, claim):
    tracing.set_backend(None)
    assert not tracing.enabled()
    assert _run(deps, claim, "CLM-0003").decision.value == "approve"


def test_every_name_the_dashboards_filter_on_is_emitted(deps, claim, golden_cases, lf):
    from pathlib import Path

    spec = json.loads((Path(__file__).parents[1] / "dashboards" / "claimsight.json").read_text())
    wanted = {
        (w["query"]["view"].startswith("scores"), f["value"])
        for board in spec["dashboards"]
        for w in board["widgets"]
        for f in w["query"]["filters"]
        if f["column"] == "name"
    }
    _run(deps, claim, "CLM-0003")
    client = TestClient(create_app(deps))
    client.post("/claims", json=golden_cases["CLM-0004"]["claim"])
    emitted = {(True, name) for _, name, _, _ in lf.scores}
    emitted |= {(False, o["name"]) for o in lf.observations}
    assert wanted <= emitted, wanted - emitted
