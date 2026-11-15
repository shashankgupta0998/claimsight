from datetime import date

import pytest

from claimsight.agents.coverage import CoverageAgent
from claimsight.config import Settings
from claimsight.deps import build_replay_deps
from claimsight.retrieval.base import RetrievedChunk
from claimsight.retrieval.dense import DenseRetriever, rrf_fuse
from claimsight.retrieval.embed import (
    CassetteEmbedder,
    MissingEmbedding,
    decode_vector,
    encode_vector,
)
from claimsight.retrieval.rerank import (
    CassetteReranker,
    MissingRerankScore,
    RerankingRetriever,
)
from claimsight.schemas import ClaimInput, Location


def _chunk(cid: str, text: str = "") -> RetrievedChunk:
    return RetrievedChunk(clause_id=cid, doc_id="d", page=1, text=text or cid, score=0.0)


def test_rrf_rewards_agreement_between_lanes():
    a = [_chunk("x"), _chunk("y"), _chunk("z")]
    b = [_chunk("y"), _chunk("w")]
    fused = [c.clause_id for c in rrf_fuse([a, b], k=4)]
    assert fused[0] == "y"  # ranked by both lanes
    assert set(fused) == {"x", "y", "z", "w"}


def test_vector_roundtrip_is_float16_close():
    vec = [0.1, -0.25, 0.333, 0.9]
    assert decode_vector(encode_vector(vec)) == pytest.approx(vec, abs=1e-3)


class FakeEmbedder:
    name, dim = "fake", 2

    def __init__(self):
        self.calls = 0

    def embed(self, texts):
        self.calls += 1
        return [[1.0, 0.0] if "hail" in t else [0.0, 1.0] for t in texts]


def test_embedding_cassette_records_then_replays(tmp_path):
    path = tmp_path / "emb.json"
    live = CassetteEmbedder(path, FakeEmbedder())
    chunks = [
        {"clause_id": "H", "doc_id": "d", "text": "hail and windstorm"},
        {"clause_id": "C", "doc_id": "d", "text": "collision with an object"},
    ]
    assert DenseRetriever(chunks, live).search("hail dents", 1)[0].clause_id == "H"

    offline = CassetteEmbedder(path)  # no model
    assert DenseRetriever(chunks, offline).search("hail dents", 1)[0].clause_id == "H"
    with pytest.raises(MissingEmbedding):
        offline.embed(["never recorded"])


def test_prune_keeps_only_used_entries(tmp_path):
    path = tmp_path / "emb.json"
    c = CassetteEmbedder(path, FakeEmbedder())
    c.embed(["a", "b"])
    fresh = CassetteEmbedder(path, FakeEmbedder())
    fresh.embed(["a"])
    fresh.save(prune=True)
    offline = CassetteEmbedder(path)
    assert "a" in offline and "b" not in offline


class FakeReranker:
    name = "fake-ce"

    def score(self, query, texts):
        return [float(len(set(query.split()) & set(t.split()))) for t in texts]


class ListRetriever:
    name = "list"

    def __init__(self, chunks):
        self.chunks = chunks

    def search(self, query, k=5):
        return self.chunks[:k]


def test_reranker_reorders_and_replays(tmp_path):
    first = ListRetriever([_chunk("A", "deductible page"), _chunk("B", "hail damage to auto")])
    path = tmp_path / "rr.json"
    live = RerankingRetriever(first, CassetteReranker(path, FakeReranker()), pool=2)
    assert [c.clause_id for c in live.search("hail damage", 2)] == ["B", "A"]

    offline = RerankingRetriever(first, CassetteReranker(path), pool=2)
    assert [c.clause_id for c in offline.search("hail damage", 2)] == ["B", "A"]
    with pytest.raises(MissingRerankScore):
        offline.search("new query", 2)


@pytest.mark.parametrize(
    "statement",
    [
        "Someone keyed both doors while I was parked at the office.",
        "My car was vandalized overnight. The windshield is smashed.",
    ],
)
def test_coverage_retrieves_by_peril_not_statement_noise(statement):
    """Incidental details (keyed, windshield) used to pull the wrong clauses."""
    deps = build_replay_deps(Settings())
    claim = ClaimInput(
        claim_id="T-1",
        policy_id="P-100",
        loss_type="vandalism",
        loss_date=date(2026, 7, 1),
        location=Location(lat=30.0, lon=-97.0),
        statement=statement,
    )
    finding, retrieved = CoverageAgent(deps.retriever, deps.policies).run(claim, statement)
    assert {"PP-D-COMP", "PP-DEC-DED"} <= set(retrieved)
    assert finding.applies is True and finding.coverage_part == "comprehensive"
