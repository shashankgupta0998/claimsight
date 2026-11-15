import json

import pytest

from claimsight.retrieval.base import RetrievedChunk
from claimsight.retrieval.rerank import RerankingRetriever
from claimsight.retrieval.rewrite import (
    PROMPT_VERSION,
    CassetteRewriter,
    ExpandedQueryRetriever,
    MissingRewrite,
    RewrittenQueryRetriever,
    joint_query,
)


class FakeRewriter:
    name = "fake-llm"

    def __init__(self, table: dict[str, str]):
        self.table, self.calls = table, 0

    def rewrite(self, query: str) -> str:
        self.calls += 1
        return self.table.get(query, query)


class TableRetriever:
    """Returns a fixed ranking per query, so tests can see which query each stage used."""

    name = "table"

    def __init__(self, table: dict[str, list[str]]):
        self.table, self.seen = table, []

    def search(self, query: str, k: int = 5) -> list[RetrievedChunk]:
        self.seen.append(query)
        ids = self.table.get(query, [])[:k]
        return [RetrievedChunk(cid, "d", 1, cid, 1.0) for cid in ids]


class RecordingReranker:
    name = "rr"

    def __init__(self, prefer: str):
        self.prefer, self.queries = prefer, []

    def score(self, query, texts):
        self.queries.append(query)
        return [1.0 if t == self.prefer else 0.0 for t in texts]


Q, R = "mice chewed through my wiring", "contact with a bird or animal"


def test_cassette_records_once_then_replays_offline(tmp_path):
    path = tmp_path / "rewrites.json"
    fake = FakeRewriter({Q: R})
    live = CassetteRewriter(path, fake)
    assert live.rewrite(Q) == R and live.rewrite(Q) == R
    assert fake.calls == 1  # second call came from the cassette

    data = json.loads(path.read_text())
    assert data["model"] == "fake-llm" and data["prompt"] == PROMPT_VERSION

    offline = CassetteRewriter(path)
    assert offline.rewrite(Q) == R
    with pytest.raises(MissingRewrite):
        offline.rewrite("never recorded")


def test_cassette_refuses_a_different_prompt_or_model(tmp_path):
    path = tmp_path / "rewrites.json"
    path.write_text(json.dumps({"model": "fake-llm", "prompt": "old", "rewrites": {}}))
    with pytest.raises(ValueError, match="re-record"):
        CassetteRewriter(path, FakeRewriter({}))


def test_prune_drops_unused_rewrites(tmp_path):
    path = tmp_path / "rewrites.json"
    live = CassetteRewriter(path, FakeRewriter({Q: R}))
    live.rewrite(Q)
    live.rewrite("stale question")
    fresh = CassetteRewriter(path, FakeRewriter({}))
    fresh.rewrite(Q)
    fresh.save(prune=True)
    assert len(json.loads(path.read_text())["rewrites"]) == 1


def test_replace_searches_with_the_rewrite_only():
    inner = TableRetriever({Q: ["WIRING"], R: ["ANIMAL"]})
    got = RewrittenQueryRetriever(inner, FakeRewriter({Q: R})).search(Q, k=1)
    assert [c.clause_id for c in got] == ["ANIMAL"] and inner.seen == [R]


def test_expand_fuses_both_rankings():
    inner = TableRetriever({Q: ["WIRING", "SHARED"], R: ["ANIMAL", "SHARED"]})
    got = [c.clause_id for c in ExpandedQueryRetriever(inner, FakeRewriter({Q: R})).search(Q)]
    assert got[0] == "SHARED"  # ranked by both searches
    assert set(got) == {"WIRING", "ANIMAL", "SHARED"}


def test_expand_skips_the_second_search_when_rewrite_is_unchanged():
    inner = TableRetriever({Q: ["WIRING"]})
    ExpandedQueryRetriever(inner, FakeRewriter({})).search(Q)
    assert inner.seen == [Q]


def test_joint_mode_reranks_against_question_plus_rewrite():
    rewriter = FakeRewriter({Q: R})
    inner = TableRetriever({Q: ["WIRING"], R: ["ANIMAL"]})
    reranker = RecordingReranker(prefer="ANIMAL")
    stack = RerankingRetriever(
        ExpandedQueryRetriever(inner, rewriter),
        reranker,
        rerank_query=lambda q: joint_query(q, rewriter.rewrite(q)),
    )
    assert stack.search(Q, k=1)[0].clause_id == "ANIMAL"
    assert reranker.queries == [f"{Q}\n{R}"]
    assert joint_query(Q, Q) == Q


def test_dual_adds_question_and_rewrite_rerank_scores():
    from claimsight.retrieval.rewrite import DualQueryReranker

    class TableReranker:
        name = "t"

        def score(self, query, texts):
            return [
                {(Q, "A"): 2.0, (Q, "B"): 0.0, (R, "A"): 0.0, (R, "B"): 3.0}[(query, t)]
                for t in texts
            ]

    dual = DualQueryReranker(TableReranker(), FakeRewriter({Q: R}))
    assert dual.score(Q, ["A", "B"]) == [2.0, 3.0]
    unchanged = DualQueryReranker(TableReranker(), FakeRewriter({}))
    assert unchanged.score(Q, ["A", "B"]) == [2.0, 0.0]
