"""In-memory dense and hybrid retrieval.

These mirror the Postgres lanes in `pgvector.py` (dense cosine, sparse keyword, RRF fusion)
so replay mode and CI can evaluate hybrid retrieval without a database. The pgvector
integration tests check the two implementations agree on the dense lane.
"""

from __future__ import annotations

from claimsight.retrieval.base import RetrievedChunk, Retriever
from claimsight.retrieval.embed import Embedder, cosine

RRF_K = 60


def chunk_to_result(chunk: dict, score: float) -> RetrievedChunk:
    return RetrievedChunk(
        clause_id=chunk["clause_id"],
        doc_id=chunk["doc_id"],
        page=chunk.get("page"),
        text=chunk["text"],
        score=round(score, 4),
        metadata=chunk.get("metadata", {}),
    )


class DenseRetriever:
    def __init__(self, chunks: list[dict], embedder: Embedder, query_prefix: str = ""):
        self._chunks = chunks
        self._embedder = embedder
        self._vectors = embedder.embed([c["text"] for c in chunks])
        self.query_prefix = query_prefix
        self.name = f"dense:{embedder.name}"

    def search(self, query: str, k: int = 5) -> list[RetrievedChunk]:
        (qvec,) = self._embedder.embed([self.query_prefix + query])
        scored = sorted(
            ((cosine(qvec, v), c) for v, c in zip(self._vectors, self._chunks, strict=True)),
            key=lambda x: x[0],
            reverse=True,
        )
        return [chunk_to_result(c, s) for s, c in scored[:k]]


def rrf_fuse(
    rankings: list[list[RetrievedChunk]], k: int, rrf_k: int = RRF_K
) -> list[RetrievedChunk]:
    """Reciprocal rank fusion: sum 1/(rrf_k + rank) over the lists each item appears in."""
    scores: dict[str, float] = {}
    first: dict[str, RetrievedChunk] = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking, start=1):
            scores[item.clause_id] = scores.get(item.clause_id, 0.0) + 1.0 / (rrf_k + rank)
            first.setdefault(item.clause_id, item)
    order = sorted(scores, key=lambda cid: (-scores[cid], cid))
    return [
        RetrievedChunk(**{**first[cid].__dict__, "score": round(scores[cid], 6)})
        for cid in order[:k]
    ]


class HybridRetriever:
    """Run each lane for a candidate pool, then fuse with RRF."""

    def __init__(self, lanes: list[Retriever], pool: int = 20, rrf_k: int = RRF_K):
        self.lanes, self.pool, self.rrf_k = lanes, pool, rrf_k
        self.name = "hybrid(" + "+".join(lane.name for lane in lanes) + ")"

    def search(self, query: str, k: int = 5) -> list[RetrievedChunk]:
        return rrf_fuse([lane.search(query, k=self.pool) for lane in self.lanes], k, self.rrf_k)
