"""Build the retrieval stack described by Settings.

    first stage  keyword | dense | hybrid      backend  memory | pgvector
    then         optional cross-encoder rerank of the top `rerank_pool`
    around both  optional LLM query rewriting: replace | expand | joint | dual

Models sit behind cassettes in evals/golden/cassettes/. Offline (replay, CI) reads them and
never loads a model; live mode or `record_retrieval` fills in anything missing.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from claimsight.config import Settings
from claimsight.retrieval.base import Retriever
from claimsight.retrieval.dense import DenseRetriever, HybridRetriever
from claimsight.retrieval.embed import CassetteEmbedder
from claimsight.retrieval.keyword import BM25Retriever
from claimsight.retrieval.rerank import CassetteReranker, RerankingRetriever
from claimsight.retrieval.rewrite import (
    CassetteRewriter,
    DualQueryReranker,
    ExpandedQueryRetriever,
    RewrittenQueryRetriever,
    joint_query,
)


@dataclass
class RetrievalStack:
    retriever: Retriever
    embedder: CassetteEmbedder | None = None
    reranker: CassetteReranker | None = None
    rewriter: CassetteRewriter | None = None

    def save_cassettes(self, prune: bool = False) -> None:
        for cassette in (self.embedder, self.reranker, self.rewriter):
            if cassette is not None:
                cassette.save(prune=prune)


def load_chunks(golden: Path) -> list[dict]:
    lines = (golden / "policy_clauses.jsonl").read_text().splitlines()
    return [json.loads(x) for x in lines if x.strip()]


def cassette_dir(s: Settings) -> Path:
    return Path(s.golden_dir) / "cassettes"


def build_embedder(s: Settings, online: bool) -> CassetteEmbedder:
    inner = None
    if online:
        from claimsight.retrieval.embed import SentenceTransformerEmbedder

        inner = SentenceTransformerEmbedder(s.embed_model)
    return CassetteEmbedder(cassette_dir(s) / "embeddings.json", inner)


def build_rewriter(s: Settings, online: bool) -> CassetteRewriter:
    inner = None
    if online:
        from claimsight.retrieval.rewrite import ClaudeQueryRewriter

        inner = ClaudeQueryRewriter(s.rewrite_model, s.rewrite_effort)
    return CassetteRewriter(cassette_dir(s) / "rewrites.json", inner)


def build_retrieval(
    s: Settings,
    online: bool = False,
    embedder: CassetteEmbedder | None = None,
    reranker: CassetteReranker | None = None,
    rewriter: CassetteRewriter | None = None,
) -> RetrievalStack:
    """`embedder` / `reranker` / `rewriter` reuse existing cassettes (the recorder shares
    one of each)."""
    online = online or s.record_retrieval
    chunks = load_chunks(Path(s.golden_dir))
    if s.retriever != "keyword" and embedder is None:
        embedder = build_embedder(s, online)
    if s.retriever == "keyword":
        embedder = None

    if s.vector_store == "pgvector":
        from claimsight.retrieval.pgvector import PgDenseRetriever, PgSparseRetriever, connect

        conn = connect(s.database_url)
        sparse: Retriever = PgSparseRetriever(conn)
        dense = PgDenseRetriever(conn, embedder, s.embed_query_prefix) if embedder else None
    else:
        sparse = BM25Retriever(chunks)
        dense = DenseRetriever(chunks, embedder, s.embed_query_prefix) if embedder else None

    first: Retriever = {
        "keyword": sparse,
        "dense": dense,
        "hybrid": HybridRetriever([dense, sparse], pool=s.rerank_pool) if dense else sparse,
    }[s.retriever]

    if s.query_rewrite == "off":
        rewriter = None
    elif rewriter is None:
        rewriter = build_rewriter(s, online)
    if s.query_rewrite in ("expand", "joint"):
        first = ExpandedQueryRetriever(first, rewriter, pool=s.rerank_pool)
    elif s.query_rewrite == "dual":
        # The first stage sees only the rewrite; the reranker below sees both.
        first = RewrittenQueryRetriever(first, rewriter)

    if s.rerank:
        if reranker is None:
            inner = None
            if online:
                from claimsight.retrieval.rerank import CrossEncoderReranker

                inner = CrossEncoderReranker(s.rerank_model)
            reranker = CassetteReranker(cassette_dir(s) / "rerank.json", inner)
        rerank_query = None
        if s.query_rewrite == "joint":
            rerank_query = lambda q: joint_query(q, rewriter.rewrite(q))  # noqa: E731
        scorer = DualQueryReranker(reranker, rewriter) if s.query_rewrite == "dual" else reranker
        first = RerankingRetriever(first, scorer, pool=s.rerank_pool, rerank_query=rerank_query)
    else:
        reranker = None
    if s.query_rewrite == "replace":
        first = RewrittenQueryRetriever(first, rewriter)
    return RetrievalStack(first, embedder, reranker, rewriter)
