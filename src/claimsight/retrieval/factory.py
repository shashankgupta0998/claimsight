"""Build the retrieval stack described by Settings.

    first stage  keyword | dense | hybrid      backend  memory | pgvector
    then         optional cross-encoder rerank of the top `rerank_pool`

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


@dataclass
class RetrievalStack:
    retriever: Retriever
    embedder: CassetteEmbedder | None = None
    reranker: CassetteReranker | None = None

    def save_cassettes(self, prune: bool = False) -> None:
        for cassette in (self.embedder, self.reranker):
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


def build_retrieval(
    s: Settings,
    online: bool = False,
    embedder: CassetteEmbedder | None = None,
    reranker: CassetteReranker | None = None,
) -> RetrievalStack:
    """`embedder` / `reranker` reuse existing cassettes (the recorder shares one of each)."""
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

    if s.rerank:
        if reranker is None:
            inner = None
            if online:
                from claimsight.retrieval.rerank import CrossEncoderReranker

                inner = CrossEncoderReranker(s.rerank_model)
            reranker = CassetteReranker(cassette_dir(s) / "rerank.json", inner)
        first = RerankingRetriever(first, reranker, pool=s.rerank_pool)
    else:
        reranker = None
    return RetrievalStack(first, embedder, reranker)
