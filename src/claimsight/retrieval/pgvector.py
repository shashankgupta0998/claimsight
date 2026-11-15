"""Policy retrieval on Postgres: pgvector HNSW for the dense lane, tsvector for the sparse
lane. Needs the `pg` extra and the schema in infra/sql/001_schema.sql.

The lanes return the same `RetrievedChunk`s as the in-memory retrievers, so fusion
(`dense.HybridRetriever`) and reranking (`rerank.RerankingRetriever`) work unchanged on
top of either backend.
"""

from __future__ import annotations

import json
import re

from claimsight.retrieval.base import RetrievedChunk
from claimsight.retrieval.embed import Embedder

_DENSE_SQL = """
SELECT clause_id, doc_id, page, text, metadata, 1 - (embedding <=> %(qvec)s) AS score
FROM policy_chunks
WHERE %(doc_id)s::text IS NULL OR doc_id = %(doc_id)s
ORDER BY embedding <=> %(qvec)s
LIMIT %(k)s
"""

_SPARSE_SQL = """
SELECT clause_id, doc_id, page, text, metadata, ts_rank_cd(tsv, q) AS score
FROM policy_chunks, to_tsquery('english', %(tsq)s) AS q
WHERE tsv @@ q AND (%(doc_id)s::text IS NULL OR doc_id = %(doc_id)s)
ORDER BY score DESC, clause_id
LIMIT %(k)s
"""

_UPSERT_DOC = """
INSERT INTO policy_docs (doc_id, title) VALUES (%s, %s)
ON CONFLICT (doc_id) DO UPDATE SET title = EXCLUDED.title, ingested_at = now()
"""

_UPSERT_CHUNK = """
INSERT INTO policy_chunks (clause_id, doc_id, page, text, metadata, embedding)
VALUES (%s, %s, %s, %s, %s, %s)
ON CONFLICT (clause_id) DO UPDATE SET
    doc_id = EXCLUDED.doc_id, page = EXCLUDED.page, text = EXCLUDED.text,
    metadata = EXCLUDED.metadata, embedding = EXCLUDED.embedding
"""


def to_or_tsquery(text: str) -> str:
    """Claimant statements are long; AND semantics would match nothing, so OR the terms."""
    terms = sorted(set(re.findall(r"[a-z0-9]{3,}", text.lower())))
    return " | ".join(terms) or "''"


def connect(database_url: str):
    import psycopg
    from pgvector.psycopg import register_vector

    conn = psycopg.connect(database_url, autocommit=True)
    register_vector(conn)
    return conn


def _row(r) -> RetrievedChunk:
    return RetrievedChunk(
        clause_id=r[0], doc_id=r[1], page=r[2], text=r[3], metadata=r[4], score=float(r[5])
    )


def ingest(conn, chunks: list[dict], embedder: Embedder, titles: dict[str, str] | None = None):
    """Upsert policy documents and their clause chunks with embeddings. Idempotent."""
    import numpy as np

    vectors = embedder.embed([c["text"] for c in chunks])
    with conn.cursor() as cur:
        for doc_id in sorted({c["doc_id"] for c in chunks}):
            cur.execute(_UPSERT_DOC, (doc_id, (titles or {}).get(doc_id, doc_id)))
        for chunk, vec in zip(chunks, vectors, strict=True):
            cur.execute(
                _UPSERT_CHUNK,
                (
                    chunk["clause_id"],
                    chunk["doc_id"],
                    chunk.get("page"),
                    chunk["text"],
                    json.dumps(chunk.get("metadata", {})),
                    np.asarray(vec, dtype=np.float32),
                ),
            )
    return len(chunks)


class PgDenseRetriever:
    def __init__(self, conn, embedder: Embedder, query_prefix: str = "", ef_search: int = 100):
        self._conn, self._embedder, self.query_prefix = conn, embedder, query_prefix
        self.name = f"pg-dense:{embedder.name}"
        with conn.cursor() as cur:
            # Default ef_search (40) can drop true neighbours; recall matters more than speed here.
            cur.execute(f"SET hnsw.ef_search = {int(ef_search)}")

    def search(self, query: str, k: int = 5, doc_id: str | None = None) -> list[RetrievedChunk]:
        import numpy as np

        (qvec,) = self._embedder.embed([self.query_prefix + query])
        params = {"qvec": np.asarray(qvec, dtype=np.float32), "k": k, "doc_id": doc_id}
        with self._conn.cursor() as cur:
            cur.execute(_DENSE_SQL, params)
            return [_row(r) for r in cur.fetchall()]


class PgSparseRetriever:
    name = "pg-tsvector"

    def __init__(self, conn):
        self._conn = conn

    def search(self, query: str, k: int = 5, doc_id: str | None = None) -> list[RetrievedChunk]:
        with self._conn.cursor() as cur:
            cur.execute(_SPARSE_SQL, {"tsq": to_or_tsquery(query), "k": k, "doc_id": doc_id})
            return [_row(r) for r in cur.fetchall()]
