"""Hybrid retrieval on Postgres: dense (pgvector HNSW) + sparse (tsvector), fused with RRF.

Requires the `rag` extra and the schema in infra/sql/001_schema.sql.

Roadmap (weeks 3-4):
  * add a cross-encoder rerank step (bge-reranker-v2-m3) over the fused top-20
  * add the visual lane: ColQwen2 MaxSim over `policy_page_patches`, fused as a third list
  * measure each lane's contribution with evals/runner.py (Recall@5, citation precision)
"""

from __future__ import annotations

import re

from claimsight.retrieval.base import RetrievedChunk

_RRF_K = 60

_HYBRID_SQL = """
WITH dense AS (
    SELECT clause_id, row_number() OVER (ORDER BY embedding <=> %(qvec)s) AS r
    FROM policy_chunks
    ORDER BY embedding <=> %(qvec)s
    LIMIT %(pool)s
),
sparse AS (
    SELECT clause_id, row_number() OVER (ORDER BY ts_rank_cd(tsv, q) DESC) AS r
    FROM policy_chunks, to_tsquery('english', %(tsq)s) AS q
    WHERE tsv @@ q
    ORDER BY ts_rank_cd(tsv, q) DESC
    LIMIT %(pool)s
)
SELECT c.clause_id, c.doc_id, c.page, c.text, c.metadata,
       COALESCE(1.0 / (%(rrf_k)s + d.r), 0) + COALESCE(1.0 / (%(rrf_k)s + s.r), 0) AS score
FROM policy_chunks c
LEFT JOIN dense d USING (clause_id)
LEFT JOIN sparse s USING (clause_id)
WHERE d.r IS NOT NULL OR s.r IS NOT NULL
ORDER BY score DESC
LIMIT %(k)s
"""


def to_or_tsquery(text: str) -> str:
    """Claimant statements are long; AND semantics would match nothing, so OR the terms."""
    terms = sorted(set(re.findall(r"[a-z0-9]{3,}", text.lower())))
    return " | ".join(terms) or "''"


class PgVectorRetriever:
    name = "pgvector-hybrid"

    def __init__(self, database_url: str, embed_model: str, pool: int = 20):
        import psycopg
        from pgvector.psycopg import register_vector
        from sentence_transformers import SentenceTransformer

        self._conn = psycopg.connect(database_url, autocommit=True)
        register_vector(self._conn)
        self._embedder = SentenceTransformer(embed_model)
        self._pool = pool

    def search(self, query: str, k: int = 5) -> list[RetrievedChunk]:
        qvec = self._embedder.encode(query, normalize_embeddings=True)
        params = {
            "qvec": qvec,
            "tsq": to_or_tsquery(query),
            "pool": self._pool,
            "rrf_k": _RRF_K,
            "k": k,
        }
        with self._conn.cursor() as cur:
            cur.execute(_HYBRID_SQL, params)
            rows = cur.fetchall()
        return [
            RetrievedChunk(
                clause_id=r[0], doc_id=r[1], page=r[2], text=r[3], metadata=r[4], score=float(r[5])
            )
            for r in rows
        ]
