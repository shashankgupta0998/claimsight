"""Integration tests against a real Postgres + pgvector.

    CLAIMSIGHT_TEST_DATABASE_URL=postgresql://user@localhost:5432/db uv run pytest -m pg

Skipped when the variable is unset. Each session builds the schema in its own Postgres
schema and drops it afterwards, so it never touches existing tables.
"""

from __future__ import annotations

import os
import uuid
from pathlib import Path
from urllib.parse import quote

import pytest

from claimsight.config import Settings
from claimsight.retrieval.factory import build_embedder, build_retrieval, load_chunks
from evals.retrieval_eval import load_queries, score_retriever

DB_URL = os.environ.get("CLAIMSIGHT_TEST_DATABASE_URL")
pytestmark = [
    pytest.mark.pg,
    pytest.mark.skipif(not DB_URL, reason="CLAIMSIGHT_TEST_DATABASE_URL not set"),
]
SCHEMA_SQL = Path(__file__).resolve().parents[1] / "infra" / "sql" / "001_schema.sql"


@pytest.fixture(scope="module")
def db():
    from claimsight.retrieval.pgvector import connect, ingest

    schema = f"test_{uuid.uuid4().hex[:8]}"
    admin = connect(DB_URL)
    admin.execute("CREATE EXTENSION IF NOT EXISTS vector")
    admin.execute(f"CREATE SCHEMA {schema}")
    sep = "&" if "?" in DB_URL else "?"
    url = f"{DB_URL}{sep}options={quote(f'-csearch_path={schema},public')}"
    conn = connect(url)
    conn.execute(SCHEMA_SQL.read_text())

    settings = Settings(database_url=url, vector_store="pgvector")
    chunks = load_chunks(Path(settings.golden_dir))
    ingest(conn, chunks, build_embedder(settings, online=False))
    yield {"conn": conn, "url": url, "settings": settings, "chunks": chunks}
    conn.close()
    admin.execute(f"DROP SCHEMA {schema} CASCADE")
    admin.close()


def test_ingest_is_idempotent(db):
    from claimsight.retrieval.pgvector import ingest

    ingest(db["conn"], db["chunks"], build_embedder(db["settings"], online=False))
    (count,) = db["conn"].execute("SELECT count(*) FROM policy_chunks").fetchone()
    (docs,) = db["conn"].execute("SELECT count(*) FROM policy_docs").fetchone()
    assert count == len(db["chunks"]) and docs == 1


def test_dense_lane_matches_in_memory(db):
    """HNSW is approximate; with ef_search above the corpus size it must equal exact search."""
    queries = [q["query"] for q in load_queries()]
    mem = build_retrieval(Settings(retriever="dense", rerank=False)).retriever
    pg = build_retrieval(db["settings"].model_copy(update={"rerank": False})).retriever
    for q in queries:
        assert [c.clause_id for c in pg.search(q, 5)] == [c.clause_id for c in mem.search(q, 5)]


def test_sparse_lane_finds_keywords(db):
    from claimsight.retrieval.pgvector import PgSparseRetriever

    hits = PgSparseRetriever(db["conn"]).search("windshield glass breakage", k=3)
    assert {"PP-D-GLASS", "END-GLASS"} & {h.clause_id for h in hits}


def test_doc_filter(db):
    from claimsight.retrieval.pgvector import PgSparseRetriever

    assert PgSparseRetriever(db["conn"]).search("collision", k=3, doc_id="other-form") == []


def test_full_stack_scores_like_memory(db):
    """Default stack (dense > rerank) on Postgres reproduces the in-memory policy-QA score."""
    queries = load_queries()
    mem = score_retriever(build_retrieval(Settings()).retriever, queries)["metrics"]
    pg = score_retriever(build_retrieval(db["settings"]).retriever, queries)["metrics"]
    assert pg == mem
