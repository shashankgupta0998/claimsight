"""Load the policy corpus into Postgres + pgvector.

    uv run --extra pg python scripts/ingest_policies.py --init-schema       # replay vectors
    uv run --extra rag python scripts/ingest_policies.py --live             # embed with the model

Reads evals/golden/policy_clauses.jsonl and upserts policy_docs and policy_chunks, so it is
safe to re-run. Without --live, vectors come from the embedding cassette, which makes the
database contents identical to what the offline evals use.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from claimsight.config import get_settings  # noqa: E402
from claimsight.retrieval.factory import build_embedder, load_chunks  # noqa: E402
from claimsight.retrieval.pgvector import connect, ingest  # noqa: E402

SCHEMA = ROOT / "infra" / "sql" / "001_schema.sql"
TITLES = {"PP-00-01-specimen": "Personal auto policy (paraphrased specimen)"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--live", action="store_true", help="Embed with the real model")
    ap.add_argument("--init-schema", action="store_true", help="Create tables if missing")
    args = ap.parse_args()

    settings = get_settings()
    conn = connect(settings.database_url)
    if args.init_schema:
        (exists,) = conn.execute("SELECT to_regclass('policy_chunks') IS NOT NULL").fetchone()
        if not exists:
            conn.execute(SCHEMA.read_text())
            print(f"Created schema from {SCHEMA.relative_to(ROOT)}")

    chunks = load_chunks(Path(settings.golden_dir))
    embedder = build_embedder(settings, online=args.live)
    n = ingest(conn, chunks, embedder, TITLES)
    if args.live:
        embedder.save()
    print(f"Ingested {n} clauses with {embedder.name} ({embedder.dim}-d)")


if __name__ == "__main__":
    main()
