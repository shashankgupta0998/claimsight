.PHONY: install test test-pg lint data eval gate baseline variants retrieval record ingest api up down

install:
	uv sync --group dev

test:
	uv run pytest -q

test-pg:  # needs Postgres with pgvector (make up); override with CLAIMSIGHT_TEST_DATABASE_URL
	CLAIMSIGHT_TEST_DATABASE_URL=$${CLAIMSIGHT_TEST_DATABASE_URL:-postgresql://claimsight:claimsight@localhost:5432/claimsight} uv run --extra pg pytest -q -m pg

lint:
	uv run ruff check .

eval:
	uv run python -m evals.runner --golden evals/golden/claims.jsonl evals/golden/variants.jsonl evals/golden/synthetic.jsonl --out evals/reports/latest.json

gate: eval
	uv run python -m evals.gate --report evals/reports/latest.json --baseline evals/baseline.json

baseline: eval
	cp evals/reports/latest.json evals/baseline.json

variants:
	uv run python scripts/make_fraud_variants.py

data: variants
	uv run python scripts/generate_claims.py

retrieval:  # policy-QA ablation: keyword vs dense vs hybrid, with and without rerank
	uv run python -m evals.retrieval_eval

record:  # re-record embedding, rerank and query-rewrite cassettes with the real models
	uv run --extra embed --extra llm python scripts/record_retrieval.py

ingest:  # load the policy corpus into Postgres + pgvector
	uv run --extra pg python scripts/ingest_policies.py --init-schema

api:
	uv run uvicorn claimsight.api.main:app --reload

up:
	docker compose up -d

down:
	docker compose down
