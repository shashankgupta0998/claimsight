.PHONY: install test lint data eval gate baseline variants api up down

install:
	uv sync --group dev

test:
	uv run pytest -q

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

api:
	uv run uvicorn claimsight.api.main:app --reload

up:
	docker compose up -d

down:
	docker compose down
