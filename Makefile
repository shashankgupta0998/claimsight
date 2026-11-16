.PHONY: install test test-pg lint data eval gate baseline variants retrieval coverage record record-coverage photo-check record-photo-check pages detector train-detector ingest api up down

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

coverage:  # LLM coverage review on the labelled exclusion statements (replay)
	uv run python -m evals.coverage_eval

record-coverage:  # record missing coverage reviews (needs Anthropic credentials)
	uv run --extra llm python -m evals.coverage_eval --record
	CLAIMSIGHT_COVERAGE_REVIEW=llm CLAIMSIGHT_RECORD_COVERAGE_REVIEW=1 uv run --extra llm python -m evals.runner --golden evals/golden/claims.jsonl evals/golden/variants.jsonl evals/golden/synthetic.jsonl --out evals/reports/review_on.json

photo-check:  # VLM statement/photo check on 188 labelled statements (replay)
	uv run python -m evals.photo_eval

record-photo-check:  # record missing photo checks (needs the CC0 photos in data/ and credentials)
	uv run --extra llm python -m evals.photo_eval --record

detector:  # damage detector mAP on the held-out test photos (replay)
	uv run python -m evals.detector_eval

train-detector:  # fine-tune RT-DETRv2 on the CC0 damage photos (see models/damage_data.py)
	uv run --extra train python scripts/train_detector.py

pages:  # draw the specimen policy as page images for the ColQwen2 page lane
	uv run python scripts/render_pages.py

record:  # re-record embedding, rerank, query-rewrite and page-score cassettes with the real models
	uv run --extra embed --extra llm python scripts/record_retrieval.py

ingest:  # load the policy corpus into Postgres + pgvector
	uv run --extra pg python scripts/ingest_policies.py --init-schema

api:
	uv run uvicorn claimsight.api.main:app --reload

up:
	docker compose up -d

down:
	docker compose down
