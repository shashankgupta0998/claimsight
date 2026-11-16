.PHONY: install test test-pg lint data eval gate baseline variants retrieval coverage record record-coverage routing injection photo-check record-photo-check duplicates record-duplicates pages detector train-detector ingest api up down

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

injection:  # prompt-injection red team against the coverage review (replay)
	uv run python -m evals.injection_eval

routing:  # coverage review cost vs quality: Sonnet vs Haiku first, with and without caching
	uv run python -m evals.routing_eval

record-coverage:  # record missing coverage reviews for every routing x layout (needs credentials)
	for r in off haiku_first; do for l in statement_first cached_prefix; do \
	  export CLAIMSIGHT_COVERAGE_REVIEW=llm CLAIMSIGHT_RECORD_COVERAGE_REVIEW=1 \
	    CLAIMSIGHT_COVERAGE_REVIEW_ROUTING=$$r CLAIMSIGHT_COVERAGE_REVIEW_LAYOUT=$$l; \
	  uv run --extra llm python -m evals.coverage_eval --record || exit 1; \
	  uv run --extra llm python -m evals.runner --golden evals/golden/claims.jsonl evals/golden/variants.jsonl evals/golden/synthetic.jsonl --out evals/reports/review_$${r}_$${l}.json || exit 1; \
	done; done
	uv run --extra llm python -m evals.injection_eval --record

photo-check:  # VLM statement/photo check on 188 labelled statements (replay)
	uv run python -m evals.photo_eval

record-photo-check:  # record missing photo checks (needs the CC0 photos in data/ and credentials)
	uv run --extra llm python -m evals.photo_eval --record

duplicates:  # reused-photo search: dHash vs DINOv2 on edited held-out photos (replay)
	uv run python -m evals.duplicate_eval

record-duplicates:  # embed the photos and edits with DINOv2 (needs the photos in data/)
	uv run --extra ml python -m evals.duplicate_eval --record

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
