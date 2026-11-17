.PHONY: install test test-pg worker loadtest loadtest-matrix lint data eval gate baseline variants retrieval coverage record record-coverage routing injection trace-eval dashboards dashboards-build photo-check record-photo-check duplicates record-duplicates pages detector train-detector detector-hub ingest api demo space space-docker up down

install:
	uv sync --group dev

test:
	uv run pytest -q

test-pg:  # needs Postgres with pgvector and Redis (make up); override the CLAIMSIGHT_TEST_* URLs
	CLAIMSIGHT_TEST_DATABASE_URL=$${CLAIMSIGHT_TEST_DATABASE_URL:-postgresql://claimsight:claimsight@localhost:5432/claimsight} CLAIMSIGHT_TEST_REDIS_URL=$${CLAIMSIGHT_TEST_REDIS_URL:-redis://localhost:6379/0} uv run --extra pg --extra queue pytest -q -m pg

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

trace-eval:  # run the 257-case eval in replay with Langfuse tracing on (keys in .env)
	uv run --extra llm --env-file .env python -m evals.runner --golden evals/golden/claims.jsonl evals/golden/variants.jsonl evals/golden/synthetic.jsonl --out evals/reports/traced.json

dashboards:  # run every dashboard widget's query against your Langfuse project (keys in .env)
	uv run --extra llm --env-file .env python scripts/langfuse_dashboards.py --check

dashboards-build:  # create/update the dashboards in Langfuse (dry run; APPLY=1 to make them)
	uv run --extra llm --env-file .env python scripts/langfuse_dashboards.py --build $(if $(APPLY),--apply)

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

detector-hub:  # weights + model card in build/detector, checked against the cassette (README)
	uv run --extra ml python scripts/push_detector.py

pages:  # draw the specimen policy as page images for the ColQwen2 page lane
	uv run python scripts/render_pages.py

record:  # re-record embedding, rerank, query-rewrite and page-score cassettes with the real models
	uv run --extra embed --extra llm python scripts/record_retrieval.py

ingest:  # load the policy corpus into Postgres + pgvector
	uv run --extra pg python scripts/ingest_policies.py --init-schema

api:
	uv run uvicorn claimsight.api.main:app --reload

demo:  # the Space's page at http://localhost:8000 (replay mode, in memory)
	CLAIMSIGHT_DEMO=true uv run uvicorn claimsight.api.main:app --reload

space:  # the free static Hugging Face Space in build/space (README, "Hugging Face Space")
	uv run python scripts/build_space.py

space-docker:  # the Docker Space in build/space (Hugging Face hosts it only with PRO)
	uv run python scripts/build_space.py --kind docker

HOST ?= http://localhost:8000
USERS ?= 20
DURATION ?= 60s
loadtest:  # drive a running API with Locust (make api first; replay mode makes no model calls)
	uv run --extra load locust -f loadtest/locustfile.py --headless -u $(USERS) -r $(USERS) -t $(DURATION) --host $(HOST) --only-summary

loadtest-matrix:  # every scenario: inline/memory, inline/postgres, queued (set DB_URL, REDIS_URL)
	uv run --extra load --extra pg --extra queue python loadtest/run.py --db $(DB_URL) --redis $(REDIS_URL)

worker:  # run queued claims (CLAIMSIGHT_QUEUE=redis, CLAIMSIGHT_CHECKPOINTER=postgres)
	uv run --extra pg --extra queue python -m claimsight.queue

up:
	docker compose up -d

down:
	docker compose down
