from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="CLAIMSIGHT_", env_file=".env", extra="ignore")

    # replay: recorded model outputs + API cassettes (deterministic, free, used in CI)
    # live:   real models and real APIs
    mode: Literal["replay", "live"] = "replay"
    golden_dir: Path = REPO_ROOT / "evals" / "golden"

    # CLAIMSIGHT_DATABASE_URL, or plain DATABASE_URL (what `fly postgres attach` sets).
    database_url: str = Field(
        default="postgresql://claimsight:claimsight@localhost:5432/claimsight",
        validation_alias=AliasChoices("CLAIMSIGHT_DATABASE_URL", "DATABASE_URL"),
    )
    # Graph state between steps (claimsight/checkpoint.py). `postgres` keeps claims waiting
    # for an adjuster across restarts; `memory` is for tests and a single dev process.
    checkpointer: Literal["memory", "postgres"] = "memory"
    checkpoint_pool_size: int = 10
    redis_url: str = "redis://localhost:6379/0"
    # Claims run inline in the request (`off`) or on a Redis queue with a separate worker
    # (`redis`, claimsight/queue.py; needs checkpointer=postgres and the `queue` extra).
    queue: Literal["off", "redis"] = "off"
    # When set, every /claims route needs this in the X-API-Key header (/healthz stays open).
    # Set it on any deployment reachable from the internet.
    api_key: str | None = None
    # Public demo (api/demo.py, the Hugging Face Space): a page at `/` that files the golden
    # claims, and a cap on claims kept in memory, since anyone on the internet can file them.
    demo: bool = False
    demo_max_claims: int = 5_000
    queue_max_tries: int = 3
    queue_retry_delay_s: float = 5.0
    # Jobs one worker runs at once, and how often an idle worker checks Redis. arq's default
    # check is every 0.5 s, which capped each worker at a few claims a second (load test).
    queue_concurrency: int = 16
    queue_poll_s: float = 0.05
    queue_job_timeout_s: int = 300
    queue_keep_result_s: int = 86_400
    # Policy retrieval. First stage: keyword (BM25 / tsvector), dense, or hybrid (RRF of both).
    # Optional cross-encoder rerank of the first stage's top `rerank_pool`.
    # Backend: in-memory (replay, CI) or Postgres + pgvector.
    retriever: Literal["keyword", "dense", "hybrid"] = "dense"
    rerank: bool = True
    rerank_pool: int = 20
    vector_store: Literal["memory", "pgvector"] = "memory"
    # LLM query rewriting into policy vocabulary (retrieval/rewrite.py):
    # off | replace (search with the rewrite) | expand (fuse question + rewrite, rerank on the
    # question) | joint (fuse, rerank on question + rewrite) | dual (search with the rewrite,
    # rerank score = question's + rewrite's). Off by default: coverage already searches in
    # policy vocabulary (agents/coverage.py), so it would pay an LLM call per claim for nothing.
    query_rewrite: Literal["off", "replace", "expand", "joint", "dual"] = "off"
    rewrite_model: str = "claude-opus-5-5"
    rewrite_effort: Literal["low", "medium", "high"] = "low"
    # ColQwen2 page lane (retrieval/pages.py): rank rendered policy page images by late
    # interaction and fuse the clauses on the top pages into the first stage by RRF.
    page_lane: Literal["off", "colqwen2"] = "off"
    page_model: str = "vidore/colqwen2-v1.0-hf"
    # Replay records missing embeddings / rerank scores / rewrites instead of failing
    # (needs `embed`, and `llm` plus Anthropic credentials when query_rewrite is on).
    record_retrieval: bool = False

    # LLM coverage review (agents/coverage_review.py): reads the statement against the clauses
    # and every Part D exclusion; quotes are checked against the clause text. It can turn an
    # approval into an escalation, never into a denial. Replay reads cassettes/coverage.json
    # (a claim with no recorded review keeps the rules' finding); `record_coverage_review`
    # fills in missing reviews (needs `llm` and credentials).
    coverage_review: Literal["off", "llm"] = "llm"
    coverage_review_effort: Literal["low", "medium", "high"] = "medium"
    # Routing: `haiku_first` asks routing_model first and keeps its answer only when it is
    # `covered`; anything else goes to reasoning_model. Same exclusion recall and specificity
    # as Sonnet alone on the labelled set, 16% of reviews reach Sonnet, ~77% cheaper (README).
    # Haiku's answers replay from cassettes/coverage.haiku.json.
    coverage_review_routing: Literal["off", "haiku_first"] = "haiku_first"
    # Prompt layout: `cached_prefix` puts the screening clauses first so they can be cached
    # (half the cost per call) but changes Sonnet's answer on two labelled cases; see README.
    coverage_review_layout: Literal["statement_first", "cached_prefix"] = "statement_first"
    record_coverage_review: bool = False

    # VLM photo check (agents/photo_check.py): do the photos fit the statement in kind and
    # severity? `inconsistent` escalates, never denies. Runs only on photos with files; replay
    # reads cassettes/photo_check.json. `record_photo_check` calls the API on a miss.
    # Off by default: at 0.80 specificity on the held-out photos it would escalate about one
    # in five consistent claims, and there are no real claim photos yet to measure that on.
    photo_check: Literal["off", "llm"] = "off"
    photo_check_effort: Literal["low", "medium", "high"] = "medium"
    record_photo_check: bool = False

    # RT-DETRv2 fine-tuned by scripts/train_detector.py (local until pushed to the Hub).
    detector_model: str = str(REPO_ROOT / "models" / "rtdetr-hitl-damage")
    part_detector_model: str = "google/owlv2-base-patch16-ensemble"
    asr_model: str = "openai/whisper-large-v3-turbo"
    embed_model: str = "BAAI/bge-small-en-v1.5"
    embed_query_prefix: str = "Represent this sentence for searching relevant passages: "
    rerank_model: str = "BAAI/bge-reranker-v2-m3"
    reasoning_model: str = "claude-sonnet-5-5"
    routing_model: str = "claude-haiku-5-5"

    min_detection_confidence: float = 0.5
    escalation_confidence: float = 0.6
    auto_approve_limit_usd: float = 5000.0
    hail_radius_km: float = 25.0
    flood_min_precip_mm: float = 25.0
    max_photo_distance_km: float = 50.0
    duplicate_max_hamming: int = 6
    # DINOv2 reused-photo search (tools/photo_embed.py) for photos with image files, next to
    # the hash. Off until there's an index of earlier claims' embeddings to search (live mode
    # reads evals/golden/prior_photo_embeddings.jsonl if present; production: pgvector).
    # The cosine threshold is set on val edits at <= 1% false alarms (evals/duplicate_eval.py).
    photo_embeddings: Literal["off", "dinov2"] = "off"
    photo_embed_model: str = "facebook/dinov2-small"
    duplicate_min_cosine: float = 0.916
    http_timeout_s: float = 10.0


@lru_cache
def get_settings() -> Settings:
    return Settings()
