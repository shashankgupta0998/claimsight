from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="CLAIMSIGHT_", env_file=".env", extra="ignore")

    # replay: recorded model outputs + API cassettes (deterministic, free, used in CI)
    # live:   real models and real APIs
    mode: Literal["replay", "live"] = "replay"
    golden_dir: Path = REPO_ROOT / "evals" / "golden"

    database_url: str = "postgresql://claimsight:claimsight@localhost:5432/claimsight"
    redis_url: str = "redis://localhost:6379/0"
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
    http_timeout_s: float = 10.0


@lru_cache
def get_settings() -> Settings:
    return Settings()
