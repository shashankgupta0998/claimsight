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
    # Replay records missing embeddings / rerank scores instead of failing (needs `embed`).
    record_retrieval: bool = False

    detector_model: str = "your-hf-username/rtdetr-cardd"
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
