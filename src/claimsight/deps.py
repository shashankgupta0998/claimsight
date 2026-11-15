"""Wire concrete implementations to the interfaces: replay (offline) or live."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from claimsight.agents.impact import (
    CascadeImpactClassifier,
    ImpactClassifier,
    LLMImpactClassifier,
    RuleImpactClassifier,
)
from claimsight.config import Settings, get_settings
from claimsight.interfaces import Detector, PhotoIndex, Transcriber, VinDecoder, WeatherEvidence
from claimsight.models.detector import ReplayDetector
from claimsight.policies import JsonPolicyStore
from claimsight.retrieval import BM25Retriever, Retriever
from claimsight.tools.nhtsa import NHTSAVinDecoder, ReplayVinDecoder
from claimsight.tools.photos import InMemoryPhotoIndex
from claimsight.tools.weather import LiveWeather, ReplayWeather


@dataclass
class Deps:
    settings: Settings
    detector: Detector
    retriever: Retriever
    policies: JsonPolicyStore
    vin_decoder: VinDecoder
    weather: WeatherEvidence
    photo_index: PhotoIndex
    transcriber: Transcriber | None = None
    impact_classifier: ImpactClassifier = field(default_factory=RuleImpactClassifier)

    def model_versions(self) -> dict[str, str]:
        parts = {
            "detector": self.detector,
            "retriever": self.retriever,
            "vin_decoder": self.vin_decoder,
            "weather": self.weather,
            "photo_index": self.photo_index,
            "transcriber": self.transcriber,
            "impact_classifier": self.impact_classifier,
        }
        return {k: getattr(v, "name", type(v).__name__) for k, v in parts.items() if v is not None}


def _merge_json(paths) -> dict:
    """Merge cassette files: hand-curated (vin.json) and generated (vin.synthetic.json)."""
    merged: dict = {}
    for path in sorted(paths):
        merged.update(json.loads(path.read_text()))
    return merged


def _shared(golden: Path) -> dict:
    return {
        "retriever": BM25Retriever.from_jsonl(golden / "policy_clauses.jsonl"),
        "policies": JsonPolicyStore.from_file(golden / "policies.json"),
        "photo_index": InMemoryPhotoIndex.from_files(sorted(golden.glob("prior_photos*.jsonl"))),
    }


def _impact(s: Settings, offline: bool) -> ImpactClassifier:
    # Live runs record LLM answers into the cassette; replay reads them back offline.
    cassette = Path(s.golden_dir) / "cassettes" / "impact.llm.json"
    llm = LLMImpactClassifier(s.routing_model, cassette, offline=offline)
    return CascadeImpactClassifier(RuleImpactClassifier(), llm)


def build_replay_deps(settings: Settings | None = None) -> Deps:
    s = settings or get_settings()
    golden = Path(s.golden_dir)
    return Deps(
        settings=s,
        detector=ReplayDetector(),
        vin_decoder=ReplayVinDecoder(_merge_json(golden.glob("cassettes/vin*.json"))),
        weather=ReplayWeather(_merge_json(golden.glob("cassettes/weather*.json"))),
        impact_classifier=_impact(s, offline=True),
        **_shared(golden),
    )


def build_live_deps(settings: Settings | None = None) -> Deps:
    s = settings or get_settings()
    from claimsight.models.asr import WhisperTranscriber
    from claimsight.models.detector import HFDetector

    shared = _shared(Path(s.golden_dir))
    if s.retriever == "pgvector":
        from claimsight.retrieval.pgvector import PgVectorRetriever

        shared["retriever"] = PgVectorRetriever(s.database_url, s.embed_model)
    return Deps(
        settings=s,
        detector=HFDetector(s.detector_model, s.part_detector_model),
        vin_decoder=NHTSAVinDecoder(timeout=s.http_timeout_s),
        weather=LiveWeather(timeout=s.http_timeout_s),
        transcriber=WhisperTranscriber(s.asr_model),
        impact_classifier=_impact(s, offline=False),
        **shared,
    )


def build_deps(settings: Settings | None = None) -> Deps:
    s = settings or get_settings()
    return build_live_deps(s) if s.mode == "live" else build_replay_deps(s)
