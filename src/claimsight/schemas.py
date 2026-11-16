"""Typed contracts shared by the agents, the API and the eval harness."""

from __future__ import annotations

from datetime import date, datetime
from enum import StrEnum

from pydantic import BaseModel, Field, computed_field


class LossType(StrEnum):
    HAIL = "hail"
    FLOOD = "flood"
    COLLISION = "collision"
    ANIMAL = "animal"  # contact with a bird or animal: comprehensive, not collision, in PAP forms
    GLASS = "glass"
    THEFT = "theft"
    VANDALISM = "vandalism"


class Location(BaseModel):
    lat: float = Field(ge=-90, le=90)
    lon: float = Field(ge=-180, le=180)
    label: str | None = None


class DamageItem(BaseModel):
    part: str
    damage_type: str
    severity: float = Field(ge=0, le=1)
    confidence: float = Field(ge=0, le=1)
    photo_id: str | None = None
    box: tuple[float, float, float, float] | None = None


class Photo(BaseModel):
    photo_id: str
    uri: str | None = None
    exif_taken_at: datetime | None = None
    exif_lat: float | None = None
    exif_lon: float | None = None
    phash: str | None = Field(default=None, description="64-bit perceptual hash as 16 hex chars")
    recorded_detections: list[DamageItem] | None = Field(
        default=None,
        description="Replay cassette: detector output recorded once, reused in offline evals.",
    )


class ClaimInput(BaseModel):
    claim_id: str
    policy_id: str
    loss_type: LossType
    loss_date: date
    location: Location
    vin: str | None = None
    reported_vehicle: str | None = Field(default=None, examples=["2021 Toyota Camry"])
    statement: str = ""
    voice_note_uri: str | None = None
    photos: list[Photo] = Field(default_factory=list)


class Citation(BaseModel):
    doc_id: str
    clause_id: str
    page: int | None = None
    snippet: str


class QuoteCheck(BaseModel):
    clause_id: str
    quote: str
    valid: bool
    problem: str | None = None


class CoverageReview(BaseModel):
    """What the LLM coverage reviewer said, and whether its quotes held up."""

    reviewer: str
    verdict: str  # covered | excluded | unclear
    explanation: str
    quotes: list[QuoteCheck] = Field(default_factory=list)
    applied: bool = False  # did the review change the finding?


class CoverageFinding(BaseModel):
    applies: bool | None
    coverage_part: str | None = None
    deductible: float | None = None
    citations: list[Citation] = Field(default_factory=list)
    rationale: str
    review: CoverageReview | None = None


class SignalStatus(StrEnum):
    PASS = "pass"
    FLAG = "flag"
    UNKNOWN = "unknown"


class VerificationSignal(BaseModel):
    check: str
    status: SignalStatus
    detail: str
    evidence: dict = Field(default_factory=dict)


class CostEstimate(BaseModel):
    low: float
    high: float
    currency: str = "USD"
    line_items: list[dict] = Field(default_factory=list)


class Decision(StrEnum):
    APPROVE = "approve"
    DENY = "deny"
    ESCALATE = "escalate"


class DecisionPacket(BaseModel):
    claim_id: str
    decision: Decision
    decided_by: str = "system"
    confidence: float
    damage: list[DamageItem]
    coverage: CoverageFinding
    signals: list[VerificationSignal]
    estimate: CostEstimate
    reasons: list[str]
    model_versions: dict[str, str] = Field(default_factory=dict)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def fraud_flags(self) -> list[str]:
        return [s.check for s in self.signals if s.status == SignalStatus.FLAG]
