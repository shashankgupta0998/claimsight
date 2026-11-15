from __future__ import annotations

import operator
from typing import Annotated, TypedDict

from claimsight.schemas import (
    ClaimInput,
    CostEstimate,
    CoverageFinding,
    DamageItem,
    Decision,
    DecisionPacket,
    VerificationSignal,
)


class ClaimState(TypedDict, total=False):
    claim: ClaimInput
    statement: str
    transcribed: bool
    damage: list[DamageItem]
    coverage: CoverageFinding
    retrieved_clause_ids: list[str]
    # Verification and the critic both add signals; the reducer concatenates them.
    signals: Annotated[list[VerificationSignal], operator.add]
    estimate: CostEstimate
    confidence: float
    decision: Decision
    reasons: list[str]
    decided_by: str
    packet: DecisionPacket
