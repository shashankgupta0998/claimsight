"""Critic: cross-check the agents' outputs against each other, then decide or escalate.

The statement-vs-photos check lives here because it needs both the transcript and the
vision output. It compares where the claimant says the car was hit (agents/impact.py)
with where the photos show damage, weighting each damaged part by severity x detector
confidence so one faint false-positive detection can't make a contradictory story look
consistent.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from claimsight.agents.impact import ImpactClassifier, RuleImpactClassifier
from claimsight.schemas import (
    ClaimInput,
    CostEstimate,
    CoverageFinding,
    DamageItem,
    Decision,
    LossType,
    SignalStatus,
    VerificationSignal,
)

PART_ZONES = {
    **dict.fromkeys(("bumper_front", "hood", "headlamp", "grille", "fender"), "front"),
    **dict.fromkeys(("bumper_rear", "trunk", "taillamp", "rear_windshield"), "rear"),
    **dict.fromkeys(("door", "mirror", "quarter_panel"), "side"),
}
MIN_SEVERITY = 0.1  # a detected part always carries some weight


def statement_zone(statement: str) -> str | None:
    """Rule-only reading of the impact side. Kept as a helper for tests and notebooks."""
    return RuleImpactClassifier().classify(statement)


def damage_zone_weights(damage: list[DamageItem]) -> dict[str, float]:
    weights: dict[str, float] = {}
    for d in damage:
        zone = PART_ZONES.get(d.part)
        if zone:
            weights[zone] = weights.get(zone, 0.0) + d.confidence * max(d.severity, MIN_SEVERITY)
    return weights


@dataclass
class CriticResult:
    decision: Decision
    confidence: float
    reasons: list[str]
    new_signals: list[VerificationSignal] = field(default_factory=list)


@dataclass
class Critic:
    escalation_confidence: float = 0.6
    auto_approve_limit_usd: float = 5000.0
    impact: ImpactClassifier = field(default_factory=RuleImpactClassifier)
    # The stated impact zone must carry at least this share of the zoned damage weight.
    min_zone_share: float = 0.25

    def consistency(self, claim: ClaimInput, statement: str, damage: list[DamageItem]):
        if claim.loss_type != LossType.COLLISION:
            return None
        said = self.impact.classify(statement)
        weights = damage_zone_weights(damage)
        total = sum(weights.values())
        if not said or not total:
            return VerificationSignal(
                check="statement_consistency",
                status=SignalStatus.UNKNOWN,
                detail="Statement or photos do not indicate an impact side.",
                evidence={"stated_zone": said, "zone_weights": weights},
            )
        share = weights.get(said, 0.0) / total
        evidence = {
            "stated_zone": said,
            "zone_weights": {k: round(v, 3) for k, v in weights.items()},
            "stated_zone_share": round(share, 3),
            "classifier": self.impact.name,
        }
        if share >= self.min_zone_share:
            return VerificationSignal(
                check="statement_consistency",
                status=SignalStatus.PASS,
                detail=f"Statement describes a {said} impact; {share:.0%} of the damage is {said}.",
                evidence=evidence,
            )
        main = max(weights, key=weights.get)
        return VerificationSignal(
            check="statement_consistency",
            status=SignalStatus.FLAG,
            detail=f"Statement describes a {said} impact, but {share:.0%} of the damage is "
            f"{said} and most of it is {main}.",
            evidence=evidence,
        )

    def run(
        self,
        claim: ClaimInput,
        statement: str,
        damage: list[DamageItem],
        coverage: CoverageFinding,
        signals: list[VerificationSignal],
        estimate: CostEstimate,
    ) -> CriticResult:
        new = [s for s in [self.consistency(claim, statement, damage)] if s is not None]
        all_signals = signals + new
        flags = [s for s in all_signals if s.status == SignalStatus.FLAG]
        unknowns = [s for s in all_signals if s.status == SignalStatus.UNKNOWN]

        confidence = sum(d.confidence for d in damage) / len(damage) if damage else 0.4
        confidence -= 0.25 * len(flags) + 0.05 * len(unknowns)
        if coverage.applies is None:
            confidence -= 0.2
        confidence = round(max(0.0, min(1.0, confidence)), 3)

        reasons = [f"{s.check}: {s.detail}" for s in flags]
        if flags:
            decision = Decision.ESCALATE
        elif coverage.applies is False:
            decision = Decision.DENY
            reasons.append(coverage.rationale)
        elif coverage.applies is None:
            decision = Decision.ESCALATE
            reasons.append(f"Coverage unclear: {coverage.rationale}")
        elif estimate.high > self.auto_approve_limit_usd:
            decision = Decision.ESCALATE
            reasons.append(
                f"Estimate up to ${estimate.high:,.0f} exceeds the "
                f"${self.auto_approve_limit_usd:,.0f} auto-approval limit."
            )
        elif confidence < self.escalation_confidence:
            decision = Decision.ESCALATE
            reasons.append(f"Confidence {confidence:.2f} is below {self.escalation_confidence}.")
        else:
            decision = Decision.APPROVE
            reasons.append(coverage.rationale)
        return CriticResult(
            decision=decision, confidence=confidence, reasons=reasons, new_signals=new
        )
