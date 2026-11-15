from __future__ import annotations

from dataclasses import dataclass

from claimsight.interfaces import Detector
from claimsight.schemas import ClaimInput, DamageItem


@dataclass
class VisionAgent:
    detector: Detector
    min_confidence: float = 0.5

    def run(self, claim: ClaimInput) -> list[DamageItem]:
        """Detect damage in every photo and keep one item per (part, damage type).

        Several photos usually show the same dent; keeping the most confident view stops
        the estimator from paying for it twice.
        """
        best: dict[tuple[str, str], DamageItem] = {}
        for photo in claim.photos:
            for item in self.detector.detect(photo):
                if item.confidence < self.min_confidence:
                    continue
                key = (item.part, item.damage_type)
                if key not in best or item.confidence > best[key].confidence:
                    best[key] = item
        return sorted(best.values(), key=lambda d: (d.part, d.damage_type))
