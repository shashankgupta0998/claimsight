"""Where did the claimant's car take the hit? front, rear, side, or unknown (None).

Two classifiers behind one protocol, chained:

* RuleImpactClassifier: weighted regex patterns that read *who hit whom*, not keywords.
  "I rear-ended a Honda" is front damage; "a Honda rear-ended me" is rear damage.
  Free, deterministic, explainable. Abstains (None) when the evidence is weak or tied.
* LLMImpactClassifier: asks the routing model only when the rules abstain. Every answer
  is cached on disk, so a cached run is a replay cassette and evals stay deterministic.

Accuracy is measured on tests/data/impact_statements.jsonl, phrasings written separately
from the synthetic generator's templates so the rules can't be tuned to the test set.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Protocol

ZONES = ("front", "rear", "side")

_VEHICLE = r"(car|truck|vehicle|suv|van|jeep|sedan)"
# (pattern, zone, weight). Weights: 3-4 = who-hit-whom, 2 = a named part of the claimant's
# car, 1 = weak hint.
PATTERNS: list[tuple[str, str, float]] = [
    # Claimant was hit from behind
    (r"\b(rear-ended|rear-end)\s+(me|us|my)\b", "rear", 3),
    (r"\b(got|was|were|been|being|i'm|i am)\s+rear-ended\b", "rear", 3),
    (r"\bfrom behind\b", "rear", 3),
    (r"\b(back|rear|tail)[\s-]?(end|bumper|gate|lights?|lamps?)\b", "rear", 2),
    (rf"\b(back|rear) of (my|our|the) {_VEHICLE}\b", "rear", 2),
    (r"\bbehind (me|us)\b", "rear", 1),
    (r"\b(backed|backing|reversed|reversing)\b", "rear", 3),
    # Claimant hit something ahead
    (r"\b(i|we)\s+rear-ended\b", "front", 5),
    (
        r"\b(i|we)\s+(hit|struck|ran into|crashed into|slammed into|plowed into|bumped into|"
        r"collided with|tapped|smashed into|drove into|went into|slid into|nudged|rammed)\b",
        "front",
        3,
    ),
    (
        rf"\b(my|our) {_VEHICLE}\s+(hit|struck|ran into|crashed into|slid into|went into)\b",
        "front",
        3,
    ),
    (r"\b(in front of|ahead of) (me|us)\b", "front", 2),
    (rf"\bfront(\s|-)?(end|bumper|of (my|our|the) {_VEHICLE})\b", "front", 2),
    (r"\bhead[\s-]?on\b", "front", 3),
    (r"\b(hood|headlights?|grille|radiator)\b", "front", 1),
    (r"\b(i|we) (couldn'?t|could not|didn'?t|did not) (stop|brake)\b", "front", 2),
    # Side impact
    (r"\bside[\s-]?swip", "side", 3),
    (r"\bt[\s-]?boned\b", "side", 3),
    (r"\b(driver'?s?|passenger)[\s-]?side\b", "side", 2),
    (rf"\bside of (my|our|the) {_VEHICLE}\b", "side", 2),
    (r"\b(doors?|mirror)\b", "side", 1),
]
_COMPILED = [(re.compile(p), zone, w) for p, zone, w in PATTERNS]


def _normalise(text: str) -> str:
    s = text.lower().replace("’", "'")
    return re.sub(r"\brear[\s-]?end", "rear-end", s)


class ImpactClassifier(Protocol):
    name: str

    def classify(self, statement: str) -> str | None: ...


class RuleImpactClassifier:
    name = "impact-rules-v1"

    def __init__(self, min_score: float = 1.0, min_margin: float = 1.0):
        self.min_score, self.min_margin = min_score, min_margin

    def scores(self, statement: str) -> dict[str, float]:
        text = _normalise(statement)
        totals = dict.fromkeys(ZONES, 0.0)
        for pattern, zone, weight in _COMPILED:
            if pattern.search(text):
                totals[zone] += weight
        return totals

    def classify(self, statement: str) -> str | None:
        ranked = sorted(self.scores(statement).items(), key=lambda kv: kv[1], reverse=True)
        (top, best), (_, second) = ranked[0], ranked[1]
        if best < self.min_score or best - second < self.min_margin:
            return None  # abstain rather than guess: a wrong guess raises a false fraud flag
        return top


LLM_SYSTEM = """You read a motor-insurance claimant's statement and decide where the
claimant's own vehicle was struck. Answer with exactly one word:
front  - the claimant's car hit something with its front, or was hit head-on
rear   - the claimant's car was hit from behind, or reversed into something
side   - the claimant's car was hit on a door, flank or mirror
unknown - the statement does not say"""


class LLMImpactClassifier:
    """Fallback for statements the rules can't read. Needs the `llm` extra."""

    def __init__(self, model: str, cache_path: Path, offline: bool = False):
        self.name = f"impact-llm:{model}"
        self.model, self.cache_path, self.offline = model, cache_path, offline
        self._cache: dict[str, str] = (
            json.loads(cache_path.read_text()) if cache_path.exists() else {}
        )
        self._client = None

    @staticmethod
    def key(statement: str) -> str:
        return hashlib.sha256(statement.strip().encode()).hexdigest()[:16]

    def classify(self, statement: str) -> str | None:
        k = self.key(statement)
        if k not in self._cache:
            if self.offline:
                return None
            self._cache[k] = self._ask(statement)
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            self.cache_path.write_text(json.dumps(self._cache, indent=1, sort_keys=True))
        label = self._cache[k]
        return label if label in ZONES else None

    def _ask(self, statement: str) -> str:
        if self._client is None:
            import anthropic

            self._client = anthropic.Anthropic()
        msg = self._client.messages.create(
            model=self.model,
            max_tokens=5,
            system=LLM_SYSTEM,
            messages=[{"role": "user", "content": statement}],
        )
        text = "".join(b.text for b in msg.content if b.type == "text").strip().lower()
        word = text.split()[0].strip(".") if text else "unknown"
        return word if word in ZONES else "unknown"


class CascadeImpactClassifier:
    """First classifier that commits to an answer wins."""

    def __init__(self, *classifiers: ImpactClassifier):
        self.classifiers = classifiers
        self.name = "+".join(c.name for c in classifiers)

    def classify(self, statement: str) -> str | None:
        for c in self.classifiers:
            zone = c.classify(statement)
            if zone is not None:
                return zone
        return None
