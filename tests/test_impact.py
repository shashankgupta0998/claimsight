import json
from datetime import date
from pathlib import Path

import pytest

from claimsight.agents.critic import Critic
from claimsight.agents.impact import (
    CascadeImpactClassifier,
    LLMImpactClassifier,
    RuleImpactClassifier,
)
from claimsight.schemas import ClaimInput, DamageItem, Location, SignalStatus

HELD_OUT = Path(__file__).parent / "data" / "impact_statements.jsonl"


def _held_out():
    return [json.loads(x) for x in HELD_OUT.read_text().splitlines() if x.strip()]


def test_rules_on_held_out_phrasings():
    """Phrasings written apart from the generator's templates. Don't tune rules to this file."""
    rules = RuleImpactClassifier()
    rows = _held_out()
    predictions = [rules.classify(r["text"]) for r in rows]
    accuracy = sum(p == r["zone"] for p, r in zip(predictions, rows, strict=True)) / len(rows)
    wrong_guesses = [
        r["text"]
        for p, r in zip(predictions, rows, strict=True)
        if p is not None and p != r["zone"]
    ]
    assert accuracy >= 0.85
    # Abstaining costs a little confidence; a wrong guess raises a false fraud flag.
    assert wrong_guesses == []


@pytest.mark.parametrize(
    ("text", "zone"),
    [
        ("Someone rear-ended me at a light", "rear"),
        ("I rear-ended the car ahead", "front"),
        ("I wasn't the one who hit anyone; a van rear-ended me.", "rear"),
        ("I backed into a pole", "rear"),
        ("Hail fell on my car", None),
    ],
)
def test_who_hit_whom(text, zone):
    assert RuleImpactClassifier().classify(text) == zone


class FakeClient:
    def __init__(self, answer):
        self.calls = 0
        self.messages = self
        self.answer = answer

    def create(self, **_):
        self.calls += 1
        block = type("Block", (), {"type": "text", "text": self.answer})()
        return type("Msg", (), {"content": [block]})()


def test_llm_fallback_is_cached_and_replayable(tmp_path):
    cache = tmp_path / "impact.llm.json"
    llm = LLMImpactClassifier("claude-haiku-5-5", cache)
    llm._client = FakeClient("Front.")
    cascade = CascadeImpactClassifier(RuleImpactClassifier(), llm)

    statement = "I slid on ice and went into a ditch nose first."
    assert cascade.classify(statement) == "front"
    assert cascade.classify(statement) == "front"
    assert llm._client.calls == 1  # second answer came from the cache

    offline = LLMImpactClassifier("claude-haiku-5-5", cache, offline=True)
    assert offline.classify(statement) == "front"  # replay reads the recorded answer
    assert offline.classify("never seen before") is None  # and never calls the API


def _claim(statement: str) -> ClaimInput:
    return ClaimInput(
        claim_id="T-1",
        policy_id="P-100",
        loss_type="collision",
        loss_date=date(2026, 7, 1),
        location=Location(lat=30.0, lon=-97.0),
        statement=statement,
    )


def test_faint_false_positive_does_not_hide_a_contradiction():
    damage = [
        DamageItem(part="bumper_rear", damage_type="dent", severity=0.85, confidence=0.9),
        DamageItem(part="taillamp", damage_type="lamp_broken", severity=0.7, confidence=0.84),
        DamageItem(part="fender", damage_type="scratch", severity=0.2, confidence=0.56),
    ]
    signal = Critic().consistency(
        _claim("I rear-ended a pickup truck."), "I rear-ended a pickup truck.", damage
    )
    assert signal.status == SignalStatus.FLAG
    assert signal.evidence["stated_zone"] == "front"
    assert signal.evidence["stated_zone_share"] < 0.25


def test_matching_story_passes_despite_false_positive():
    damage = [
        DamageItem(part="bumper_rear", damage_type="dent", severity=0.6, confidence=0.9),
        DamageItem(part="bumper_front", damage_type="scratch", severity=0.2, confidence=0.55),
    ]
    text = "A sedan hit me from behind."
    assert Critic().consistency(_claim(text), text, damage).status == SignalStatus.PASS
