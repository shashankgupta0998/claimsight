import json
import shutil

import pytest

from claimsight.config import Settings
from claimsight.schemas import ClaimInput
from claimsight.tools.photos import hamming
from claimsight.tools.weather import cassette_key
from evals.metrics import word_error_rate
from evals.runner import evaluate
from evals.synth import generate


@pytest.fixture(scope="module")
def data():
    return generate(300, seed=3)


def test_same_seed_same_data():
    a, b = generate(40, seed=11), generate(40, seed=11)
    assert json.dumps(a.cases) == json.dumps(b.cases)
    assert a.weather == b.weather and a.vins == b.vins


def test_claims_match_the_api_schema(data):
    for case in data.cases:
        ClaimInput.model_validate(case["claim"])


def test_labels_follow_business_rules(data):
    for case in data.cases:
        claim, labels = case["claim"], case["labels"]
        fraud = [t for t in labels["tags"] if t.startswith("fraud:")]
        if fraud:
            assert labels["decision"] == "escalate" and len(labels["fraud_flags"]) == 1
        elif "deny" in labels["tags"]:
            assert labels["decision"] == "deny"
            assert claim["loss_type"] == "collision" and claim["policy_id"] == "P-200"
        elif {"major", "low_evidence"} & set(labels["tags"]):
            assert labels["decision"] == "escalate"
        else:
            assert labels["decision"] == "approve" and labels["fraud_flags"] == []


def test_every_external_lookup_has_a_cassette(data):
    for case in data.cases:
        claim = case["claim"]
        assert claim["vin"] in data.vins
        if claim["loss_type"] in ("hail", "flood"):
            loc = claim["location"]
            from datetime import date

            key = cassette_key(loc["lat"], loc["lon"], date.fromisoformat(claim["loss_date"]))
            assert key in data.weather


def test_only_duplicate_fraud_reuses_a_photo(data):
    priors = [p["phash"] for p in data.prior_photos]
    for case in data.cases:
        nearest = min(hamming(p["phash"], q) for p in case["claim"]["photos"] for q in priors)
        if "fraud:duplicate_photo" in case["labels"]["tags"]:
            assert nearest <= 4
        else:
            assert nearest > 6  # near-duplicate hard negatives sit at 9-14 bits


def test_scenario_mix_is_roughly_as_configured(data):
    stats = data.stats()
    assert 0.08 < stats["tags"].get("fraud", 0) / 300 < 0.25
    assert set(stats["decisions"]) == {"approve", "deny", "escalate"}


def test_pipeline_runs_end_to_end_on_generated_claims(tmp_path, settings):
    golden = tmp_path / "golden"
    (golden / "cassettes").mkdir(parents=True)
    for name in ("policy_clauses.jsonl", "policies.json"):
        shutil.copy(settings.golden_dir / name, golden / name)
    data = generate(60, seed=5)
    (golden / "cassettes" / "weather.json").write_text(json.dumps(data.weather))
    (golden / "cassettes" / "vin.json").write_text(json.dumps(data.vins))
    (golden / "prior_photos.jsonl").write_text(
        "".join(json.dumps(p) + "\n" for p in data.prior_photos)
    )

    report = evaluate(data.cases, Settings(mode="replay", golden_dir=golden))
    assert report["metrics"]["n_cases"] == 60
    assert report["metrics"]["decision_accuracy"] > 0.6
    assert "legit" in report["slices"]


def test_word_error_rate():
    assert word_error_rate("I hit a deer", "i hit a deer.") == 0
    assert word_error_rate("I hit a deer", "I hit deer") == pytest.approx(0.25)
    assert word_error_rate("", "") == 0
