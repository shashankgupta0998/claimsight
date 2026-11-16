import json

import pytest
from PIL import Image

from claimsight.agents.critic import Critic
from claimsight.agents.photo_check import (
    PROMPT_VERSION,
    CassettePhotoChecker,
    MissingPhotoCheck,
    check_key,
    image_sha,
    run_check,
    to_signal,
)
from claimsight.schemas import (
    CostEstimate,
    CoverageFinding,
    Decision,
    Photo,
    SignalStatus,
)

SMASHED = {
    "verdict": "inconsistent",
    "visible_damage": ["front bumper torn off", "headlight smashed"],
    "explanation": "A shopping cart can't tear a bumper off.",
}


class FakeChecker:
    name = "fake-vlm"

    def __init__(self, answer):
        self.answer, self.calls = answer, []

    def check(self, statement, photos):
        self.calls.append((statement, photos))
        return self.answer


@pytest.fixture
def photo(tmp_path):
    path = tmp_path / "car.jpg"
    Image.new("RGB", (64, 48), (120, 30, 30)).save(path)
    return str(path)


def test_verdicts_map_to_signals():
    assert to_signal(SMASHED, "m").status == SignalStatus.FLAG
    assert "front bumper torn off" in to_signal(SMASHED, "m").detail
    assert to_signal({**SMASHED, "verdict": "consistent"}, "m").status == SignalStatus.PASS
    assert to_signal({**SMASHED, "verdict": "unclear"}, "m").status == SignalStatus.UNKNOWN
    assert to_signal(None, "m") is None  # refusal or cut-off: no signal


def test_nothing_to_check_without_photo_files_or_statement(photo):
    fake = FakeChecker(SMASHED)
    assert run_check(fake, "statement", [None, "/no/such/file.jpg"]) is None
    assert run_check(fake, "  ", [photo]) is None
    assert run_check(None, "statement", [photo]) is None
    assert fake.calls == []


def test_cassette_records_once_then_replays_by_photo_hash(tmp_path, photo):
    path = tmp_path / "photo_check.json"
    fake = FakeChecker(SMASHED)
    live = CassettePhotoChecker(path, fake)
    assert live.check("cart scratch", [photo]) == live.check("cart scratch", [photo]) == SMASHED
    assert len(fake.calls) == 1
    assert json.loads(path.read_text())["prompt"] == PROMPT_VERSION

    offline = CassettePhotoChecker(path)
    assert offline.check_hashed("cart scratch", [image_sha(photo)]) == SMASHED  # no file needed
    with pytest.raises(MissingPhotoCheck):
        offline.check("a different story", [photo])
    assert run_check(offline, "a different story", [photo]) is None  # unrecorded: no signal


def test_key_depends_on_statement_and_photos_not_order():
    assert check_key("s", ["a", "b"]) == check_key("s", ["b", "a"])
    assert check_key("s", ["a"]) != check_key("t", ["a"])


def test_cassette_from_an_old_prompt_is_ignored_offline_and_refused_online(tmp_path):
    path = tmp_path / "photo_check.json"
    path.write_text(json.dumps({"model": "fake-vlm", "prompt": "old", "answers": {"k": {}}}))
    with pytest.raises(MissingPhotoCheck):
        CassettePhotoChecker(path).check_hashed("s", ["a"])
    with pytest.raises(ValueError, match="re-record"):
        CassettePhotoChecker(path, FakeChecker(None))


def test_inconsistent_photos_escalate_never_deny(claim, photo):
    c = claim("CLM-0001").model_copy(update={"photos": [Photo(photo_id="p", uri=photo)]})
    critic = Critic(photo_checker=FakeChecker(SMASHED))
    covered = CoverageFinding(applies=True, rationale="Covered.")
    result = critic.run(c, "A cart scratched it.", [], covered, [], CostEstimate(low=1, high=2))
    assert result.decision == Decision.ESCALATE
    assert any(s.check == "photo_statement" for s in result.new_signals)
    assert any(r.startswith("photo_statement:") for r in result.reasons)
