import math

import pytest
from PIL import Image

from claimsight.agents.verification import VerificationAgent
from claimsight.retrieval.embed import encode_vector
from claimsight.schemas import Photo, SignalStatus
from claimsight.tools.photo_embed import EmbeddingPhotoIndex
from claimsight.tools.photos import InMemoryPhotoIndex
from evals.duplicate_eval import CASSETTE, TRANSFORMS, calibrate, edit, hit, score


def unit(*xs):
    n = math.sqrt(sum(x * x for x in xs))
    return [x / n for x in xs]


class FakeEmbedder:
    name = "fake-dino"

    def __init__(self, vec):
        self.vec = vec

    def embed(self, path):
        return self.vec


def test_index_returns_best_cosine_and_skips_the_same_claim():
    index = EmbeddingPhotoIndex(
        [
            {"claim_id": "A", "photo_id": "a", "embedding": unit(1, 0)},
            {"claim_id": "B", "photo_id": "b", "embedding": encode_vector(unit(1, 1))},
        ]
    )
    entry, sim = index.nearest(unit(1, 0), exclude_claim_id="X")
    assert entry["claim_id"] == "A" and sim == pytest.approx(1.0)
    entry, sim = index.nearest(unit(1, 0), exclude_claim_id="A")
    assert entry["claim_id"] == "B" and sim == pytest.approx(math.sqrt(0.5), abs=1e-3)


def test_embedding_match_flags_a_reused_photo_the_hash_missed(claim, tmp_path):
    path = tmp_path / "p.jpg"
    Image.new("RGB", (32, 32)).save(path)
    c = claim("CLM-0001").model_copy(update={"photos": [Photo(photo_id="p", uri=str(path))]})
    index = EmbeddingPhotoIndex([{"claim_id": "OLD", "photo_id": "o", "embedding": unit(1, 0)}])

    def agent(vec, threshold=0.9):
        return VerificationAgent(
            None,
            None,
            InMemoryPhotoIndex(),
            photo_embedder=FakeEmbedder(vec),
            embedding_index=index,
            duplicate_min_cosine=threshold,
        )

    flagged = agent(unit(1, 0.1)).check_duplicates(c)
    assert flagged.status == SignalStatus.FLAG and flagged.evidence["matches"][0]["cosine"] > 0.9
    assert agent(unit(1, 1)).check_duplicates(c).status == SignalStatus.PASS
    no_embedder = VerificationAgent(None, None, InMemoryPhotoIndex())
    assert no_embedder.check_duplicates(c) is None  # no hash and no embedder: nothing to check


def test_edits_are_deterministic_and_change_the_image():
    img = Image.effect_noise((120, 80), 40).convert("RGB")
    for t in TRANSFORMS:
        a, b = edit(img, t, "x" + t), edit(img, t, "x" + t)
        assert a.tobytes() == b.tobytes()
        assert a.size != img.size or a.tobytes() != img.tobytes()


def test_a_match_counts_only_on_the_right_photo():
    q = {"photo": "a.png", "seen": True, "dinov2": {"match": "b.png", "cosine": 0.99}}
    assert hit(q, "dinov2", 0.9) == (True, False)


def test_calibration_allows_at_most_one_percent_false_alarms():
    negs = [{"seen": False, "dinov2": {"cosine": c / 1000}} for c in range(300)]
    thr = calibrate(negs, "dinov2")
    assert sum(q["dinov2"]["cosine"] >= thr for q in negs) <= 3


@pytest.mark.skipif(not CASSETTE.exists(), reason="no duplicates cassette")
def test_recorded_comparison_replays():
    import json

    m = score(json.loads(CASSETTE.read_text()))
    assert m["test"]["dinov2"]["recall"] > m["test"]["dhash"]["recall"]
    assert m["test_without_twins"]["dinov2"]["fpr"] <= 0.01
