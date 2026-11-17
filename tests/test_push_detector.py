import hashlib
import importlib.util
from pathlib import Path

import pytest

from evals.detector_eval import detection_map, load

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def push():
    spec = importlib.util.spec_from_file_location(
        "push_detector", ROOT / "scripts/push_detector.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def fake_model(tmp_path, push):
    """Stand-ins for the weights (not in git): the card only needs the files to exist."""
    d = tmp_path / "model"
    d.mkdir()
    for f in push.FILES:
        (d / f).write_bytes(f"fake {f}".encode())
    return d


def test_card_carries_the_measured_numbers_and_hub_metadata(push, fake_model, tmp_path):
    out = tmp_path / "hub"
    built = push.build(out, fake_model)
    assert sorted(p.name for p in out.iterdir()) == sorted([*push.FILES, "README.md"])
    readme = (out / "README.md").read_text()
    front = readme.split("---\n")[1]
    for line in (
        "license: apache-2.0",
        "pipeline_tag: object-detection",
        "base_model: PekingU/rtdetr_v2_r18vd",
        "- DrBimmer/car-parts-and-damage-dataset",
    ):
        assert line in front
    m = detection_map(*load()[:2])
    assert f"value: {m['map']}" in front and f"value: {m['map_50']}" in front
    assert f"| **all (mAP)** | **{m['map']:.3f}** | **{m['map_50']:.3f}** |" in readme
    assert "| crack (too few to mean much) |" in readme
    digest = hashlib.sha256(b"fake model.safetensors").hexdigest()
    assert built["sha256"] == digest and digest in readme


def test_rebuild_replaces_its_own_folder_but_not_someone_elses(push, fake_model, tmp_path):
    out = tmp_path / "hub"
    push.build(out, fake_model)
    push.build(out, fake_model)  # its own output: replaced
    (out / "notes.txt").write_text("mine")
    with pytest.raises(SystemExit):
        push.build(out, fake_model)
    assert (out / "notes.txt").exists()


def test_missing_weights_say_how_to_make_them(push, tmp_path):
    with pytest.raises(SystemExit, match="make train-detector"):
        push.build(tmp_path / "hub", tmp_path / "nowhere")


def test_card_example_loads_the_named_repo(push, fake_model, tmp_path):
    push.build(tmp_path / "a", fake_model)
    assert f'model="{push.PLACEHOLDER}"' in (tmp_path / "a" / "README.md").read_text()
    push.build(tmp_path / "b", fake_model, repo="someone/claimsight-damage-rtdetr")
    readme = (tmp_path / "b" / "README.md").read_text()
    assert 'model="someone/claimsight-damage-rtdetr"' in readme and push.PLACEHOLDER not in readme
