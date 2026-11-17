import importlib.util
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from claimsight.api.main import create_app

ROOT = Path(__file__).resolve().parents[1]


def demo_client(deps, **update) -> TestClient:
    s = deps.settings.model_copy(update={"demo": True, **update})
    return TestClient(create_app(type(deps)(**{**deps.__dict__, "settings": s})))


def test_demo_page_is_off_by_default(deps):
    client = TestClient(create_app(deps))
    assert client.get("/").status_code == 404
    assert client.get("/demo/samples").status_code == 404


def test_demo_serves_the_page_and_every_golden_claim(deps):
    client = demo_client(deps)
    page = client.get("/")
    assert page.status_code == 200 and "text/html" in page.headers["content-type"]
    assert "/demo/samples" in page.text
    samples = client.get("/demo/samples").json()
    ids = [s["id"] for s in samples]
    assert len(ids) == 17 and "CLM-0001" in ids and "CLM-0003-V-FLIP" in ids
    flip = next(s for s in samples if s["id"] == "CLM-0003-V-FLIP")
    assert flip["title"] == "Same claim as CLM-0003, but the statement says the claimant caused it"
    assert flip["expected"] == "escalate"


def test_every_demo_sample_runs_to_a_decision_or_the_adjuster(deps):
    client = demo_client(deps)
    for s in client.get("/demo/samples").json():
        claim = {**s["claim"], "claim_id": s["id"] + "-demo"}
        body = client.post("/claims", json=claim).json()
        assert body["status"] in ("decided", "awaiting_review"), s["id"]
        if body["status"] == "awaiting_review":
            review = {"decision": "approve", "reviewer": "demo-adjuster"}
            done = client.post(f"/claims/{claim['claim_id']}/review", json=review).json()
            assert done["status"] == "decided"


def test_an_edited_statement_still_gets_a_decision(deps):
    """No recorded coverage review for new wording: the rules' finding stands, no crash."""
    client = demo_client(deps)
    claim = client.get("/demo/samples").json()[0]["claim"]
    claim = {**claim, "claim_id": "EDITED", "statement": "A branch fell on the roof in a storm."}
    assert client.post("/claims", json=claim).json()["status"] in ("decided", "awaiting_review")


def test_demo_caps_the_claims_it_keeps(deps):
    client = demo_client(deps, demo_max_claims=2)
    claim = client.get("/demo/samples").json()[0]["claim"]
    for i in range(2):
        assert client.post("/claims", json={**claim, "claim_id": f"C{i}"}).status_code == 200
    full = client.post("/claims", json={**claim, "claim_id": "C2"})
    assert full.status_code == 503 and "restarts" in full.json()["detail"]
    assert client.get("/claims/C0").status_code == 200  # reading still works


@pytest.fixture(scope="module")
def build_space():
    spec = importlib.util.spec_from_file_location("build_space", ROOT / "scripts/build_space.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_space_has_the_front_matter_the_demo_and_no_local_files(build_space, tmp_path):
    out = tmp_path / "space"
    files = build_space.build(out)
    readme = (out / "README.md").read_text()
    assert readme.startswith("---\n") and "sdk: docker" in readme and "app_port: 8000" in readme
    dockerfile = (out / "Dockerfile").read_text()
    assert dockerfile.index("ENV CLAIMSIGHT_DEMO=true") < dockerfile.index("\nCMD ")
    assert (out / "src/claimsight/api/static/demo.html").exists()
    assert (out / "evals/golden/claims.jsonl").exists() and (out / "uv.lock").exists()
    for f in files:
        assert not f.startswith(("data/", "models/", "tests/", ".env", "evals/reports/")), f
    assert not (out / ".env").exists()


def test_space_build_refuses_to_delete_a_folder_it_did_not_make(build_space, tmp_path):
    (tmp_path / "keep.txt").write_text("mine")
    with pytest.raises(SystemExit):
        build_space.build(tmp_path)
    assert (tmp_path / "keep.txt").exists()
