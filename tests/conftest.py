import json

import pytest

from claimsight.config import Settings
from claimsight.deps import build_replay_deps
from claimsight.schemas import ClaimInput


@pytest.fixture(scope="session")
def settings() -> Settings:
    return Settings(mode="replay")


@pytest.fixture(scope="session")
def deps(settings):
    return build_replay_deps(settings)


@pytest.fixture(scope="session")
def golden_cases(settings) -> dict[str, dict]:
    path = settings.golden_dir / "claims.jsonl"
    cases = [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
    return {c["claim"]["claim_id"]: c for c in cases}


@pytest.fixture
def claim(golden_cases):
    def _get(claim_id: str) -> ClaimInput:
        return ClaimInput.model_validate(golden_cases[claim_id]["claim"])

    return _get
