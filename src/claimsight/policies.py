from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel


class Policy(BaseModel):
    policy_id: str
    coverages: list[str]  # e.g. ["comprehensive", "collision"]
    deductibles: dict[str, float]
    vin: str | None = None


class JsonPolicyStore:
    """Declarations-page data per policy. Swap for the `policies` table in Postgres."""

    def __init__(self, policies: dict[str, Policy]):
        self._policies = policies

    @classmethod
    def from_file(cls, path: Path) -> JsonPolicyStore:
        raw = json.loads(path.read_text())
        return cls({pid: Policy(policy_id=pid, **body) for pid, body in raw.items()})

    def get(self, policy_id: str) -> Policy | None:
        return self._policies.get(policy_id)
