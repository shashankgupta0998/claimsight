"""The public demo (`CLAIMSIGHT_DEMO=true`): a page at `/` for filing the golden claims.

The page is one static file (static/demo.html) that talks to the ordinary API: it lists the
samples from `GET /demo/samples`, files one with `POST /claims` under a fresh id, and posts
the adjuster's review when the claim waits for one. Replay mode, so nothing calls a model:
photo detections come with each sample, and an edited statement that has no recorded
coverage review keeps the rules' finding (the decision says so).
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import HTMLResponse

PAGE = Path(__file__).with_name("static") / "demo.html"

# What each fraud variant changes, for the sample picker (scripts/make_fraud_variants.py).
VARIANTS = {
    "EXIF": "a photo taken a month before the loss",
    "DUP": "a photo already seen on an earlier claim",
    "FLIP": "the statement says the claimant caused it",
}


def describe(claim_id: str, statement: str) -> str:
    for suffix, what in VARIANTS.items():
        if claim_id.endswith(f"-V-{suffix}"):
            return f"Same claim as {claim_id.split('-V-')[0]}, but {what}"
    return statement


@lru_cache
def load_samples(golden_dir: str) -> list[dict]:
    samples = []
    for name in ("claims.jsonl", "variants.jsonl"):
        for line in (Path(golden_dir) / name).read_text().splitlines():
            if not line.strip():
                continue
            case = json.loads(line)
            claim = case["claim"]
            samples.append(
                {
                    "id": claim["claim_id"],
                    "title": describe(claim["claim_id"], claim["statement"]),
                    "expected": case["labels"]["decision"],
                    "claim": claim,
                }
            )
    return samples


def router(golden_dir: Path) -> APIRouter:
    r = APIRouter()

    @r.get("/", response_class=HTMLResponse, include_in_schema=False)
    async def page() -> str:
        return PAGE.read_text()

    @r.get("/demo/samples")
    async def samples() -> list[dict]:
        return load_samples(str(golden_dir))

    return r
