"""Turn clean golden claims into labelled fraud variants.

Each approved claim yields cases a real fraudster would try: backdated photos, a photo
reused from an earlier claim and, for collisions, a statement that contradicts the damage.
Location and date stay the same, so the existing API cassettes still apply.

    uv run python scripts/make_fraud_variants.py
"""

from __future__ import annotations

import argparse
import copy
import json
from datetime import datetime, timedelta
from pathlib import Path

GOLDEN = Path(__file__).resolve().parents[1] / "evals" / "golden"


def _variant(case: dict, suffix: str, flag: str) -> dict:
    v = copy.deepcopy(case)
    v["claim"]["claim_id"] = f"{case['claim']['claim_id']}-V-{suffix}"
    v["labels"]["decision"] = "escalate"
    v["labels"]["fraud_flags"] = [flag]
    return v


def variants(case: dict, prior_phash: str):
    if case["labels"]["decision"] != "approve":
        return

    v = _variant(case, "EXIF", "exif_consistency")
    for p in v["claim"]["photos"]:
        if p.get("exif_taken_at"):
            taken = datetime.fromisoformat(p["exif_taken_at"]) - timedelta(days=30)
            p["exif_taken_at"] = taken.isoformat()
    yield v

    v = _variant(case, "DUP", "duplicate_photos")
    v["claim"]["photos"][0]["phash"] = prior_phash
    yield v

    rear_only = all(
        d["part"] in {"bumper_rear", "taillamp", "trunk"}
        for p in case["claim"]["photos"]
        for d in p.get("recorded_detections") or []
    )
    if case["claim"]["loss_type"] == "collision" and rear_only:
        v = _variant(case, "FLIP", "statement_consistency")
        v["claim"]["statement"] = "I rear-ended the car ahead of me when traffic stopped suddenly."
        yield v


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=GOLDEN / "claims.jsonl")
    ap.add_argument("--out", type=Path, default=GOLDEN / "variants.jsonl")
    args = ap.parse_args()

    prior = json.loads((GOLDEN / "prior_photos.jsonl").read_text().splitlines()[0])["phash"]
    cases = [json.loads(x) for x in args.src.read_text().splitlines() if x.strip()]
    out = [v for c in cases for v in variants(c, prior)]
    args.out.write_text("".join(json.dumps(v) + "\n" for v in out))
    print(f"Wrote {len(out)} variants to {args.out}")


if __name__ == "__main__":
    main()
