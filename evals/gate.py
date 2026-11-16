"""Regression gate: fail CI when a quality metric drops more than the tolerance vs baseline.

    uv run python -m evals.gate --report evals/reports/latest.json --baseline evals/baseline.json

Latency is reported but not gated: on shared CI runners it is too noisy to block merges.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

GATED = (
    "decision_accuracy",
    "retrieval_recall_at_5",
    "policy_qa_hit_at_1",
    "policy_qa_recall_at_5",
    "policy_qa_ndcg_at_5",
    "citation_precision",
    "damage_precision",
    "damage_recall",
    "fraud_precision",
    "fraud_recall",
    # Real detector on held-out photos, from cassettes/detector.json.
    "detector_map",
    "detector_map_50",
    # VLM photo check on the held-out photos (only when photo_check is on).
    "photo_check_flag_recall",
    "photo_check_specificity",
    # Only when coverage_review is on (skipped if the baseline doesn't have them).
    "coverage_review_exclusion_recall",
    "coverage_review_specificity",
    "coverage_review_quote_validity",
)


def compare(current: dict, baseline: dict, tolerance: float) -> list[str]:
    failures = []
    for name in GATED:
        if name not in baseline:
            continue
        before, after = baseline[name], current.get(name)
        if after is None:
            failures.append(f"{name}: missing from report")
        elif before - after > tolerance:
            failures.append(f"{name}: {before:.4f} → {after:.4f} (drop {before - after:.4f})")
    return failures


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", type=Path, required=True)
    ap.add_argument("--baseline", type=Path, required=True)
    ap.add_argument("--tolerance", type=float, default=0.02)
    args = ap.parse_args(argv)

    current = json.loads(args.report.read_text())["metrics"]
    baseline = json.loads(args.baseline.read_text())["metrics"]
    failures = compare(current, baseline, args.tolerance)
    if failures:
        print("Eval regression. Merge blocked:")
        for f in failures:
            print(f"  ✗ {f}")
        return 1
    checked = sum(name in baseline for name in GATED)
    print(f"Eval gate passed ({checked} metrics within {args.tolerance} of baseline).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
