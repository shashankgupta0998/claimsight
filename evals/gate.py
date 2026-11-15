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
    print(f"Eval gate passed ({len(GATED)} metrics within {args.tolerance} of baseline).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
