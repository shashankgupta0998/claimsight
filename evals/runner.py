"""Run the claims graph over labelled cases and score every stage.

    uv run python -m evals.runner --golden evals/golden/claims.jsonl --out evals/reports/latest.json

Replay mode (default) uses recorded model outputs and API cassettes, so it is free and
deterministic. Live mode runs real models and APIs over the same labels.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

from claimsight.config import Settings
from claimsight.deps import build_deps
from claimsight.graph import build_graph
from claimsight.schemas import ClaimInput
from evals.metrics import (
    ndcg_at_k,
    percentile,
    precision_recall,
    recall_at_k,
    safe_div,
    word_error_rate,
)

K = 5


def load_cases(paths: list[Path]) -> list[dict]:
    cases = []
    for path in paths:
        if not path.exists():
            print(f"skip: {path} not found", file=sys.stderr)
            continue
        cases += [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return cases


def evaluate(cases: list[dict], settings: Settings) -> dict:
    graph = build_graph(build_deps(settings))
    rows = []
    for case in cases:
        claim = ClaimInput.model_validate(case["claim"])
        labels = case["labels"]
        start = time.perf_counter()
        state = graph.invoke({"claim": claim})
        latency_ms = (time.perf_counter() - start) * 1000
        packet = state["packet"]
        retrieved = state.get("retrieved_clause_ids", [])
        cited = [c.clause_id for c in packet.coverage.citations]
        rows.append(
            {
                "claim_id": claim.claim_id,
                "expected": labels["decision"],
                "predicted": packet.decision.value,
                "correct": packet.decision.value == labels["decision"],
                "expected_flags": sorted(labels["fraud_flags"]),
                "predicted_flags": sorted(packet.fraud_flags),
                "recall_at_k": recall_at_k(retrieved, labels["relevant_clause_ids"], K),
                "ndcg_at_k": ndcg_at_k(retrieved, labels["relevant_clause_ids"], K),
                "cited": cited,
                "cited_relevant": sum(c in labels["relevant_clause_ids"] for c in cited),
                "damage_pr": precision_recall(
                    {(d.part, d.damage_type) for d in packet.damage},
                    {tuple(x) for x in labels["damage"]},
                ),
                "confidence": packet.confidence,
                "estimate_high": packet.estimate.high,
                "latency_ms": round(latency_ms, 2),
                "tags": labels.get("tags", ["hand_curated"]),
                "wer": (
                    word_error_rate(labels["reference_transcript"], state["statement"])
                    if state.get("transcribed") and labels.get("reference_transcript")
                    else None
                ),
            }
        )
    return {
        "metrics": aggregate(rows),
        "slices": slices(rows),
        "cases": rows,
        "mode": settings.mode,
    }


def slices(rows: list[dict], min_n: int = 3) -> dict[str, dict]:
    """Decision accuracy per tag (loss type, scenario, fraud type, hard negative...).

    An aggregate score hides where the system fails; slices show it.
    """
    by_tag: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        for tag in r["tags"]:
            by_tag[tag].append(r)
    out = {}
    for tag, group in sorted(by_tag.items()):
        if len(group) < min_n:
            continue
        out[tag] = {
            "n": len(group),
            "decision_accuracy": round(sum(r["correct"] for r in group) / len(group), 4),
            "flag_exact_match": round(
                sum(r["expected_flags"] == r["predicted_flags"] for r in group) / len(group), 4
            ),
        }
    return out


def aggregate(rows: list[dict]) -> dict:
    n = len(rows)
    flag_tp = flag_pred = flag_gold = 0
    for r in rows:
        tp, p, g = precision_recall(
            {(r["claim_id"], f) for f in r["predicted_flags"]},
            {(r["claim_id"], f) for f in r["expected_flags"]},
        )
        flag_tp, flag_pred, flag_gold = flag_tp + tp, flag_pred + p, flag_gold + g
    dmg_tp = sum(r["damage_pr"][0] for r in rows)
    dmg_pred = sum(r["damage_pr"][1] for r in rows)
    dmg_gold = sum(r["damage_pr"][2] for r in rows)
    cited = sum(len(r["cited"]) for r in rows)
    lat = [r["latency_ms"] for r in rows]
    wers = [r["wer"] for r in rows if r.get("wer") is not None]
    asr = {"asr_wer": round(sum(wers) / len(wers), 4), "asr_n": len(wers)} if wers else {}
    return {
        "n_cases": n,
        "decision_accuracy": round(safe_div(sum(r["correct"] for r in rows), n), 4),
        "escalation_rate": round(safe_div(sum(r["predicted"] == "escalate" for r in rows), n), 4),
        f"retrieval_recall_at_{K}": round(safe_div(sum(r["recall_at_k"] for r in rows), n), 4),
        f"retrieval_ndcg_at_{K}": round(safe_div(sum(r["ndcg_at_k"] for r in rows), n), 4),
        "citation_precision": round(safe_div(sum(r["cited_relevant"] for r in rows), cited), 4),
        "damage_precision": round(safe_div(dmg_tp, dmg_pred), 4),
        "damage_recall": round(safe_div(dmg_tp, dmg_gold), 4),
        "fraud_precision": round(safe_div(flag_tp, flag_pred), 4),
        "fraud_recall": round(safe_div(flag_tp, flag_gold), 4),
        "latency_p50_ms": round(percentile(lat, 0.5), 2),
        "latency_p95_ms": round(percentile(lat, 0.95), 2),
        **asr,
    }


def print_report(report: dict, max_misses: int = 25) -> None:
    print("\nClaimSight eval report")
    for key, value in report["metrics"].items():
        print(f"  {key:<26} {value}")
    if report.get("slices"):
        print(f"\n  {'slice':<28} {'n':>4}  {'decision':>8}  {'flags':>6}")
        for tag, s in report["slices"].items():
            print(
                f"  {tag:<28} {s['n']:>4}  {s['decision_accuracy']:>8.2f}  "
                f"{s['flag_exact_match']:>6.2f}"
            )
    misses = [
        r
        for r in report["cases"]
        if not r["correct"] or r["expected_flags"] != r["predicted_flags"]
    ]
    if misses:
        print(f"\nMismatches ({len(misses)}, showing up to {max_misses})")
        for r in misses[:max_misses]:
            print(
                f"  {r['claim_id']:<20} decision {r['expected']}→{r['predicted']}  "
                f"flags {r['expected_flags']}→{r['predicted_flags']}"
            )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--golden", nargs="+", type=Path, default=[Path("evals/golden/claims.jsonl")])
    ap.add_argument("--out", type=Path, default=Path("evals/reports/latest.json"))
    ap.add_argument("--mode", choices=["replay", "live"], default="replay")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args(argv)

    cases = load_cases(args.golden)[: args.limit]
    if not cases:
        print("No cases to evaluate.", file=sys.stderr)
        return 1
    report = evaluate(cases, Settings(mode=args.mode))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, default=list))
    print_report(report)
    print(f"\nWrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
