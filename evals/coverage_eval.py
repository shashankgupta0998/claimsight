"""Coverage review eval: does the LLM catch exclusions in the statement, with real quotes?

    uv run python -m evals.coverage_eval            # replay from cassettes/coverage.json
    uv run --extra llm python -m evals.coverage_eval --record   # call the API for misses

35 hand-written statements on a policy that carries both physical damage coverages, so the
rules approve every one of them. Each is labelled:

* excluded - a stated fact triggers a Part D exclusion or the territory condition
* unclear  - the statement raises an exclusion that more facts would settle
* covered  - hard negatives: the trigger words are there ("Uber", "Mexico", "freezing",
             "race", "police") but the exclusion doesn't apply

plus two prompt-injection statements. A case counts as caught only if the review escalated
it and cited the right clause with a quote that passed the check.

`score_reviews` is also called by evals/runner.py when coverage_review is on, so the gate
tracks it.
"""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

from claimsight.agents.coverage import CoverageAgent
from claimsight.config import Settings
from claimsight.deps import Deps, build_replay_deps
from claimsight.schemas import ClaimInput, Location
from evals.metrics import safe_div

CASES = Path(__file__).parent / "golden" / "coverage_cases.jsonl"


def load_cases(path: Path = CASES) -> list[dict]:
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def _claim(case: dict) -> ClaimInput:
    return ClaimInput(
        claim_id=case["cid"],
        policy_id=case["policy_id"],
        loss_type=case["loss_type"],
        loss_date=date(2026, 6, 1),
        location=Location(lat=39.74, lon=-104.99),
        statement=case["statement"],
    )


def score_reviews(deps: Deps, cases: list[dict]) -> dict:
    agent = CoverageAgent(
        deps.retriever, deps.policies, reviewer=deps.coverage_reviewer, screening=deps.screening
    )
    rows = []
    for case in cases:
        finding, _ = agent.run(_claim(case), case["statement"])
        review = finding.review
        flagged = bool(review and review.applied)
        cited = [c.clause_id for c in finding.citations]
        quotes = review.quotes if review else []
        rows.append(
            {
                "cid": case["cid"],
                "expected": case["expected"],
                "verdict": review.verdict if review else None,
                "flagged": flagged,
                "should_flag": case["expected"] != "covered",
                "right_clause": flagged and case["exclusion"] in cited,
                "quotes": len(quotes),
                "valid_quotes": sum(q.valid for q in quotes),
                "rejected": [q.model_dump() for q in quotes if not q.valid],
                "unbacked": bool(review and review.verdict != "covered" and not review.applied),
                "tags": case["tags"],
                "rationale": finding.rationale,
            }
        )

    pos = [r for r in rows if r["should_flag"]]
    neg = [r for r in rows if not r["should_flag"]]
    n_quotes = sum(r["quotes"] for r in rows)
    metrics = {
        "n": len(rows),
        "reviewed": sum(r["verdict"] is not None for r in rows),
        "exclusion_recall": round(safe_div(sum(r["right_clause"] for r in pos), len(pos)), 4),
        "flag_recall": round(safe_div(sum(r["flagged"] for r in pos), len(pos)), 4),
        "specificity": round(safe_div(sum(not r["flagged"] for r in neg), len(neg)), 4),
        "quote_validity": round(safe_div(sum(r["valid_quotes"] for r in rows), n_quotes), 4),
        "unbacked_findings": sum(r["unbacked"] for r in rows),
    }
    return {"metrics": metrics, "rows": rows}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--record", action="store_true", help="call the API for missing reviews")
    ap.add_argument("--out", type=Path, default=Path("evals/reports/coverage.json"))
    args = ap.parse_args(argv)

    cases = load_cases()
    s = Settings(coverage_review="llm", record_coverage_review=args.record)
    deps = build_replay_deps(s)
    report = score_reviews(deps, cases)
    if args.record:
        deps.coverage_reviewer.save()

    m = report["metrics"]
    pos = sum(c["expected"] != "covered" for c in cases)
    print(f"Coverage review ({m['n']} statements: {pos} should escalate, {m['n'] - pos} covered)")
    print(f"  reviewer           {deps.coverage_reviewer.name}  ({m['reviewed']} reviewed)")
    print("  rules only         exclusion recall 0.00, specificity 1.00 (never reads statement)")
    for key in ("exclusion_recall", "flag_recall", "specificity", "quote_validity"):
        print(f"  {key:<18} {m[key]:.2f}")
    print(f"  unbacked_findings  {m['unbacked_findings']}  (exclusion claimed, no checked quote)")

    misses = [
        r
        for r in report["rows"]
        if r["should_flag"] != r["flagged"] or (r["should_flag"] and not r["right_clause"])
    ]
    if misses:
        print(f"\nMisses ({len(misses)}):")
        for r in misses:
            print(f"  {r['cid']} want {r['expected']:<8} got {r['verdict']}  {r['rationale'][:90]}")
    rejected = [(r["cid"], q) for r in report["rows"] for q in r["rejected"]]
    if rejected:
        print(f"\nRejected quotes ({len(rejected)}):")
        for cid, q in rejected:
            print(f"  {cid} {q['clause_id']}: {q['quote'][:70]!r} ({q['problem']})")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
