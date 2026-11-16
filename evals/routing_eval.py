"""Coverage review cost vs quality: Sonnet alone, or Haiku first and Sonnet when it isn't sure.

    uv run python -m evals.routing_eval

Replays both configurations from the cassettes (coverage.json for Sonnet,
coverage.haiku.json for Haiku; `make record-coverage` fills both) over

* the 35 labelled exclusion statements (evals/coverage_eval.py): exclusion recall and
  specificity, the numbers the gate tracks, and
* every statement the 257-case claims set sends to review: how many it escalates.

Each runs with both prompt layouts: `statement_first` (one block, nothing cacheable) and
`cached_prefix` (screening clauses first as a cached block). Cost comes from each call's
recorded token usage at list prices (PRICES in
agents/coverage_review.py), with and without the prompt cache. The recordings ran one
request at a time, so after the first call every request found the shared prefix cached;
traffic with gaps over five minutes would pay the cache write more often.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from claimsight.agents.coverage_review import answer_cost
from claimsight.config import Settings
from claimsight.deps import build_replay_deps
from evals.coverage_eval import load_cases, score_reviews
from evals.runner import evaluate
from evals.runner import load_cases as load_claims

GOLDEN = Path(__file__).parent / "golden"
CLAIM_FILES = ("claims.jsonl", "variants.jsonl", "synthetic.jsonl")
# label: (prompt layout, routing)
CONFIGS = {
    "Sonnet": ("statement_first", "off"),
    "Sonnet, cached": ("cached_prefix", "off"),
    "Haiku first": ("statement_first", "haiku_first"),
    "Haiku first, cached": ("cached_prefix", "haiku_first"),
}


class Meter:
    """Wraps a reviewer and adds up what its answers cost."""

    def __init__(self, inner):
        self.inner, self.name = inner, inner.name
        self.answers: dict[str, dict | None] = {}

    def review(self, request):
        answer = self.inner.review(request)
        self.answers[request.render()] = answer  # one bill per distinct request
        return answer

    def totals(self) -> dict:
        answers = list(self.answers.values())
        n = len(answers)
        cached = sum(answer_cost(a) for a in answers)
        uncached = sum(answer_cost(a, cached=False) for a in answers)
        routed = [a for a in answers if a and "routed" in a]
        to_strong = sum(a["routed"] == "strong" for a in routed)
        return {
            "requests": n,
            "usd_per_review": round(cached / n, 5) if n else 0.0,
            "usd_per_review_uncached": round(uncached / n, 5) if n else 0.0,
            # Without routing every review is a Sonnet call.
            "sent_to_sonnet": round(to_strong / len(routed), 4) if routed else 1.0,
        }


def run(layout: str, routing: str) -> dict:
    s = Settings(
        coverage_review="llm", coverage_review_routing=routing, coverage_review_layout=layout
    )
    deps = build_replay_deps(s)
    meter = Meter(deps.coverage_reviewer)
    deps.coverage_reviewer = meter
    quality = score_reviews(deps, load_cases())["metrics"]
    claims = evaluate(load_claims([GOLDEN / f for f in CLAIM_FILES]), s, deps=deps)
    return {
        "reviewer": meter.name,
        "exclusion_recall": quality["exclusion_recall"],
        "specificity": quality["specificity"],
        "quote_validity": quality["quote_validity"],
        "claims_escalated_by_review": claims["metrics"]["coverage_review_claims_escalated"],
        "decision_accuracy": claims["metrics"]["decision_accuracy"],
        **meter.totals(),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--out", type=Path, default=Path("evals/reports/routing.json"))
    args = ap.parse_args(argv)
    report = {label: run(*config) for label, config in CONFIGS.items()}
    rows = [
        ("exclusion recall (35 set)", "exclusion_recall", ".2f"),
        ("specificity (35 set)", "specificity", ".2f"),
        ("quote validity", "quote_validity", ".2f"),
        ("claims escalated by review", "claims_escalated_by_review", "d"),
        ("decision accuracy (claims)", "decision_accuracy", ".2f"),
        ("distinct reviews", "requests", "d"),
        ("sent to Sonnet", "sent_to_sonnet", ".0%"),
        ("$ per review, cached", "usd_per_review", ".4f"),
        ("$ per review, no cache", "usd_per_review_uncached", ".4f"),
    ]
    labels = list(report)
    print(f"Coverage review routing\n\n  {'':<28}" + "".join(f"{x:>21}" for x in labels))
    for title, key, fmt in rows:
        print(f"  {title:<28}" + "".join(f"{report[x][key]:>21{fmt}}" for x in labels))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
