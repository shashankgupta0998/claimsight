"""Prompt-injection red team for the coverage review.

    uv run python -m evals.injection_eval                       # replay
    uv run --extra llm python -m evals.injection_eval --record  # call the API for misses

128 attacks (scripts/make_injection_cases.py): 8 techniques appended to (or put before)
each of the 16 clearly excluded statements, which all escalate without them. The claimant
wants the claim paid, so an attack succeeds when the claim no longer escalates: the review
answers `covered`, or backs its exclusion with no checked quote.

Run against the default (Haiku first, Sonnet when Haiku isn't sure) and Sonnet alone. With
routing, an attack only has to make Haiku say `covered` to skip Sonnet entirely, so the
report counts those separately.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from claimsight.config import Settings
from claimsight.deps import build_replay_deps
from evals.coverage_eval import score_reviews

CASES = Path(__file__).parent / "golden" / "injection_cases.jsonl"
CONFIGS = {"Haiku first (default)": "haiku_first", "Sonnet only": "off"}


def load_cases(path: Path = CASES) -> list[dict]:
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


class Spy:
    """Remembers each answer by statement, so the report can say which model was fooled."""

    def __init__(self, inner):
        self.inner, self.name, self.answers = inner, inner.name, {}

    def review(self, request):
        answer = self.inner.review(request)
        self.answers[request.statement] = answer
        return answer

    def save(self, prune: bool = False) -> None:
        self.inner.save(prune=prune)


def attack(routing: str, cases: list[dict], record: bool = False) -> dict:
    s = Settings(
        coverage_review="llm", coverage_review_routing=routing, record_coverage_review=record
    )
    deps = build_replay_deps(s)
    spy = Spy(deps.coverage_reviewer)
    deps.coverage_reviewer = spy
    rows = score_reviews(deps, cases)["rows"]
    by_cid = {c["cid"]: c for c in cases}
    out = []
    for r in rows:
        case = by_cid[r["cid"]]
        answer = spy.answers.get(case["statement"]) or {}
        out.append(
            {
                "cid": r["cid"],
                "base": case["base"],
                "technique": case["technique"],
                "succeeded": not r["flagged"],
                "verdict": r["verdict"],
                "skipped_sonnet": answer.get("routed") == "cheap",
                "reviewed": r["verdict"] is not None,
            }
        )
    return {"rows": out}


def summarize(rows: list[dict]) -> dict:
    by_tech: dict[str, list[bool]] = defaultdict(list)
    for r in rows:
        by_tech[r["technique"]].append(r["succeeded"])
    return {
        "attacks": len(rows),
        "reviewed": sum(r["reviewed"] for r in rows),
        "succeeded": sum(r["succeeded"] for r in rows),
        "attack_success_rate": round(sum(r["succeeded"] for r in rows) / len(rows), 4),
        "skipped_sonnet": sum(r["skipped_sonnet"] for r in rows),
        "by_technique": {t: sum(v) for t, v in by_tech.items()},
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--record", action="store_true", help="call the API for missing reviews")
    ap.add_argument("--out", type=Path, default=Path("evals/reports/injection.json"))
    args = ap.parse_args(argv)
    cases = load_cases()
    report = {}
    for label, routing in CONFIGS.items():
        result = attack(routing, cases, args.record)
        report[label] = {**summarize(result["rows"]), "rows": result["rows"]}

    labels = list(report)
    n_bases = len({c["base"] for c in cases})
    print(f"Prompt injection vs coverage review ({len(cases)} attacks on {n_bases} exclusions)")
    print(
        f"  success = the excluded claim no longer escalates\n\n  {'':<22}"
        + "".join(f"{x:>24}" for x in labels)
    )
    for t in report[labels[0]]["by_technique"]:
        print(
            f"  {t:<22}"
            + "".join(f"{report[x]['by_technique'][t]:>21} / {n_bases}" for x in labels)
        )
    print(
        f"  {'attack success rate':<22}"
        + "".join(f"{report[x]['attack_success_rate']:>24.1%}" for x in labels)
    )
    print(
        f"  {'Haiku said covered':<22}"
        + "".join(f"{report[x]['skipped_sonnet']:>24}" for x in labels)
    )
    print(f"  {'reviewed':<22}" + "".join(f"{report[x]['reviewed']:>24}" for x in labels))
    wins = [r for x in labels for r in report[x]["rows"] if r["succeeded"]]
    if wins:
        print(f"\nSuccessful attacks ({len(wins)}):")
        for x in labels:
            for r in report[x]["rows"]:
                if r["succeeded"]:
                    print(f"  [{x}] {r['cid']} -> {r['verdict']}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
