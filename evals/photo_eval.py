"""Photo check eval: does the VLM flag statements the photos contradict, and only those?

    uv run python -m evals.photo_eval                          # replay
    uv run --extra llm python -m evals.photo_eval --record     # needs the photos + API

188 cases on the detector's 94 held-out CC0 photos, one statement that fits each photo's
labelled damage and one that can't have produced it (scripts/make_photo_cases.py). Two of
the contradicting statements carry a prompt injection.

* flag_recall: contradicting statements the check flagged (FLAG -> escalate)
* specificity: fitting statements it did not flag
* unclear_rate: answers of `unclear` (UNKNOWN: no flag, a small confidence penalty)

The rule check in critic.py can't score here: it needs a stated impact side and part-level
detections, and these statements are about kind and size of damage, not side.

`score_photo_checks` is also called by evals/runner.py, so the gate tracks it.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from claimsight.agents.photo_check import MissingPhotoCheck, to_signal
from claimsight.config import Settings
from claimsight.deps import build_photo_checker
from claimsight.models.damage_data import DAMAGE_DIR
from claimsight.schemas import SignalStatus
from evals.metrics import safe_div

CASES = Path(__file__).parent / "golden" / "photo_cases.jsonl"


def load_cases(path: Path = CASES) -> list[dict]:
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def score_photo_checks(checker, cases: list[dict], record: bool = False) -> dict:
    rows = []
    for case in cases:
        photo = DAMAGE_DIR / "img" / case["file"]
        files = [str(photo)] if record and photo.exists() else None
        try:
            answer = checker.check_hashed(case["statement"], [case["sha256"]], files)
        except MissingPhotoCheck:
            answer = None
        signal = to_signal(answer, checker.name)
        rows.append(
            {
                "cid": case["cid"],
                "expected": case["expected"],
                "profile": case["profile"],
                "tags": case.get("tags", []),
                "verdict": answer.get("verdict") if answer else None,
                "flagged": bool(signal and signal.status == SignalStatus.FLAG),
                "detail": signal.detail if signal else None,
            }
        )
    pos = [r for r in rows if r["expected"] == "inconsistent"]
    neg = [r for r in rows if r["expected"] == "consistent"]
    answered = [r for r in rows if r["verdict"]]
    metrics = {
        "n": len(rows),
        "answered": len(answered),
        "flag_recall": round(safe_div(sum(r["flagged"] for r in pos), len(pos)), 4),
        "specificity": round(safe_div(sum(not r["flagged"] for r in neg), len(neg)), 4),
        "unclear_rate": round(
            safe_div(sum(r["verdict"] == "unclear" for r in answered), len(answered)), 4
        ),
        "injection_resisted": sum(r["flagged"] for r in pos if "injection" in r["tags"]),
    }
    return {"metrics": metrics, "rows": rows}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--record", action="store_true", help="call the API for missing answers")
    ap.add_argument("--out", type=Path, default=Path("evals/reports/photo_check.json"))
    args = ap.parse_args(argv)

    cases = load_cases()
    s = Settings(photo_check="llm", record_photo_check=args.record)
    checker = build_photo_checker(s, online=False)
    report = score_photo_checks(checker, cases, record=args.record)
    if args.record:
        checker.save(prune=True)
    m = report["metrics"]
    print(f"Photo check ({m['n']} statements on {m['n'] // 2} held-out photos)")
    print(f"  checker        {checker.name}  ({m['answered']} answered)")
    print(f"  flag_recall    {m['flag_recall']:.2f}  (contradicting statements flagged)")
    print(f"  specificity    {m['specificity']:.2f}  (fitting statements left alone)")
    print(f"  unclear_rate   {m['unclear_rate']:.2f}")
    print(f"  injections     {m['injection_resisted']} / 2 still flagged")
    for profile in ("heavy", "light"):
        rows = [r for r in report["rows"] if r["profile"] == profile]
        pos = [r for r in rows if r["expected"] == "inconsistent"]
        neg = [r for r in rows if r["expected"] == "consistent"]
        print(
            f"  {profile:<6} photos ({len(neg)}): flagged {sum(r['flagged'] for r in pos)}/"
            f"{len(pos)} contradicting, {sum(r['flagged'] for r in neg)}/{len(neg)} fitting"
        )
    misses = [r for r in report["rows"] if r["flagged"] != (r["expected"] == "inconsistent")]
    if misses:
        print(f"\nMisses ({len(misses)}):")
        for r in misses:
            print(
                f"  {r['cid']} want {r['expected']:<12} got {r['verdict']}  "
                f"{(r['detail'] or '')[:110]}"
            )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
