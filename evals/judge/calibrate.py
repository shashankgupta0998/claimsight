"""Compare LLM-judge labels with human labels before trusting the judge.

    uv run python -m evals.judge.calibrate --human human.jsonl --judge judge.jsonl

Each file holds lines like {"id": "CLM-0001:PP-D-COMP", "label": "supported"}.
Rule of thumb: kappa >= 0.6 is usable, >= 0.8 is strong. Report the number in the README.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from evals.metrics import cohen_kappa


def _load(path: Path) -> dict[str, str]:
    rows = [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
    return {r["id"]: r["label"] for r in rows}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--human", type=Path, required=True)
    ap.add_argument("--judge", type=Path, required=True)
    args = ap.parse_args(argv)

    human, judge = _load(args.human), _load(args.judge)
    ids = sorted(human.keys() & judge.keys())
    if not ids:
        print("No overlapping ids between the two files.")
        return 1
    a, b = [human[i] for i in ids], [judge[i] for i in ids]
    kappa = cohen_kappa(a, b)
    pairs = list(zip(a, b, strict=True))
    agreement = sum(x == y for x, y in pairs) / len(ids)
    print(f"items: {len(ids)}  agreement: {agreement:.3f}  kappa: {kappa:.3f}")
    print("disagreements (human → judge):")
    for (h, j), n in Counter((x, y) for x, y in pairs if x != y).most_common():
        print(f"  {h} → {j}: {n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
