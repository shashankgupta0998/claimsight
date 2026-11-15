"""Policy-QA retrieval eval: 50 labelled questions over the 42-clause policy corpus.

    uv run python -m evals.retrieval_eval                    # ablation table, replay
    uv run python -m evals.retrieval_eval --vector-store pgvector

The claims golden set only asks for a handful of clauses (collision, other than collision,
glass, declarations), so it can't tell retrievers apart. These questions are written the way
claimants and adjusters ask them, including a `lexical_gap` slice whose wording shares few
words with the clause that answers it ("someone keyed my car" -> malicious mischief).

`score_retriever` is also called by evals/runner.py, so the gate tracks the retriever the
graph actually uses.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from claimsight.config import Settings
from claimsight.retrieval.base import Retriever
from claimsight.retrieval.factory import build_retrieval
from evals.metrics import ndcg_at_k, recall_at_k, safe_div

QUERIES = Path(__file__).parent / "golden" / "retrieval_queries.jsonl"

# (label, retriever, rerank): the configurations the ablation compares.
CONFIGS = [
    ("keyword", "keyword", False),
    ("dense", "dense", False),
    ("hybrid (RRF)", "hybrid", False),
    ("keyword > rerank", "keyword", True),
    ("hybrid > rerank", "hybrid", True),
    ("dense > rerank", "dense", True),
]


def load_queries(path: Path = QUERIES) -> list[dict]:
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def reciprocal_rank(retrieved: list[str], relevant: list[str]) -> float:
    for rank, cid in enumerate(retrieved, start=1):
        if cid in relevant:
            return 1.0 / rank
    return 0.0


def score_retriever(retriever: Retriever, queries: list[dict], k: int = 5) -> dict:
    rows = []
    for q in queries:
        got = [c.clause_id for c in retriever.search(q["query"], k=k)]
        rows.append(
            {
                "qid": q["qid"],
                "retrieved": got,
                "relevant": q["relevant"],
                "tags": q.get("tags", []),
                "r1": recall_at_k(got, q["relevant"], 1),
                "rk": recall_at_k(got, q["relevant"], k),
                "hit1": float(got[:1] != [] and got[0] in q["relevant"]),
                "ndcg": ndcg_at_k(got, q["relevant"], k),
                "rr": reciprocal_rank(got, q["relevant"]),
            }
        )

    def mean(key: str, subset: list[dict]) -> float:
        return round(safe_div(sum(r[key] for r in subset), len(subset)), 4)

    gap = [r for r in rows if "lexical_gap" in r["tags"]]
    return {
        "metrics": {
            "n": len(rows),
            "hit_at_1": mean("hit1", rows),
            f"recall_at_{k}": mean("rk", rows),
            f"ndcg_at_{k}": mean("ndcg", rows),
            "mrr": mean("rr", rows),
            f"lexical_gap_recall_at_{k}": mean("rk", gap),
        },
        "rows": rows,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--vector-store", choices=["memory", "pgvector"], default="memory")
    ap.add_argument("--out", type=Path, default=Path("evals/reports/retrieval.json"))
    ap.add_argument("--show-misses", action="store_true")
    args = ap.parse_args(argv)

    queries = load_queries()
    report = {}
    print(f"Policy-QA retrieval ({len(queries)} questions, k=5, {args.vector_store})\n")
    print(f"  {'config':<18} {'hit@1':>6} {'R@5':>6} {'nDCG@5':>7} {'MRR':>6} {'gap R@5':>8}")
    for label, retriever, rerank in CONFIGS:
        s = Settings(retriever=retriever, rerank=rerank, vector_store=args.vector_store)
        result = score_retriever(build_retrieval(s).retriever, queries)
        report[label] = result
        m = result["metrics"]
        print(
            f"  {label:<18} {m['hit_at_1']:>6.2f} {m['recall_at_5']:>6.2f} "
            f"{m['ndcg_at_5']:>7.2f} {m['mrr']:>6.2f} {m['lexical_gap_recall_at_5']:>8.2f}"
        )
    if args.show_misses:
        default = report["dense > rerank"]["rows"]
        print("\nMisses for dense > rerank (no relevant clause in the top 5):")
        for r in default:
            if r["rk"] == 0:
                q = next(x for x in queries if x["qid"] == r["qid"])
                got = r["retrieved"][:3]
                print(f"  {r['qid']} {q['query'][:60]!r} want {r['relevant']} got {got}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
