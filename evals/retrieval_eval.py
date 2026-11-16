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
from claimsight.retrieval.factory import build_page_scorer, build_retrieval, load_chunks
from claimsight.retrieval.pages import PageLaneRetriever
from evals.metrics import ndcg_at_k, recall_at_k, safe_div

QUERIES = Path(__file__).parent / "golden" / "retrieval_queries.jsonl"
ALL = 100  # more than the corpus holds: a full ranking

# (label, retriever, rerank, query_rewrite, page_lane): the configurations the ablation
# compares.
CONFIGS = [
    ("keyword", "keyword", False, "off", "off"),
    ("dense", "dense", False, "off", "off"),
    ("hybrid (RRF)", "hybrid", False, "off", "off"),
    ("keyword > rerank", "keyword", True, "off", "off"),
    ("hybrid > rerank", "hybrid", True, "off", "off"),
    ("dense > rerank", "dense", True, "off", "off"),
    ("rewrite: replace", "dense", True, "replace", "off"),
    ("rewrite: expand", "dense", True, "expand", "off"),
    ("rewrite: joint", "dense", True, "joint", "off"),
    ("rewrite: dual", "dense", True, "dual", "off"),
    ("dense + pages", "dense", False, "off", "colqwen2"),
    ("dense + pages > rerank", "dense", True, "off", "colqwen2"),
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


def page_ranking(retriever: Retriever, query: str) -> list[int]:
    """Pages in the order their clauses first appear in the retriever's full ranking."""
    if isinstance(retriever, PageLaneRetriever):
        return [p for p, _ in retriever.rank_pages(query)]
    return list(dict.fromkeys(c.page for c in retriever.search(query, k=ALL)))


def score_pages(retriever: Retriever, queries: list[dict], pages: dict[str, int]) -> dict:
    """Page-level retrieval: is a page holding a relevant clause ranked first / in the top 3?"""
    hit1 = r3 = 0.0
    for q in queries:
        want = {pages[c] for c in q["relevant"]}
        got = page_ranking(retriever, q["query"])
        hit1 += got[:1] != [] and got[0] in want
        r3 += len(want & set(got[:3])) / len(want)
    n = len(queries)
    return {"page_hit_at_1": round(hit1 / n, 4), "page_recall_at_3": round(r3 / n, 4)}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--vector-store", choices=["memory", "pgvector"], default="memory")
    ap.add_argument("--out", type=Path, default=Path("evals/reports/retrieval.json"))
    ap.add_argument("--show-misses", action="store_true")
    ap.add_argument(
        "--misses-for", default="dense > rerank", help="config whose misses --show-misses prints"
    )
    args = ap.parse_args(argv)

    queries = load_queries()
    report = {}
    print(f"Policy-QA retrieval ({len(queries)} questions, k=5, {args.vector_store})\n")
    print(f"  {'config':<22} {'hit@1':>6} {'R@5':>6} {'nDCG@5':>7} {'MRR':>6} {'gap R@5':>8}")
    for label, retriever, rerank, rewrite, page_lane in CONFIGS:
        s = Settings(
            retriever=retriever,
            rerank=rerank,
            query_rewrite=rewrite,
            page_lane=page_lane,
            vector_store=args.vector_store,
        )
        result = score_retriever(build_retrieval(s).retriever, queries)
        report[label] = result
        m = result["metrics"]
        print(
            f"  {label:<22} {m['hit_at_1']:>6.2f} {m['recall_at_5']:>6.2f} "
            f"{m['ndcg_at_5']:>7.2f} {m['mrr']:>6.2f} {m['lexical_gap_recall_at_5']:>8.2f}"
        )
    # Page level: what a page-image index would hand an adjuster, before any clause split.
    chunks = load_chunks(Settings().golden_dir)
    page_of = {c["clause_id"]: c["page"] for c in chunks}
    page_lane = PageLaneRetriever(chunks, build_page_scorer(Settings(), online=False))
    print(f"\nPage-level ({len({c['page'] for c in chunks})} rendered pages)\n")
    print(f"  {'config':<22} {'page hit@1':>10} {'page R@3':>9}")
    for label, retriever in [
        ("dense", build_retrieval(Settings(retriever="dense", rerank=False)).retriever),
        ("dense > rerank", build_retrieval(Settings()).retriever),
        ("pages (ColQwen2)", page_lane),
    ]:
        m = score_pages(retriever, queries, page_of)
        report[f"page-level: {label}"] = m
        print(f"  {label:<22} {m['page_hit_at_1']:>10.2f} {m['page_recall_at_3']:>9.2f}")
    if args.show_misses:
        default = report[args.misses_for]["rows"]
        print(f"\nMisses for {args.misses_for} (no relevant clause in the top 5):")
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
