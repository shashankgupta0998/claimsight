"""Record embedding and rerank cassettes for every text the evals ask about.

    uv run --extra embed python scripts/record_retrieval.py

Runs the claims eval (all golden files) and every retrieval ablation config with the real
embedding model and cross-encoder filling in anything missing, then writes
evals/golden/cassettes/embeddings.json and rerank.json with only the entries still in use.

Also records LLM query rewrites (evals/golden/cassettes/rewrites.json) for the rewrite
configs; that part needs the `llm` extra and Anthropic credentials, and costs a few cents.

Re-run after changing policy clauses, retrieval queries, the peril queries in
agents/coverage.py, the embedding model, the reranker, or the rewrite prompt or model.
Claim statements don't matter: coverage retrieves by peril. Replay fails loudly on a
missing entry, so CI catches a stale cassette.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from claimsight.agents.coverage import DECLARATIONS_QUERY, PERIL_QUERIES  # noqa: E402
from claimsight.config import Settings  # noqa: E402
from claimsight.deps import build_replay_deps  # noqa: E402
from claimsight.retrieval.factory import build_retrieval, load_chunks  # noqa: E402
from claimsight.retrieval.rewrite import joint_query  # noqa: E402
from evals.retrieval_eval import CONFIGS, load_queries, score_retriever  # noqa: E402
from evals.runner import evaluate, load_cases  # noqa: E402

GOLDEN = ROOT / "evals" / "golden"


def main() -> None:
    settings = Settings(record_retrieval=True)
    deps = build_replay_deps(settings)
    stack = deps.retrieval
    cases = load_cases([GOLDEN / f for f in ("claims.jsonl", "variants.jsonl", "synthetic.jsonl")])
    print(f"Recording the claims eval ({len(cases)} cases) with {deps.retriever.name}")
    evaluate(cases, settings, deps=deps)
    # Every loss type, not only those the claims set happens to use (the coverage review
    # eval has thefts; the claims set has none).
    for q in [*PERIL_QUERIES.values(), DECLARATIONS_QUERY]:
        deps.retriever.search(q)

    queries = load_queries()
    rewriter = None
    for label, retriever, rerank, rewrite in CONFIGS:
        s = Settings(
            record_retrieval=True, retriever=retriever, rerank=rerank, query_rewrite=rewrite
        )
        ablation = build_retrieval(
            s, embedder=stack.embedder, reranker=stack.reranker, rewriter=rewriter
        )
        rewriter = rewriter or ablation.rewriter
        score_retriever(ablation.retriever, queries)
        print(f"Recorded ablation config: {label}")

    # Score every (query, clause) pair, so any first stage replays on either backend
    # (Postgres's tsvector lane returns a different candidate pool than BM25).
    texts = [c["text"] for c in load_chunks(GOLDEN)]
    all_queries = [q["query"] for q in queries] + [*PERIL_QUERIES.values(), DECLARATIONS_QUERY]
    if rewriter is not None:
        # Rewrites and question+rewrite pairs, for the replace and joint configs.
        for q in [q["query"] for q in queries]:
            r = rewriter.rewrite(q)
            all_queries += [r, joint_query(q, r)]
        all_queries = list(dict.fromkeys(all_queries))
    for q in all_queries:
        stack.reranker.score(q, texts)
    print(f"Scored {len(all_queries)} queries x {len(texts)} clauses with the reranker")

    stack.save_cassettes(prune=True)
    if rewriter is not None:
        rewriter.save(prune=True)
    for cassette in (stack.embedder, stack.reranker, rewriter):
        if cassette is not None:
            print(f"Wrote {cassette.path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
