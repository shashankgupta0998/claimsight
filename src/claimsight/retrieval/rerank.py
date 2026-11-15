"""Second-stage reranking with a cross-encoder, recorded to a cassette for replay.

The first stage (dense, keyword or hybrid) returns a candidate pool; the cross-encoder
reads each (query, clause) pair together and reorders them. On the policy-QA set this is
where top-1 accuracy comes from: see the ablation in `evals/retrieval_eval.py`.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Protocol

from claimsight.retrieval.base import RetrievedChunk, Retriever
from claimsight.retrieval.embed import text_key


class Reranker(Protocol):
    name: str

    def score(self, query: str, texts: list[str]) -> list[float]: ...


class MissingRerankScore(KeyError):
    pass


class CrossEncoderReranker:
    def __init__(self, model: str):
        from sentence_transformers import CrossEncoder

        self._model = CrossEncoder(model)
        self.name = model

    def score(self, query: str, texts: list[str]) -> list[float]:
        return [float(s) for s in self._model.predict([(query, t) for t in texts])]


def pair_key(query: str, text: str) -> str:
    return text_key(query + "\x1f" + text)


class CassetteReranker:
    def __init__(self, path: Path, inner: Reranker | None = None):
        """`inner=None` means offline: read recorded scores, never load a model."""
        self.path, self.inner = path, inner
        data = json.loads(path.read_text()) if path.exists() else {}
        self.name = data.get("model", inner.name if inner else "unknown")
        self._scores: dict[str, float] = data.get("scores", {})
        self._used: set[str] = set()
        if inner and data and data.get("model") != inner.name:
            raise ValueError(f"{path} was recorded with {data.get('model')}, not {inner.name}")

    def score(self, query: str, texts: list[str]) -> list[float]:
        keys = [pair_key(query, t) for t in texts]
        missing = [(k, t) for k, t in zip(keys, texts, strict=True) if k not in self._scores]
        if missing:
            if self.inner is None:
                raise MissingRerankScore(
                    f"{len(missing)} pair(s) for {query[:60]!r} not in {self.path.name}. "
                    "Re-record with: uv run python scripts/record_retrieval.py"
                )
            fresh = self.inner.score(query, [t for _, t in missing])
            for (k, _), s in zip(missing, fresh, strict=True):
                self._scores[k] = round(s, 4)
            self.save()
        self._used.update(keys)
        return [self._scores[k] for k in keys]

    def save(self, prune: bool = False) -> None:
        scores = {k: v for k, v in self._scores.items() if not prune or k in self._used}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"model": self.name, "scores": dict(sorted(scores.items()))}
        self.path.write_text(json.dumps(payload, indent=0) + "\n")


class RerankingRetriever:
    def __init__(
        self,
        first_stage: Retriever,
        reranker: Reranker,
        pool: int = 20,
        rerank_query: Callable[[str], str] | None = None,
    ):
        """`rerank_query` maps the search query to the text the cross-encoder reads."""
        self.first_stage, self.reranker, self.pool = first_stage, reranker, pool
        self.rerank_query = rerank_query or (lambda q: q)
        self.name = f"{first_stage.name}>rerank:{reranker.name}"

    def search(self, query: str, k: int = 5) -> list[RetrievedChunk]:
        candidates = self.first_stage.search(query, k=self.pool)
        if not candidates:
            return []
        scores = self.reranker.score(self.rerank_query(query), [c.text for c in candidates])
        ranked = sorted(zip(scores, candidates, strict=True), key=lambda x: (-x[0], x[1].clause_id))
        return [RetrievedChunk(**{**c.__dict__, "score": round(s, 4)}) for s, c in ranked[:k]]
