"""BM25 over policy clauses. The sparse baseline every dense/visual retriever must beat."""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path

from claimsight.retrieval.base import RetrievedChunk

_TOKEN = re.compile(r"[a-z0-9]+")
_STOP = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "in",
    "is",
    "it",
    "its",
    "me",
    "my",
    "of",
    "on",
    "or",
    "the",
    "to",
    "was",
    "we",
    "were",
    "while",
    "with",
    "you",
    "your",
    "i",
}


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in _STOP]


class BM25Retriever:
    name = "bm25"

    def __init__(self, chunks: list[dict], k1: float = 1.5, b: float = 0.75):
        self._chunks = chunks
        self._docs = [Counter(tokenize(c["text"])) for c in chunks]
        self._lens = [sum(d.values()) for d in self._docs]
        self._avg = sum(self._lens) / max(len(self._lens), 1)
        df: Counter[str] = Counter()
        for d in self._docs:
            df.update(d.keys())
        n = len(chunks)
        self._idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}
        self.k1, self.b = k1, b

    @classmethod
    def from_jsonl(cls, path: Path) -> BM25Retriever:
        return cls([json.loads(line) for line in path.read_text().splitlines() if line.strip()])

    def search(self, query: str, k: int = 5) -> list[RetrievedChunk]:
        q = tokenize(query)
        scored = []
        for chunk, tf, length in zip(self._chunks, self._docs, self._lens, strict=True):
            score = 0.0
            for term in q:
                if term not in tf:
                    continue
                f = tf[term]
                norm = (
                    f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * length / self._avg))
                )
                score += self._idf[term] * norm
            if score > 0:
                scored.append((score, chunk))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [
            RetrievedChunk(
                clause_id=c["clause_id"],
                doc_id=c["doc_id"],
                page=c.get("page"),
                text=c["text"],
                score=round(s, 4),
                metadata=c.get("metadata", {}),
            )
            for s, c in scored[:k]
        ]
