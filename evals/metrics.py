"""Metric functions: pure, dependency-free and unit-tested.

A bug in a metric would otherwise show up as a model gain.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Hashable, Iterable, Sequence


def recall_at_k(retrieved: Sequence[str], relevant: Iterable[str], k: int) -> float:
    rel = set(relevant)
    if not rel:
        return 1.0
    return len(rel & set(retrieved[:k])) / len(rel)


def ndcg_at_k(retrieved: Sequence[str], relevant: Iterable[str], k: int) -> float:
    rel = set(relevant)
    dcg = sum(1 / math.log2(i + 2) for i, doc in enumerate(retrieved[:k]) if doc in rel)
    ideal = sum(1 / math.log2(i + 2) for i in range(min(len(rel), k)))
    return dcg / ideal if ideal else 1.0


def precision_recall(
    predicted: Iterable[Hashable], gold: Iterable[Hashable]
) -> tuple[int, int, int]:
    """Return (true positives, predicted count, gold count) so callers can micro-average."""
    p, g = set(predicted), set(gold)
    return len(p & g), len(p), len(g)


def safe_div(num: float, den: float, empty: float = 1.0) -> float:
    return num / den if den else empty


def cohen_kappa(a: Sequence[Hashable], b: Sequence[Hashable]) -> float:
    """Agreement between two raters beyond chance. Calibrates the LLM judge against humans."""
    if len(a) != len(b) or not a:
        raise ValueError("Raters must label the same, non-empty set of items.")
    n = len(a)
    observed = sum(x == y for x, y in zip(a, b, strict=True)) / n
    ca, cb = Counter(a), Counter(b)
    expected = sum(ca[k] * cb.get(k, 0) for k in ca) / (n * n)
    return 1.0 if expected == 1 else (observed - expected) / (1 - expected)


def percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    idx = (len(s) - 1) * q
    lo, hi = math.floor(idx), math.ceil(idx)
    return s[lo] + (s[hi] - s[lo]) * (idx - lo)


def word_error_rate(reference: str, hypothesis: str) -> float:
    """Word-level edit distance / reference length, ignoring case and punctuation."""
    ref = re.findall(r"[a-z0-9']+", reference.lower())
    hyp = re.findall(r"[a-z0-9']+", hypothesis.lower())
    if not ref:
        return 0.0 if not hyp else 1.0
    prev = list(range(len(hyp) + 1))
    for i, rw in enumerate(ref, 1):
        cur = [i] + [0] * len(hyp)
        for j, hw in enumerate(hyp, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (rw != hw))
        prev = cur
    return prev[-1] / len(ref)
