from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol


@dataclass(frozen=True)
class RetrievedChunk:
    clause_id: str
    doc_id: str
    page: int | None
    text: str
    score: float
    metadata: dict = field(default_factory=dict)


class Retriever(Protocol):
    name: str

    def search(self, query: str, k: int = 5) -> list[RetrievedChunk]: ...
