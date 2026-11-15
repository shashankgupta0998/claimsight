"""Text embedders behind one protocol, with a disk cassette for replay.

* SentenceTransformerEmbedder: the real model (needs the `embed` extra, pulls in torch).
* CassetteEmbedder: wraps a real embedder and records every vector it returns. With
  `offline=True` it never loads a model and only reads the cassette, so CI runs dense and
  hybrid retrieval without torch or a GPU.

Vectors are L2-normalised and stored as base64 float16. That keeps a few hundred
384-d vectors under 1 MB and rounds well below anything that changes a ranking.
"""

from __future__ import annotations

import base64
import hashlib
import json
import struct
from pathlib import Path
from typing import Protocol

Vector = list[float]


class Embedder(Protocol):
    name: str
    dim: int

    def embed(self, texts: list[str]) -> list[Vector]: ...


class MissingEmbedding(KeyError):
    """Replay was asked for a text that was never recorded."""


def text_key(text: str) -> str:
    return hashlib.sha256(text.strip().encode()).hexdigest()[:16]


def encode_vector(vec: Vector) -> str:
    return base64.b64encode(struct.pack(f"<{len(vec)}e", *vec)).decode()


def decode_vector(blob: str) -> Vector:
    raw = base64.b64decode(blob)
    return list(struct.unpack(f"<{len(raw) // 2}e", raw))


def cosine(a: Vector, b: Vector) -> float:
    # Inputs are normalised, so the dot product is the cosine.
    return sum(x * y for x, y in zip(a, b, strict=True))


class SentenceTransformerEmbedder:
    def __init__(self, model: str):
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model)
        self.name = model
        self.dim = self._model.get_embedding_dimension()

    def embed(self, texts: list[str]) -> list[Vector]:
        vecs = self._model.encode(texts, normalize_embeddings=True, batch_size=32)
        return [v.tolist() for v in vecs]


class CassetteEmbedder:
    def __init__(self, path: Path, inner: Embedder | None = None):
        """`inner=None` means offline: read the cassette, never call a model."""
        self.path, self.inner = path, inner
        data = json.loads(path.read_text()) if path.exists() else {}
        self.name = data.get("model", inner.name if inner else "unknown")
        self.dim = data.get("dim", inner.dim if inner else 0)
        self._vectors: dict[str, str] = data.get("vectors", {})
        self._used: set[str] = set()
        if inner and data and data.get("model") != inner.name:
            raise ValueError(f"{path} was recorded with {data.get('model')}, not {inner.name}")

    def __contains__(self, text: str) -> bool:
        return text_key(text) in self._vectors

    def embed(self, texts: list[str]) -> list[Vector]:
        missing = [t for t in dict.fromkeys(texts) if t not in self]
        if missing:
            if self.inner is None:
                raise MissingEmbedding(
                    f"{len(missing)} text(s) not in {self.path.name}, e.g. {missing[0][:80]!r}. "
                    "Re-record with: uv run python scripts/record_retrieval.py"
                )
            for text, vec in zip(missing, self.inner.embed(missing), strict=True):
                self._vectors[text_key(text)] = encode_vector(vec)
            self.save()
        keys = [text_key(t) for t in texts]
        self._used.update(keys)
        return [decode_vector(self._vectors[k]) for k in keys]

    def save(self, prune: bool = False) -> None:
        """Write the cassette. `prune` drops vectors this process never asked for."""
        vectors = {k: v for k, v in self._vectors.items() if not prune or k in self._used}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"model": self.name, "dim": self.dim, "vectors": dict(sorted(vectors.items()))}
        self.path.write_text(json.dumps(payload, indent=0) + "\n")
