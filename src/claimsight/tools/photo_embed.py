"""Reused-photo search with DINOv2 embeddings, alongside the 64-bit dHash in photos.py.

dHash survives resizing and recompression, but a crop, a mirror image or a photo of a
screen changes most of its bits. DINOv2 (self-supervised ViT, `facebook/dinov2-small`,
Apache-2.0) maps each photo to a 384-d vector that describes what's in it, so an edited
copy of an earlier claim's photo still lands next to the original.

* DinoEmbedder: the model (needs the `ml` extra). CLS token, L2-normalised.
* EmbeddingPhotoIndex: earlier claims' vectors; nearest() returns the best cosine. In
  production this is a pgvector HNSW index, like the policy clauses.

Measured against dHash in evals/duplicate_eval.py.
"""

from __future__ import annotations

from pathlib import Path

from claimsight.retrieval.embed import Vector, cosine, decode_vector

DINO_MODEL = "facebook/dinov2-small"


class DinoEmbedder:
    def __init__(self, model: str = DINO_MODEL, device: str | None = None):
        import torch
        from transformers import AutoImageProcessor, AutoModel

        if device is None:
            device = "mps" if torch.backends.mps.is_available() else "cpu"
        self._torch, self._device, self.name = torch, device, model
        self._processor = AutoImageProcessor.from_pretrained(model)
        self._model = AutoModel.from_pretrained(model).to(device).eval()

    def embed_images(self, images: list) -> list[Vector]:
        with self._torch.no_grad():
            batch = self._processor(
                images=[im.convert("RGB") for im in images], return_tensors="pt"
            )
            out = self._model(**batch.to(self._device)).pooler_output
            out = self._torch.nn.functional.normalize(out.float(), dim=-1)
        return [[round(v, 5) for v in row] for row in out.cpu().tolist()]

    def embed(self, path: str | Path) -> Vector:
        from PIL import Image, ImageOps

        with Image.open(path) as im:
            return self.embed_images([ImageOps.exif_transpose(im)])[0]


class EmbeddingPhotoIndex:
    """Linear scan over earlier claims' photo vectors (base64 float16 or plain lists)."""

    name = "dinov2-index"

    def __init__(self, entries: list[dict] | None = None):
        self._entries = []
        for e in entries or []:
            vec = e["embedding"]
            self._entries.append(
                {**e, "embedding": decode_vector(vec) if isinstance(vec, str) else vec}
            )

    def add(self, claim_id: str, photo_id: str, embedding: Vector) -> None:
        self._entries.append({"claim_id": claim_id, "photo_id": photo_id, "embedding": embedding})

    def nearest(self, embedding: Vector, exclude_claim_id: str) -> tuple[dict, float] | None:
        best = None
        for e in self._entries:
            if e["claim_id"] == exclude_claim_id:
                continue
            s = cosine(embedding, e["embedding"])
            if best is None or s > best[1]:
                best = (e, s)
        return best
