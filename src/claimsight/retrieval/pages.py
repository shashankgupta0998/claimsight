"""Page lane: retrieve policy pages as images with ColQwen2, then hand back their clauses.

The text lanes search 42 clauses that were cut out of the policy by hand. Real policies
arrive as PDFs: scanned declarations, tables, running headers, endorsements stapled on.
A visual retriever skips text extraction and chunking. It embeds each page image into one
vector per image patch, the query into one vector per token, and scores a page by summing,
over query tokens, the best-matching patch (late interaction, "MaxSim").

This corpus has no PDF, so `render_pages` draws the specimen policy one page per image:
running header, the section title, each clause under its heading, page footer, and a
coverage table on the declarations page. The images are committed under
`evals/golden/pages/` so every machine scores the same pixels.

The lane ranks pages, then returns the clauses printed on them (best page first), so it
fuses with the text lanes by RRF and goes through the same cross-encoder.

Scores sit behind a cassette like every other model output: one score per (query, page),
tagged with a fingerprint of the page images. Replay never loads the model.
"""

from __future__ import annotations

import hashlib
import json
import textwrap
from pathlib import Path
from typing import Protocol

from claimsight.retrieval.base import RetrievedChunk
from claimsight.retrieval.dense import chunk_to_result
from claimsight.retrieval.embed import text_key

PAGE_W, PAGE_H = 1275, 1650  # US letter at 150 dpi
FORM = "PP 00 01 (specimen)"
RENDER_VERSION = "1"

SECTIONS = {
    "DEC": "DECLARATIONS",
    "DEF": "DEFINITIONS",
    "A": "PART A - LIABILITY COVERAGE",
    "B": "PART B - MEDICAL PAYMENTS COVERAGE",
    "C": "PART C - UNINSURED MOTORISTS COVERAGE",
    "D": "PART D - COVERAGE FOR DAMAGE TO YOUR AUTO",
    "E": "PART E - DUTIES AFTER AN ACCIDENT OR LOSS",
    "F": "PART F - GENERAL PROVISIONS",
    "END": "ENDORSEMENTS",
}

DECLARATIONS_TABLE = [
    ("Coverage", "Limit of liability", "Deductible"),
    ("A  Liability", "$ ________ each accident", "-"),
    ("B  Medical Payments", "$ ________ each person", "-"),
    ("C  Uninsured Motorists", "$ ________ each accident", "-"),
    ("D  Other Than Collision", "Actual cash value", "$ ________"),
    ("D  Collision", "Actual cash value", "$ ________"),
    ("Towing and Labor", "$ ________ per disablement", "-"),
]


def section_of(clause_id: str) -> str:
    key = clause_id.split("-")[0] if clause_id.startswith("END") else clause_id.split("-")[1]
    return SECTIONS[key]


def split_heading(text: str) -> tuple[str | None, str]:
    """`"Exclusions. We do not pay..."` -> `("Exclusions", "We do not pay...")`."""
    head, sep, rest = text.partition(". ")
    words = head.split()
    title_case = all(w[0].isupper() for w in words if len(w) > 3)
    if sep and len(words) <= 8 and title_case:
        return head, rest
    return None, text


def render_pages(chunks: list[dict]):
    """One PIL image per page that holds at least one clause, keyed by page number."""
    from PIL import Image, ImageDraw, ImageFont

    def font(size: int):
        return ImageFont.load_default(size=size)

    small, body, head, title = font(20), font(25), font(29), font(36)
    pages: dict[int, list[dict]] = {}
    for c in chunks:
        pages.setdefault(c["page"], []).append(c)

    images = {}
    margin, width_chars = 110, 80
    for page, clauses in sorted(pages.items()):
        img = Image.new("L", (PAGE_W, PAGE_H), 255)
        d = ImageDraw.Draw(img)
        d.text((margin, 60), "PERSONAL AUTO POLICY", font=small, fill=0)
        d.text((PAGE_W - margin, 60), FORM, font=small, fill=0, anchor="ra")
        d.line((margin, 95, PAGE_W - margin, 95), fill=0, width=2)
        sections = list(dict.fromkeys(section_of(c["clause_id"]) for c in clauses))
        y = 130
        for s in sections:
            d.text((margin, y), s, font=title, fill=0)
            y += 52
        y += 20
        if page == 1:
            y = _draw_table(d, y, margin, head, body)
        for c in clauses:
            heading, text = split_heading(c["text"])
            if heading:
                d.text((margin, y), heading, font=head, fill=0)
                y += 42
            for line in textwrap.wrap(text, width_chars):
                d.text((margin, y), line, font=body, fill=0)
                y += 34
            y += 26
        d.line((margin, PAGE_H - 95, PAGE_W - margin, PAGE_H - 95), fill=0, width=1)
        d.text((PAGE_W // 2, PAGE_H - 70), f"Page {page} of 17", font=small, fill=0, anchor="ma")
        d.text((PAGE_W - margin, PAGE_H - 70), FORM, font=small, fill=0, anchor="ra")
        images[page] = img
    return images


def _draw_table(d, y: int, x: int, head, body) -> int:
    cols = (x, x + 420, x + 820)
    for i, row in enumerate(DECLARATIONS_TABLE):
        for cx, cell in zip(cols, row, strict=True):
            d.text((cx + 10, y + 10), cell, font=head if i == 0 else body, fill=0)
        y += 50
        d.line((x, y, PAGE_W - x, y), fill=0, width=2 if i == 0 else 1)
    for cx in (*cols, PAGE_W - x):
        d.line((cx, y - 50 * len(DECLARATIONS_TABLE), cx, y), fill=0, width=1)
    d.line((x, y - 50 * len(DECLARATIONS_TABLE), PAGE_W - x, y - 50 * len(DECLARATIONS_TABLE)))
    return y + 40


def page_paths(pages_dir: Path) -> dict[int, Path]:
    return {int(p.stem.split("-")[1]): p for p in sorted(pages_dir.glob("page-*.png"))}


def fingerprint(pages_dir: Path) -> str:
    """Hash of every committed page image: a cassette is valid only for these pixels."""
    h = hashlib.sha256()
    for page, path in sorted(page_paths(pages_dir).items()):
        h.update(f"{page}:".encode() + hashlib.sha256(path.read_bytes()).digest())
    return h.hexdigest()[:16]


class PageScorer(Protocol):
    name: str

    def score(self, query: str) -> dict[int, float]: ...


class MissingPageScore(KeyError):
    pass


class ColQwen2PageScorer:
    """Embeds every page image once, then scores each query by MaxSim. Needs `embed`."""

    def __init__(self, model: str, pages_dir: Path):
        import torch
        from PIL import Image
        from transformers import ColQwen2ForRetrieval, ColQwen2Processor

        device = "mps" if torch.backends.mps.is_available() else "cpu"
        if torch.cuda.is_available():
            device = "cuda"
        self._torch, self._device, self.name = torch, device, model
        self._model = ColQwen2ForRetrieval.from_pretrained(
            model, dtype=torch.bfloat16 if device != "cpu" else torch.float32
        ).to(device)
        self._model.eval()
        self._processor = ColQwen2Processor.from_pretrained(model)
        paths = page_paths(pages_dir)
        self._pages = list(paths)
        self._page_emb = []
        with torch.no_grad():
            for page in self._pages:
                image = Image.open(paths[page]).convert("RGB")
                batch = self._processor(images=[image], return_tensors="pt").to(device)
                self._page_emb.append(self._model(**batch).embeddings[0])

    def score(self, query: str) -> dict[int, float]:
        with self._torch.no_grad():
            batch = self._processor(text=[query], return_tensors="pt").to(self._device)
            q = self._model(**batch).embeddings
        scores = self._processor.score_retrieval(list(q), self._page_emb)[0]
        return {p: float(s) for p, s in zip(self._pages, scores.tolist(), strict=True)}


class CassettePageScorer:
    """Records one score per (query, page); offline (`inner=None`) only replays."""

    def __init__(self, path: Path, pages_dir: Path, inner: PageScorer | None = None):
        self.path, self.inner = path, inner
        self.pages = fingerprint(pages_dir)
        data = json.loads(path.read_text()) if path.exists() else {}
        self.name = data.get("model", inner.name if inner else "unknown")
        self._scores: dict[str, dict[str, float]] = data.get("scores", {})
        self._used: set[str] = set()
        if data and data.get("pages") != self.pages:
            if inner is None:
                raise ValueError(
                    f"{path.name} was scored on other page images. Re-record with: make record"
                )
            self._scores = {}
        if inner and data and data.get("model") != inner.name:
            raise ValueError(f"{path} was recorded with {data.get('model')}, not {inner.name}")

    def score(self, query: str) -> dict[int, float]:
        k = text_key(query)
        if k not in self._scores:
            if self.inner is None:
                raise MissingPageScore(
                    f"{query[:60]!r} not in {self.path.name}. Re-record with: make record"
                )
            self._scores[k] = {str(p): round(s, 3) for p, s in self.inner.score(query).items()}
            self.save()
        self._used.add(k)
        return {int(p): s for p, s in self._scores[k].items()}

    def save(self, prune: bool = False) -> None:
        scores = {k: v for k, v in self._scores.items() if not prune or k in self._used}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"model": self.name, "pages": self.pages, "scores": dict(sorted(scores.items()))}
        self.path.write_text(json.dumps(payload, separators=(",", ":"), indent=0) + "\n")


class PageLaneRetriever:
    """Rank pages by the visual score, then return their clauses, best page first."""

    def __init__(self, chunks: list[dict], scorer: PageScorer):
        self._by_page: dict[int, list[dict]] = {}
        for c in chunks:
            self._by_page.setdefault(c["page"], []).append(c)
        self.scorer = scorer
        self.name = f"pages:{scorer.name}"

    def rank_pages(self, query: str) -> list[tuple[int, float]]:
        scores = self.scorer.score(query)
        return sorted(
            ((p, s) for p, s in scores.items() if p in self._by_page), key=lambda x: (-x[1], x[0])
        )

    def search(self, query: str, k: int = 5) -> list[RetrievedChunk]:
        out = []
        for page, s in self.rank_pages(query):
            out += [chunk_to_result(c, s) for c in self._by_page[page]]
            if len(out) >= k:
                break
        return out[:k]
