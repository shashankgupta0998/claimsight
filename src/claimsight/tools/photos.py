"""Photo forensics: perceptual hashing, EXIF extraction and a reused-photo index."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

_EXIF_IFD = 0x8769
_GPS_IFD = 0x8825
_DATETIME_ORIGINAL = 36867
_DATETIME = 306


def hamming(a: str, b: str) -> int:
    return (int(a, 16) ^ int(b, 16)).bit_count()


def dhash(path: str | Path, size: int = 8) -> str:
    """64-bit difference hash. Survives resizing and recompression, unlike a file checksum."""
    from PIL import Image

    with Image.open(path) as img:
        gray = img.convert("L").resize((size + 1, size), Image.Resampling.LANCZOS)
    px = list(gray.tobytes())
    bits = 0
    for r in range(size):
        for c in range(size):
            left, right = px[r * (size + 1) + c], px[r * (size + 1) + c + 1]
            bits = (bits << 1) | int(left > right)
    return f"{bits:0{size * size // 4}x}"


def _dms_to_deg(value, ref) -> float | None:
    if not value:
        return None
    deg = float(value[0]) + float(value[1]) / 60 + float(value[2]) / 3600
    return -deg if ref in ("S", "W") else deg


def read_exif(path: str | Path) -> tuple[datetime | None, float | None, float | None]:
    from PIL import Image

    with Image.open(path) as img:
        exif = img.getexif()
    raw = exif.get_ifd(_EXIF_IFD).get(_DATETIME_ORIGINAL) or exif.get(_DATETIME)
    taken = None
    if raw:
        try:
            taken = datetime.strptime(str(raw), "%Y:%m:%d %H:%M:%S")
        except ValueError:
            taken = None
    gps = exif.get_ifd(_GPS_IFD)
    return taken, _dms_to_deg(gps.get(2), gps.get(1)), _dms_to_deg(gps.get(4), gps.get(3))


class InMemoryPhotoIndex:
    """Linear scan over prior-claim photo hashes. DINOv2 search is in photo_embed.py."""

    name = "phash-index"

    def __init__(self, entries: list[dict] | None = None):
        self._entries = list(entries or [])

    @classmethod
    def from_jsonl(cls, path: Path) -> InMemoryPhotoIndex:
        return cls.from_files([path])

    @classmethod
    def from_files(cls, paths: list[Path]) -> InMemoryPhotoIndex:
        entries = []
        for path in paths:
            entries += [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
        return cls(entries)

    def add(self, claim_id: str, photo_id: str, phash: str) -> None:
        self._entries.append({"claim_id": claim_id, "photo_id": photo_id, "phash": phash})

    def nearest(self, phash: str, exclude_claim_id: str) -> tuple[dict, int] | None:
        best = None
        for e in self._entries:
            if e["claim_id"] == exclude_claim_id:
                continue
            d = hamming(phash, e["phash"])
            if best is None or d < best[1]:
                best = (e, d)
        return best
