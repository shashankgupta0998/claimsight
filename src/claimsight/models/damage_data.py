"""Training data for the damage detector: Humans in the Loop "Car Parts and Car Damages".

The damage half is 814 photos with polygon labels in eight classes, released under CC0 1.0
(https://humansintheloop.org/resources/datasets/car-parts-and-car-damages-dataset/). We use
the copy on the Hugging Face Hub (DrBimmer/car-parts-and-damage-dataset), where the two
halves sit in swapped folders: the damage photos are under "Car parts dataset".

CarDD would match the pipeline's labels better, but its licence needs the lab's signed
consent and forbids redistribution, so weights and cassettes trained on it couldn't ship.

Polygons become boxes (the extent of the exterior ring). The split is a hash of the file
name, so it never depends on download order, and is committed as evals/golden/detector/.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

DAMAGE_DIR = Path("data/hitl-car-damage/Car parts dataset/File1")

# Dataset class -> detector label.
CLASSES = {
    "Dent": "dent",
    "Scratch": "scratch",
    "Cracked": "crack",
    "Broken part": "broken_part",
    "Paint chip": "paint_chip",
    "Flaking": "flaking",
    "Missing part": "missing_part",
    "Corrosion": "corrosion",
}
LABELS = list(CLASSES.values())


def split_of(name: str) -> str:
    """80 / 10 / 10 by a stable hash of the file name."""
    bucket = int(hashlib.sha256(name.encode()).hexdigest(), 16) % 10
    return "test" if bucket == 0 else "val" if bucket == 1 else "train"


def read_annotation(path: Path) -> dict:
    """One Supervisely annotation -> {file, width, height, boxes: [(label, x0, y0, x1, y1)]}."""
    data = json.loads(path.read_text())
    boxes = []
    for obj in data["objects"]:
        label = CLASSES.get(obj["classTitle"])
        pts = obj.get("points", {}).get("exterior") or []
        if label is None or len(pts) < 2:
            continue
        xs, ys = [p[0] for p in pts], [p[1] for p in pts]
        x0, y0, x1, y1 = min(xs), min(ys), max(xs), max(ys)
        if x1 - x0 >= 2 and y1 - y0 >= 2:
            boxes.append((label, float(x0), float(y0), float(x1), float(y1)))
    return {
        "file": path.name.removesuffix(".json"),
        "width": data["size"]["width"],
        "height": data["size"]["height"],
        "boxes": boxes,
    }


def load_split(root: Path = DAMAGE_DIR) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {"train": [], "val": [], "test": []}
    for ann in sorted((root / "ann").glob("*.json")):
        rec = read_annotation(ann)
        if rec["boxes"] and (root / "img" / rec["file"]).exists():
            out[split_of(rec["file"])].append(rec)
    return out
