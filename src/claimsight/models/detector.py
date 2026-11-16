"""Damage detection: a fine-tuned damage detector plus a zero-shot part finder.

The detector (RT-DETRv2 fine-tuned on CC0 car-damage photos, scripts/train_detector.py)
says *what* the damage is but not *where*. Repair cost depends on the part, so each damage
box is assigned to the zero-shot part box that contains most of it, and the part then
settles labels the detector can't: a broken headlamp is `lamp_broken`, a broken windshield
`glass_shatter` (the pipeline's types, from CarDD's six classes).
"""

from __future__ import annotations

from claimsight.schemas import DamageItem, Photo

Box = tuple[float, float, float, float]  # xmin, ymin, xmax, ymax

# Prompt text for the zero-shot detector -> canonical part id used by the estimator.
PART_PROMPTS: dict[str, str] = {
    "front bumper": "bumper_front",
    "rear bumper": "bumper_rear",
    "car hood": "hood",
    "car roof": "roof",
    "car door": "door",
    "windshield": "windshield",
    "headlight": "headlamp",
    "taillight": "taillamp",
    "fender": "fender",
    "trunk lid": "trunk",
    "tire": "tire",
}


# Detector label -> pipeline damage type. Paint damage of any kind costs like a scratch.
DAMAGE_TYPES = {
    "dent": "dent",
    "scratch": "scratch",
    "paint_chip": "scratch",
    "flaking": "scratch",
    "crack": "crack",
    "broken_part": "broken_part",
    "missing_part": "missing_part",
    "corrosion": "corrosion",
}
BROKEN_BY_PART = {
    "headlamp": "lamp_broken",
    "taillamp": "lamp_broken",
    "windshield": "glass_shatter",
}


# Live score cutoff. On the held-out photos, 0.3 kept ~26 boxes a photo at 13% precision;
# 0.5 keeps ~4 at 36% (IoU 0.5, right class), with a correct box on 72 of 94 photos.
DAMAGE_THRESHOLD = 0.5


def pipeline_damage_type(label: str, part: str) -> str:
    """Map a detector label onto the estimator's damage types, using the part it sits on."""
    label = label.lower().replace(" ", "_")
    kind = DAMAGE_TYPES.get(label, label)
    return BROKEN_BY_PART.get(part, kind) if kind == "broken_part" else kind


def box_area(b: Box) -> float:
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def overlap_fraction(inner: Box, outer: Box) -> float:
    """Share of `inner` covered by `outer`. Damage sits inside a part, so IoU would undercount."""
    ix = max(0.0, min(inner[2], outer[2]) - max(inner[0], outer[0]))
    iy = max(0.0, min(inner[3], outer[3]) - max(inner[1], outer[1]))
    area = box_area(inner)
    return (ix * iy) / area if area else 0.0


class ReplayDetector:
    """Returns detections recorded on the photo. Deterministic and free, so CI can run it."""

    name = "replay-detector"

    def detect(self, photo: Photo) -> list[DamageItem]:
        return [
            d.model_copy(update={"photo_id": photo.photo_id})
            for d in photo.recorded_detections or []
        ]


class HFDetector:
    """Live detector. Needs the `ml` extra and a damage model fine-tuned on CarDD."""

    def __init__(self, damage_model: str, part_model: str, device: int | str | None = None):
        from transformers import pipeline

        self._damage = pipeline("object-detection", model=damage_model, device=device)
        self._parts = pipeline("zero-shot-object-detection", model=part_model, device=device)
        self.name = damage_model

    @staticmethod
    def _box(det: dict) -> Box:
        b = det["box"]
        return (b["xmin"], b["ymin"], b["xmax"], b["ymax"])

    def detect(self, photo: Photo) -> list[DamageItem]:
        if not photo.uri:
            return []
        from PIL import Image

        with Image.open(photo.uri) as im:
            image = im.convert("RGB")
        damages = self._damage(image, threshold=DAMAGE_THRESHOLD)
        parts = self._parts(image, candidate_labels=list(PART_PROMPTS), threshold=0.15)

        items = []
        for det in damages:
            dbox = self._box(det)
            best_part, best_cover, part_box = "unknown", 0.0, None
            for p in parts:
                cover = overlap_fraction(dbox, self._box(p))
                if cover > best_cover:
                    best_part, best_cover, part_box = PART_PROMPTS[p["label"]], cover, self._box(p)
            # Severity proxy: share of the part's area that is damaged. Replace with a
            # learned severity head once you have labelled severities.
            severity = min(1.0, box_area(dbox) / box_area(part_box)) if part_box else 0.5
            items.append(
                DamageItem(
                    part=best_part,
                    damage_type=pipeline_damage_type(det["label"], best_part),
                    severity=round(severity, 3),
                    confidence=round(float(det["score"]), 3),
                    photo_id=photo.photo_id,
                    box=dbox,
                )
            )
        return items
