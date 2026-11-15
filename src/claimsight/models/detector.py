"""Damage detection: a fine-tuned damage detector plus a zero-shot part finder.

CarDD labels say *what* the damage is (dent, scratch, crack, glass shatter, lamp broken,
tire flat) but not *where*. Repair cost depends on the part, so each damage box is
assigned to the zero-shot part box that contains most of it.
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
        damages = self._damage(image, threshold=0.3)
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
                    damage_type=det["label"].lower().replace(" ", "_"),
                    severity=round(severity, 3),
                    confidence=round(float(det["score"]), 3),
                    photo_id=photo.photo_id,
                    box=dbox,
                )
            )
        return items
