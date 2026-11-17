"""Assemble the damage detector for the Hugging Face Hub: the weights and a model card.

    make detector-hub                     # writes build/detector/, checks the weights
    make detector-hub REPO=<you>/claimsight-damage-rtdetr   # the card names that repo
    hf upload <you>/claimsight-damage-rtdetr build/detector . --repo-type model

The card's numbers come from the recorded test-split eval (evals/detector_eval.py), not
from a copy, so the card can't drift from what CI measures. `--check` (the default when
the `ml` extra and the photos are present) runs the assembled weights through the same
transformers pipeline the live detector uses on a few test photos and compares the boxes
with the cassette, so what goes up is the model that was evaluated.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from evals.detector_eval import detection_map, load  # noqa: E402

MODEL_DIR = ROOT / "models" / "rtdetr-hitl-damage"
FILES = ("config.json", "preprocessor_config.json", "model.safetensors")
PHOTOS = ROOT / "data" / "hitl-car-damage" / "Car parts dataset" / "File1" / "img"
DATASET = "DrBimmer/car-parts-and-damage-dataset"
SOURCE = "https://github.com/shashankgupta0998/claimsight"
PLACEHOLDER = "<this repo>"
CHECK_PHOTOS, CHECK_MIN_SCORE, SCORE_TOL, BOX_TOL = 5, 0.5, 0.001, 1.0

PRETTY = {
    "missing_part": "missing part",
    "broken_part": "broken part",
    "paint_chip": "paint chip",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def card(metrics: dict, base: str, size: int, digest: str, repo: str = PLACEHOLDER) -> str:
    m = metrics
    rows = "\n".join(
        f"| {PRETTY.get(lab, lab)}{' (too few to mean much)' if c['n'] < 10 else ''} "
        f"| {c['ap']:.3f} | {c['ap50']:.3f} | {c['n']:,} |"
        for lab, c in sorted(m["per_class"].items(), key=lambda x: -x[1]["ap50"])
    )
    labels = ", ".join(f"`{lab}`" for lab in m["per_class"])
    return f"""---
license: apache-2.0
library_name: transformers
pipeline_tag: object-detection
base_model: {base}
datasets:
- {DATASET}
tags:
- object-detection
- rt-detr
- car-damage
- insurance
model-index:
- name: claimsight-damage-rtdetr
  results:
  - task:
      type: object-detection
    dataset:
      name: Humans in the Loop car damages (CC0), held-out test split
      type: {DATASET}
      split: test
    metrics:
    - type: mean_average_precision
      name: mAP@[.50:.95]
      value: {m["map"]}
    - type: mean_average_precision
      name: mAP@.50
      value: {m["map_50"]}
---

# ClaimSight damage detector (RT-DETRv2 r18)

Finds car damage in photos: {labels}. It is the damage detector of
[ClaimSight]({SOURCE}), an auto-insurance claims triage project, where its boxes are
matched to car parts and fed to the claim's estimate.

**It is weak** (test mAP {m["map"]:.3f}): good enough to show the pipeline working on real
photos, not to price a repair. Treat its output as a signal for a human, not a measurement.

## Use

```python
from transformers import pipeline

detect = pipeline("object-detection", model="{repo}")
detect("car.jpg", threshold=0.5)
# [{{"score": 0.65, "label": "dent", "box": {{"xmin": 424, "ymin": 168, ...}}}}, ...]
```

0.5 is the threshold ClaimSight uses: on the test photos it returns about 4 boxes a
photo, 36% of them right (IoU 0.5, right class), with at least one right box on 72 of 94
photos. At 0.3 it returns about 26 a photo at 13% precision.

## Training

- Base: [`{base}`](https://huggingface.co/{base}) (Apache-2.0), fine-tuned end to end.
- Data: Humans in the Loop's "Car Parts and Car Damages", damage half: 814 photos, 8
  classes, about 11 boxes a photo, released under CC0 1.0 (the copy at
  [`{DATASET}`](https://huggingface.co/datasets/{DATASET})). Polygons were turned into
  their bounding boxes.
- Split 80/10/10 by a hash of the file name: 651 train, 69 val, 94 test.
- {size} x {size} input, 30 epochs, AdamW with one-cycle (lr 1e-4, backbone 1e-5), best
  validation checkpoint kept (epoch 17). About 100 minutes on an Apple M4.
- Script: `scripts/train_detector.py` in the source repo.

CarDD's labels fit insurance work better, but its licence needs the lab's signed consent
and forbids redistribution, so these CC0 photos were used instead.

## Evaluation

COCO-style AP on the {m["images"]} held-out test photos ({m["boxes"]:,} boxes), top 100
detections a photo. The photos were shrunk to at most 800 px a side (JPEG, quality 95)
before the model's 640 px resize, as in training; on full-size photos the scores shift by
a couple of points.

| class | AP | AP50 | test boxes |
|---|---|---|---|
| **all (mAP)** | **{m["map"]:.3f}** | **{m["map_50"]:.3f}** | {m["boxes"]:,} |
{rows}

## Limitations

- Big, distinct damage (missing and broken parts) is learnable; thin or tiny damage
  (scratches, paint chips, flaking) is not at this input size: a polygon's box is mostly
  undamaged paint, and 640 px shrinks a chip to a few pixels.
- Validation mAP plateaued at about 0.06 from epoch 11. More data, tiling or a larger
  input are the likely fixes, not more epochs.
- 814 photos, mostly close-ups; it has not been tested on claim photos taken by drivers.
- Labels describe the damage, not the part; ClaimSight finds the part with a separate
  zero-shot model (OWLv2).

`model.safetensors` sha256: `{digest}`
"""


def check(out: Path, cassette: dict, truth: list[dict]) -> str:
    """Run the assembled weights the way the live detector does; compare with the cassette."""
    try:
        from transformers import pipeline

        # The photos exactly as the cassette's run saw them (800 px JPEG copies), so the
        # scores match to rounding, not just roughly.
        from scripts.train_detector import load_image
    except ImportError:
        return "skipped: needs the `ml` extra (uv run --extra ml ...)"
    images = [i for i in cassette["images"] if (PHOTOS / i["file"]).exists()][:CHECK_PHOTOS]
    if not images:
        return f"skipped: no test photos in {PHOTOS}"
    detect = pipeline("object-detection", model=str(out), device="cpu")
    width = {t["file"]: t["width"] for t in truth}
    compared = 0
    for rec in images:
        image = load_image(rec)
        # The pipeline answers in the copy's pixels (whole pixels); the cassette in the
        # original photo's.
        scale = width[rec["file"]] / image.width
        tol = BOX_TOL * scale
        got = [
            [
                d["label"],
                d["score"],
                *(d["box"][k] * scale for k in ("xmin", "ymin", "xmax", "ymax")),
            ]
            for d in detect(image, threshold=CHECK_MIN_SCORE)
        ]
        want = [b for b in rec["boxes"] if b[1] >= CHECK_MIN_SCORE + SCORE_TOL]
        for w in want:
            ok = any(
                g[0] == w[0]
                and abs(g[1] - w[1]) <= SCORE_TOL
                and all(abs(a - b) <= tol for a, b in zip(g[2:], w[2:], strict=True))
                for g in got
            )
            if not ok:
                raise SystemExit(f"{rec['file']}: recorded box {w} not reproduced; got {got}")
            compared += 1
    return f"ok: {compared} recorded boxes on {len(images)} test photos reproduced"


def build(out: Path, model_dir: Path = MODEL_DIR, repo: str = PLACEHOLDER) -> dict:
    if out.exists():
        if {p.name for p in out.iterdir()} - {*FILES, "README.md"}:
            raise SystemExit(f"{out} exists and wasn't made by this script; not deleting it.")
        shutil.rmtree(out)
    missing = [f for f in FILES if not (model_dir / f).exists()]
    if missing:
        raise SystemExit(f"{model_dir} lacks {missing}; train it first (make train-detector).")
    preds, truth, cassette = load()
    metrics = detection_map(preds, truth)
    out.mkdir(parents=True)
    for f in FILES:
        shutil.copy2(model_dir / f, out / f)
    digest = sha256(out / "model.safetensors")
    (out / "README.md").write_text(card(metrics, cassette["base"], cassette["size"], digest, repo))
    return {"metrics": metrics, "sha256": digest, "cassette": cassette, "truth": truth}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=ROOT / "build" / "detector")
    ap.add_argument("--repo", default=PLACEHOLDER, help="the Hub repo the card's example loads")
    ap.add_argument("--no-check", action="store_true", help="skip running the weights")
    args = ap.parse_args()
    built = build(args.out, repo=args.repo)
    size = sum(p.stat().st_size for p in args.out.iterdir())
    m = built["metrics"]
    print(f"{args.out}: {size / 1e6:.1f} MB, test mAP {m['map']:.3f} / AP50 {m['map_50']:.3f}")
    print(f"model.safetensors sha256 {built['sha256']}")
    if not args.no_check:
        print("check:", check(args.out, built["cassette"], built["truth"]))


if __name__ == "__main__":
    main()
