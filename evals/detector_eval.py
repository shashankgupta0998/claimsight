"""Damage detector eval: COCO box mAP on the held-out test split, from the cassette.

    uv run python -m evals.detector_eval

`scripts/train_detector.py` writes the ground truth (evals/golden/detector/test.json, CC0
labels) and the fine-tuned model's raw predictions (evals/golden/cassettes/detector.json).
This scores them in pure Python, so CI checks the numbers without torch or the images.

mAP follows COCO: IoU thresholds 0.50:0.05:0.95, 101-point interpolated precision, up to
100 detections per image, averaged over classes that appear in the ground truth.
`tests/test_detector_eval.py` checks it against hand-worked cases; on 40 random images it
agreed with torchmetrics/pycocotools to within 0.002 mAP.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

GOLDEN = Path(__file__).parent / "golden"
IOUS = [0.5 + 0.05 * i for i in range(10)]
RECALLS = [i / 100 for i in range(101)]
MAX_DETS = 100


def iou(a: list[float], b: list[float]) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def average_precision(dets: list[tuple[float, bool]], n_truth: int) -> float:
    """dets: (score, is_true_positive), any order. COCO 101-point interpolation."""
    if n_truth == 0:
        return float("nan")
    dets = sorted(dets, key=lambda d: -d[0])
    tp = fp = 0
    prec, rec = [], []
    for _, hit in dets:
        tp, fp = tp + hit, fp + (not hit)
        prec.append(tp / (tp + fp))
        rec.append(tp / n_truth)
    for i in range(len(prec) - 2, -1, -1):  # make precision monotone non-increasing
        prec[i] = max(prec[i], prec[i + 1])
    total, j = 0.0, 0
    for r in RECALLS:
        while j < len(rec) and rec[j] < r:
            j += 1
        total += prec[j] if j < len(rec) else 0.0
    return total / len(RECALLS)


def match(preds: list, truth: list, thr: float) -> list[tuple[float, bool]]:
    """Greedy COCO matching for one image and class: best score first, best unused IoU."""
    used, out = set(), []
    for p in sorted(preds, key=lambda p: -p[0]):
        best, best_i = thr, None
        for i, t in enumerate(truth):
            if i not in used and (o := iou(p[1], t)) >= best:
                best, best_i = o, i
        if best_i is not None:
            used.add(best_i)
        out.append((p[0], best_i is not None))
    return out


def detection_map(preds: list[dict], truth: list[dict]) -> dict:
    """preds: [{file, boxes: [[label, score, x0, y0, x1, y1]]}]; truth: [{file, boxes:
    [[label, x0, y0, x1, y1]]}]. Returns map, map_50, map_75 and per-class AP."""
    by_file = {p["file"]: p["boxes"] for p in preds}
    labels = sorted({b[0] for t in truth for b in t["boxes"]})
    per_class: dict[str, dict] = {}
    for lab in labels:
        aps = []
        n_truth = sum(1 for t in truth for b in t["boxes"] if b[0] == lab)
        for thr in IOUS:
            dets = []
            for t in truth:
                gt = [list(b[1:]) for b in t["boxes"] if b[0] == lab]
                top = sorted(by_file.get(t["file"], []), key=lambda b: -b[1])[:MAX_DETS]
                pd = [(b[1], list(b[2:])) for b in top if b[0] == lab]
                dets += match(pd, gt, thr)
            aps.append(average_precision(dets, n_truth))
        per_class[lab] = {"n": n_truth, "ap": round(sum(aps) / len(aps), 4), "ap50": aps[0]}
        per_class[lab]["ap50"] = round(per_class[lab]["ap50"], 4)
        per_class[lab]["ap75"] = round(aps[5], 4)

    def mean(key: str) -> float:
        vals = [c[key] for c in per_class.values()]
        return round(sum(vals) / len(vals), 4) if vals else 0.0

    return {
        "map": mean("ap"),
        "map_50": mean("ap50"),
        "map_75": mean("ap75"),
        "images": len(truth),
        "boxes": sum(len(t["boxes"]) for t in truth),
        "per_class": per_class,
    }


def load(golden: Path = GOLDEN) -> tuple[list[dict], list[dict], dict]:
    truth = json.loads((golden / "detector" / "test.json").read_text())["images"]
    cassette = json.loads((golden / "cassettes" / "detector.json").read_text())
    return cassette["images"], truth, cassette


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("evals/reports/detector.json"))
    args = ap.parse_args(argv)
    preds, truth, cassette = load()
    m = detection_map(preds, truth)
    print(f"Damage detector: {cassette['model']} (from {cassette['base']}), test split")
    print(f"  {m['images']} images, {m['boxes']} boxes")
    print(f"  mAP {m['map']:.3f}   mAP@50 {m['map_50']:.3f}   mAP@75 {m['map_75']:.3f}\n")
    print(f"  {'class':<14} {'boxes':>5} {'AP':>6} {'AP50':>6}")
    for lab, c in sorted(m["per_class"].items(), key=lambda x: -x[1]["n"]):
        print(f"  {lab:<14} {c['n']:>5} {c['ap']:>6.3f} {c['ap50']:>6.3f}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(m, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
