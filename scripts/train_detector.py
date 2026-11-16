"""Fine-tune RT-DETRv2 (r18) on the CC0 Humans in the Loop car-damage photos.

    uv run --extra train python scripts/train_detector.py                 # train + test
    uv run --extra train python scripts/train_detector.py --epochs 1 --limit 16   # smoke

Data: see claimsight/models/damage_data.py (download notes there). Starts from
PekingU/rtdetr_v2_r18vd (COCO-pretrained, Apache-2.0) with a new 8-class head. Keeps the
checkpoint with the best val mAP, then scores the held-out test split once and writes:

* models/rtdetr-hitl-damage/          the checkpoint (gitignored; push to the Hub separately)
* evals/golden/detector/test.json     ground-truth boxes for the test split (CC0 labels)
* evals/golden/cassettes/detector.json  the model's raw test predictions, for replay
* evals/reports/detector_train.json   per-epoch loss and val mAP
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

import torch  # noqa: E402
from PIL import Image, ImageOps  # noqa: E402

from claimsight.models.damage_data import DAMAGE_DIR, LABELS, load_split  # noqa: E402

BASE = "PekingU/rtdetr_v2_r18vd"
OUT = ROOT / "models" / "rtdetr-hitl-damage"
GOLDEN = ROOT / "evals" / "golden"
SIZE = 640


def device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    return "mps" if torch.backends.mps.is_available() else "cpu"


CACHE = DAMAGE_DIR.parent / "cache-800"


def load_image(rec: dict) -> Image.Image:
    """The photo shrunk to at most 800 px a side (cached as JPEG): the model sees 640 px, and
    decoding the full-size PNGs every epoch was the bottleneck."""
    path = CACHE / (rec["file"] + ".jpg")
    if not path.exists():
        CACHE.mkdir(exist_ok=True)
        with Image.open(DAMAGE_DIR / "img" / rec["file"]) as im:
            img = im.convert("RGB")
        img.thumbnail((800, 800))
        img.save(path, quality=95)
    with Image.open(path) as im:
        return im.convert("RGB")


def scaled(rec: dict, img: Image.Image) -> dict:
    """Boxes in the cached image's pixels (labels are in the original's)."""
    f = img.width / rec["width"]
    boxes = [(lab, *(v * f for v in b)) for lab, *b in rec["boxes"]]
    return {**rec, "width": img.width, "height": img.height, "boxes": boxes}


def coco_target(rec: dict, image_id: int, flip: bool = False) -> dict:
    anns = []
    for label, x0, y0, x1, y1 in rec["boxes"]:
        if flip:
            x0, x1 = rec["width"] - x1, rec["width"] - x0
        w, h = x1 - x0, y1 - y0
        anns.append(
            {
                "bbox": [x0, y0, w, h],
                "category_id": LABELS.index(label),
                "area": w * h,
                "iscrowd": 0,
            }
        )
    return {"image_id": image_id, "annotations": anns}


def batches(records, processor, batch_size, train):
    order = list(range(len(records)))
    if train:
        random.shuffle(order)
    for i in range(0, len(order), batch_size):
        idx = order[i : i + batch_size]
        images, targets = [], []
        for j in idx:
            rec = records[j]
            img = load_image(rec)
            rec = scaled(rec, img)
            flip = train and random.random() < 0.5
            if flip:
                img = ImageOps.mirror(img)
            images.append(img)
            targets.append(coco_target(rec, j, flip))
        yield idx, processor(images=images, annotations=targets, return_tensors="pt")


@torch.no_grad()
def predict(model, processor, records, dev, batch_size=8, threshold=0.0):
    model.eval()
    preds = []
    for i in range(0, len(records), batch_size):
        chunk = records[i : i + batch_size]
        images = [load_image(r) for r in chunk]
        inputs = processor(images=images, return_tensors="pt").to(dev)
        out = model(**inputs)
        # Boxes come back in the original photo's pixels, like the labels.
        sizes = torch.tensor([[r["height"], r["width"]] for r in chunk])
        res = processor.post_process_object_detection(out, target_sizes=sizes, threshold=threshold)
        for rec, r in zip(chunk, res, strict=True):
            preds.append(
                {
                    "file": rec["file"],
                    "boxes": [
                        [LABELS[int(lab)], round(float(s), 4), *[round(float(v), 1) for v in b]]
                        for s, lab, b in zip(r["scores"], r["labels"], r["boxes"], strict=True)
                    ],
                }
            )
    return preds


def score(preds: list[dict], truth: list[dict]) -> dict:
    from evals.detector_eval import detection_map

    return detection_map(preds, truth)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--limit", type=int, default=0, help="use only N training images (smoke)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from transformers import RTDetrImageProcessor, RTDetrV2ForObjectDetection

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    split = load_split()
    train, val, test = split["train"], split["val"], split["test"]
    if args.limit:
        train, val, test = train[: args.limit], val[: args.limit // 2], test[: args.limit // 2]
    print(f"train {len(train)}  val {len(val)}  test {len(test)} images")

    dev = device()
    processor = RTDetrImageProcessor.from_pretrained(BASE, size={"height": SIZE, "width": SIZE})
    model = RTDetrV2ForObjectDetection.from_pretrained(
        BASE,
        id2label=dict(enumerate(LABELS)),
        label2id={lab: i for i, lab in enumerate(LABELS)},
        ignore_mismatched_sizes=True,
    ).to(dev)

    backbone = [p for n, p in model.named_parameters() if "backbone" in n]
    rest = [p for n, p in model.named_parameters() if "backbone" not in n]
    opt = torch.optim.AdamW(
        [{"params": backbone, "lr": args.lr / 10}, {"params": rest, "lr": args.lr}],
        weight_decay=1e-4,
    )
    steps = args.epochs * -(-len(train) // args.batch_size)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=[args.lr / 10, args.lr], total_steps=steps, pct_start=0.05
    )

    history, best = [], -1.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0, total, n = time.time(), 0.0, 0
        for _, batch in batches(train, processor, args.batch_size, train=True):
            labels = [{k: v.to(dev) for k, v in t.items()} for t in batch["labels"]]
            out = model(pixel_values=batch["pixel_values"].to(dev), labels=labels)
            opt.zero_grad()
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
            opt.step()
            sched.step()
            total, n = total + out.loss.item(), n + 1
            del out, labels, batch
        if dev == "mps":  # the MPS allocator otherwise keeps every batch's buffers
            torch.mps.empty_cache()
        val_map = score(predict(model, processor, val, dev), val)
        row = {"epoch": epoch, "loss": round(total / n, 4), "secs": round(time.time() - t0)}
        row |= {"val_map": val_map["map"], "val_map50": val_map["map_50"]}
        if dev == "mps":
            row["gpu_gb"] = round(torch.mps.driver_allocated_memory() / 1e9, 1)
        history.append(row)
        print(json.dumps(row), flush=True)
        if val_map["map"] > best:
            best = val_map["map"]
            model.save_pretrained(OUT)
            processor.save_pretrained(OUT)

    model = RTDetrV2ForObjectDetection.from_pretrained(OUT).to(dev)
    preds = predict(model, processor, test, dev)
    test_map = score(preds, test)
    print("test", json.dumps({k: v for k, v in test_map.items() if k != "per_class"}))

    if not args.limit:
        (GOLDEN / "detector").mkdir(parents=True, exist_ok=True)
        (GOLDEN / "detector" / "test.json").write_text(
            json.dumps({"labels": LABELS, "images": test}, indent=0) + "\n"
        )
        # COCO scores at most 100 detections per image; keep only those.
        top = [{**p, "boxes": sorted(p["boxes"], key=lambda b: -b[1])[:100]} for p in preds]
        cassette = {"model": "rtdetr-hitl-damage", "base": BASE, "size": SIZE, "images": top}
        (GOLDEN / "cassettes" / "detector.json").write_text(
            json.dumps(cassette, separators=(",", ":"), indent=0) + "\n"
        )
    report = {"history": history, "best_val_map": best, "test": test_map}
    (ROOT / "evals" / "reports").mkdir(exist_ok=True)
    (ROOT / "evals" / "reports" / "detector_train.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
