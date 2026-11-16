"""Reused-photo search: dHash vs DINOv2 on edited copies of real damage photos.

    uv run python -m evals.duplicate_eval                       # replay
    uv run --extra ml python -m evals.duplicate_eval --record   # needs the photos in data/

The golden claims carry only hashes, so this uses the 814 CC0 damage photos
(models/damage_data.py). The index of "earlier claims" holds the 651 training photos plus
a hash-chosen half of the val and test photos. Every val and test photo is then edited
seven ways, the way a reused photo turns up again:

    recompress   60% size, JPEG quality 60
    crop         centre 70%
    mirror       left-right flip
    rotate       7 degrees, cropped to hide the corners
    color        brighter, flatter, less saturated
    screen       photo of a screen: bezel, keystone, blur, noise, JPEG
    crop_mirror  off-centre 60% crop, flipped

An edit is a positive when its original is in the index (it must match *that* photo) and a
negative when it isn't (any match is a false alarm). Other photos of damaged cars, some of
the same car from another angle, are the hard negatives.

dHash keeps its production threshold (Hamming <= 6). DINOv2's cosine threshold is chosen
on the val queries (best recall with at most 1% false alarms) and reported on test.

--record writes cassettes/duplicates.json with each query's best match under both
methods; replay scores from it without the photos or the model.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import random
from pathlib import Path

GOLDEN = Path(__file__).parent / "golden"
CASSETTE = GOLDEN / "cassettes" / "duplicates.json"
TRANSFORMS = ["recompress", "crop", "mirror", "rotate", "color", "screen", "crop_mirror"]
DHASH_MAX = 6  # Settings.duplicate_max_hamming
MAX_FPR = 0.01


def seen(name: str) -> bool:
    """Half of the val/test photos are in the index, chosen by a hash of the file name."""
    return int(hashlib.sha256(("seen:" + name).encode()).hexdigest(), 16) % 2 == 0


def edit(img, how: str, seed: str):
    """Deterministic edit of a PIL image."""
    from PIL import Image, ImageEnhance, ImageFilter, ImageOps

    rng = random.Random(seed)
    w, h = img.size

    def crop(frac: float, dx: float = 0.0, dy: float = 0.0):
        cw, ch = int(w * frac), int(h * frac)
        x0 = int((w - cw) * (0.5 + dx)) if dx else (w - cw) // 2
        y0 = int((h - ch) * (0.5 + dy)) if dy else (h - ch) // 2
        return img.crop((x0, y0, x0 + cw, y0 + ch))

    def jpeg(im, q: int):
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=q)
        return Image.open(io.BytesIO(buf.getvalue())).convert("RGB")

    if how == "recompress":
        return jpeg(img.resize((int(w * 0.6), int(h * 0.6))), 60)
    if how == "crop":
        return crop(0.7)
    if how == "mirror":
        return ImageOps.mirror(img)
    if how == "rotate":
        r = img.rotate(7, resample=Image.Resampling.BICUBIC)
        return r.crop((int(w * 0.075), int(h * 0.075), int(w * 0.925), int(h * 0.925)))
    if how == "color":
        out = ImageEnhance.Brightness(img).enhance(1.3)
        out = ImageEnhance.Contrast(out).enhance(0.8)
        return ImageEnhance.Color(out).enhance(0.7)
    if how == "screen":
        bezel = Image.new("RGB", (int(w * 1.12), int(h * 1.12)), (18, 18, 20))
        bezel.paste(img, (int(w * 0.06), int(h * 0.06)))
        bw, bh = bezel.size
        k = 0.06
        # Source corners UL, LL, LR, UR: a keystone, as if shot from a little to the right.
        quad = (0, bh * k, 0, bh * (1 - k), bw, bh, bw, 0)
        out = bezel.transform((bw, bh), Image.Transform.QUAD, quad, Image.Resampling.BICUBIC)
        out = out.filter(ImageFilter.GaussianBlur(1.2))
        # Seeded sensor noise: grey 128, sigma 12 (what Image.effect_noise makes, unseeded).
        px = bytes(min(255, max(0, int(rng.gauss(128, 12)))) for _ in range(bw * bh))
        noise = Image.frombytes("L", (bw, bh), px).convert("RGB")
        out = Image.blend(out, noise, 0.06)
        return jpeg(out, 70)
    if how == "crop_mirror":
        return ImageOps.mirror(crop(0.6, dx=rng.uniform(-0.4, 0.4), dy=rng.uniform(-0.4, 0.4)))
    raise ValueError(how)


def record(out: Path = CASSETTE) -> dict:
    from PIL import Image, ImageOps

    from claimsight.models.damage_data import DAMAGE_DIR, load_split
    from claimsight.tools.photo_embed import DINO_MODEL, DinoEmbedder
    from claimsight.tools.photos import dhash, hamming

    def dhash_image(im) -> str:
        buf = io.BytesIO()
        im.save(buf, "PNG")
        return dhash(io.BytesIO(buf.getvalue()))

    def load(name: str):
        with Image.open(DAMAGE_DIR / "img" / name) as im:
            img = im.convert("RGB")
        img.thumbnail((1024, 1024))
        return img

    split = load_split()
    index_names = [r["file"] for r in split["train"]]
    index_names += [r["file"] for s in ("val", "test") for r in split[s] if seen(r["file"])]
    model = DinoEmbedder(DINO_MODEL)

    def embed_all(images: list) -> list:
        vecs = []
        for i in range(0, len(images), 16):
            vecs += model.embed_images(images[i : i + 16])
        return vecs

    print(f"Indexing {len(index_names)} photos")
    index_imgs = [load(n) for n in index_names]
    index_vecs = embed_all(index_imgs)
    index_hash = [dhash_image(im) for im in index_imgs]
    del index_imgs

    queries = []
    for s in ("val", "test"):
        for r in split[s]:
            base = load(r["file"])
            edits = [edit(base, t, r["file"] + t) for t in TRANSFORMS]
            vecs = embed_all(edits)
            flipped = embed_all([ImageOps.mirror(im) for im in edits])
            for t, im, v, fv in zip(TRANSFORMS, edits, vecs, flipped, strict=True):
                h = dhash_image(im)
                d_i = min(range(len(index_names)), key=lambda i: hamming(h, index_hash[i]))
                sims = [sum(a * b for a, b in zip(v, iv, strict=True)) for iv in index_vecs]
                e_i = max(range(len(index_names)), key=sims.__getitem__)
                # Mirror-augmented query: the better of the photo and its mirror image.
                fsims = [
                    max(x, sum(a * b for a, b in zip(fv, iv, strict=True)))
                    for x, iv in zip(sims, index_vecs, strict=True)
                ]
                f_i = max(range(len(index_names)), key=fsims.__getitem__)
                queries.append(
                    {
                        "split": s,
                        "photo": r["file"],
                        "transform": t,
                        "seen": seen(r["file"]),
                        "dhash": {
                            "match": index_names[d_i],
                            "distance": hamming(h, index_hash[d_i]),
                        },
                        "dinov2": {"match": index_names[e_i], "cosine": round(sims[e_i], 4)},
                        "dinov2_flip": {
                            "match": index_names[f_i],
                            "cosine": round(fsims[f_i], 4),
                        },
                    }
                )
        print(f"Queried {s}: {sum(q['split'] == s for q in queries)} edits")
    cassette = {"model": DINO_MODEL, "index_size": len(index_names), "queries": queries}
    out.write_text(json.dumps(cassette, indent=0) + "\n")
    return cassette


def hit(q: dict, method: str, threshold: float) -> tuple[bool, bool]:
    """(matched at this threshold, matched the right photo)."""
    m = q[method]
    ok = m["distance"] <= threshold if method == "dhash" else m["cosine"] >= threshold
    return ok, ok and m["match"] == q["photo"]


def rates(queries: list[dict], method: str, threshold: float) -> dict:
    pos = [q for q in queries if q["seen"]]
    neg = [q for q in queries if not q["seen"]]
    return {
        "recall": round(sum(hit(q, method, threshold)[1] for q in pos) / len(pos), 4),
        "fpr": round(sum(hit(q, method, threshold)[0] for q in neg) / len(neg), 4),
    }


def calibrate(queries: list[dict], method: str, max_fpr: float = MAX_FPR) -> float:
    """Lowest cosine threshold whose false-alarm rate on these queries is <= max_fpr."""
    neg = sorted((q[method]["cosine"] for q in queries if not q["seen"]), reverse=True)
    allowed = int(max_fpr * len(neg))  # this many negatives may score at or above it
    return round(neg[allowed] + 1e-4, 4) if allowed < len(neg) else 0.0


METHODS = ("dhash", "dinov2", "dinov2_flip")
# Unseen test photos with a near-identical twin in the index: same car, same spot, taken
# seconds apart (checked by eye after the first run). Every false alarm on test is one of
# these; the metrics are reported with and without them.
TWINS = {"Car damages 1180.png", "Car damages 1286.png"}


def score(cassette: dict) -> dict:
    qs = cassette["queries"]
    val = [q for q in qs if q["split"] == "val"]
    test = [q for q in qs if q["split"] == "test"]
    no_twins = [q for q in test if q["photo"] not in TWINS]
    thr = {"dhash": DHASH_MAX} | {m: calibrate(val, m) for m in METHODS[1:]}
    out = {
        "thresholds": thr,
        "test": {m: rates(test, m, thr[m]) for m in METHODS},
        "test_without_twins": {m: rates(no_twins, m, thr[m]) for m in METHODS},
        "val_recall": {m: rates(val, m, thr[m])["recall"] for m in METHODS},
        "by_transform": {},
        "test_queries": len(test),
        "test_positives": sum(q["seen"] for q in test),
    }
    for t in TRANSFORMS:
        sub = [q for q in test if q["transform"] == t]
        out["by_transform"][t] = {m: rates(sub, m, thr[m])["recall"] for m in METHODS}
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--record", action="store_true", help="embed the photos and edits")
    ap.add_argument("--out", type=Path, default=Path("evals/reports/duplicates.json"))
    args = ap.parse_args(argv)
    cassette = record() if args.record else json.loads(CASSETTE.read_text())
    m = score(cassette)
    print(
        f"Reused-photo search: {cassette['index_size']} indexed photos, "
        f"{m['test_queries']} test edits ({m['test_positives']} of indexed photos)"
    )
    t = m["thresholds"]
    print(
        f"  cosine thresholds (val, <= {MAX_FPR:.0%} false alarms): "
        f"DINOv2 {t['dinov2']}, + mirrored query {t['dinov2_flip']}\n"
    )
    print(f"  {'test recall':<20} {'dHash <= 6':>10} {'DINOv2':>8} {'+ mirror':>9}")

    def row(label: str, vals: dict, fmt: str = ".2f") -> None:
        print(
            f"  {label:<20} "
            + " ".join(f"{vals[k]:{w}{fmt}}" for k, w in zip(METHODS, (10, 8, 9), strict=True))
        )

    for tr, r in m["by_transform"].items():
        row(tr, r)
    row("all", {k: v["recall"] for k, v in m["test"].items()})
    row("false alarms", {k: v["fpr"] for k, v in m["test"].items()}, ".3f")
    row("  without twins", {k: v["fpr"] for k, v in m["test_without_twins"].items()}, ".3f")
    row("val recall", m["val_recall"])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(m, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
