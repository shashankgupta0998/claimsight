"""Draw the specimen policy as page images for the ColQwen2 page lane.

    uv run python scripts/render_pages.py

Writes evals/golden/pages/page-NN.png, one per page that holds a clause. Re-run after
changing policy_clauses.jsonl or the layout in retrieval/pages.py, then re-record the page
scores (scripts/record_retrieval.py --pages): the cassette is tied to these exact pixels.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src")]

from claimsight.retrieval.factory import load_chunks  # noqa: E402
from claimsight.retrieval.pages import fingerprint, render_pages  # noqa: E402

GOLDEN = ROOT / "evals" / "golden"


def main() -> None:
    out = GOLDEN / "pages"
    out.mkdir(exist_ok=True)
    for old in out.glob("page-*.png"):
        old.unlink()
    images = render_pages(load_chunks(GOLDEN))
    for page, img in images.items():
        img.save(out / f"page-{page:02d}.png", optimize=True)
    print(f"Wrote {len(images)} pages to {out.relative_to(ROOT)} (fingerprint {fingerprint(out)})")


if __name__ == "__main__":
    main()
