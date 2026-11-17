"""Assemble the Hugging Face Space: the deploy image, in demo mode, with the Space's README.

    make space            # writes build/space/
    hf upload <you>/claimsight build/space . --repo-type space

A Docker Space builds the Dockerfile at its root and needs a README whose front matter says
`sdk: docker` and which port to open (space/README.md). Only files git tracks or would track
(not ignored) are copied, so nothing local (.env, data/, models/, reports) ends up in the Space.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# What the Dockerfile copies, and the Dockerfile itself.
KEEP = ("Dockerfile", ".dockerignore", "pyproject.toml", "uv.lock", "src/", "evals/", "infra/")
SKIP = ("evals/reports/",)
DEMO_ENV = (
    "\n# Hugging Face Space (scripts/build_space.py): the demo page at /, claims capped.\n"
    "ENV CLAIMSIGHT_DEMO=true\n"
)


def tracked_files() -> list[str]:
    out = subprocess.run(
        ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return [
        f
        for f in out.split("\0")
        if f and f.startswith(KEEP) and not f.startswith(SKIP) and (ROOT / f).is_file()
    ]


def build(out: Path) -> list[str]:
    if out.exists():
        if not (out / ".claimsight-space").exists():
            raise SystemExit(f"{out} exists and wasn't made by this script; not deleting it.")
        shutil.rmtree(out)
    files = tracked_files()
    for f in files:
        (out / f).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / f, out / f)
    shutil.copy2(ROOT / "space" / "README.md", out / "README.md")
    dockerfile = out / "Dockerfile"
    text = dockerfile.read_text()
    # Before CMD, so the setting is part of the final image.
    head, sep, cmd = text.rpartition("\nCMD ")
    dockerfile.write_text(head + "\n" + DEMO_ENV + sep + cmd)
    (out / ".claimsight-space").write_text("Made by scripts/build_space.py\n")
    return files + ["README.md"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=ROOT / "build" / "space")
    args = ap.parse_args()
    files = build(args.out)
    size = sum((args.out / f).stat().st_size for f in files)
    print(f"{len(files)} files, {size / 1e6:.1f} MB -> {args.out}")


if __name__ == "__main__":
    main()
