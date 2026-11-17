"""Assemble the Hugging Face Space: the free static demo, or the Docker one.

    make space              # static: build/space/  (free)
    make space-docker       # Docker: build/space/  (needs Hugging Face PRO to run)
    hf upload <you>/claimsight build/space . --repo-type space

Static (the default): Hugging Face hosts static Spaces free; Docker and Gradio Spaces need a
PRO subscription. Replay mode is deterministic, so this runs the real API once per sample
claim (and, for claims that wait for an adjuster, once per adjuster choice) and saves each
answer as JSON next to the demo page in static mode. The page is the same file the API
serves at `/` in demo mode (api/static/demo.html), so the two can't drift; a test checks
the saved answers against the API.

Docker: the deploy image with the demo page on and `space/README.docker.md`. Only files git
tracks or would track (not ignored) are copied, so nothing local (.env, data/, models/,
reports) ends up in the Space.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MARKER = ".claimsight-space"
PAGE = ROOT / "src" / "claimsight" / "api" / "static" / "demo.html"
# What the Dockerfile copies, and the Dockerfile itself.
KEEP = ("Dockerfile", ".dockerignore", "pyproject.toml", "uv.lock", "src/", "evals/", "infra/")
SKIP = ("evals/reports/",)
DEMO_ENV = (
    "\n# Hugging Face Space (scripts/build_space.py): the demo page at /, claims capped.\n"
    "ENV CLAIMSIGHT_DEMO=true\n"
)
DECISIONS = ("approve", "deny")


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


def fresh(out: Path) -> None:
    if out.exists():
        if not (out / MARKER).exists():
            raise SystemExit(f"{out} exists and wasn't made by this script; not deleting it.")
        shutil.rmtree(out)
    out.mkdir(parents=True)


def build_docker(out: Path) -> list[str]:
    fresh(out)
    files = tracked_files()
    for f in files:
        (out / f).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / f, out / f)
    shutil.copy2(ROOT / "space" / "README.docker.md", out / "README.md")
    dockerfile = out / "Dockerfile"
    text = dockerfile.read_text()
    # Before CMD, so the setting is part of the final image.
    head, sep, cmd = text.rpartition("\nCMD ")
    dockerfile.write_text(head + "\n" + DEMO_ENV + sep + cmd)
    (out / MARKER).write_text("Made by scripts/build_space.py\n")
    return files + ["README.md"]


def demo_client():
    """A fresh API in demo mode, replay, in-memory state."""
    sys.path[:0] = [str(ROOT / "src")]
    from fastapi.testclient import TestClient

    from claimsight.api.main import create_app
    from claimsight.config import Settings
    from claimsight.deps import build_replay_deps

    settings = Settings(_env_file=None, mode="replay", demo=True, checkpointer="memory")
    return TestClient(create_app(build_replay_deps(settings)))


def answers() -> dict[str, dict | list]:
    """Every answer the static page needs, from the real API: name -> JSON body."""
    samples = demo_client().get("/demo/samples").json()
    saved: dict[str, dict | list] = {"samples": samples}
    for s in samples:
        client = demo_client()
        filed = client.post("/claims", json=s["claim"])
        filed.raise_for_status()
        saved[s["id"]] = filed.json()
        if filed.json()["status"] != "awaiting_review":
            continue
        for decision in DECISIONS:
            client = demo_client()  # each choice from the same starting point
            client.post("/claims", json=s["claim"]).raise_for_status()
            body = {"decision": decision, "reviewer": "demo-adjuster", "note": None}
            reviewed = client.post(f"/claims/{s['id']}/review", json=body)
            reviewed.raise_for_status()
            saved[f"{s['id']}.{decision}"] = reviewed.json()
    return saved


def build_static(out: Path) -> list[str]:
    fresh(out)
    page = PAGE.read_text()
    assert page.count('<html lang="en">') == 1
    (out / "index.html").write_text(
        page.replace('<html lang="en">', '<html lang="en" data-mode="static">')
    )
    shutil.copy2(ROOT / "space" / "README.static.md", out / "README.md")
    files = ["index.html", "README.md"]
    (out / "data").mkdir()
    for name, body in answers().items():
        path = f"data/{name}.json"
        (out / path).write_text(json.dumps(body, indent=1, ensure_ascii=False) + "\n")
        files.append(path)
    (out / MARKER).write_text("Made by scripts/build_space.py\n")
    return files


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=ROOT / "build" / "space")
    ap.add_argument("--kind", choices=("static", "docker"), default="static")
    args = ap.parse_args()
    files = (build_static if args.kind == "static" else build_docker)(args.out)
    size = sum((args.out / f).stat().st_size for f in files)
    print(f"{args.kind}: {len(files)} files, {size / 1e6:.2f} MB -> {args.out}")


if __name__ == "__main__":
    main()
