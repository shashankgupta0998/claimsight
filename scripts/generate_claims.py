"""Generate the synthetic part of the golden set.

    uv run python scripts/generate_claims.py                    # 240 claims, seed 7
    uv run python scripts/generate_claims.py --paraphrase       # + natural statements (Claude)
    uv run python scripts/generate_claims.py --tts 20           # + voice notes for 20 claims

Writes, next to the hand-curated files in evals/golden/:
    synthetic.jsonl                   labelled claims
    cassettes/weather.synthetic.json  recorded SPC / Open-Meteo responses
    cassettes/vin.synthetic.json      recorded vPIC responses
    prior_photos.synthetic.jsonl      earlier-claim photo hashes for the duplicate check
    synthetic.stats.json, README.md   counts and the dataset card

Same seed, same files: CI regenerates and fails on any diff.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from evals.synth import KIND_WEIGHTS, generate  # noqa: E402

GOLDEN = ROOT / "evals" / "golden"


def existing_hashes() -> list[str]:
    hashes = []
    for line in (GOLDEN / "prior_photos.jsonl").read_text().splitlines():
        if line.strip():
            hashes.append(json.loads(line)["phash"])
    for line in (GOLDEN / "claims.jsonl").read_text().splitlines():
        if line.strip():
            hashes += [p["phash"] for p in json.loads(line)["claim"]["photos"] if p.get("phash")]
    return hashes


def dataset_card(stats: dict, seed: int) -> str:
    def table(counts: dict) -> str:
        rows = "\n".join(f"| {k} | {v} |" for k, v in sorted(counts.items()))
        return "| value | count |\n|---|---|\n" + rows

    hand = sum(1 for x in (GOLDEN / "claims.jsonl").read_text().splitlines() if x.strip())
    return f"""---
license: cc-by-4.0
task_categories:
  - object-detection
  - automatic-speech-recognition
  - text-classification
language:
  - en
tags:
  - insurance
  - fraud-detection
  - rag-evaluation
  - synthetic
pretty_name: ClaimSight golden set
size_categories:
  - n<1K
---

# ClaimSight golden set

Labelled auto-insurance claims for evaluating a multimodal claims-triage pipeline:
damage detection, policy-clause retrieval and citation, fraud checks, and the final
approve / deny / escalate decision.

All data is synthetic. No real claimant, vehicle or policy is represented. VINs are
random strings with real manufacturer prefixes.

## Composition

- `claims.jsonl`: {hand} hand-written claims, one per behaviour the pipeline must get right.
- `variants.jsonl`: fraud variants of the hand-written approved claims.
- `synthetic.jsonl`: {stats["n_cases"]} scenario-generated claims (seed {seed}).

### Synthetic decisions

{table(stats["decisions"])}

### Synthetic loss types

{table(stats["loss_types"])}

### Injected fraud signals

{table(stats["fraud_flags"])}

### Scenario tags

{table(stats["tags"])}

## How labels are made

Labels come from the generator's scenario, never from running the pipeline. Scenario
mix: {", ".join(f"{k} {int(v * 100)}%" for k, v in KIND_WEIGHTS.items())}.

1. Any injected fraud signal → `escalate`
2. Coverage part missing from the policy → `deny`
3. Major (structural) loss over the auto-approval limit → `escalate`
4. Low-quality photo evidence → `escalate`
5. Otherwise → `approve`

The evidence is noisy on purpose: false-positive and missed detections, photos without
EXIF, statement phrasings outside the critic's cue list, and hard negatives (photo taken
the evening before, a similar but different photo, a photo from a nearby town, hail
logged by the weather station with no SPC report nearby).

## Fields

`claim` matches `claimsight.schemas.ClaimInput`. Each photo's `recorded_detections` is
the detector output replayed in offline evals. `labels` holds `decision`, `fraud_flags`,
`relevant_clause_ids`, `damage` (true `[part, damage_type]` pairs) and `tags`.

## Limitations

- Detector outputs are simulated, not produced by a model on real images. Pair the
  labels with CarDD images and recorded model outputs before quoting detection numbers.
- Statements come from templates unless generated with `--paraphrase`.
- The policy corpus is a short paraphrase of a personal auto policy's structure.
"""


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--n", type=int, default=240)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--paraphrase", action="store_true", help="Rewrite statements with Claude")
    ap.add_argument("--paraphrase-model", default="claude-haiku-5-5")
    ap.add_argument("--tts", type=int, default=0, metavar="N", help="Voice notes for the first N")
    ap.add_argument("--tts-model", default="facebook/mms-tts-eng")
    args = ap.parse_args()

    data = generate(args.n, seed=args.seed, existing_hashes=existing_hashes())

    if args.paraphrase:
        from evals.synth_media import paraphrase_statements

        calls = paraphrase_statements(
            data.cases, GOLDEN / "cache" / "paraphrases.json", model=args.paraphrase_model
        )
        print(f"Paraphrased statements ({calls} new model calls, rest from cache)")
    if args.tts:
        from evals.synth_media import synthesize_voice_notes

        done = synthesize_voice_notes(
            data.cases, ROOT / "data" / "voice", model=args.tts_model, limit=args.tts
        )
        print(f"Wrote {done} voice notes to data/voice/")

    def dump(path: Path, obj) -> None:
        path.write_text(json.dumps(obj, indent=1, sort_keys=True) + "\n")

    (GOLDEN / "synthetic.jsonl").write_text("".join(json.dumps(c) + "\n" for c in data.cases))
    dump(GOLDEN / "cassettes" / "weather.synthetic.json", data.weather)
    dump(GOLDEN / "cassettes" / "vin.synthetic.json", data.vins)
    (GOLDEN / "prior_photos.synthetic.jsonl").write_text(
        "".join(json.dumps(p) + "\n" for p in data.prior_photos)
    )
    stats = data.stats()
    dump(GOLDEN / "synthetic.stats.json", stats)
    (GOLDEN / "README.md").write_text(dataset_card(stats, args.seed))

    print(f"Wrote {stats['n_cases']} claims to {GOLDEN / 'synthetic.jsonl'}")
    print("decisions:", stats["decisions"])
    print("fraud flags:", stats["fraud_flags"])


if __name__ == "__main__":
    main()
