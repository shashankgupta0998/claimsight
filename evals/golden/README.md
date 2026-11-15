---
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

- `claims.jsonl`: 10 hand-written claims, one per behaviour the pipeline must get right.
- `variants.jsonl`: fraud variants of the hand-written approved claims.
- `synthetic.jsonl`: 240 scenario-generated claims (seed 7).

### Synthetic decisions

| value | count |
|---|---|
| approve | 142 |
| deny | 26 |
| escalate | 72 |

### Synthetic loss types

| value | count |
|---|---|
| animal | 25 |
| collision | 107 |
| flood | 26 |
| glass | 24 |
| hail | 40 |
| vandalism | 18 |

### Injected fraud signals

| value | count |
|---|---|
| duplicate_photos | 10 |
| exif_consistency | 11 |
| statement_consistency | 9 |
| vin_decode | 2 |
| weather_corroboration | 6 |

### Scenario tags

| value | count |
|---|---|
| animal | 25 |
| collision | 107 |
| deny | 26 |
| flood | 26 |
| fraud | 38 |
| fraud:duplicate_photo | 10 |
| fraud:exif_backdate | 7 |
| fraud:exif_location | 4 |
| fraud:statement_flip | 9 |
| fraud:vin_mismatch | 2 |
| fraud:weather | 6 |
| glass | 24 |
| hail | 40 |
| hard:hail_code_only | 2 |
| hard:near_duplicate | 8 |
| hard:photo_nearby_town | 9 |
| hard:photo_prev_evening | 8 |
| hard_negative | 27 |
| legit | 115 |
| low_evidence | 15 |
| major | 19 |
| no_exif | 17 |
| vandalism | 18 |
| zone:front | 43 |
| zone:rear | 51 |
| zone:side | 13 |

## How labels are made

Labels come from the generator's scenario, never from running the pipeline. Scenario
mix: legit 52%, fraud 15%, hard_negative 10%, deny 8%, major 8%, low_evidence 7%.

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
