# ClaimSight

Multimodal, agent-based triage for auto-insurance claims. A claimant's photos, voice note
and documents go in. A decision packet comes out, in which every finding points to its
evidence: a model output, a cited policy page, or an outside API response. Claims the
system is unsure about pause for a human adjuster.

> Research prototype on synthetic and public data. Not an insurance decision tool.

## How it works

```
intake ─┬─ vision ── estimator ─┐
        ├─ coverage ────────────┼─ critic ─┬─ finalize
        └─ verification ────────┘          └─ human_review ─ finalize
```

| Node | Does | v0 (this scaffold) | Target |
|---|---|---|---|
| intake | Transcribes the voice note | Whisper wrapper | + PII redaction (GLiNER, Presidio) |
| vision | Finds damage, maps it to parts | Replay detector | RT-DETR on CarDD + OWLv2 parts |
| coverage | Retrieves and cites policy clauses | BM25 + clause metadata | pgvector hybrid + ColQwen2 + reranker + LLM |
| verification | Checks outside evidence | NHTSA vPIC, NOAA SPC, Open-Meteo, EXIF, photo hashes | + DINOv2 near-duplicates |
| estimator | Repair cost range | Parts table | LightGBM |
| critic | Cross-checks agents, decides | Rules + statement/damage consistency | + LLM consistency pass |
| human_review | Adjuster approval | LangGraph `interrupt` | Postgres checkpointer |

## Quickstart

```bash
make install        # uv sync --group dev
make test           # unit + graph + API tests, fully offline
make eval           # score the graph on the golden set
make gate           # fail if quality dropped vs evals/baseline.json
make api            # http://localhost:8000/docs
```

Optional local stack (Postgres + pgvector, Redis, MinIO, Langfuse): `make up`.

Extras: `uv sync --extra ml` (real models), `--extra rag` (pgvector retrieval),
`--extra llm` (Anthropic judge, Langfuse tracing).

## Replay vs live

Every model and API sits behind a protocol in `src/claimsight/interfaces.py` with two
implementations:

- **replay**: recorded detector outputs on each photo plus API cassettes in
  `evals/golden/cassettes/`. Free, deterministic, runs in CI.
- **live**: real models and APIs. Set `CLAIMSIGHT_MODE=live`.

The graph doesn't know which one it got. To grow the golden set, run new claims in live
mode once and record the responses as cassettes.

## The eval harness

Built before the agents, per the project plan.

- `evals/golden/claims.jsonl`: labelled claims (decision, fraud flags, relevant clauses,
  damage).
- `scripts/make_fraud_variants.py`: turns clean claims into labelled fraud cases
  (backdated EXIF, reused photo, contradictory statement).
- `scripts/generate_claims.py` (`make data`): 240 scenario-generated claims plus the API
  cassettes they need. Labels come from the scenario, never from running the pipeline.
  The evidence is noisy on purpose (false-positive and missed detections, missing EXIF,
  hard negatives). `--paraphrase` rewrites statements with Claude (cached);
  `--tts N` adds voice notes so live mode exercises ASR and reports WER. CI regenerates
  the set from its seed and fails on any diff. See the dataset card in
  `evals/golden/README.md`.
- `evals/runner.py`: decision accuracy, escalation rate, Recall@5, nDCG@5, citation
  precision, damage P/R, fraud P/R, ASR WER, latency p50/p95, and per-slice accuracy
  (loss type, scenario, fraud type, hard negative).
- `evals/gate.py`: CI blocks a merge when a gated metric drops more than 0.02.
- `evals/judge/`: LLM citation-support judge plus `calibrate.py`, which reports Cohen's
  kappa against your own labels. Don't publish judge scores without that number.

### Baseline (257 cases, replay)

| Metric | Before critic fix | Now |
|---|---|---|
| Decision accuracy | 0.98 | 1.00 |
| Fraud recall (precision) | 0.92 (1.00) | 1.00 (1.00) |
| Statement-flip fraud caught | 5 / 9 | 9 / 9 |
| Damage precision / recall | 0.98 / 0.98 | 0.98 / 0.98 |
| Clause Recall@5 / citation precision | 1.00 / 1.00 | 1.00 / 1.00 |

The statement check used to miss contradictions for two reasons. A keyword list didn't
know phrasings like "hit me from behind". And any detected part counted as damage, so one
faint false-positive fender made a front-impact story look plausible. The fix
(`agents/impact.py`, `agents/critic.py`):

- Weighted patterns that read who hit whom ("I rear-ended a Honda" is front damage,
  "a Honda rear-ended me" is rear). The classifier abstains rather than guess. On 35
  held-out phrasings written apart from the generator's templates
  (`tests/data/impact_statements.jsonl`) it scores **33 / 35, up from 8 / 35**, with no
  wrong guesses. The two misses abstain.
- An LLM fallback (routing model) for statements the rules abstain on. Answers are cached
  as a cassette (`cassettes/impact.llm.json`), so live runs record them and replay reads
  them back.
- Damage zones weighted by severity × detector confidence. The stated zone must hold at
  least 25% of the damage weight.

The synthetic set's 1.00 reflects its template statements. The held-out score and a
`--paraphrase` run are the honest tests of language coverage.

Detector outputs are simulated, so damage scores are an upper bound until real model
outputs are recorded on CarDD images.

## Data sources

- [NHTSA vPIC](https://vpic.nhtsa.dot.gov/api/): VIN decoding
- [NOAA SPC storm reports](https://www.spc.noaa.gov/climo/reports/): hail size and location
- [Open-Meteo archive](https://open-meteo.com/en/docs/historical-weather-api): daily weather
- CarDD: car damage detection dataset (detector training and eval)
- Policy text in `evals/golden/policy_clauses.jsonl` is paraphrased in the structure of a
  personal auto policy. Replace it with a state-filed specimen policy when you ingest real
  documents.

## Roadmap

- [x] Weeks 1–2: schema, graph, API, eval harness, golden set v0, CI gate
- [x] Weeks 1–2: synthetic claim generator (scenarios, LLM paraphrase, TTS voice notes) → 257 cases
- [ ] Weeks 3–4: ingest policies into pgvector, ColQwen2 page lane, reranker, LLM coverage reasoning
- [ ] Weeks 5–6: fine-tune RT-DETR on CarDD, push to the HF Hub, record detector cassettes
- [ ] Weeks 7–8: VLM statement/damage check, DINOv2 duplicate search, Postgres checkpointer
- [ ] Weeks 9–10: model routing, caching, prompt-injection red-team set, Langfuse dashboards
- [ ] Weeks 11–12: Redis queue, Locust load test, deploy (Modal + Fly.io), HF Space demo

## Layout

```
src/claimsight/
  agents/        vision, coverage, verification, estimator, critic
  graph/         LangGraph state and wiring
  models/        detector (damage + parts), ASR
  retrieval/     BM25 baseline, pgvector hybrid
  tools/         NHTSA, weather, photo forensics, geo
  api/           FastAPI app with human review
evals/           golden set, runner, gate, LLM judge
infra/sql/       Postgres + pgvector schema
```
