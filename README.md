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

| Node | Does | Now | Target |
|---|---|---|---|
| intake | Transcribes the voice note | Whisper wrapper | + PII redaction (GLiNER, Presidio) |
| vision | Finds damage, maps it to parts | Replay detector | RT-DETR on CarDD + OWLv2 parts |
| coverage | Retrieves and cites policy clauses | Dense (bge-small) → cross-encoder rerank, in memory or on pgvector | + ColQwen2 page lane, LLM coverage reasoning |
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
make retrieval      # policy-QA ablation: keyword / dense / hybrid, with and without rerank
make ingest         # load the policy corpus into Postgres + pgvector
make test-pg        # integration tests against Postgres + pgvector
make api            # http://localhost:8000/docs
```

Optional local stack (Postgres + pgvector, Redis, MinIO, Langfuse): `make up`.

Extras: `uv sync --extra ml` (real models), `--extra pg` (Postgres client),
`--extra embed` (embedding model and reranker; pulls in torch), `--extra rag` (both),
`--extra llm` (Anthropic judge, query rewriting, Langfuse tracing).

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
| Policy-QA hit@1 / Recall@5 (added with retrieval) | | 0.62 / 0.77 |

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

## Policy retrieval

The corpus is 42 clauses in the structure of a personal auto policy: insuring agreements,
definitions, every Part D exclusion, conditions, duties and endorsements. Many of them are
near-misses for each other on purpose. `evals/golden/retrieval_queries.jsonl` holds 50
questions phrased the way claimants and adjusters ask them, each labelled with the clauses
that answer it. 23 are tagged `lexical_gap`: they share few words with their answer ("someone
keyed my car" is malicious mischief, "I took it to a track day" is the racing exclusion).

`make retrieval` (replay, 50 questions, k=5):

| First stage | Rerank | hit@1 | Recall@5 | nDCG@5 | Lexical-gap R@5 |
|---|---|---|---|---|---|
| BM25 | | 0.38 | 0.44 | 0.40 | 0.20 |
| Dense (bge-small-en-v1.5) | | 0.34 | 0.72 | 0.53 | 0.65 |
| Hybrid (RRF of both) | | 0.42 | 0.60 | 0.49 | 0.37 |
| BM25 | bge-reranker-v2-m3 | 0.56 | 0.58 | 0.55 | 0.28 |
| Hybrid | bge-reranker-v2-m3 | 0.64 | 0.76 | 0.68 | 0.63 |
| **Dense (default)** | **bge-reranker-v2-m3** | **0.62** | **0.77** | **0.68** | **0.67** |
| LLM rewrite → dense | bge-reranker-v2-m3 on the rewrite | 0.60 | 0.98 | 0.82 | 1.00 |
| **LLM rewrite → dense** | **question + rewrite scores added** | **0.64** | **0.98** | **0.82** | **1.00** |

What the numbers say:

- **Dense beats keyword search by a wide margin** on recall, most of all on the lexical-gap
  slice (0.65 vs 0.20).
- **Fusion with BM25 hurts.** RRF pulls BM25's wrong matches into the top 5. I also tried
  weighting dense 2x and 3x in the fusion; it still did worse than dense alone.
- **The reranker is where top-1 comes from** (0.34 → 0.62). Turning it off drops claims
  decision accuracy from 1.00 to 0.80; the gate blocks that.
- **The 7 misses left after rerank needed world knowledge**, not better embeddings: "keyed"
  means vandalism, "seized by police" means confiscation, "mice chewed the wiring" is contact
  with an animal. An LLM knows those mappings.

Caveat: I chose the default from these six configurations (plus bge-base and the query-prefix
on/off) on the same 50 questions, so treat the numbers as a dev-set score.

**LLM query rewriting** (`retrieval/rewrite.py`). Claude (Opus 5.5, low effort) restates the
question in the policy's own vocabulary before retrieval. I wrote the prompt once, with no
examples, without looking at the questions. `make retrieval` compares four ways to use the rewrite:

- **Search with the rewrite only** finds every lexical-gap clause (R@5 0.67 → 1.00) and all 7
  old misses. But the rewrites are long (36 words on average) and general, often mentioning
  "physical damage coverage" or the deductible, so the declarations page took rank 1 on
  several questions the original wording got right: hit@1 fell from 0.62 to 0.60.
- **Searching with both and fusing, reranking on the question** barely moves recall (0.78):
  the cross-encoder reading the question is exactly what fails on these questions.
- **Search with the rewrite, then add the cross-encoder's score for the question to its score
  for the rewrite** keeps the recall and wins back top-1: hit@1 0.64, R@5 0.98, nDCG@5 0.82.

I tried four ways of combining the two rerank scores on the same 50 questions. The top ones
differ by 2–3 questions, so I picked the simplest one that kept full recall and stopped there.

Rewriting is **off by default** (`CLAIMSIGHT_QUERY_REWRITE=off`). The only consumer today is
the coverage agent, and it already searches in policy vocabulary (below), so turning it on
would add an LLM call per claim for nothing. It becomes the default for free-text policy
questions once the LLM coverage step reads the claimant's statement. CI replays the rewrites
from a cassette, so the comparison table is checked on every push.

**Coverage retrieves by peril, not by statement.** Searching with the claimant's whole
statement let incidental details win: "the windshield is smashed" pulled glass clauses ahead
of vandalism, "keyed both doors" matched nothing useful. Intake already gives a structured
loss type, so the coverage agent searches with the policy's term for that peril ("malicious
mischief or vandalism") plus a lookup for the declarations page. Scanning the statement for
exclusions is the job of the LLM coverage step on the roadmap. A side effect: claim
statements no longer affect retrieval, so `make data` never invalidates the cassettes.

**Two backends, one interface.** In-memory lanes run in replay and CI. On Postgres, the dense
lane is pgvector HNSW (cosine, `ef_search` 100) and the keyword lane is `tsvector`. Fusion and
reranking sit on top of either. The CI `pgvector` job checks that ingest is idempotent, that
HNSW returns exactly the in-memory neighbours, and that the full stack scores identically on
both backends. Postgres's `ts_rank_cd` is weaker than BM25 (hit@1 0.28 vs 0.38), another
reason dense is the first stage.

**Cassettes.** `cassettes/embeddings.json` (base64 float16 vectors), `cassettes/rerank.json`
(a score for every query × clause pair) and `cassettes/rewrites.json` (one rewrite per
question, tagged with the model and a hash of the prompt) let CI run all of this without
torch or an API key. Replay raises on anything not recorded. `make record` re-records after
changing clauses, queries, the peril queries, any model or the rewrite prompt; the rewrite
part needs Anthropic credentials and costs a few cents.

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
- [x] Weeks 3–4: 42-clause corpus, policy-QA eval, dense + rerank retrieval, pgvector ingest and lanes, CI integration job
- [x] Weeks 3–4: LLM query rewriting (policy-QA R@5 0.77 → 0.98, off by default until a free-text consumer exists)
- [ ] Weeks 3–4, rest: LLM coverage reasoning with quote validation, ColQwen2 page lane
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
  retrieval/     BM25, dense, RRF hybrid, cross-encoder rerank, LLM query rewrite, pgvector
  tools/         NHTSA, weather, photo forensics, geo
  api/           FastAPI app with human review
evals/           golden set, runner, gate, LLM judge
infra/sql/       Postgres + pgvector schema
```
