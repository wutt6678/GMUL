# Granularity-Controlled Multimodal Unlearning (GMUL)

Can a multimodal model be made to forget an entity–attribute association **at a
chosen level of granularity** — neither leaking the finer detail it was asked to
drop, nor destroying the coarser branch that detail lived on?

That is the whole question this repository exists to answer. Package
`granunlearn`, Python ≥ 3.10, MIT.

The setting is a vision-language model (Qwen3.5-9B with LoRA adapters) over
entities that carry a hierarchy: MLLMU persons whose attributes nest
(city → region → country) and iNaturalist species whose attributes nest
(genus → family → order). "Forget that this person lives in San Francisco" has
at least three defensible outcomes — the model still says San Francisco
(*leakage*), it says California-adjacent coarser truth (*correct granularity*),
or it says nothing at all about the person (*over-forgetting*). Those three are
different results and this repository never collapses them into "incorrect".

---

## Status: what has run and what has not

This is a preregistered confirmation design, so "not yet run" is a load-bearing
fact rather than an omission.

| Stage | Artifact | State |
| --- | --- | --- |
| Exploratory pipeline (Iterations 1–11R) | `data/reports/mllmu_pilot100_final_evaluation.json`, 30 prediction files bound by `mllmu_pilot100_prediction_manifest.json` | **run, committed** |
| Confirmation split (11C stage 3) | `data/mllmu_hier_confirm100/` — 1,209 queries, 402 photographs sealed by hash | **built, sealed, committed** |
| Protocol freeze (11C stage 2, re-frozen through 5R) | `data/reports/mllmu_pilot100_confirmation_freeze.json` — `refusals: []`, `--check-only` clean | **frozen, committed** |
| Confirmation scoring (11C stage 5) | `data/mllmu_hier_confirm100/predictions/`, `data/reports/mllmu_confirm100_final_analysis.json` | **NOT RUN — both absent** |

**No preregistered claim has a verdict yet.** The stage-5 code is committed,
CPU-tested and hash-bound into the freeze, and is deliberately unrun: the
confirmation is scored exactly once, so the code that scores it is reviewed
before any GPU hour is spent on it. The exploratory numbers below are *not* that
verdict.

---

## What is measured

Definitions live in `src/granunlearn/evaluation/hierarchy_metrics.py` and are
frozen there.

| Metric | Meaning |
| --- | --- |
| **FILR** | Fine Information Leakage Rate — fraction of outputs matching a level *finer* than the unlearning target. Lower is better. |
| **TGA** | Target Granularity Accuracy — fraction of outputs matching the post-unlearning acceptable answer, i.e. answering at exactly the intended granularity. Higher is better. |
| **ancestor_retention** | Accuracy on probes requesting a level *above* the target. |
| **retain_same / retain_other** | Accuracy on non-target associations of the same entity, and of other entities. |

Failure taxonomy for target probes — `under_forgetting` (finer than target; the
same set as FILR hits), `over_forgetting` (an ancestor of the target),
`wrong_branch` (a value from a different association's hierarchy), `refusal`,
`hallucination`.

Every rate is published in **two averaging units that are never blended**:
*entity-macro* (each entity weighted equally — the primary estimand, and the
unit the bootstrap resamples) and *row-micro* (per query row). Stratifications:
route (T→T, I→T, I+T→T), hierarchy type, target depth, and — since 11R —
**image provenance** (`held_out_photo` vs `seen_photo_unseen_wording`), which is
what separates a genuinely unseen photograph from unseen wording over a
photograph training already consumed.

## Model states

| | |
| --- | --- |
| Reference | `BASE` (un-finetuned), `MF` (fine-tuned), `MG` (granularity-target oracle), `MN` (negation) |
| Unlearning candidates | `B0` (no-op: a byte-identical copy of MF), `B1` (complete-forget), `B2` (coarse-positive SFT), `B2R`, `B3` (granularity-aware) |
| Scored by the confirmation | **exactly `B3`, `B0`, `MG`** — the freeze records every other state as excluded and `evaluate_confirmation_split.py` refuses to generate one |

`B0` is the no-op baseline and is proven so by hashing both adapter contracts
rather than by inheriting 11R's output comparison; on the exploratory split `B0`
reproduced `MF` with 0 raw-output mismatches over 2,259 test queries.

## Datasets

| Directory | Version | Contents |
| --- | --- | --- |
| `data/mllmu_hier_smoke/` | smoke | end-to-end plumbing, tiny |
| `data/mllmu_hier_pilot100/` | `pilot100_v2` | 100 entities (64 persons + 36 species), 477 associations, 496 unique images; the exploratory split — training, unlearning groups and 2,259 test queries |
| `data/mllmu_hier_confirm100/` | `confirm100_v1` | 1,209 evaluation-only queries over 100 entities / 317 associations, 402 unique images sealed by sha256. No training or validation use: the adapters already exist |
| `data/salmu_*`, `data/raw/inaturalist/` | — | SalmuBench cross-dataset comparison; iNaturalist fetch pools with `PROVENANCE.json` |

The confirmation's 1,209 probes are 504 wording probes (42 target persons × 12),
360 photograph probes (30 target species × 12) and 345 retention probes. Its
402 photographs are the 360 selected hash-disjoint from exploratory media plus
the 42 exempted target-person portraits, and the exemption is recorded with its
cost rather than argued away.

---

## Layout

```
src/granunlearn/
  hierarchy/     hierarchy engine: parsers, canonicalization, taxonomy, validation
  datasets/      MLLMU + iNaturalist adapters, splits, image-split policy
  evaluation/    scoring, hierarchy_metrics, query_generation, paired_ci,
                 prediction_provenance, reference_eval, confirmation_templates
  training/      reference and unlearning trainers, candidate grid, state datasets
  schema/        pydantic records: Association, Query, Prediction, Hierarchy
  salmu/, qwen/  cross-dataset comparison; the VLM client and semantic pipeline
scripts/         38 pipeline entry points, build_* → train_* → evaluate_* → analyze_*
scripts/lanes/   6 GPU chain scripts, incl. wait_for_gpu.sh (shared device claimer)
tests/unit/      CPU-only suite, runs in CI
tests/integration/  GPU suite, deliberately NOT run in CI
data/reports/    38 committed JSON/CSV evidence artifacts
requirements/    base, qwen, qwen-tested, salmu, ci-unit
```

## Install

```bash
# GPU / artifact machine
python -m pip install -e ".[qwen]"          # or: -r requirements/qwen-tested.txt

# CPU-only, exactly what CI installs
python -m pip install -r requirements/ci-unit.txt
python -m pip install -e . --no-deps
```

`requirements/ci-unit.txt` is pytest, numpy, pandas, pyarrow, pydantic, pyyaml
and pillow — no torch. That is a contract rather than an aspiration: the GPU
stack is imported inside the functions that need it, and
`tests/unit/test_ci_dependency_closure.py` derives the closure from the
requirements file and AST-walks the **module-level** imports of every test module
and the repository modules it reaches, so a stray top-level `import torch` fails
here instead of hiding every other result behind a collection error in CI.

## Tests

```bash
pytest tests/unit -q          # 1502 tests, CPU only, ~2.5 min on the artifact box
```

The same 1502 pass with `torch`, `transformers`, `peft`, `accelerate` and
`datasets` made unimportable, which is how the CPU-only contract is checked on a
machine that has the GPU stack installed. `.github/workflows/tests.yml` runs the
unit suite plus a step that loads every committed report the evidence claims
depend on.

Two coverage boundaries are stated rather than left to a skip count, and both
are gitignored inputs rather than unfinished work:

* fabricating a **provenance-verified** prediction file needs the real adapter
  bytes, because `PredictionFingerprint.build` hashes them;
* asserting an exact answer from the image preflight — or running the scorer's
  `main()`, which re-hashes the pool in every mode — needs the 402 sealed
  photographs.

Both guards skip naming what is absent, and both run on the machine that trained
the adapters and fetched the pool, which is the machine that will run the three
GPU passes. Measured: **1431 passed / 71 skipped** in a bare clone with a venv
built from `requirements/ci-unit.txt` alone, **1502 passed / 0 skipped** here.
The committed half of each boundary — the manifest pinning 402 paths and hashes,
the pool's disjointness from exploratory media, the fetch provenance behind it —
is asserted in `test_iteration11c_probes.py` and needs no bytes at all.

---

## Evidence discipline

The interesting property of this repository is not the model code; it is that a
result produced here is hard to produce accidentally.

* **Frozen before inference.** `scripts/freeze_confirmation_protocol.py` binds
  every parameter the claims depend on — estimand, family, α, test, seeds,
  bootstrap counts, generation configuration, dataset and code hashes.
  `--check-only` re-derives the freeze from the repository and reports DRIFT if
  anything moved; re-freezing needs `--allow-refreeze` and is refused once a
  confirmation prediction exists.
* **Analysis code is hash-bound.** Seven scripts under
  `code.analysis_scripts_sha256`, plus `src/granunlearn/evaluation/paired_ci.py`
  pinned separately because it implements the sign-flip test, the paired CI and
  the Holm step-down while deliberately sitting outside the code fingerprint.
  Both the scorer and the analyzer re-hash all of them at runtime: a pin that is
  recorded but never compared describes the protocol instead of enforcing it.
* **Every prediction carries a provenance sidecar** binding the adapter bytes
  and `adapter_config.json`, the dataset version and artifact hashes, the image
  manifest, the generation configuration, the code fingerprint and the base-model
  revision. Reuse of an existing prediction file is a *verified* decision, never
  a filename match. `scripts/verify_prediction_provenance.py` audits them.
* **The eight sealed-split invariants** are enforced structurally where possible.
  "Partial state results are not inspected before every scored state has
  finished" is a rule about discipline if it is only written down, so
  `evaluate_confirmation_split.py` imports no aggregation code at all — a run
  that generated one state and stopped has no way to print a rate, an interval
  or a p-value. A test asserts this against the module's AST.
* **Amendments are outcome-blind and measured.** The two protocol amendments
  recorded after the freeze each carry ordering claims *computed* from committed
  artifacts (freeze timestamps, the pool's provenance, the count of prediction
  files that existed), not typed. Where a claim turned out to be false — a
  "sealed before the fetch" timeline in 11C-3b — it was retracted in the freeze
  with the narrower defensible claim in its place, and a repo-wide sweep for the
  stale wording is itself a test.

## Running the confirmation pass (stage 5)

```bash
bash scripts/lanes/confirm100_11c5_chain.sh
```

One invocation, six steps:

| Step | What it does |
| --- | --- |
| `lock` | One chain at a time (`mkdir`, pid recorded). A lock whose pid is dead is reclaimed; a lock naming no pid is left alone. Two chains would queue the same states and write the same parquet |
| `pre` | Fails before any GPU hour: freeze present and unrefused, split sealed against its query-id hash, all three adapters re-hashed against the pinned contracts, every frozen analysis-script and `paired_ci` hash compared with the bytes on disk, all 402 photographs re-hashed, every image-route query resolved to a photograph, and the base-model revision resolved to the pinned snapshot directory |
| `todo` | `--list-outstanding`: which states still need work, decided by **verification**, not by whether a filename exists |
| `gen` | One lane per outstanding state, each claiming a GPU through `wait_for_gpu.sh`, generating all 1,209 queries, verifying **its own** state and exiting on that alone |
| `gate` | `--verify-only`: all three sidecars verify and all three were generated over the same query order |
| `anal` | `analyze_confirmation_split.py`: no GPU, assembles the report |

No batch size, image batch size or token budget appears on any command line —
all three are frozen and read from the freeze, so a lane cannot drift the
generation configuration by editing a variable in the chain.

The **gate, not the lanes' exit statuses, is the completeness authority**: a lane
can exit nonzero after its evidence was sealed, and stopping there would leave a
complete confirmation unassembled. Such an anomaly is investigated rather than
obeyed — `--list-outstanding` is asked again and the chain proceeds only if
nothing is missing. Exit codes: `2` preconditions or todo list, `3` already
scored, `4` the gate refused, `5` the analysis refused or wrote nothing, `6`
another chain holds the lock. `SIGTERM` stops the lanes instead of orphaning
them.

Resumption is by verified sidecar reuse, so an interrupted pass is cheap to
restart *and* actually restartable: a state whose parquet exists but does not
verify is regenerated, where a filename test would skip it, the gate would then
refuse it, and the chain would stop with the offending file still in place.
Logs are under `outputs/lanes/`; the chain log is the journal and every other log
is truncated per run so a tail of one is never a previous run's.

The same steps by hand:

```bash
python scripts/evaluate_confirmation_split.py --list-outstanding
python scripts/evaluate_confirmation_split.py --device cuda:0 --state B3
python scripts/evaluate_confirmation_split.py --verify-only
python scripts/analyze_confirmation_split.py --check-only
```

Prerequisites: the three adapter directories on disk, the pinned base-model
revision present in the local HF cache (a moved or absent ref refuses rather than
silently loading a different model), and the 402 sealed photographs.

---

## Exploratory result (Iteration 11R) — not the confirmation

Row-micro rates on the frozen `pilot100_v2` test split, all nine states over
2,259 queries, one uniform batch layout:

| State | FILR ↓ | TGA ↑ |
| --- | --- | --- |
| `BASE` | 0.2617 | 0.1119 |
| `MF` = `B0` | 0.4469 | 0.2107 |
| `MN` | 0.2642 | 0.1778 |
| `B1` | 0.3506 | 0.2148 |
| `B2` | 0.3119 | 0.2897 |
| `B2R` | 0.3012 | 0.3119 |
| **`B3`** | **0.2091** | **0.3909** |
| `MG` (oracle) | 0.1407 | 0.4395 |

Entity-macro paired 95% percentile CIs for `B3 − B0` (72 entity clusters):
FILR **−0.2442** [−0.2870, −0.2028], TGA **+0.1982** [+0.1620, +0.2352],
retain_same −0.0164 [−0.0674, +0.0282], retain_other +0.0213 [−0.0602, +0.1056],
over_forgetting +0.0199 [+0.0091, +0.0310].

So the granularity-aware baseline moves a substantial part of the way from the
no-op toward the oracle on the exploratory split, without measurably costing
retention. Two caveats that are part of the result and not decoration:

* These are **exploratory** numbers over the split that selected `B3`. The
  preregistered verdict is the confirmation's — entity-macro `B3 − B0` on TGA and
  FILR over all 72 target entities, one-sided sign-flip permutation tests
  (10,000 draws, seed 20260908) with a Holm step-down over the two claims at
  familywise α = 0.05 — and it has not been computed.
* `B3`'s TGA interval against `MG` crosses zero with a half-width larger than the
  prespecified δ = 0.05 margin, so it is reported **INDETERMINATE**, not
  equivalent. The margin is a reporting yardstick; no equivalence test is run on
  the confirmation split and the freeze records that.

## What is committed, and what is re-derivable

Committed: both dataset directories' parquets and manifests, both image
manifests, `PROVENANCE.json` for each iNaturalist pool, all 38 stage reports, the
power analysis and the freeze.

Gitignored, with the record that makes each one auditable anyway:

| Ignored | Re-derivable from |
| --- | --- |
| Photographs (432 + 720 CC-licensed files) | `PROVENANCE.json`: source URL, licence, attribution, observation id and sha256 per file, re-fetchable with `scripts/fetch_inat_species.py` |
| Adapter checkpoints (~300 MB each) | the committed training/unlearning JSONLs, the candidate grid and each recipe recorded in the freeze |
| Prediction parquets | 30 exploratory ones are bound by hash in `mllmu_pilot100_prediction_manifest.json`; the confirmation's will be beside their own sidecars |

The one exception, and the reason for it: `data/reports/*` is ignored, so each
research-provenance report carries an explicit `!` negation — including the
confirmation analysis, which does not exist yet and whose absence from the
repository is itself the record that stage 5 has not run.

## Iteration history

`git log --oneline` is the authoritative narrative and its commit messages carry
the measurements. In brief: **0–3** schemas, hierarchy engine, dataset adapters
and validation gates · **4–7** smoke experiment, reference states MF/MG/MN,
frozen evaluator · **8–9** the failure taxonomy and the first MF→MU unlearning
baselines · **10–11** pilot-100, SalmuBench cross-dataset comparison ·
**11R** the visual-split repair (the image route had been serving the photograph
training consumed), provenance-gated reuse, sharded regeneration of all evidence
· **11C** the confirmation: power analysis, protocol freeze, photograph
selection, the 1,209-query split, five review repairs, and the dedicated scorer
and analyzer that will produce the verdict.
