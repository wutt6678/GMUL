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

This is a preregistered confirmation design, so which stage has run is a
load-bearing fact rather than an omission. Every stage has now run.

| Stage | Artifact | State |
| --- | --- | --- |
| Exploratory pipeline (Iterations 1–11R) | `data/reports/mllmu_pilot100_final_evaluation.json`, 30 prediction files bound by `mllmu_pilot100_prediction_manifest.json` | **run, committed** |
| Confirmation split (11C stage 3) | `data/mllmu_hier_confirm100/` — 1,209 queries, 402 photographs sealed by hash | **built, sealed, committed** |
| Protocol freeze (11C stage 2, re-frozen through 5R) | `data/reports/mllmu_pilot100_confirmation_freeze.json` — `refusals: []`, `--check-only` clean | **frozen, committed** |
| Confirmation scoring (11C stage 5) | `data/mllmu_hier_confirm100/predictions/` — three files of 1,209 rows each, bound by sidecar; `data/reports/mllmu_confirm100_final_analysis.json` | **run once, scored, committed** |
| Post-run execution audit | `data/reports/mllmu_confirm100_execution_provenance.json`, derived from the committed `data/mllmu_hier_confirm100/execution_logs/` | **filed post-hoc; `post_hoc: true`, binds nothing** |
| Iteration 12 selection protocol (exploratory) | `data/reports/mllmu_iter12_retention_probe.json` — the fit/probe partition of the 70 retained entities; `data/reports/mllmu_iter12_selection_protocol_freeze.json` — D_G's direction, tolerance and tie-break, the retention floor, the 9-row grid | **frozen, committed; frozen before any candidate was trained, and the freeze now refuses to be rewritten** |
| Iteration 12 Stage 1 (exploratory) | `data/reports/mllmu_iter12_retention_selection.json` — 9 rows scored, 8 disqualified by the retention floor, B0 selected | **run once, 2026-09-12 → 09-13; a negative result: no replay setting preserves retention on never-rehearsed entities** |
| Iteration 12 Stage 1b (exploratory) | `data/reports/mllmu_iter12_seed_replication.json` — two near-miss parents at four seeds each, scored by the frozen mean-based floor; `data/reports/mllmu_iter12_seed_replication_freeze.json`; `data/reports/mllmu_iter12_seed_replication.REPAIR.json` | **run once, 2026-09-13; a negative result: neither parent clears the floor on a four-seed mean, and every one of the eight numbers straddles the anchor** |
| Iteration 12 Stage 1c (exploratory) | `data/reports/mllmu_iter12_route_probe.json` — the route-stratified measurement basis, derived from the dataset with no prediction read; `data/reports/mllmu_iter12_route_stratification_freeze.json` — the eight-number floor, bound to 19 prediction parquets and one generation contract; `data/reports/mllmu_iter12_route_stratified.json` | **re-scored from existing predictions, 2026-09-14, zero GPU-hours; verdict unchanged — 0/8 eligible on eight numbers exactly as on four — and Stage 1b's straddling failures become decisive on the image route, where every seed of `ep5` fell below the anchor on all four numbers** |

**Both preregistered claims were confirmed.** Entity-macro `B3 − B0` over all 72
target entities (42 persons, 30 species), 864 paired target probes, one-sided
cluster sign-flip permutation tests with 10,000 draws at seed 20260908, Holm
step-down over the two claims at familywise α = 0.05:

| Primary claim | B0 | B3 | Diff | 95% CI | p | Holm |
| --- | --- | --- | --- | --- | --- | --- |
| FILR (leakage rate, lower is better) | 0.3056 | 0.1481 | **−0.1574** | [−0.2176, −0.0937] | 0.0001 | rejected, step 1 |
| TGA (accuracy, higher is better) | 0.3438 | 0.5023 | **+0.1586** | [+0.1007, +0.2211] | 0.0001 | rejected, step 2 |

`retained: []` — nothing survived the step-down, so both nulls were rejected in
the directions the freeze declared. `0.0001` is `p_smallest_reportable`: no draw
was as extreme as the observed statistic, so it reads as "at the resolution floor
of 10,000 draws", not as four-digit precision, and the report records that the
resolution comes from 41 and 44 *flippable* entity clusters rather than 72,
since roughly 40% of per-entity differences are exactly zero.

Scored once, on 2026-09-09, and nothing re-tuned on it: the report carries
`anything_tuned_on_these_predictions: false`,
`partial_results_were_inspectable_before_completion: false`,
`checkpoint_selection_invoked: false` and `reference_state_gate_invoked: false`.

Three things the result does **not** say, all recorded in the report and stated
here because they are part of it rather than decoration:

* retention is *lower* under `B3` — `retain_same` −0.0476 CI [−0.119, +0.019],
  `retain_other` −0.1111 CI [−0.2224, +0.0002]. Both are descriptive: no
  preregistered claim covers retention and no p-value is computed. Both intervals
  **include zero**, so neither is a demonstrated cost — `retain_other`'s upper
  bound of +0.0002 is as near to excluding zero as an interval can be without
  excluding it, which is a reason to measure retention properly in its own study,
  not a finding about it.
* over-forgetting on the person stratum is 0.0198 under `B3` against 0.0000
  under `B0`, CI [+0.0000, +0.0595]. The photograph stratum shows none.
* `B3 − MG` on FILR is +0.0938 CI [+0.0451, +0.1493] with
  `equivalence_concluded: false`. The δ = 0.05 margin is a reporting yardstick;
  no equivalence or non-inferiority test was run, so this is not "`B3` ≈ `MG`".

The exploratory numbers further below are still not this verdict. They are what
selected `B3` in the first place, which is exactly why they cannot also confirm
it.

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
pytest tests/unit -q          # 1845 tests, CPU only, ~3.2 min on the artifact box
```

The same 1845 pass with `torch`, `transformers`, `peft`, `accelerate` and
`datasets` made unimportable, which is how the CPU-only contract is checked on a
machine that has the GPU stack installed. `.github/workflows/tests.yml` runs the
unit suite plus a step that loads every committed report the evidence claims
depend on, and checks out with **`fetch-depth: 0`**: four tests re-derive claims
from the repository's *history* rather than trusting the report that states them
— the 11C audit's verifier-binding gap and the 11C-5R seal's module hash — and
under `actions/checkout`'s default depth of 1 those commits are absent, so the
checks would skip and report green without verifying anything.

Three coverage boundaries are stated rather than left to a skip count, and all
three are gitignored inputs rather than unfinished work:

* fabricating a **provenance-verified** prediction file needs the real adapter
  bytes, because `PredictionFingerprint.build` hashes them;
* asserting an exact answer from the image preflight — or running the scorer's
  `main()`, which re-hashes the pool in every mode — needs the 402 sealed
  photographs;
* re-hashing the 19 prediction parquets Stage 1c is bound to, and re-running its
  analyzer, needs those parquets. What a clone *can* check is everything the
  committed freeze and report assert about them, so the skip is limited to the
  byte re-verification: the freeze still records 19 sha256s and one measurement
  contract, and the tests that read that record run everywhere. Its
  `--check-only` reports "19 parquet(s) without one" rather than passing over a
  shorter list, and the analyzer refuses rather than silently re-scoring fewer
  states.

All three guards skip naming what is absent, and all three run on the machine
that trained the adapters and fetched the pool — the machine that ran the GPU
passes.
Measured: **1763 passed / 82 skipped / 0 failed** in a bare clone with a venv
built from `requirements/ci-unit.txt` alone, and **1845 passed / 0 skipped**
here. The same clone taken at `--depth 1` gives **1759 / 86 / 0**: the four
extra skips are exactly the history-derived checks, which is what
`fetch-depth: 0` in the workflow exists to prevent. The whole suite also passes
with `PytestRemovedIn10Warning` promoted to an error.
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
  anything moved. Overwriting the freeze needs `--allow-refreeze`; without it the
  script refuses whenever a freeze already exists. **The flag is the only gate.**
  The code does not check whether confirmation predictions exist, so "re-freeze
  only before scoring" is a procedural rule and not an enforced one — its refusal
  message asks the operator to pass the flag only if the confirmation is unscored,
  which is a request, not a check. What actually makes the rule stick is
  consequence; see the next bullet.
* **Code and data are hash-bound, and the hashes are compared.** Four sets of
  pins are re-checked against the bytes on disk at every runtime entry point —
  the scorer, the analyzer and the chain's own preconditions: the ten
  `code.fingerprinted_modules` that turn queries into scores, the seven
  `code.analysis_scripts_sha256` that turn scores into claims,
  `src/granunlearn/evaluation/paired_ci.py` pinned separately because it
  implements the sign-flip test, the paired CI and the Holm step-down while
  deliberately sitting outside the code fingerprint, and the confirmation
  dataset's own `queries.parquet`, `associations.parquet`, `manifest.json` and
  image-manifest roll-up under `confirmation_dataset`. A pin that is recorded
  but never compared describes the protocol instead of enforcing it, and the
  prediction sidecars cannot substitute: they hash the same modules and the same
  dataset files, but the expectation is *re-derived from the same disk* at
  verification time, so an edit moves both sides together and every sidecar still
  verifies. Only the freeze holds a value that was fixed before the bytes could
  be touched — which is why a prompt rewritten behind an unchanged `query_id` is
  refused now and was not before.
* **Scoring seals the code permanently, and a re-freeze cannot undo it.** Two
  different comparisons bind the same modules. `verify_frozen_code` compares the
  *freeze* against the bytes on disk, so re-freezing satisfies it. `verify_sidecar`
  compares each sidecar's **recorded** `code.modules_sha256` against a fingerprint
  built fresh from disk, and a sidecar is immutable history written at generation
  time — so editing any of the eighteen pinned paths (the ten modules, the seven
  analysis scripts, `paired_ci.py`) moves the disk side only and the committed
  predictions can never verify again. Measured, not reasoned: `verify_sidecar`
  returns 0 refusals on the tree as committed and returns `module hash differs
  (scoring or generation logic changed since this file was generated)` when one
  entry is altered. The sealing date is therefore the *generation* date, not the
  freeze date.
* **What the run did not record is on the record.** `scripts/audit_confirmation_execution.py`
  files `data/reports/mllmu_confirm100_execution_provenance.json`, clearly marked
  `post_hoc: true` and binding nothing. It records that the three states were
  generated at one commit (`73f49fd`) with identical package environments; that
  `prediction_provenance.py` — the module implementing `verify_sidecar` itself —
  was pinned by no hash; and that no GPU model, UUID, driver version, CUDA version
  or compute capability was captured anywhere, so **physical-GPU equivalence across
  the three states cannot be established retrospectively**. Only the device index
  survives, and all three successful generations claimed index 3. None of this
  licenses rescoring: a repeat is a new replication with its own protocol, never a
  re-run of this one.
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
* **An exploratory protocol can enforce what the confirmatory one only asks
  for.** Iteration 12's `scripts/freeze_iter12_selection_protocol.py` refuses to
  write a freeze at all once any candidate adapter exists — unconditionally, with
  no flag reaching past it, which is precisely the gate the confirmation's
  `--allow-refreeze` does not have. Its `--refreeze` path is permitted only
  *before* training, requires `--reason`, and appends an amendment carrying the
  sha256 of the protocol it supersedes.
  `scripts/select_iter12_retention_checkpoints.py` verifies the freeze before it
  generates anything, so changing the criterion afterwards breaks the run instead
  of quietly redefining it: setting the floor's tolerance to the rejected `+0.02`
  margin makes six tests fail, including the freeze check the selector runs.

## Running the confirmation pass (stage 5)

```bash
bash scripts/lanes/confirm100_11c5_chain.sh
```

This ran once, on 2026-09-09, and produced the committed result above. It cannot
be re-run: after its preconditions it checks for the analysis report and, finding
it, stops with `STOPPING: … already exists` and **exit status 3** — before it asks
which states are outstanding and before it queues a single lane. That guard is what
makes "scored exactly once" a property of the script rather than of operator
discipline, and `tests/unit/test_iteration11c5_chain.py` asserts the status.
Regenerating a state after seeing the verdict would be a protocol amendment that
belongs in the freeze, not a retry. The steps are documented because the run that
mattered went through them, and the logs it wrote — including the two
out-of-memory retries and every `CLAIMED GPU` line — are committed under
`data/mllmu_hier_confirm100/execution_logs/`.

One invocation, six steps:

| Step | What it does |
| --- | --- |
| `lock` | One chain at a time (`mkdir`, pid recorded). A lock whose pid is dead is reclaimed; a lock naming no pid is left alone. Two chains would queue the same states and write the same parquet. The per-run logs are reset **after** the lock is claimed, so a refused duplicate leaves the active run's evidence untouched |
| `pre` | Fails before any GPU hour: freeze present and unrefused, split sealed against its query-id hash, all three adapters re-hashed against the pinned contracts, every frozen module, analysis-script and `paired_ci` hash compared with the bytes on disk, the three confirmation dataset artifacts and the image-manifest roll-up compared with the freeze, all 402 photographs re-hashed, every image-route query resolved to a photograph, and the base-model revision resolved to the pinned snapshot directory |
| `todo` | `--list-outstanding`: which states still need work, decided by **verification**, not by whether a filename exists. The answer is written to a structured JSON file the chain reads back, never parsed out of stdout, and every name in it is checked to be exactly `B3`, `B0` or `MG` before anything is queued |
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
is reset per run — after the lock is claimed, never before — so a tail of one is
never a previous run's and a refused second invocation erases nothing.

The same steps by hand:

```bash
python scripts/evaluate_confirmation_split.py --list-outstanding \
    --outstanding-report outputs/lanes/outstanding.json
python scripts/evaluate_confirmation_split.py --device cuda:0 --state B3
python scripts/evaluate_confirmation_split.py --verify-only
python scripts/analyze_confirmation_split.py --check-only
```

`--outstanding-report` is what the chain reads: it carries `outstanding_states`,
`verified_states` and one `reasons` list per outstanding state, so a resumed run
says *why* a state needs regenerating rather than only that it does. Without the
flag the same names are printed to stdout, which is clean because that mode
disables logging process-wide — every logger in the import graph writes to stdout
through `setup_logger`, and one truncated sidecar was once enough to put an error
line where a state name was expected.

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
  familywise α = 0.05 — and it is reported in **Status** above, where both claims
  were rejected in their declared directions.
* `B3`'s TGA interval against `MG` crosses zero with a half-width larger than the
  prespecified δ = 0.05 margin, so it is reported **INDETERMINATE**, not
  equivalent. The margin is a reporting yardstick; no equivalence test is run on
  the confirmation split and the freeze records that.

## What is committed, and what is re-derivable

Committed: both dataset directories' parquets and manifests, both image
manifests, `PROVENANCE.json` for each iNaturalist pool, all 39 stage reports, the
power analysis and the freeze.

Gitignored, with the record that makes each one auditable anyway:

| Ignored | Re-derivable from |
| --- | --- |
| Photographs (432 + 720 CC-licensed files) | `PROVENANCE.json`: source URL, licence, attribution, observation id and sha256 per file, re-fetchable with `scripts/fetch_inat_species.py` |
| Adapter checkpoints (~300 MB each) | the committed training/unlearning JSONLs, the candidate grid and each recipe recorded in the freeze |
| Prediction parquets | 30 exploratory ones are bound by hash in `mllmu_pilot100_prediction_manifest.json`; the confirmation's three are committed beside their own sidecars, and the report's `inputs_bound` carries each parquet's sha256 |

The one exception, and the reason for it: `data/reports/*` is ignored, so each
research-provenance report carries an explicit `!` negation — including the
confirmation analysis, whose absence from the repository was the record that
stage 5 had not run, and whose presence is now the record that it has.

The confirmation's three prediction files are the one place this repository
tracks predictions rather than only their hashes. They are 132 KB, they are the
bytes the committed report's `inputs_bound` pins, and they are not regenerable:
batched greedy decoding is not bit-stable across batch compositions, so a re-run
would not reproduce them, and the protocol scores exactly once regardless.
Committing them lets a reviewer recompute every number in the report with no GPU,
no adapter and no photograph. The exploratory parquets stay ignored as large and
regenerable; these are neither.

## Iteration 12 (exploratory): retention, and a protocol frozen first

The confirmation's retention diagnostics are the open scientific problem:
retain-same −0.0476 CI [−0.119, +0.019], retain-other −0.1111 CI [−0.2224,
+0.0002], person-stratum over-forgetting +0.0198. Both retention intervals
include zero, so neither is a demonstrated cost — which is a reason to measure
retention properly in its own study, not a finding about it.

Iteration 12 is that study. It is **exploratory**: it selects a checkpoint, tests
no hypothesis, controls no error rate, and reads nothing from the sealed
confirmation. The successor it develops is method `B4` — B3's three components
with the replay strength swept — in two stages: retain replay first, and an
MF-preservation regularizer only if Stage 1 fails to beat its own reference row.

**The measurement problem, found before any training.** The `retain` replay group
is exactly the 387 retained associations, and those associations back **100% of
the retention queries in all three splits** — 387/387 for `retain_same_entity`
and 65/65 for `retain_other_entity`, and 198/198 and 64/64 in the confirmation
split as well. A candidate carrying `sft retain` is therefore scored in-sample on
retention while B0, the no-op, is scored out-of-sample. A floor saying "retention
must not fall below B0" then compares two different quantities and cannot
discriminate: applied to the committed pilot-100 grid it passes exactly the two
candidates carrying a replay group and fails all nine that do not. It selects
*for* replay instead of testing it.

So `scripts/build_iter12_retention_probe.py` splits the 70 retained **entities**
by a seeded hash into a `fit` half (183 associations, replayed) and a `probe`
half (204 associations, never replayed by any candidate), and the floor is
measured on probe queries only, where B0 and a replay candidate stand on the same
footing. `fine_target.jsonl` and `target_level.jsonl` are copied byte-for-byte
from the pilot-100 groups, so the target-side objective does not move. Splitting
at entity level is what makes "never rehearsed" true of a whole entity's
knowledge: an association-level split would leave every probe fact belonging to a
partly rehearsed entity.

Three things the freeze pins down, because each can change which candidate wins
and none is implied by "closest to MG subject to retention not falling":

* **the direction of D_G** — minimise; every component enters as
  `|v_j(M_U) − v_j(M_G)|`. It does **not** reward low leakage: a candidate that
  drives FILR below MG's level is penalised exactly as much as one that leaves it
  above, which a test asserts rather than a sentence claims. The floor is the only
  monotone requirement in the protocol — more retained knowledge is never
  disqualifying.
* **the numerical tolerance** — distances are compared exactly as
  `distance_to_reference` rounds them, to 6 decimals, with no added epsilon, so
  two candidates are tied iff their rounded distances are equal. The floor uses
  `1e-9`. The smallest difference the probe measurement can express is 0.00204
  (one flipped outcome in the widest of 35 probe entities), so `1e-9` sits six
  orders of magnitude below anything resolvable: it absorbs float representation
  error and cannot decide a real comparison. That is what separates a tolerance
  from a margin.
* **the tie-break** — `(distance_to_mg, candidate_id)` ascending. The id encodes
  hyperparameters and no outcome, so the order cannot be gamed by looking at
  results, and it does not depend on grid order, filesystem order, or which
  adapters happen to exist on disk. What it replaces — the pilot selector's strict
  `<` over dict insertion order — was deterministic but written down nowhere.

The floor requires **both** estimands. Row-micro and entity-macro disagree in
*sign* for retain-other on the pilot-100 evidence: the incumbent is −0.1056
against B0 row-micro and +0.0093 entity-macro. Freezing either alone would be
choosing the estimand that produces the preferred answer. A `+0.02` retention
margin was proposed and rejected — it is not a resolvable gap, not an interval
width and not a pre-registered effect size, so requiring improvement beyond "not
below B0" would invent a hypothesis and file it as a constraint.

The constraint is not vacuous. B0's pilot-100 floor is retain-same 0.5594 /
retain-other 0.6056 row-micro, and the checkpoint the confirmation just scored —
`B3_lam0.5_lr2e-05_ep5` — sits at 0.6111 / **0.5000**: it fails. Two of the
fifteen trained candidates pass.

Two consequences of halving the replay group are recorded rather than absorbed.
The retain term's share of each epoch's interleaved gradient stream falls from
0.683 to 0.504, so weight 1.0 does *not* reproduce the incumbent's effective
influence — the grid therefore includes the derived weight 1.3539 that does,
computed from the committed group counts and recomputed from them by a test. And
the probe half carries 1 of the 6 taxonomic retained associations, so taxonomic
retention is recorded as unmeasured, not as preserved.

None of this touches the eighteen sealed paths: `hierarchy_metrics.py` supplies
the row-micro rate through filtered inputs, `paired_ci.py` and
`select_unlearning_checkpoints.py` are imported rather than edited, and a test
cross-checks all five sealed dependencies against the hashes the confirmation
freeze recorded. The analysis report is still `b330b3c0488af6a0`.

### Stage 1 result: the floor rejected every replay setting

Stage 1 ran 2026-09-12 13:48 → 2026-09-13 00:34. The four training lanes waited
roughly four hours for a quiet device, trained 13:48→18:04, then MG and B0 were
generated serially (2h14m), the remaining eight in parallel (4h16m), and the
selection itself took 17 seconds. Every B4 trained on the **fit half**: retain
183 examples, 363-example stream, 230 optimizer steps at ep5 (138 at ep3, 368 at
ep8); B0 took 0 steps, being the no-op copy. No candidate trained on the pilot's
387-example retain group, so no probe association was ever rehearsed.

**All eight replay candidates are disqualified. B0 is the only eligible row and is
selected**, at D_G = 0.101757 against the best unlearner's 0.027343. The frozen
Stage-2 gate opens: the best eligible row does not strictly improve on the
reference row `B4_w1.0_lam0.5_lr2e-05_ep5` (0.046729), which was itself
disqualified.

Retain-**same** shortfalls are robust — 408 queries, so one flipped outcome is
0.00245, and the observed shortfalls run from −0.0073 to −0.1446 row-micro, i.e.
3 to 59 queries. A dose-response is visible: w0.5 −0.0760, w1.0 −0.0613,
w1.3539 −0.0368, w2.0 −0.0539, w4.0 −0.0073. Stronger replay does protect
retention and never reaches the no-op.

Retain-**other** verdicts are not robust — 76 queries, so one flipped outcome is
0.0132. `B4_w2.0_lam0.5_lr2e-05_ep8` is the only candidate to exceed B0 there
(+0.0131, exactly one query) and it holds the second-best D_G (0.030686); it
fails retain-same by −0.0564 (23 queries) and is therefore ineligible. That is
not revisited: freezing the rule before training is what makes a result like this
one a result rather than a starting point for choosing a different estimand.

The structural finding is about the constraint. The floor is anchored on B0,
which performs **no unlearning at all**, so "retain at least as well as B0 on
entities you never rehearsed" can only be met by (almost) not unlearning. That is
what was measured: the single eligible row is the no-op, 3.7× the best
unlearner's distance to MG. Stage 2's MF-preservation regularizer faces the same
anchor, and will return the no-op again unless the anchor itself is reconsidered
— which is a design decision to make *before* Stage 2 is frozen, not after it is
scored.

**A defect, deliberately left in place.** The selector hardcodes the method label
`B4` in its staging path and its log line, so the winner B0 is staged at
`data/checkpoints/mllmu_iter12_unlearn/selected/B4/adapters`. The bytes are
correct — byte-identical to MF, sha256 `4ee78bd1bb5cf6a7…`, exactly what a no-op
copy should be — and the report says `"selected": "B0"`, but the directory name is
wrong and would mislead anyone browsing the checkpoint tree. It cannot be fixed:
the selector is one of the eight hash-bound protocol paths, and the freeze now
refuses to be rewritten because nine candidates have adapters (`--refreeze` exits
1), which is the arm built for exactly this situation. Correcting the label would
permanently break `--check-only` and the selector's own freeze verification — a
worse artifact than a misnamed directory. Recorded here instead.

### Stage 1b: seed replication, frozen before any replicate was trained

Two candidates missed retain-same by 3 and 4 queries out of 408. A point floor
over a single training seed cannot distinguish "this method loses retention"
from "this draw lost retention", so Stage 1b re-runs those two at four seeds and
asks the floor of a **mean**.

The parents are `B4_w4.0_lam0.5_lr2e-05_ep5` and `B4_w2.0_lam0.5_lr2e-05_ep3`,
chosen as the two smallest retain-same shortfalls expressed in queries. They are
deliberately *not* the two best on D_G — selecting on the criterion being
revisited would be selecting on the outcome. Seeds are 42 (reused from Stage 1,
since re-running an identical RNG state says nothing about between-seed
variance) plus 43, 44 and 45, fixed before training.

**A measurement that cost nothing and changed the design.** Two B0 parquets
already existed, generated from *identical* adapter bytes under different
`image_batch_size` (8 in the pilot run, 1 in Stage 1). They disagree on 60 of
4,518 raw outputs and flip `is_correct_branch` on **15** rows — every one of
them image-route. The floor's 408 + 76 probe queries are **100% `text_to_text`**,
and **zero** of the 15 flips land in either subset. So generation instability is
real, confined to the image route, and does not reach the retention floor:
replicate retention differences will be attributable to the training seed alone.
The same comparison moved the full train+val vector's `filr` by −0.0004 and
`wrong` by +0.0008, which puts D_G's generation noise near 2e-4 — about 17×
smaller than the gap between Stage 1's two best candidates (0.027343 vs
0.030686), so it does not reorder them, but the sixth decimal the frozen
tie-break compares at is not physically meaningful.

What that comparison *cannot* show is whether generation is deterministic under
a **fixed** contract, which is the assumption every replicate comparison rests
on. So the chain runs a control first and on its own: regenerate B0 under the
Stage-1 contract and compare row by row. If any of the four floor numbers moves,
the study stops before training anything — a point floor over a
non-reproducible anchor is not interpretable, and replication cannot repair it.

The primary rule is the mean over replicates, at or above the Stage-1 B0 anchor
on all four numbers, scored by the **frozen** `floor_check` with the **frozen**
`1e-9` epsilon and no margin. The mean is compared unrounded: rounding it to the
four decimals its inputs carry would let a rounding step decide a comparison the
inputs did not. Min, max, sd, the count of replicates individually at or above
the anchor, and a non-parametric range classification (`every_replicate_below` /
`straddles` / `every_replicate_at_or_above`) are all reported and none is
decision-bearing — with k=4 an interval would import a distributional
assumption the design does not need.

Nothing frozen was edited to build this. The Stage-1 freeze refuses to be
rewritten now that nine candidates have adapters, and `candidate_grid.py`,
`retention_selection.py`, `unlearning_trainer.py` and the sealed
`select_unlearning_checkpoints.py` are all hash-bound, so Stage 1b is four new
protocol paths plus its own freeze, which *imports* the Stage-1 freeze's hashing
and document-flattening rather than copying them — so "does this still match the
repository" means the same thing in both studies. The new freeze refuses to be
written once any replicate adapter exists, refuses a bare overwrite, and refuses
`--refreeze` without `--reason`; its four amendments all moved only protocol-path
hashes and no criterion field, and the fifth was refused — the six replicates
were on disk by then, so the protocol can no longer be edited by anyone.

Cost, taken from Stage 1's own measurements rather than guessed: 6 trainings at
1609 s (ep5) and ~965 s (ep3), plus 7 generations at a 40.2-minute median —
roughly 7 GPU-hours.

**Result: both parents fail on the mean.** The control ran first and passed more
strongly than the design required — regenerating B0 from the Stage-1 adapter
under the Stage-1 contract reproduced all 4,518 predictions with **zero** rows
differing on any of the seven recorded fields, `raw_output` included. So the
anchor is a point, not a draw, and every difference below is the training seed
and nothing else.

| parent | ret-same row-micro | ret-same entity-macro | ret-other row-micro | ret-other entity-macro | clears |
| --- | --- | --- | --- | --- | --- |
| `B4_w2.0_lam0.5_lr2e-05_ep3` | +0.0055 | −0.0078 | −0.0165 | +0.0032 | 2/4 |
| `B4_w4.0_lam0.5_lr2e-05_ep5` | −0.0037 | −0.0073 | −0.0263 | −0.0176 | 0/4 |

Each cell is the four-seed mean minus the Stage-1 B0 anchor. The rule is all
four at or above, with no margin, so both are disqualified — the verdict Stage 1
reached from one seed, now reached from four.

**The informative part is the spread, not the verdict.** All eight numbers are
classified `straddles`: for none of them did every seed fall below the anchor,
and for none did every seed clear it. On retain-other row-micro, the binding
failure, the seed-to-seed sd is 0.0225 (ep3) and 0.0515 (ep5) against shortfalls
of −0.0165 and −0.0263 — **the noise is larger than the gap it is being asked to
adjudicate**. In the only units this measure has, those shortfalls are 1.25 and
2.0 queries out of 76, while one seed alone moved the number by up to 9 (ep5's
range is 0.3947–0.5132).

That answers a sharper question than the one asked. The floor is not failing to
detect a real retention loss; it is being applied at a resolution the probe half
cannot support. This is a limit on measurement power, not a finding about
replay.

D_G, reported and not decision-bearing, moved as well: ep3's Stage-1 single-seed
0.061929 turns out to have been the *worst* of its four seeds (mean 0.055779, sd
0.006951), and ep5's 0.041957 sits above its mean of 0.040861 (sd 0.007779). A
seed sd of ~0.007 on D_G exceeds the 0.003343 that separated Stage 1's two best
candidates, which supersedes the generation-noise bound quoted above: the frozen
tie-break's sixth decimal is dominated by training-seed variance rather than by
generation, and Stage 1's ordering of near-tied candidates was not resolvable.

**The frozen analyzer crashed, and a separate script finished it.**
`analyze_iter12_seed_replication.py:325` stored `rs.distance_to_mg(...)` whole,
but that function returns `tuple[float | None, list[str]]` — its own caller in
`retention_selection.py` unpacks it into `dist, used`. Line 386 collected those
tuples into `dists` and line 394 handed them to `aggregate`, whose first
statement is `float(v)`, so the run died with a `TypeError` *after* all eight
replicates had been scored and *before* the report was assembled.

Both files on that path are frozen protocol paths, and the freeze refuses
amendment now that six replicates exist — correctly, since a protocol edited
after its replicates could be fitted to them. So the frozen bytes stayed frozen
and `scripts/repair_iter12_seed_replication_scoring.py` runs them with that one
call unwrapped, proving rather than asserting that this cannot move the verdict:
the four protocol paths still hash to what the freeze records; an AST walk
recomputed at run time shows no decision-path function reaches
`distance_to_mg`; the analyzer has exactly one call site for it; and all eight
calls the patch served were recorded by caller frame as coming from that site.
`distance_to_mg` is not decision-bearing in any case — the report says so itself:
"the floor decides eligibility".

What that cannot claim is that the preregistered analysis ran to completion
unaided. It did not. The disclosure is filed *beside* the result in
`mllmu_iter12_seed_replication.REPAIR.json` rather than inside it, because
adding it to the report would mean editing a file the freeze has locked.

Replication did not increase the query count, which was known before it ran: the
probe half is fixed at 35 entities / 408 and 23 donors / 76 by the association
pool, so retain-other stays resolvable only in units of 1/76 = 0.0132 however
many seeds are run. Four seeds per parent bounds training-seed variance loosely
and licenses no interval claim, and only the two near-miss parents were
replicated — this says nothing about the other six Stage-1 candidates.

### Stage 1c: the floor was reading one of two routes

Stage 1b ended on a measurement-power limit, so the next question was whether
the probe could be made powerful enough to decide anything. `query_generation.py`
is one of the eighteen paths the confirmation freeze sealed, so generating more
queries per association is unavailable. But each of the probe half's 204
associations backs **two** retention families per split — `retain_same_entity` /
`retain_other_entity`, whose route is `text_to_text`, and `retain_same_entity_image`
/ `retain_other_entity_image`, whose route is `image_to_text` — and the sealed
`compute_hierarchy_metrics` has always emitted blocks for both, plus a pooled
`*_all_routes`. The frozen floor read only the text pair.

Because Stage 1 generated all 4,518 train+val queries for ten states and Stage 1b
did the same for seven, the image-route predictions were already on disk. Stage 1c
therefore costs **zero GPU-hours**: it re-scores what exists on eight numbers
instead of four.

Stratified rather than pooled, because the two routes are not interchangeable
instruments. Across the eight Stage-1 candidates, difference against B0:

| number | text route | image route | candidates failing |
| --- | --- | --- | --- |
| retain-same row-micro | −0.0558 (sd 0.0434) | **−0.1373** (sd 0.0620) | 8/8 vs 8/8 |
| retain-other row-micro | −0.0559 (sd 0.0335) | **−0.1382** (sd 0.0617) | 7/8 vs 8/8 |
| retain-other entity-macro | −0.0237 (sd 0.0434) | **−0.1562** (sd 0.0826) | 6/8 vs 8/8 |

Averaging those would report a single number describing neither. So each route
keeps its own four numbers and the floor requires all eight — which makes it
*harder* to pass than the rule it extends. **0/8 candidates are eligible on the
eight numbers, exactly as 0/8 were on the four**, so no Stage-1 verdict changes.

The route asymmetry is not a property of the image route. MG, the
granularity-controlled reference that D_G targets, loses almost nothing on it
(retain-same **+0.0074**, retain-other −0.0263) against text-route losses of
−0.0269 and −0.0132. The 2.5× asymmetry belongs to the replay candidates, not to
photograph-cued recall in general.

**What stratifying does and does not buy.** It does not enlarge any denominator:
each of the eight numbers still sits on 408 or 76 queries, because the association
pool is exhausted (90 target + 387 retained = all 477) and retain-other reaches
only 23 probe donors. The pooled `*_all_routes` rate, which *would* double n to
816 and 152, is reported beside the decision and floors nothing. The absolute
ceiling inside pilot-100 is 1,224 and 228 queries — every split and both routes —
so a mechanism whose retention effect is smaller than about 0.0044 is not
resolvable on this dataset at any number of seeds.

What the second instrument does buy is decidability. Recomputing Stage 1b's four
seeds per parent:

| parent | route | numbers where sd < \|shortfall\| | numbers where every seed fell below |
| --- | --- | --- | --- |
| `B4_w2.0_lam0.5_lr2e-05_ep3` | text | 0/4 | 0/4 (all `straddles`) |
| `B4_w2.0_lam0.5_lr2e-05_ep3` | image | **3/4** | **3/4** |
| `B4_w4.0_lam0.5_lr2e-05_ep5` | text | 0/4 | 0/4 (all `straddles`) |
| `B4_w4.0_lam0.5_lr2e-05_ep5` | image | **4/4** | **4/4** |

For ep5 the image-route retain-same shortfall is −0.0980 — **40.0 queries of
408** — against a between-seed sd of 0.0120, about 5 queries. The ratio of spread
to shortfall falls from 1.96 on the text route to 0.12. Stage 1b could only say
"the seeds straddle the anchor and the noise exceeds the gap"; Stage 1c can say
"every independent seed fell below it, by more than the spread". Both parents
still fail, and now the failure is not inside the noise.

Two guards make this a re-scoring rather than a re-deciding. The analyzer
recomputes the frozen four-number floor for every state and **refuses to write**
unless the stratified verdict restricted to the text stratum equals the filed one
number by number and flag by flag — it compared 36 Stage-1 numbers and 8
Stage-1b numbers with zero problems. And because the image route is the one whose
correctness flips when batch composition changes, the freeze binds the sha256 of
all 19 prediction parquets *and* their sidecars, and verifies they are one
measurement contract (`image_batch_size 1` throughout); the seed-42 replicates are
read from Stage 1's directory and the other three from Stage 1b's, so a contract
difference between them would have put the headline comparison on two
instruments.

Disclosed: this basis was identified **after** Stage 1b was scored, so it is not
blind. Two things keep it from being a rule chosen for its answer. It was found in
the dataset's structure — the image families cover identical association sets
(387/387), share zero templates with the text ones, and are emitted by the sealed
metric — and every field of the basis report is derived from `queries.parquet`,
`associations.parquet` and the frozen partition with **no prediction read**, so
`--check-only` re-derives the whole document and a reviewer can confirm that
without loading a single model output. And its direction is against interest: it
adds four conditions and enlarges the failure it reports. Pooling, stratifying and
dropping the image route were all computed; stratifying is frozen, and the pooled
numbers are in the report so that choice is visible.

The freeze refuses to be written once the analysis report exists — the analogue of
Stage 1b's post-training refusal for a study that trains nothing, since here the
thing a rule must not postdate is the score. It was amended twice before any score
was filed, both recorded with reasons and both moving only protocol-path hashes.
Nothing frozen was edited: `retention_selection.py` and the sealed
`hierarchy_metrics.py` are imported and called, and Stage 1c is four new protocol
paths plus its own freeze. 93 tests, and 8 mutations of the module and the freeze
— image failures made non-disqualifying, the consistency gate turned into a
no-op, each cross-check removed or inverted, the pooled families admitted to the
floor, a contract conflict made harmless, the text stratum un-delegated — were
each caught.

### Stage 2: the anchor was the defect, and the mechanism was chosen by measurement

Stage 1c ended with 0/8 eligible and a floor whose sensitivity was no longer the
binding limit, and this README had already said what Stage 2 therefore faced: the
same anchor would return the no-op again, so the anchor itself had to be
reconsidered, and that was a decision to make **before** Stage 2 was frozen rather
than after it was scored. Stage 2 does both things it preregistered — it moves the
anchor and it changes the objective — and it derives the second from a measurement
of adapters that already existed, at zero GPU cost.

**The anchor.** `D_G`, the criterion this project has used since Iteration 9,
measures distance *to MG*. Under the floor Stage 1 froze, whose reference point is
the no-op `B0`, **MG itself clears only 2 of the eight numbers**, its worst
shortfall −0.087 on `image.retain_other_entity_image.entity_macro`. A floor whose
reference point the oracle fails disqualifies the state the criterion is aiming at,
so it can only rank candidates by how little they did — which is what Stage 1
observed, selecting the no-op. The floor is therefore re-anchored at MG.

Two checks keep that from being a rule moved to change an answer. Re-anchoring
**changes no filed verdict**: all eight Stage-1 candidates are ineligible on `B0`
and on MG alike, 0/8 either way, so nothing already committed moves. And the new
floor **does not licence inaction**: `B0` clears only 6 of 8 under the MG anchor,
because MG sits above the no-op on both image retain-same numbers, so declining to
unlearn is not eligible either. `B0`'s eight values are still computed and reported
beside the primary gate on every candidate, so a reader can see both rules at once.
Disclosed: `floor_check_stratified` labels its baseline field `b0` and its
comparison string `candidate >= b0 - eps` whichever baseline is passed, and that
module is hash-bound by the Stage-1c freeze, so under Stage 2's primary gate the
field named `b0` holds MG's values. The names cannot be corrected; the report says
so explicitly instead.

**The mechanism.** `build_mf_reference_logprobs.py --calibrate` loaded all eleven
adapters the repository already had — MF, MG, `B0` and the eight Stage-1
candidates — and measured each one's KL from MF on the 183 fit-half prompts the
anchor would pin, plus each one's NLL on all three knowledge groups, reading **no
retention number, no prediction parquet and no evaluation query**. Joined
afterwards against the eight numbers Stage 1c filed:

* the quantity Stage 2 was preregistered to control **does not determine
  retention**. MG sits 0.0296 nats from MF and `B4_w4.0_lam0.5_lr2e-05_ep5` sits
  0.0308, yet their image retain-same row-micro differs by 0.1054. Two further
  pairs within 0.015 of each other in drift differ by 0.1496 and 0.1692 in
  retention, against a floor that resolves 0.0044.
* the anchor's own optimum is **the wrong place**: `KL = 0` is MF, which performs
  no unlearning at all, while the oracle sits at 0.0296. A strong beta drives a
  candidate toward MF rather than toward MG.
* what *does* rank them is how far the suppression term drove the fine fact.
  Spearman rho between `fine_target` NLL and image retention is **−0.976 and
  −1.000**, against −0.571 to −0.619 for the drift.
* and the overshoot buys nothing: rho between `fine_target` NLL and `D_G` is
  **−0.024**. MG sits at 1.3643 nats, the nearest candidate at 1.7083, the rest
  running to 12.7768 — every candidate pushed the fine fact further than the state
  it is scored by distance to.

So Stage 2 trains **both** mechanisms and lets the frozen floor decide, because an
eight-point observational argument is not an experiment and the mechanism Stage 2
was preregistered with deserves to be run rather than argued away. `B5` bounds the
ascent — `-λ·min(NLL, cap)`, gradient exactly zero above the cap — which is the
coordinate the measurement says governs. `B6` is the MF-preservation anchor at
`beta = 13.642`, derived as the strength at which `beta · KL` equals the
suppression term's magnitude at the end of the incumbent's training
(0.5 × 6.6061 / 0.242124), so it is read from committed training summaries and
never from a retention number.

Seven rows train: four caps (0.5, 1.0, 2.0, 4.0 nats) at the incumbent budget;
the 2.0-nat cap repeated at eight epochs, because Stage 1 coupled "how far to
suppress" to "how long to train" through the epoch count and a cap decouples them;
and two anchor rows — `B6` with replay *replaced* by the KL, and `B6R` with the KL
*beside* it, spent through `anchor_weight` rather than a second group entry so the
epoch keeps its 363 micro-batches and the schedule under test does not move. The
caps are swept, not derived from the oracle: the training procedure never reads MG,
whose own level is reported afterwards and bracketed by the range. Everything else
is the incumbent recipe — λ 0.5, target-level weight 1.0, replay weight 1.0, lr,
budget — and the incumbent row itself is the sweep's `cap = ∞` point, reused from
Stage 1 for the price of a sha256.

**Why the loop is copied, and how the copy is policed.** `train_unlearning` is
hash-bound and cannot grow a mode, so `train_with_preservation` is a second copy of
it — and a second copy can drift, which would appear as a difference between Stage
1 and Stage 2 having nothing to do with either mechanism. Two independent checks
bound that: `--control` trains the incumbent objective three times in one process
under one pinned `PYTHONHASHSEED` — A through the frozen loop, A2 through the
frozen loop again, B through the new loop with every group in a plain `sft`/`gd`
mode and neither cap nor anchor — and **refuses to train a single candidate**
unless the gap between A and B is no wider than the gap between A and A2, with
bitwise equality demanded outright when that floor is exactly zero, compared per
tensor rather than by digest because a digest says "different" and cannot
distinguish float noise from a different objective; and 102 tests drive both loops
against one stub model on CPU and assert bit-identical parameters *and* identical
epoch summaries, so a divergence is caught without a GPU.

**The control was first frozen with a different criterion, and it refused.** That
version compared one run of the new loop against the adapter Stage 1 *filed*, by
sha256 over the whole directory. It ran 1187.8 s over 230 optimizer steps and
reported different digests — with an identical recipe in all fifteen fields, an
identical init adapter, and `adapter_config.json` differing only in the *order* of
the same seven target modules. That order is not reproducible:
`PeftConfig.target_modules` is a Python `set`, and Python randomises the string
hash seed per process unless `PYTHONHASHSEED` is exported, so the order PEFT
serialises it in — and the LoRA injection order behind it, which decides which
module draws which dropout mask — varies run to run. Four seeds were measured to
give four different orders over the same seven modules, a pinned seed gives one
order every time, and the filed order matched none of the seven seeds tried, so
the process that wrote it is not recoverable and no later run can match its bytes.

To establish that this indicted the criterion rather than the new loop, the
**frozen** loop was re-run unedited on the same spec from the same init adapter. It
did not reproduce the filed adapter either, and missed it by about as much as the
new loop had: max per-tensor weight gap 1.726e-03 for the frozen re-run against
filed, 1.812e-03 for the new loop against filed, 1.800e-03 between the two re-runs,
over 256 tensors. Three gaps of one order, so the new loop is no further from the
filed adapter than the frozen loop is from a file the frozen loop itself wrote.
Both runs are kept under `outputs/superseded/iter12_stage2_control_v1/`, whose
README records the digests, the three gaps and the three `target_modules` orders,
and whose small files are committed so the disclosure can be checked rather than
believed. The filed adapter is still loaded and its gap still reported beside the
gate; it is simply not what the gate reads, and `train_iter12_stage2.py` refuses to
run the control at all without a pinned seed because a control under a randomised
one measures the interpreter.

The freeze was amended to record all of this, and the amendment touches no
criterion: the anchor, its values, the eight numbers, the epsilon, the tie-break
and the grid are byte-identical to what was frozen, which is checkable because the
run that refused produced no candidate, no prediction, no retention number and no
score — there was no result for any of them to have been fitted to. The retired
criterion was unsatisfiable by construction, so it tested the hash seed rather than
the two loops, and a gate that cannot pass says nothing when it fails. Because
widening what an amendment may touch needs something holding the other side, eight
tests now pin the control *structurally*, by parsing the script that holds it: the
hash seed is checked before anything trains and an unpinned one is refused by code
that is executed rather than merely inspected, three runs happen with the frozen
loop twice, each side of the gate is built from the right pair of them, the gate
has exactly three branches and never assigns a bare `True`, the filed adapter never
reaches it, and `main` refuses to train both when the gate did not pass and when
the control was never run. The freeze lists all eight by name under
`faithfulness_control.what_pins_this_in_the_suite`, and a test reads that list back
against the test file, so the disclosure is checkable rather than descriptive. The
anchor's correctness rests on three guards that run
at every step: the NLL recomputed from the extracted log-probabilities must equal
the loss the model returned (a causal LM predicts `labels[i]` from `logits[i-1]`,
and an off-by-one there is invisible in the numbers — it just anchors every
position against its neighbour's reference distribution; measured agreement
7.5e-09), the cached supervised tokens must equal the current ones, and the cache's
recorded MF digest must match the adapter the run starts from. The reference is
cached exact rather than top-k truncated — 1,489 positions × 248,320 vocabulary =
1.479 GB in fp32 — so no second model is resident beside a co-tenant.

**The control ran, and passed.** Three trainings of the incumbent objective, two
epochs and 92 optimizer steps each, in one process under `PYTHONHASHSEED=0`. The
frozen loop against itself — the measured noise floor — landed max **5.810e-04**
per tensor, mean 1.649e-04. The frozen loop against the new one landed max
**5.032e-04**, mean 1.384e-04: narrower on both, a ratio of **0.866**, so the two
loops are not distinguishable at the resolution this stack has. All three runs
share one `target_modules` order, and it is not the filed one. The filed adapter
sits **2.820e-03** away, **4.85×** the in-process floor and against 1.726e-03 for
the superseded cross-process measurement, so pinning the seed cut the run-to-run
gap **3.0-fold** and what is left is GPU floating-point nondeterminism.

Two things about that result are worth stating because neither was predicted. The
floor is **not zero**: one process and a pinned seed do not buy bitwise
reproduction on this stack, and the gate passed on its magnitude branch. Had the
control assumed a zero floor — which is what the CPU stub tests show, and what the
design expected — it would have refused a second time on a criterion the hardware
cannot meet. And the gate is a comparison of two draws from the same noise
distribution, so it can fail by chance; that it came out at 0.866, narrower on the
mean as well as the maximum, is the reassuring part, and the number is filed rather
than paraphrased.

**Disclosed: the control's own summary sentence is wrong, and cannot be
corrected.** `CONTROL.json`'s `detail` field says the two loops "produced
bitwise-identical adapters under a pinned hash seed", and the log line says they
"agree bitwise". They do not: `gate.gap_frozen_vs_new.bitwise_identical` is
`false` in the same document, and those structured fields — with `gate.required` —
are the authoritative ones. The sentence was written for the zero-floor branch and
was not made conditional when the magnitude branch decided. It cannot be fixed
where it was written: `train_iter12_stage2.py` is one of the eight protocol paths
the freeze binds by sha256, and the freeze locked the moment the control wrote its
first adapter, so editing it would fail `--check-only` and the selector refuses to
run when the freeze does not match. The correction is filed in
`data/reports/mllmu_iter12_stage2_control_reading.json` instead, which also copies
the control's numbers out of the gitignored checkpoint directory so a clone can
check them, and is regenerated by `scripts/annotate_iter12_stage2_control.py` —
whose prose is computed from the numbers beside it, so it cannot drift away from
them, and which refuses to overwrite a filed reading.

**Disclosed: a row interrupted mid-training looks trained, and the chain catches
that rather than preventing it.** `preservation_trainer` creates `<row>/adapters`
on the way in and fills it only at the end, and both the lane planner and the
trainer's own resume path skip a row when that directory *exists*. So a chain
stopped between the two leaves a directory that looks exactly like a trained row
and holds nothing, and a relaunch would skip it and never retrain it. The skip
cannot be tightened onto the adapter file, because it is in
`train_iter12_stage2.py`, which the freeze binds. What catches it is the chain's
`check` phase: it gates afterwards on `adapters/adapter_model.safetensors` for all
seven rows that need a GPU — enumerated from `stage2_grid` rather than from the rows
the planner happened to train, which is what makes it a completeness gate and not an
echo of the skip — and exits **4** naming the gaps before `gen` spends any GPU. An
interrupted relaunch therefore stops loudly instead of filing a report over a
partial grid, and the recovery is to remove the empty directory and relaunch — not
`--force`, which retrains the row but also re-runs the control's three trainings,
since it means "ignore what already exists" rather than "retrain this one row".
Two tests hold the pair: one asserts the
directory really is made *before* it is written, so the empty directory stays
reachable and cannot be quietly forgotten, and one asserts the gate is on the file,
over the whole grid, and still exits. A `check` "simplified" to match the skip is
the one edit that would turn this from a loud refusal into a silent partial grid.

The freeze binds eight protocol paths, seventeen modules Stage 2 imports and must
not edit, and eighteen committed data paths, plus — separately, because they are
gitignored — the sha256 of the three prediction parquets Stage 2 *reuses* (`B0`,
MG and the incumbent, read from Stage 1's directory rather than regenerated) and of
the cache. The generation contract is **inherited** from the Stage-1 freeze rather
than restated, and the selector refuses a run whose batch layout differs, because
the image route is precisely the one whose correctness flips with batch
composition. It refuses to write a report unless the whole grid has predictions,
recomputes all eight numbers for each reused state and refuses unless they equal
the values Stage 1c filed, and refuses any path resolving inside the sealed 11C
confirmation. Its refusal is keyed on adapters and evaluated before any flag, so
`--refreeze` cannot reach past it — the control's adapter included, since a control
run under an unfrozen loop proves nothing about the frozen one. It was amended four
times, all four recorded with reasons, all four recording that no adapter existed
under the root it governs, and none moving the criterion: once for a self-describing
field; once to correct four cache figures in a docstring that had been estimated
before the cache was built and was wrong on every one of them (1,434 rows of
248,077 at 1.42 GB, against the 1,489 rows of 248,320 at 1.479 GB the sidecar
records), which a test now reads back from the committed sidecar because an estimate
in a docstring is otherwise compared against nothing; once to replace the
faithfulness control's comparison basis; and once to name in the freeze itself the
eight tests that hold that replacement, and to correct a count the previous entry
got wrong. The third was made only after the failed control's adapter had been
*moved* out of the root the freeze governs, so its precondition was genuinely
satisfied rather than bypassed, and the moved run is committed so the reason is
inspectable. 32 mutations — the clamp removed, capped rows made to descend, the
capped value booked as the group's NLL, the accumulation tail normalised by the full
window, supervised positions read one place late, the alignment and NLL guards
disabled, an unknown cache row served from a neighbour, the anchor permitted on a
target group, the seal compared as a string so `..` walks past it, the floor gate
made to pass whatever it is given, the control's adapter made invisible to the
freeze, four grid edits, eleven aimed at the redesigned control (the gate made
to pass whatever the two loops did, the bitwise demand conditioned away, the gate
fed the filed adapter so the unsatisfiable criterion returns, the noise floor
measured against the wrong run, A2 dropped so the floor is assumed rather than
measured, the hash seed not pinned before training and then accepted rather than
refused, a failed control and an unrun control each no longer stopping the
candidates, and the freeze's disclosure and its named test list each quietly
removed), and four aimed at the annotator that files the control's reading (the
gate's verdict computed the wrong way round, any number of distinct
`target_modules` orders counted as one, the pinned order reported as the filed one
whatever it is, and the gate's prose claiming the mean went whichever way it did
not) — were each caught, with the freeze's own hash check deselected so that
every mutation had to be caught by a test exercising the behaviour rather than by
the file having changed.

The last four were held back until every GPU lane had finished, because eleven of
the earlier mutations edit `train_iter12_stage2.py` and two of the training lanes
start a fresh `python train_iter12_stage2.py` whenever a card frees up: a lane that
loaded the file inside the ten-second window a mutation was applied would have
trained a candidate under edited code, and nothing downstream would have noticed,
because the adapter would have looked perfectly well formed. One of the four then
**survived**, and the reason is a limit on comparing an output against a filed copy
rather than a limit on the suite. `--check-only` rebuilds the reading and diffs it
against the committed one, but `len(distinct) >= 1` and `len(distinct) == 1` agree
whenever there is exactly one distinct order — which is what this control has — so
no comparison against these numbers can separate them, and the claim that makes the
three runs comparable at all was unpinned. Three tests now drive `build()` from a
synthetic control in a temporary directory whose three runs *disagree*, in both
directions and against the filed adapter's order, so the field is pinned rather
than merely falsifiable. They are built from scratch rather than copied out of
`data/checkpoints/`, so they run in a clone too, where the gitignored directory the
reading was copied from does not exist and a skip would have hidden exactly the gap
they close.

### Stage 2 ran: the MF-preservation anchor missed by fourteen queries, on a floor with no margin

All seven rows trained, generated and scored under the freeze, and `selected` is
**None**: 0 of 9 candidates clear the primary MG-anchored floor on all eight numbers,
1 of 9 clears the `B0`-anchored floor reported beside it — and that one is `B0`
itself, which under that anchor is compared against its own values. The Stage-3 gate
opens.

The result is not "nothing worked", and the difference from Stage 1 is the point.
`B6_beta13.642_lam0.5_lr2e-05_ep5`, the MF-preservation anchor, fails **3** of the
eight numbers where the four ascent caps fail between four and seven of them — four,
six, seven and seven, and not in cap order — and the uncapped Stage-1
incumbent fails 7. It clears MG on all four text-route numbers, and its fine-target
NLL is 0.6857 nats — 0.6469 for `B6R`, against 0.7832 to 2.9924 for capped ascent —
so the anchor holds the unlearning objective *and* text retention at once rather than
trading one for the other. All three of its failures are image-route:
`image.retain_same_entity_image.entity_macro` −0.0339, its `.row_micro` −0.0123, and
`image.retain_other_entity_image.row_micro` −0.0263. At the denominators Stage 1c
measured — 408 retain-same and 76 retain-other queries per route, so 816 and 152 rows
across train+val — those are **10 rows of 816 and 4 rows of 152**. The frozen floor's
margin is 0.0 and its epsilon 1e-09, so four queries decide it.

The cap sweep moves in the direction the calibration predicted, which is the
direction that disqualifies it. Loosening the cap raises the fine-target NLL
monotonically (0.7832 → 1.1852 → 1.8007 → 2.3548 at caps 0.5 → 1.0 → 2.0 → 4.0 nats)
and lowers image retain-same row-micro (0.4510 → 0.4216 → 0.3505 → 0.3750), while
`frac_at_cap` falls from 0.80 to 0.2667 — a looser cap binds on fewer tokens, so more
ascent is actually applied, and the retention cost tracks the ascent rather than the
nominal cap. Repeating the 2.0-nat cap at eight epochs instead of five takes the NLL
to 2.9924 with `frac_at_cap` 0.9111 and image retain-same to 0.3260, the worst in the
grid: suppression and budget are **not** decoupled, and buying more of the first costs
the second.

So Stage 1 returned the no-op because its anchor disqualified every state including
the oracle, and Stage 2 re-anchored at MG and returns a mechanism that nearly clears.
The shortfall is now three image-route numbers totalling fourteen queries rather than
a floor no state can pass — but the measurement limit is unchanged and is still the
binding one, 76 retain-other queries per route against pilot-100's ceiling of 228. The
honest reading is that the MF-preservation anchor is the mechanism Stage 3 should
carry forward and that this dataset cannot yet resolve whether it clears the floor.

**The result is not the frozen selector's own, and the record filed beside it says so
in a field named `what_still_cannot_be_claimed`.** `select_iter12_stage2.py` is one of
the eight protocol paths the freeze binds by sha256, and it crashed twice.

1. Lines **550** and **579** call `rt.pooled_all_routes()` with three positional
   arguments where the frozen callee requires four. The fourth, `probe_entities`, is
   already bound at line 499 from `mllmu_iter12_retention_probe.json` — one of the
   eighteen data paths the freeze binds — and the sibling calls two lines away pass it
   correctly, so there is exactly one value that can go there and it is the protocol's
   rather than the repair's. It is in the reused-states loop, which runs *before* any
   generation, so every lane died having written nothing. The field it fills floors
   nothing: `floor_vector` reads only
   `by_route["routes"]`, and the selector copies the field into a report row at line
   227 without passing it to `summary_vector`, `distance_to_mg`,
   `floor_check_stratified` or `rank_key` — asserted by walking the argument trees of
   those eight calls, not by reading the code and agreeing with it.
2. With that out of the way the run reached line **630** and died on `KeyError: 'MG'`,
   because the anchor won its own selection. `eligible` and `ranked` at lines 232–233
   omit the `cid != "MG"` filter the same file applies at 294, 299, 304 and 331, and
   MG's distance to MG is 0 by construction, so it topped the ranking whatever the
   seven candidates measured; line 630 then looked the winner up in a dictionary the
   same function had already excluded it from. The patch removes `"MG"` from the rows
   handed to `build_report`, which receives the anchor through three parameters of its
   own — `reused_eight`, `reused_vec`, `mg_vec` — so the floor, the anchor values and
   the reported MG block are computed from inputs it does not touch, and dropping one
   key from a sort's input cannot reorder the rest. **That patch decides which name
   lands in `selected`**: unpatched it is `"MG"`, a reference state that was never a
   candidate and that would have been selected whatever the data said. It is defended
   by the freeze's own rule being stated over a *candidate* and naming MG as the
   anchor, by MG not being a row of the frozen grid, and by the selector's own report
   declaring it `excluded_from_the_ranking` with the reason that ranking it would let
   the reference state win its own selection — but it is a judgement made by a script
   written after the protocol was frozen, and it is recorded as one rather than buried
   in a field that looks computed.

No frozen byte was edited. The freeze refuses amendment once any Stage-2 adapter
exists and seven do, and the selector verifies the freeze as its own Step 1 before
anything expensive, so an edited copy would refuse to run — that refusal is the
control, and it was honoured rather than worked around: two module attributes are
rebound for the duration of one `main()` call and both are restored in a `finally`.
`scripts/repair_iter12_stage2_selection.py` follows the Stage-1b precedent, and proves
before either patch is allowed near the model that the bound paths still match the
freeze, that those two sites are the only ones, that **every** call the selector makes
into a bound module has the right arity (27 audited, 0 unresolved, the two known sites
the only tolerated problems), that the patched field cannot reach a decision, and that
the first crash reproduces unpatched in a subprocess with the report hashed *and
timed* either side. The scoring run rewrote the selection report **byte-identically** —
the five recorded scoring invocations left one distinct sha256 between them, and the
first of them is the one that created the file — which is possible because the report
carries no timestamp of its own, and which is filed as the determinism evidence it is
rather than as "nothing changed".

A third defect sits in the freeze rather than the selector and is **not** patched.
`_git` is defined once, as `_git(repo_root, *args)`, and both
`freeze_iter12_stage2.py` (lines 305–306) and `freeze_iter12_route_stratification.py`
(294–295) call it as `_git("rev-parse", "HEAD")`, binding a git subcommand to
`repo_root`. Git is then run inside a directory called `rev-parse`, which does not
exist, and the helper's `except OSError: return None` turns that into silence rather
than an error. Both freezes therefore record `git_commit: null`; and because
`git_dirty` is written as `_git("status", "--porcelain") is not None`, the same `None`
makes it `false`, so each asserts a clean tree. The Stage-2 tree was dirty: 40 of its
43 bound paths match HEAD reconstructed from its own `frozen_at_utc` (`dae584a`), and
the three that differ are exactly the three its amendment log names as re-hashed. For
Stage 1c, 17 match and the 5 absent are all present in `e94d1f8`, the commit that
filed it. Both fields are in `VOLATILE_FIELDS` and `verify_freeze` reads neither,
which is why the freezes still verify and this study ran at all; and each freeze script
binds *itself* among its own `protocol_paths`, so editing either would make the
selector refuse to run the study it governs. What a freeze exists to answer — which
bytes, from which commit, in what tree state — is, for these two, nothing, and the
reconstruction is labelled a reconstruction rather than filed as provenance.

It also does not pretend to work in a checkout that has no history to reconstruct
from. A depth-1 clone holds one commit, dated after every freeze, so
`rev-list --before=<frozen_at_utc>` answers with nothing. The first version of this
raised, which failed eight tests in a shallow clone for a reason that was about the
clone and was indistinguishable from the repair being broken. It now returns a
disclosure naming the timestamp it could not resolve, and carrying none of the
fields whose values would read as findings — because `the_tree_was_therefore_dirty`
is `bool(differing or absent)`, so measuring nothing computes to `False` and files
as a measurement that found the tree clean: the opposite of the finding, arrived at
by comparing nothing. The workflow clones with `fetch-depth: 0`, so CI takes the
reconstructing branch, and the disclosure is driven deterministically anyway by a
timestamp before the first commit — a branch no test reaches is a branch a mutation
can rewrite unnoticed.

72 tests in `tests/unit/test_iteration12_stage2_repair.py` hold all of this, torch-free
and GPU-free; the arity audit that would have caught the first defect before the grid's
generation ran — 96.72 minutes for the seven lanes, from the chain log's own two
timestamps — is one of them, and it pins 27 audited calls
because a first version resolved half the call sites, reported no problems, and was
worse than no audit at all for looking like one. 31 mutations — the patched name
changed, the anchor named as a candidate that exists, a three-argument call declared
sufficient, `rank_key` dropped from the decision-bearing set, the key tokeniser
reverted to substring matching (which reports a timestamp in a report that has none,
because "candidates" contains "date"), the disclosure made not to refuse once the
defect is fixed, a mis-bound `_git` call no longer recognised, the amendment log split
on the wrong dot, a byte-identical rewrite reported as a change, both chain-log
refusals returned to silent `None`s, every call reported as supplied including the
callee's own, one of the two patches no longer restored, the scoring lane allowed to
skip the crash proof, a generation lane pointed at the frozen selector directly, the
`check` phase weakened from the adapter file to the directory, a checkout with no
history reporting the tree it never looked at and the same disclosure claiming it had
compared the bound paths, and nine edits to the filed record itself — were each
caught, 0 survived, and every file was restored byte-identically. `--check-only`
re-derives the record from the frozen bytes, the
committed evidence log and both earlier filings of itself, and refuses a record whose
anchor proof is null — a gate verified against the filing it replaced rather than
assumed to bite. Those two earlier filings are committed under `outputs/superseded/`
with a measured key-set diff against this one: the first filed its anchor proof as a
null where a byte-identical reproduction belonged, and the second said the first was
unreachable from a clone when it was about to become reachable. The evidence log and
the chain log are committed at the paths the repair reads them from, because a clone
without them would pass `--check-only`'s invocation checks *vacuously* — an empty log
holds no failed invocation and no unexpected call site either.

One limit on that log is worth stating rather than leaving to be discovered, and it is
why the chain's *other* log is committed beside it. `iter12_stage2_nohup.log` is
whatever the caller redirected stdout to, and the chain is launched with `nohup > `,
which truncates — so what is committed is the **last** chain run's stdout, not a
history of them, and the relaunch that carried the repair overwrote the run in which
the seven generation lanes died on the first defect. That log therefore evidences the
*second* crash, the `KeyError: 'MG'` the record quotes, and not the first. What
evidences the first is stronger anyway: the repair re-reproduces it on demand,
unpatched, in a subprocess, with the report hashed and timed either side, so it is a
repeatable demonstration rather than a log line a later run can delete.
`the_second_crash_as_it_happened()` refuses rather than returning `None` if the log is
ever missing or holds no `KeyError:`, which is the same rule — a parser that cannot
parse must not be indistinguishable from a log that says something else, because the
first version of it reported `so_the_crashed_run_filed_nothing: false`, the opposite of
the truth.

`iter12_stage2_chain.log` is the one that keeps history: the lane script writes it with
`tee -a` and `>>`, so it holds every phase of every run — the control passing on real
bytes, seven training lanes, the `check` phase finding all seven adapters, seven
generation lanes launched at 12:31:19, `received SIGTERM — stopping 7 lane(s) before
releasing the lock` at 12:33:37 and `lanes stopped; releasing the lock and exiting 143`
four seconds later, and both score failures. That is the only committed evidence the
hardened stop path ran for real rather than only against a stub, and it is why the
generation figure above is quotable at all: `gen: one lane per candidate` at 12:50:51
and `gen: all generation lanes finished` at 14:27:34 are two of its lines, 96.72
minutes apart.

## Iteration history

`git log --oneline` is the authoritative narrative and its commit messages carry
the measurements. In brief: **0–3** schemas, hierarchy engine, dataset adapters
and validation gates · **4–7** smoke experiment, reference states MF/MG/MN,
frozen evaluator · **8–9** the failure taxonomy and the first MF→MU unlearning
baselines · **10–11** pilot-100, SalmuBench cross-dataset comparison ·
**11R** the visual-split repair (the image route had been serving the photograph
training consumed), provenance-gated reuse, sharded regeneration of all evidence
· **11C** the confirmation: power analysis, protocol freeze, photograph
selection, the 1,209-query split, two rounds of blocking review repairs (five
findings in 11C-5R, four in 11C-5R2), the dedicated scorer and analyzer, and the
scored result they produced — both preregistered claims confirmed · **11C-5R3**
the post-run audit of what that scoring run did not record, and three README
claims corrected · **12** retention, as a separate exploratory study: the
fit/probe partition that makes a retention floor measurable at all, a selection
protocol frozen before any candidate was trained, and Stage 1 run under it —
which disqualified all eight replay settings and selected the no-op, leaving the
anchor, not the mechanism, as the thing Stage 2 has to reconsider · **12b** the
two near-misses re-run at four seeds each, frozen first, after a free
measurement showed the floor's probe queries are entirely text-route and so
unaffected by the generation instability that does move the image route — and
run, to a second negative result. The determinism control passed with **zero**
differing rows across all 4,518 predictions and all seven recorded fields, so
the anchor is a point; neither parent then cleared the floor on a four-seed mean
(2/4 and 0/4 of the four numbers), every one of the eight straddled the anchor,
and the seed sd on retain-other row-micro (0.0225, 0.0515) exceeded the very
shortfall it was adjudicating (−0.0165, −0.0263). The binding limit is therefore
measurement power, not the anchor and not the mechanism. The frozen analyzer
crashed on a field that is not decision-bearing and was completed by a separate
script that leaves every frozen byte untouched. · **12c** the power limit was
attacked from the dataset instead of from the GPU: the probe's associations turn
out to back image-route retention families that the sealed metric has always
emitted and the frozen floor never read, and their predictions were already
generated, so the floor was extended to eight numbers at **zero GPU-hours**. The
verdict did not move — 0/8 eligible on eight exactly as on four, and the text
stratum reproduces all 36 filed Stage-1 numbers and all 8 Stage-1b numbers
exactly — but the second instrument is more sensitive: the image route loses
−0.1373 against the text route's −0.0558 on retain-same across the eight
candidates, and MG loses almost nothing on it. Of the sixteen numbers now
measured, the same seven have every seed below the anchor *and* a spread smaller
than the shortfall they adjudicate; of the original eight, none had either.
Stratifying enlarged no denominator; pilot-100's ceiling is 228 retain-other
queries, so the binding limit is still the dataset. · **12d** Stage 2 took the
point Stage 1c left — that the anchor, not the sensitivity, was what returned the
no-op — and acted on it: MG, the state `D_G` measures distance *to*, clears only
**2 of 8** numbers under a floor referenced to `B0`, so the floor was re-anchored
at MG, changing **no filed verdict** (all eight Stage-1 candidates are ineligible
on both anchors) and not licencing inaction (`B0` clears 6 of 8 under the new one).
The mechanism was then chosen by measuring the eleven adapters the repository
already had, at zero GPU cost: the drift the preregistered anchor controls does not
determine retention (MG and `B4_w4.0` are 0.0012 apart in drift and 0.1054 apart in
image retain-same, against a floor resolving 0.0044), its optimum is MF rather than
the oracle, and what *does* rank the candidates is how far the suppression term
drove the fine fact (rho **−0.976 / −1.000** with image retention versus −0.571 to
−0.619 for the drift, and **−0.024** with `D_G`, so the overshoot past MG's 1.3643
nats buys nothing). Both mechanisms are trained anyway and the frozen floor
decides: seven rows — four ascent caps, the 2.0-nat cap repeated at eight epochs to
test the decoupling of suppression from budget, and the MF anchor alone and beside
replay at `beta = 13.642` derived from committed training summaries. Because
`train_unlearning` is hash-bound and cannot grow a mode, the new loop is a second
copy of it, policed twice: a GPU control that refuses to train anything unless the
two loops agree, and 102 tests that drive both loops against one stub model and
require bit-identical parameters. **The control was first frozen as
byte-reproduction against the incumbent's filed adapter, ran, and refused** — and
the refusal turned out to be about the criterion, not the loop: `PeftConfig
.target_modules` is a Python `set`, so the order PEFT serialises it in, and the
LoRA injection order behind it, vary with the per-process hash seed, and the filed
order is not recoverable. Re-running the *frozen* loop unedited missed those bytes
by about as much as the new loop had (1.726e-03 against 1.812e-03 max per-tensor
gap), so the control now trains the incumbent objective three times in one process
under a pinned `PYTHONHASHSEED` and gates the across-loop gap against the
same-loop gap it *measures*, with the filed adapter still reported beside the gate.
Both runs are kept under `outputs/superseded/iter12_stage2_control_v1/`. 32
mutations caught, 0 survived — four of them aimed at the annotator and held back
until the GPU lanes had finished, one of which survived a comparison against the
filed reading until a synthetic control pinned what that comparison could not see.
The freeze was amended four times; the third only
after the failed control's adapter was moved out of the root the freeze governs, so
its precondition was genuinely satisfied rather than bypassed, and the fourth
corrects a count the third got wrong — it said seven structural tests hold the
widened seal, and there are eight, because the test pinning which pair of runs each
side of the gate is built from was written after that reason was drafted. The
criterion did not move: the anchor, its values, the eight numbers, the epsilon, the
tie-break and the grid are byte-identical to what was frozen, which the run that
refused could not have influenced because it produced no candidate, no prediction,
no retention number and no score. · **12e** Stage 2 ran and returned a second
negative with far more in it than the first: `selected` is **None**, 0 of 9
candidates clear the MG-anchored floor on all eight numbers, and the Stage-3 gate
opens. The MF-preservation anchor is the best row in the grid and not narrowly — it
fails **3** of 8 where the four ascent caps fail four, six, seven and seven between
them and the uncapped Stage-1 incumbent fails 7, it clears MG on all four text-route
numbers, and it holds
the lowest fine-target NLL of the trained rows (0.6857; `B6R` 0.6469) against
0.7832–2.9924 for capped ascent. All three of its failures are image-route and
amount to **10 rows of 816 and 4 rows of 152** against a floor whose margin is 0.0.
The cap sweep costs retention monotonically as it loosens (image retain-same
row-micro 0.4510 → 0.4216 → 0.3505 → 0.3750 while `frac_at_cap` falls 0.80 → 0.2667,
so a looser cap applies *more* ascent), and repeating the 2.0-nat cap at eight epochs
is the worst cell in the grid — suppression and budget are not decoupled. The result
is **not the frozen selector's own**: it crashed twice, on a missing fourth argument
at lines 550 and 579 to a field that floors nothing, and then on `KeyError: 'MG'` at
line 630 because `eligible` and `ranked` omit the anchor filter the same file applies
four times elsewhere, so the anchor won its own selection. A separate repair script
patched two names for one `main()` call and restored both in a `finally`, editing no
frozen byte; the second patch **does** change `selected` and the record filed beside
the report says so in a field named `what_still_cannot_be_claimed` rather than
leaving it to inference. A third defect, disclosed and not patched, is in the freezes
themselves: `_git(repo_root, *args)` is called as `_git("rev-parse", "HEAD")` by both
`freeze_iter12_stage2.py` and `freeze_iter12_route_stratification.py`, so the
subcommand binds to `repo_root`, the helper's `except OSError` swallows it, and both
filed freezes record `git_commit: null` and `git_dirty: false` — the second falsely,
since 40 of the Stage-2 freeze's 43 bound paths match HEAD reconstructed from its own
timestamp and the three that differ are the three its amendment log names. Both fields
are volatile, so no verdict depends on them, and each script binds itself, so neither
can be edited. 72 tests, 31 mutations caught and 0 survived, and the five recorded
scoring invocations left one distinct sha256 of the selection report between them.
The provenance reconstruction also refuses to fabricate an answer in a checkout with
no history: a depth-1 clone gets a disclosure naming the timestamp it could not
resolve, not a `the_tree_was_therefore_dirty: false` computed from nothing.
