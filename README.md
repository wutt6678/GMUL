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
pytest tests/unit -q          # 1574 tests, CPU only, ~3.2 min on the artifact box
```

The same 1574 pass with `torch`, `transformers`, `peft`, `accelerate` and
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
the adapters and fetched the pool — the machine that ran the three GPU passes.
Measured: **1496 passed / 78 skipped / 0 failed** in a bare clone with a venv
built from `requirements/ci-unit.txt` alone — with and without CI's
`--maxfail=5` — and **1574 passed / 0 skipped** here. The whole suite also passes
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
manifests, `PROVENANCE.json` for each iNaturalist pool, all 38 stage reports, the
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
scored result they produced — both preregistered claims confirmed.
