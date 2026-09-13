"""Freeze the Iteration 12 Stage-1b seed-replication protocol.

    python scripts/freeze_iter12_seed_replication.py
    python scripts/freeze_iter12_seed_replication.py --check-only

Written BEFORE any replicate is trained, and it refuses to be written
afterwards.  Stage 1 exists because a criterion chosen after seeing results is
not a criterion; a replication study is exactly the place that discipline is
most tempting to skip, because the results it revisits are already known.  The
two parents, the three new seeds, the primary rule, the epsilon and the range
classification are all recorded here while they could still have been chosen
differently.

``_flat``, ``_sha256``, ``_hash_all`` and ``_git`` are IMPORTED from the
Stage-1 freeze rather than reimplemented, so "does this document still match the
repository" means the same thing in both studies.

The Stage-1 freeze is left completely alone: it is hash-bound, nine candidates
have adapters, and it now refuses to be rewritten.  Nothing here edits it or any
other frozen path.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

#: Hashing, git state and document flattening come from the Stage-1 freeze so
#: both studies verify identically.  Importing also means a change to that
#: frozen file shows up here as a change in behaviour rather than as a silent
#: divergence between two copies.
from freeze_iter12_selection_protocol import (
    _flat,
    _git,
    _hash_all,
    _sha256,
)

from granunlearn.config import _find_repo_root
from granunlearn.evaluation import retention_selection as rs
from granunlearn.training.seed_replication import (
    ALL_SEEDS,
    BASE_SEED,
    FLOOR_NUMBERS,
    NEW_SEEDS,
    PARENTS,
    PROBE_REPORT,
    SEED_CKPT_ROOT,
    SEED_PREDICTIONS_SUBDIR,
    STAGE1_CKPT_ROOT,
    STAGE1_FREEZE_REPORT,
    STAGE1_PREDICTIONS_SUBDIR,
    STAGE1_SELECTION_REPORT,
    replicate_id,
    replicates,
    to_train,
)
from granunlearn.training.seed_replication import (
    FREEZE_REPORT as OUT_REPORT,
)

#: The four new files this study is run by.  Hashed so that the rule cannot be
#: edited between training and scoring.
PROTOCOL_PATHS = (
    "scripts/freeze_iter12_seed_replication.py",
    "src/granunlearn/training/seed_replication.py",
    "scripts/train_iter12_seed_replicates.py",
    "scripts/analyze_iter12_seed_replication.py",
)

#: Imported by the four above and therefore part of the protocol: the frozen
#: criterion, the frozen grid, the frozen trainer and the sealed generation
#: path.  A change to any of them changes what this study measures.
DEPENDS_ON_UNCHANGED = (
    "src/granunlearn/evaluation/retention_selection.py",
    "src/granunlearn/evaluation/reference_eval.py",
    "src/granunlearn/evaluation/hierarchy_metrics.py",
    "src/granunlearn/training/candidate_grid.py",
    "src/granunlearn/training/unlearning_trainer.py",
    "src/granunlearn/training/reference_trainer.py",
    "scripts/select_unlearning_checkpoints.py",
    "scripts/freeze_iter12_selection_protocol.py",
)

#: Committed evidence this protocol is derived from.  The anchor values and the
#: parents' Stage-1 shortfalls are read out of the filed Stage-1 report, not
#: restated, so the freeze cannot disagree with it.
DATA_PATHS = (
    STAGE1_FREEZE_REPORT,
    STAGE1_SELECTION_REPORT,
    PROBE_REPORT,
    "data/mllmu_hier_pilot100/unlearning_iter12/fine_target.jsonl",
    "data/mllmu_hier_pilot100/unlearning_iter12/target_level.jsonl",
    "data/mllmu_hier_pilot100/unlearning_iter12/retain.jsonl",
)

VOLATILE_FIELDS = ("frozen_at_utc", "git_commit", "git_dirty", "amendments")

#: Keys of the Stage-1 generation contract that must MATCH, because they are
#: inside the measurement, and keys that must DIFFER, because they identify the
#: study.  Frozen here so the analyzer's enforcement is checking a recorded
#: decision rather than its own opinion.
CONTRACT_MUST_MATCH = ("batch_size", "image_batch_size", "max_new_tokens",
                       "max_length", "max_image_pixels", "do_sample",
                       "model_id")
CONTRACT_MUST_DIFFER = ("predictions_dir", "experiment_id")


def trained_replicates(repo_root: Path) -> list[str]:
    """Replicate ids that already have adapters on disk."""
    root = repo_root / SEED_CKPT_ROOT
    if not root.exists():
        return []
    return sorted(p.parent.parent.name for p in
                  root.glob("*/adapters/adapter_model.safetensors"))


def _stage1(repo_root: Path) -> dict[str, Any]:
    return json.loads((repo_root / STAGE1_SELECTION_REPORT).read_text())


def _shortfall_in_queries(stage1: dict, parent: str) -> dict[str, Any]:
    """The parent's Stage-1 retain-same shortfall expressed in QUERIES.

    A rate difference on its own does not say whether a single seed could
    decide it; the same difference over a known denominator does.  Recomputed
    here from the filed report rather than quoted.
    """
    checks = stage1["candidates"][parent]["floor"]["checks"]
    out = {}
    for family, estimand in FLOOR_NUMBERS:
        c = checks[f"{family}.{estimand}"]
        n = stage1["floor"]["b0"][family]["num_queries"]
        out[f"{family}.{estimand}"] = {
            "candidate": c["candidate"], "b0": c["b0"],
            "difference": c["difference"], "passes": c["passes"],
            "num_queries": n,
            "difference_in_queries": round(c["difference"] * n, 2),
        }
    return out


def build_freeze(repo_root: Path) -> dict[str, Any]:
    stage1 = _stage1(repo_root)
    contract = json.loads(
        (repo_root / STAGE1_FREEZE_REPORT).read_text())["generation_contract"]
    probe = json.loads((repo_root / PROBE_REPORT).read_text())
    return {
        "iteration": 12,
        "stage": "1b — seed replication of the two near-miss candidates",
        "exploratory": True,
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": _git(repo_root, "rev-parse", "HEAD"),
        "git_dirty": _git(repo_root, "status", "--porcelain") or "",
        "question": ("are the two smallest Stage-1 retain-same shortfalls "
                     "stable across independent training seeds, or one draw?"),
        "why_this_study_exists": {
            "stage1_result": "all eight B4 replay candidates disqualified; B0 "
                             "selected; the Stage-2 gate opened",
            "what_stage1_could_not_decide": (
                "the floor is a point comparison over one training seed, so a "
                "shortfall of 3 queries and a shortfall of 59 are the same "
                "kind of verdict"),
            "post_hoc_check_that_shaped_this_design": (
                "re-scoring the same eight candidates against a more lenient "
                "MG anchor also rejects all eight, so the binding limitation "
                "is the measurement, not the choice of anchor"),
        },
        "parents": {
            "ids": list(PARENTS),
            "selection_rule": (
                "the two candidates whose retain-same row-micro shortfall "
                "against B0 is smallest when expressed in QUERIES, taken from "
                "the filed Stage-1 report"),
            "not_selected_on": (
                "D_G. Selecting the two best on the criterion would be "
                "selecting on the outcome being revisited"),
            "stage1_shortfalls": {p: _shortfall_in_queries(stage1, p)
                                  for p in PARENTS},
        },
        "seeds": {
            "base_reused_from_stage1": BASE_SEED,
            "why_reused": (
                "its adapter and predictions were produced and scored by "
                "Stage 1; re-running the same seed would produce a second draw "
                "from an identical RNG state and say nothing about between-seed "
                "variance"),
            "new": list(NEW_SEEDS),
            "why_these": ("the three integers after the recipe default, fixed "
                          "here before training so they cannot be chosen after "
                          "seeing results"),
            "all": list(ALL_SEEDS),
            "k": len(ALL_SEEDS),
            "chosen_before_any_replicate_was_trained": True,
        },
        "what_a_seed_varies": {
            "measured_not_assumed": True,
            "consumed_at": [
                "unlearning_trainer.set_recipe_seeds(recipe.seed) — LoRA "
                "initialisation and dropout",
                'random.Random(f"{recipe.seed}:{epoch}:{gi}").shuffle(order) '
                "— the per-group per-epoch order of the interleaved stream",
            ],
            "does_not_vary": (
                "which examples are trained on, so the fit/probe partition is "
                "untouched by replication"),
            "reachability_asserted_at_runtime": (
                "the trainer raises unless ReferenceRecipe(**overrides).seed "
                "equals the replicate's seed"),
        },
        "primary_rule": {
            "statement": ("for each of the four frozen numbers, the MEAN over "
                          "replicates must be at or above the Stage-1 B0 "
                          "anchor"),
            "scored_by": ("granunlearn.evaluation.retention_selection."
                          "floor_check — imported, not reimplemented"),
            "epsilon": rs.FLOOR_EPSILON,
            "epsilon_is_the_stage1_epsilon": True,
            "mean_is_compared_unrounded": True,
            "why_unrounded": (
                "rounding the mean to the four decimals its inputs carry would "
                "let a rounding step decide a comparison the inputs did not"),
            "no_margin": True,
            "margin_required": 0.0,
            "rejected_alternative": {
                "value": 0.02,
                "status": "rejected in Stage 1 and not reintroduced here",
                "why": ("it is not a resolvable gap, not an interval width and "
                        "not a pre-registered effect size"),
            },
        },
        "reported_but_not_decision_bearing": {
            "per_number": ["mean", "min", "max", "sd",
                           "replicates_at_or_above_anchor",
                           "range_classification"],
            "range_values": ["every_replicate_below", "straddles",
                             "every_replicate_at_or_above"],
            "why_not_an_interval": (
                "with k=4 an interval imports a distributional assumption the "
                "design does not need; the observed range answers whether every "
                "independent seed fell below the anchor"),
            "distance_to_mg": (
                "reported per replicate and as a mean; the floor decides "
                "eligibility, exactly as in Stage 1"),
        },
        "anchor": {
            "source": STAGE1_SELECTION_REPORT,
            "field": "floor.b0",
            "values": {f"{f}.{e}": stage1["floor"]["b0"][f][e]
                       for f, e in FLOOR_NUMBERS},
            "denominators": {f: stage1["floor"]["b0"][f]["num_queries"]
                             for f, _ in FLOOR_NUMBERS},
            "why_a_point": (
                "B0 is the no-op copy of MF and takes zero training steps, so "
                "it has no seed variance; the determinism control below is what "
                "tests whether it has generation variance"),
        },
        "generation_determinism_control": {
            "runs_before": "any replicate verdict is reported",
            "what_it_does": (
                "regenerates B0 into this study's own predictions directory "
                "under the Stage-1 contract and compares it row by row with "
                "Stage 1's B0 parquet"),
            "stops_the_study_if": "any of the four floor numbers moves",
            "why": ("a point floor over a non-reproducible anchor is not "
                    "interpretable and replication cannot repair it"),
            "already_measured_from_two_existing_parcets": {
                "comparison": ("identical B0 adapter bytes generated under "
                               "image_batch_size 8 (pilot-100) and 1 "
                               "(Stage 1)"),
                "correctness_flips": 15,
                "rows": 4518,
                "raw_output_differences": 60,
                "flips_by_family": {"retain_same_entity_image": 9,
                                    "retain_other_entity_image": 2,
                                    "image_fine_direct": 3,
                                    "multimodal_image_text": 1},
                "flips_inside_either_probe_subset": 0,
                "why_zero": ("the floor's 408 + 76 probe queries are 100% "
                             "text_to_text; every flip is image-route"),
                "d_g_components_that_moved": {"filr": -0.0004,
                                              "wrong": 0.0008},
                "consequence_for_d_g": (
                    "generation noise of order 2e-4, ~17x smaller than the gap "
                    "between Stage 1's two best candidates (0.027343 vs "
                    "0.030686), so it does not reorder them — but D_G's sixth "
                    "decimal is not physically meaningful, and the frozen "
                    "tie-break compares at exactly six decimals"),
                "what_it_cannot_show": (
                    "whether generation is deterministic under a FIXED "
                    "contract, which is the assumption every replicate "
                    "comparison rests on — hence the control"),
            },
        },
        "generation_contract": {
            "inherited_from": STAGE1_FREEZE_REPORT,
            "must_match": {k: contract[k] for k in CONTRACT_MUST_MATCH},
            "must_differ": {k: contract[k] for k in CONTRACT_MUST_DIFFER},
            "this_study_uses": {
                "experiment_id": "mllmu_iter12_seeds",
                "predictions_dir":
                    f"data/mllmu_hier_pilot100/{SEED_PREDICTIONS_SUBDIR}",
            },
            "why_a_separate_directory": (
                "the control regenerates B0, so this study needs a file called "
                "predictions_tv_B0.parquet. In Stage 1's directory that name is "
                "already taken by filed evidence whose sidecar carries a "
                "different experiment_id, so the sidecar would refuse reuse and "
                "REGENERATE OVER IT. A separate directory makes that impossible "
                "rather than unlikely — the same reasoning Stage 1 recorded for "
                "not using the pilot-100 directory"),
        },
        "evidence_base": {
            "reads": ["pilot-100 train+val predictions",
                      "the fit-half training groups",
                      "the filed Stage-1 selection report"],
            "never_read": list(rs.FORBIDDEN_EVIDENCE),
            "reused_artifacts": {
                "seed_42_replicate_predictions":
                    f"data/mllmu_hier_pilot100/{STAGE1_PREDICTIONS_SUBDIR}/",
                "seed_42_replicate_adapters": f"{STAGE1_CKPT_ROOT}/",
                "mg_reference_vector": f"{STAGE1_SELECTION_REPORT} "
                                       "(reference.vector), read not "
                                       "regenerated",
            },
        },
        "replicate_set": {
            "ids": [r.replicate_id for r in replicates()],
            "needing_a_gpu": [r.replicate_id for r in to_train()],
            "reused": [replicate_id(p, BASE_SEED) for p in PARENTS],
            "cost_measured_from_stage1": {
                "training_seconds_per_candidate_ep5": 1609.2,
                "generation_minutes_per_state_median": 40.2,
                "trainings": len(to_train()),
                "generations": len(to_train()) + 1,
                "plus_one_for": "the determinism control",
            },
        },
        "refusal_policy": {
            "refuses_to_freeze_when": "any replicate adapter exists",
            "unconditional": True,
            "no_flag_reaches_past_it": True,
            "reason": ("a replication protocol written after its replicates "
                       "were trained could be fitted to them, which is the "
                       "one thing a replication is supposed to make "
                       "impossible"),
        },
        "hashes": {
            "protocol_paths": _hash_all(repo_root, PROTOCOL_PATHS),
            "depends_on_unchanged": _hash_all(repo_root, DEPENDS_ON_UNCHANGED),
            "data_paths": _hash_all(repo_root, DATA_PATHS),
        },
        "limitations": [
            "Replication does not increase the query count. The probe half is "
            "35 entities / 408 retain-same and 23 donors / 76 retain-other "
            "queries, fixed by the association pool, so retain-other stays "
            "resolvable only in units of 1/76 = 0.0132 however many seeds are "
            "run.",
            "k=4 per parent. A mean over four draws bounds training-seed "
            "variance loosely and licenses no interval claim.",
            "Two parents only. This says nothing about the other six Stage-1 "
            "candidates.",
            "The probe half carries 1 of the 6 taxonomic retained "
            "associations, so taxonomic retention remains unmeasured, as in "
            "Stage 1.",
            "Exploratory: no hypothesis is tested and no error rate is "
            "controlled. A pass is a reason to design a confirmatory protocol, "
            "not a claim.",
        ],
        "probe_partition": {
            "source": PROBE_REPORT,
            "rule": probe["partition_rule"],
            "probe_entities": probe["halves"]["probe"]["num_entities"],
            "probe_associations": probe["halves"]["probe"]["num_associations"],
            "replayed_by_any_candidate":
                probe["halves"]["probe"]["replayed_by_any_candidate"],
        },
        "amendments": [],
    }


def verify_freeze(repo_root: Path, path: Path | None = None) -> list[str]:
    """Recompute the freeze against the repository. Empty list == matches."""
    path = path or repo_root / OUT_REPORT
    if not path.exists():
        return [f"{path} does not exist -- the protocol is not frozen"]
    old = _flat(json.loads(path.read_text()))
    fresh = _flat(build_freeze(repo_root))
    reasons = []
    for key in sorted(set(old) | set(fresh)):
        if any(key.startswith(f"{v}.") or key == v for v in VOLATILE_FIELDS):
            continue
        if old.get(key) != fresh.get(key):
            reasons.append(f"{key}: frozen={old.get(key)!r} "
                           f"now={fresh.get(key)!r}")
    return reasons


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check-only", action="store_true",
                    help="Verify the committed freeze instead of writing it")
    ap.add_argument("--refreeze", action="store_true",
                    help="Amend an existing freeze, recording the amendment. "
                         "Refused once any replicate has been trained.")
    ap.add_argument("--reason", default=None,
                    help="Why the freeze is being amended (with --refreeze)")
    ap.add_argument("--repo-root", default=None)
    args = ap.parse_args()
    repo_root = Path(args.repo_root) if args.repo_root \
        else (_find_repo_root(Path.cwd()) or Path.cwd())
    out = repo_root / OUT_REPORT

    if args.check_only:
        reasons = verify_freeze(repo_root)
        if reasons:
            print("MISMATCHES:")
            for r in reasons:
                print(f"  {r}")
            raise SystemExit(1)
        print(f"OK — the Stage-1b seed-replication freeze still matches the "
              f"repository (volatile fields excluded: "
              f"{list(VOLATILE_FIELDS)})")
        return

    #: Evaluated FIRST and unconditionally: no flag, including --refreeze,
    #: reaches past it.
    trained = trained_replicates(repo_root)
    if trained:
        raise SystemExit(
            f"REFUSING to freeze: {len(trained)} seed replicate(s) already have "
            f"adapters on disk ({trained[:3]}...). A replication protocol "
            f"written now would postdate the replicates it is supposed to "
            f"govern, and could be fitted to them.")

    document = build_freeze(repo_root)
    if out.exists():
        previous = json.loads(out.read_text())
        if not args.refreeze:
            raise SystemExit(
                f"REFUSING to overwrite {out} (sha256 {_sha256(out)}). It is "
                f"the record that this protocol predates its own replicates; "
                f"deleting or replacing it is the one act that would make that "
                f"claim unfalsifiable. --refreeze --reason amends it with the "
                f"change recorded.")
        if not args.reason:
            raise SystemExit("--refreeze requires --reason: an amendment whose "
                             "reason is not written down is indistinguishable "
                             "from a quiet rewrite.")
        old_flat, new_flat = _flat(previous), _flat(document)
        changed = sorted(
            k for k in set(old_flat) | set(new_flat)
            if not any(k.startswith(f"{v}.") or k == v
                       for v in VOLATILE_FIELDS)
            and old_flat.get(k) != new_flat.get(k))
        #: The refusal above is what makes this assertion safe to record: no
        #: flag reaches past it, so an amendment can only exist from a time
        #: when no replicate had been trained.
        document["amendments"] = [*previous.get("amendments", []), {
            "utc": document["frozen_at_utc"],
            "reason": args.reason,
            "supersedes_sha256": _sha256(out),
            "fields_that_changed": changed,
            "no_replicate_had_been_trained": True,
        }]
        out.write_text(json.dumps(document, indent=2) + "\n")
        print(f"Seed-replication freeze AMENDED -> {out}")
        print(f"  reason: {args.reason}")
        print(f"  fields that changed: {changed or '(none)'}")
        return

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(document, indent=2) + "\n")
    print(f"Seed-replication protocol frozen -> {out}")
    print(f"  parents {list(PARENTS)}")
    print(f"  seeds {list(ALL_SEEDS)} (k={len(ALL_SEEDS)}, {BASE_SEED} reused "
          f"from Stage 1)")
    print(f"  primary rule: mean over replicates >= the Stage-1 B0 anchor on "
          f"all four numbers, epsilon {rs.FLOOR_EPSILON}, no margin")
    print(f"  trainings needed: {len(to_train())}")


if __name__ == "__main__":
    main()
