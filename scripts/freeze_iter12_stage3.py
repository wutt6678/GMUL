"""Freeze the Iteration 12 Stage-3 protocol before a single B7 adapter exists.

    python scripts/freeze_iter12_stage3.py
    python scripts/freeze_iter12_stage3.py --check-only
    python scripts/freeze_iter12_stage3.py --refreeze --reason "..."

Stage 3 is a successor study to a filed Stage-2 result, which makes the
pre-training discipline more important rather than less: the parent, its
fourteen-query shortfall and its fine-target NLL are all known.  This freeze
records what may be varied (one additional image-conditioned anchor weight,
at four values), what may not (the eight MG-anchored floors, D_G, the B6
objective, the fit/probe split, the seed, the batch layouts, the cache and the
tie-break), and how the zero-weight B6 control is judged.

The refusal is keyed on adapters and evaluated unconditionally before any
flag other than ``--check-only``: once a B7 adapter exists, a protocol written
or amended now would postdate the training it governs.

Nothing here edits a frozen path.  Hashing and git helpers are imported from
the Stage-1 freeze; the Stage-2 criterion, cache and trainer are imported and
bound, never modified.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from freeze_iter12_route_stratification import (
    contract_signature,
    prediction_filename,
    sidecar_for,
)
from freeze_iter12_selection_protocol import (
    _flat,
    _git,
    _hash_all,
    _sha256,
)

from granunlearn.config import _find_repo_root
from granunlearn.evaluation import retention_selection as rs
from granunlearn.evaluation import route_stratified_retention as rt
from granunlearn.evaluation.selection import SUMMARY_COMPONENTS
from granunlearn.logging_utils import setup_logger
from granunlearn.training import stage2_grid as s2g
from granunlearn.training import stage3_grid as s3g
from granunlearn.training import stage3b_replication as s3b
from granunlearn.training.preservation_anchor import (
    CACHE_SIDECAR,
    CACHE_TENSOR,
    CACHE_VERSION,
)

log = setup_logger("freeze_iter12_stage3")

OUT_REPORT = s3g.FREEZE_REPORT
BASIS_REPORT = s3g.BASIS_REPORT
CONTROL_REPORT = s3g.CONTROL_REPORT
SELECTION_REPORT = s3g.OUT_REPORT
STAGE1_FREEZE_REPORT = "data/reports/mllmu_iter12_selection_protocol_freeze.json"

#: The code that IMPLEMENTS Stage 3.  A change to one of these changes the
#: mechanism, the control, the grid or the scoring rule, and must fail
#: --check-only before another row is trained or generated.
PROTOCOL_PATHS = (
    "scripts/freeze_iter12_stage3.py",
    "scripts/build_iter12_stage3_basis.py",
    "scripts/train_iter12_stage3.py",
    "scripts/select_iter12_stage3.py",
    "src/granunlearn/training/stage3_grid.py",
    "src/granunlearn/training/stage3b_replication.py",
)

#: Frozen code Stage 3 DEPENDS ON but must not modify.  Several of these are
#: already bound by earlier Iteration-12 freezes; importing them makes drift
#: visible here as well rather than letting Stage 3 inherit it silently.
DEPENDS_ON_UNCHANGED = (
    "src/granunlearn/evaluation/retention_selection.py",
    "src/granunlearn/evaluation/route_stratified_retention.py",
    "src/granunlearn/evaluation/hierarchy_metrics.py",
    "src/granunlearn/evaluation/reference_eval.py",
    "src/granunlearn/evaluation/prediction_provenance.py",
    "src/granunlearn/evaluation/scoring.py",
    "src/granunlearn/training/candidate_grid.py",
    "src/granunlearn/training/stage2_grid.py",
    "src/granunlearn/training/preservation_anchor.py",
    "src/granunlearn/training/preservation_trainer.py",
    "src/granunlearn/training/unlearning_trainer.py",
    "src/granunlearn/training/unlearning_datasets.py",
    "src/granunlearn/training/state_datasets.py",
    "src/granunlearn/training/reference_trainer.py",
    "src/granunlearn/training/seed_replication.py",
    "scripts/select_unlearning_checkpoints.py",
    "scripts/select_iter12_retention_checkpoints.py",
    "scripts/freeze_iter12_selection_protocol.py",
    "scripts/freeze_iter12_seed_replication.py",
    "scripts/freeze_iter12_route_stratification.py",
    "scripts/freeze_iter12_stage2.py",
    "scripts/build_iter12_stage2_basis.py",
    "scripts/build_mf_reference_logprobs.py",
)

#: Committed evidence this protocol is derived from.  The control and basis
#: reports are bound because they are the pre-training record a reviewer reads.
DATA_PATHS = (
    BASIS_REPORT,
    CONTROL_REPORT,
    s2g.OUT_REPORT,
    s2g.FREEZE_REPORT,
    s2g.BASIS_REPORT,
    s2g.CALIBRATION_REPORT,
    "data/reports/mllmu_iter12_stage2_control_reading.json",
    s2g.ROUTE_STRATIFIED_REPORT,
    s2g.STAGE1C_FREEZE_REPORT,
    "data/reports/mllmu_iter12_route_probe.json",
    STAGE1_FREEZE_REPORT,
    "data/reports/mllmu_iter12_retention_selection.json",
    "data/reports/mllmu_iter12_seed_replication_freeze.json",
    "data/reports/mllmu_iter12_seed_replication.json",
    "data/reports/mllmu_iter12_seed_replication.REPAIR.json",
    "data/reports/mllmu_iter12_retention_probe.json",
    "data/mllmu_hier_pilot100/queries.parquet",
    "data/mllmu_hier_pilot100/associations.parquet",
    "data/mllmu_hier_pilot100/unlearning_iter12/fine_target.jsonl",
    "data/mllmu_hier_pilot100/unlearning_iter12/target_level.jsonl",
    "data/mllmu_hier_pilot100/unlearning_iter12/retain.jsonl",
    "data/mllmu_hier_pilot100/training/MF.jsonl",
    "data/mllmu_hier_pilot100/training/MG.jsonl",
)

VOLATILE_FIELDS = ("frozen_at_utc", "git_commit", "git_dirty", "amendments")
PREDICTIONS_FIELD = "reused_predictions_bound"
CACHE_FIELD = "image_conditioned_cache_bound"
CONTRACT_FIELD = "generation_contract_inherited"
CLONE_DEPENDENT_FIELDS = (PREDICTIONS_FIELD, CACHE_FIELD)
CONTRACT_MUST_MATCH = ("batch_size", "image_batch_size", "max_new_tokens",
                       "max_length", "max_image_pixels", "do_sample")


def reused_prediction_paths(repo_root: Path) -> dict[str, str]:
    """The three parquets Stage 3 reads instead of regenerating."""
    calibration = json.loads((repo_root / s2g.CALIBRATION_REPORT).read_text())
    b6 = s3g.control_row(
        s3g.stage3_grid(calibration["grid_rule"]["beta_star"])).candidate_id
    out = {
        "stage1.B0": str((s3g.stage1_predictions_dir(repo_root) /
                          prediction_filename("B0")).relative_to(repo_root)),
        "stage1.MG": str((s3g.stage1_predictions_dir(repo_root) /
                          prediction_filename("MG")).relative_to(repo_root)),
        "stage2.B6": str((s3g.stage2_predictions_dir(repo_root) /
                          prediction_filename(b6)).relative_to(repo_root)),
    }
    for key, rel in out.items():
        if rel.startswith("data/data/") or not rel.startswith("data/"):
            raise SystemExit(
                f"{key} resolved to {rel!r}, not a repo-relative prediction "
                f"path; a resolver has doubled a prefix")
    return out


def cache_paths(repo_root: Path) -> dict[str, str]:
    cdir = s3g.reference_cache_dir(repo_root)
    out = {}
    for name in (CACHE_SIDECAR, CACHE_TENSOR):
        rel = str((cdir / name).relative_to(repo_root))
        if rel.startswith("data/data/"):
            raise SystemExit(f"{name} resolved to {rel!r}: a doubled prefix")
        out[name] = rel
    return out


def inherited_contract(repo_root: Path) -> dict[str, Any]:
    """The generation contract, read from the Stage-1 freeze."""
    stage1_freeze = json.loads((repo_root / STAGE1_FREEZE_REPORT).read_text())
    contract = stage1_freeze["generation_contract"]
    stage1c = json.loads((repo_root / s2g.STAGE1C_FREEZE_REPORT).read_text())
    verified = stage1c["measurement_contract"]
    return {
        "contract": contract,
        "keys_compared": sorted(contract),
        "read_from": STAGE1_FREEZE_REPORT,
        "stage1c_verified_one_instrument": verified["one_instrument"],
        "stage1c_distinct_contracts": verified["distinct_contracts"],
        "agrees_with_the_stage1c_verified_contract": all(
            verified["contract"].get(k) == v
            for k, v in contract.items()
            if k in CONTRACT_MUST_MATCH),
        "this_study_uses": {
            "experiment_id": "mllmu_iter12_stage3",
            "predictions_dir":
                f"data/mllmu_hier_pilot100/{s3g.STAGE3_PREDICTIONS_SUBDIR}",
        },
        "why_it_is_load_bearing": (
            "Stage 3 compares newly generated B7 predictions with reused B0, "
            "MG and B6 parquets. Batch layout is inside D_G and inside the "
            "image route's correctness, so all of them must be one "
            "instrument."),
    }


def adapters_exist(repo_root: Path) -> list[str]:
    root = s3g.stage3_ckpt_root(repo_root)
    if not root.exists():
        return []
    return sorted(str(p.parent.relative_to(repo_root))
                  for p in root.glob("**/adapters") if p.is_dir())


def post_stage3_decision_tree(repo_root: Path, grid: list) -> dict[str, Any]:
    """The downstream tree, derived rather than transcribed.

    The envelope is recomputed here from the frozen MG anchor and B6's filed
    floor values, so the numbers in this document cannot drift from the code
    that applies them.  ``select_iter12_stage3.py`` re-derives the same
    envelope from the bound Stage-2 parquet before every scoring run and
    refuses to write a report if the bytes no longer reproduce it.
    """
    stage2_freeze = json.loads((repo_root / s2g.FREEZE_REPORT).read_text())
    anchor = stage2_freeze["what_is_frozen"]["anchor_values"]
    env = s3b.incumbent_envelope(anchor)
    b7 = s3g.b7_rows(grid)
    return {
        "preregistration": {
            "filed_by": "an amendment to this freeze, before any B7 adapter "
                        "or B7 prediction existed",
            "amendment_reason_is_recorded_in": "amendments[-1].reason",
            "why_it_had_to_be_filed_now": (
                "the tree was absent from the original freeze, and a "
                "near-miss threshold chosen after Stage 3 is scored can be "
                "fitted to whichever row it admits. This freeze refuses "
                "amendment once a B7 adapter exists, so this is the last "
                "moment the rule can predate the result it governs."),
        },
        "shortfall": {
            "definition": "d_j(c) = max(0, a_j - v_j(c) - epsilon) for each "
                          "of the eight frozen floor numbers j",
            "epsilon": rs.FLOOR_EPSILON,
            "epsilon_is_the_frozen_tolerance_not_a_margin": True,
            "a_j": "the frozen MG anchor in what_is_frozen.anchor_values",
            "v_j_c": "the candidate's own eight numbers, recomputed from its "
                     "prediction parquet",
            "no_further_rounding": True,
            "why_not_the_reported_differences": (
                "the selection report's difference fields are rounded to four "
                "decimals; reading them would drop the 1e-9 and could flip a "
                "<= comparison against the envelope at exactly the boundary "
                "the epsilon exists to guard"),
            "a_missing_number_disqualifies": (
                "an unmeasurable retention number is not a zero shortfall, so "
                "it cannot be ranked as a near miss"),
            "computed_by":
                "granunlearn.training.stage3b_replication.shortfall_vector",
        },
        "incumbent_relative_envelope": {
            "incumbent": s3g.control_row(grid).candidate_id,
            "b6_floor_values_source": {
                "recomputed_from_the_bound_stage2_parquet": True,
                "parquet": reused_prediction_paths(repo_root)["stage2.B6"],
                "sha256_bound_as":
                    "reused_predictions_bound.sha256['stage2.B6']",
                "not_transcribed_from_a_report": True,
                "re_verified_before_every_scoring_run_by":
                    "select_iter12_stage3.py, which refuses to score if the "
                    "bound parquet no longer reproduces these values",
            },
            "b6_floor_values": dict(s3b.B6_FLOOR_VALUES_AT_AMENDMENT_TIME),
            "per_metric_shortfall": env["per_metric"],
            "K": env["K"],
            "M": env["M"],
            "S": env["S"],
            "failed_metrics": env["failed_metrics"],
            "summation": env["summation"],
            "query_shortfall_behind_it": dict(s3b.B6_QUERY_SHORTFALL),
            "why_incumbent_relative": (
                "the image route's between-seed training variance has never "
                "been measured -- Stage 1b measured sd on the four TEXT "
                "numbers only (0.0121 and 0.0127 retain-same, 0.0225 and "
                "0.0195 retain-other) -- and generation noise under the fixed "
                "contract is measured at zero over all 4,518 rows, so "
                "'within measured seed noise' has no referent for the three "
                "image numbers that decide Stage 3. 'No worse than the best "
                "existing mechanism's retention shortfall' needs no variance "
                "estimate and no distributional assumption."),
        },
        "qualification": {
            "exact_pass": "all eight d_j(c) == 0",
            "incumbent_relative_near_miss":
                "K(c) <= K_B6 AND M(c) <= M_B6 AND S(c) <= S_B6",
            "all_three_conjuncts_are_required": True,
            "why_all_three": (
                "K bounds how many numbers may fail, M how badly any one may "
                "fail, S the total. Without S a candidate could fail three "
                "numbers each just under M; without M it could fail one "
                "number arbitrarily far below the anchor."),
            "computed_by":
                "granunlearn.training.stage3b_replication.qualification",
        },
        "parent_eligibility": {
            "only_these_rows_can_be_stage3b_parents":
                [c.candidate_id for c in b7],
            "num_b7_rows": len(b7),
            "b6_zero_control_is_excluded": True,
            "why_b6_is_excluded": (
                "it validates reproduction and defines the envelope, but it is "
                "the filed Stage-2 incumbent rather than a new successor. The "
                "exclusion is load-bearing: B6 satisfies its own envelope "
                "with equality on all three conjuncts, so without it the "
                "control would select itself."),
            "b0_is_excluded": True,
            "why_b0_is_excluded": "the no-op reference is not a mechanism",
            "max_parents": s3b.MAX_PARENTS,
            "ranking": [
                "exact pass before near miss",
                "among exact passers: smaller D_G",
                "among near misses: fewer failed metrics",
                "then smaller maximum shortfall",
                "then smaller total shortfall",
                "then smaller D_G",
                "then lexicographic candidate_id",
            ],
            "ranking_key":
                "(0 if exact_pass else 1, K, M, S, distance_to_mg, "
                "candidate_id)",
            "if_no_b7_candidate_qualifies": s3b.CLOSE_ITERATION_12,
            "what_closing_means": (
                "Iteration 12 ends as a documented negative result. No "
                "Stage-3b replicate is trained and no successor is adopted. A "
                "non-B7 row in the selection report's `selected` field does "
                "not open Stage 3b."),
        },
        "stage3b_mechanics": {
            "seeds": {"reused": s3b.BASE_SEED, "trained": list(s3b.NEW_SEEDS),
                      "all": list(s3b.ALL_SEEDS),
                      "frozen_by": "the Stage-1b seed list, recorded before "
                                   "any replicate existed"},
            "seed_42_is_reused_not_retrained": (
                "the parent's Stage-3 adapter and predictions are read from "
                "the Stage-3 namespaces; re-running seed 42 would measure GPU "
                "nondeterminism rather than between-seed variance"),
            "per_seed": "all eight floor numbers are computed separately for "
                        "every seed, from that seed's own predictions",
            "mean": "the arithmetic mean of each metric over 42/43/44/45, "
                    "UNROUNDED",
            "mean_floor_rule": "every mean metric must satisfy "
                               "mean_j >= a_j - 1e-9",
            "scored_by": "granunlearn.evaluation.route_stratified_retention"
                         ".floor_check_stratified",
            "the_stage1b_analyzer_must_not_be_reused": True,
            "why_not": (
                "Stage 1b's mean_candidate/floor_check path is text-only "
                "(four numbers) and B0-anchored. Reusing it would drop the "
                "four image numbers that decide Stage 3 and re-anchor the "
                "floor on a state Stage 3 does not use."),
            "the_text_stratum_cross_check_still_runs": (
                "text_stratum_reproduces_the_frozen_floor is asserted on the "
                "mean, so the stratified verdict still equals the frozen "
                "floor's verdict on the four numbers they share"),
            "d_g": {
                "rule": "average the COMPLETE summary vector componentwise "
                        "over the four seeds, then apply the frozen "
                        "distance_to_mg to that mean vector",
                "components": list(SUMMARY_COMPONENTS),
                "forbidden": "averaging already-computed per-seed D_G values",
                "why_forbidden": (
                    "distance_to_reference is a weighted L1 over absolute "
                    "component differences, rounded to six decimals inside "
                    "each call. abs is convex and the rounding is per call, so "
                    "the mean of four distances is not the distance of the "
                    "mean vector and the two can order two parents "
                    "differently."),
                "per_seed_distances_are_reported_but_not_used": True,
            },
            "own_namespaces": {
                "checkpoints": s3b.STAGE3B_CKPT_ROOT,
                "predictions":
                    f"data/mllmu_hier_pilot100/"
                    f"{s3b.STAGE3B_PREDICTIONS_SUBDIR}",
                "report": s3b.OUT_REPORT,
                "resolvers_refuse_to_collide_with":
                    "the Stage-1, Stage-1b, Stage-2 and Stage-3 roots",
            },
        },
        "terminal_branches": {
            "no_mean_eligible_parent": s3b.STOP_ITERATION_12,
            "exactly_one_mean_eligible_parent": {
                "decision": s3b.SELECT_THE_ONLY_PARENT,
                "d_g_is_descriptive": True,
                "why": "with one survivor there is nothing to choose "
                       "between, so its D_G describes the state and decides "
                       "nothing",
            },
            "more_than_one_mean_eligible_parent": {
                "decision": s3b.SELECT_BY_MINIMUM_D_G,
                "tie_break": "(distance_to_mg, candidate_id) ascending",
            },
            "selected_parent": s3b.ITERATION_13,
            "iteration_13_is_separately_frozen": True,
            "iteration_12_carries_no_confirmatory_claim": True,
        },
        "implemented_by": {
            "module": "src/granunlearn/training/stage3b_replication.py",
            "sha256_bound_in": "hashes.protocol_paths",
            "executor_scripts_do_not_exist_yet": True,
            "constraint_on_them": (
                "a Stage-3b trainer or analyzer may only IMPORT that module. "
                "It may not restate or reinterpret these rules, and this "
                "freeze cannot be amended again once a B7 adapter exists."),
        },
    }


def build_freeze(repo_root: Path) -> dict[str, Any]:
    basis_path = repo_root / BASIS_REPORT
    control_path = repo_root / CONTROL_REPORT
    if not basis_path.exists() or not control_path.exists():
        missing = [str(p) for p in (basis_path, control_path) if not p.exists()]
        raise SystemExit(
            f"REFUSING: Stage-3 basis/control evidence is missing: {missing}. "
            f"The basis records the design and the control proves gamma=0 is "
            f"B6; freezing without them would bind a rule nobody wrote down.")
    basis = json.loads(basis_path.read_text())
    control = json.loads(control_path.read_text())
    calibration = json.loads((repo_root / s2g.CALIBRATION_REPORT).read_text())
    beta_star = calibration["grid_rule"]["beta_star"]
    grid = s3g.stage3_grid(beta_star)
    kept, filtered = s3g.validate_stage3_grid(grid, beta_star)
    if kept:
        raise SystemExit(f"REFUSING: the Stage-3 grid is invalid: {kept}")

    paths = reused_prediction_paths(repo_root)
    present = {k: v for k, v in paths.items() if (repo_root / v).exists()}
    cpaths = cache_paths(repo_root)
    cpresent = {k: v for k, v in cpaths.items() if (repo_root / v).exists()}
    sidecar = json.loads((repo_root / cpaths[CACHE_SIDECAR]).read_text()) \
        if CACHE_SIDECAR in cpresent else {}
    stage2_freeze = json.loads((repo_root / s2g.FREEZE_REPORT).read_text())
    stage2_rule = stage2_freeze["what_is_frozen"]

    return {
        "iteration": "12",
        "stage": 3,
        "protocol": "route-aware B6 successor (method B7), exploratory",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(
            timespec="seconds"),
        "git_commit": _git(repo_root, "rev-parse", "HEAD"),
        "git_dirty": _git(repo_root, "status", "--porcelain") or "",
        "amendments": [],
        "status": {
            "exploratory": True,
            "preregistered_before_training": True,
            "carries_no_confirmatory_claim": True,
            "probe_entities_remain_evaluation_only": True,
        },

        "what_is_frozen": {
            "floor_numbers": list(rt.FLOOR_NUMBER_KEYS),
            "num_numbers": len(rt.FLOOR_NUMBER_KEYS),
            "primary_anchor": stage2_rule["primary_anchor"],
            "primary_rule": stage2_rule["primary_rule"],
            "epsilon": stage2_rule["epsilon"],
            "margin": stage2_rule["margin"],
            "no_margin_is_used": stage2_rule["no_margin_is_used"],
            "anchor_values": stage2_rule["anchor_values"],
            "reported_baseline": stage2_rule["reported_baseline"],
            "reported_baseline_values":
                stage2_rule["reported_baseline_values"],
            "denominators": stage2_rule["denominators"],
            "among_eligible": stage2_rule["among_eligible"],
            "computed_by": stage2_rule["computed_by"],
            "distance_to_mg": {
                "direction": rs.D_G_DIRECTION,
                "rounding": "6 decimals, as distance_to_reference rounds",
                "tie_break": "(distance_to_mg, candidate_id) ascending",
                "mg_is_excluded_from_the_ranking": True,
            },
            "unchanged_from_the_stage2_freeze": True,
        },

        "b6_result": basis["b6_result"],
        "image_anchor_examples": basis["image_anchor_examples"],

        "mechanism": {
            "method": s3g.METHOD_ROUTE_ANCHOR,
            "parent": basis["b6_result"]["candidate_id"],
            "what_moves": (
                "only the coefficient on the image-conditioned MF anchor "
                "stream: effective weight = B6 beta + additional image weight"),
            "what_does_not_move": [
                "fine_target gd at lambda 0.5",
                "target_level sft at weight 1.0",
                "the fit-half retain examples",
                "the MF reference cache",
                "learning rate 2e-05",
                "five epochs",
                "training seed 42",
                "PYTHONHASHSEED 0",
                "the frozen trainer and its starting MF adapter",
            ],
            "additional_image_weight_multipliers":
                list(s3g.IMAGE_WEIGHT_MULTIPLIERS),
            "additional_image_weights":
                basis["grid"]["additional_image_weights"],
            "effective_anchor_weights":
                basis["grid"]["effective_anchor_weights"],
            "why_these_weights": basis["grid"]["why_these_weights"],
        },

        "grid": {
            "num_rows": len(grid),
            "num_trained": len(s3g.trained_rows(grid)),
            "rows": [c.describe() for c in grid],
            "the_frozen_validator_was_run": {
                "residual_errors": kept,
                "filtered_messages": filtered,
                "filtered_gap_substrings": list(s3g.FROZEN_VALIDATOR_GAP),
                "filtered_because": basis["grid"][
                    "the_frozen_validator_was_run"]["filtered_because"],
            },
        },

        "seeds_and_layout": basis["seeds_and_layout"],
        "zero_weight_control": {
            "report": CONTROL_REPORT,
            "sha256": _sha256(control_path),
            "control": control["control"],
            "rule": control["rule"],
            "tolerances": control["tolerances"],
            "holds_at_freeze_time": control["holds"],
            "bound_bytes": control["bound_bytes"],
            "it_is_a_gate_not_a_report": (
                "train_iter12_stage3.py recomputes this control and refuses "
                "to train any B7 row if it no longer holds; "
                "select_iter12_stage3.py applies the same gate before "
                "generation."),
        },

        "post_stage3_decision_tree": post_stage3_decision_tree(repo_root, grid),

        "evidence_base": {
            "used": basis["evidence_base"]["used"],
            "never_read": list(rs.FORBIDDEN_EVIDENCE),
            "never_read_enforced_by": (
                "train_iter12_stage3.py and select_iter12_stage3.py refuse "
                "any argument that resolves inside those paths, on RESOLVED "
                "paths rather than strings."),
            "probe_half_never_rehearsed": (
                "The image-conditioned anchor stream covers 0 of the 204 "
                "probe-half associations, measured in the basis."),
        },

        "basis": {
            "report": BASIS_REPORT,
            "sha256": _sha256(basis_path),
            "reads_no_prediction_parquet":
                basis["reads_no_prediction_parquet"],
        },
        "measurement_limits": stage2_freeze["measurement_limits"],
        "extends": {
            "stage2_freeze": s2g.FREEZE_REPORT,
            "stage2_selection": s2g.OUT_REPORT,
            "stage1c_freeze": s2g.STAGE1C_FREEZE_REPORT,
            "stage1_freeze": STAGE1_FREEZE_REPORT,
            "note": (
                "Stage 3 preserves Stage 2's criterion exactly and reuses its "
                "B6 row as the zero-weight control. It re-decides no filed "
                "Stage-1 or Stage-2 verdict."),
        },
        "refusal_policy": {
            "refuses_to_freeze_when": "any Stage-3 adapter exists",
            "unconditional": True,
            "no_flag_reaches_past_it": True,
            "reason": (
                "the parent and its shortfall are already known, so a grid "
                "or control amended after training could be fitted to the "
                "outcome it is supposed to govern -- including which "
                "shortfalls earn replication"),
        },

        "hashes": {
            "protocol_paths": _hash_all(repo_root, PROTOCOL_PATHS),
            "depends_on_unchanged": _hash_all(repo_root, DEPENDS_ON_UNCHANGED),
            "data_paths": _hash_all(repo_root, DATA_PATHS),
        },

        PREDICTIONS_FIELD: {
            "count_expected": len(paths),
            "count_present_at_freeze_time": len(present),
            "absent_at_freeze_time": sorted(
                k for k in paths if k not in present),
            "sha256": {k: _sha256(repo_root / v) for k, v in present.items()},
            "sidecar_contract": {
                k: contract_signature(
                    json.loads(sidecar_for(repo_root, v).read_text()))
                for k, v in present.items()
                if sidecar_for(repo_root, v).exists()
            },
            "paths": paths,
            "reused_not_regenerated": True,
            "why_reused": (
                "B0 and MG are the bound floor references and B6 is the "
                "zero-weight control. Regenerating them would produce a "
                "second draw where hashed evidence already sits."),
            "note": (
                "Gitignored and therefore absent from a clone. Verified "
                "apart from the rest of the freeze, reporting match, "
                "mismatch or absent per path."),
        },

        CACHE_FIELD: {
            "count_expected": len(cpaths),
            "count_present_at_freeze_time": len(cpresent),
            "absent_at_freeze_time": sorted(
                k for k in cpaths if k not in cpresent),
            "sha256": {k: _sha256(repo_root / v) for k, v in cpresent.items()},
            "paths": cpaths,
            "sidecar": {k: sidecar.get(k) for k in (
                "cache_version", "model_id", "base_model_revision",
                "mf_adapter_sha256", "preserve_group_sha256", "max_length",
                "max_image_pixels", "vocab_size", "num_examples",
                "num_positions", "nll_agreement_max", "kl_direction",
                "probe_half_excluded", "target_groups_excluded")},
            "cache_version": CACHE_VERSION,
            "why_bound": (
                "Every B7 row trains against these bytes. A cache rebuilt "
                "from a different MF adapter or encoding contract would "
                "preserve a different function while looking ordinary."),
        },

        CONTRACT_FIELD: inherited_contract(repo_root),
    }


# ── verification ────────────────────────────────────────────────────────

def verify_freeze(repo_root: Path, path: Path | None = None) -> list[str]:
    """Recompute the freeze against the repository. Empty list == matches."""
    path = path or repo_root / OUT_REPORT
    if not path.exists():
        return [f"{path} does not exist -- Stage 3 is not frozen"]
    old = _flat(json.loads(path.read_text()))
    fresh = _flat(build_freeze(repo_root))
    skip = VOLATILE_FIELDS + CLONE_DEPENDENT_FIELDS
    reasons = []
    for key in sorted(set(old) | set(fresh)):
        if any(key.startswith(f"{v}.") or key == v for v in skip):
            continue
        if old.get(key) != fresh.get(key):
            reasons.append(f"{key}: frozen={old.get(key)!r} "
                           f"now={fresh.get(key)!r}")
    return reasons


def verify_clone_dependent(repo_root: Path,
                           path: Path | None = None,
                           ) -> dict[str, dict[str, list[str]]]:
    """Three-valued check of the gitignored bytes Stage 3 depends on."""
    path = path or repo_root / OUT_REPORT
    doc = json.loads(path.read_text())
    out: dict[str, dict[str, list[str]]] = {}
    for field in CLONE_DEPENDENT_FIELDS:
        block = doc[field]
        frozen: dict[str, str] = block["sha256"]
        paths: dict[str, str] = block["paths"]
        res: dict[str, list[str]] = {"match": [], "mismatch": [], "absent": []}
        for key, rel in sorted(paths.items()):
            p = repo_root / rel
            if key not in frozen:
                res["absent"].append(
                    f"{key}: not present when the protocol was frozen")
            elif not p.exists():
                res["absent"].append(f"{key}: {rel} is absent from this clone")
            elif _sha256(p) != frozen[key]:
                res["mismatch"].append(
                    f"{key}: frozen={frozen[key][:16]} "
                    f"now={_sha256(p)[:16]}")
            else:
                res["match"].append(key)
        out[field] = res
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check-only", action="store_true",
                    help="Verify the committed freeze instead of writing it")
    ap.add_argument("--refreeze", action="store_true",
                    help="Amend an existing freeze, recording the amendment. "
                         "Refused once any Stage-3 adapter exists.")
    ap.add_argument("--reason", default=None,
                    help="Why the freeze is being amended (with --refreeze)")
    ap.add_argument("--repo-root", default=None)
    args = ap.parse_args()
    repo_root = Path(args.repo_root) if args.repo_root \
        else (_find_repo_root(Path.cwd()) or Path.cwd())
    out = repo_root / OUT_REPORT

    if args.check_only:
        reasons = verify_freeze(repo_root)
        clone = verify_clone_dependent(repo_root) if out.exists() else {}
        if reasons or any(v["mismatch"] for v in clone.values()):
            for r in reasons:
                print(f"  MISMATCH {r}")
            for field, res in clone.items():
                for m in res["mismatch"]:
                    print(f"  MISMATCH {field}: {m}")
            raise SystemExit(1)
        print(f"OK — the Stage-3 freeze still matches the repository "
              f"(volatile fields excluded: {list(VOLATILE_FIELDS)}; "
              f"clone-dependent fields verified separately: "
              f"{list(CLONE_DEPENDENT_FIELDS)})")
        for field, res in clone.items():
            print(f"  {field}: {len(res['match'])} match, "
                  f"{len(res['absent'])} absent from this clone")
            for a in res["absent"]:
                print(f"    absent: {a}")
        return

    #: Evaluated FIRST and unconditionally: no flag, including --refreeze,
    #: reaches past it.  Stage 3 trains, so the protocol must predate the
    #: adapters it governs.
    existing = adapters_exist(repo_root)
    if existing:
        raise SystemExit(
            f"REFUSING to freeze: {len(existing)} Stage-3 adapter(s) already "
            f"exist ({existing[:4]}). A protocol written now would postdate "
            f"the training it is supposed to govern, and could be fitted to "
            f"it -- which image weights to sweep and how the B6 control is "
            f"judged are choices whose answer changes once adapters are on "
            f"disk.")

    document = build_freeze(repo_root)
    if out.exists():
        previous = json.loads(out.read_text())
        if not args.refreeze:
            raise SystemExit(
                f"REFUSING to overwrite {out}. It already exists; pass "
                f"--refreeze --reason \"...\" to amend it deliberately.")
        if not args.reason:
            raise SystemExit(
                "REFUSING: --refreeze without --reason. An amendment whose "
                "reason is not recorded cannot be distinguished from a rule "
                "changed to fit its own result.")
        changed = []
        old_flat, new_flat = _flat(previous), _flat(document)
        for key in sorted(set(old_flat) | set(new_flat)):
            if any(key.startswith(f"{v}.") or key == v
                   for v in VOLATILE_FIELDS):
                continue
            if old_flat.get(key) != new_flat.get(key):
                changed.append(key)
        for section in ("protocol_paths", "depends_on_unchanged", "data_paths"):
            merged = dict(previous["hashes"].get(section, {}))
            merged.update(document["hashes"][section])
            document["hashes"][section] = merged
        for field in CLONE_DEPENDENT_FIELDS:
            merged = dict(previous.get(field, {}).get("sha256", {}))
            merged.update(document[field]["sha256"])
            document[field]["sha256"] = merged
        document["amendments"] = list(previous.get("amendments", [])) + [{
            "amended_at_utc": document["frozen_at_utc"],
            "reason": args.reason,
            "supersedes_sha256": _sha256(out),
            "fields_changed": changed,
            "num_fields_changed": len(changed),
            "no_adapter_existed_at_amendment_time": not existing,
        }]
        log.warning("AMENDING the Stage-3 freeze: %d field(s) change (%s)",
                    len(changed), ", ".join(changed[:8]))
    out.write_text(json.dumps(document, indent=2, ensure_ascii=False) + "\n")
    log.info("Stage-3 freeze -> %s", out)
    log.info("  %d protocol paths, %d dependencies, %d data paths bound",
             len(document["hashes"]["protocol_paths"]),
             len(document["hashes"]["depends_on_unchanged"]),
             len(document["hashes"]["data_paths"]))
    log.info("  B7 rows: %d trained; B6 zero-weight control holds: %s",
             document["grid"]["num_trained"],
             document["zero_weight_control"]["holds_at_freeze_time"])
    log.info("  the next Stage-3 adapter locks this protocol")


if __name__ == "__main__":
    main()
