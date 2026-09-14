"""Freeze the Iteration 12 Stage-2 protocol before a single adapter is trained.

    python scripts/freeze_iter12_stage2.py
    python scripts/freeze_iter12_stage2.py --check-only
    python scripts/freeze_iter12_stage2.py --refreeze --reason "..."

Stage 2 makes two decisions that could otherwise have been made after seeing
its result: what the retention floor is ANCHORED at, and which mechanisms are
trained.  Both are recorded in ``mllmu_iter12_stage2_basis.json``, and this
script hash-binds that document together with the code that implements it, so
neither the rule nor the mechanism can be edited between being written down and
being applied.

WHY THE REFUSAL IS KEYED ON ADAPTERS
------------------------------------
Stage 1c trained nothing, so its freeze refused once the SCORE existed.  Stage 2
trains, which restores the ordinary and stronger protection: the protocol must
predate the training.  The refusal here fires the moment any adapter exists
under ``data/checkpoints/mllmu_iter12_stage2/`` — the faithfulness control's
included, because the control is part of the study and a control run under an
unfrozen loop proves nothing about the frozen one.  It is evaluated first and
unconditionally, before any flag, so ``--refreeze`` cannot reach past it.

WHAT IS BOUND
-------------
* the eight protocol paths that implement Stage 2;
* the modules they import whose behaviour is part of the measurement, including
  four that earlier freezes already hash-bound and one that the confirmation
  freeze SEALED — imported and called here, never edited;
* the basis and calibration reports, the filed Stage-1/1b/1c reports, the
  knowledge-group jsonl files and the dataset parquets;
* separately, because they are gitignored and absent from a clone: the sha256 of
  the two prediction parquets Stage 2 REUSES (B0 and MG, read from Stage 1's
  directory rather than regenerated) and of the MF reference cache the anchor
  rows train against.

Both clone-dependent blocks report three outcomes per path — match, mismatch,
absent — rather than folding absence into a pass, so a clone says it has fewer
bytes to check instead of silently verifying less.

THE GENERATION CONTRACT IS INHERITED, NOT RESTATED
--------------------------------------------------
Stage 2 generates predictions for its own candidates and reads two of Stage 1's.
If those came from different instruments the comparison would span two
measurements, and the image route is precisely the one whose correctness flips
with batch composition — 15 flips out of 4,518 between two B0 parquets from
identical adapter bytes.  So the contract is read from the Stage-1 freeze and
bound here, and the selector refuses a run whose contract differs from it.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

#: Hashing, git state and document flattening come from the Stage-1 freeze, and
#: the parquet-naming and sidecar rules from Stage 1c's, so all four
#: Iteration-12 studies verify identically.  Importing also means a change to a
#: frozen file shows up here as a change in behaviour rather than as a silent
#: divergence between copies.
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
from granunlearn.logging_utils import setup_logger
from granunlearn.training import stage2_grid as sg
from granunlearn.training.candidate_grid import ITER12_LAM
from granunlearn.training.preservation_anchor import (
    CACHE_SIDECAR,
    CACHE_TENSOR,
    CACHE_VERSION,
    NLL_AGREEMENT_TOLERANCE,
)
from granunlearn.training.preservation_trainer import CAPPED_MODE

log = setup_logger("freeze_iter12_stage2")

OUT_REPORT = sg.FREEZE_REPORT
BASIS_REPORT = sg.BASIS_REPORT
CALIBRATION_REPORT = sg.CALIBRATION_REPORT
SELECTION_REPORT = sg.OUT_REPORT
STAGE1_FREEZE_REPORT = "data/reports/mllmu_iter12_selection_protocol_freeze.json"

#: The code that IMPLEMENTS Stage 2.  If one of these changes, the mechanism or
#: the criterion has changed, and --check-only must say so before another
#: candidate is trained.
PROTOCOL_PATHS = (
    "scripts/freeze_iter12_stage2.py",
    "scripts/build_iter12_stage2_basis.py",
    "scripts/build_mf_reference_logprobs.py",
    "scripts/train_iter12_stage2.py",
    "scripts/select_iter12_stage2.py",
    "src/granunlearn/training/stage2_grid.py",
    "src/granunlearn/training/preservation_anchor.py",
    "src/granunlearn/training/preservation_trainer.py",
)

#: Code Stage 2 DEPENDS ON but must not modify.  ``hierarchy_metrics`` and
#: ``select_unlearning_checkpoints`` are among the eighteen paths the
#: confirmation freeze sealed; ``candidate_grid``, ``retention_selection``,
#: ``unlearning_trainer``, ``unlearning_datasets``, ``reference_trainer`` and
#: ``state_datasets`` are bound by the Stage-1 freeze; ``seed_replication`` and
#: ``route_stratified_retention`` by Stage 1b's and 1c's.  All are imported,
#: never edited.
DEPENDS_ON_UNCHANGED = (
    "src/granunlearn/evaluation/retention_selection.py",
    "src/granunlearn/evaluation/route_stratified_retention.py",
    "src/granunlearn/evaluation/hierarchy_metrics.py",
    "src/granunlearn/evaluation/reference_eval.py",
    "src/granunlearn/evaluation/prediction_provenance.py",
    "src/granunlearn/evaluation/scoring.py",
    "src/granunlearn/training/candidate_grid.py",
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
)

#: Committed evidence the protocol is derived from.  Every prior Iteration-12
#: freeze and report is listed, so Stage 2 cannot outlive a change to any of
#: them.
DATA_PATHS = (
    BASIS_REPORT,
    CALIBRATION_REPORT,
    sg.ROUTE_STRATIFIED_REPORT,
    sg.STAGE1C_FREEZE_REPORT,
    STAGE1_FREEZE_REPORT,
    "data/reports/mllmu_iter12_retention_selection.json",
    "data/reports/mllmu_iter12_seed_replication_freeze.json",
    "data/reports/mllmu_iter12_seed_replication.json",
    "data/reports/mllmu_iter12_seed_replication.REPAIR.json",
    "data/reports/mllmu_iter12_retention_probe.json",
    "data/reports/mllmu_iter12_route_probe.json",
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
CACHE_FIELD = "reference_cache_bound"
CONTRACT_FIELD = "generation_contract_inherited"

#: Fields whose value depends on gitignored bytes being present.  Excluded from
#: the flat document comparison and checked by their own three-valued
#: verifiers, so a clone reports "fewer things to check" rather than either a
#: false mismatch or a silent pass over a shorter list.
CLONE_DEPENDENT_FIELDS = (PREDICTIONS_FIELD, CACHE_FIELD)

#: Sidecar keys that must be identical between the reused parquets and whatever
#: Stage 2 generates.  Inherited from the Stage-1c freeze rather than restated.
CONTRACT_MUST_MATCH = ("batch_size", "image_batch_size", "max_new_tokens",
                       "max_length", "max_image_pixels", "do_sample")

#: The Stage-1 states whose predictions Stage 2 reads rather than regenerates.
#: Named here and in the selector, which imports this tuple, so the two cannot
#: disagree about how many bytes the floor rests on.
REUSED_STATES: tuple[str, ...] = ("B0", "MG", sg.incumbent_row())


def reused_prediction_paths(repo_root: Path) -> dict[str, str]:
    """The Stage-1 parquets Stage 2 reads instead of regenerating.

    B0 supplies the reported baseline, MG supplies the decision-bearing anchor,
    and the incumbent Stage-1 row is the cap sweep's ``cap = infinity`` point --
    the same recipe with no cap at all, which already exists, so including it
    costs a sha256 and gives the sweep its own control.

    All three were bound by sha256 in the Stage-1c freeze under the single
    verified generation contract, so reusing them ties Stage 2's floor to bytes
    that are already hashed rather than to a fresh generation whose only purpose
    would be to reproduce them.
    """
    stage1 = sg.stage1_predictions_dir(repo_root)
    out = {}
    for state in REUSED_STATES:
        rel = str((stage1 / prediction_filename(state)).relative_to(repo_root))
        if rel.startswith("data/data/") or not rel.startswith("data/"):
            raise SystemExit(
                f"{state} resolved to {rel!r}, which is not a repo-relative "
                f"prediction path; the resolvers and this script disagree "
                f"about who supplies the data/ prefix")
        out[f"stage1.{state}"] = rel
    return out


def cache_paths(repo_root: Path) -> dict[str, str]:
    """The MF reference cache the anchor rows train against."""
    cdir = sg.reference_cache_dir(repo_root)
    out = {}
    for name in (CACHE_SIDECAR, CACHE_TENSOR):
        rel = str((cdir / name).relative_to(repo_root))
        if rel.startswith("data/data/"):
            raise SystemExit(f"{name} resolved to {rel!r}: a doubled prefix")
        out[name] = rel
    return out


def inherited_contract(repo_root: Path) -> dict[str, Any]:
    """The generation contract, read from the Stage-1 freeze.

    Read rather than retyped: a contract restated here could differ from the
    one the reused parquets were produced under, and the difference would be
    invisible until someone compared two instruments' numbers.
    """
    stage1_freeze = json.loads((repo_root / STAGE1_FREEZE_REPORT).read_text())
    contract = stage1_freeze["generation_contract"]
    stage1c = json.loads((repo_root / sg.STAGE1C_FREEZE_REPORT).read_text())
    verified = stage1c["measurement_contract"]
    return {
        "contract": contract,
        "keys_compared": sorted(contract),
        "read_from": STAGE1_FREEZE_REPORT,
        "stage1c_verified_one_instrument": verified["one_instrument"],
        "stage1c_distinct_contracts": verified["distinct_contracts"],
        "stage1c_bound_parquets": verified["num_bound_parquets"],
        "agrees_with_the_stage1c_verified_contract": all(
            verified["contract"].get(k) == v
            for k, v in contract.items()
            if k in CONTRACT_MUST_MATCH),
        "why_it_is_load_bearing": (
            "Stage 2 generates predictions for its own candidates and reads two "
            "of Stage 1's. The image route is the one whose correctness flips "
            "when batch composition changes -- 15 flips out of 4,518 between "
            "two B0 parquets from identical adapter bytes, all of them "
            "image-route -- so a contract difference between the reused and the "
            "generated parquets would put the anchor and the candidates on two "
            "instruments. The selector refuses a run whose contract differs."),
    }


def adapters_exist(repo_root: Path) -> list[str]:
    """Every adapter directory under the Stage-2 root, the control's included.

    One recursive glob rather than a glob per nesting depth: the control sits
    one level deeper than the candidates (``_control/<id>/adapters``), and a
    fixed-depth pattern would have missed it -- which is exactly the adapter
    that most needs to lock the freeze, since a control run under an unfrozen
    loop proves nothing about the frozen one.
    """
    root = sg.stage2_ckpt_root(repo_root)
    if not root.exists():
        return []
    return sorted(str(p.parent.relative_to(repo_root))
                  for p in root.glob("**/adapters") if p.is_dir())


def trained_yet(repo_root: Path) -> bool:
    return bool(adapters_exist(repo_root))


def build_freeze(repo_root: Path) -> dict[str, Any]:
    basis_path = repo_root / BASIS_REPORT
    if not basis_path.exists():
        raise SystemExit(
            f"REFUSING: {BASIS_REPORT} does not exist. The basis records the "
            f"two decisions this freeze exists to protect -- the anchor and the "
            f"mechanisms -- so freezing without it would bind a rule nobody "
            f"wrote down.")
    basis = json.loads(basis_path.read_text())
    calibration = json.loads((repo_root / CALIBRATION_REPORT).read_text())
    grid = sg.stage2_grid(calibration["grid_rule"]["beta_star"])
    kept, filtered = sg.validate_stage2_grid(grid)
    if kept:
        raise SystemExit(f"REFUSING: the Stage-2 grid is invalid: {kept}")

    paths = reused_prediction_paths(repo_root)
    present = {k: v for k, v in paths.items() if (repo_root / v).exists()}
    cpaths = cache_paths(repo_root)
    cpresent = {k: v for k, v in cpaths.items() if (repo_root / v).exists()}
    sidecar = json.loads((repo_root / cpaths[CACHE_SIDECAR]).read_text()) \
        if CACHE_SIDECAR in cpresent else {}

    anchor = basis["anchor_decision"]
    return {
        "iteration": "12",
        "stage": 2,
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(
            timespec="seconds"),
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": _git("status", "--porcelain") is not None,
        "amendments": [],
        "trains_seven_candidates": len(sg.trained_rows(grid)),
        "gpu_hours_estimated": (
            "about 10: seven trainings at 24-40 minutes each plus seven "
            "generations at a 40.2-minute median, on a box where only one GPU "
            "currently has the ~26 GiB the recipe needs"),

        # ── 1. the floor ──
        "what_is_frozen": {
            "floor_numbers": list(rt.FLOOR_NUMBER_KEYS),
            "num_numbers": len(rt.FLOOR_NUMBER_KEYS),
            "routes": [route for route, _ in rt.ROUTES],
            "route_families": {r: list(f) for r, f in rt.ROUTES},
            "estimands": list(rs.RETENTION_ESTIMANDS),
            "primary_anchor": anchor["anchor_state"],
            "primary_rule": (
                "candidate >= MG - epsilon on all eight numbers; a number "
                "missing on either side disqualifies"),
            "epsilon": rs.FLOOR_EPSILON,
            "margin": 0.0,
            "no_margin_is_used": True,
            "epsilon_is_a_tolerance_not_a_margin": True,
            "direction": rs.FLOOR_DIRECTION,
            "anchor_values": anchor["anchor_values"],
            "reported_baseline": anchor["reported_baseline_state"],
            "reported_baseline_values": anchor["reported_baseline_values"],
            "reported_baseline_is_not_decision_bearing": True,
            "computed_by": (
                "route_stratified_retention.floor_check_stratified -- the same "
                "frozen function Stage 1c used, with MG's vector passed as the "
                "baseline. That function is generic in its baseline, so no new "
                "metric code was written and none was edited."),
            "denominators": anchor["denominators"],
            "scope_splits": list(rt.SCOPE_SPLITS),
            "pooled_reported_but_not_floored": list(rt.POOLED_FAMILIES),
            "among_eligible": (
                "minimise D_G with the frozen tie-break, exactly as Stage 1 "
                "did. The floor decides eligibility; D_G only orders the "
                "eligible."),
        },

        # ── 2. why the anchor moved, and what that does NOT do ──
        "anchor_change": {
            #: Constant rather than ``not trained_yet(repo_root)``.  The claim
            #: is true because ``main`` REFUSES to write this document once any
            #: Stage-2 adapter exists, so re-evaluating it later would report a
            #: guarantee the refusal already provides as though it had been
            #: withdrawn -- and --check-only would then mismatch forever after
            #: the control ran.
            "decided_before_any_stage2_adapter": True,
            "enforced_by": (
                "main() evaluates adapters_exist() first and unconditionally, "
                "before any flag including --refreeze"),
            "pre_registered_by": anchor["pre_registered_by"],
            "matched_comparison": anchor["matched_comparison"],
            "the_oracle_fails_the_frozen_floor":
                anchor["the_oracle_fails_the_frozen_floor"],
            "re_anchoring_changes_no_filed_verdict":
                anchor["re_anchoring_changes_no_filed_verdict"],
            "the_new_floor_does_not_licence_inaction":
                anchor["the_new_floor_does_not_licence_inaction"],
            "the_frozen_stage1c_floor_is_untouched": (
                "route_stratified_retention.py and analyze_iter12_route_"
                "stratified.py are hash-bound by the Stage-1c freeze and are "
                "not edited. The filed Stage-1, Stage-1b and Stage-1c reports "
                "are byte-identical, and their B0-anchored verdicts stand as "
                "filed."),
            "a_defect_left_in_place": anchor["a_defect_left_in_place"],
        },

        # ── 3. the mechanisms ──
        "mechanisms": {
            "preregistered": basis["mechanism_decision"][
                "preregistered_mechanism"],
            "preregistered_by": basis["mechanism_decision"]["preregistered_by"],
            "trained": basis["mechanism_decision"]["mechanisms_trained"],
            "capped_mode": CAPPED_MODE,
            "capped_term": (
                "-lambda * min(NLL(fine_target), cap), whose gradient is "
                "exactly zero above the cap; verified on a stub model before "
                "this freeze was written"),
            "fine_suppression_weight": ITER12_LAM,
            "anchor_term": "beta * KL(p_MF || p_theta) over supervised positions",
            "anchor_direction": sidecar.get("kl_direction"),
            "beta_star": calibration["grid_rule"]["beta_star"],
            "beta_star_derivation": calibration["grid_rule"],
            "why_the_anchor_is_trained_anyway": (
                "The calibration argues it cannot work: its optimum is MF, "
                "which is no unlearning, and states at equal drift from MF "
                "differ in retention by far more than the floor resolves. That "
                "is an observational argument over eight points. A mechanism "
                "this study was preregistered with gets run rather than argued "
                "away, and if the argument is wrong the run is what finds out."),
        },

        # ── 4. the grid ──
        "grid": {
            "num_rows": len(grid),
            "num_trained": len(sg.trained_rows(grid)),
            "rows": [c.describe() for c in grid],
            "cap_values": list(sg.STAGE2_CAPS),
            "cap_units": "nats of per-token NLL on the fine completion",
            "cap_derived_from_the_oracle": False,
            "oracle_position_reported_not_used":
                calibration["measurement"]["adapters"]["MG"][
                    "nll_by_group"]["fine_target"],
            "uncapped_point_of_the_sweep": sg.incumbent_row(),
            "the_frozen_validator_was_run": {
                "residual_errors": kept,
                "filtered_messages": filtered,
                "filtered_gap_substrings": list(sg.FROZEN_VALIDATOR_GAP),
                "filtered_because": (
                    "candidate_grid.py is hash-bound, so its METHODS tuple and "
                    "its mode vocabulary were frozen before Stage 2 existed and "
                    "cannot be extended. Only 'unknown method' and 'bad mode' "
                    "are filtered, by exact substring, and the residual must be "
                    "empty -- so every other structural check the frozen "
                    "validator makes still applies to this grid."),
            },
        },

        # ── 5. faithfulness ──
        "faithfulness_control": {
            "what_it_is": (
                "train_with_preservation is a second copy of a training loop, "
                "because unlearning_trainer.train_unlearning is hash-bound and "
                "cannot grow a mode. --control trains the incumbent Stage-1 "
                "objective THREE times in ONE process under ONE pinned "
                "PYTHONHASHSEED: A through the frozen loop, A2 through the "
                "frozen loop again, B through the new loop with every group in "
                "a plain sft/gd mode and neither a cap nor an anchor."),
            "reproduces_the_objective_of": sg.incumbent_row(),
            "spec_read_from": "the frozen Stage-1 grid, not retyped",
            "the_gate": (
                "A2 exists to MEASURE the noise floor rather than assume it: "
                "gap(A, A2) is how far two runs of the same code land apart on "
                "this stack. The gate is that gap(A, B) does not exceed it -- "
                "and bitwise equality is demanded outright when the floor is "
                "exactly zero, which is what one process and a pinned seed are "
                "supposed to buy. A sha256 alone cannot express this: it says "
                "'different' and stops, so it cannot distinguish float noise "
                "from a different objective, which is the whole question. The "
                "control therefore compares per-tensor gaps and reports the "
                "worst tensor."),
            "it_is_a_gate_not_a_report": (
                "scripts/train_iter12_stage2.py refuses to train a single "
                "candidate unless the control passed, so a drift between the "
                "two loops stops the study instead of being interpreted as a "
                "difference between mechanisms."),
            "cpu_half_already_verified": (
                "tests/unit/test_iteration12_stage2.py drives both loops "
                "against one stub model and asserts bit-identical parameters, "
                "forward-call counts, optimizer steps and epoch summaries, so "
                "a divergence is caught without a GPU as well."),
            "control_dir_is_inside_the_refusal": (
                "The control's adapters live under the Stage-2 checkpoint root, "
                "so running it locks this freeze against amendment. That is "
                "deliberate: a control run under an unfrozen loop proves "
                "nothing about the frozen one."),
            "a_criterion_this_replaced": (
                "The control as first frozen compared ONE run of the new loop "
                "against the adapter Stage 1 FILED, by sha256 over the whole "
                "directory. It ran, and refused: identical recipe in all "
                "fifteen fields, identical init adapter, identical 230 "
                "optimizer steps, and adapter_config.json differing only in the "
                "ORDER of the same seven target modules. That order is not "
                "reproducible. PeftConfig.target_modules is a Python set, and "
                "Python randomises the string hash seed per process unless "
                "PYTHONHASHSEED is exported, so the serialised order -- and the "
                "LoRA injection order behind it, which decides which module "
                "draws which dropout mask -- varies run to run. Four seeds were "
                "measured to give four different orders, a pinned seed gives "
                "one order every time, and the filed order matched none of the "
                "seven seeds tried, so the process that wrote it is not "
                "recoverable."),
            "the_evidence_that_it_was_the_stack_not_the_new_loop": (
                "The FROZEN loop, unedited and hash-bound, was re-run on the "
                "same spec from the same init adapter. It did not reproduce the "
                "filed adapter either, and landed the same distance from it: "
                "max per-tensor weight gap 1.726e-03 for the frozen re-run "
                "against filed, 1.812e-03 for the new loop against filed, "
                "1.800e-03 between the two re-runs -- three gaps of one order, "
                "so the new loop is no further from the filed adapter than the "
                "frozen loop is from itself. All three adapters hold the same "
                "seven target modules in three different orders. Both runs are "
                "kept under outputs/superseded/iter12_stage2_control_v1/, whose "
                "README.md records the orders and the gaps."),
            "why_replacing_it_is_not_moving_the_goalposts": (
                "The retired criterion was unsatisfiable by construction, so it "
                "tested the interpreter's hash seed rather than the two loops, "
                "and a gate that cannot pass says nothing when it fails. The "
                "replacement asks the question the original was written to ask "
                "-- do the two copies compute the same function? -- and asks it "
                "in the only way that has an answer, with the environment held "
                "still and the noise floor measured instead of assumed. The "
                "filed adapter is still loaded, compared and REPORTED beside "
                "the gate, so its gap stays visible; it is simply not what the "
                "gate reads. Nothing about the anchor, the eight numbers, the "
                "epsilon, the tie-break, the grid or the mechanisms moved: the "
                "control produced no candidate, no prediction, no retention "
                "number and no score, so there was no result for any of them to "
                "be fitted to. What changed is a fact about the apparatus. The "
                "failed control's directory was MOVED to outputs/superseded "
                "rather than deleted, so the adapter that demonstrated the "
                "defect is still on disk and inspectable."),
            "the_hash_seed_is_pinned_by_the_lane_script": (
                "scripts/lanes/iter12_stage2.sh exports PYTHONHASHSEED, and "
                "train_iter12_stage2.py refuses to run the control at all "
                "without it, because a control under a randomised seed measures "
                "the seed."),
            "what_pins_this_in_the_suite": [
                "test_the_control_pins_the_hash_seed_before_it_trains_anything",
                "test_an_unpinned_hash_seed_refuses_rather_than_measuring_the_"
                "interpreter",
                "test_the_control_runs_the_frozen_loop_twice_and_the_new_loop_"
                "once",
                "test_the_gate_never_passes_unconditionally",
                "test_the_gate_compares_the_right_pair_of_runs_on_each_side",
                "test_the_filed_adapter_is_reported_and_never_gates",
                "test_main_refuses_to_train_when_the_control_did_not_pass",
                "test_the_freeze_discloses_the_criterion_the_control_replaced",
            ],
            "why_those_tests_are_listed_here": (
                "Replacing this control's criterion widened what a later "
                "amendment may touch to include faithfulness_control.*, so the "
                "seal needs something holding the other side. These eight tests "
                "are that. They are STRUCTURAL rather than behavioural because "
                "the script imports the trainers at module level and CI "
                "installs a closure without torch: each parses "
                "scripts/train_iter12_stage2.py and asserts which branch does "
                "what -- that the hash seed is checked first, that three runs "
                "happen with the frozen loop twice, that each side of the gate "
                "is built from the right pair of them, that the gate has exactly "
                "three branches and never assigns a bare True, that the filed "
                "adapter never reaches it, and that main refuses to train both "
                "when the gate failed and when the control was never run. "
                "Naming them here rather than describing them in prose makes "
                "the list checkable against the test file, and makes a later "
                "amendment that quietly drops one visible as a change to this "
                "field. One of them is not purely structural: "
                "test_an_unpinned_hash_seed_refuses_rather_than_measuring_the_"
                "interpreter lifts that single function out by "
                "ast.get_source_segment and EXECUTES it, because 'there is a "
                "raise statement somewhere in here' passes just as happily for "
                "'if False: raise'."),
        },

        # ── 6. what is never read ──
        "evidence_base": {
            "used": [
                "data/mllmu_hier_pilot100/queries.parquet, splits train+val",
                "data/mllmu_hier_pilot100/associations.parquet",
                "the fit-half groups of unlearning_iter12/",
                BASIS_REPORT,
                CALIBRATION_REPORT,
                sg.ROUTE_STRATIFIED_REPORT,
            ],
            "never_read": list(rs.FORBIDDEN_EVIDENCE),
            "never_read_enforced_by": (
                "scripts/train_iter12_stage2.py and "
                "scripts/select_iter12_stage2.py refuse any argument that "
                "resolves inside those paths, on RESOLVED paths rather than on "
                "strings, and write only under the Stage-2 checkpoint and "
                "predictions directories."),
            "probe_half_never_rehearsed": (
                "The preserve group covers 0 of the 204 probe-half "
                "associations, recorded in the basis and re-checked by the "
                "matched comparison. An anchor fitted on probe-half prompts "
                "would rehearse the quantity the floor measures."),
        },

        "basis": {
            "report": BASIS_REPORT,
            "sha256": _sha256(basis_path),
            "reads_no_prediction_parquet":
                basis["reads_no_prediction_parquet"],
            "re_derives_with_check_only": True,
            "calibration_report": CALIBRATION_REPORT,
            "calibration_sha256": _sha256(repo_root / CALIBRATION_REPORT),
            "calibration_controls": basis["mechanism_decision"]["controls"],
            "nll_agreement_max":
                basis["mechanism_decision"]["nll_agreement_max"],
            "nll_agreement_tolerance": NLL_AGREEMENT_TOLERANCE,
            "cache_version": CACHE_VERSION,
        },

        "measurement_limits": basis["measurement_limits"],

        "extends": {
            "stage1_freeze": STAGE1_FREEZE_REPORT,
            "stage1b_freeze": "data/reports/"
                              "mllmu_iter12_seed_replication_freeze.json",
            "stage1c_freeze": sg.STAGE1C_FREEZE_REPORT,
            "stage1c_report": sg.ROUTE_STRATIFIED_REPORT,
            "note": (
                "Stage 2 extends the Stage-1c floor to a different anchor and "
                "reuses two of the parquets that freeze bound. It re-decides "
                "nothing that was filed: the B0-anchored verdicts stand, and "
                "they are recomputed and reported beside the new ones."),
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
                "All three parquets are bound by sha256 in the Stage-1c freeze "
                "under the single verified contract. Regenerating them would "
                "spend about two GPU-hours producing a second draw from "
                "identical adapter bytes, and would put fresh files where "
                "hashed evidence already sits. The selector re-derives each "
                "state's eight numbers from the reused bytes and refuses to "
                "write unless all of them equal the values Stage 1c filed, so "
                "reuse is checked rather than trusted."),
            "note": (
                "Gitignored and therefore absent from a clone. Verified apart "
                "from the rest of the freeze, reporting match, mismatch or "
                "absent per path."),
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
            "why_bound": (
                "The anchor rows train against these bytes. A cache rebuilt "
                "from a different MF adapter, or at a different truncation "
                "length, would preserve a different function, and the "
                "resulting adapter would look perfectly ordinary."),
            "note": (
                "Gitignored: 1.48 GB of fp32 log-probabilities. Verified apart "
                "from the rest of the freeze, reporting match, mismatch or "
                "absent per path."),
        },

        CONTRACT_FIELD: inherited_contract(repo_root),

        "disclosure": {
            "the_anchor_was_changed_after_stage1_was_scored": True,
            "not_blind": True,
            "why_that_is_not_a_goalpost_move": (
                "Three properties, all computed in the basis rather than "
                "asserted: the new anchor is justified by a comparison that "
                "needs no prediction at all (MG and MF have byte-identical "
                "retained training data); it changes no filed verdict, because "
                "Stage 1 is 0 of 8 eligible under either anchor; and it does "
                "not licence inaction, because B0 itself fails it on 2 of 8."),
            "the_mechanism_was_changed_after_the_calibration": True,
            "the_calibration_reads_no_retention_number":
                calibration["measurement"]["reads_no_retention_number"],
            "the_cap_range_was_chosen_after_measuring_the_oracle": True,
            "cap_range_disclosure": basis["cap_grid"]["why_this_range"],
            "direction_is_against_interest": (
                "The new anchor is HARDER on the rows that matter: the "
                "candidates that came closest under B0 are not the ones that "
                "come closest under MG, and the cap sweep's own uncapped point "
                "is an existing Stage-1 row that fails."),
            "what_this_legitimately_preregisters": (
                "Stage 3: the next mechanism is judged on all eight numbers "
                "against the anchor frozen here, and its protocol is frozen "
                "before it is trained."),
        },
    }


# ── verification ────────────────────────────────────────────────────────

def verify_freeze(repo_root: Path, path: Path | None = None) -> list[str]:
    """Recompute the freeze against the repository.  Empty list == matches.

    The prediction and cache blocks are excluded here and verified by
    :func:`verify_clone_dependent`, because those files are gitignored and
    their absence from a clone is not a mismatch.
    """
    path = path or repo_root / OUT_REPORT
    if not path.exists():
        return [f"{path} does not exist -- Stage 2 is not frozen"]
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
    """Three-valued check of the gitignored bytes Stage 2 depends on."""
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
                    f"{key}: frozen={frozen[key][:16]} now={_sha256(p)[:16]}")
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
                         "Refused once any Stage-2 adapter exists.")
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
        print(f"OK — the Stage-2 freeze still matches the repository "
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
    #: reaches past it.  Stage 2 trains, so the ordinary protection applies --
    #: the protocol must predate the adapters it governs.  The control's adapter
    #: counts, because a control run under an unfrozen loop proves nothing about
    #: the frozen one.
    existing = adapters_exist(repo_root)
    if existing:
        raise SystemExit(
            f"REFUSING to freeze: {len(existing)} Stage-2 adapter(s) already "
            f"exist ({existing[:4]}). A protocol written now would postdate the "
            f"training it is supposed to govern, and could be fitted to it -- "
            f"which anchor to use, which mechanisms to sweep, and how wide the "
            f"grid is, are all choices whose answer changes once adapters are "
            f"on disk. The faithfulness control's adapter counts too.")

    document = build_freeze(repo_root)
    if out.exists():
        previous = json.loads(out.read_text())
        if not args.refreeze:
            raise SystemExit(
                f"REFUSING to overwrite {out}. It already exists; pass "
                f"--refreeze --reason \"...\" to amend it deliberately. A "
                f"silent rewrite is how a preregistration stops being one.")
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
        #: The previous document's hashes are carried forward and MERGED, so an
        #: amendment cannot drop a path that was bound before it.  A byte that
        #: was hashed and then quietly unhashed is worse than one never hashed.
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
        log.warning("AMENDING the Stage-2 freeze: %d field(s) change (%s)",
                    len(changed), ", ".join(changed[:8]))
    with open(out, "w") as f:
        json.dump(document, f, indent=2, ensure_ascii=False)
    log.info("Stage-2 freeze -> %s", out)
    log.info("  %d protocol paths, %d dependencies, %d data paths bound",
             len(document["hashes"]["protocol_paths"]),
             len(document["hashes"]["depends_on_unchanged"]),
             len(document["hashes"]["data_paths"]))
    log.info("  anchor: %s on %d numbers | rows: %d (%d trained)",
             document["what_is_frozen"]["primary_anchor"],
             document["what_is_frozen"]["num_numbers"],
             document["grid"]["num_rows"], document["grid"]["num_trained"])
    log.info("  this freeze can no longer be amended: the next adapter trained "
             "under it locks the protocol")


if __name__ == "__main__":
    main()
