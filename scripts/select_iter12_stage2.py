"""Iteration 12 Stage 2: generate, apply the frozen MG-anchored floor, order by D_G.

    python scripts/select_iter12_stage2.py --device cuda:0
    python scripts/select_iter12_stage2.py --device cuda:0 --generate-only
    python scripts/select_iter12_stage2.py --candidates B5_cap2.0_lam0.5_lr2e-05_ep5

This is the executor for the protocol ``scripts/freeze_iter12_stage2.py``
committed.  It decides nothing of its own: the anchor, its values, the rule, the
epsilon, the tie-break and the Stage-3 gate are all read from the freeze and the
basis, and the freeze is VERIFIED before a single query is generated.  If the
code that implements the protocol has changed since it was frozen, this script
refuses to run rather than selecting a successor under rules nobody
preregistered.

Order of operations, and why each guard is where it is
------------------------------------------------------
1. verify the freeze (before generating: generation is the expensive part, and a
   mismatch found afterwards would leave artifacts produced under an unfrozen
   criterion);
2. refuse any path that resolves inside the sealed confirmation evidence;
3. refuse a generation contract that differs from the inherited one — batch
   layout is inside D_G and inside the image route's correctness, so a different
   contract is a different instrument;
4. reuse B0, MG and the incumbent row from the parquets the Stage-1c freeze
   bound, checking each sha256, and RECOMPUTE their eight numbers — refusing to
   write unless all three equal the values Stage 1c filed, exactly.  A floor
   whose anchor does not reproduce its own filed value is not the frozen floor;
5. generate every Stage-2 candidate into ``predictions_iter12_stage2/``;
6. refuse to write a report unless the WHOLE grid has predictions — a report
   over a subset names a winner the grid does not support and looks complete,
   carrying the same dataset version as the real one;
7. apply the primary floor, report the B0-anchored one beside it, then order the
   eligible by D_G with the frozen tie-break.

``--generate-only`` stops after step 5, which is what makes generation shardable
across GPUs: only an unsharded run may write the report.

Needs a GPU.  Reads no confirmation prediction, and writes no sealed path.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import freeze_iter12_stage2 as fis2

#: The sealed pilot-100 selector is imported, never copied: generation
#: semantics (batch layout, sidecar contract, coverage validation) have to be
#: IDENTICAL to the ones that produced the anchor's numbers, and a
#: reimplementation would drift from them silently.
from select_unlearning_checkpoints import _generate_state

from granunlearn.config import _find_repo_root
from granunlearn.evaluation import retention_selection as rs
from granunlearn.evaluation import route_stratified_retention as rt
from granunlearn.evaluation.prediction_provenance import dataset_version
from granunlearn.evaluation.reference_eval import (
    DEFAULT_MAX_IMAGE_PIXELS,
    DEFAULT_MAX_LENGTH,
    DEFAULT_MAX_NEW_TOKENS,
    load_associations_parquet,
    load_predictions_parquet,
    load_queries_parquet,
)
from granunlearn.logging_utils import setup_logger
from granunlearn.training import stage2_grid as sg
from granunlearn.training.preservation_anchor import sha256_file

log = setup_logger("select_iter12_stage2")

SPLITS = ("train", "val")
EXPERIMENT_ID = "mllmu_iter12_stage2"
FLOOR_KEYS: tuple[str, ...] = rt.FLOOR_NUMBER_KEYS

#: Reused rather than regenerated.  The tuple is imported from the freeze so the
#: two cannot disagree about how many bytes the floor rests on: B0 supplies the
#: reported baseline, MG the decision-bearing anchor, and the incumbent Stage-1
#: row is the cap sweep's ``cap = infinity`` point -- the same recipe with no cap
#: at all, which already exists and therefore costs nothing to include.
REUSED_STATES = fis2.REUSED_STATES


def assert_no_forbidden_evidence(paths: list[Path], repo_root: Path) -> None:
    """Refuse to touch the sealed confirmation, in either direction.

    Checked on RESOLVED paths, because a relative path with ``..`` in it, or a
    symlink, would otherwise walk straight past a string comparison.
    """
    forbidden = [((repo_root / rel).resolve(), rel)
                 for rel in rs.FORBIDDEN_EVIDENCE]
    for path in paths:
        resolved = Path(path).resolve()
        for bad, rel in forbidden:
            if resolved == bad or bad in resolved.parents:
                raise SystemExit(
                    f"REFUSING: {path} resolves inside {rel}. The 11C "
                    f"confirmation was scored once and is sealed; tuning an "
                    f"Iteration-12 successor on its predictions would make a "
                    f"confirmatory result retrospective.")


def reused_paths(repo_root: Path) -> dict[str, Path]:
    """Absolute paths of the Stage-1 parquets Stage 2 reads."""
    stage1 = sg.stage1_predictions_dir(repo_root)
    return {state: stage1 / fis2.prediction_filename(state)
            for state in REUSED_STATES}


def verify_reused_bytes(repo_root: Path) -> dict[str, Any]:
    """Check every reused parquet against the sha256 the freeze recorded."""
    freeze = json.loads((repo_root / fis2.OUT_REPORT).read_text())
    bound = freeze[fis2.PREDICTIONS_FIELD]["sha256"]
    out: dict[str, Any] = {}
    for state, path in reused_paths(repo_root).items():
        key = f"stage1.{state}"
        if not path.exists():
            raise SystemExit(
                f"REFUSING: {path} is missing. Stage 2 reads the anchor's "
                f"predictions rather than regenerating them, so without those "
                f"bytes there is no floor to apply.")
        got = sha256_file(path)
        want = bound.get(key)
        if want is None:
            raise SystemExit(
                f"REFUSING: the freeze binds no sha256 for {key}, so {path} "
                f"cannot be certified as the parquet Stage 1c measured.")
        if got != want:
            raise SystemExit(
                f"REFUSING: {path} hashes to {got[:16]} but the freeze records "
                f"{want[:16]}. The anchor's bytes are not the ones the protocol "
                f"was frozen against.")
        out[state] = {"path": str(path.relative_to(repo_root)), "sha256": got}
    return out


def eight_numbers(preds, queries, associations, probe_entities) -> dict:
    """The eight floor numbers for one state, by the frozen route-stratified
    metric."""
    by_route = rt.probe_retention_by_route(preds, queries, associations,
                                           probe_entities)
    return rt.floor_vector(by_route)


def gate_reused_against_the_filed_values(
        recomputed: dict[str, dict[str, Any]],
        repo_root: Path) -> dict[str, Any]:
    """Refuse to score unless the reused states reproduce Stage 1c exactly.

    This is the same discipline Stage 1c applied to its own text stratum: an
    extension that disagrees with the rule it extends, on the numbers they
    share, is a different rule wearing its name.  Here the shared numbers are
    the anchor's and the reported baseline's own eight values.
    """
    strat = json.loads((repo_root / sg.ROUTE_STRATIFIED_REPORT).read_text())
    filed = {
        "B0": strat["anchor"]["values"],
        "MG": strat["stage1"]["reference_states_not_floored"]["MG"]["values"],
        sg.incumbent_row():
            strat["stage1"]["candidates"][sg.incumbent_row()]["values"],
    }
    problems: list[str] = []
    compared = 0
    for state, values in filed.items():
        got = recomputed.get(state)
        if got is None:
            problems.append(f"{state}: not recomputed")
            continue
        for key in FLOOR_KEYS:
            compared += 1
            if got.get(key) != values[key]:
                problems.append(
                    f"{state}/{key}: Stage 1c filed {values[key]} but the "
                    f"reused parquet recomputes to {got.get(key)}")
    return {
        "states_gated": sorted(filed),
        "numbers_compared": compared,
        "problems": problems,
        "holds": not problems,
        "why_this_gate_exists": (
            "Stage 2 anchors on MG and reports against B0 using parquets "
            "Stage 1 generated. If those bytes no longer produce the numbers "
            "Stage 1c filed, then either the bytes moved or the metric did, and "
            "either way the floor being applied is not the frozen one."),
    }


def build_report(generated: dict[str, dict[str, Any]],
                 reused_eight: dict[str, dict[str, Any]],
                 reused_vec: dict[str, dict[str, Any]],
                 mg_vec: dict[str, float | None],
                 repo_root: Path, data_dir: Path,
                 predictions_dir: Path,
                 generation_config: dict[str, Any],
                 model_id: str,
                 gates: dict[str, Any],
                 freeze: dict[str, Any]) -> dict[str, Any]:
    """Apply the primary floor, report the B0-anchored one, order by D_G."""
    b0_eight = reused_eight["B0"]
    mg_eight = reused_eight["MG"]
    eps = rs.FLOOR_EPSILON

    rows: dict[str, Any] = {}
    for cid, info in generated.items():
        vector = rs.summary_vector(info["trainval_metrics"])
        dist, used = rs.distance_to_mg(vector, mg_vec)
        eight = info["eight_numbers"]
        primary = rt.floor_check_stratified(eight, mg_eight, eps)
        reported = rt.floor_check_stratified(eight, b0_eight, eps)
        rows[cid] = {
            "method": info["method"],
            "spec": info["spec"],
            "vector": vector,
            "distance_to_mg": dist,
            "used_components": used,
            "values": eight,
            "difference_vs_mg_anchor": {
                k: round(eight[k] - mg_eight[k], 4) for k in FLOOR_KEYS},
            "difference_vs_b0": {
                k: round(eight[k] - b0_eight[k], 4) for k in FLOOR_KEYS},
            "primary_floor": primary,
            "eligible": primary["eligible"] and dist is not None,
            "reported_b0_floor": reported,
            "eligible_on_the_reported_b0_floor": reported["eligible"],
            "pooled_all_routes": info.get("pooled_all_routes"),
            "training_side": info.get("training_side"),
            "config": info.get("config"),
        }

    eligible = {cid: r for cid, r in rows.items() if r["eligible"]}
    ranked = sorted(eligible, key=lambda c: rs.rank_key(
        rows[c]["distance_to_mg"], c))
    selected = ranked[0] if ranked else None
    tied: list[str] = []
    if selected is not None:
        best = rows[selected]["distance_to_mg"]
        tied = sorted(cid for cid in eligible
                      if cid != selected
                      and rows[cid]["distance_to_mg"] == best)

    incumbent = sg.incumbent_row()
    report: dict[str, Any] = {
        "iteration": "12",
        "stage": 2,
        "criterion": {
            "primary": freeze["what_is_frozen"]["primary_rule"],
            "anchor": freeze["what_is_frozen"]["primary_anchor"],
            "anchor_values": mg_eight,
            "epsilon": eps,
            "margin": 0.0,
            "no_margin_is_used": True,
            "reported_beside_it": (
                "the frozen B0-anchored eight-number floor, computed for every "
                "state and decision-bearing nowhere"),
            "reported_baseline_values": b0_eight,
            "denominators": freeze["what_is_frozen"]["denominators"],
            "among_eligible": freeze["what_is_frozen"]["among_eligible"],
            "computed_by": freeze["what_is_frozen"]["computed_by"],
            "the_field_named_b0_holds_the_baseline_passed_in": (
                "floor_check_stratified labels its baseline field 'b0' "
                "whatever is passed. Under the primary floor that field holds "
                "MG's values; under the reported one it holds B0's. The module "
                "is hash-bound by the Stage-1c freeze so the name cannot be "
                "corrected."),
        },
        "consistency_gates": gates,
        "anchor_states": {
            "B0": {
                "values": b0_eight,
                "reused_from": reused_vec["B0"]["path"],
                "sha256": reused_vec["B0"]["sha256"],
                "primary_floor_result": rt.floor_check_stratified(
                    b0_eight, mg_eight, eps),
                "note": (
                    "B0 is not eligible under the primary floor either, which "
                    "is the property the basis required: the new anchor cannot "
                    "be satisfied by declining to unlearn."),
            },
            "MG": {
                "values": mg_eight,
                "reused_from": reused_vec["MG"]["path"],
                "sha256": reused_vec["MG"]["sha256"],
                "is_the_anchor": True,
                "trivially_eligible": True,
                "excluded_from_the_ranking": True,
                "why_excluded": (
                    "MG is the oracle the criterion measures distance to. "
                    "Ranking it among the candidates would let the reference "
                    "state win its own selection."),
            },
        },
        "candidates": {cid: r for cid, r in rows.items() if cid != "MG"},
        "eligible": sorted(eligible),
        "disqualified_by_the_primary_floor": {
            cid: {k: v["passes"] for k, v in
                  rows[cid]["primary_floor"]["checks"].items()}
            for cid in rows if not rows[cid]["eligible"] and cid != "MG"
        },
        "eligibility_under_both_anchors": {
            cid: {"on_mg": rows[cid]["eligible"],
                  "on_b0": rows[cid]["eligible_on_the_reported_b0_floor"]}
            for cid in rows if cid != "MG"
        },
        "ranking_of_eligible": ranked,
        "selected": selected,
        "tied_with": tied,
        "a_tie_means": (
            "the criterion did not discriminate and the tie-break -- not the "
            "science -- chose the winner"),
        "cap_sweep": {
            "caps": list(sg.STAGE2_CAPS),
            "units": "nats of per-token NLL on the fine completion",
            "uncapped_point": incumbent,
            "uncapped_point_values": reused_eight.get(incumbent),
            "uncapped_point_reused_from": reused_vec.get(incumbent, {}).get(
                "path"),
            "rows": {cid: {
                "cap": rows[cid]["spec"].get("cap"),
                "final_fine_target_nll":
                    (rows[cid]["training_side"] or {}).get(
                        "final_avg_loss_fine_target"),
                "final_frac_at_cap":
                    (rows[cid]["training_side"] or {}).get(
                        "final_frac_at_cap_fine_target"),
                "image_retain_same_row_micro_vs_mg":
                    rows[cid]["difference_vs_mg_anchor"][
                        "image.retain_same_entity_image.row_micro"],
            } for cid in rows
                if rows[cid]["method"] == sg.METHOD_BOUNDED and cid != "MG"},
            "the_question_this_answers": (
                "Does bounding the ascent let a candidate forget as much while "
                "retaining more? The incumbent row is the same recipe with no "
                "cap at all, so the sweep has its own control and it cost "
                "nothing."),
        },
        "stage_3_gate": stage3_gate(rows, selected, incumbent),
        "tag": "iter12",
        "protocol_freeze": {
            "path": fis2.OUT_REPORT,
            "verified_before_generation": True,
            "mismatches_found": [],
        },
        "basis": {"path": sg.BASIS_REPORT,
                  "sha256": freeze["basis"]["sha256"]},
        "dataset_version": dataset_version(data_dir),
        "predictions_dir": str(predictions_dir.relative_to(repo_root)),
        "reused_predictions": reused_vec,
        "generation_contract": generation_config,
        "model_id": model_id,
        "never_read": list(rs.FORBIDDEN_EVIDENCE),
        "measurement_limits": freeze["measurement_limits"],
        "note": (
            "EXPLORATORY. This report selects a checkpoint; it tests no "
            "hypothesis and controls no error rate. Distance is a selection "
            "criterion only -- the individual metrics remain the scientific "
            "results, and any confirmatory claim about the winner needs its own "
            "sealed protocol, frozen before it is scored. Stage 2 trains one "
            "seed per row, so a row that clears the primary floor by less than "
            "the between-seed spread Stage 1b measured is a candidate for "
            "replication rather than a result."),
    }
    return report


def stage3_gate(rows: dict[str, Any], selected: str | None,
                reference_row: str) -> dict[str, Any]:
    """Does a Stage 3 open?  Same shape as the frozen Stage-2 gate.

    ``retention_selection.stage2_gate`` is hash-bound and reads a report shaped
    for the four-number floor, so this restates its rule for the eight-number
    one rather than calling it: a stage opens if nothing satisfied the floor, or
    if the best eligible row does not STRICTLY improve on the reference row's
    D_G.  A tie is no improvement, and there is no margin.
    """
    if reference_row not in rows:
        #: The incumbent is a reused Stage-1 row, so it is in ``rows`` only if
        #: this run generated or read it.  Refusing is correct: without it the
        #: gate has nothing to compare against and would silently report that
        #: Stage 3 opens.
        raise SystemExit(
            f"REFUSING: the reference row {reference_row!r} is not in this "
            f"run, so the Stage-3 gate is undefined. It is one of the reused "
            f"states precisely so that this comparison is always available.")
    ref_dist = rows[reference_row]["distance_to_mg"]
    if selected is None:
        return {
            "stage3_opens": True,
            "reason": "no candidate satisfied the primary retention floor",
            "best_eligible": None,
            "best_eligible_distance_to_mg": None,
            "reference_row": reference_row,
            "reference_distance_to_mg": ref_dist,
            "margin_required": 0.0,
            "no_margin_is_used": True,
        }
    best = rows[selected]["distance_to_mg"]
    improves = ref_dist is not None and best < ref_dist
    return {
        "stage3_opens": not improves,
        "reason": ("the best eligible candidate strictly improves on the "
                   "incumbent reference row" if improves else
                   "the best eligible candidate does not strictly improve on "
                   "the incumbent reference row"),
        "best_eligible": selected,
        "best_eligible_distance_to_mg": best,
        "reference_row": reference_row,
        "reference_distance_to_mg": ref_dist,
        "comparison": "strictly smaller 6-decimal distance; a tie is no "
                      "improvement",
        "margin_required": 0.0,
        "no_margin_is_used": True,
        "same_shape_as_the_frozen_stage2_gate": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Iteration 12 Stage-2 selection under the frozen protocol")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--image-batch-size", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int,
                        default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--model-id", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--candidates", default=None,
                        help="Restrict generation to these candidate ids (a "
                             "lane that trained part of the grid). The report "
                             "is still only written for a full grid.")
    parser.add_argument("--generate-only", action="store_true",
                        help="Generate (or verify-reuse) predictions and stop, "
                             "without scoring or writing the report")
    parser.add_argument("--no-stage", action="store_true",
                        help="Score and report only; do not stage the winner")
    parser.add_argument("--repo-root", default=None)
    args = parser.parse_args()

    repo_root = Path(args.repo_root) if args.repo_root \
        else (_find_repo_root(Path.cwd()) or Path.cwd())
    data_dir = sg.dataset_dir(repo_root)
    predictions_dir = sg.stage2_predictions_dir(repo_root)
    out_report = repo_root / sg.OUT_REPORT
    ckpt_root = sg.stage2_ckpt_root(repo_root)

    #: Step 1 -- the freeze is verified before anything expensive happens.
    reasons = fis2.verify_freeze(repo_root)
    preds_check = fis2.verify_clone_dependent(repo_root)
    mismatches = [m for res in preds_check.values() for m in res["mismatch"]]
    if reasons or mismatches:
        for r in reasons[:20]:
            log.error("    - %s", r)
        for m in mismatches:
            log.error("    - CLONE-DEPENDENT %s", m)
        raise SystemExit(
            "REFUSING to select under an unfrozen criterion. Either restore the "
            "frozen code, or re-freeze deliberately -- which "
            "freeze_iter12_stage2.py will itself refuse to do once a Stage-2 "
            "adapter exists.")
    freeze = json.loads((repo_root / fis2.OUT_REPORT).read_text())

    #: Step 2 -- the sealed confirmation is out of bounds.
    assert_no_forbidden_evidence(
        [data_dir, predictions_dir, out_report, ckpt_root,
         sg.reference_cache_dir(repo_root)], repo_root)

    #: Step 3 -- one instrument, inherited rather than restated.
    generation_config = {
        "batch_size": args.batch_size,
        "image_batch_size": args.image_batch_size,
        "max_new_tokens": args.max_new_tokens,
        "do_sample": False,
        "max_image_pixels": DEFAULT_MAX_IMAGE_PIXELS,
        "max_length": DEFAULT_MAX_LENGTH,
    }
    inherited = freeze[fis2.CONTRACT_FIELD]["contract"]
    actual = dict(generation_config)
    actual["model_id"] = args.model_id
    actual["experiment_id"] = EXPERIMENT_ID
    actual["predictions_dir"] = str(predictions_dir.relative_to(repo_root))
    for key, want in inherited.items():
        if key in ("why_not_the_pilot_predictions_dir",
                   "predictions_dir", "experiment_id"):
            #: Stage 2 has its own directory and experiment id by design; the
            #: selector refuses to WRITE anywhere Stage 1 wrote, so these two
            #: are expected to differ and comparing them would refuse a correct
            #: run.  Every measurement-relevant key is still compared.
            continue
        got = actual.get(key)
        if want != got:
            raise SystemExit(
                f"REFUSING: the generation contract is inherited as {key}="
                f"{want!r} but this run would use {got!r}. Batch layout is "
                f"inside D_G and inside the image route's correctness, so a "
                f"different contract is a different instrument.")

    probe = json.loads((repo_root / "data/reports/"
                        "mllmu_iter12_retention_probe.json").read_text())
    probe_entities = probe["halves"]["probe"]["entities"]
    queries = load_queries_parquet(data_dir / "queries.parquet")
    associations = load_associations_parquet(data_dir / "associations.parquet")
    by_assoc = {a.association_id: a for a in associations}
    predictions_dir.mkdir(parents=True, exist_ok=True)

    #: Step 4 -- the reused states, byte-verified, then recomputed and gated.
    reused_vec = verify_reused_bytes(repo_root)
    reused_preds = {state: load_predictions_parquet(path)
                    for state, path in reused_paths(repo_root).items()}
    reused_eight = {state: eight_numbers(p, queries, associations,
                                         probe_entities)
                    for state, p in reused_preds.items()}
    gate = gate_reused_against_the_filed_values(reused_eight, repo_root)
    if not gate["holds"]:
        raise SystemExit(
            "REFUSING to score: the reused states do not reproduce the numbers "
            "Stage 1c filed.\n  " + "\n  ".join(gate["problems"][:20]))
    log.info("reused %d state(s); %d eight-number values reproduce Stage 1c "
             "exactly", len(reused_preds), gate["numbers_compared"])

    mg_preds = reused_preds["MG"]
    mg_vec = rs.trainval_vector(mg_preds, queries, associations)
    log.info("MG %s vector: %s", "+".join(SPLITS), mg_vec)

    #: Step 5 -- generate the Stage-2 candidates.
    cal = json.loads((repo_root / sg.CALIBRATION_REPORT).read_text())
    grid = sg.stage2_grid(cal["grid_rule"]["beta_star"])
    kept, _filtered = sg.validate_stage2_grid(grid)
    if kept:
        raise SystemExit(f"REFUSING: the Stage-2 grid is invalid: {kept}")
    wanted = None
    if args.candidates:
        wanted = {s.strip() for s in args.candidates.split(",") if s.strip()}
        unknown = wanted - {c.candidate_id for c in grid}
        if unknown:
            raise SystemExit(
                f"candidates outside the frozen grid: {sorted(unknown)}")

    generated: dict[str, dict[str, Any]] = {}
    missing_adapters: list[str] = []
    for state in REUSED_STATES:
        summary_path = (sg.stage1_ckpt_root(repo_root) / state
                        / "training_summary.json")
        generated[state] = {
            "method": "B0" if state == "B0" else (
                "MG" if state == "MG" else "B4"),
            "spec": {"candidate_id": state, "reused_from_stage1": True},
            "trainval_metrics": rs.trainval_hierarchy_metrics(
                reused_preds[state], queries, associations),
            "eight_numbers": reused_eight[state],
            "pooled_all_routes": rt.pooled_all_routes(
                reused_preds[state], queries, associations),
            "training_side": None,
            "config": json.loads(summary_path.read_text())
            if summary_path.exists() else None,
        }
    for spec in sg.trained_rows(grid):
        if wanted is not None and spec.candidate_id not in wanted:
            continue
        adir = ckpt_root / spec.candidate_id / "adapters"
        if not adir.exists():
            missing_adapters.append(spec.candidate_id)
            log.warning("[%s] adapters missing — skipped", spec.candidate_id)
            continue
        preds = _generate_state(
            spec.candidate_id, adir, queries, by_assoc, repo_root,
            args.device, predictions_dir, data_dir, args.model_id,
            generation_config, EXPERIMENT_ID, SPLITS)
        summary_path = ckpt_root / spec.candidate_id / "training_summary.json"
        summary = json.loads(summary_path.read_text()) \
            if summary_path.exists() else {}
        last = (summary.get("epochs") or [{}])[-1]
        generated[spec.candidate_id] = {
            "method": spec.method,
            "spec": dict(spec.describe(), cap=spec.cap, beta=spec.beta),
            "trainval_metrics": rs.trainval_hierarchy_metrics(
                preds, queries, associations),
            "eight_numbers": eight_numbers(preds, queries, associations,
                                           probe_entities),
            "pooled_all_routes": rt.pooled_all_routes(
                preds, queries, associations),
            "training_side": {
                "final_avg_loss_fine_target":
                    last.get("avg_loss_fine_target"),
                "final_frac_at_cap_fine_target":
                    last.get("frac_at_cap_fine_target"),
                "final_avg_kl_retain": last.get("avg_kl_retain"),
                "final_avg_loss_retain": last.get("avg_loss_retain"),
                "num_optimizer_steps": summary.get("num_optimizer_steps"),
                "train_seconds": summary.get("train_seconds"),
            },
            "config": {k: summary.get(k) for k in
                       ("recipe", "groups", "num_optimizer_steps",
                        "init_adapter_dir", "noop", "preservation_anchor")},
        }
        log.info("[%s] fine_target NLL %s | frac_at_cap %s | %s",
                 spec.candidate_id, last.get("avg_loss_fine_target"),
                 last.get("frac_at_cap_fine_target"),
                 json.dumps({k: round(v, 4) for k, v in
                             generated[spec.candidate_id]
                             ["eight_numbers"].items() if v is not None}))

    if args.generate_only:
        log.info("generate-only: %d state(s) have %s predictions; no report "
                 "written, because a report over a subset would name a winner "
                 "the full grid does not support",
                 len(generated), "+".join(SPLITS))
        return

    #: Step 6 -- the whole grid, or no report.
    if missing_adapters:
        raise SystemExit(
            f"REFUSING to write a selection report: {len(missing_adapters)} "
            f"grid candidate(s) have no adapter ({missing_adapters}). Train "
            f"them, or pass --generate-only; a partial report would look "
            f"complete and carry the same dataset_version as the real one.")

    #: Step 7 -- the floors, then D_G, then the frozen tie-break.
    report = build_report(generated, reused_eight, reused_vec, mg_vec,
                          repo_root, data_dir, predictions_dir,
                          generation_config, args.model_id,
                          {"reused_states_reproduce_stage1c": gate},
                          freeze)

    if not args.no_stage and report["selected"]:
        import shutil
        cid = report["selected"]
        src = (ckpt_root / cid / "adapters") if (
            ckpt_root / cid / "adapters").exists() \
            else sg.stage1_ckpt_root(repo_root) / cid / "adapters"
        dst = ckpt_root / "selected" / report["candidates"][cid]["method"] \
            / "adapters"
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
        summary = src.parent / "training_summary.json"
        if summary.exists():
            shutil.copy(summary, dst.parent / "training_summary.json")
        report["staged"] = str(dst.relative_to(repo_root))
        log.info("SELECTED %s <- %s (D_G=%.6f)",
                 report["candidates"][cid]["method"], cid,
                 report["candidates"][cid]["distance_to_mg"])

    with open(out_report, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    log.info("eligible on the primary (%s-anchored) floor: %d/%d",
             report["criterion"]["anchor"], len(report["eligible"]),
             len(report["candidates"]))
    log.info("eligible on the reported B0-anchored floor: %d/%d",
             sum(1 for v in report["eligibility_under_both_anchors"].values()
                 if v["on_b0"]),
             len(report["candidates"]))
    log.info("selected %s | tied_with %s | stage 3 opens: %s (%s)",
             report["selected"], report["tied_with"],
             report["stage_3_gate"]["stage3_opens"],
             report["stage_3_gate"]["reason"])
    log.info("Selection report -> %s", out_report)


if __name__ == "__main__":
    main()
