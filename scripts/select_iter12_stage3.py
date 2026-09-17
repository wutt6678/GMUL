"""Iteration 12 Stage 3: generate, apply the frozen MG floor, order by D_G.

    python scripts/select_iter12_stage3.py --device cuda:0
    python scripts/select_iter12_stage3.py --device cuda:0 --generate-only
    python scripts/select_iter12_stage3.py --candidates B7_imgw3.4105_beta13.642_lam0.5_lr2e-05_ep5

This executor decides nothing of its own.  The eight-number MG-anchored floor,
its epsilon, D_G, the tie-break, the B6 zero-weight control and the four B7
weights are read from the Stage-3 freeze, which is verified before generation.
B0 and MG are reused from Stage 1's bound parquets; B6 is reused from Stage
2's bound parquet; only the four B7 adapters are newly generated.

The selector writes the MG exclusion correctly at source.  Stage 2's frozen
selector ranked the anchor against itself and needed a runtime repair; Stage
3 keeps MG in ``anchor_states`` and never puts it in ``candidates`` or
``ranking_of_eligible``.

Needs a GPU for generation.  Reads no confirmation prediction and writes no
sealed path.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import freeze_iter12_stage3 as fis3
from freeze_iter12_route_stratification import prediction_filename
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
from granunlearn.training import stage2_grid as s2g
from granunlearn.training import stage3_grid as s3g
from granunlearn.training.preservation_anchor import sha256_file

log = setup_logger("select_iter12_stage3")

SPLITS = ("train", "val")
EXPERIMENT_ID = "mllmu_iter12_stage3"
FLOOR_KEYS: tuple[str, ...] = rt.FLOOR_NUMBER_KEYS


def assert_no_forbidden_evidence(paths: list[Path], repo_root: Path) -> None:
    """Refuse to touch the sealed confirmation, on resolved paths."""
    forbidden = [((repo_root / rel).resolve(), rel)
                 for rel in rs.FORBIDDEN_EVIDENCE]
    for path in paths:
        resolved = Path(path).resolve()
        for bad, rel in forbidden:
            if resolved == bad or bad in resolved.parents:
                raise SystemExit(
                    f"REFUSING: {path} resolves inside {rel}. The 11C "
                    f"confirmation is sealed evidence; Stage 3 is selected "
                    f"from exploratory train+val evidence only.")


def control_row_id(repo_root: Path) -> str:
    calibration = json.loads((repo_root / s2g.CALIBRATION_REPORT).read_text())
    return s3g.control_row(
        s3g.stage3_grid(calibration["grid_rule"]["beta_star"])).candidate_id


def reused_paths(repo_root: Path) -> dict[str, Path]:
    b6 = control_row_id(repo_root)
    return {
        "B0": s3g.stage1_predictions_dir(repo_root) / prediction_filename("B0"),
        "MG": s3g.stage1_predictions_dir(repo_root) / prediction_filename("MG"),
        b6: s3g.stage2_predictions_dir(repo_root) / prediction_filename(b6),
    }


def verify_reused_bytes(repo_root: Path) -> dict[str, Any]:
    freeze = json.loads((repo_root / fis3.OUT_REPORT).read_text())
    bound = freeze[fis3.PREDICTIONS_FIELD]["sha256"]
    out: dict[str, Any] = {}
    for state, path in reused_paths(repo_root).items():
        key = ("stage1.B0" if state == "B0" else
               "stage1.MG" if state == "MG" else "stage2.B6")
        if not path.exists():
            raise SystemExit(
                f"REFUSING: {path} is missing. Stage 3 reuses the anchor and "
                f"B6 bytes rather than regenerating them.")
        got = sha256_file(path)
        want = bound.get(key)
        if want is None:
            raise SystemExit(
                f"REFUSING: the Stage-3 freeze binds no sha256 for {key}.")
        if got != want:
            raise SystemExit(
                f"REFUSING: {path} hashes to {got[:16]} but the freeze "
                f"records {want[:16]}.")
        out[state] = {"path": str(path.relative_to(repo_root)),
                      "sha256": got}
    return out


def eight_numbers(preds, queries, associations, probe_entities) -> dict:
    return rt.floor_vector(rt.probe_retention_by_route(
        preds, queries, associations, probe_entities))


def summary_vector(preds, queries, associations) -> dict:
    return rs.summary_vector(rs.trainval_hierarchy_metrics(
        preds, queries, associations))


def gate_reused_states(recomputed: dict[str, dict[str, Any]],
                       repo_root: Path) -> dict[str, Any]:
    """The reused-state and zero-weight gates, decided before any report."""
    b6 = control_row_id(repo_root)
    strat = json.loads((repo_root / s2g.ROUTE_STRATIFIED_REPORT).read_text())
    stage2 = json.loads((repo_root / s2g.OUT_REPORT).read_text())
    filed = {
        "B0": strat["anchor"]["values"],
        "MG": strat["stage1"]["reference_states_not_floored"]["MG"]["values"],
        b6: stage2["candidates"][b6]["values"],
    }
    problems: list[str] = []
    compared = 0
    for state, values in filed.items():
        got = recomputed.get(state, {}).get("eight_numbers")
        if got is None:
            problems.append(f"{state}: not recomputed")
            continue
        for key in FLOOR_KEYS:
            compared += 1
            if got.get(key) != values[key]:
                problems.append(
                    f"{state}/{key}: filed {values[key]} but the reused "
                    f"parquet recomputes to {got.get(key)}")

    mg_vec = recomputed["MG"]["vector"]
    b6_vec = recomputed[b6]["vector"]
    dg, used = rs.distance_to_mg(b6_vec, mg_vec)
    filed_dg = stage2["candidates"][b6]["distance_to_mg"]
    dg_gap = None if dg is None else abs(dg - filed_dg)
    zero_control_report = json.loads(
        (repo_root / s3g.CONTROL_REPORT).read_text())
    zero_control = {
        "control": zero_control_report["control"],
        "filed_report": s3g.CONTROL_REPORT,
        "filed_report_holds": zero_control_report["holds"],
        "structural": zero_control_report["structural"],
        "b6_d_g": {
            "filed": filed_dg,
            "recomputed": dg,
            "abs_difference": dg_gap,
            "used_components": used,
            "within_tolerance": dg_gap is not None
                and dg_gap <= s3g.CONTROL_DG_TOLERANCE,
        },
        "floor_tolerance": s3g.CONTROL_FLOOR_TOLERANCE,
        "d_g_tolerance": s3g.CONTROL_DG_TOLERANCE,
    }
    zero_control["holds"] = (
        zero_control_report["holds"] and not problems
        and zero_control["b6_d_g"]["within_tolerance"])
    return {
        "reused_states_reproduce_their_filed_values": {
            "states_gated": sorted(filed),
            "numbers_compared": compared,
            "problems": problems,
            "holds": not problems,
        },
        "zero_image_anchor_weight_reproduces_b6": zero_control,
    }


def build_report(states: dict[str, dict[str, Any]],
                 reused_vec: dict[str, Any],
                 mg_vec: dict[str, float | None],
                 repo_root: Path,
                 data_dir: Path,
                 predictions_dir: Path,
                 generation_config: dict[str, Any],
                 model_id: str,
                 gates: dict[str, Any],
                 freeze: dict[str, Any],
                 grid: list[s3g.Stage3CandidateSpec]) -> dict[str, Any]:
    """Apply the primary floor, report B0 beside it, then order by D_G."""
    b6 = control_row_id(repo_root)
    b0_eight = states["B0"]["eight_numbers"]
    mg_eight = freeze["what_is_frozen"]["anchor_values"]
    eps = rs.FLOOR_EPSILON

    rows: dict[str, Any] = {}
    for cid, info in states.items():
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
            "reused_from": info.get("reused_from"),
            "training_side": info.get("training_side"),
            "config": info.get("config"),
        }

    #: MG is not in ``rows`` at all.  This is the defect Stage 2's repair had
    #: to work around; Stage 3 does not introduce it.
    eligible = {cid: r for cid, r in rows.items() if r["eligible"]}
    ranked = sorted(eligible, key=lambda c: rs.rank_key(
        rows[c]["distance_to_mg"], c))
    selected = ranked[0] if ranked else None
    tied = sorted(cid for cid in eligible
                  if selected is not None and cid != selected
                  and rows[cid]["distance_to_mg"] ==
                  rows[selected]["distance_to_mg"])
    b7 = [c for c in grid if c.method == s3g.METHOD_ROUTE_ANCHOR]
    return {
        "iteration": "12",
        "stage": 3,
        "criterion": {
            "primary": freeze["what_is_frozen"]["primary_rule"],
            "anchor": freeze["what_is_frozen"]["primary_anchor"],
            "anchor_values": mg_eight,
            "epsilon": eps,
            "margin": 0.0,
            "no_margin_is_used": True,
            "reported_beside_it": (
                "the frozen B0-anchored eight-number floor, computed for "
                "every state and decision-bearing nowhere"),
            "reported_baseline_values": b0_eight,
            "denominators": freeze["what_is_frozen"]["denominators"],
            "among_eligible": freeze["what_is_frozen"]["among_eligible"],
            "tie_break": "(distance_to_mg, candidate_id) ascending",
        },
        "consistency_gates": gates,
        "anchor_states": {
            "B0": {
                "values": b0_eight,
                "reused_from": reused_vec["B0"]["path"],
                "sha256": reused_vec["B0"]["sha256"],
                "primary_floor_result": rt.floor_check_stratified(
                    b0_eight, mg_eight, eps),
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
        "candidates": rows,
        "eligible": sorted(eligible),
        "disqualified_by_the_primary_floor": {
            cid: {k: v["passes"] for k, v in
                  rows[cid]["primary_floor"]["checks"].items()}
            for cid in rows if not rows[cid]["eligible"]
        },
        "eligibility_under_both_anchors": {
            cid: {"on_mg": rows[cid]["eligible"],
                  "on_b0": rows[cid]["eligible_on_the_reported_b0_floor"]}
            for cid in rows
        },
        "ranking_of_eligible": ranked,
        "selected": selected,
        "tied_with": tied,
        "a_tie_means": (
            "the criterion did not discriminate and the tie-break -- not the "
            "science -- chose the winner"),
        "mechanism": {
            "parent": b6,
            "additional_image_weight_grid": [
                c.image_anchor_weight for c in b7],
            "effective_anchor_weight_grid": [
                c.effective_anchor_weight for c in b7],
            "image_conditioned_examples":
                freeze["image_anchor_examples"]["image_conditioned_examples"],
            "fit_half_examples":
                freeze["image_anchor_examples"]["fit_half_examples"],
            "the_important_disclosure":
                freeze["image_anchor_examples"]["the_important_disclosure"],
        },
        "tag": "iter12",
        "protocol_freeze": {
            "path": fis3.OUT_REPORT,
            "verified_before_generation": True,
            "mismatches_found": [],
        },
        "basis": {"path": s3g.BASIS_REPORT,
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
            "hypothesis and controls no error rate. Stage 3 trains one seed "
            "per B7 row, so a row that clears the floor by less than the "
            "between-seed spread Stage 1b measured is a candidate for "
            "replication rather than a result."),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Iteration 12 Stage-3 selection under the frozen protocol")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--image-batch-size", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int,
                        default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--model-id", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--candidates", default=None,
                        help="Restrict generation to these B7 ids. The report "
                             "is still only written for a full grid.")
    parser.add_argument("--generate-only", action="store_true")
    parser.add_argument("--repo-root", default=None)
    args = parser.parse_args()

    repo_root = Path(args.repo_root) if args.repo_root \
        else (_find_repo_root(Path.cwd()) or Path.cwd())
    data_dir = s3g.dataset_dir(repo_root)
    predictions_dir = s3g.stage3_predictions_dir(repo_root)
    out_report = repo_root / s3g.OUT_REPORT
    ckpt_root = s3g.stage3_ckpt_root(repo_root)

    #: Step 1 -- the freeze is verified before anything expensive happens.
    reasons = fis3.verify_freeze(repo_root)
    clone = fis3.verify_clone_dependent(repo_root) \
        if (repo_root / fis3.OUT_REPORT).exists() else {}
    mismatches = [m for res in clone.values() for m in res["mismatch"]]
    if reasons or mismatches:
        for r in reasons[:20]:
            log.error("    - %s", r)
        for m in mismatches:
            log.error("    - CLONE-DEPENDENT %s", m)
        raise SystemExit(
            "REFUSING to select under an unfrozen Stage-3 criterion.")
    freeze = json.loads((repo_root / fis3.OUT_REPORT).read_text())

    #: Step 2 -- the sealed confirmation is out of bounds.
    assert_no_forbidden_evidence(
        [data_dir, predictions_dir, out_report, ckpt_root,
         s3g.reference_cache_dir(repo_root)], repo_root)

    #: Step 3 -- one instrument, inherited rather than restated.
    generation_config = {
        "batch_size": args.batch_size,
        "image_batch_size": args.image_batch_size,
        "max_new_tokens": args.max_new_tokens,
        "do_sample": False,
        "max_image_pixels": DEFAULT_MAX_IMAGE_PIXELS,
        "max_length": DEFAULT_MAX_LENGTH,
    }
    inherited = freeze[fis3.CONTRACT_FIELD]["contract"]
    actual = dict(generation_config)
    actual["model_id"] = args.model_id
    actual["experiment_id"] = EXPERIMENT_ID
    actual["predictions_dir"] = str(predictions_dir.relative_to(repo_root))
    for key, want in inherited.items():
        if key in ("why_not_the_pilot_predictions_dir",
                   "predictions_dir", "experiment_id"):
            continue
        got = actual.get(key)
        if want != got:
            raise SystemExit(
                f"REFUSING: the generation contract is inherited as {key}="
                f"{want!r} but this run would use {got!r}.")

    calibration = json.loads((repo_root / s2g.CALIBRATION_REPORT).read_text())
    grid = s3g.stage3_grid(calibration["grid_rule"]["beta_star"])
    kept, _filtered = s3g.validate_stage3_grid(
        grid, calibration["grid_rule"]["beta_star"])
    if kept:
        raise SystemExit(f"REFUSING: the Stage-3 grid is invalid: {kept}")
    b7 = s3g.b7_rows(grid)
    wanted = None
    if args.candidates:
        wanted = {c.strip() for c in args.candidates.split(",") if c.strip()}
        unknown = wanted - {c.candidate_id for c in b7}
        if unknown:
            raise SystemExit(f"candidates outside the frozen B7 grid: "
                             f"{sorted(unknown)}")

    probe = json.loads((repo_root / "data/reports/"
                                  "mllmu_iter12_retention_probe.json").read_text())
    probe_entities = probe["halves"]["probe"]["entities"]
    queries = load_queries_parquet(data_dir / "queries.parquet")
    associations = load_associations_parquet(data_dir / "associations.parquet")
    by_assoc = {a.association_id: a for a in associations}

    #: Step 4 -- reused states and the zero-weight control, before generation.
    reused_vec = verify_reused_bytes(repo_root)
    reused_preds = {state: load_predictions_parquet(path)
                    for state, path in reused_paths(repo_root).items()}
    recomputed = {state: {
        "eight_numbers": eight_numbers(preds, queries, associations,
                                       probe_entities),
        "vector": summary_vector(preds, queries, associations),
        "trainval_metrics": rs.trainval_hierarchy_metrics(
            preds, queries, associations),
    } for state, preds in reused_preds.items()}
    gates = gate_reused_states(recomputed, repo_root)
    if not gates["reused_states_reproduce_their_filed_values"]["holds"] or \
            not gates["zero_image_anchor_weight_reproduces_b6"]["holds"]:
        raise SystemExit(
            "REFUSING to generate B7 predictions: the reused anchors or the "
            "zero-weight B6 control failed, so a B7 row would not be "
            "interpretable as B6 plus image weight.")

    stage2 = json.loads((repo_root / s2g.OUT_REPORT).read_text())
    b6 = control_row_id(repo_root)
    states: dict[str, dict[str, Any]] = {
        "B0": {
            "method": "B0",
            "spec": {"candidate_id": "B0", "reused_from_stage1": True},
            "trainval_metrics": recomputed["B0"]["trainval_metrics"],
            "eight_numbers": recomputed["B0"]["eight_numbers"],
            "reused_from": "stage1",
        },
        b6: {
            "method": s2g.METHOD_ANCHOR,
            "spec": s3g.control_row(grid).describe(),
            "trainval_metrics": recomputed[b6]["trainval_metrics"],
            "eight_numbers": recomputed[b6]["eight_numbers"],
            "reused_from": "stage2",
            "training_side": stage2["candidates"][b6].get("training_side"),
            "config": stage2["candidates"][b6].get("config"),
        },
    }
    mg_vec = recomputed["MG"]["vector"]

    predictions_dir.mkdir(parents=True, exist_ok=True)
    missing_adapters: list[str] = []
    for spec in b7:
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
        states[spec.candidate_id] = {
            "method": spec.method,
            "spec": spec.describe(),
            "trainval_metrics": rs.trainval_hierarchy_metrics(
                preds, queries, associations),
            "eight_numbers": eight_numbers(preds, queries, associations,
                                           probe_entities),
            "reused_from": None,
            "training_side": {
                "final_avg_loss_fine_target":
                    (summary.get("epochs") or [{}])[-1].get(
                        "avg_loss_fine_target"),
                "final_avg_kl_retain":
                    (summary.get("epochs") or [{}])[-1].get("avg_kl_retain"),
            },
            "config": {k: summary.get(k) for k in
                       ("recipe", "groups", "num_optimizer_steps",
                        "init_adapter_dir", "preservation_anchor")},
        }
        log.info("[%s] %s", spec.candidate_id,
                 json.dumps(states[spec.candidate_id]["eight_numbers"]))

    if args.generate_only:
        log.info("generate-only: %d B7 row(s) have %s predictions in %s",
                 len([s for s in states if s.startswith("B7_")]),
                 "+".join(SPLITS), predictions_dir.name)
        return

    missing = [c.candidate_id for c in b7 if c.candidate_id not in states]
    if missing:
        raise SystemExit(
            f"REFUSING to write a Stage-3 report over a subset. Missing "
            f"predictions for {missing}; adapters missing for "
            f"{missing_adapters}. A partial report would name a winner the "
            f"full grid does not support.")
    report = build_report(states, reused_vec, mg_vec, repo_root, data_dir,
                          predictions_dir, generation_config, args.model_id,
                          gates, freeze, grid)
    if out_report.exists():
        raise SystemExit(
            f"REFUSING to overwrite {out_report}. A Stage-3 selection is "
            f"filed once; preserve or supersede it deliberately.")
    out_report.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    log.info("Stage-3 selection -> %s (selected=%s)", out_report,
             report["selected"])


if __name__ == "__main__":
    main()
