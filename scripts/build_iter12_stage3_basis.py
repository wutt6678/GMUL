"""The Stage-3 measurement basis: written before any Stage-3 adapter exists.

    python scripts/build_iter12_stage3_basis.py
    python scripts/build_iter12_stage3_basis.py --check-only

Stage 3 starts from a filed result, not from a preference for the anchor
mechanism.  Stage 2's B6 row clears every text-route retention number, fails
only three image-route numbers, misses by fourteen rows in total, and preserves
the fine-target objective better than the bounded-ascent rows.  This document
records the design that follows from that result -- one additional
image-conditioned anchor weight, swept at four frozen values beside the B6
zero-weight control -- and records the dataset facts that make the design
honest.

WHAT THIS DOCUMENT READS
------------------------
Committed reports, the fit-half retain group and the MF cache's committed
sidecar.  It reads NO prediction parquet and NO confirmation evidence, so
``--check-only`` can re-derive it byte for byte in a clean clone.  B6's
prediction bytes are bound separately by the freeze and by the zero-weight
control; they are not needed to state the protocol.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from freeze_iter12_selection_protocol import _flat, _sha256

from granunlearn.config import _find_repo_root
from granunlearn.evaluation import retention_selection as rs
from granunlearn.evaluation import route_stratified_retention as rt
from granunlearn.logging_utils import setup_logger
from granunlearn.training import stage2_grid as s2g
from granunlearn.training import stage3_grid as s3g

log = setup_logger("build_iter12_stage3_basis")

OUT_REPORT = s3g.BASIS_REPORT
STAGE2_SELECTION = s2g.OUT_REPORT
PROBE_REPORT = "data/reports/mllmu_iter12_retention_probe.json"
ROUTE_PROBE_REPORT = "data/reports/mllmu_iter12_route_probe.json"
STAGE2_FREEZE = s2g.FREEZE_REPORT

IMAGE_FLOOR_KEYS = tuple(k for k in rt.FLOOR_NUMBER_KEYS if k.startswith("image."))
TEXT_FLOOR_KEYS = tuple(k for k in rt.FLOOR_NUMBER_KEYS if k.startswith("text."))


def _round(x: float, nd: int = 4) -> float:
    return round(float(x), nd)


def b6_outcome(repo_root: Path) -> dict[str, Any]:
    """The filed Stage-2 result Stage 3 is allowed to start from."""
    selection = json.loads((repo_root / STAGE2_SELECTION).read_text())
    candidates = selection["candidates"]
    b6_ids = [cid for cid, row in candidates.items()
              if row["method"] == s2g.METHOD_ANCHOR
              and cid.startswith("B6_beta")]
    if len(b6_ids) != 1:
        raise SystemExit(
            f"REFUSING: expected exactly one B6 row in {STAGE2_SELECTION}, "
            f"got {b6_ids}")
    cid = b6_ids[0]
    row = candidates[cid]
    checks = row["primary_floor"]["checks"]
    failed = [k for k in rt.FLOOR_NUMBER_KEYS if not checks[k]["passes"]]
    denominators = selection["criterion"]["denominators"]

    same_key = "image.retain_same_entity_image.row_micro"
    other_key = "image.retain_other_entity_image.row_micro"
    same_rows = round(abs(row["difference_vs_mg_anchor"][same_key]) *
                      denominators[same_key])
    other_rows = round(abs(row["difference_vs_mg_anchor"][other_key]) *
                       denominators[other_key])
    bounded = [v["final_fine_target_nll"] for v in
               selection["cap_sweep"]["rows"].values()]
    return {
        "candidate_id": cid,
        "method": row["method"],
        "selected_stage2": selection["selected"],
        "stage3_gate": selection["stage_3_gate"],
        "distance_to_mg": row["distance_to_mg"],
        "fine_target_nll": row["training_side"]["final_avg_loss_fine_target"],
        "bounded_ascent_fine_target_nll_range": [min(bounded), max(bounded)],
        "preserves_the_target_transformation_better_than_bounded_ascent":
            row["training_side"]["final_avg_loss_fine_target"] < min(bounded),
        "values": row["values"],
        "difference_vs_mg_anchor": row["difference_vs_mg_anchor"],
        "text_route": {
            "keys": list(TEXT_FLOOR_KEYS),
            "passes": {k: checks[k]["passes"] for k in TEXT_FLOOR_KEYS},
            "all_four_pass": all(checks[k]["passes"] for k in TEXT_FLOOR_KEYS),
        },
        "image_route": {
            "keys": list(IMAGE_FLOOR_KEYS),
            "passes": {k: checks[k]["passes"] for k in IMAGE_FLOOR_KEYS},
            "failed": failed,
            "exactly_three_fail": len(failed) == 3,
        },
        "the_fourteen_query_shortfall": {
            "same_image_rows": same_rows,
            "same_image_denominator": denominators[same_key],
            "other_image_rows": other_rows,
            "other_image_denominator": denominators[other_key],
            "expressed_over_trainval_both_route_denominators": {
                "same": f"{same_rows * 2}/816",
                "other": f"{other_rows * 2}/152",
                "total": same_rows * 2 + other_rows * 2,
            },
            "why_the_entity_macro_failure_adds_no_rows": (
                "entity_macro and row_micro are two estimands over the same "
                "408 image retain-same rows. Counting both as new queries "
                "would double-count the same outcomes; the distinct row "
                "shortfall is five of 408 retain-same plus two of 76 "
                "retain-other, written as ten of 816 and four of 152 when the "
                "two routes' train+val denominators are used."),
        },
        "why_this_is_the_parent": (
            "B6 is the only row that holds all four text floors while keeping "
            "the fine-target NLL below every bounded-ascent row. Its remaining "
            "failure is confined to the image route, so Stage 3 varies the "
            "image-conditioned anchor strength and leaves the rest of B6 "
            "unchanged."),
    }


def image_anchor_examples(repo_root: Path) -> dict[str, Any]:
    """The image-conditioned fit-half stream, measured from the group file."""
    group = s3g.retain_group_path(repo_root)
    if not group.exists():
        raise SystemExit(f"REFUSING: {group} is missing")
    rows = [json.loads(line) for line in group.read_text().splitlines()
            if line.strip()]
    image_rows = [r for r in rows if s3g.image_conditioned(r)]
    probe = json.loads((repo_root / PROBE_REPORT).read_text())
    probe_assoc = set(probe["halves"]["probe"]["associations"])
    probe_entities = set(probe["halves"]["probe"]["entities"])

    cache_dir = s3g.reference_cache_dir(repo_root)
    sidecar_path = cache_dir / "sidecar.json"
    if not sidecar_path.exists():
        raise SystemExit(
            f"REFUSING: {sidecar_path} is missing. The image-conditioned MF "
            f"cache's provenance is part of the protocol, not an anecdote.")
    sidecar = json.loads(sidecar_path.read_text())
    cache_rows = sidecar.get("rows", [])
    cache_example_ids = {r["example_id"] for r in cache_rows}
    group_example_ids = {r["example_id"] for r in image_rows}
    cache_assoc = {r["association_id"] for r in cache_rows}
    return {
        "source_group": str(group.relative_to(repo_root)),
        "source_group_sha256": _sha256(group),
        "predicate": "modality == 'image_text' and image_path is non-empty",
        "fit_half_examples": len(rows),
        "image_conditioned_examples": len(image_rows),
        "not_image_conditioned": len(rows) - len(image_rows),
        "all_fit_half_examples_are_image_conditioned":
            len(image_rows) == len(rows),
        "entities": len({r["entity_id"] for r in image_rows}),
        "associations": len({r["association_id"] for r in image_rows}),
        "probe_half_overlap": {
            "associations": len(probe_assoc & cache_assoc),
            "entities": len(probe_entities & {r["entity_id"] for r in image_rows}),
            "must_be_zero": True,
        },
        "the_important_disclosure": (
            "Every fit-half retention example is image-conditioned, so the "
            "route-aware term is additional weight on the existing 183-example "
            "stream, not a restriction to a smaller subset. Any prose that "
            "calls it a subset without this sentence would overstate the "
            "mechanism."),
        "mf_cache": {
            "directory": str(cache_dir.relative_to(repo_root)),
            "sidecar": str(sidecar_path.relative_to(repo_root)),
            "sidecar_sha256": _sha256(sidecar_path),
            "tensor_sha256": sidecar.get("tensor_sha256"),
            "mf_adapter_sha256": sidecar.get("mf_adapter_sha256"),
            "base_model_revision": sidecar.get("base_model_revision"),
            "cache_version": sidecar.get("cache_version"),
            "num_examples": sidecar.get("num_examples"),
            "num_positions": sidecar.get("num_positions"),
            "cache_rows_match_the_image_conditioned_group":
                cache_example_ids == group_example_ids,
            "cache_is_the_image_conditioned_mf_cache":
                cache_example_ids == group_example_ids
                and len(image_rows) == len(rows),
            "probe_half_excluded": sidecar.get("probe_half_excluded"),
            "target_groups_excluded": sidecar.get("target_groups_excluded"),
            "kl_direction": sidecar.get("kl_direction"),
            "encoding": {
                "max_length": sidecar.get("max_length"),
                "max_image_pixels": sidecar.get("max_image_pixels"),
                "vocab_size": sidecar.get("vocab_size"),
            },
            "no_new_cache_is_needed": True,
            "why": (
                "The Stage-2 cache already covers exactly these 183 "
                "image-conditioned examples at every supervised position, "
                "with MF-adapter and encoding provenance. Rebuilding the same "
                "1.48 GB tensor under a Stage-3 name would add bytes, not "
                "information."),
        },
    }


def grid_summary(repo_root: Path) -> dict[str, Any]:
    calibration = json.loads((repo_root / s2g.CALIBRATION_REPORT).read_text())
    beta_star = calibration["grid_rule"]["beta_star"]
    beta = round(beta_star, 4)
    grid = s3g.stage3_grid(beta_star)
    kept, filtered = s3g.validate_stage3_grid(grid, beta_star)
    if kept:
        raise SystemExit(f"REFUSING: the Stage-3 grid is invalid: {kept}")
    return {
        "beta_star": beta_star,
        "beta_used_by_b6": beta,
        "additional_image_weight_multipliers":
            list(s3g.IMAGE_WEIGHT_MULTIPLIERS),
        "additional_image_weights": [
            round(beta * m, 4) for m in s3g.IMAGE_WEIGHT_MULTIPLIERS],
        "effective_anchor_weights": [
            round(beta + round(beta * m, 4), 4)
            for m in s3g.IMAGE_WEIGHT_MULTIPLIERS],
        "why_these_weights": (
            "B6 is the local optimum to improve on, so the sweep stays local: "
            "one eighth, one quarter, one half and one times its calibrated "
            "beta as ADDITIONAL image-conditioned weight. The zero row is B6 "
            "itself, not a fifth training run."),
        "num_rows": len(grid),
        "num_trained": len(s3g.trained_rows(grid)),
        "rows": [c.describe() for c in grid],
        "the_frozen_validator_was_run": {
            "residual_errors": kept,
            "filtered_messages": filtered,
            "filtered_gap_substrings": list(s3g.FROZEN_VALIDATOR_GAP),
            "filtered_because": (
                "candidate_grid.py is hash-bound and its METHODS/mode "
                "vocabulary predates B6/B7 and anchor. Only 'unknown method' "
                "and 'bad mode' are filtered; every structural check it can "
                "still make applies unchanged."),
        },
    }


def selection_rule(repo_root: Path) -> dict[str, Any]:
    """The Stage-2 criterion, inherited without amendment."""
    stage2 = json.loads((repo_root / STAGE2_FREEZE).read_text())
    frozen = stage2["what_is_frozen"]
    return {
        "inherited_from": STAGE2_FREEZE,
        "floor_numbers": frozen["floor_numbers"],
        "num_numbers": frozen["num_numbers"],
        "primary_anchor": frozen["primary_anchor"],
        "primary_rule": frozen["primary_rule"],
        "epsilon": frozen["epsilon"],
        "margin": frozen["margin"],
        "no_margin_is_used": frozen["no_margin_is_used"],
        "anchor_values": frozen["anchor_values"],
        "reported_baseline": frozen["reported_baseline"],
        "reported_baseline_values": frozen["reported_baseline_values"],
        "among_eligible": frozen["among_eligible"],
        "tie_break": "(distance_to_mg, candidate_id) ascending",
        "distance_rounding": "6 decimals, as distance_to_reference rounds",
        "mg_is_excluded_from_the_ranking": True,
        "why_mg_is_excluded": (
            "MG is the reference D_G measures distance to. Stage 2's frozen "
            "selector omitted that exclusion at the ranking bindings and the "
            "repair had to remove it at runtime; Stage 3 writes the exclusion "
            "into its selector instead of repeating the defect."),
        "unchanged_from_stage2": True,
    }


def build_report(repo_root: Path) -> dict[str, Any]:
    probe = json.loads((repo_root / PROBE_REPORT).read_text())
    route_probe = json.loads((repo_root / ROUTE_PROBE_REPORT).read_text())
    return {
        "iteration": "12",
        "stage": 3,
        "purpose": (
            "Develop a route-aware successor to B6 by increasing only the "
            "image-conditioned MF-anchor weight, while preserving B6's "
            "fine-target ascent, target-level objective, the eight MG-anchored "
            "floors and D_G."),
        "reads_no_prediction_parquet": True,
        "b6_result": b6_outcome(repo_root),
        "image_anchor_examples": image_anchor_examples(repo_root),
        "grid": grid_summary(repo_root),
        "selection_rule": selection_rule(repo_root),
        "partition": {
            "source": PROBE_REPORT,
            "rule": probe["partition_rule"],
            "fit_entities": probe["halves"]["fit"]["num_entities"],
            "probe_entities": probe["halves"]["probe"]["num_entities"],
            "probe_never_replayed":
                not probe["halves"]["probe"]["replayed_by_any_candidate"],
            "route_basis": ROUTE_PROBE_REPORT,
            "route_denominators": route_probe["measurement_basis"]["per_route"],
        },
        "seeds_and_layout": {
            "training_seed": 42,
            "python_hash_seed": "0",
            "why_the_seed_is_held": (
                "Every B7 row is B6 with one coefficient moved, so the recipe "
                "seed, LoRA shape, optimizer, learning rate, epochs, "
                "per-device batch, accumulation, max length and image-pixel "
                "budget are held at B6's values."),
            "training": {
                "learning_rate": s2g.STAGE2_LR,
                "num_epochs": s2g.STAGE2_EPOCHS,
                "per_device_batch_size": 1,
                "gradient_accumulation_steps": 8,
                "max_length": 1536,
                "max_image_pixels": 384 * 384,
            },
            "generation_contract_inherited_from":
                "data/reports/mllmu_iter12_selection_protocol_freeze.json",
        },
        "zero_weight_control": {
            "report": s3g.CONTROL_REPORT,
            "rule": (
                "the Stage-3 row with additional image-anchor weight 0 must "
                "map exactly to B6 structurally, and B6's bound prediction "
                "bytes must reproduce its filed eight retention values and "
                "D_G within the frozen tolerances"),
            "floor_tolerance": s3g.CONTROL_FLOOR_TOLERANCE,
            "d_g_tolerance": s3g.CONTROL_DG_TOLERANCE,
            "why_b6_is_reused_not_retrained": (
                "The zero-weight row is the same objective, the same trainer, "
                "the same seed and the same starting adapter. A second "
                "same-seed training run would measure GPU nondeterminism; "
                "reuse of the bound B6 bytes is the comparison that answers "
                "whether gamma=0 is B6."),
        },
        "evidence_base": {
            "used": [
                STAGE2_SELECTION,
                STAGE2_FREEZE,
                s2g.CALIBRATION_REPORT,
                PROBE_REPORT,
                ROUTE_PROBE_REPORT,
                "data/mllmu_hier_pilot100/unlearning_iter12/retain.jsonl",
                "data/mllmu_hier_pilot100/mf_reference_logprobs/sidecar.json",
            ],
            "never_read": list(rs.FORBIDDEN_EVIDENCE),
            "probe_entities_evaluation_only": True,
        },
    }


def check_only(repo_root: Path) -> list[str]:
    path = repo_root / OUT_REPORT
    if not path.exists():
        return [f"{path} does not exist"]
    old = _flat(json.loads(path.read_text()))
    fresh = _flat(build_report(repo_root))
    return [f"{k}: filed={old.get(k)!r} now={fresh.get(k)!r}"
            for k in sorted(set(old) | set(fresh))
            if old.get(k) != fresh.get(k)]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check-only", action="store_true")
    ap.add_argument("--repo-root", default=None)
    args = ap.parse_args()
    repo_root = Path(args.repo_root) if args.repo_root \
        else (_find_repo_root(Path.cwd()) or Path.cwd())
    out = repo_root / OUT_REPORT
    if args.check_only:
        reasons = check_only(repo_root)
        if reasons:
            print("MISMATCHES:")
            for r in reasons:
                print(f"  {r}")
            raise SystemExit(1)
        print(f"OK — {OUT_REPORT} re-derives from committed artifacts")
        return
    document = build_report(repo_root)
    if out.exists() and json.loads(out.read_text()) != document:
        raise SystemExit(
            f"REFUSING to overwrite {out}: it differs from the re-derived "
            f"basis. Inspect the mismatch with --check-only; a silent rewrite "
            f"is how a basis becomes post hoc.")
    out.write_text(json.dumps(document, indent=2, ensure_ascii=False) + "\n")
    log.info("Stage-3 basis -> %s", out)
    log.info("  parent: %s", document["b6_result"]["candidate_id"])
    log.info("  image-conditioned examples: %d of %d",
             document["image_anchor_examples"]["image_conditioned_examples"],
             document["image_anchor_examples"]["fit_half_examples"])
    log.info("  trained rows: %d", document["grid"]["num_trained"])


if __name__ == "__main__":
    main()
