"""Train the Iteration 12 Stage-3 route-aware B6 successors.

    python scripts/train_iter12_stage3.py --control
    python scripts/train_iter12_stage3.py --device cuda:0
    python scripts/train_iter12_stage3.py --device cuda:0 \
        --candidates B7_imgw3.4105_beta13.642_lam0.5_lr2e-05_ep5
    python scripts/train_iter12_stage3.py --plan-lanes 2 --emit-sh

Stage 3 does not add another copy of a training loop.  Every B7 row is
expressed as B6's three groups with one changed coefficient -- the effective
anchor weight on the image-conditioned retain stream -- and is trained by the
FROZEN ``train_with_preservation`` that Stage 2 used.  The zero-weight control
is B6 reused from its bound Stage-2 bytes, because a same-seed second run
would measure GPU nondeterminism rather than whether ``gamma = 0`` is B6.

Order of operations, and why each guard is where it is
------------------------------------------------------
1. verify the Stage-3 freeze and the clone-dependent bytes before GPU work;
2. refuse any path that resolves inside the sealed confirmation evidence;
3. require the zero-weight control to hold, so a B7 row cannot be interpreted
   as "B6 plus image weight" unless ``gamma = 0`` really is B6;
4. train only the four B7 rows, skipping rows whose adapters already exist.

Needs a GPU for training.  ``--control`` is CPU-only and is what files the
committed control report before the freeze is written.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import freeze_iter12_stage3 as fis3
from freeze_iter12_route_stratification import prediction_filename

from granunlearn.config import _find_repo_root
from granunlearn.evaluation import retention_selection as rs
from granunlearn.evaluation import route_stratified_retention as rt
from granunlearn.evaluation.reference_eval import (
    load_associations_parquet,
    load_predictions_parquet,
    load_queries_parquet,
)
from granunlearn.logging_utils import setup_logger
from granunlearn.training import stage2_grid as s2g
from granunlearn.training import stage3_grid as s3g
from granunlearn.training.preservation_anchor import sha256_file
from granunlearn.training.preservation_trainer import (
    PreservationGroupSpec,
    train_with_preservation,
    validate_stage2_groups,
)
from granunlearn.training.reference_trainer import ReferenceRecipe

log = setup_logger("train_iter12_stage3")

CONTROL_ID = "control_zero_image_anchor_weight_reproduces_b6"


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
                    f"confirmation is sealed evidence; Stage 3 is developed "
                    f"from exploratory train+val evidence only.")


def resolve_group_path(repo_root: Path, name: str,
                       declared: str) -> Path:
    """The group's real path, checked against the one the grid declares."""
    resolved = (s3g.groups_dir(repo_root) / f"{name}.jsonl").resolve()
    declared_path = (repo_root / declared).resolve()
    if resolved != declared_path:
        raise SystemExit(
            f"REFUSING: the Stage-3 grid declares {declared} for group "
            f"{name!r} but the fit-half directory resolves it to {resolved}.")
    if not resolved.exists():
        raise SystemExit(f"REFUSING: {resolved} does not exist")
    return resolved


def as_training_specs(repo_root: Path,
                      spec: s3g.Stage3CandidateSpec,
                      ) -> list[PreservationGroupSpec]:
    return [PreservationGroupSpec(
        name=g.name,
        path=resolve_group_path(repo_root, g.name, str(g.path)),
        mode=g.mode,
        weight=g.weight,
        cap=g.cap,
        anchor_weight=g.anchor_weight) for g in spec.groups]


def recipe_for(spec: s3g.Stage3CandidateSpec) -> ReferenceRecipe:
    allowed = {"learning_rate", "num_epochs", "lora_r", "lora_alpha",
               "lora_dropout", "weight_decay", "seed"}
    unknown = set(spec.overrides) - allowed
    if unknown:
        raise SystemExit(
            f"{spec.candidate_id}: overrides outside the swept-knob "
            f"allowlist: {sorted(unknown)}")
    return ReferenceRecipe(**spec.overrides)


def _eight_numbers(preds, queries, associations, probe_entities) -> dict:
    return rt.floor_vector(rt.probe_retention_by_route(
        preds, queries, associations, probe_entities))


def _vector(preds, queries, associations) -> dict:
    return rs.summary_vector(rs.trainval_hierarchy_metrics(
        preds, queries, associations))


def zero_weight_control(repo_root: Path) -> dict[str, Any]:
    """Recompute the B6 zero-weight control from bound bytes.

    Structural half: the Stage-3 ``gamma = 0`` row is exactly the Stage-2 B6
    objective.  Numerical half: the B6 prediction parquet reproduces the eight
    filed retention values and D_G within the frozen tolerances.  Both halves
    are required; either alone would leave a way for the control to pass while
    meaning something else.
    """
    calibration = json.loads((repo_root / s2g.CALIBRATION_REPORT).read_text())
    beta_star = calibration["grid_rule"]["beta_star"]
    grid = s3g.stage3_grid(beta_star)
    control = s3g.control_row(grid)
    parent = s3g.b6_parent(beta_star)
    recipe = recipe_for(control)

    retain = s3g.retain_group_path(repo_root)
    rows = [json.loads(line) for line in retain.read_text().splitlines()
            if line.strip()]
    image_rows = [r for r in rows if s3g.image_conditioned(r)]
    sidecar_path = s3g.reference_cache_dir(repo_root) / "sidecar.json"
    sidecar = json.loads(sidecar_path.read_text())
    cache_ids = {r["example_id"] for r in sidecar["rows"]}
    structural_checks = {
        "zero_row_maps_exactly_to_b6": s3g.same_objective_as_b6(
            control, parent),
        "all_fit_half_examples_are_image_conditioned":
            len(image_rows) == len(rows),
        "cache_covers_exactly_the_image_conditioned_group":
            cache_ids == {r["example_id"] for r in image_rows},
        "cache_was_built_from_the_same_mf_adapter_field":
            sidecar.get("mf_adapter_sha256") is not None,
        "training_seed_is_b6s": recipe.seed == 42,
        "learning_rate_is_b6s": recipe.learning_rate == s2g.STAGE2_LR,
        "epoch_budget_is_b6s": recipe.num_epochs == s2g.STAGE2_EPOCHS,
        "batch_layout_is_b6s": recipe.per_device_batch_size == 1
            and recipe.gradient_accumulation_steps == 8,
        "encoding_is_the_cache_contract":
            recipe.max_length == sidecar.get("max_length")
            and recipe.max_image_pixels == sidecar.get("max_image_pixels"),
    }

    stage2 = json.loads((repo_root / s2g.OUT_REPORT).read_text())
    filed = stage2["candidates"][control.candidate_id]
    data_dir = s3g.dataset_dir(repo_root)
    queries = load_queries_parquet(data_dir / "queries.parquet")
    associations = load_associations_parquet(data_dir / "associations.parquet")
    probe = json.loads((repo_root / "data/reports/"
                                  "mllmu_iter12_retention_probe.json").read_text())
    probe_entities = probe["halves"]["probe"]["entities"]

    b6_path = s3g.stage2_predictions_dir(repo_root) / prediction_filename(
        control.candidate_id)
    mg_path = s3g.stage1_predictions_dir(repo_root) / prediction_filename("MG")
    if not b6_path.exists() or not mg_path.exists():
        missing = [str(p) for p in (b6_path, mg_path) if not p.exists()]
        raise SystemExit(
            f"REFUSING: the zero-weight control needs the bound B6 and MG "
            f"prediction bytes; missing {missing}")
    b6_preds = load_predictions_parquet(b6_path)
    mg_preds = load_predictions_parquet(mg_path)

    recomputed_values = _eight_numbers(b6_preds, queries, associations,
                                       probe_entities)
    value_checks = {}
    for key in rt.FLOOR_NUMBER_KEYS:
        got = recomputed_values[key]
        want = filed["values"][key]
        diff = None if got is None else abs(got - want)
        value_checks[key] = {
            "filed": want,
            "recomputed": got,
            "abs_difference": diff,
            "within_tolerance": diff is not None
                and diff <= s3g.CONTROL_FLOOR_TOLERANCE,
        }
    mg_vec = _vector(mg_preds, queries, associations)
    b6_vec = _vector(b6_preds, queries, associations)
    dg, used = rs.distance_to_mg(b6_vec, mg_vec)
    dg_diff = None if dg is None else abs(dg - filed["distance_to_mg"])
    dg_check = {
        "filed": filed["distance_to_mg"],
        "recomputed": dg,
        "abs_difference": dg_diff,
        "used_components": used,
        "within_tolerance": dg_diff is not None
            and dg_diff <= s3g.CONTROL_DG_TOLERANCE,
    }

    holds = all(structural_checks.values()) and \
        all(v["within_tolerance"] for v in value_checks.values()) and \
        dg_check["within_tolerance"]
    return {
        "control": CONTROL_ID,
        "parent": control.candidate_id,
        "rule": (
            "additional image-anchor weight 0 is B6: same groups, modes, "
            "weights, paths, overrides, seed, batch layout, cache and "
            "starting adapter, and the bound B6 bytes reproduce the filed "
            "eight values and D_G within tolerance"),
        "tolerances": {
            "floor_values": s3g.CONTROL_FLOOR_TOLERANCE,
            "d_g": s3g.CONTROL_DG_TOLERANCE,
            "why": (
                "the floor values are filed at four decimals and must "
                "reproduce to float precision; D_G is filed at six decimals, "
                "so one unit in its last place is allowed"),
        },
        "structural": structural_checks,
        "numerical": {
            "eight_values": value_checks,
            "d_g": dg_check,
        },
        "bound_bytes": {
            "b6_predictions": str(b6_path.relative_to(repo_root)),
            "b6_predictions_sha256": sha256_file(b6_path),
            "mg_predictions": str(mg_path.relative_to(repo_root)),
            "mg_predictions_sha256": sha256_file(mg_path),
            "retain_group": str(retain.relative_to(repo_root)),
            "retain_group_sha256": sha256_file(retain),
            "cache_sidecar": str(sidecar_path.relative_to(repo_root)),
            "cache_sidecar_sha256": sha256_file(sidecar_path),
            "cache_tensor_sha256": sidecar.get("tensor_sha256"),
        },
        "image_conditioned_examples": len(image_rows),
        "fit_half_examples": len(rows),
        "why_b6_is_reused_not_retrained": (
            "The zero-weight row is the same objective, trainer, seed, cache "
            "and starting adapter. A second same-seed run would measure GPU "
            "nondeterminism; reuse of the bound B6 bytes is the control that "
            "answers whether gamma=0 is B6."),
        "holds": holds,
    }


def filed_control(repo_root: Path) -> dict[str, Any]:
    path = repo_root / s3g.CONTROL_REPORT
    if not path.exists():
        raise SystemExit(
            f"REFUSING: {s3g.CONTROL_REPORT} is missing. Run "
            f"train_iter12_stage3.py --control before freezing Stage 3.")
    return json.loads(path.read_text())


def require_control(repo_root: Path, recompute: bool = True) -> dict[str, Any]:
    filed = filed_control(repo_root)
    current = zero_weight_control(repo_root) if recompute else filed
    if current != filed:
        raise SystemExit(
            f"REFUSING: the zero-weight control no longer re-derives the "
            f"filed {s3g.CONTROL_REPORT}. The control is part of the protocol, "
            f"not a one-time note.")
    if not current["holds"]:
        raise SystemExit(
            "REFUSING to train B7 rows: the zero-weight control failed, so "
            "'B6 plus image weight' would not have a verified B6 baseline.")
    return current


def group_sizes(repo_root: Path) -> dict[str, int]:
    gdir = s3g.groups_dir(repo_root)
    return {p.stem: sum(1 for line in p.read_text().splitlines() if line.strip())
            for p in sorted(gdir.glob("*.jsonl"))}


def training_cost(spec: s3g.Stage3CandidateSpec, sizes: dict[str, int]) -> int:
    epochs = int(spec.overrides.get("num_epochs", 10))
    return epochs * sum(sizes[g.name] for g in spec.groups)


def plan_lanes(rows, sizes: dict[str, int], n_lanes: int) -> list[list]:
    if n_lanes < 1:
        raise ValueError("n_lanes must be >= 1")
    lanes: list[list] = [[] for _ in range(n_lanes)]
    loads = [0] * n_lanes
    for spec in sorted(rows, key=lambda s: (-training_cost(s, sizes),
                                            s.candidate_id)):
        i = min(range(n_lanes), key=lambda k: (loads[k], k))
        lanes[i].append(spec)
        loads[i] += training_cost(spec, sizes)
    return [lane for lane in lanes if lane]


def emit_plan(args, repo_root: Path) -> None:
    calibration = json.loads((repo_root / s2g.CALIBRATION_REPORT).read_text())
    grid = s3g.stage3_grid(calibration["grid_rule"]["beta_star"])
    rows = s3g.trained_rows(grid)
    sizes = group_sizes(repo_root)
    lanes = plan_lanes(rows, sizes, args.plan_lanes)
    total = sum(training_cost(c, sizes) for c in rows)
    print(f"# group sizes: {json.dumps(sizes)}")
    print(f"# {total} micro-batches over {len(rows)} row(s) in "
          f"{len(lanes)} lane(s)")
    for lane in lanes:
        ids = ",".join(c.candidate_id for c in lane)
        if args.emit_sh:
            print(ids)
        else:
            cost = sum(training_cost(c, sizes) for c in lane)
            print(f"lane: cost={cost} ({cost / max(total, 1):.0%}) "
                  f"n={len(lane)} -> {ids}")


def verify_or_refuse(repo_root: Path) -> None:
    reasons = fis3.verify_freeze(repo_root)
    clone = fis3.verify_clone_dependent(repo_root) \
        if (repo_root / fis3.OUT_REPORT).exists() else {}
    mismatches = [m for res in clone.values() for m in res["mismatch"]]
    if reasons or mismatches:
        raise SystemExit(
            "REFUSING to train under an unfrozen Stage-3 protocol:\n  "
            + "\n  ".join([*reasons[:20], *mismatches]))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--control", action="store_true",
                    help="Recompute and file/check the zero-weight B6 control; "
                         "CPU-only")
    ap.add_argument("--candidates", default=None,
                    help="Comma-separated B7 ids (default: all still missing)")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--plan-lanes", type=int, default=None, metavar="N")
    ap.add_argument("--emit-sh", action="store_true")
    ap.add_argument("--repo-root", default=None)
    args = ap.parse_args()
    repo_root = Path(args.repo_root) if args.repo_root \
        else (_find_repo_root(Path.cwd()) or Path.cwd())

    if args.plan_lanes is not None:
        emit_plan(args, repo_root)
        return

    if args.control:
        #: The first control run predates the freeze and writes the report the
        #: freeze will bind.  Later runs verify the freeze first and refuse to
        #: rewrite a differing report.
        if (repo_root / fis3.OUT_REPORT).exists():
            verify_or_refuse(repo_root)
        current = zero_weight_control(repo_root)
        out = repo_root / s3g.CONTROL_REPORT
        if out.exists() and json.loads(out.read_text()) != current:
            raise SystemExit(
                f"REFUSING to overwrite {s3g.CONTROL_REPORT}: the control "
                f"re-derives differently. Inspect it rather than replacing it.")
        out.write_text(json.dumps(current, indent=2, ensure_ascii=False) + "\n")
        if not current["holds"]:
            raise SystemExit("the zero-weight B6 control FAILED")
        print(f"OK — zero image-anchor weight reproduces B6 -> {out}")
        return

    verify_or_refuse(repo_root)
    assert_no_forbidden_evidence(
        [s3g.stage3_ckpt_root(repo_root), s3g.groups_dir(repo_root),
         s3g.reference_cache_dir(repo_root),
         repo_root / s3g.CONTROL_REPORT], repo_root)
    require_control(repo_root, recompute=True)

    calibration = json.loads((repo_root / s2g.CALIBRATION_REPORT).read_text())
    grid = s3g.stage3_grid(calibration["grid_rule"]["beta_star"])
    kept, _filtered = s3g.validate_stage3_grid(
        grid, calibration["grid_rule"]["beta_star"])
    if kept:
        raise SystemExit(f"REFUSING: the Stage-3 grid is invalid: {kept}")
    rows = s3g.trained_rows(grid)
    if args.candidates:
        wanted = {s.strip() for s in args.candidates.split(",") if s.strip()}
        unknown = wanted - {c.candidate_id for c in rows}
        if unknown:
            raise SystemExit(
                f"--candidates {sorted(unknown)} matches no trainable B7 row; "
                f"trainable ids: {sorted(c.candidate_id for c in rows)}")
        rows = [c for c in rows if c.candidate_id in wanted]

    ckpt_root = s3g.stage3_ckpt_root(repo_root)
    mf = s3g.mf_adapters(repo_root)
    cache_dir = s3g.reference_cache_dir(repo_root)
    if not mf.exists():
        raise SystemExit(f"REFUSING: the canonical MF adapter {mf} is missing")

    trained: list[dict[str, Any]] = []
    for spec in rows:
        out_dir = ckpt_root / spec.candidate_id
        if (out_dir / "adapters").exists() and not args.force:
            log.info("[%s] adapters exist — skipped", spec.candidate_id)
            continue
        groups = as_training_specs(repo_root, spec)
        errors = validate_stage2_groups(groups)
        if errors:
            raise SystemExit(
                f"REFUSING: {spec.candidate_id}'s objective is invalid: "
                f"{errors}")
        summary = train_with_preservation(
            spec.candidate_id, groups, out_dir, device=args.device,
            recipe=recipe_for(spec), init_adapter_dir=mf,
            cache_dir=cache_dir, mf_adapter_dir=mf)
        last = (summary["epochs"] or [{}])[-1]
        trained.append({
            "candidate_id": spec.candidate_id,
            "image_anchor_weight": spec.image_anchor_weight,
            "effective_anchor_weight": spec.effective_anchor_weight,
            "num_optimizer_steps": summary["num_optimizer_steps"],
            "train_seconds": summary["train_seconds"],
            "final_avg_loss_fine_target": last.get("avg_loss_fine_target"),
            "final_avg_kl_retain": last.get("avg_kl_retain"),
        })
        log.info("[%s] done in %.0fs | fine_target NLL %s | KL %s",
                 spec.candidate_id, summary["train_seconds"],
                 last.get("avg_loss_fine_target"), last.get("avg_kl_retain"))
    log.info("trained %d of %d requested B7 row(s)", len(trained), len(rows))
    if trained:
        print(json.dumps(trained, indent=2))


if __name__ == "__main__":
    main()
