"""Iteration 12 Stage 1: generate, apply the frozen floor, minimise D_G.

    python scripts/select_iter12_retention_checkpoints.py --device cuda:0

This is the executor for the protocol that
``scripts/freeze_iter12_selection_protocol.py`` committed.  It decides nothing
of its own: the criterion, its direction, the numerical tolerance, the
tie-break, the floor and the Stage-2 gate are all read from the freeze, and
the freeze is VERIFIED before a single query is generated.  If the code that
implements the protocol has changed since it was frozen, this script refuses
to run rather than selecting a successor under rules nobody preregistered.

Order of operations, and why each guard is where it is
------------------------------------------------------
1. verify the freeze (before generating: generation is the expensive part,
   and a mismatch found afterwards would leave artifacts produced under an
   unfrozen criterion);
2. refuse any path that resolves inside the sealed confirmation evidence;
3. generate MG and every candidate into ``predictions_iter12/`` under one
   generation contract, reusing the sealed pilot-100 generator so the
   predictions are comparable with the incumbent's rather than merely similar;
4. refuse to write a selection report unless the WHOLE grid has predictions --
   a report over a subset names a winner the grid does not support and looks
   complete, carrying the same dataset version as the real one;
5. apply the floor, then minimise D_G with the frozen tie-break.

``--generate-only`` stops after step 3, which is what makes generation
shardable across GPUs: only an unsharded run may write the report.

Needs a GPU.  Reads no confirmation prediction, and writes no sealed path.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import freeze_iter12_selection_protocol as fip

#: The sealed pilot-100 selector is imported, never copied: generation
#: semantics (batch layout, sidecar contract, coverage validation) have to be
#: IDENTICAL to the ones that produced the incumbent's numbers, and a
#: reimplementation would drift from them silently.  Importing also means a
#: change to that sealed file shows up as a change in behaviour here rather
#: than as a divergence nobody notices.
from select_unlearning_checkpoints import (
    _generate_state,
    _mg_reference_predictions,
)

from granunlearn.config import _find_repo_root
from granunlearn.evaluation import retention_selection as rs
from granunlearn.evaluation.prediction_provenance import dataset_version
from granunlearn.evaluation.reference_eval import (
    DEFAULT_MAX_IMAGE_PIXELS,
    DEFAULT_MAX_LENGTH,
    DEFAULT_MAX_NEW_TOKENS,
    load_associations_parquet,
    load_queries_parquet,
)
from granunlearn.logging_utils import setup_logger
from granunlearn.training.candidate_grid import (
    ITER12_REFERENCE_ROW,
    dataset_dir_for_tag,
    grid_for_tag,
    validate_grid,
)

log = setup_logger("select_iter12_retention")

TAG = "iter12"
SPLITS = ("train", "val")
EXPERIMENT_ID = "mllmu_iter12_stage1"
PREDICTIONS_SUBDIR = "predictions_iter12"
OUT_REPORT = "data/reports/mllmu_iter12_retention_selection.json"


def assert_no_forbidden_evidence(paths: list[Path],
                                 repo_root: Path) -> None:
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


def predictions_dir_for(repo_root: Path) -> Path:
    """The study's own predictions directory, never the pilot-100 one."""
    out = repo_root / dataset_dir_for_tag(TAG) / PREDICTIONS_SUBDIR
    shared = repo_root / dataset_dir_for_tag(TAG) / "predictions"
    if out.resolve() == shared.resolve():
        raise SystemExit(
            f"REFUSING: {out} is the pilot-100 predictions directory. Both "
            f"grids contain a candidate called B0, so sharing a directory "
            f"would regenerate over committed pilot-100 evidence the moment a "
            f"differing experiment_id made the sidecar refuse reuse.")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Iteration 12 Stage-1 selection under the frozen protocol")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--image-batch-size", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int,
                        default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--model-id", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--candidates", default=None,
                        help="Restrict generation to these candidate ids "
                             "(a lane that trained part of the grid). The "
                             "report is still only written for a full grid.")
    parser.add_argument("--generate-only", action="store_true",
                        help="Generate (or verify-reuse) predictions and stop, "
                             "without scoring or writing the report")
    parser.add_argument("--no-stage", action="store_true",
                        help="Score and report only; do not stage the winner")
    parser.add_argument("--repo-root", default=None)
    args = parser.parse_args()

    repo_root = Path(args.repo_root) if args.repo_root \
        else (_find_repo_root(Path.cwd()) or Path.cwd())
    data_dir = repo_root / dataset_dir_for_tag(TAG)
    predictions_dir = predictions_dir_for(repo_root)
    out_report = repo_root / OUT_REPORT

    #: Step 1 -- the freeze is verified before anything expensive happens.
    reasons = fip.verify_freeze(repo_root)
    if reasons:
        log.error("The frozen protocol does not match the repository: %d "
                  "field(s) differ", len(reasons))
        for r in reasons[:20]:
            log.error("    - %s", r)
        raise SystemExit(
            "REFUSING to select under an unfrozen criterion. Either restore "
            "the frozen code, or re-freeze deliberately -- which "
            "freeze_iter12_selection_protocol.py will itself refuse to do "
            "once a candidate adapter exists.")

    #: Step 2 -- the sealed confirmation is out of bounds.
    assert_no_forbidden_evidence(
        [data_dir, predictions_dir, out_report,
         repo_root / fip.CANDIDATE_ROOT], repo_root)

    generation_config = {
        "batch_size": args.batch_size,
        "image_batch_size": args.image_batch_size,
        "max_new_tokens": args.max_new_tokens,
        "do_sample": False,
        "max_image_pixels": DEFAULT_MAX_IMAGE_PIXELS,
        "max_length": DEFAULT_MAX_LENGTH,
    }
    frozen_contract = json.loads(
        (repo_root / fip.OUT_REPORT).read_text())["generation_contract"]
    #: Compared as a dict of what THIS run will actually do, rather than by
    #: reaching into two differently-shaped objects: ``experiment_id`` and
    #: ``model_id`` are not generation-config keys, and looking them up in the
    #: wrong place would compare None against a frozen string and refuse a
    #: correct run.
    actual = dict(generation_config)
    actual["model_id"] = args.model_id
    actual["experiment_id"] = EXPERIMENT_ID
    actual["predictions_dir"] = str(predictions_dir.relative_to(repo_root))
    for key, want in frozen_contract.items():
        if key == "why_not_the_pilot_predictions_dir":
            continue
        got = actual.get(key)
        if want != got:
            raise SystemExit(
                f"REFUSING: the generation contract is frozen at {key}="
                f"{want!r} but this run would use {got!r}. Batch layout is "
                f"inside D_G, so a different contract is a different "
                f"criterion.")

    probe = json.loads((repo_root / fip.PROBE_REPORT).read_text())
    probe_entities = probe["halves"]["probe"]["entities"]

    grid = grid_for_tag(TAG)
    errors = validate_grid(grid)
    if errors:
        raise SystemExit(f"the {TAG} grid is invalid: {errors}")
    wanted = None
    if args.candidates:
        wanted = {c.strip() for c in args.candidates.split(",") if c.strip()}
        unknown = wanted - {c.candidate_id for c in grid}
        if unknown:
            raise SystemExit(f"candidates outside the frozen grid: "
                             f"{sorted(unknown)}")

    queries = load_queries_parquet(data_dir / "queries.parquet")
    associations = load_associations_parquet(data_dir / "associations.parquet")
    by_assoc = {a.association_id: a for a in associations}
    predictions_dir.mkdir(parents=True, exist_ok=True)
    ckpt_root = repo_root / fip.CANDIDATE_ROOT

    #: Step 3 -- MG first, in this study's own directory and layout.
    mg_adapters = repo_root / "data" / "checkpoints" / "mllmu_pilot100" \
        / "MG" / "adapters"
    if not mg_adapters.exists():
        raise FileNotFoundError(
            f"MG adapter missing: {mg_adapters}. Selection must never "
            f"approximate the oracle with the bare base model.")
    mg_preds = _mg_reference_predictions(
        queries, by_assoc, repo_root, data_dir, predictions_dir, mg_adapters,
        args.model_id, args.device, generation_config, EXPERIMENT_ID, SPLITS)
    ref_vec = rs.trainval_vector(mg_preds, queries, associations)
    log.info("MG %s vector: %s", "+".join(SPLITS), ref_vec)

    generated: dict[str, dict[str, Any]] = {}
    missing_adapters: list[str] = []
    for spec in grid:
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
        generated[spec.candidate_id] = {
            "method": spec.method,
            "spec": spec.describe(),
            "trainval_metrics": rs.trainval_hierarchy_metrics(
                preds, queries, associations),
            "probe_retention": rs.probe_retention(
                preds, queries, associations, probe_entities),
            "selection_scope": list(SPLITS),
            "config": {k: summary.get(k) for k in
                       ("recipe", "groups", "num_optimizer_steps",
                        "init_adapter_dir", "noop")},
        }
        log.info("[%s] %s", spec.candidate_id,
                 json.dumps(generated[spec.candidate_id]["probe_retention"]))

    if args.generate_only:
        log.info("generate-only: %d candidate(s) have %s predictions in %s; "
                 "no report written, because a report over a subset would "
                 "name a winner the full grid does not support",
                 len(generated), "+".join(SPLITS), predictions_dir.name)
        return

    #: Step 4 -- the whole grid, or no report.
    if missing_adapters:
        raise SystemExit(
            f"REFUSING to write a selection report: {len(missing_adapters)} "
            f"grid candidate(s) have no adapter ({missing_adapters}). Train "
            f"them, or pass --generate-only; a partial report would look "
            f"complete and carry the same dataset_version as the real one.")

    if "B0" not in generated:
        raise SystemExit("REFUSING: B0 is the floor and is not in the run")
    b0_probe = generated["B0"]["probe_retention"]

    #: Step 5 -- floor first, then D_G, then the frozen tie-break.
    report = rs.select_successor(
        {cid: info for cid, info in generated.items()}, ref_vec, b0_probe)
    report["stage_2_gate"] = rs.stage2_gate(report, ITER12_REFERENCE_ROW)
    report["tag"] = TAG
    report["iteration"] = "12"
    report["stage"] = 1
    report["protocol_freeze"] = {
        "path": fip.OUT_REPORT,
        "verified_before_generation": True,
        #: The list ``verify_freeze`` actually returned -- empty because a
        #: non-empty one raised above.  Recorded rather than asserted, so the
        #: claim in this report is the value that was checked.
        "mismatches_found": reasons,
    }
    report["dataset_version"] = dataset_version(data_dir)
    report["probe_partition"] = {
        "path": fip.PROBE_REPORT,
        "num_probe_entities": len(probe_entities),
        "probe_entities_never_replayed": True,
    }
    report["predictions_dir"] = str(
        predictions_dir.relative_to(repo_root))
    report["generation_contract"] = generation_config
    report["model_id"] = args.model_id
    report["never_read"] = list(rs.FORBIDDEN_EVIDENCE)
    report["note"] = (
        "EXPLORATORY. This report selects a checkpoint; it tests no "
        "hypothesis and controls no error rate. Distance is a selection "
        "criterion only -- the individual metrics remain the scientific "
        "results, and any confirmatory claim about the winner needs its own "
        "sealed protocol, frozen before it is scored.")

    if not args.no_stage and report["selected"]:
        import shutil
        cid = report["selected"]
        src = ckpt_root / cid / "adapters"
        dst = ckpt_root / "selected" / "B4" / "adapters"
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
        summary = ckpt_root / cid / "training_summary.json"
        if summary.exists():
            shutil.copy(summary, dst.parent / "training_summary.json")
        report["staged"] = str(dst)
        log.info("SELECTED B4 <- %s (D_G=%.6f)", cid,
                 report["candidates"][cid]["distance_to_mg"])

    with open(out_report, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    log.info("eligible: %d/%d | disqualified by the floor: %s",
             len(report["eligible"]), len(report["candidates"]),
             report["disqualified_by_the_floor"])
    log.info("selected %s | tied_with %s | stage 2 opens: %s (%s)",
             report["selected"], report["tied_with"],
             report["stage_2_gate"]["stage2_opens"],
             report["stage_2_gate"]["reason"])
    log.info("Selection report -> %s", out_report)


if __name__ == "__main__":
    main()
