"""Select unlearning checkpoints by distance to MG on TRAIN+VAL only.

    python scripts/select_unlearning_checkpoints.py --tag pilot100 \
        --device cuda:0

Protocol: hyperparameters are tuned on train/val probes ONLY.  This
script generates completions for every trained candidate, computes v(M)
and D_G(M_U) on train+val, selects the best candidate per method, stages
it as ``selected/{METHOD}``, and writes
``data/reports/mllmu_{tag}_unlearning_selection.json``.

The FROZEN TEST SPLIT is never used here.  For ``--tag pilot100`` the
default ``--splits train,val`` means test queries are not even GENERATED
at selection time (``--splits all``, the Iteration 9 smoke default,
materialises every split and filters afterwards).  Prediction parquets
are named for their split scope — ``predictions_{id}.parquet`` (all
splits) vs ``predictions_tv_{id}.parquet`` (train+val) — so a
selection-time artifact can never be mistaken for a full one.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from granunlearn.config import _find_repo_root
from granunlearn.evaluation.reference_eval import (
    ReferenceStateGenerator,
    load_associations_parquet,
    load_predictions_parquet,
    load_queries_parquet,
)
from granunlearn.evaluation.scoring import score_query
from granunlearn.evaluation.selection import (
    filter_predictions_by_splits,
    select_checkpoints,
    summary_vector,
    trainval_hierarchy_metrics,
)
from granunlearn.logging_utils import setup_logger
from granunlearn.training.candidate_grid import grid_for_tag

log = setup_logger("select_unlearning")

ALL_SPLITS = ("train", "val", "test")


def prediction_filename(state_id: str, splits: tuple[str, ...]) -> str:
    """Split-scoped prediction parquet name (scope is part of the name).

    Two different scopes must never produce the same filename, so the
    fallback spells the split names out in full rather than abbreviating
    them to initials ('val+test' would otherwise collide with the
    'train+val' selection scope).
    """
    scope = set(splits)
    if not scope:
        raise ValueError("at least one split is required")
    if scope >= set(ALL_SPLITS):
        return f"predictions_{state_id}.parquet"
    if scope == {"train", "val"}:
        return f"predictions_tv_{state_id}.parquet"
    if scope == {"test"}:
        return f"predictions_test_{state_id}.parquet"
    return f"predictions_{'-'.join(sorted(scope))}_{state_id}.parquet"


def _generate_state(state_id: str, adapter_dir: Path | None,
                    queries, by_assoc, repo_root, device,
                    predictions_dir: Path, model_id: str,
                    batch_size: int, image_batch_size: int,
                    experiment_id: str,
                    splits: tuple[str, ...]):
    """Generate (or reuse) predictions for one checkpoint over `splits`."""
    ppath = predictions_dir / prediction_filename(state_id, splits)
    if ppath.exists():
        log.info("[%s] reusing persisted predictions %s", state_id,
                 ppath.name)
        return load_predictions_parquet(ppath)
    subset = queries if set(splits) >= set(ALL_SPLITS) else \
        [q for q in queries if q.split in splits]
    log.info("[%s] generating %d/%d queries (%s)...", state_id,
             len(subset), len(queries), ",".join(splits))
    generator = ReferenceStateGenerator(
        model_id, device, adapter_dir=adapter_dir)
    raws = generator.generate_for_queries(
        subset, by_assoc, repo_root, batch_size=batch_size,
        image_batch_size=image_batch_size)
    generator.unload()
    preds = [score_query(q, by_assoc[q.association_id], raw,
                         experiment_id=experiment_id,
                         checkpoint_id=state_id)
             for q, raw in zip(subset, raws)]
    import pandas as pd
    pd.DataFrame([json.loads(p.model_dump_json()) for p in preds]
                 ).to_parquet(ppath, index=False)
    return preds


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Distance-to-MG selection on train+val")
    parser.add_argument("--tag", default="smoke",
                        choices=("smoke", "pilot100"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--image-batch-size", type=int, default=1)
    parser.add_argument("--model-id", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--splits", default=None,
                        help="Comma-separated generation scope (default: "
                             "train,val for pilot100; all for smoke)")
    parser.add_argument("--candidates", default=None,
                        help="Restrict to these candidate ids (a lane that "
                             "trained only part of the grid)")
    parser.add_argument("--no-stage", action="store_true",
                        help="Score + report only; do not stage selected "
                             "adapters (used by partial lanes)")
    args = parser.parse_args()

    repo_root = _find_repo_root(Path.cwd()) or Path.cwd()
    data_dir = repo_root / "data" / f"mllmu_hier_{args.tag}"
    unlearn_ckpt = repo_root / "data" / "checkpoints" / \
        f"mllmu_{args.tag}_unlearn"
    predictions_dir = data_dir / "predictions"
    predictions_dir.mkdir(parents=True, exist_ok=True)

    splits = tuple(s.strip() for s in args.splits.split(",")) \
        if args.splits else \
        (("train", "val") if args.tag == "pilot100" else ALL_SPLITS)
    unknown = set(splits) - set(ALL_SPLITS)
    if unknown:
        raise SystemExit(f"unknown splits: {sorted(unknown)}")
    experiment_id = f"mllmu_{args.tag}_iter11" if args.tag == "pilot100" \
        else "mllmu_smoke_iter9"

    queries = load_queries_parquet(data_dir / "queries.parquet")
    associations = load_associations_parquet(data_dir / "associations.parquet")
    by_assoc = {a.association_id: a for a in associations}

    # Candidate registry from the frozen grid: only candidates whose
    # adapters actually exist on disk are scored (a lane may have trained
    # a subset); B0 is included because it is the pipeline self-test.
    grid = grid_for_tag(args.tag)
    wanted = None
    if args.candidates:
        wanted = {c.strip() for c in args.candidates.split(",") if c.strip()}
    candidates_dirs: dict[str, tuple[str, Path | None]] = {}
    for spec in grid:
        if wanted is not None and spec.candidate_id not in wanted:
            continue
        # B0 is a real checkpoint directory (the MF adapter COPIED
        # unchanged), not the bare base model: loading it through the
        # same adapter path is what makes "B0 must reproduce MF" a
        # pipeline self-test rather than an assumption.
        adir = unlearn_ckpt / spec.candidate_id / "adapters"
        if not adir.exists():
            log.warning("[%s] adapters missing — skipped", spec.candidate_id)
            continue
        candidates_dirs[spec.candidate_id] = (spec.method, adir)
    if not candidates_dirs:
        raise SystemExit("no trained candidates found — run "
                         "scripts/train_unlearning_baselines.py first")

    # MG reference vector on the SELECTION scope (oracle to approximate).
    # MG's predictions come from the reference-state evaluation; when
    # those were generated over all splits they are filtered down here,
    # so the reference vector never sees a different scope than the
    # candidates.
    mg_path = predictions_dir / prediction_filename("MG", splits)
    if not mg_path.exists():
        mg_full = predictions_dir / prediction_filename("MG", ALL_SPLITS)
        if not mg_full.exists():
            raise FileNotFoundError(
                f"MG predictions missing ({mg_path.name} / {mg_full.name}) — "
                f"run the reference-state evaluation first")
        mg_preds = filter_predictions_by_splits(
            load_predictions_parquet(mg_full), queries, splits)
    else:
        mg_preds = load_predictions_parquet(mg_path)
    mg_tv = trainval_hierarchy_metrics(mg_preds, queries, associations)
    ref_vec = summary_vector(mg_tv)
    log.info("MG %s vector: %s", "+".join(splits), ref_vec)

    candidates: dict[str, dict] = {}
    for cid, (method, adir) in candidates_dirs.items():
        preds = _generate_state(
            cid, adir, queries, by_assoc, repo_root, args.device,
            predictions_dir, args.model_id, args.batch_size,
            args.image_batch_size, experiment_id, splits)
        tv_metrics = trainval_hierarchy_metrics(preds, queries, associations)
        summary_path = unlearn_ckpt / cid / "training_summary.json"
        summary = json.loads(summary_path.read_text()) \
            if summary_path.exists() else {}
        candidates[cid] = {
            "method": method,
            "trainval_metrics": tv_metrics,
            "selection_scope": list(splits),
            "config": {k: summary.get(k) for k in
                       ("recipe", "groups", "num_optimizer_steps",
                        "init_adapter_dir", "noop")},
        }
        log.info("[%s] train+val FILR=%s TGA=%s retain_same=%s "
                 "retain_other=%s wrong=%s", cid,
                 tv_metrics.get("filr"), tv_metrics.get("tga"),
                 (tv_metrics.get("retain_same_entity") or {}).get(
                     "baseline_accuracy"),
                 (tv_metrics.get("retain_other_entity") or {}).get(
                     "baseline_accuracy"),
                 (tv_metrics.get("failure_rates") or {}).get("wrong_branch"))

    report = select_checkpoints(candidates, ref_vec)
    report["tag"] = args.tag
    report["selection_scope"] = list(splits)
    report["scope_note"] = (
        "test queries were NOT generated at selection time"
        if set(splits) < set(ALL_SPLITS) else
        "all splits generated; metrics filtered to train+val")

    if not args.no_stage:
        import shutil
        for method, cid in report["selected"].items():
            if cid is None:
                continue
            src = unlearn_ckpt / cid / "adapters"
            dst = unlearn_ckpt / "selected" / method / "adapters"
            dst.parent.mkdir(parents=True, exist_ok=True)
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(src, dst)
            shutil.copy(unlearn_ckpt / cid / "training_summary.json",
                        unlearn_ckpt / "selected" / method /
                        "training_summary.json")
            log.info("SELECTED %s <- %s (D_G=%.4f)", method, cid,
                     report["candidates"][cid]["distance_to_mg"])
        report["staged"] = {
            m: str(unlearn_ckpt / "selected" / m / "adapters")
            for m, c in report["selected"].items() if c is not None}

    out = repo_root / "data" / "reports" / \
        f"mllmu_{args.tag}_unlearning_selection.json"
    with open(out, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    log.info("Selection report -> %s", out)


if __name__ == "__main__":
    main()
