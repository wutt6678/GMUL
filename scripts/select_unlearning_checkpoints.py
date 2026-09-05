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
from granunlearn.evaluation.prediction_provenance import (
    SUPERSEDED_V1_COMMIT,
    PredictionFingerprint,
    dataset_version,
    validate_prediction_coverage,
    verify_sidecar,
    write_sidecar,
)
from granunlearn.evaluation.reference_eval import (
    DEFAULT_MAX_IMAGE_PIXELS,
    DEFAULT_MAX_LENGTH,
    DEFAULT_MAX_NEW_TOKENS,
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
                    predictions_dir: Path, data_dir: Path, model_id: str,
                    generation_config: dict,
                    experiment_id: str,
                    splits: tuple[str, ...]):
    """Generate (or reuse) predictions for one checkpoint over `splits`.

    Reuse requires a provenance sidecar that matches this run's adapter,
    base-model revision, dataset version and artifact hashes, generation
    configuration and code fingerprint.  Selection decides which candidate
    is reported, so a silently stale parquet here would pick the winner on
    someone else's numbers.
    """
    ppath = predictions_dir / prediction_filename(state_id, splits)
    subset = queries if set(splits) >= set(ALL_SPLITS) else \
        [q for q in queries if q.split in splits]
    expected = PredictionFingerprint.build(
        experiment_id=experiment_id, checkpoint_id=state_id,
        repo_root=repo_root, data_dir=data_dir, model_id=model_id,
        adapter_dir=adapter_dir, generation_config=generation_config,
        num_rows=len(subset))
    if ppath.exists():
        reasons = verify_sidecar(ppath, expected)
        if reasons:
            log.warning("[%s] REFUSING to reuse %s — %d provenance "
                        "mismatch(es):", state_id, ppath.name, len(reasons))
            for r in reasons:
                log.warning("    - %s", r)
        else:
            preds = load_predictions_parquet(ppath)
            problems = validate_prediction_coverage(
                preds, [q.query_id for q in subset], experiment_id, state_id)
            if problems:
                raise SystemExit(
                    f"[{state_id}] {ppath.name} is provenance-valid but "
                    f"row-invalid:\n  " + "\n  ".join(problems))
            log.info("[%s] reusing provenance-validated predictions %s "
                     "(%d rows)", state_id, ppath.name, len(preds))
            return preds
    log.info("[%s] generating %d/%d queries (%s)...", state_id,
             len(subset), len(queries), ",".join(splits))
    generator = ReferenceStateGenerator(
        model_id, device, adapter_dir=adapter_dir)
    raws = generator.generate_for_queries(
        subset, by_assoc, repo_root,
        batch_size=generation_config["batch_size"],
        image_batch_size=generation_config["image_batch_size"],
        max_new_tokens=generation_config["max_new_tokens"])
    generator.unload()
    preds = [score_query(q, by_assoc[q.association_id], raw,
                         experiment_id=experiment_id,
                         checkpoint_id=state_id)
             for q, raw in zip(subset, raws)]
    problems = validate_prediction_coverage(
        preds, [q.query_id for q in subset], experiment_id, state_id)
    if problems:
        raise SystemExit(f"[{state_id}] freshly generated predictions are "
                         f"row-invalid:\n  " + "\n  ".join(problems))
    import pandas as pd
    pd.DataFrame([json.loads(p.model_dump_json()) for p in preds]
                 ).to_parquet(ppath, index=False)
    write_sidecar(ppath, expected)
    return preds


def _mg_reference_predictions(
    queries, by_assoc, repo_root, data_dir, predictions_dir,
    mg_adapters: Path, model_id: str, device: str,
    generation_config: dict, experiment_id: str,
    splits: tuple[str, ...],
):
    """MG's behaviour vector on the SELECTION scope (the oracle to approach).

    Two provenance-verified sources are acceptable, in order: a file
    already scoped to ``splits``, or the reference-state gate's all-split
    file filtered down — filtering is legitimate because the rows are the
    same generations, only the scope narrows, and the sidecar still has to
    match this run's dataset, adapter, configuration and code.  Neither
    being reusable regenerates the scoped file, which requires MG's
    adapter: scoring MG with ``adapter_dir=None`` would silently substitute
    the bare base model for the oracle.
    """
    scoped = predictions_dir / prediction_filename("MG", splits)
    expected = PredictionFingerprint.build(
        experiment_id=experiment_id, checkpoint_id="MG",
        repo_root=repo_root, data_dir=data_dir, model_id=model_id,
        adapter_dir=mg_adapters, generation_config=generation_config)
    subset_ids = [q.query_id for q in queries
                  if set(splits) >= set(ALL_SPLITS) or q.split in splits]
    for path, filter_to_scope in ((scoped, False),
                                  (predictions_dir / prediction_filename(
                                      "MG", ALL_SPLITS), True)):
        if not path.exists():
            continue
        reasons = verify_sidecar(path, expected)
        if reasons:
            log.warning("[MG] not reusing %s — %d provenance mismatch(es):",
                        path.name, len(reasons))
            for r in reasons:
                log.warning("    - %s", r)
            continue
        preds = load_predictions_parquet(path)
        if filter_to_scope:
            preds = filter_predictions_by_splits(preds, queries, splits)
        problems = validate_prediction_coverage(
            preds, subset_ids, experiment_id, "MG")
        if problems:
            raise SystemExit(f"[MG] {path.name} is provenance-valid but "
                             f"row-invalid:\n  " + "\n  ".join(problems))
        log.info("[MG] reusing provenance-validated predictions %s (%d rows)",
                 path.name, len(preds))
        return preds
    return _generate_state("MG", mg_adapters, queries, by_assoc, repo_root,
                           device, predictions_dir, data_dir, model_id,
                           generation_config, experiment_id, splits)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Distance-to-MG selection on train+val")
    parser.add_argument("--tag", default="smoke",
                        choices=("smoke", "pilot100"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--image-batch-size", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int,
                        default=DEFAULT_MAX_NEW_TOKENS)
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

    # One generation contract for the whole selection run: MG and every
    # candidate are scored under identical batch sizes, so D_G compares
    # models rather than batch layouts.  It is also what the prediction
    # sidecars are verified against.
    generation_config = {
        "batch_size": args.batch_size,
        "image_batch_size": args.image_batch_size,
        "max_new_tokens": args.max_new_tokens,
        "do_sample": False,
        "max_image_pixels": DEFAULT_MAX_IMAGE_PIXELS,
        "max_length": DEFAULT_MAX_LENGTH,
    }

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
    mg_adapters = repo_root / "data" / "checkpoints" / \
        f"mllmu_{args.tag}" / "MG" / "adapters"
    if not mg_adapters.exists():
        raise FileNotFoundError(
            f"MG adapter missing: {mg_adapters} — run the reference-state "
            f"training first (selection must never approximate the oracle "
            f"with the bare base model)")
    mg_preds = _mg_reference_predictions(
        queries, by_assoc, repo_root, data_dir, predictions_dir,
        mg_adapters, args.model_id, args.device, generation_config,
        experiment_id, splits)
    mg_tv = trainval_hierarchy_metrics(mg_preds, queries, associations)
    ref_vec = summary_vector(mg_tv)
    log.info("MG %s vector: %s", "+".join(splits), ref_vec)

    candidates: dict[str, dict] = {}
    for cid, (method, adir) in candidates_dirs.items():
        preds = _generate_state(
            cid, adir, queries, by_assoc, repo_root, args.device,
            predictions_dir, data_dir, args.model_id, generation_config,
            experiment_id, splits)
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
    report["iteration"] = "11R"
    report["dataset_version"] = dataset_version(data_dir)
    report["selection_scope"] = list(splits)
    report["scope_note"] = (
        "test queries were NOT generated at selection time"
        if set(splits) < set(ALL_SPLITS) else
        "all splits generated; metrics filtered to train+val")
    report["supersedes"] = {
        "commit": SUPERSEDED_V1_COMMIT,
        "dataset_version": "pilot100_v1",
        "reason": (
            "The v1 selection ranked candidates on train+val predictions "
            "generated over assoc.images[0] — the photograph every "
            "checkpoint was trained on — for all three splits. Iteration "
            "11R re-froze the dataset so val queries are served "
            "photographs disjoint from training, which changes the val "
            "half of the selection basis, so the ranking is recomputed "
            "rather than inherited."),
    }

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
