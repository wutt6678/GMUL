"""Select unlearning checkpoints by distance to MG on TRAIN+VAL only.

    python scripts/select_unlearning_checkpoints.py --device cuda:0

Protocol (Iteration 9): hyperparameters are tuned on train/val probes
ONLY.  This script generates completions for every baseline candidate,
computes v(M) and D_G(M_U) on train+val, selects the best candidate
per method, stages it as selected/{METHOD}, and writes
data/reports/mllmu_smoke_unlearning_selection.json.

The FROZEN TEST SPLIT is never used here — it remains a genuine final
held-out evaluation run afterwards by evaluate_reference_states.py.
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
    select_checkpoints,
    summary_vector,
    trainval_hierarchy_metrics,
)
from granunlearn.logging_utils import setup_logger

log = setup_logger("select_unlearning")


def _generate_state(state_id: str, adapter_dir: Path | None,
                    queries, by_assoc, repo_root, device,
                    predictions_dir: Path, model_id: str,
                    batch_size: int):
    ppath = predictions_dir / f"predictions_{state_id}.parquet"
    if ppath.exists():
        log.info("[%s] reusing persisted predictions", state_id)
        return load_predictions_parquet(ppath)
    generator = ReferenceStateGenerator(
        model_id, device, adapter_dir=adapter_dir)
    raws = generator.generate_for_queries(
        queries, by_assoc, repo_root, batch_size=batch_size)
    generator.unload()
    preds = [score_query(q, by_assoc[q.association_id], raw,
                         experiment_id="mllmu_smoke_iter9",
                         checkpoint_id=state_id)
             for q, raw in zip(queries, raws)]
    import pandas as pd
    pd.DataFrame([json.loads(p.model_dump_json()) for p in preds]
                 ).to_parquet(ppath, index=False)
    return preds


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Distance-to-MG selection on train+val")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--model-id", default="Qwen/Qwen3.5-9B")
    args = parser.parse_args()

    repo_root = _find_repo_root(Path.cwd()) or Path.cwd()
    smoke = repo_root / "data" / "mllmu_hier_smoke"
    unlearn_ckpt = repo_root / "data" / "checkpoints" / "mllmu_smoke_unlearn"
    ref_ckpt = repo_root / "data" / "checkpoints" / "mllmu_smoke"
    predictions_dir = smoke / "predictions"

    queries = load_queries_parquet(smoke / "queries.parquet")
    associations = load_associations_parquet(smoke / "associations.parquet")
    by_assoc = {a.association_id: a for a in associations}

    # Candidate registry: candidate_id -> (method, adapter_dir)
    candidates_dirs: dict[str, tuple[str, Path | None]] = {
        "B0": ("B0", unlearn_ckpt / "B0" / "adapters"),
        "B1_lr2e-05": ("B1", unlearn_ckpt / "B1_lr2e-05" / "adapters"),
        "B1_lr1e-04": ("B1", unlearn_ckpt / "B1_lr1e-04" / "adapters"),
        "B2_lr1e-04": ("B2", unlearn_ckpt / "B2_lr1e-04" / "adapters"),
        "B3_lam1.0": ("B3", unlearn_ckpt / "B3_lam1.0" / "adapters"),
        "B3_lam0.5": ("B3", unlearn_ckpt / "B3_lam0.5" / "adapters"),
    }
    for cid, (_, adir) in candidates_dirs.items():
        if adir is not None and not adir.exists():
            raise FileNotFoundError(f"Missing candidate adapters: {adir}")

    # MG reference vector on TRAIN+VAL (oracle we want to approximate)
    mg_preds = load_predictions_parquet(
        predictions_dir / "predictions_MG.parquet")
    mg_tv = trainval_hierarchy_metrics(mg_preds, queries, associations)
    ref_vec = summary_vector(mg_tv)
    log.info("MG train+val vector: %s", ref_vec)

    candidates: dict[str, dict] = {}
    for cid, (method, adir) in candidates_dirs.items():
        preds = _generate_state(
            cid, adir, queries, by_assoc, repo_root, args.device,
            predictions_dir, args.model_id, args.batch_size)
        tv_metrics = trainval_hierarchy_metrics(preds, queries, associations)
        summary = json.loads(
            (unlearn_ckpt / cid / "training_summary.json").read_text()) \
            if (unlearn_ckpt / cid / "training_summary.json").exists() \
            else {}
        candidates[cid] = {
            "method": method,
            "trainval_metrics": tv_metrics,
            "config": {k: summary.get(k) for k in
                       ("recipe", "groups", "num_optimizer_steps")},
        }
        log.info("[%s] train+val FILR=%s TGA=%s", cid,
                 tv_metrics.get("filr"), tv_metrics.get("tga"))

    report = select_checkpoints(candidates, ref_vec)

    # Stage selected adapters under selected/{METHOD} and reuse their
    # predictions for the final evaluation run.
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
        shutil.copy(predictions_dir / f"predictions_{cid}.parquet",
                    predictions_dir / f"predictions_{method}.parquet")
        log.info("SELECTED %s <- %s (D_G=%.4f)", method, cid,
                 report["candidates"][cid]["distance_to_mg"])

    out = repo_root / "data" / "reports" / \
        "mllmu_smoke_unlearning_selection.json"
    with open(out, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    log.info("Selection report -> %s", out)


if __name__ == "__main__":
    main()
