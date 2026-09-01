"""Train the SALMU unlearning candidates B0-B3 (Iteration 10 stage 4).

    python scripts/train_salmu_unlearning_baselines.py --device cuda:3

All candidates continue from MF^SALMU (full checkpoint, not LoRA).
The B3 group STRUCTURE is fixed from Iteration 9:
gd fine_target + sft target_level + sft retain.  Only lr / lambda are
varied, and selection will later use train/val probe personas only.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from granunlearn.config import _find_repo_root
from granunlearn.logging_utils import setup_logger
from granunlearn.salmu.adapter import REPOS, locate_repo
from granunlearn.salmu.clip_trainer import ClipRecipe
from granunlearn.salmu.paths import SalmuPaths
from granunlearn.salmu.state_datasets import load_state_pairs
from granunlearn.salmu.unlearning import (
    SalmuGroupSpec,
    build_salmu_unlearning_groups,
    make_salmu_noop_checkpoint,
    train_salmu_unlearning,
)

log = setup_logger("train_salmu_unlearning_baselines")

# (method, lr, num_epochs, batch_size, lambda_gd, constrained_gd)
# v1 (unconstrained ascent, kept for the registry): full-model gradient
# ascent collapsed the embedding space (sims -> -0.9 everywhere).
# v2: constrained ascent — stop when fine-pair similarity reaches the
# MG anchor level; B2 gets lower lrs after the 1e-4 collapse.
CANDIDATES = [
    ("B1", 1e-5, 10, 128, None, False),
    ("B1", 2e-5, 10, 128, None, False),
    ("B1", 1e-4, 10, 128, None, False),
    ("B2", 1e-4, 10, 128, None, False),
    ("B3", 1e-5, 3, 256, 0.5, False),
    ("B3", 1e-5, 3, 256, 1.0, False),
    ("B1", 2e-6, 10, 128, None, True),
    ("B1", 5e-6, 10, 128, None, True),
    ("B1", 1e-5, 10, 128, None, True),
    ("B2", 2e-5, 10, 128, None, False),
    ("B2", 5e-5, 10, 128, None, False),
    ("B2_retain", 5e-6, 10, 128, None, False),   # B2 + explicit retain group
    ("B2_retain", 1e-5, 10, 128, None, False),
    ("B2_retain", 2e-5, 10, 128, None, False),
    ("B2_retain", 5e-5, 10, 128, None, False),
    ("B3", 2e-6, 3, 256, 0.5, True),
    ("B3", 2e-6, 3, 256, 1.0, True),
    ("B3", 5e-6, 3, 256, 0.5, True),
    ("B3", 5e-6, 3, 256, 1.0, True),
]


def candidate_id(method: str, lr: float, lam: float | None,
                 constrained: bool = False) -> str:
    tag = f"{method}_lr{lr:g}"
    if lam is not None:
        tag += f"_lam{lam:g}"
    if constrained:
        tag += "_c"
    return tag


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train SALMU unlearning candidates")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--only", default=None,
                        help="comma-separated candidate ids to run")
    parser.add_argument("--suffix", default="",
                        help="Iteration tag (e.g. r5 -> holdout-clean "
                             "artifacts under *_r5 paths)")
    args = parser.parse_args()

    repo_root = _find_repo_root(Path.cwd()) or Path.cwd()
    paths = SalmuPaths(repo_root, suffix=args.suffix)
    train_ds = locate_repo(REPOS["training_dataset"]["repo_id"], "dataset")
    bench = locate_repo(REPOS["benchmark_dataset"]["repo_id"], "dataset")
    hier_dir = paths.hier_dir
    mf_dir = paths.ref_ckpt_root / "MF"
    mg_dir = paths.ref_ckpt_root / "MG"
    out_root = paths.unlearn_root

    identities = json.loads((bench / "identities_metadata.json").read_text())
    hierarchies = json.loads((repo_root / "data" / "salmu_hierarchical" /
                              "associations.json").read_text())
    mf_pairs = load_state_pairs(paths.mf_pairs_path)
    groups = build_salmu_unlearning_groups(
        mf_pairs, hierarchies, identities,
        out_dir=paths.groups_dir)

    only = set(args.only.split(",")) if args.only else None

    make_salmu_noop_checkpoint(mf_dir, out_root / "B0")

    for method, lr, epochs, bs, lam, constrained in CANDIDATES:
        cid = candidate_id(method, lr, lam, constrained)
        if only and cid not in only:
            continue
        out_dir = out_root / cid
        if (out_dir / "pytorch_model.bin").exists():
            log.info("[%s] checkpoint exists — skipping", cid)
            continue
        recipe = replace(ClipRecipe(), learning_rate=lr,
                         num_epochs=epochs, batch_size=bs)
        if method == "B1":
            specs = [SalmuGroupSpec("fine_target",
                                    groups["fine_target"], "gd", 1.0)]
        elif method == "B2":
            specs = [SalmuGroupSpec("target_level",
                                    groups["target_level"], "sft", 1.0)]
        elif method == "B2_retain":
            specs = [
                SalmuGroupSpec("target_level", groups["target_level"],
                               "sft", 1.0),
                SalmuGroupSpec("retain", groups["retain"], "sft", 1.0),
            ]
        else:  # B3: fixed structure, lambda only scales the gd term
            specs = [
                SalmuGroupSpec("fine_target", groups["fine_target"],
                               "gd", lam),
                SalmuGroupSpec("target_level", groups["target_level"],
                               "sft", 1.0),
                SalmuGroupSpec("retain", groups["retain"], "sft", 1.0),
            ]
        train_salmu_unlearning(
            candidate_id=cid,
            group_specs=specs,
            parquet_dir=train_ds / "data",
            init_checkpoint=mf_dir / "pytorch_model.bin",
            output_dir=out_dir,
            device=args.device,
            recipe=recipe,
            anchor_checkpoint=mg_dir / "pytorch_model.bin"
            if constrained and method != "B2" else None,
            gd_stop_sim=None,  # stop AT the MG anchor level
            gd_probe_pairs=groups["fine_target"]
            if constrained and method != "B2" else None,
        )
    log.info("SALMU unlearning candidates done.")


if __name__ == "__main__":
    main()
