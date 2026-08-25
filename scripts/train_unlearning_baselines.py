"""Train Iteration 9 MF->MU baselines (small comparative sweep).

    python scripts/train_unlearning_baselines.py --device cuda:0

Methods (all continuing from the canonical MF adapter):
* B0 no-op:       MF adapter copied unchanged (sanity baseline)
* B1 complete-forget:      gradient ascent on fine_target
* B2 coarse-positive SFT:  SFT on target_level completions
* B3 granularity-aware:    gd fine_target + sft target_level + sft retain

Sweep knobs are tuned on train/val probes ONLY by
scripts/select_unlearning_checkpoints.py; the frozen test split is a
genuine held-out evaluation (Iteration 9 protocol).

Candidates (kept deliberately small):
* B1: lr in {2e-5, 1e-4}, 10 epochs
* B2: lr 1e-4, 10 epochs
* B3: suppress weight lam in {1.0, 0.5}, lr 1e-4, 10 epochs
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from granunlearn.config import _find_repo_root
from granunlearn.logging_utils import setup_logger
from granunlearn.training.reference_trainer import ReferenceRecipe
from granunlearn.training.unlearning_trainer import (
    GroupSpec,
    make_noop_checkpoint,
    train_unlearning,
)

log = setup_logger("train_unlearning_baselines")


def candidate_configs() -> list[dict]:
    """The full small sweep: (candidate_id, method, groups, recipe overrides)."""
    g = lambda name: "data/mllmu_hier_smoke/unlearning"  # noqa: E731
    configs = []
    for lr in (2e-5, 1e-4):
        configs.append({
            "candidate_id": f"B1_lr{lr:g}",
            "method": "B1",
            "groups": [GroupSpec("fine_target",
                                 f"{g('')}/fine_target.jsonl", "gd", 1.0)],
            "overrides": {"learning_rate": lr},
        })
    configs.append({
        "candidate_id": "B2_lr1e-04",
        "method": "B2",
        "groups": [GroupSpec("target_level",
                             f"{g('')}/target_level.jsonl", "sft", 1.0)],
        "overrides": {"learning_rate": 1e-4},
    })
    for lam in (1.0, 0.5):
        configs.append({
            "candidate_id": f"B3_lam{lam}",
            "method": "B3",
            "groups": [
                GroupSpec("fine_target", f"{g('')}/fine_target.jsonl",
                          "gd", lam),
                GroupSpec("target_level", f"{g('')}/target_level.jsonl",
                          "sft", 1.0),
                GroupSpec("retain", f"{g('')}/retain.jsonl", "sft", 1.0),
            ],
            "overrides": {"learning_rate": 1e-4},
        })
    return configs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train MF->MU baseline candidates")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--methods", default="B0,B1,B2,B3",
                        help="Comma-separated method filter")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override epoch budget for all candidates")
    args = parser.parse_args()

    repo_root = _find_repo_root(Path.cwd()) or Path.cwd()
    out_root = repo_root / "data" / "checkpoints" / "mllmu_smoke_unlearn"
    mf_adapters = repo_root / "data" / "checkpoints" / "mllmu_smoke" / \
        "MF" / "adapters"
    if not mf_adapters.exists():
        raise FileNotFoundError(f"Canonical MF adapter missing: {mf_adapters}")
    methods = {m.strip().upper() for m in args.methods.split(",")}

    if "B0" in methods:
        make_noop_checkpoint("B0", mf_adapters, out_root / "B0")

    for cfg in candidate_configs():
        if cfg["method"] not in methods:
            continue
        recipe = ReferenceRecipe(**cfg["overrides"])
        if args.epochs is not None:
            recipe = replace(recipe, num_epochs=args.epochs)
        train_unlearning(
            method_id=cfg["candidate_id"],
            groups=cfg["groups"],
            output_dir=out_root / cfg["candidate_id"],
            device=args.device,
            recipe=recipe,
            init_adapter_dir=mf_adapters,
        )
    log.info("All requested baseline candidates trained -> %s", out_root)


if __name__ == "__main__":
    main()
