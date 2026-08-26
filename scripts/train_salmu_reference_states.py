"""Train the SALMU reference states MF/MG/MN from the Clean CLIP.

    python scripts/train_salmu_reference_states.py --device cuda:0

Identical ClipRecipe for every state; only D_F/D_G/D_N differ.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from granunlearn.config import _find_repo_root
from granunlearn.logging_utils import setup_logger
from granunlearn.salmu.adapter import REPOS, locate_repo
from granunlearn.salmu.clip_trainer import ClipRecipe, train_salmu_state

log = setup_logger("train_salmu_reference_states")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train SALMU reference states")
    parser.add_argument("--states", default="MF,MG,MN")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args()

    repo_root = _find_repo_root(Path.cwd()) or Path.cwd()
    train_dir = repo_root / "data" / "salmu_hierarchical" / "training"
    out_root = repo_root / "data" / "checkpoints" / "salmu"
    train_ds = locate_repo(REPOS["training_dataset"]["repo_id"], "dataset")
    clean = locate_repo(REPOS["clean_model"]["repo_id"], "model")
    init_ckpt = clean / "open_clip_model.safetensors"
    if not init_ckpt.exists():
        raise FileNotFoundError(f"Clean checkpoint missing: {init_ckpt}")

    recipe = ClipRecipe()
    if args.epochs is not None:
        from dataclasses import replace
        recipe = replace(recipe, num_epochs=args.epochs)

    for state in [s.strip().upper() for s in args.states.split(",")]:
        pairs_path = train_dir / f"{state}.jsonl"
        if not pairs_path.exists():
            raise FileNotFoundError(
                f"Missing state pairs: {pairs_path} "
                f"(run scripts/build_salmu_state_pairs.py)")
        train_salmu_state(
            state=state,
            pairs_path=pairs_path,
            parquet_dir=train_ds / "data",
            init_checkpoint=init_ckpt,
            output_dir=out_root / state,
            device=args.device,
            recipe=recipe,
        )
    log.info("All requested SALMU reference states trained.")


if __name__ == "__main__":
    main()
