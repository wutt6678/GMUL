"""Train the MF/MG/MN reference states with an IDENTICAL recipe.

    python scripts/train_reference_states.py --states MF,MG,MN --device cuda:0

MF = Train(D_F), MG = Train(D_G), MN = Train(D_N).  The recipe (LoRA,
seed, optimizer, epochs, multimodal formatting) is shared verbatim; only
the knowledge dataset differs.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from granunlearn.config import _find_repo_root
from granunlearn.logging_utils import setup_logger
from granunlearn.training.reference_trainer import ReferenceRecipe, train_state

log = setup_logger("train_reference_states")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train reference states (identical recipe)")
    parser.add_argument("--states", default="MF,MG,MN")
    parser.add_argument("--tag", default="smoke",
                        choices=("smoke", "pilot100"))
    parser.add_argument("--smoke-dir", default=None,
                        help="Override the dataset directory "
                             "(default: data/mllmu_hier_<tag>)")
    parser.add_argument("--checkpoints-dir", default=None,
                        help="Override the checkpoint directory "
                             "(default: data/checkpoints/mllmu_<tag>)")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args()

    repo_root = _find_repo_root(Path.cwd()) or Path.cwd()
    smoke_dir = Path(args.smoke_dir or f"data/mllmu_hier_{args.tag}")
    ckpt_dir = Path(args.checkpoints_dir or
                    f"data/checkpoints/mllmu_{args.tag}")
    smoke_dir = smoke_dir if smoke_dir.is_absolute() else repo_root / smoke_dir
    ckpt_dir = ckpt_dir if ckpt_dir.is_absolute() else repo_root / ckpt_dir

    recipe = ReferenceRecipe()
    if args.epochs is not None:
        recipe = ReferenceRecipe(num_epochs=args.epochs)

    for state in [s.strip().upper() for s in args.states.split(",")]:
        dataset = smoke_dir / "training" / f"{state}.jsonl"
        if not dataset.exists():
            raise FileNotFoundError(
                f"{dataset} — run scripts/build_state_datasets.py first")
        train_state(state, dataset, ckpt_dir / state,
                    device=args.device, recipe=recipe)
    log.info("All requested states trained.")


if __name__ == "__main__":
    main()
