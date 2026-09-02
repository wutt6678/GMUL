"""Build the MF/MG/MN reference-state knowledge datasets (Iteration 7).

    python scripts/build_state_datasets.py --config configs/datasets/mllmu.yaml

Training data is derived BY STATE from the committed smoke associations +
F/R partition — never from evaluation queries (the adversarial
negation_correction family stays evaluation-only).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from granunlearn.config import _find_repo_root
from granunlearn.evaluation.reference_eval import (
    load_associations_parquet,
)
from granunlearn.logging_utils import setup_logger
from granunlearn.training.state_datasets import write_state_datasets

log = setup_logger("build_state_datasets")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build D_F / D_G / D_N for the reference states")
    parser.add_argument("--tag", default="smoke",
                        help="Dataset tag: smoke -> data/mllmu_hier_"
                             "smoke + mllmu_smoke_* reports; pilot100 "
                             "-> the Iteration-11 mixed dataset")
    parser.add_argument("--smoke-dir", default=None,
                        help="Override the dataset directory "
                             "(default: data/mllmu_hier_<tag>)")
    args = parser.parse_args()

    repo_root = _find_repo_root(Path.cwd()) or Path.cwd()
    smoke_dir = Path(args.smoke_dir or f"data/mllmu_hier_{args.tag}")
    if not smoke_dir.is_absolute():
        smoke_dir = repo_root / smoke_dir

    associations = load_associations_parquet(
        smoke_dir / "associations.parquet")
    partition = json.loads(
        (repo_root / "data" / "reports" /
         f"mllmu_{args.tag}_target_retain.json").read_text())

    manifest = write_state_datasets(
        associations, partition, smoke_dir / "training",
        repo_root=repo_root)
    for state, info in manifest["states"].items():
        log.info(
            "%s: %d examples (target %d / retain %d) | image_text %d | "
            "target levels %s",
            state, info["num_examples"], info["num_target"],
            info["num_retain"], info["num_image_text"],
            info["target_level_distribution"])
    log.info("Wrote state datasets -> %s", smoke_dir / "training")


if __name__ == "__main__":
    main()
