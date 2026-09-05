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
from granunlearn.evaluation.image_splits import assert_no_training_drift
from granunlearn.evaluation.prediction_provenance import dataset_version
from granunlearn.evaluation.reference_eval import (
    load_associations_parquet,
)
from granunlearn.logging_utils import setup_logger
from granunlearn.training.state_datasets import write_state_datasets

log = setup_logger("build_state_datasets")

#: The three files MF/MG/MN are fitted on.
STATE_JSONLS = ("training/MF.jsonl", "training/MG.jsonl",
                "training/MN.jsonl")


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
    # Iteration 11R freeze gate: the repaired visual split reserves
    # images[0] as the training photograph precisely so that rebuilding on
    # pilot100_v2 reproduces these three files byte-for-byte.  If a future
    # edit re-picks the training photograph, MF/MG/MN and every candidate
    # fitted on them become stale, and the build must say so instead of
    # writing new bytes under an old version's adapters.
    measured = assert_no_training_drift(
        smoke_dir, STATE_JSONLS, dataset_version(smoke_dir))
    if measured:
        log.info("No training drift: %d state file(s) byte-identical to "
                 "pilot100_v1", len(measured))
    log.info("Wrote state datasets -> %s", smoke_dir / "training")


if __name__ == "__main__":
    main()
