"""Build the Iteration 9 unlearning knowledge groups.

    python scripts/build_unlearning_groups.py

Writes fine_target / target_level / retain jsonls (same controlled
template as the reference-state datasets, repo-relative image paths)
plus a manifest, derived ONLY from the committed association pool and
the F/R partition.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from granunlearn.config import _find_repo_root
from granunlearn.evaluation.image_splits import assert_no_training_drift
from granunlearn.evaluation.prediction_provenance import dataset_version
from granunlearn.evaluation.reference_eval import load_associations_parquet
from granunlearn.logging_utils import setup_logger
from granunlearn.training.unlearning_datasets import write_unlearning_groups

log = setup_logger("build_unlearning_groups")

#: The three group files every unlearning candidate is fitted on.
UNLEARNING_GROUP_JSONLS = ("unlearning/fine_target.jsonl",
                           "unlearning/target_level.jsonl",
                           "unlearning/retain.jsonl")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build unlearning knowledge groups")
    parser.add_argument("--tag", default="smoke")
    parser.add_argument("--smoke-dir", default=None,
                        help="Override the dataset directory "
                             "(default: data/mllmu_hier_<tag>)")
    args = parser.parse_args()

    repo_root = _find_repo_root(Path.cwd()) or Path.cwd()
    smoke = Path(args.smoke_dir or f"data/mllmu_hier_{args.tag}")
    smoke = smoke if smoke.is_absolute() else repo_root / smoke

    associations = load_associations_parquet(smoke / "associations.parquet")
    partition = json.loads(
        (repo_root / "data" / "reports" /
         f"mllmu_{args.tag}_target_retain.json").read_text())

    manifest = write_unlearning_groups(
        associations, partition, smoke / "unlearning")
    for group, info in manifest["groups"].items():
        log.info("%s: %d examples -> %s", group, info["num_examples"],
                 info["path"])
    # Iteration 11R freeze gate: the repaired visual split may relabel
    # evaluation photographs, but it must not move one byte of the data the
    # unlearning candidates were fitted on.  If it does, the correct
    # response is to retrain, not to keep reporting stale adapters.
    measured = assert_no_training_drift(
        smoke, UNLEARNING_GROUP_JSONLS, dataset_version(smoke))
    if measured:
        log.info("No training drift: %d group file(s) byte-identical to "
                 "pilot100_v1", len(measured))
    log.info("Wrote unlearning groups -> %s", smoke / "unlearning")


if __name__ == "__main__":
    main()
