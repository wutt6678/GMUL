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
from granunlearn.evaluation.reference_eval import load_associations_parquet
from granunlearn.logging_utils import setup_logger
from granunlearn.training.unlearning_datasets import write_unlearning_groups

log = setup_logger("build_unlearning_groups")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build unlearning knowledge groups")
    parser.add_argument("--smoke-dir", default="data/mllmu_hier_smoke")
    args = parser.parse_args()

    repo_root = _find_repo_root(Path.cwd()) or Path.cwd()
    smoke = Path(args.smoke_dir)
    smoke = smoke if smoke.is_absolute() else repo_root / smoke

    associations = load_associations_parquet(smoke / "associations.parquet")
    partition = json.loads(
        (repo_root / "data" / "reports" /
         "mllmu_smoke_target_retain.json").read_text())

    manifest = write_unlearning_groups(
        associations, partition, smoke / "unlearning")
    for group, info in manifest["groups"].items():
        log.info("%s: %d examples -> %s", group, info["num_examples"],
                 info["path"])
    log.info("Wrote unlearning groups -> %s", smoke / "unlearning")


if __name__ == "__main__":
    main()
