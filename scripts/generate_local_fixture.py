#!/usr/bin/env python3
"""Generate the local iNaturalist-format dataset used for the smoke build.

This writes a COCO-style dataset (annotations.json + real JPEG images)
under ``data/raw/inaturalist/local_v1/`` using the test taxonomy fixture.
When real iNaturalist data is synced, replace this directory's contents
with the actual data — the adapter reads the same format.

Usage
-----
    python scripts/generate_local_fixture.py [--images-per-species 12]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "tests"))

from fixtures.inat_fixture import write_local_dataset  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="data/raw/inaturalist/local_v1")
    parser.add_argument("--images-per-species", type=int, default=12)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    out = REPO_ROOT / args.output
    summary = write_local_dataset(
        out,
        images_per_species=args.images_per_species,
        seed=args.seed,
    )
    print(f"Wrote {summary['num_species']} species, {summary['num_images']} images → {out}")


if __name__ == "__main__":
    main()
