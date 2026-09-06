"""Freeze a manifest of every referenced image's BYTES (Iteration 11R1).

    python scripts/build_image_manifest.py --tag pilot100

``associations.parquet`` records where each photograph lives, and the
prediction sidecars hash that parquet — so before this script existed the
contract pinned the POINTER and not the photograph.  Swapping one species'
image for a different one left all thirty sidecars verifying cleanly while
the model answered about a different picture, and nothing in the evidence
could tell.

This writes ``<data_dir>/image_manifest.json`` holding a sha256 per
referenced image plus a single ``manifest_sha256`` roll-up.
:func:`dataset_fingerprint` binds that roll-up, so a rebuilt manifest
invalidates every sidecar; :func:`verify_image_manifest` re-hashes the
images against the frozen file, so a swapped photograph is caught too.

Re-freezing is refused unless ``--allow-refreeze`` is passed, because
silently re-pinning the images would make existing evidence describe bytes
that are no longer on disk.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from granunlearn.config import _find_repo_root
from granunlearn.evaluation.prediction_provenance import (
    build_image_manifest,
    dataset_version,
    image_manifest_path,
    referenced_image_paths,
    verify_image_manifest,
)
from granunlearn.logging_utils import setup_logger

log = setup_logger("build_image_manifest")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Freeze sha256s for every image the dataset references")
    parser.add_argument("--tag", default="pilot100",
                        help="Dataset tag: pilot100 -> data/mllmu_hier_"
                             "pilot100, smoke -> data/mllmu_hier_smoke")
    parser.add_argument("--data-dir", default=None,
                        help="Override the dataset directory")
    parser.add_argument("--allow-refreeze", action="store_true",
                        help="Overwrite an existing frozen manifest. Off by "
                             "default: re-pinning image bytes changes what "
                             "every existing prediction sidecar describes, "
                             "so it must be a deliberate act.")
    parser.add_argument("--check-only", action="store_true",
                        help="Verify the frozen manifest against the images "
                             "on disk and change nothing")
    args = parser.parse_args()

    repo_root = _find_repo_root(Path.cwd()) or Path.cwd()
    data_dir = Path(args.data_dir or f"data/mllmu_hier_{args.tag}")
    if not (data_dir / "associations.parquet").exists():
        raise SystemExit(f"{data_dir}/associations.parquet does not exist; "
                         f"build the dataset first")
    manifest_path = image_manifest_path(data_dir)

    if args.check_only:
        if not manifest_path.exists():
            raise SystemExit(f"no frozen manifest at {manifest_path}")
        problems = verify_image_manifest(data_dir, repo_root)
        frozen = json.loads(manifest_path.read_text())
        print(f"{manifest_path}: {frozen['num_images']} images pinned, "
              f"rollup {frozen['manifest_sha256']}")
        if problems:
            print(f"FAILED — {len(problems)} image(s) disagree with the "
                  f"frozen manifest:")
            for p in problems:
                print(f"  {p}")
            raise SystemExit(1)
        print("OK — every referenced image matches its frozen hash")
        return

    refs = referenced_image_paths(data_dir)
    if manifest_path.exists() and not args.allow_refreeze:
        existing = json.loads(manifest_path.read_text())
        raise SystemExit(
            f"{manifest_path} is already frozen (dataset "
            f"{existing.get('dataset_version')}, "
            f"{existing.get('num_images')} images, rollup "
            f"{existing.get('manifest_sha256')}).\n"
            f"Re-freezing would change what every existing prediction "
            f"sidecar describes. Re-run with --allow-refreeze if that is "
            f"really intended, or --check-only to verify the images on disk "
            f"against it.\n"
            f"The dataset currently references {len(refs)} image path(s).")

    log.info("hashing %d referenced images under %s (version %s)",
             len(refs), data_dir, dataset_version(data_dir))
    written = build_image_manifest(data_dir, repo_root)
    if written["unresolved_paths"]:
        # Fail closed: a manifest that silently omits images would let an
        # unhashed photograph pass every downstream check.
        print(f"FAILED — {len(written['unresolved_paths'])} referenced "
              f"image(s) are not on disk and cannot be pinned:")
        for rel in written["unresolved_paths"][:20]:
            print(f"  {rel}")
        raise SystemExit(1)

    manifest_path.write_text(
        json.dumps(written, indent=2, sort_keys=True))
    print(f"froze {manifest_path}")
    print(f"  dataset_version    : {written['dataset_version']}")
    print(f"  num_images         : {written['num_images']}")
    print(f"  manifest_sha256    : {written['manifest_sha256']}")
    print(f"  unresolved_paths   : {len(written['unresolved_paths'])}")

    problems = verify_image_manifest(data_dir, repo_root)
    if problems:
        print(f"FAILED self-check — {len(problems)} problem(s):")
        for p in problems[:20]:
            print(f"  {p}")
        raise SystemExit(1)
    print("  self-check         : OK (re-hashed every image, all match)")


if __name__ == "__main__":
    main()
