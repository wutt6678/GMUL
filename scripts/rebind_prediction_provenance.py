"""Re-stamp v1 prediction sidecars onto provenance contract v2 (11R1).

    python scripts/rebind_prediction_provenance.py --tag pilot100
    python scripts/rebind_prediction_provenance.py --tag pilot100 --dry-run

Contract v1 recorded the adapter's WEIGHTS, the dataset's tabular artifacts
and six scoring modules.  It recorded no hash of the prediction parquet
itself, nothing from ``adapter_config.json``, and no image bytes — so a
fabricated ``raw_output``, a flipped ``is_finer_than_target`` bit, an edited
LoRA rank and a swapped photograph all verified cleanly (all four measured
during 11R1).  v2 binds all of them, which leaves thirty existing sidecars
that cannot satisfy the new contract.

Regenerating them means regenerating the predictions: hours of GPU work to
re-derive bytes that are already on disk.  This script re-stamps the RECORD
instead, and is only sound because it refuses to run unless three
preconditions hold:

1. A frozen image manifest exists for the dataset AND every referenced image
   on disk still matches it — otherwise the newly bound image hash would
   describe bytes nobody has checked.
2. For every sidecar, all ten :data:`CODE_FINGERPRINT_MODULES` are
   byte-identical to the blobs at that sidecar's own recorded ``git_commit``.
   The six v1 modules were hashed at generation; the four ``schema`` modules
   are new to the contract, and this is what licenses recording today's hash
   as the value generation would have recorded.
3. Every sidecar's ``checkpoint_id`` resolves to an adapter directory whose
   weights hash equals the recorded ``adapter_sha256``.  Resolution is by
   the same naming convention the generating scripts used, not by searching
   for a matching hash: ``selected/B0`` and ``MF`` hold byte-identical
   weights, so a hash search is ambiguous for twelve of the thirty.

What rebinding CANNOT establish is stated in the sidecar itself: the new
``prediction_sha256`` asserts that the bytes on disk are the bytes the
recorded adapter produced.  That is an assumption about the past which only
re-running inference could verify.  Everything else is recomputed from
artifacts on disk.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from granunlearn.config import _find_repo_root
from granunlearn.evaluation.prediction_provenance import (
    CODE_FINGERPRINT_MODULES,
    PROVENANCE_CONTRACT_VERSION,
    adapter_contract,
    adapter_sha256,
    dataset_fingerprint,
    environment_fingerprint,
    image_manifest_path,
    parquet_num_rows,
    resolve_adapter_dir,
    sha256_file,
    verify_image_manifest,
)
from granunlearn.logging_utils import setup_logger

log = setup_logger("rebind_prediction_provenance")


def _blob_sha256_at(commit: str, rel: str, repo_root: Path) -> str | None:
    """sha256 of a path's bytes as committed at ``commit``."""
    out = subprocess.run(["git", "show", f"{commit}:{rel}"], cwd=repo_root,
                         capture_output=True)
    if out.returncode != 0:
        return None
    return hashlib.sha256(out.stdout).hexdigest()


def check_module_soundness(sidecars: list[dict[str, Any]],
                           repo_root: Path) -> list[str]:
    """Precondition 2, per sidecar.  Empty list == sound."""
    problems: list[str] = []
    current = {rel: sha256_file(repo_root / rel)
               if (repo_root / rel).exists() else None
               for rel in CODE_FINGERPRINT_MODULES}
    commits = sorted({s["code"].get("git_commit") for s in sidecars
                      if s["code"].get("git_commit")})
    for commit in commits:
        for rel in CODE_FINGERPRINT_MODULES:
            then = _blob_sha256_at(commit, rel, repo_root)
            if then is None:
                problems.append(f"{rel} does not exist at {commit[:8]}, so "
                                f"its hash at generation time is unknown")
            elif then != current[rel]:
                problems.append(
                    f"{rel} CHANGED between {commit[:8]} and the working "
                    f"tree, so today's hash is not the hash generation used")
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Re-stamp v1 prediction sidecars onto contract v2")
    parser.add_argument("--tag", default="pilot100")
    parser.add_argument("--predictions-dir", default=None,
                        help="Override the predictions directory")
    parser.add_argument("--dry-run", action="store_true",
                        help="Run every precondition and report what would "
                             "change, but write nothing")
    args = parser.parse_args()

    repo_root = _find_repo_root(Path.cwd()) or Path.cwd()
    data_dir = repo_root / "data" / f"mllmu_hier_{args.tag}"
    pred_dir = Path(args.predictions_dir or
                    data_dir / "predictions")
    sidecar_paths = sorted(pred_dir.glob("*.parquet.provenance.json"))
    if not sidecar_paths:
        print(f"no sidecars under {pred_dir}")
        return 1

    sidecars = []
    for sp in sidecar_paths:
        try:
            sidecars.append(json.loads(sp.read_text()))
        except json.JSONDecodeError as exc:
            print(f"FAILED: unreadable sidecar {sp.name}: {exc}")
            return 1
    print(f"{len(sidecars)} sidecars under {pred_dir}")

    # ---------- precondition 1: frozen, matching image manifest ----------
    manifest = image_manifest_path(data_dir)
    if not manifest.exists():
        print(f"FAILED precondition 1: no frozen image manifest at "
              f"{manifest}\n  run: python scripts/build_image_manifest.py "
              f"--tag {args.tag}")
        return 1
    img_problems = verify_image_manifest(data_dir, repo_root)
    if img_problems:
        print(f"FAILED precondition 1: {len(img_problems)} referenced "
              "image(s) disagree with the frozen manifest:")
        for p in img_problems[:10]:
            print(f"  {p}")
        return 1
    print("precondition 1 OK: image manifest frozen and every referenced "
          "image matches it")

    # ---------- precondition 2: module bytes unchanged since generation ----
    mod_problems = check_module_soundness(sidecars, repo_root)
    if mod_problems:
        print(f"FAILED precondition 2: {len(mod_problems)} fingerprinted "
              f"module(s) are not byte-identical to what generation used:")
        for p in mod_problems[:10]:
            print(f"  {p}")
        print("  These sidecars cannot be re-stamped; regenerate the "
              "predictions instead.")
        return 1
    print(f"precondition 2 OK: all {len(CODE_FINGERPRINT_MODULES)} "
          f"fingerprinted modules byte-identical at every recorded commit")

    # ---------- precondition 3: adapter resolution cross-check ----------
    adapters: dict[str, Path | None] = {}
    bad = []
    for sc in sidecars:
        cid = sc["checkpoint_id"]
        ad = resolve_adapter_dir(cid, repo_root, args.tag)
        if cid in adapters:
            continue
        if ad is None:
            if sc.get("adapter_sha256") is not None:
                bad.append(f"{cid}: BASE resolves to no adapter but the "
                           f"sidecar records weights")
            adapters[cid] = None
            continue
        if not ad.exists():
            bad.append(f"{cid}: {ad} does not exist")
            continue
        actual = adapter_sha256(ad)
        if actual != sc.get("adapter_sha256"):
            bad.append(f"{cid}: {ad} holds weights {actual}, the sidecar "
                       f"records {sc.get('adapter_sha256')}")
            continue
        adapters[cid] = ad
    if bad:
        print(f"FAILED precondition 3: {len(bad)} adapter(s) do not resolve:")
        for b in bad[:10]:
            print(f"  {b}")
        return 1
    print(f"precondition 3 OK: all {len(adapters)} distinct checkpoint_ids "
          f"resolve to an adapter whose weights match the recorded hash")

    # ---------- re-stamp ----------
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    env = environment_fingerprint()
    rebound = skipped = 0
    for sp, old in zip(sidecar_paths, sidecars):
        parquet = Path(str(sp)[: -len(".provenance.json")])
        ver = old.get("contract_version", 1)
        if ver == PROVENANCE_CONTRACT_VERSION:
            skipped += 1
            continue
        cid = old["checkpoint_id"]
        ad = adapters[cid]
        ds = dataset_fingerprint(data_dir, repo_root)
        new: dict[str, Any] = {
            "experiment_id": old["experiment_id"],
            "checkpoint_id": cid,
            "adapter_sha256": old.get("adapter_sha256"),
            "base_model_revision": old.get("base_model_revision"),
            "dataset": ds,
            "generation_config": old.get("generation_config") or {},
            # git_commit/git_dirty stay the GENERATION-TIME values: they
            # attribute the evidence to the tree that produced it.  The
            # module hashes are current bytes, which precondition 2 just
            # proved identical to that tree's.
            "code": {
                "git_commit": (old.get("code") or {}).get("git_commit"),
                "git_dirty": (old.get("code") or {}).get("git_dirty"),
                "modules_sha256": {
                    rel: sha256_file(repo_root / rel)
                    if (repo_root / rel).exists() else None
                    for rel in CODE_FINGERPRINT_MODULES},
            },
            "created_utc": old.get("created_utc", ""),
            "num_rows": parquet_num_rows(parquet),
            "adapter_contract": adapter_contract(ad),
            "prediction_sha256": sha256_file(parquet),
            "contract_version": PROVENANCE_CONTRACT_VERSION,
            "environment": env,
            "rebind": {
                "rebound_utc": now,
                "from_contract_version": ver,
                "to_contract_version": PROVENANCE_CONTRACT_VERSION,
                "original_sidecar_sha256": sha256_file(sp),
                "adapter_dir": str(ad.relative_to(repo_root)) if ad else None,
                "adapter_resolved_by": (
                    "checkpoint_id naming convention, cross-checked against "
                    "the recorded adapter_sha256"),
                "image_manifest_sha256": ds.get("image_manifest_sha256"),
                "preconditions_passed": [
                    "frozen image manifest matches every referenced image",
                    f"all {len(CODE_FINGERPRINT_MODULES)} fingerprinted "
                    f"modules byte-identical to their blobs at this "
                    f"sidecar's recorded git_commit",
                    "adapter directory weights match the recorded hash",
                ],
                "assertion": (
                    "RE-STAMPED, NOT REGENERATED. prediction_sha256 asserts "
                    "that the parquet bytes on disk at rebind time are the "
                    "bytes the recorded adapter and generation config "
                    "produced. Nothing here can prove that: only re-running "
                    "inference could. It is recorded so that any FUTURE edit "
                    "to the parquet is detected, and so the artifact is "
                    "pinned from this moment onward."),
            },
        }
        if args.dry_run:
            print(f"  would re-stamp {parquet.name}: "
                  f"prediction_sha256={new['prediction_sha256'][:12]} "
                  f"rows={new['num_rows']} "
                  f"contract={new['adapter_contract']['sha256'][:12] if new['adapter_contract'] else None}")
            rebound += 1
            continue
        sp.write_text(json.dumps(new, indent=2))
        rebound += 1

    verb = "would re-stamp" if args.dry_run else "re-stamped"
    print(f"\n{verb} {rebound}, already on v{PROVENANCE_CONTRACT_VERSION}: "
          f"{skipped}, total {len(sidecars)}")
    if not args.dry_run and rebound:
        print("next: python scripts/verify_prediction_provenance.py "
              f"--tag {args.tag}   (writes the committed consolidated "
              "manifest)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
