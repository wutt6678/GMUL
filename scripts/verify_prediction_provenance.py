"""Verify every prediction sidecar and freeze a consolidated manifest (11R1).

    python scripts/verify_prediction_provenance.py --tag pilot100
    python scripts/verify_prediction_provenance.py --tag pilot100 --check-only

The thirty pilot-100 prediction parquets and their sidecars are gitignored,
so "all sidecars verify" was a claim about a working tree nobody else could
see: nothing in the commit let a reader check it, or even know which bytes
were supposed to be there.

This writes ``data/reports/mllmu_<tag>_prediction_manifest.json``, which IS
committed, holding for every prediction file its parquet sha256, its sidecar
sha256, its row count, its adapter contract, the image-manifest roll-up and
the ten module hashes — plus one ``manifest_sha256`` roll-up over all of
them.  The parquets themselves stay untracked (they are large and
regenerable), but their identities are now pinned in git, so:

* ``--check-only`` re-derives the whole structure from disk and compares it
  to the committed manifest, which is what the test suite runs;
* anyone holding the parquets can confirm they are the ones this evidence
  was computed from;
* anyone without them knows exactly what to expect, and a substituted file
  fails loudly instead of quietly.

Verification rebuilds each expected fingerprint from the artifacts on disk
today — the adapter directory, the dataset, the frozen image manifest, the
current module bytes — rather than trusting what the sidecar says about
itself.  Any refusal is fatal.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from granunlearn.config import _find_repo_root
from granunlearn.evaluation.prediction_provenance import (
    CODE_FINGERPRINT_MODULES,
    PROVENANCE_CONTRACT_VERSION,
    PredictionFingerprint,
    adapter_sha256,
    base_model_revision,
    dataset_fingerprint,
    dataset_version,
    git_commit,
    image_manifest_sha256,
    parquet_num_rows,
    resolve_adapter_dir,
    sha256_file,
    sidecar_path,
    verify_image_manifest,
    verify_sidecar,
)
from granunlearn.logging_utils import setup_logger

log = setup_logger("verify_prediction_provenance")

#: Batch layout implied by the file name.  The three layouts carry different
#: query sets and must never be compared as if interchangeable, so the
#: manifest records which one each file is.
LAYOUT_BY_PREFIX = (
    ("predictions_test_", "test_only"),
    ("predictions_tv_", "train_val"),
    ("predictions_", "all_split"),
)


def _layout(name: str) -> str:
    for prefix, layout in LAYOUT_BY_PREFIX:
        if name.startswith(prefix):
            return layout
    return "unknown"


def build_consolidated_manifest(
    repo_root: Path, tag: str, pred_dir: Path, data_dir: Path,
    model_id: str,
) -> tuple[dict[str, Any], list[str]]:
    """Re-derive the manifest from disk and report every refusal."""
    problems: list[str] = []
    ds_fp = dataset_fingerprint(data_dir, repo_root)
    img_problems = verify_image_manifest(data_dir, repo_root)
    if img_problems:
        problems.extend(f"image manifest: {p}" for p in img_problems)

    records: list[dict[str, Any]] = []
    for parquet in sorted(pred_dir.glob("predictions*.parquet")):
        sc_path = sidecar_path(parquet)
        if not sc_path.exists():
            problems.append(f"{parquet.name}: no provenance sidecar")
            continue
        found = json.loads(sc_path.read_text())
        cid = found.get("checkpoint_id")
        ad = resolve_adapter_dir(cid, repo_root, tag)
        if ad is not None and not ad.exists():
            problems.append(f"{parquet.name}: adapter dir {ad} is missing")
        elif ad is not None:
            actual = adapter_sha256(ad)
            if actual != found.get("adapter_sha256"):
                problems.append(
                    f"{parquet.name}: {ad} holds weights {actual}, the "
                    f"sidecar records {found.get('adapter_sha256')}")
        expected = PredictionFingerprint(
            experiment_id=found.get("experiment_id"),
            checkpoint_id=cid,
            adapter_sha256=found.get("adapter_sha256"),
            base_model_revision=base_model_revision(model_id),
            dataset=ds_fp,
            generation_config=found.get("generation_config") or {},
            code={
                "git_commit": git_commit(repo_root),
                "git_dirty": None,
                "modules_sha256": {
                    rel: sha256_file(repo_root / rel)
                    if (repo_root / rel).exists() else None
                    for rel in CODE_FINGERPRINT_MODULES},
            },
            adapter_contract=found.get("adapter_contract"),
        )
        reasons = verify_sidecar(parquet, expected)
        for r in reasons:
            problems.append(f"{parquet.name}: {r}")
        rebind = found.get("rebind") or {}
        records.append({
            "parquet": parquet.name,
            "layout": _layout(parquet.name),
            "parquet_sha256": sha256_file(parquet),
            "sidecar_sha256": sha256_file(sc_path),
            "num_rows": parquet_num_rows(parquet),
            "experiment_id": found.get("experiment_id"),
            "checkpoint_id": cid,
            "adapter_sha256": found.get("adapter_sha256"),
            "adapter_contract_sha256": (
                (found.get("adapter_contract") or {}).get("sha256")),
            "base_model_revision": found.get("base_model_revision"),
            "generation_config": found.get("generation_config"),
            "dataset_version": (found.get("dataset") or {}).get("version"),
            "image_manifest_sha256": (
                (found.get("dataset") or {}).get("image_manifest_sha256")),
            "code_git_commit": (found.get("code") or {}).get("git_commit"),
            "code_modules_sha256": (found.get("code") or {}).get(
                "modules_sha256"),
            "created_utc": found.get("created_utc"),
            "contract_version": found.get("contract_version"),
            "rebound_utc": rebind.get("rebound_utc"),
            "rebind_assertion": rebind.get("assertion"),
            "verify_reasons": reasons,
        })

    canon = json.dumps(
        [{k: v for k, v in r.items() if k != "verify_reasons"}
         for r in records], sort_keys=True, separators=(",", ":"))
    rollup = hashlib.sha256(canon.encode("utf-8")).hexdigest()
    manifest = {
        "tag": tag,
        "dataset_version": dataset_version(data_dir),
        "generated_utc": datetime.now(timezone.utc).isoformat(
            timespec="seconds"),
        "provenance_contract_version": PROVENANCE_CONTRACT_VERSION,
        "repo_commit": git_commit(repo_root),
        "image_manifest_sha256": image_manifest_sha256(data_dir),
        "num_images_pinned": ds_fp.get("num_images_pinned"),
        "code_modules_sha256": {
            rel: sha256_file(repo_root / rel)
            if (repo_root / rel).exists() else None
            for rel in CODE_FINGERPRINT_MODULES},
        "num_predictions": len(records),
        "num_refusals": sum(len(r["verify_reasons"]) for r in records)
        + len([p for p in problems if p.startswith("image manifest")]),
        "manifest_sha256": rollup,
        "predictions": records,
        "note": (
            "The prediction parquets and sidecars themselves are gitignored "
            "(large and regenerable). This manifest is committed so their "
            "identities are pinned in git: --check-only re-derives it from "
            "disk and compares, which is what makes 'all sidecars verify' a "
            "reproducible claim rather than an assertion about one machine's "
            "working tree. Records whose verify_reasons is non-empty were "
            "REFUSED."),
    }
    return manifest, problems


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify prediction sidecars and freeze/compare a "
                    "consolidated manifest")
    parser.add_argument("--tag", default="pilot100")
    parser.add_argument("--model-id", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--predictions-dir", default=None)
    parser.add_argument("--output", default=None,
                        help="Manifest path (default: data/reports/mllmu_"
                             "<tag>_prediction_manifest.json)")
    parser.add_argument("--check-only", action="store_true",
                        help="Compare disk against the committed manifest "
                             "and write nothing")
    args = parser.parse_args()

    repo_root = _find_repo_root(Path.cwd()) or Path.cwd()
    data_dir = repo_root / "data" / f"mllmu_hier_{args.tag}"
    pred_dir = Path(args.predictions_dir or data_dir / "predictions")
    out_path = Path(args.output or
                    repo_root / "data" / "reports"
                    / f"mllmu_{args.tag}_prediction_manifest.json")
    if not pred_dir.exists():
        print(f"no predictions directory at {pred_dir}")
        return 1

    manifest, problems = build_consolidated_manifest(
        repo_root, args.tag, pred_dir, data_dir, args.model_id)

    if args.check_only:
        if not out_path.exists():
            print(f"FAILED: no committed manifest at {out_path}")
            return 1
        committed = json.loads(out_path.read_text())
        drift = []
        if committed.get("manifest_sha256") != manifest["manifest_sha256"]:
            drift.append(
                f"manifest_sha256: committed "
                f"{committed.get('manifest_sha256')}, on disk "
                f"{manifest['manifest_sha256']}")
        by_name = {r["parquet"]: r for r in committed.get("predictions", [])}
        for rec in manifest["predictions"]:
            old = by_name.get(rec["parquet"])
            if old is None:
                drift.append(f"{rec['parquet']}: not in the committed "
                             f"manifest")
                continue
            for key in ("parquet_sha256", "sidecar_sha256", "num_rows",
                        "adapter_contract_sha256",
                        "image_manifest_sha256"):
                if old.get(key) != rec[key]:
                    drift.append(f"{rec['parquet']}.{key}: committed "
                                 f"{old.get(key)}, on disk {rec[key]}")
        for name in sorted(set(by_name) -
                           {r["parquet"] for r in manifest["predictions"]}):
            drift.append(f"{name}: in the committed manifest but missing "
                         f"on disk")
        print(f"checked {manifest['num_predictions']} predictions against "
              f"{out_path.name}")
        if drift:
            print(f"FAILED — {len(drift)} difference(s):")
            for d in drift[:20]:
                print(f"  {d}")
            return 1
        if problems:
            print(f"FAILED — {len(problems)} verification refusal(s):")
            for p in problems[:20]:
                print(f"  {p}")
            return 1
        print("OK — every prediction matches the committed manifest and "
              "every sidecar verifies")
        return 0

    if problems:
        print(f"FAILED — {len(problems)} verification refusal(s):")
        for p in problems:
            print(f"  {p}")
        print("\nno manifest written: a consolidated manifest that pins "
              "refused evidence would launder the refusal into the commit")
        return 1

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    print(f"verified {manifest['num_predictions']} predictions, "
          f"0 refusals")
    print(f"  dataset_version    : {manifest['dataset_version']}")
    print(f"  contract version   : {manifest['provenance_contract_version']}")
    print(f"  image_manifest     : {manifest['image_manifest_sha256']}")
    print(f"  num_images_pinned  : {manifest['num_images_pinned']}")
    print(f"  manifest_sha256    : {manifest['manifest_sha256']}")
    print(f"wrote {out_path}")
    layouts: dict[str, int] = {}
    for r in manifest["predictions"]:
        layouts[r["layout"]] = layouts.get(r["layout"], 0) + 1
    print(f"  layouts: {layouts}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
