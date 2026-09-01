"""Write SALMU provenance record (Iteration 10R4).

    python scripts/write_salmu_provenance.py

Produces data/reports/salmu_reference_provenance.json with:
* dataset pair counts (from state_pairs_manifest.json)
* target attribute distribution
* PINNED HF repos: repo_id + consumed snapshot revision (the
  reproducibility contract), with the local cache path recorded as a
  DIAGNOSTIC machine-specific field only
* complete-file SHA-256 checkpoint hashes
* environment versions

Path semantics: artifacts inside the project repository are recorded
repo-relative; released HF snapshots are pinned by repo_id + revision
(their absolute cache location varies per machine and carries no
provenance weight).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from collections import Counter
from pathlib import Path

from granunlearn.config import _find_repo_root
from granunlearn.logging_utils import setup_logger
from granunlearn.salmu.adapter import REPOS, locate_repo
from granunlearn.salmu.paths import SalmuPaths

log = setup_logger("write_salmu_provenance")


def sha256_file(path: Path) -> str:
    """Complete-file SHA-256 (reads in 1 MiB chunks)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _repo_relative(path: Path, repo_root: Path) -> str:
    """Return a repo-relative path string for portability."""
    try:
        return str(path.relative_to(repo_root))
    except ValueError:
        return f"<external> {path}"


def _snapshot_revision(snapshot_path: Path) -> str | None:
    """Extract the consumed HF snapshot revision from the cache path
    (``.../snapshots/<revision>``)."""
    parts = Path(snapshot_path).parts
    if "snapshots" in parts:
        idx = parts.index("snapshots")
        if idx + 1 < len(parts):
            return parts[idx + 1]
    return None


def _pin_repo(key: str, repo_id: str, repo_type: str) -> dict:
    """Resolve a repo snapshot and pin its revision.

    The snapshot is located, its revision extracted, and then
    re-located THROUGH the pinned revision so the consumed snapshot
    provably matches the claimed one.
    """
    entry: dict = {"repo_id": repo_id, "repo_type": repo_type,
                   "role": REPOS[key].get("role", "")}
    try:
        local = locate_repo(repo_id, repo_type)
        revision = _snapshot_revision(local)
        entry["revision"] = revision
        if revision is not None:
            # Prove the pinned revision resolves to the same snapshot
            from granunlearn.salmu.adapter import locate_repo_pinned
            pinned = locate_repo_pinned(repo_id, repo_type, revision)
            entry["revision_verified"] = (
                pinned.resolve() == Path(local).resolve())
            local = pinned
        else:
            entry["revision_verified"] = False
        entry["local_cache_path"] = str(local)
        entry["local_cache_path_note"] = (
            "diagnostic only (machine-specific HF cache location); "
            "the reproducibility contract is repo_id + revision")
    except Exception as exc:
        entry["revision"] = None
        entry["revision_verified"] = False
        entry["local_cache_path"] = None
        entry["error"] = str(exc)
    return entry


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Write SALMU provenance record")
    parser.add_argument("--suffix", default="",
                        help="Iteration tag (e.g. r5 -> holdout-clean "
                             "provenance under *_r5 paths)")
    args = parser.parse_args()

    repo_root = _find_repo_root(Path.cwd()) or Path.cwd()
    paths = SalmuPaths(repo_root, suffix=args.suffix)
    ckpt_root = paths.ref_ckpt_root
    unlearn_root = paths.unlearn_root

    # Dataset pair counts from manifest
    manifest = json.loads(paths.manifest_path.read_text())
    pair_counts = {}
    for state, info in manifest["states"].items():
        pair_counts[state] = {
            "num_pairs": info["num_pairs"],
            "num_target_associations": info["num_target_associations"],
            "num_same_entity_retain": info["num_same_entity_retain"],
            "num_other_entity_retain": info["num_other_entity_retain"],
            "num_identities": info["num_identities"],
        }

    # Target attribute distribution
    tam = manifest.get("target_attr_map", {})
    attr_dist = dict(Counter(tam.values()))

    # Pinned HF repos (repo_id + consumed snapshot revision)
    hf_repos = {key: _pin_repo(key, info["repo_id"],
                               info.get("repo_type", "model"))
                for key, info in REPOS.items()}

    # Complete-file checkpoint hashes
    checkpoints = {}
    for state in ("MF", "MG", "MN"):
        ckpt = ckpt_root / state / "pytorch_model.bin"
        if ckpt.exists():
            checkpoints[state] = sha256_file(ckpt)
    # Clean base model hash (the starting point, NOT MF)
    clean_local = hf_repos.get("clean_model", {}).get("local_cache_path")
    base_model_hash = ""
    if clean_local:
        clean_ckpt = Path(clean_local) / "open_clip_model.safetensors"
        if clean_ckpt.exists():
            base_model_hash = sha256_file(clean_ckpt)
    # Selected unlearning checkpoints
    selected_dir = unlearn_root / "selected"
    if selected_dir.exists():
        for method_dir in sorted(selected_dir.iterdir()):
            if method_dir.is_dir():
                ckpt = method_dir / "pytorch_model.bin"
                if ckpt.exists():
                    key = f"selected_{method_dir.name}"
                    checkpoints[key] = sha256_file(ckpt)

    # Environment
    import torch
    import open_clip
    gpu_name = None
    try:
        gpu_name = torch.cuda.get_device_name(0)
    except Exception:
        pass

    from granunlearn.salmu.unlearning import split_target_personas
    persona_split = split_target_personas(
        manifest["partition"]["target_identity_ids"])
    split_note = ("Probe-persona split: "
                  f"{len(persona_split['train'])} train / "
                  f"{len(persona_split['val'])} val / "
                  f"{len(persona_split['test'])} test "
                  "(sha256, seed 42).")
    if args.suffix:
        holdout_note = (
            f"10R5 holdout-clean protocol (suffix={args.suffix}): "
            "the pair universe is restricted to the official forget "
            "split and target personas keep only designations with "
            ">=1 forget target pair, so NO holdout_identity / "
            "holdout_association pair enters any training group. "
            "The official `retain` split carries no sensitive "
            "associations (generic utility captions only), so forget "
            "is the only permitted sensitive training data. "
            "Released-split results for the retrained states are "
            "therefore untouched external evaluation; COMPROMISED "
            "alone is the benchmark authors' checkpoint trained on "
            "forget + holdouts (in-sample reference).")
    else:
        holdout_note = (
            "10R4b protocol caveat: the released sensitive TRAINING "
            "dataset is the union of the forget and holdout splits, "
            "so the current MF/MG/MN pair sets and ALL unlearning "
            "groups consume released holdout pairs (see "
            "holdout_consumption in salmu_official_splits.json). "
            "Released-split results are transfer diagnostics, not "
            "untouched external evaluation; Iteration 10R5 retrains "
            "holdout-clean.")

    provenance = {
        "experiment_id": ("salmu_iter10r5" if args.suffix == "r5"
                          else "salmu_iter10r4"),
        "iteration_suffix": args.suffix or None,
        "dataset": "salmu-512-redistributed (sensitive split, core "
                   "attrs city/job/blood_type)",
        "protocol": manifest.get("protocol"),
        "dataset_partition": {
            "seed": manifest["partition"]["seed"],
            "num_targets": manifest["partition"]["num_targets"],
            "num_retain": manifest["partition"]["num_retain"],
            "target_identity_ids": manifest["partition"][
                "target_identity_ids"],
        },
        "target_attr_distribution": attr_dist,
        "target_attr_assignment": f"round-robin (balanced {attr_dist})",
        "dataset_pair_counts": pair_counts,
        "hf_repos": hf_repos,
        "base_model": {
            "arch": "ViT-B-16",
            "checkpoint": "open_clip_model.safetensors",
            "sha256": base_model_hash,
            "params_M": 149.6,
            "repo": REPOS["clean_model"]["repo_id"],
            "revision": hf_repos.get("clean_model", {}).get("revision"),
            "note": "Clean CLIP — the shared starting checkpoint for "
                    "MF/MG/MN (NOT the MF fine-tuned checkpoint).",
        },
        "recipe": {
            "arch": "ViT-B-16",
            "learning_rate": 1e-5,
            "weight_decay": 0.0,
            "num_epochs": 3,
            "batch_size": 256,
            "seed": 42,
            "bf16": True,
            "logit_scale_frozen": True,
        },
        "checkpoints_sha256": checkpoints,
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "open_clip": open_clip.__version__,
            "gpu": gpu_name,
        },
        "path_semantics": "Repo artifacts use repo-relative paths "
                          "(see dataset/checkpoint keys). Released HF "
                          "snapshots are pinned by repo_id + revision "
                          "in hf_repos; their local_cache_path is "
                          "diagnostic only and machine-specific.",
        "notes": [
            "MF/MG/MN fine-tuned from SAME Clean CLIP with IDENTICAL "
            "recipe; only D_F/D_G/D_N differ.",
            "Per-attribute targeting: each target persona has ONE "
            "target attr (city/job/blood_type) via round-robin "
            "assignment.",
            "MG: target personas get ONE generalized target caption "
            "per target attribute.",
            "MN: target attr omitted; same-entity retain attrs kept "
            "with fine captions.",
            "Unlearning candidates continue from MF with constrained "
            "ascent (v2).",
            split_note,
            "Selection on TARGET-ONLY train+val probes, "
            "association-weighted (each (identity, attribute) counts "
            "once); internal test identities evaluated for the "
            "SELECTED checkpoints only since 10R4 — but the split "
            "was inspected candidate-wide in 10R2/10R3, so those "
            "test numbers are exploratory.",
            holdout_note,
            "B2_retain_* excluded from B2 candidate family.",
            "Sibling probes use correct hierarchy level: same-sector "
            "different-profession-class for job; different country "
            "for city; different ABO group for blood_type.",
            "Gate runs on target-attribute probes only; same-entity "
            "and other-entity retention reported separately.",
            "Checkpoint hashes are complete-file SHA-256.",
        ],
    }

    out = paths.report("salmu_reference_provenance")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(provenance, f, indent=2, ensure_ascii=False)
    log.info("Wrote SALMU provenance -> %s", out)


if __name__ == "__main__":
    main()
