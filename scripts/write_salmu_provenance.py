"""Write SALMU provenance record (Iteration 10R2).

    python scripts/write_salmu_provenance.py

Produces data/reports/salmu_reference_provenance.json with:
* dataset pair counts (from state_pairs_manifest.json)
* target attribute distribution
* pinned HF repo IDs + local snapshot paths
* checkpoint hashes
* environment versions
"""

from __future__ import annotations

import hashlib
import json
import platform
from collections import Counter
from pathlib import Path

from granunlearn.config import _find_repo_root
from granunlearn.logging_utils import setup_logger
from granunlearn.salmu.adapter import REPOS, locate_repo

log = setup_logger("write_salmu_provenance")


def sha256_file_prefix(path: Path, n_bytes: int = 16 * 1024 * 1024) -> str:
    """SHA-256 prefix (first 16 hex chars of first n_bytes)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read(n_bytes))
    return h.hexdigest()[:16]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    repo_root = _find_repo_root(Path.cwd()) or Path.cwd()
    hier_dir = repo_root / "data" / "salmu_hierarchical"
    reports = repo_root / "data" / "reports"
    ckpt_root = repo_root / "data" / "checkpoints" / "salmu"
    unlearn_root = repo_root / "data" / "checkpoints" / "salmu_unlearn"

    # Dataset pair counts from manifest
    manifest = json.loads(
        (hier_dir / "training" / "state_pairs_manifest.json").read_text())
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

    # Pinned HF repo info with actual local snapshot paths
    hf_repos = {}
    for key, info in REPOS.items():
        repo_id = info["repo_id"]
        repo_type = info.get("repo_type", "model")
        try:
            local_path = locate_repo(repo_id, repo_type)
            snapshot = str(local_path)
        except Exception:
            snapshot = "not available"
        hf_repos[key] = {"repo_id": repo_id,
                         "repo_type": repo_type,
                         "local_path": snapshot,
                         "role": info.get("role", "")}

    # Checkpoint hashes
    checkpoints = {}
    for state in ("MF", "MG", "MN"):
        ckpt = ckpt_root / state / "pytorch_model.bin"
        if ckpt.exists():
            checkpoints[state] = sha256_file_prefix(ckpt)
    # Selected unlearning checkpoints
    selected_dir = unlearn_root / "selected"
    if selected_dir.exists():
        for method_dir in sorted(selected_dir.iterdir()):
            if method_dir.is_dir():
                ckpt = method_dir / "pytorch_model.bin"
                if ckpt.exists():
                    key = f"selected_{method_dir.name}"
                    checkpoints[key] = sha256_file_prefix(ckpt)

    # Environment
    import torch
    import open_clip
    gpu_name = None
    try:
        gpu_name = torch.cuda.get_device_name(0)
    except Exception:
        pass

    provenance = {
        "experiment_id": "salmu_iter10r2",
        "dataset": "salmu-512-redistributed (sensitive split, core "
                   "attrs city/job/blood_type)",
        "dataset_partition": {
            "seed": manifest["partition"]["seed"],
            "num_targets": manifest["partition"]["num_targets"],
            "num_retain": manifest["partition"]["num_retain"],
            "target_identity_ids": manifest["partition"][
                "target_identity_ids"],
        },
        "target_attr_distribution": attr_dist,
        "target_attr_assignment": "round-robin (balanced 20/20/20)",
        "dataset_pair_counts": pair_counts,
        "hf_repos": hf_repos,
        "base_model": {
            "arch": "ViT-B-16",
            "checkpoint": "open_clip_model.safetensors",
            "sha256_prefix16": checkpoints.get("MF", ""),
            "params_M": 149.6,
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
        "checkpoints_sha256_prefix16": checkpoints,
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "open_clip": open_clip.__version__,
            "gpu": gpu_name,
        },
        "notes": [
            "MF/MG/MN fine-tuned from SAME Clean CLIP with IDENTICAL "
            "recipe; only D_F/D_G/D_N differ.",
            "Per-attribute targeting: each target persona has ONE "
            "target attr (city/job/blood_type) via round-robin "
            "assignment (balanced 20/20/20).",
            "MG: target personas get ONE generalized target caption "
            "per target attribute.",
            "MN: target attr omitted; same-entity retain attrs kept "
            "with fine captions.",
            "Unlearning candidates continue from MF with constrained "
            "ascent (v2).",
            "Probe-persona split: 40 train / 10 val / 10 test "
            "(sha256, seed 42).",
            "Selection on train+val only; test frozen.",
            "Sibling probes use different-branch alternatives "
            "(different country/ABO group/sector).",
            "Gate runs on target-attribute probes only; same-entity "
            "and other-entity retention reported separately.",
            "Paths are repo-relative for portability.",
        ],
    }

    out = reports / "salmu_reference_provenance.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(provenance, f, indent=2, ensure_ascii=False)
    log.info("Wrote SALMU provenance -> %s", out)


if __name__ == "__main__":
    main()
