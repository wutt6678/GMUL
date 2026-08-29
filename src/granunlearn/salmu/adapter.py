"""Read-only adapter over the RELEASED SALMUBench artifacts.

The original benchmark is never modified: this module only LOCATES the
Hugging Face snapshots and records their provenance (repo ids, revision
hashes, file sha256 for metadata) in ``data/salmu_original/manifest.json``.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from granunlearn.logging_utils import setup_logger

log = setup_logger("salmu_adapter")

REPOS: dict[str, dict[str, str]] = {
    "benchmark_dataset": {
        "repo_id": "cvc-mmu/salmubench-512-redistributed",
        "repo_type": "dataset",
        "role": "official evaluation splits: forget / forget_target / "
                "retain_synth / holdout_identity / holdout_association",
    },
    "training_dataset": {
        "repo_id": "cvc-mmu/salmu-512-redistributed",
        "repo_type": "dataset",
        "role": "knowledge-injection dataset used to train the "
                "Compromised model",
    },
    "clean_model": {
        "repo_id": "cvc-mmu/clip-vit-b-16-salmu-clean",
        "repo_type": "model",
        "role": "CLIP ViT-B/16 trained WITHOUT the sensitive "
                "associations — the shared starting checkpoint for "
                "MF/MG/MN reference states",
    },
    "compromised_model": {
        "repo_id": "cvc-mmu/clip-vit-b-16-salmu-compromised",
        "repo_type": "model",
        "role": "CLIP ViT-B/16 trained WITH the sensitive associations "
                "(SALMUBench's unlearning start point; reference only "
                "for our counterfactual design)",
    },
}

# Metadata files whose integrity we pin with sha256 (small; the parquet
# image shards are pinned by HF revision + dataset size).
METADATA_FILES = (
    "identities_metadata.json",
    "sensitive_set_captions_metadata.json",
    "sensitive_set_generic_captions.json",
    "fragile_set_ids.json",
    "fragile_set_all_similarities.json",
)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def locate_repo(repo_id: str, repo_type: str) -> Path:
    """Local HF snapshot directory for a repo (must already be
    downloaded)."""
    from huggingface_hub import snapshot_download
    return Path(snapshot_download(repo_id=repo_id, repo_type=repo_type,
                                  local_files_only=True))


def locate_repo_pinned(repo_id: str, repo_type: str,
                       revision: str) -> Path:
    """Local HF snapshot at a PINNED revision.

    Unlike ``locate_repo``, this ensures the consumed snapshot matches
    the claimed revision (no drift between remote and local).
    """
    from huggingface_hub import snapshot_download
    return Path(snapshot_download(repo_id=repo_id, repo_type=repo_type,
                                  revision=revision,
                                  local_files_only=True))


def load_original_metadata() -> dict[str, Any]:
    """Load the benchmark's original metadata (read-only)."""
    bench = locate_repo(REPOS["benchmark_dataset"]["repo_id"], "dataset")
    return {
        "identities": json.loads(
            (bench / "identities_metadata.json").read_text()),
        "captions": json.loads(
            (bench / "sensitive_set_captions_metadata.json").read_text()),
        "generic_captions": json.loads(
            (bench / "sensitive_set_generic_captions.json").read_text()),
        "benchmark_root": str(bench),
    }


def write_original_manifest(out_path: str | Path) -> dict:
    """Pin the released artifacts: repo ids + revisions + metadata
    hashes.  Never rewrites anything inside the HF snapshots.

    Paths are stored RELATIVE to the repository root so that the
    manifest is portable across machines.
    """
    from huggingface_hub import HfApi
    from granunlearn.config import _find_repo_root
    api = HfApi()
    repo_root = _find_repo_root(Path(out_path).parent) or Path(out_path).parent
    manifest: dict[str, Any] = {"artifacts": {}}
    for key, info in REPOS.items():
        try:
            rev = api.repo_info(info["repo_id"],
                                repo_type=info["repo_type"]).sha
        except Exception as exc:  # offline fallback
            rev = None
            log.warning("Could not fetch revision for %s: %s",
                        info["repo_id"], exc)
        entry: dict[str, Any] = {
            "repo_id": info["repo_id"],
            "repo_type": info["repo_type"],
            "revision": rev,
            "role": info["role"],
        }
        try:
            # Use pinned revision when available to ensure the
            # consumed snapshot matches the claimed revision.
            if rev is not None:
                local = locate_repo_pinned(
                    info["repo_id"], info["repo_type"], rev)
            else:
                local = locate_repo(
                    info["repo_id"], info["repo_type"])
            # Local cache path is diagnostic only; the reproducibility
            # contract is repo_id + revision.  Store as optional field.
            entry["local_cache_path"] = str(local)
            if info["repo_type"] == "dataset":
                entry["metadata_sha256"] = {
                    f: _sha256(local / f) for f in METADATA_FILES
                    if (local / f).exists()}
        except Exception as exc:
            entry["local_cache_path"] = None
            log.warning("Snapshot not available for %s: %s",
                        info["repo_id"], exc)
        manifest["artifacts"][key] = entry
    manifest["note"] = (
        "Read-only adapter: released SALMUBench artifacts are never "
        "modified; all derived data lives under salmu_hierarchical/ or "
        "salmu_aux_redaction/. "
        "Reproducibility contract: repo_id + revision (pinned). "
        "local_cache_path is diagnostic (machine-specific HF cache location).")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    log.info("Wrote salmu_original manifest -> %s", out_path)
    return manifest
