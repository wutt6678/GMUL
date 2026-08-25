"""Compact committed provenance record for the Iteration 7 reference
states (Iteration 7 review).

One small JSON binding the experiment to its exact inputs:

* dataset artifact hashes (smoke parquet, training jsonl, partition)
* base-model revision (HF cache ref)
* the training recipe (verbatim)
* environment versions + hardware
* adapter/checkpoint hashes per state

    python scripts/write_provenance.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path

from granunlearn.config import _find_repo_root
from granunlearn.logging_utils import setup_logger
from granunlearn.training.reference_trainer import ReferenceRecipe

log = setup_logger("write_provenance")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def base_model_revision(model_id: str) -> dict:
    """Resolve the local HF cache revision for the base model."""
    info = {"model_id": model_id, "revision": None, "cache_path": None}
    cache_root = Path.home() / ".cache" / "huggingface" / "hub"
    model_dir = cache_root / ("models--" + model_id.replace("/", "--"))
    ref = model_dir / "refs" / "main"
    if ref.exists():
        info["revision"] = ref.read_text().strip()
        info["cache_path"] = str(model_dir)
    return info


def main() -> None:
    parser = argparse.ArgumentParser(description="Write provenance record")
    parser.add_argument("--model-id", default="Qwen/Qwen3.5-9B")
    args = parser.parse_args()

    repo_root = _find_repo_root(Path.cwd()) or Path.cwd()
    smoke = repo_root / "data" / "mllmu_hier_smoke"
    reports = repo_root / "data" / "reports"
    ckpt = repo_root / "data" / "checkpoints" / "mllmu_smoke"

    dataset_files = [
        smoke / "associations.parquet",
        smoke / "queries.parquet",
        smoke / "manifest.json",
        smoke / "training" / "MF.jsonl",
        smoke / "training" / "MG.jsonl",
        smoke / "training" / "MN.jsonl",
        smoke / "training" / "state_datasets_manifest.json",
        reports / "mllmu_smoke_target_retain.json",
    ]
    dataset_hashes = {
        str(p.relative_to(repo_root)): sha256_file(p)
        for p in dataset_files if p.exists()
    }

    checkpoints = {}
    for state in ("MF", "MG", "MN"):
        state_dir = ckpt / state
        entry: dict = {}
        adapter = state_dir / "adapters" / "adapter_model.safetensors"
        if adapter.exists():
            entry["adapter_sha256"] = sha256_file(adapter)
        summary = state_dir / "training_summary.json"
        if summary.exists():
            entry["training_summary_sha256"] = sha256_file(summary)
            ts = json.loads(summary.read_text())
            entry["num_optimizer_steps"] = ts.get("num_optimizer_steps")
            entry["final_loss"] = ts.get("final_loss")
        checkpoints[state] = entry

    import torch
    import transformers
    import peft
    gpu_name = None
    try:
        gpu_name = torch.cuda.get_device_name(0)
    except Exception:
        pass

    provenance = {
        "experiment_id": "mllmu_smoke_iter7",
        "dataset": "mllmu_hier_smoke (10 entities, 68 associations)",
        "dataset_hashes_sha256": dataset_hashes,
        "base_model": base_model_revision(args.model_id),
        "recipe": ReferenceRecipe().to_dict(),
        "checkpoints": checkpoints,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "peft": peft.__version__,
            "cuda_runtime": torch.version.cuda,
            "gpu": gpu_name,
            "platform": platform.platform(),
        },
        "notes": [
            "MF/MG/MN share the recipe verbatim; only D_F/D_G/D_N differ.",
            "Optimizer steps differ only via dataset size (68/68/48 "
            "examples; accum-normalized trailing groups).",
            "Evaluation metrics are reported pooled AND per paraphrase "
            "split; adversarial probes excluded from core slices.",
        ],
    }

    out = reports / "mllmu_smoke_reference_provenance.json"
    with open(out, "w") as f:
        json.dump(provenance, f, indent=2, ensure_ascii=False)
    log.info("Wrote provenance -> %s", out)


if __name__ == "__main__":
    main()
