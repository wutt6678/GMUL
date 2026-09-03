"""Compact committed provenance record for a tagged MLLMU experiment.

One small JSON binding the experiment to its exact inputs:

* dataset artifact hashes (parquets, manifest, training + unlearning
  jsonls, F/R partition report);
* for pilot-100: the iNaturalist fetch provenance hash and its
  resolution gate (the taxonomic stratum's photos are gitignored, so the
  gate record + per-photo SHA-256s ARE the reproducibility contract);
* base-model revision (HF cache ref);
* the training recipe (verbatim);
* environment versions + hardware;
* adapter/checkpoint hashes per reference state and per unlearning
  candidate.

    python scripts/write_provenance.py --tag pilot100
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path

from granunlearn.config import _find_repo_root
from granunlearn.logging_utils import setup_logger
from granunlearn.training.candidate_grid import grid_for_tag
from granunlearn.training.reference_trainer import ReferenceRecipe

log = setup_logger("write_provenance")

EXPERIMENT_IDS = {"smoke": "mllmu_smoke_iter7",
                  "pilot100": "mllmu_pilot100_iter11"}
DATASET_NOTES = {
    "smoke": "mllmu_hier_smoke (10 entities, 68 associations)",
    "pilot100": ("mllmu_hier_pilot100 (100 entities = 64 MLLMU persons + "
                 "36 real iNaturalist species; 477 associations; balanced "
                 "30/30/30 semantic/numeric/taxonomic targets)"),
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def base_model_revision(model_id: str) -> dict:
    """Resolve the local HF cache revision for the base model.

    The pinned ``revision`` is the reproducibility contract; the local
    cache path is diagnostic environment metadata only and is
    deliberately separated from it.
    """
    info = {"model_id": model_id, "revision": None,
            "diagnostics": {"local_cache_path": None}}
    cache_root = Path.home() / ".cache" / "huggingface" / "hub"
    model_dir = cache_root / ("models--" + model_id.replace("/", "--"))
    ref = model_dir / "refs" / "main"
    if ref.exists():
        info["revision"] = ref.read_text().strip()
        info["diagnostics"]["local_cache_path"] = str(model_dir)
    return info


def dataset_hashes(repo_root: Path, tag: str) -> dict[str, str]:
    """Hash every committed/derived dataset artifact of the tag."""
    data_dir = repo_root / "data" / f"mllmu_hier_{tag}"
    reports = repo_root / "data" / "reports"
    files = [
        data_dir / "associations.parquet",
        data_dir / "queries.parquet",
        data_dir / "manifest.json",
        reports / f"mllmu_{tag}_target_retain.json",
        reports / f"mllmu_{tag}_query_report.json",
        data_dir / "training" / "state_datasets_manifest.json",
        data_dir / "unlearning" / "unlearning_groups_manifest.json",
    ]
    files += [data_dir / "training" / f"{s}.jsonl"
              for s in ("MF", "MG", "MN")]
    files += [data_dir / "unlearning" / f"{g}.jsonl"
              for g in ("fine_target", "target_level", "retain")]
    if tag == "pilot100":
        files.append(repo_root / "configs" / "datasets" / "pilot100.yaml")
    out = {}
    for p in files:
        if p.exists():
            out[str(p.relative_to(repo_root))] = sha256_file(p)
        else:
            log.warning("missing dataset artifact: %s", p)
    return out


def inaturalist_provenance(repo_root: Path) -> dict:
    """The taxonomic stratum's fetch contract (photos stay gitignored)."""
    root = repo_root / "data" / "raw" / "inaturalist" / "pilot_v1"
    prov_path = root / "PROVENANCE.json"
    ann_path = root / "annotations.json"
    if not prov_path.exists():
        return {"available": False}
    prov = json.loads(prov_path.read_text())
    photos = prov.get("photos", [])
    widths = [p.get("width") for p in photos if p.get("width")]
    return {
        "available": True,
        "provenance_file": str(prov_path.relative_to(repo_root)),
        "provenance_sha256": sha256_file(prov_path),
        "annotations_sha256": sha256_file(ann_path)
        if ann_path.exists() else None,
        "num_photos": len(photos),
        "resolution_gate": prov.get("resolution_gate"),
        "num_rejected_candidates": sum(
            len(r.get("rejected", []))
            for r in prov.get("rejected_candidates", [])),
        "min_longest_edge_px": min(widths) if widths else None,
        "max_longest_edge_px": max(widths) if widths else None,
        "photo_sha256_recorded": all(p.get("sha256") for p in photos),
        "licenses": sorted({p.get("license_code") for p in photos}),
        "note": ("photo bytes are gitignored; PROVENANCE.json pins every "
                 "observation/photo id, license, attribution, source URL, "
                 "SHA-256 and measured resolution, so the frozen image set "
                 "is exactly re-fetchable via scripts/fetch_inat_species.py"),
    }


def checkpoint_hashes(root: Path, ids: list[str]) -> dict:
    out = {}
    for cid in ids:
        d = root / cid
        entry: dict = {}
        adapter = d / "adapters" / "adapter_model.safetensors"
        if adapter.exists():
            entry["adapter_sha256"] = sha256_file(adapter)
        summary = d / "training_summary.json"
        if summary.exists():
            entry["training_summary_sha256"] = sha256_file(summary)
            ts = json.loads(summary.read_text())
            for k in ("num_optimizer_steps", "final_loss", "train_seconds",
                      "device", "noop", "init_adapter_dir"):
                if k in ts:
                    entry[k] = ts[k]
            if ts.get("epochs"):
                entry["epochs"] = ts["epochs"]
        if entry:
            out[cid] = entry
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Write provenance record")
    parser.add_argument("--tag", default="smoke",
                        choices=("smoke", "pilot100"))
    parser.add_argument("--model-id", default="Qwen/Qwen3.5-9B")
    args = parser.parse_args()

    repo_root = _find_repo_root(Path.cwd()) or Path.cwd()
    reports = repo_root / "data" / "reports"
    ckpt = repo_root / "data" / "checkpoints" / f"mllmu_{args.tag}"
    unlearn_ckpt = repo_root / "data" / "checkpoints" / \
        f"mllmu_{args.tag}_unlearn"

    states = checkpoint_hashes(ckpt, ["BASE", "MF", "MG", "MN"])
    grid = grid_for_tag(args.tag)
    candidate_ids = [c.candidate_id for c in grid]
    selected_ids = [f"selected/{m}" for m in ("B0", "B1", "B2", "B2R", "B3")
                    if (unlearn_ckpt / "selected" / m).exists()]
    candidates = checkpoint_hashes(unlearn_ckpt, candidate_ids + selected_ids)

    selection_path = reports / f"mllmu_{args.tag}_unlearning_selection.json"
    selection = json.loads(selection_path.read_text()) \
        if selection_path.exists() else None

    # The final evaluation is the iteration's headline artifact, so bind it
    # here too: without this the provenance record covers the dataset, the
    # checkpoints and the selection but not the report the claims are read
    # off, and a reader cannot tell which final-evaluation file belongs to
    # these exact weights.
    final_path = reports / f"mllmu_{args.tag}_final_evaluation.json"
    final = json.loads(final_path.read_text()) if final_path.exists() else None
    final_one_shot = (final or {}).get("one_shot") or {}
    final_sens = (final or {}).get("batch_composition_sensitivity") or {}

    import torch
    import transformers
    import peft
    gpu_name = None
    try:
        gpu_name = torch.cuda.get_device_name(0)
    except Exception:
        pass

    provenance = {
        "experiment_id": EXPERIMENT_IDS[args.tag],
        "iteration": 11 if args.tag == "pilot100" else 7,
        "tag": args.tag,
        "dataset": DATASET_NOTES[args.tag],
        "dataset_hashes_sha256": dataset_hashes(repo_root, args.tag),
        "base_model": base_model_revision(args.model_id),
        "recipe": ReferenceRecipe().to_dict(),
        "reference_state_checkpoints": states,
        "candidate_grid": [c.describe() for c in grid],
        "unlearning_checkpoints": candidates,
        "selection": {
            "report": str(selection_path.relative_to(repo_root))
            if selection else None,
            "selected": (selection or {}).get("selected"),
            "basis": (selection or {}).get("basis"),
            "selection_scope": (selection or {}).get("selection_scope"),
        },
        "final_evaluation": {
            "report": str(final_path.relative_to(repo_root))
            if final else None,
            "report_sha256":
                hashlib.sha256(final_path.read_bytes()).hexdigest()
                if final else None,
            "selection_scope_used": final_one_shot.get("selection_scope"),
            "num_test_queries": final_one_shot.get("num_test_queries"),
            "assembled_without_generation":
                final_one_shot.get("assembled_without_generation"),
            "b0_equals_mf_passed":
                ((final or {}).get("b0_equals_mf_invariant") or {}).get(
                    "passed"),
            "batch_layout_noise_floor": {
                "max_abs_metric_delta":
                    final_sens.get("max_abs_metric_delta"),
                "max_abs_retain_delta": final_sens.get("max_abs_retain_delta"),
            },
        },
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
            "Every unlearning candidate continues from the SAME MF adapter "
            "and may override only swept knobs (learning_rate, num_epochs, "
            "group weights).",
            "Evaluation metrics are reported pooled AND per paraphrase "
            "split; adversarial probes are excluded from core slices.",
            "Checkpoint selection used TRAIN+VAL probes only; the frozen "
            "test split was generated once, afterwards, by "
            "scripts/evaluate_pilot100_final.py.",
            "Reproducibility contract = dataset hashes + base-model "
            "revision + recipe + adapter hashes. Fields under "
            "'diagnostics' and 'environment' are machine-specific "
            "metadata, not part of the contract.",
            "Training jsonl image paths are repo-relative; resolved at "
            "load time.",
        ],
    }
    if args.tag == "pilot100":
        provenance["inaturalist_stratum"] = inaturalist_provenance(repo_root)
        provenance["notes"].append(
            "max_image_pixels is enforced through "
            "granunlearn.imaging.image_size_kwargs: Qwen3VLProcessor "
            "silently ignores a bare max_pixels kwarg, which had left "
            "every image at native resolution (1024 vision tokens for a "
            "1024x1024 portrait instead of the intended 144).")

    out = reports / f"mllmu_{args.tag}_reference_provenance.json"
    with open(out, "w") as f:
        json.dump(provenance, f, indent=2, ensure_ascii=False)
    log.info("Wrote provenance -> %s (%d dataset artifacts, %d states, "
             "%d candidates)", out, len(provenance["dataset_hashes_sha256"]),
             len(states), len(candidates))


if __name__ == "__main__":
    main()
