"""Evaluate SALMU reference states on hierarchy-aware embedding probes.

    python scripts/evaluate_salmu_reference_states.py --device cuda:0

Loads BASE (released Clean CLIP) + MF/MG/MN checkpoints, encodes the
target-persona probes (fine / target / ancestor / sibling / generic),
applies the Iteration 10 separation gate, and writes
data/reports/salmu_reference_eval.json.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from granunlearn.config import _find_repo_root
from granunlearn.logging_utils import setup_logger
from granunlearn.salmu.adapter import REPOS, locate_repo
from granunlearn.salmu.clip_trainer import ClipRecipe
from granunlearn.salmu.embedding_metrics import (
    PROBE_KINDS,
    aggregate_scores,
    build_target_probes,
    reference_state_gate,
)

log = setup_logger("evaluate_salmu_reference_states")


class SalmuImageIndex:
    """file_name -> PIL image, from the released sensitive parquet."""

    def __init__(self, parquet_dir: str | Path):
        from datasets import concatenate_datasets, load_dataset
        shards = sorted(Path(parquet_dir).glob("sensitive-*.parquet"))
        self.ds = concatenate_datasets(
            [load_dataset("parquet", data_files=str(s), split="train")
             for s in shards])
        self.idx = {row["file_name"]: i
                    for i, row in enumerate(self.ds)}

    def image(self, file_name: str):
        return self.ds[self.idx[file_name]]["image"].convert("RGB")


def load_clip(state: str, repo_root: Path, device: str):
    import open_clip
    import torch
    recipe = ClipRecipe()
    if state == "BASE":
        clean = locate_repo(REPOS["clean_model"]["repo_id"], "model")
        model, _, preprocess = open_clip.create_model_and_transforms(
            recipe.arch,
            pretrained=str(clean / "open_clip_model.safetensors"))
    else:
        clean = locate_repo(REPOS["clean_model"]["repo_id"], "model")
        model, _, preprocess = open_clip.create_model_and_transforms(
            recipe.arch,
            pretrained=str(clean / "open_clip_model.safetensors"))
        ckpt = repo_root / "data" / "checkpoints" / "salmu" / state / \
            "pytorch_model.bin"
        if not ckpt.exists():
            raise FileNotFoundError(f"Missing SALMU checkpoint: {ckpt}")
        state_dict = torch.load(ckpt, map_location="cpu")
        model.load_state_dict(state_dict)
        log.info("[%s] loaded fine-tuned weights from %s", state, ckpt)
    model = model.to(device).eval()
    tokenizer = open_clip.get_tokenizer(recipe.arch)
    return model, preprocess, tokenizer


def score_state(state: str, probes: list[dict], image_index,
                repo_root: Path, device: str) -> dict:
    import torch
    model, preprocess, tokenizer = load_clip(state, repo_root, device)
    results = []
    with torch.no_grad():
        for probe in probes:
            image = preprocess(
                image_index.image(probe["image_file"])
            ).unsqueeze(0).to(device)
            img_feat = model.encode_image(image)
            img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
            sims = {}
            for kind in PROBE_KINDS:
                caption = probe[f"{kind}_caption"]
                if caption is None:
                    continue
                text = tokenizer([caption]).to(device)
                feat = model.encode_text(text)
                feat = feat / feat.norm(dim=-1, keepdim=True)
                sims[kind] = float((img_feat @ feat.t()).item())
            results.append({"identity_id": probe["identity_id"],
                            "attribute": probe["attribute"],
                            "sims": sims})
    del model
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    return aggregate_scores(results)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate SALMU reference states")
    parser.add_argument("--states", default="BASE,MF,MG,MN")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    repo_root = _find_repo_root(Path.cwd()) or Path.cwd()
    bench = locate_repo(REPOS["benchmark_dataset"]["repo_id"], "dataset")
    train_ds = locate_repo(REPOS["training_dataset"]["repo_id"], "dataset")
    hier_dir = repo_root / "data" / "salmu_hierarchical"

    identities = json.loads((bench / "identities_metadata.json").read_text())
    hierarchies = json.loads((hier_dir / "associations.json").read_text())
    manifest = json.loads(
        (hier_dir / "training" / "state_pairs_manifest.json").read_text())
    target_ids = manifest["partition"]["target_identity_ids"]

    cap_meta = json.loads(
        (train_ds / "sensitive_set_captions_metadata.json").read_text())
    fine_caps: dict = defaultdict(lambda: defaultdict(list))
    images_by: dict = defaultdict(lambda: defaultdict(list))
    for fname in sorted(cap_meta):
        meta = cap_meta[fname]
        if meta["data_field"] not in ("city", "job", "blood_type"):
            continue
        iid = fname.split("_")[0]
        fine_caps[iid][meta["data_field"]].append(meta["caption"])
        images_by[iid][meta["data_field"]].append(fname)

    probes = build_target_probes(
        target_ids, hierarchies, identities, fine_caps, images_by)
    log.info("Built %d target probes over %d personas", len(probes),
             len(target_ids))
    image_index = SalmuImageIndex(train_ds / "data")

    scores: dict = {}
    for state in [s.strip().upper() for s in args.states.split(",")]:
        scores[state] = score_state(state, probes, image_index,
                                    repo_root, args.device)
        s = scores[state]
        log.info("[%s] fine_pref=%s target_not_fine=%s | sims=%s",
                 state, s["prefers_fine_rate"],
                 s["prefers_target_not_fine_rate"],
                 s["mean_similarities"])

    passed, reasons = reference_state_gate(scores)
    report = {
        "experiment_id": "salmu_iter10_reference_states",
        "num_target_personas": len(target_ids),
        "probes_per_kind_note": "one probe per (target persona, core "
                                "attribute) with images",
        "scores_by_state": scores,
        "reference_state_gate": {
            "passed": passed,
            "reasons": reasons,
            "definition": {
                "MF": "prefers fine over {target, sibling}: >= 0.5 and "
                      ">= MG/MN + 0.15",
                "MG": "prefers target-not-fine >= 0.5, >= MN + 0.15, "
                      "and not below its own fine preference",
                "MN": "mean fine/target similarities within 0.05 of "
                      "BASE (preference order is noise at chance level)",
            },
        },
    }
    out = repo_root / "data" / "reports" / "salmu_reference_eval.json"
    with open(out, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    log.info("SALMU reference-state gate: %s%s",
             "PASSED" if passed else "FAILED",
             f" ({'; '.join(reasons)})" if reasons else "")


if __name__ == "__main__":
    main()
