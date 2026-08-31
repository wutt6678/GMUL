"""Shared CLIP evaluation utilities for SALMU reference + MU states.

Both the reference-state evaluator and the MU checkpoint selector score
the SAME target-persona probes; this module centralizes checkpoint
loading, image access, and per-probe similarity computation so the two
entry points can never drift apart.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from granunlearn.logging_utils import setup_logger
from granunlearn.salmu.adapter import REPOS, locate_repo
from granunlearn.salmu.clip_trainer import ClipRecipe
from granunlearn.salmu.embedding_metrics import PROBE_KINDS

log = setup_logger("salmu_eval_utils")


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


def load_clip(state: str, repo_root: Path, device: str,
              unlearn_root: Path | None = None):
    """Load BASE (Clean), COMPROMISED, MF/MG/MN reference states, or MU.

    MU candidates live under data/checkpoints/salmu_unlearn/{state}.
    COMPROMISED is the released salmu-compromised CLIP checkpoint.
    """
    import open_clip
    import torch
    recipe = ClipRecipe()

    if state == "COMPROMISED":
        comp = locate_repo(REPOS["compromised_model"]["repo_id"], "model")
        ckpt_path = comp / "open_clip_pytorch_model.bin"
        model, _, preprocess = open_clip.create_model_and_transforms(
            recipe.arch, pretrained=str(ckpt_path))
        log.info("[COMPROMISED] loaded released checkpoint from %s",
                 ckpt_path)
    else:
        clean = locate_repo(REPOS["clean_model"]["repo_id"], "model")
        model, _, preprocess = open_clip.create_model_and_transforms(
            recipe.arch,
            pretrained=str(clean / "open_clip_model.safetensors"))
        if state != "BASE":
            ckpt = repo_root / "data" / "checkpoints" / "salmu" / state / \
                "pytorch_model.bin"
            if not ckpt.exists() and unlearn_root is not None:
                ckpt = unlearn_root / state / "pytorch_model.bin"
            if not ckpt.exists():
                raise FileNotFoundError(f"Missing SALMU checkpoint: {ckpt}")
            model.load_state_dict(torch.load(ckpt, map_location="cpu"))
            log.info("[%s] loaded weights from %s", state, ckpt)

    model = model.to(device).eval()
    tokenizer = open_clip.get_tokenizer(recipe.arch)
    return model, preprocess, tokenizer


def score_probes(state: str, probes: list[dict[str, Any]],
                 image_index: SalmuImageIndex, repo_root: Path,
                 device: str,
                 unlearn_root: Path | None = None) -> list[dict[str, Any]]:
    """Per-probe similarities for one checkpoint.

    Returns one dict per probe, propagating all probe metadata
    (``identity_id``, ``attribute``, ``is_target_attr``) alongside the
    per-kind similarities so that downstream aggregation can split on
    target vs retain without losing the distinction.
    """
    import torch
    model, preprocess, tokenizer = load_clip(
        state, repo_root, device, unlearn_root)
    results: list[dict[str, Any]] = []
    with torch.no_grad():
        for probe in probes:
            image = preprocess(
                image_index.image(probe["image_file"])
            ).unsqueeze(0).to(device)
            img_feat = model.encode_image(image)
            img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
            sims: dict[str, float] = {}
            for kind in PROBE_KINDS:
                caption = probe[f"{kind}_caption"]
                if caption is None:
                    continue
                text = tokenizer([caption]).to(device)
                feat = model.encode_text(text)
                feat = feat / feat.norm(dim=-1, keepdim=True)
                sims[kind] = float((img_feat @ feat.t()).item())
            result: dict[str, Any] = {
                "identity_id": probe["identity_id"],
                "attribute": probe["attribute"],
                "sims": sims,
            }
            # Propagate frozen probe ID and indices for
            # reproducible image/caption-level analysis.
            for meta_key in ("probe_id", "image_idx", "caption_idx",
                            "image_file", "fine_caption"):
                if meta_key in probe:
                    result[meta_key] = probe[meta_key]
            # Propagate target/retain tagging so aggregation can
            # split on is_target_attr without losing the distinction.
            if "is_target_attr" in probe:
                result["is_target_attr"] = probe["is_target_attr"]
            results.append(result)
    del model
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    return results


def aggregate_probe_results(results: list[dict[str, Any]],
                            personas: set[str] | None = None,
                            target_attr_only: bool = False,
                            bootstrap_ci: bool = False,
                            n_bootstrap: int = 1000,
                            ) -> dict:
    """Aggregate per-probe results, optionally restricted to a set of
    persona ids (used for train/val/test persona splits).

    When ``target_attr_only=True``, only probes with
    ``is_target_attr=True`` are included (used for the selection
    summary vector so that same-entity retain probes do not dilute
    the MG-distance signal).

    Aggregation is association-weighted: image-caption variants of the
    same (identity_id, attribute) are macro-averaged first (10R4).
    With ``bootstrap_ci=True`` adds association-bootstrap CIs.
    """
    from granunlearn.salmu.embedding_metrics import aggregate_scores
    if personas is not None:
        results = [r for r in results if r["identity_id"] in personas]
    if target_attr_only:
        results = [r for r in results
                   if r.get("is_target_attr", False)]
    return aggregate_scores(results, bootstrap_ci=bootstrap_ci,
                            n_bootstrap=n_bootstrap)


def save_probe_cache(path: str | Path, per_state: dict[str, list]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(per_state, f)
    log.info("Cached per-probe sims -> %s", path)


def load_probe_cache(path: str | Path) -> dict[str, list] | None:
    path = Path(path)
    if not path.exists():
        return None
    return json.loads(path.read_text())


def build_release_probes(repo_root: Path,
                         core_attrs: tuple = ("city", "job", "blood_type"),
                         max_images: int | None = None,
                         max_captions: int | None = None,
                         ) -> tuple[list[dict[str, Any]], list[str]]:
    """Build target-persona probes from the released artifacts.

    Deterministic and identical for the reference-state evaluator and
    the MU selector: probes come from the released sensitive caption
    metadata (fine captions + images) and our built hierarchies; the
    target partition and per-attribute assignment come from the
    committed pair-set manifest.

    With per-attribute targeting, probes cover BOTH target attributes
    (is_target_attr=True) and same-entity retain attributes
    (is_target_attr=False) of the target personas.

    Each probe gets a frozen ``probe_id`` for reproducibility.
    With ``max_images`` / ``max_captions``, the number of image and
    caption variants per (persona, attribute) is capped.

    Returns (probes, target_ids).
    """
    from collections import defaultdict

    from granunlearn.salmu.embedding_metrics import build_target_probes

    bench = locate_repo(REPOS["benchmark_dataset"]["repo_id"], "dataset")
    train_ds = locate_repo(REPOS["training_dataset"]["repo_id"], "dataset")
    hier_dir = repo_root / "data" / "salmu_hierarchical"

    identities = json.loads(
        (bench / "identities_metadata.json").read_text())
    hierarchies = json.loads((hier_dir / "associations.json").read_text())
    manifest = json.loads(
        (hier_dir / "training" / "state_pairs_manifest.json").read_text())
    target_ids = manifest["partition"]["target_identity_ids"]
    target_attr_map = manifest.get("target_attr_map")

    cap_meta = json.loads(
        (train_ds / "sensitive_set_captions_metadata.json").read_text())
    fine_caps: dict = defaultdict(lambda: defaultdict(list))
    images_by: dict = defaultdict(lambda: defaultdict(list))
    for fname in sorted(cap_meta):
        meta = cap_meta[fname]
        if meta["data_field"] not in core_attrs:
            continue
        iid = fname.split("_")[0]
        fine_caps[iid][meta["data_field"]].append(meta["caption"])
        images_by[iid][meta["data_field"]].append(fname)

    probes = build_target_probes(
        target_ids, hierarchies, identities, fine_caps, images_by,
        target_attr_map=target_attr_map,
        max_images=max_images, max_captions=max_captions)
    return probes, target_ids


def build_retain_probes(repo_root: Path, max_personas: int = 100,
                        core_attrs: tuple = ("city", "job", "blood_type"),
                        max_images: int | None = None,
                        max_captions: int | None = None,
                        ) -> list[dict[str, Any]]:
    """Collateral-damage probes over RETAIN personas.

    One probe per (retain persona, attribute, image, fine caption):
    each probe gets a frozen ``probe_id`` so results are reproducible.
    Retain personas are trained identically (fine captions) in
    MF/MG/MN and in every MU candidate's retain group, so their
    similarity is the SALMU analog of Retain_same/Retain_other.
    Deterministic, capped for cost.

    Retain personas are ALL personas NOT in the target partition
    (i.e. those carrying the ``other_entity_retain`` role in MF).
    """
    import hashlib
    from collections import defaultdict

    from granunlearn.salmu.embedding_metrics import GENERIC_CAPTION

    train_ds = locate_repo(REPOS["training_dataset"]["repo_id"], "dataset")
    hier_dir = repo_root / "data" / "salmu_hierarchical"
    manifest = json.loads(
        (hier_dir / "training" / "state_pairs_manifest.json").read_text())
    target_ids = set(manifest["partition"]["target_identity_ids"])

    # Retain personas = all personas NOT in the target partition.
    # Use the MF pairs to discover all identity ids.
    all_ids = sorted(set(
        p["identity_id"] for p in load_mf_pairs(hier_dir)))
    retain_ids = sorted(iid for iid in all_ids if iid not in target_ids)
    retain_ids = retain_ids[:max_personas]

    if not retain_ids:
        raise RuntimeError(
            "build_retain_probes: no retain personas found. "
            "Check that the manifest partition is correct.")

    cap_meta = json.loads(
        (train_ds / "sensitive_set_captions_metadata.json").read_text())
    fine_caps: dict = defaultdict(lambda: defaultdict(list))
    images_by: dict = defaultdict(lambda: defaultdict(list))
    retain_set = set(retain_ids)
    for fname in sorted(cap_meta):
        meta = cap_meta[fname]
        if meta["data_field"] not in core_attrs:
            continue
        iid = fname.split("_")[0]
        if iid not in retain_set:
            continue
        fine_caps[iid][meta["data_field"]].append(meta["caption"])
        images_by[iid][meta["data_field"]].append(fname)

    probes: list[dict[str, Any]] = []
    for iid in retain_ids:
        for attr in sorted(core_attrs):
            images = sorted(images_by.get(iid, {}).get(attr, []))
            fines = sorted(fine_caps.get(iid, {}).get(attr, []))
            if not images or not fines:
                continue
            if max_images is not None:
                images = images[:max_images]
            if max_captions is not None:
                fines = fines[:max_captions]
            for img_idx, img_file in enumerate(images):
                for cap_idx, fine_cap in enumerate(fines):
                    pid = hashlib.sha256(
                        f"{iid}|{attr}|{img_file}|{fine_cap}".encode()
                    ).hexdigest()[:16]
                    probes.append({
                        "probe_id": pid,
                        "identity_id": iid,
                        "attribute": attr,
                        "is_target_attr": False,
                        "image_file": img_file,
                        "image_idx": img_idx,
                        "fine_caption": fine_cap,
                        "caption_idx": cap_idx,
                        "target_caption": None,
                        "ancestor_caption": None,
                        "ancestor_is_target": None,
                        "sibling_caption": None,
                        "generic_caption": GENERIC_CAPTION,
                    })
    if not probes:
        raise RuntimeError(
            "build_retain_probes: zero retain probes constructed. "
            "Check caption/image availability for retain personas.")
    return probes


def load_mf_pairs(hier_dir: Path) -> list[dict]:
    """MF released pairs as plain dicts (role/identity/attr fields)."""
    pairs = []
    with open(hier_dir / "training" / "MF.jsonl") as f:
        for line in f:
            pairs.append(json.loads(line))
    return pairs

