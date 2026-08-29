"""SALMU unlearning: groups, persona probe splits, and MU trainer.

Iteration 10 stage 4: port the MF -> MU baselines (B0-B3) to the CLIP
association level.  Everything continues from MF^SALMU; the B3 group
structure is FIXED from Iteration 9 (gd fine_target + sft target_level
+ sft retain) - only hyperparameters are tuned, and ONLY on the
train/val probe personas.  Test probe personas stay held out.

With per-attribute targeting, the groups are:
* fine_target   — target (persona, attr) pairs with fine captions (gd)
* target_level  — target (persona, attr) pairs with generalized
                  target captions (sft)
* retain        — ALL retain pairs: same-entity retain (non-target
                  attributes of target personas) + other-entity retain
                  (all attributes of non-target personas) (sft)
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from granunlearn.logging_utils import setup_logger
from granunlearn.salmu.clip_trainer import ClipRecipe
from granunlearn.salmu.hierarchy import generalized_caption
from granunlearn.salmu.state_datasets import SalmuTrainingPair, \
    load_state_pairs

log = setup_logger("salmu_unlearning")

GROUPS = ("fine_target", "target_level", "retain")
MODES = ("sft", "gd")


def split_target_personas(
    target_identity_ids: list[str],
    seed: int = 42,
    val_frac: float = 1 / 6,
    test_frac: float = 1 / 6,
) -> dict[str, list[str]]:
    """Deterministic sha256 ranking into train/val/test personas.

    Probe personas are the unit of selection: hyperparameters are tuned
    on train+val personas only; test personas are the final held-out
    evaluation, mirroring the MLLMU paraphrase-split protocol.
    """
    ranked = sorted(
        target_identity_ids,
        key=lambda i: hashlib.sha256(f"{i}|salmu_split|{seed}".encode())
        .hexdigest())
    n = len(ranked)
    n_test = int(n * test_frac)
    n_val = int(n * val_frac)
    n_train = n - n_val - n_test
    return {"train": ranked[:n_train],
            "val": ranked[n_train:n_train + n_val],
            "test": ranked[n_train + n_val:]}


def build_salmu_unlearning_groups(
    mf_pairs: list[SalmuTrainingPair],
    hierarchies: dict[str, dict[str, Any]],
    identities: dict[str, dict[str, Any]],
    out_dir: str | Path | None = None,
) -> dict[str, list[SalmuTrainingPair]]:
    """Build the three knowledge groups from MF's released pairs.

    With per-attribute targeting, the role field already distinguishes
    target pairs (the designated target attribute of target personas)
    from retain pairs (everything else, including same-entity retain).

    * fine_target   — target pairs' RELEASED fine captions (gd)
    * target_level  — target pairs' generalized target captions,
                      paired with the SAME images (sft)
    * retain        — ALL retain pairs' released fine captions (sft),
                      including same-entity retain of target personas
    None of these touch SALMUBench evaluation splits.
    """
    groups: dict[str, list[SalmuTrainingPair]] = {
        "fine_target": [], "target_level": [], "retain": []}
    for pair in mf_pairs:
        if pair.role == "target_association":
            groups["fine_target"].append(pair)
        else:  # same_entity_retain or other_entity_retain
            groups["retain"].append(pair)
    # target_level: one generalized caption per (identity, attribute),
    # paired with exactly the images that carried the fine captions
    seen: set[tuple[str, str]] = set()
    for pair in sorted(groups["fine_target"],
                       key=lambda p: (p.identity_id, p.attribute)):
        key = (pair.identity_id, pair.attribute)
        if key in seen:
            continue
        seen.add(key)
        iid, attr = key
        hier = hierarchies[iid][attr]
        lv = hier["target_level"]
        caption = generalized_caption(
            identities[iid]["name"], attr, lv, hier["levels"][lv])
        for fine_pair in [p for p in groups["fine_target"]
                          if p.identity_id == iid and p.attribute == attr]:
            groups["target_level"].append(SalmuTrainingPair(
                pair_id=f"TL__{iid}__{attr}__"
                        f"{Path(fine_pair.image_file).stem}",
                state="MU",
                identity_id=iid,
                attribute=attr,
                role="target",
                level_index=lv,
                caption=caption,
                caption_source="generalized_template",
                image_file=fine_pair.image_file,
            ))
    for name, pairs in groups.items():
        groups[name] = sorted(pairs, key=lambda p: p.pair_id)
        log.info("group %s: %d pairs", name, len(groups[name]))
    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        for name, pairs in groups.items():
            with open(out_dir / f"{name}.jsonl", "w") as f:
                for p in pairs:
                    f.write(p.model_dump_json() + "\n")
        log.info("Wrote SALMU unlearning groups -> %s", out_dir)
    return groups


@dataclass(frozen=True)
class SalmuGroupSpec:
    """One training group: pairs + objective mode + loss weight."""

    name: str
    pairs: list[SalmuTrainingPair]
    mode: str
    weight: float

    def __post_init__(self) -> None:
        if self.mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        if self.weight <= 0:
            raise ValueError("weight must be positive")


def train_salmu_unlearning(
    candidate_id: str,
    group_specs: list[SalmuGroupSpec],
    parquet_dir: str | Path,
    init_checkpoint: str | Path,
    output_dir: str | Path,
    device: str = "cuda:0",
    recipe: ClipRecipe | None = None,
    anchor_checkpoint: str | Path | None = None,
    gd_stop_sim: float | None = None,
    gd_probe_pairs: list[SalmuTrainingPair] | None = None,
) -> dict[str, Any]:
    """Continue from MF^SALMU with interleaved group batches.

    sft groups minimize InfoNCE; gd groups maximize it (gradient
    ascent).  One batch per group per round, cycling short groups.

    Constrained gradient ascent (v2): when ``anchor_checkpoint`` and
    ``gd_stop_sim`` are given, the ascent on the gd group is STOPPED
    whenever the probe pairs' mean image-text similarity has dropped to
    the anchor level (e.g. MG's fine similarity).  Without the
    constraint, unconstrained ascent collapses the whole embedding
    space (Iteration 10 v1 finding: sims -> -0.9 everywhere).
    """
    import open_clip
    import torch
    from granunlearn.salmu.clip_trainer import SalmuPairDataset, \
        clip_contrastive_loss

    recipe = recipe or ClipRecipe()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    import random
    import numpy as np
    random.seed(recipe.seed)
    np.random.seed(recipe.seed)
    torch.manual_seed(recipe.seed)
    torch.cuda.manual_seed_all(recipe.seed)

    model, _, preprocess = open_clip.create_model_and_transforms(
        recipe.arch, pretrained=str(init_checkpoint))
    tokenizer = open_clip.get_tokenizer(recipe.arch)
    model = model.to(device).to(torch.bfloat16)
    model.train()
    model.logit_scale.requires_grad_(False)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        params, lr=recipe.learning_rate,
        weight_decay=recipe.weight_decay)

    loaders, specs = [], []
    for spec in group_specs:
        dataset = SalmuPairDataset(spec.pairs, parquet_dir,
                                   preprocess, tokenizer)
        loaders.append(torch.utils.data.DataLoader(
            dataset, batch_size=recipe.batch_size, shuffle=True,
            num_workers=recipe.num_workers, pin_memory=True,
            drop_last=len(dataset) > recipe.batch_size))
        specs.append(spec)
    steps_per_epoch = max(len(l) for l in loaders)

    # Anchor-probe machinery for constrained ascent
    gd_active = {s.name: True for s in specs if s.mode == "gd"}
    probe_loader = None
    gd_stop_threshold = None
    if anchor_checkpoint is not None:
        import torch.nn.functional as F
        if not gd_probe_pairs:
            raise ValueError("gd_probe_pairs required with anchor")
        anchor_model, _, _ = open_clip.create_model_and_transforms(
            recipe.arch, pretrained=str(anchor_checkpoint))
        anchor_model = anchor_model.to(device).to(
            torch.bfloat16).eval()
        probe_dataset = SalmuPairDataset(gd_probe_pairs, parquet_dir,
                                         preprocess, tokenizer)
        probe_loader = torch.utils.data.DataLoader(
            probe_dataset, batch_size=recipe.batch_size, shuffle=False,
            num_workers=recipe.num_workers, pin_memory=True)
        with torch.no_grad():
            sims = []
            for images, texts in probe_loader:
                images = images.to(device).to(torch.bfloat16)
                texts = texts.to(device)
                img_f = F.normalize(anchor_model.encode_image(images),
                                    dim=-1)
                txt_f = F.normalize(anchor_model.encode_text(texts),
                                    dim=-1)
                sims.append((img_f * txt_f).sum(dim=-1))
            anchor_sim = float(torch.cat(sims).mean().item())
        del anchor_model
        # Default stop level: the anchor state's own similarity on the
        # same pairs (i.e. forget DOWN TO the reference, not further).
        gd_stop_threshold = gd_stop_sim if gd_stop_sim is not None \
            else anchor_sim
        log.info("[%s] anchor fine-similarity %.4f | stop threshold "
                 "%.4f", candidate_id, anchor_sim, gd_stop_threshold)

    def _probe_mean_sim() -> float:
        import torch.nn.functional as F
        with torch.no_grad():
            sims = []
            for images, texts in probe_loader:
                images = images.to(device).to(torch.bfloat16)
                texts = texts.to(device)
                img_f = F.normalize(model.encode_image(images), dim=-1)
                txt_f = F.normalize(model.encode_text(texts), dim=-1)
                sims.append((img_f * txt_f).sum(dim=-1))
        return float(torch.cat(sims).mean().item())

    summary: dict[str, Any] = {
        "candidate_id": candidate_id,
        "init_checkpoint": str(init_checkpoint),
        "recipe": recipe.to_dict(),
        "groups": [{"name": s.name, "mode": s.mode, "weight": s.weight,
                    "num_pairs": len(s.pairs)} for s in specs],
        "steps_per_epoch": steps_per_epoch,
        "epochs": [],
    }
    global_step = 0
    import time
    t0 = time.time()
    for epoch in range(recipe.num_epochs):
        iters = [iter(l) for l in loaders]
        epoch_losses: dict[str, list[float]] = {s.name: [] for s in specs}
        for _ in range(steps_per_epoch):
            for g_idx in range(len(specs)):
                spec = specs[g_idx]
                if spec.mode == "gd" and not gd_active.get(spec.name,
                                                           True):
                    continue  # constrained ascent already satisfied
                try:
                    images, texts = next(iters[g_idx])
                except StopIteration:  # short group: cycle its loader
                    iters[g_idx] = iter(loaders[g_idx])
                    images, texts = next(iters[g_idx])
                images = images.to(device, non_blocking=True).to(
                    torch.bfloat16)
                texts = texts.to(device, non_blocking=True)
                img_f = model.encode_image(images)
                txt_f = model.encode_text(texts)
                loss = clip_contrastive_loss(img_f, txt_f,
                                             model.logit_scale)
                sign = -1.0 if spec.mode == "gd" else 1.0
                total = sign * spec.weight * loss
                optimizer.zero_grad()
                total.backward()
                optimizer.step()
                epoch_losses[spec.name].append(loss.item())
                global_step += 1
                if spec.mode == "gd" and probe_loader is not None and \
                        global_step % 10 == 0:
                    cur = _probe_mean_sim()
                    if cur <= gd_stop_threshold:
                        gd_active[spec.name] = False
                        summary.setdefault("gd_stopped_at_step", {})
                        summary["gd_stopped_at_step"][spec.name] = \
                            global_step
                        log.info("[%s] gd '%s' stopped at step %d "
                                 "(probe sim %.4f <= %.4f)",
                                 candidate_id, spec.name, global_step,
                                 cur, gd_stop_threshold)
        summary["epochs"].append({
            "epoch": epoch + 1,
            "losses": {name: round(sum(v) / len(v), 4)
                       for name, v in epoch_losses.items() if v},
        })
        log.info("[%s] epoch %d/%d losses=%s", candidate_id, epoch + 1,
                 recipe.num_epochs, summary["epochs"][-1]["losses"])
    summary["num_optimizer_steps"] = global_step
    summary["train_seconds"] = round(time.time() - t0, 1)

    torch.save(model.state_dict(), output_dir / "pytorch_model.bin")
    with open(output_dir / "training_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    log.info("[%s] saved -> %s (%d steps, %.1fs)", candidate_id,
             output_dir, global_step, summary["train_seconds"])
    return summary


def make_salmu_noop_checkpoint(
    mf_checkpoint: str | Path, output_dir: str | Path,
) -> None:
    """B0: MF^SALMU verbatim (the no-op 'unlearning' control)."""
    import shutil
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(Path(mf_checkpoint) / "pytorch_model.bin",
                 output_dir / "pytorch_model.bin")
    with open(output_dir / "training_summary.json", "w") as f:
        json.dump({"candidate_id": "B0", "method": "noop",
                   "init_checkpoint": str(mf_checkpoint),
                   "num_optimizer_steps": 0}, f, indent=2)
    log.info("B0 no-op checkpoint copied from %s", mf_checkpoint)
