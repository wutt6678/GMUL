"""CLIP contrastive trainer for the SALMU reference states.

MF^SALMU / MG^SALMU / MN^SALMU are fine-tuned from the SAME released
Clean CLIP ViT-B/16 checkpoint with an IDENTICAL recipe; only the
(image, caption) pair set (D_F / D_G / D_N) differs.  The logit scale
(temperature) is frozen so the three loss landscapes differ only via
the data.
"""

from __future__ import annotations

import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from granunlearn.logging_utils import setup_logger
from granunlearn.salmu.state_datasets import SalmuTrainingPair, \
    load_state_pairs

log = setup_logger("salmu_clip_trainer")


@dataclass(frozen=True)
class ClipRecipe:
    """One recipe shared verbatim by MF/MG/MN."""

    arch: str = "ViT-B-16"
    learning_rate: float = 1e-5
    weight_decay: float = 0.0
    num_epochs: int = 3
    batch_size: int = 256
    seed: int = 42
    bf16: bool = True
    num_workers: int = 4
    freeze_logit_scale: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def set_seeds(seed: int) -> None:
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class SalmuPairDataset:
    """(image, caption) dataset backed by the released `sensitive`
    parquet split, indexed by file_name."""

    def __init__(self, pairs: list[SalmuTrainingPair],
                 parquet_dir: str | Path, preprocess, tokenizer,
                 caption_limit: int = 77):
        from datasets import concatenate_datasets, load_dataset
        shard_dir = Path(parquet_dir)
        shards = sorted(shard_dir.glob("sensitive-*.parquet"))
        if not shards:
            raise FileNotFoundError(f"No sensitive shards in {shard_dir}")
        self.ds = concatenate_datasets(
            [load_dataset("parquet", data_files=str(s), split="train")
             for s in shards])
        self.idx_by_file = {row["file_name"]: i
                            for i, row in enumerate(self.ds)}
        self.rows = [(p, self.idx_by_file[p.image_file]) for p in pairs]
        self.preprocess = preprocess
        self.tokenizer = tokenizer
        self.caption_limit = caption_limit

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int):
        pair, idx = self.rows[i]
        item = self.ds[idx]
        image = item["image"].convert("RGB")
        text = self.tokenizer([pair.caption],
                              context_length=self.caption_limit)[0]
        return self.preprocess(image), text


def clip_contrastive_loss(image_features, text_features,
                          logit_scale) -> "torch.Tensor":
    import torch
    import torch.nn.functional as F
    image_features = F.normalize(image_features, dim=-1)
    text_features = F.normalize(text_features, dim=-1)
    logits = logit_scale.exp() * image_features @ text_features.t()
    labels = torch.arange(logits.shape[0], device=logits.device)
    return (F.cross_entropy(logits, labels) +
            F.cross_entropy(logits.t(), labels)) / 2


def train_salmu_state(
    state: str,
    pairs_path: str | Path,
    parquet_dir: str | Path,
    init_checkpoint: str | Path,
    output_dir: str | Path,
    device: str = "cuda:0",
    recipe: ClipRecipe | None = None,
) -> dict[str, Any]:
    """Fine-tune one reference state from the Clean checkpoint."""
    import open_clip
    import torch

    recipe = recipe or ClipRecipe()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    set_seeds(recipe.seed)

    pairs = load_state_pairs(pairs_path)
    log.info("[%s] %d pairs | init=%s | device=%s", state, len(pairs),
             Path(init_checkpoint).name, device)

    model, _, preprocess = open_clip.create_model_and_transforms(
        recipe.arch, pretrained=str(init_checkpoint))
    tokenizer = open_clip.get_tokenizer(recipe.arch)
    model = model.to(device)
    if recipe.bf16:
        model = model.to(torch.bfloat16)
    model.train()
    if recipe.freeze_logit_scale:
        model.logit_scale.requires_grad_(False)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        params, lr=recipe.learning_rate,
        weight_decay=recipe.weight_decay)

    dataset = SalmuPairDataset(pairs, parquet_dir, preprocess, tokenizer)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=recipe.batch_size, shuffle=True,
        num_workers=recipe.num_workers, pin_memory=True,
        drop_last=True)

    summary: dict[str, Any] = {
        "state": state,
        "recipe": recipe.to_dict(),
        "init_checkpoint": str(init_checkpoint),
        "num_pairs": len(pairs),
        "epochs": [],
        "num_optimizer_steps": 0,
        "device": device,
    }
    global_step = 0
    t0 = time.time()
    for epoch in range(recipe.num_epochs):
        epoch_loss, n_batches = 0.0, 0
        for images, texts in loader:
            images = images.to(device, non_blocking=True)
            texts = texts.to(device, non_blocking=True)
            if recipe.bf16:
                images = images.to(torch.bfloat16)
            image_features = model.encode_image(images)
            text_features = model.encode_text(texts)
            loss = clip_contrastive_loss(
                image_features, text_features, model.logit_scale)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
            global_step += 1
        avg = epoch_loss / max(n_batches, 1)
        summary["epochs"].append(
            {"epoch": epoch + 1, "avg_loss": round(avg, 4)})
        summary["final_loss"] = round(avg, 4)
        log.info("[%s] epoch %d/%d avg_loss=%.4f", state, epoch + 1,
                 recipe.num_epochs, avg)
    summary["num_optimizer_steps"] = global_step
    summary["train_seconds"] = round(time.time() - t0, 1)

    torch.save(model.state_dict(), output_dir / "pytorch_model.bin")
    with open(output_dir / "training_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    log.info("[%s] saved checkpoint -> %s (%d steps, %.1fs)", state,
             output_dir, global_step, summary["train_seconds"])
    return summary
