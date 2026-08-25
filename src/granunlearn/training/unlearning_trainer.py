"""MF -> MU unlearning trainer (Iteration 9).

All baselines start from the canonical MF checkpoint (the state that
knows the fine facts) and apply a small objective built from the
unlearning knowledge groups:

* ``gd``  — gradient ASCENT on the group's loss (forget/suppress);
* ``sft`` — standard supervised fine-tuning on the group's completion.

Methods:
* B0 no-op                              (MF adapter copied, no training)
* B1 complete-forget                    (gd on fine_target)
* B2 coarse-positive SFT                (sft on target_level)
* B3 fine-suppress + coarse + retain    (gd fine_target + sft
                                         target_level + sft retain)

Everything else (LoRA config, optimizer, seed, multimodal formatting,
accumulation normalization) is inherited from the Iteration 7 recipe so
comparisons stay counterfactually clean; only the swept knobs (lr,
epochs, suppress weight) vary and they are tuned on train/val probes
ONLY (Iteration 9 held-out-test protocol).
"""

from __future__ import annotations

import json
import random
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from granunlearn.logging_utils import setup_logger
from granunlearn.training.reference_trainer import (
    ReferenceRecipe,
    _encode_example,
    set_recipe_seeds,
)
from granunlearn.training.state_datasets import load_state_examples

log = setup_logger("unlearning_trainer")


@dataclass(frozen=True)
class GroupSpec:
    """One objective component: a knowledge group + how to use it."""

    name: str
    path: str | Path
    mode: Literal["sft", "gd"]
    weight: float = 1.0


def train_unlearning(
    method_id: str,
    groups: list[GroupSpec],
    output_dir: str | Path,
    device: str = "cuda:0",
    recipe: ReferenceRecipe | None = None,
    init_adapter_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Train one MF->MU baseline. Returns the training summary dict.

    ``init_adapter_dir`` must be the canonical MF adapter directory —
    every Iteration 9 baseline continues from MF.  Epochs interleave
    the groups round-robin (each group seeded-shuffled independently)
    and every micro-batch loss is ``sign * weight * loss`` normalized
    by the ACTUAL trailing accumulation-group size.
    """
    import torch
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoModelForImageTextToText, AutoProcessor

    recipe = recipe or ReferenceRecipe()
    output_dir = Path(output_dir)
    (output_dir / "adapters").mkdir(parents=True, exist_ok=True)
    set_recipe_seeds(recipe.seed)

    from granunlearn.config import _find_repo_root
    repo_root = _find_repo_root(Path.cwd()) or Path.cwd()
    loaded = {g.name: load_state_examples(g.path, repo_root=repo_root)
              for g in groups}
    for g in groups:
        log.info("[%s] group %s: %d examples (%s, weight %.2f)",
                 method_id, g.name, len(loaded[g.name]), g.mode, g.weight)

    processor = AutoProcessor.from_pretrained(recipe.model_id)
    if processor.tokenizer.padding_side != "left":
        processor.tokenizer.padding_side = "left"
    model = AutoModelForImageTextToText.from_pretrained(
        recipe.model_id, device_map={"": device},
        torch_dtype=torch.bfloat16 if recipe.bf16 else torch.float32,
    )
    if recipe.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    if init_adapter_dir is not None:
        model = PeftModel.from_pretrained(
            model, str(init_adapter_dir), is_trainable=True)
        log.info("[%s] continuing from MF adapter %s",
                 method_id, init_adapter_dir)
    else:
        lora_config = LoraConfig(
            r=recipe.lora_r, lora_alpha=recipe.lora_alpha,
            lora_dropout=recipe.lora_dropout,
            target_modules=list(recipe.lora_target_modules), bias="none")
        model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=recipe.learning_rate, weight_decay=recipe.weight_decay)

    summary: dict[str, Any] = {
        "method_id": method_id,
        "recipe": recipe.to_dict(),
        "init_adapter_dir": str(init_adapter_dir) if init_adapter_dir
        else None,
        "groups": [{"name": g.name, "mode": g.mode, "weight": g.weight,
                    "num_examples": len(loaded[g.name])} for g in groups],
        "epochs": [],
        "num_optimizer_steps": 0,
        "device": device,
    }

    global_step = 0
    t0 = time.time()
    accum = recipe.gradient_accumulation_steps
    for epoch in range(recipe.num_epochs):
        # deterministic per-group shuffles, then round-robin interleave
        streams: list[list[tuple[str, int]]] = []
        for gi, g in enumerate(groups):
            order = list(range(len(loaded[g.name])))
            random.Random(f"{recipe.seed}:{epoch}:{gi}").shuffle(order)
            streams.append([(g.name, idx) for idx in order])
        stream: list[tuple[str, int]] = []
        longest = max(len(s) for s in streams)
        for pos in range(longest):
            for s in streams:
                if pos < len(s):
                    stream.append(s[pos])

        spec_by_name = {g.name: g for g in groups}
        group_loss: dict[str, list[float]] = defaultdict(list)
        epoch_loss, n_batches = 0.0, 0
        optimizer.zero_grad()
        for i, (gname, idx) in enumerate(stream):
            spec = spec_by_name[gname]
            sign = -1.0 if spec.mode == "gd" else 1.0
            group_start = (i // accum) * accum
            group_size = min(accum, len(stream) - group_start)
            example = loaded[gname][idx]
            enc = _encode_example(
                example, processor, recipe.max_length,
                recipe.max_image_pixels)
            enc = {k: v.to(device) for k, v in enc.items()
                   if torch.is_tensor(v)}
            out = model(**enc)
            loss = sign * spec.weight * out.loss / group_size
            loss.backward()
            group_loss[gname].append(out.loss.item())
            epoch_loss += sign * spec.weight * out.loss.item()
            n_batches += 1
            if (i + 1) % accum == 0 or i + 1 == len(stream):
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

        epoch_entry: dict[str, Any] = {"epoch": epoch + 1}
        for gname, losses in group_loss.items():
            epoch_entry[f"avg_loss_{gname}"] = round(
                sum(losses) / len(losses), 4)
        epoch_entry["avg_objective"] = round(
            epoch_loss / max(n_batches, 1), 4)
        summary["epochs"].append(epoch_entry)
        log.info("[%s] epoch %d/%d %s", method_id, epoch + 1,
                 recipe.num_epochs,
                 {k: v for k, v in epoch_entry.items() if k != "epoch"})

    summary["num_optimizer_steps"] = global_step
    summary["train_seconds"] = round(time.time() - t0, 1)
    model.save_pretrained(output_dir / "adapters")
    processor.save_pretrained(output_dir / "processor")
    with open(output_dir / "training_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    log.info("[%s] saved adapters -> %s (%d steps, %.1fs)", method_id,
             output_dir, global_step, summary["train_seconds"])
    return summary


def make_noop_checkpoint(
    method_id: str,
    init_adapter_dir: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """B0: copy the MF adapter unchanged (sanity baseline — identical
    weights must reproduce MF's metrics through the whole pipeline)."""
    import shutil
    output_dir = Path(output_dir)
    init_adapter_dir = Path(init_adapter_dir)
    if (output_dir / "adapters").exists():
        shutil.rmtree(output_dir / "adapters")
    shutil.copytree(init_adapter_dir, output_dir / "adapters")
    proc_src = init_adapter_dir.parent / "processor"
    if proc_src.exists():
        if (output_dir / "processor").exists():
            shutil.rmtree(output_dir / "processor")
        shutil.copytree(proc_src, output_dir / "processor")
    summary = {
        "method_id": method_id,
        "noop": True,
        "init_adapter_dir": str(init_adapter_dir),
        "num_optimizer_steps": 0,
        "note": "MF adapter copied unchanged; must reproduce MF metrics.",
    }
    with open(output_dir / "training_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    log.info("[%s] no-op checkpoint written -> %s", method_id, output_dir)
    return summary
