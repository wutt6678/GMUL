"""Reference-state LoRA trainer (Iteration 7).

MF = Train(D_F), MG = Train(D_G), MN = Train(D_N): the base model, LoRA
configuration, seed, optimizer, training budget (epochs), multimodal
formatting and checkpoint pipeline are IDENTICAL across states — the only
intended difference is the transformed knowledge dataset.  Any behavioral
difference is therefore attributable to the state's knowledge content.

Scope notes
-----------
* LoRA adapts the language-model projection matrices; the vision tower is
  frozen.  This scope is identical for all states, which is what the
  MF/MG/MN counterfactual requires.  (MIDP experience: language-attention
  -only LoRA limits absolute visual-binding capability — that affects all
  reference states equally and is reported, not hidden.)
* Batch size is 1 with gradient accumulation (shared-GPU memory budgets);
  identical across states.
* Evaluation query families are NEVER used here; training examples come
  exclusively from ``state_datasets``.
"""

from __future__ import annotations

import json
import math
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from granunlearn.logging_utils import setup_logger
from granunlearn.training.state_datasets import TrainingExample, load_state_examples

log = setup_logger("reference_trainer")


@dataclass(frozen=True)
class ReferenceRecipe:
    """The single optimization recipe shared by MF, MG and MN."""

    model_id: str = "Qwen/Qwen3.5-9B"
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: tuple = (
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    )
    learning_rate: float = 1e-4
    weight_decay: float = 0.0
    # Budget raised after the first attempt (3 epochs): states separated
    # on target facts but retain slices underfit (retain accuracy ~0.3-0.4
    # with near-miss outputs such as '1989-03-15' vs '1989-04-15').  The
    # recipe stays identical across MF/MG/MN.
    num_epochs: int = 10
    per_device_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    max_length: int = 1536
    # 384x384 -> ~729 vision tokens (27x27 patches); must fit inside
    # max_length together with the chat-template text, otherwise the
    # processor's image-token alignment check fails under truncation.
    max_image_pixels: int = 384 * 384
    seed: int = 42
    bf16: bool = True
    gradient_checkpointing: bool = True

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["lora_target_modules"] = list(self.lora_target_modules)
        return d


def set_recipe_seeds(seed: int) -> None:
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _encode_example(example: TrainingExample, processor, max_length: int,
                    max_pixels: int):
    """Tokenize one example with the assistant completion as supervision.

    Labels are masked over everything before the assistant turn, so the
    model is trained to produce the level value given the prompt.
    """
    import torch

    image = None
    user_content: list[dict[str, Any]] = [{"type": "text",
                                           "text": example.prompt}]
    if example.image_path:
        from PIL import Image
        image = Image.open(example.image_path).convert("RGB")
        user_content.insert(0, {"type": "image"})

    messages = [
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": example.completion},
    ]
    prompt_messages = [{"role": "user", "content": user_content}]

    try:
        full_text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False,
            enable_thinking=False)
        prompt_text = processor.apply_chat_template(
            prompt_messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False)
    except TypeError:
        full_text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False)
        prompt_text = processor.apply_chat_template(
            prompt_messages, tokenize=False, add_generation_prompt=True)

    common = {}
    if image is not None:
        common = {"images": [[image]], "max_pixels": max_pixels}
    full = processor(text=[full_text], return_tensors="pt",
                     truncation=True, max_length=max_length, **common)
    prompt_only = processor(text=[prompt_text], return_tensors="pt",
                            truncation=True, max_length=max_length, **common)
    n_prompt = prompt_only["input_ids"].shape[1]

    labels = full["input_ids"].clone()
    labels[0, :min(n_prompt, labels.shape[1])] = -100
    full["labels"] = labels
    return full


def train_state(
    state: str,
    dataset_path: str | Path,
    output_dir: str | Path,
    device: str = "cuda:0",
    recipe: ReferenceRecipe | None = None,
) -> dict[str, Any]:
    """Train one reference state. Returns the training summary dict."""
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForImageTextToText, AutoProcessor

    recipe = recipe or ReferenceRecipe()
    output_dir = Path(output_dir)
    (output_dir / "adapters").mkdir(parents=True, exist_ok=True)
    set_recipe_seeds(recipe.seed)

    examples = load_state_examples(dataset_path)
    log.info("[%s] %d training examples | device=%s", state, len(examples),
             device)

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

    lora_config = LoraConfig(
        r=recipe.lora_r,
        lora_alpha=recipe.lora_alpha,
        lora_dropout=recipe.lora_dropout,
        target_modules=list(recipe.lora_target_modules),
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=recipe.learning_rate, weight_decay=recipe.weight_decay)

    rng = random.Random(recipe.seed)
    summary: dict[str, Any] = {
        "state": state,
        "recipe": recipe.to_dict(),
        "num_examples": len(examples),
        "epochs": [],
        "num_optimizer_steps": 0,
        "final_loss": None,
        "device": device,
    }
    global_step = 0
    t0 = time.time()
    accum = recipe.gradient_accumulation_steps
    for epoch in range(recipe.num_epochs):
        order = list(range(len(examples)))
        rng.shuffle(order)
        epoch_loss, n_batches = 0.0, 0
        optimizer.zero_grad()
        for i, idx in enumerate(order):
            # Normalize each micro-batch by the ACTUAL size of its
            # accumulation group: the trailing group of an epoch can be
            # smaller than `accum` (68 examples % 8 = 4) and must not be
            # divided by 8, or that update receives half its proper
            # gradient magnitude (Iteration 7 review fix).
            group_start = (i // accum) * accum
            group_size = min(accum, len(order) - group_start)
            enc = _encode_example(
                examples[idx], processor, recipe.max_length,
                recipe.max_image_pixels)
            enc = {k: v.to(device) for k, v in enc.items()
                   if torch.is_tensor(v)}
            out = model(**enc)
            loss = out.loss / group_size
            loss.backward()
            epoch_loss += out.loss.item()
            n_batches += 1
            if (i + 1) % accum == 0 or i + 1 == len(order):
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1
        avg = epoch_loss / max(n_batches, 1)
        summary["epochs"].append(
            {"epoch": epoch + 1, "avg_loss": round(avg, 4)})
        summary["final_loss"] = round(avg, 4)
        log.info("[%s] epoch %d/%d avg_loss=%.4f", state, epoch + 1,
                 recipe.num_epochs, avg)
    summary["num_optimizer_steps"] = global_step
    summary["train_seconds"] = round(time.time() - t0, 1)

    model.save_pretrained(output_dir / "adapters")
    processor.save_pretrained(output_dir / "processor")
    with open(output_dir / "training_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    log.info("[%s] saved adapters -> %s (%d steps, %.1fs)", state,
             output_dir, global_step, summary["train_seconds"])
    return summary
