#!/usr/bin/env python3
"""Environment probe for GMUL — verifies GPU, model loading, LoRA, and forward/backward.

Usage
-----
    python scripts/env_probe.py \
        --model Qwen/Qwen3.5-9B \
        --output outputs/env_probe

Acceptance criteria (RTX 6000 Ada 48 GB):
    - Qwen3.5-9B loads in BF16
    - One image+text generation succeeds
    - One text-only generation succeeds
    - LoRA adapters can be attached to language-model projection modules
    - A one-step backward pass succeeds
    - 4-bit loading is tested (optional)
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _section(msg: str) -> None:
    print(f"\n{'='*60}\n  {msg}\n{'='*60}")


def _ok(msg: str) -> None:
    print(f"  [OK]   {msg}")


def _fail(msg: str) -> None:
    print(f"  [FAIL] {msg}")


def _info(msg: str) -> None:
    print(f"  [INFO] {msg}")


# ---------------------------------------------------------------------------
# environment basics
# ---------------------------------------------------------------------------

def collect_env_basics() -> dict:
    """Collect Python / package / CUDA version info without loading any model."""
    import torch
    import transformers
    import peft

    info: dict = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cudnn": str(torch.backends.cudnn.version()),
        "transformers": transformers.__version__,
        "peft": peft.__version__,
    }

    # bitsandbytes
    try:
        import bitsandbytes
        info["bitsandbytes"] = bitsandbytes.__version__
    except ImportError:
        info["bitsandbytes"] = None

    # GPU info
    if torch.cuda.is_available():
        n = torch.cuda.device_count()
        info["gpu_count"] = n
        info["gpu_names"] = [torch.cuda.get_device_name(i) for i in range(n)]
        info["gpu_vram_gb"] = [
            round(torch.cuda.get_device_properties(i).total_memory / (1024 ** 3), 1)
            for i in range(n)
        ]
    else:
        info["gpu_count"] = 0
        info["gpu_names"] = []
        info["gpu_vram_gb"] = []

    return info


# ---------------------------------------------------------------------------
# model loading
# ---------------------------------------------------------------------------

def probe_model_load(model_name: str) -> dict:
    """Load Qwen3.5-9B in BF16 on the first available GPU and run basic generation."""
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    result: dict = {}

    # ---- Processor ----
    _section("Loading processor")
    try:
        processor = AutoProcessor.from_pretrained(model_name)
        _ok(f"Processor loaded: {type(processor).__name__}")
        result["processor_class"] = type(processor).__name__
    except Exception as e:
        _fail(f"Processor load failed: {e}")
        result["processor_class"] = None
        return result

    # ---- BF16 model load ----
    _section("Loading model (BF16)")
    t0 = time.time()
    try:
        model = AutoModelForImageTextToText.from_pretrained(
            model_name,
            dtype=torch.bfloat16,
            device_map="cuda:0",
            trust_remote_code=False,
        )
        load_time = time.time() - t0
        _ok(f"Model loaded in {load_time:.1f}s: {type(model).__name__}")
        n_params = sum(p.numel() for p in model.parameters())
        result["model_class"] = type(model).__name__
        result["n_params_b"] = round(n_params / 1e9, 2)
        result["load_time_s"] = round(load_time, 1)
        result["qwen_load_bf16"] = True

        mem_used = torch.cuda.memory_allocated(0) / (1024 ** 3)
        _info(f"GPU 0 memory allocated: {mem_used:.1f} GB")
    except Exception as e:
        _fail(f"BF16 load failed: {e}")
        result["qwen_load_bf16"] = False
        return result

    # ---- Discover linear modules for LoRA ----
    _section("Discovering linear modules for LoRA targeting")
    lang_linear_modules = set()
    mm_linear_modules = set()
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            parts = name.split(".")
            leaf = parts[-1]
            # Language-side projection modules (standard transformer naming)
            if leaf in {"q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"}:
                lang_linear_modules.add(leaf)
            else:
                mm_linear_modules.add(leaf)
    result["lang_linear_modules"] = sorted(lang_linear_modules)
    result["mm_linear_modules"] = sorted(mm_linear_modules)
    _info(f"Language-side linear modules: {sorted(lang_linear_modules)}")
    _info(f"Other linear modules: {sorted(mm_linear_modules)}")

    # ---- Text-only generation ----
    _section("Text-only generation")
    try:
        messages = [
            {"role": "user", "content": "What is the capital of France? Answer in one sentence."}
        ]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], return_tensors="pt").to(model.device)
        with torch.no_grad():
            out_ids = model.generate(**inputs, max_new_tokens=32, do_sample=False)
        text_out = processor.batch_decode(out_ids[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]
        _ok(f"Text generation output: {text_out.strip()!r}")
        result["qwen_text_forward"] = True
    except Exception as e:
        _fail(f"Text generation failed: {e}")
        result["qwen_text_forward"] = False

    # ---- Image+text generation ----
    _section("Image+text generation")
    try:
        from PIL import Image
        import numpy as np

        # Create a small synthetic test image
        img_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        test_image = Image.fromarray(img_array)

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": test_image},
                    {"type": "text", "text": "Describe this image in one sentence."},
                ],
            }
        ]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=[test_image], return_tensors="pt").to(model.device)
        with torch.no_grad():
            out_ids = model.generate(**inputs, max_new_tokens=32, do_sample=False)
        mm_out = processor.batch_decode(out_ids[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]
        _ok(f"Image+text generation output: {mm_out.strip()!r}")
        result["qwen_image_forward"] = True
    except Exception as e:
        _fail(f"Image+text generation failed: {e}")
        result["qwen_image_forward"] = False
        import traceback
        traceback.print_exc()

    # ---- LoRA attach ----
    _section("LoRA adapter attachment")
    try:
        from peft import LoraConfig, get_peft_model

        target_modules = sorted(lang_linear_modules) if lang_linear_modules else ["q_proj", "v_proj"]
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            target_modules=target_modules,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model_lora = get_peft_model(model, lora_config)
        trainable, total = model_lora.get_nb_trainable_parameters()
        _ok(f"LoRA attached — trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")
        _ok(f"Target modules: {target_modules}")
        result["lora_attach"] = True
        result["lora_trainable_params"] = trainable
        result["lora_target_modules"] = target_modules
    except Exception as e:
        _fail(f"LoRA attachment failed: {e}")
        result["lora_attach"] = False
        import traceback
        traceback.print_exc()

    # ---- Forward + backward step ----
    _section("One forward + backward step")
    try:
        model_lora.train()
        messages = [
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "2+2 equals 4."},
        ]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        inputs = processor(text=[text], return_tensors="pt").to(model_lora.device)
        input_ids = inputs["input_ids"]
        labels = input_ids.clone()
        labels[labels == processor.tokenizer.pad_token_id] = -100
        outputs = model_lora(input_ids=input_ids, labels=labels)
        loss = outputs.loss
        loss.backward()
        _ok(f"Forward+backward succeeded, loss={loss.item():.4f}")
        result["lora_backward"] = True
    except Exception as e:
        _fail(f"Forward+backward failed: {e}")
        result["lora_backward"] = False
        import traceback
        traceback.print_exc()

    # ---- 4-bit loading test ----
    _section("4-bit loading test (optional)")
    del model, model_lora
    torch.cuda.empty_cache()
    try:
        from transformers import BitsAndBytesConfig

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        model_4bit = AutoModelForImageTextToText.from_pretrained(
            model_name,
            dtype=torch.bfloat16,
            device_map="cuda:0",
            quantization_config=bnb_config,
            trust_remote_code=False,
        )
        mem_4bit = torch.cuda.memory_allocated(0) / (1024 ** 3)
        _ok(f"4-bit load succeeded, GPU memory: {mem_4bit:.1f} GB")
        result["four_bit_load"] = True
        del model_4bit
        torch.cuda.empty_cache()
    except Exception as e:
        _info(f"4-bit load failed (non-critical): {e}")
        result["four_bit_load"] = False
        torch.cuda.empty_cache()

    return result


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="GMUL environment probe")
    parser.add_argument("--model", default="Qwen/Qwen3.5-9B", help="Model name or path")
    parser.add_argument("--output", default="outputs/env_probe", help="Output directory")
    parser.add_argument("--print-linear-modules", action="store_true",
                        help="Only print linear module names and exit")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Step 1: environment basics ----
    _section("Environment basics")
    env = collect_env_basics()
    for k, v in env.items():
        _info(f"{k}: {v}")

    if env["gpu_count"] == 0:
        _fail("No GPU detected — cannot proceed")
        sys.exit(1)

    if args.print_linear_modules:
        # Quick module-name listing without running the full probe
        import torch
        from transformers import AutoModelForImageTextToText
        model = AutoModelForImageTextToText.from_pretrained(
            args.model, dtype=torch.bfloat16, device_map="cuda:0"
        )
        modules = set()
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear):
                modules.add(name.split(".")[-1])
        print(json.dumps(sorted(modules), indent=2))
        return

    # ---- Step 2: full probe ----
    probe = probe_model_load(args.model)

    # ---- Step 3: assemble and save report ----
    _section("Summary")
    report = {**env, **probe}

    checks = [
        "qwen_load_bf16",
        "qwen_text_forward",
        "qwen_image_forward",
        "lora_attach",
        "lora_backward",
    ]
    all_pass = True
    for check in checks:
        val = report.get(check, False)
        status = "PASS" if val else "FAIL"
        if not val:
            all_pass = False
        print(f"  [{status}] {check}")

    report["all_acceptance_pass"] = all_pass

    out_path = output_dir / "environment.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    _info(f"Report saved to {out_path}")

    if not all_pass:
        _fail("Some acceptance criteria FAILED")
        sys.exit(1)
    else:
        _ok("All acceptance criteria PASSED")


if __name__ == "__main__":
    main()
