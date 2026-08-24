"""Qwen3.5-9B generation client for the semantic hierarchy pipeline.

Deterministic greedy decoding; batching; explicit provenance capture
(model id, library versions, load mode).  The client never interprets
model output — all interpretation lives in ``semantic_pipeline``.
"""

from __future__ import annotations

import subprocess
from typing import Any

DEFAULT_MODEL_ID = "Qwen/Qwen3.5-9B"


def pick_free_gpu(min_free_gib: float = 8.0) -> int | None:
    """Return the GPU index with the most free memory (>= min_free_gib).

    Uses nvidia-smi (no CUDA context required) so it works even when the
    GPUs are heavily occupied and CUDA context init would itself OOM.
    """
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.free",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=15, check=True,
        ).stdout
        free_mib = [(int(idx), int(mem))
                    for idx, mem in (line.split(", ") for line in
                                     out.strip().splitlines() if line)]
        if free_mib:
            idx, mem = max(free_mib, key=lambda t: t[1])
            return idx if mem >= min_free_gib * 1024 else None
    except (OSError, subprocess.SubprocessError, ValueError):
        pass
    # Fallback: torch (requires a working CUDA context)
    import torch
    best_device, best_free = None, min_free_gib * 1024**3
    for i in range(torch.cuda.device_count()):
        try:
            free, _total = torch.cuda.mem_get_info(i)
        except Exception:
            continue
        if free > best_free:
            best_device, best_free = i, free
    return best_device


class QwenGenerator:
    """Greedy-decoding batched generator over a local Qwen3.5-9B."""

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        device: str | None = None,
        load_in_4bit: bool = True,
        max_seq_len: int = 2048,
    ):
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        if device is None or device == "auto":
            gpu = pick_free_gpu()
            if gpu is None:
                raise RuntimeError(
                    "No GPU has enough free memory for Qwen generation")
            device = f"cuda:{gpu}"
        self.device = device
        self.model_id = model_id
        self.load_in_4bit = load_in_4bit

        kwargs: dict[str, Any] = {"torch_dtype": torch.bfloat16}
        if load_in_4bit:
            from transformers import BitsAndBytesConfig
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )

        self.processor = AutoProcessor.from_pretrained(model_id)
        # Left padding is required for correct decoder-only batch generation
        if self.processor.tokenizer.padding_side != "left":
            self.processor.tokenizer.padding_side = "left"
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_id, device_map={"": device}, **kwargs
        ).eval()

    # ---- Provenance -------------------------------------------------------

    def provenance(self) -> dict[str, str]:
        import torch
        import transformers
        return {
            "model_id": self.model_id,
            "load_mode": "4bit_bnb" if self.load_in_4bit else "bf16",
            "device": self.device,
            "transformers_version": transformers.__version__,
            "torch_version": torch.__version__,
            "decoding": "greedy (do_sample=False)",
        }

    # ---- Generation -------------------------------------------------------

    def generate(
        self,
        prompts: list[str],
        max_new_tokens: int = 256,
        batch_size: int = 4,
    ) -> list[str]:
        """Greedy-generate one completion per prompt (chat template)."""
        import torch

        outputs: list[str] = []
        for start in range(0, len(prompts), batch_size):
            batch = prompts[start:start + batch_size]
            texts = []
            for p in batch:
                try:
                    # Qwen3.x thinking mode emits long reasoning before the
                    # answer and burns the token budget; the pipeline needs
                    # strict JSON only.
                    text = self.processor.apply_chat_template(
                        [{"role": "user", "content": p}],
                        tokenize=False, add_generation_prompt=True,
                        enable_thinking=False,
                    )
                except TypeError:
                    text = self.processor.apply_chat_template(
                        [{"role": "user", "content": p}],
                        tokenize=False, add_generation_prompt=True,
                    )
                texts.append(text)
            inputs = self.processor(
                text=texts, return_tensors="pt", padding=True,
                truncation=True, max_length=1536,
            ).to(self.device)
            with torch.no_grad():
                gen = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=self.processor.tokenizer.pad_token_id,
                )
            trimmed = gen[:, inputs["input_ids"].shape[1]:]
            outputs.extend(self.processor.batch_decode(
                trimmed, skip_special_tokens=True))
        return outputs

    def unload(self) -> None:
        import torch
        del self.model
        torch.cuda.empty_cache()
