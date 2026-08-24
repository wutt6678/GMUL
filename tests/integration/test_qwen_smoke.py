"""Integration smoke test for Qwen3.5-9B on RTX 6000 Ada 48GB.

Run with:
    CUDA_VISIBLE_DEVICES=0 pytest tests/integration/test_qwen_smoke.py -v -s
"""

from __future__ import annotations

import pytest
import torch

MODEL_NAME = "Qwen/Qwen3.5-9B"


@pytest.fixture(scope="module")
def processor():
    from transformers import AutoProcessor
    return AutoProcessor.from_pretrained(MODEL_NAME)


@pytest.fixture(scope="module")
def model():
    from transformers import AutoModelForImageTextToText
    m = AutoModelForImageTextToText.from_pretrained(
        MODEL_NAME,
        dtype=torch.bfloat16,
        device_map="cuda:0",
        trust_remote_code=False,
    )
    yield m
    del m
    torch.cuda.empty_cache()


# ---- Loading tests ----

class TestQwenLoad:
    def test_model_loads_bf16(self, model):
        """Qwen3.5-9B loads in BF16."""
        assert model is not None
        param = next(model.parameters())
        assert param.dtype == torch.bfloat16

    def test_processor_loads(self, processor):
        """Processor loads successfully."""
        assert processor is not None
        assert hasattr(processor, "apply_chat_template")

    def test_parameter_count(self, model):
        """Model has approximately 9B parameters."""
        n = sum(p.numel() for p in model.parameters())
        assert 8e9 < n < 12e9, f"Unexpected parameter count: {n}"


# ---- Text generation tests ----

class TestTextGeneration:
    def test_text_only_forward(self, model, processor):
        """Text-only generation succeeds."""
        messages = [{"role": "user", "content": "What is 2+2?"}]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=16, do_sample=False)
        result = processor.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]
        assert len(result.strip()) > 0


# ---- Image+text generation tests ----

class TestMultimodalGeneration:
    def test_image_text_forward(self, model, processor):
        """Image+text generation succeeds."""
        import numpy as np
        from PIL import Image

        img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": img},
                    {"type": "text", "text": "Describe this image briefly."},
                ],
            }
        ]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=[img], return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=16, do_sample=False)
        result = processor.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]
        assert len(result.strip()) > 0


# ---- LoRA tests ----

class TestLoRA:
    def test_lora_attach(self, model, processor):
        """LoRA adapters can be attached to language projection modules."""
        from peft import LoraConfig, get_peft_model

        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        config = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            target_modules=target_modules,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model_lora = get_peft_model(model, config)
        trainable, total = model_lora.get_nb_trainable_parameters()
        assert trainable > 0
        assert trainable < total
        del model_lora

    def test_lora_backward(self, model, processor):
        """One forward+backward step succeeds with LoRA."""
        from peft import LoraConfig, get_peft_model

        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        config = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            target_modules=target_modules,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model_lora = get_peft_model(model, config)
        model_lora.train()

        messages = [
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "The answer is 4."},
        ]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        inputs = processor(text=[text], return_tensors="pt").to(model_lora.device)
        input_ids = inputs["input_ids"]
        labels = input_ids.clone()
        labels[labels == processor.tokenizer.pad_token_id] = -100
        outputs = model_lora(input_ids=input_ids, labels=labels)
        loss = outputs.loss
        loss.backward()
        assert loss.item() > 0
        del model_lora

    def test_lora_save_load(self, model, processor, tmp_path):
        """LoRA adapter can be saved and reloaded."""
        from peft import LoraConfig, get_peft_model, PeftModel

        target_modules = ["q_proj", "v_proj"]
        config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=target_modules,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model_lora = get_peft_model(model, config)
        save_path = tmp_path / "lora_adapter"
        model_lora.save_pretrained(str(save_path))

        # Reload
        model_reloaded = PeftModel.from_pretrained(model, str(save_path))
        assert model_reloaded is not None
        del model_lora, model_reloaded


# ---- 4-bit loading test (optional) ----

class TestQuantization:
    def test_four_bit_load(self):
        """4-bit loading is tested (non-blocking if bitsandbytes is unavailable)."""
        try:
            from transformers import AutoModelForImageTextToText, BitsAndBytesConfig

            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
            m = AutoModelForImageTextToText.from_pretrained(
                MODEL_NAME,
                dtype=torch.bfloat16,
                device_map="cuda:0",
                quantization_config=bnb_config,
                trust_remote_code=False,
            )
            assert m is not None
            del m
            torch.cuda.empty_cache()
        except ImportError:
            pytest.skip("bitsandbytes not installed")
