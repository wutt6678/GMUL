"""Qwen-assisted semantic hierarchy generation (Iteration 5).

Pipeline: source values -> Qwen proposal (strict JSON) -> deterministic
validation -> independent Qwen verification -> ambiguity/confidence gate
-> accepted hierarchy -> manual audit sample -> AssociationRecords.

Principle: REJECTION over forced hierarchy construction.
"""

from .client import QwenGenerator, pick_free_gpu

__all__ = ["QwenGenerator", "pick_free_gpu"]
