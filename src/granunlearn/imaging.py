"""Image-resolution contract shared by training and evaluation.

The recipe pins ``max_image_pixels`` (384*384) as the multimodal
formatting budget.  In ``transformers`` 5.x the ``Qwen3VLProcessor``
does NOT accept a ``max_pixels`` keyword: the image processor reads its
area bounds from ``size={"shortest_edge", "longest_edge"}``, and an
unrecognized kwarg is silently swallowed by ``**kwargs``.  Passing
``max_pixels=...`` therefore leaves every image at its native
resolution — a 1024x1024 MLLMU portrait expands to a 64x64 patch grid
(4,096 patches / 1,024 vision tokens) instead of the intended
384x384-scale grid, which both inflates the vision-tower cost ~7x and
pushes the prompt towards the ``max_length`` truncation budget (truncated
image tokens break the processor's alignment check).

``image_size_kwargs`` is the single place that turns the recipe's pixel
budget into processor arguments, so the training and evaluation paths
can never drift apart — identical formatting across MF/MG/MN and every
unlearning candidate is the counterfactual's core requirement.
"""

from __future__ import annotations

from typing import Any

#: Qwen3-VL uses patch_size=16 with merge_size=2, so the smallest
#: meaningful image is a 2x2 merged patch = 32x32 pixels.  Used as the
#: area FLOOR so small inputs are never gratuitously upscaled.
MIN_IMAGE_AREA = 32 * 32


def image_size_kwargs(max_image_pixels: int) -> dict[str, Any]:
    """Processor kwargs that ACTUALLY enforce the pixel budget.

    ``longest_edge`` is the maximum pixel AREA (the Qwen2VL/Qwen3VL
    image-processor convention), ``shortest_edge`` the minimum.
    """
    if max_image_pixels <= 0:
        raise ValueError(
            f"max_image_pixels must be positive, got {max_image_pixels}")
    return {"size": {"longest_edge": int(max_image_pixels),
                     "shortest_edge": MIN_IMAGE_AREA}}
