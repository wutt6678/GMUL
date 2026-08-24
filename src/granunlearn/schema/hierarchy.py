"""Hierarchy level and supporting types for the granularity schema.

Convention
----------
* level 0 = finest original value
* higher level number = coarser value
* last optional level = unknown (represented separately, not as a semantic parent)
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class HierarchyLevel(BaseModel):
    """One node in a value hierarchy chain."""

    level: int = Field(ge=0, description="0 = finest, higher = coarser")
    canonical_id: str = Field(description="Unique identifier for this level's value")
    value: str = Field(description="Human-readable value string")
    normalized_value: str = Field(description="Lowercased, stripped canonical form for matching")
    parent_id: str | None = Field(
        default=None,
        description="canonical_id of the parent level (None for the coarsest node)",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Free-form metadata (e.g. taxonomy rank, numeric bounds)",
    )


class ImageRef(BaseModel):
    """Reference to an image used in an association."""

    image_id: str
    path: str = Field(description="Relative path from repo root")
    source: str = Field(default="original", description="original | synthetic | augmented")
    split: Literal["train", "val", "test"] | None = Field(
        default=None,
        description=(
            "Which split this image belongs to.  Required for datasets that "
            "use within-entity image splitting (e.g. iNaturalist)."
        ),
    )


class SplitInfo(BaseModel):
    """Train / validation / test split assignment."""

    split: str = Field(pattern=r"^(train|val|test)$")
    fold: int = Field(default=0, ge=0)


class ProvenanceInfo(BaseModel):
    """Provenance tracking for every derived record."""

    source_dataset: str
    source_entity_id: str | None = None
    source_record_id: str | None = None
    generation_model: str | None = Field(
        default=None,
        description="Model slug if LLM-generated (e.g. 'qwen3.5-9b')",
    )
    hierarchy_builder: str = Field(
        default="deterministic",
        description="Builder name: deterministic | qwen_assisted | manual",
    )
    code_version: str | None = None
    prompt_version: str | None = None
