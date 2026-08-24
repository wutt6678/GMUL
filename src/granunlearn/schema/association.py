"""Association record — one entity–attribute pair with its hierarchy.

Each ``AssociationRecord`` is the canonical unit of knowledge that can be
memorised, forgotten, or granularity-reduced.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator

from .hierarchy import HierarchyLevel, ImageRef, ProvenanceInfo, SplitInfo


class AssociationRecord(BaseModel):
    """A single entity–attribute association with hierarchical value levels."""

    association_id: str = Field(description="Globally unique association identifier")
    dataset: str = Field(description="Source dataset name (e.g. 'mllmu_hier', 'inaturalist')")
    entity_id: str = Field(description="Unique entity identifier within the dataset")
    entity_name: str | None = Field(
        default=None,
        description="Human-readable entity name (e.g. person name, species common name)",
    )

    attribute_name: str = Field(description="Attribute being described (e.g. 'residence', 'occupation')")
    hierarchy_type: Literal["semantic", "taxonomic", "numeric"] = Field(
        description="Type of hierarchy governing the value levels"
    )

    levels: list[HierarchyLevel] = Field(
        min_length=2,
        description="Ordered hierarchy chain: levels[0] = finest, levels[-1] = coarsest before unknown",
    )
    original_level: int = Field(
        ge=0,
        description="Level index of the original fine-grained value (usually 0)",
    )
    target_level: int = Field(
        ge=0,
        description="Level index the unlearning method should retain",
    )

    source_modalities: list[str] = Field(
        default_factory=list,
        description="Modalities providing evidence for this association (e.g. ['text', 'image'])",
    )
    images: list[ImageRef] = Field(default_factory=list)
    textual_context: list[str] = Field(
        default_factory=list,
        description="Source text snippets supporting this association",
    )

    retain_attribute_names: list[str] = Field(
        default_factory=list,
        description="Other attributes of the same entity that should be retained after unlearning",
    )
    split: SplitInfo
    provenance: ProvenanceInfo

    @model_validator(mode="after")
    def _check_level_bounds(self) -> "AssociationRecord":
        n = len(self.levels)
        if self.original_level >= n:
            raise ValueError(
                f"original_level ({self.original_level}) must be < len(levels) ({n})"
            )
        if self.target_level >= n:
            raise ValueError(
                f"target_level ({self.target_level}) must be < len(levels) ({n})"
            )
        if self.original_level != 0:
            raise ValueError(
                f"original_level must be 0 (got {self.original_level})"
            )
        if self.target_level < self.original_level:
            raise ValueError(
                f"target_level ({self.target_level}) must be >= "
                f"original_level ({self.original_level})"
            )
        return self

    def fine_value(self) -> HierarchyLevel:
        """Return the finest (level 0) value."""
        return self.levels[0]

    def target_value(self) -> HierarchyLevel:
        """Return the target-level value."""
        return self.levels[self.target_level]

    def num_levels(self) -> int:
        return len(self.levels)
