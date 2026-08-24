"""Query record — a single evaluation prompt derived from an association.

One association generates many queries across different routes and types.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

# Allowed cross-modal routes
Route = Literal["text_to_text", "image_to_text", "image_text_to_text"]

# Query taxonomy
QueryType = Literal[
    "fine_direct",            # Ask for the exact fine-grained value
    "target_direct",          # Ask for the target-level value
    "open",                   # Open-ended question about the attribute
    "ancestor",               # Ask about a coarser ancestor
    "retain_same_entity",     # Ask about a *different* attribute of the same entity
    "retain_other_entity",    # Ask about a completely different entity
]


class QueryRecord(BaseModel):
    """A single evaluation query linked to an association."""

    query_id: str = Field(description="Globally unique query identifier")
    association_id: str = Field(description="FK to the source AssociationRecord")
    route: Route = Field(description="Cross-modal route for this query")
    query_type: QueryType = Field(description="Semantic category of the query")

    image_ids: list[str] = Field(
        default_factory=list,
        description="Image IDs to include in the prompt (empty for T→T)",
    )
    prompt: str = Field(description="Full natural-language prompt sent to the model")

    expected_level: int | None = Field(
        default=None,
        description="Hierarchy level the correct answer should match (None for retain/other queries)",
    )
    acceptable_answer_ids: list[str] = Field(
        default_factory=list,
        description="canonical_ids that count as correct",
    )
    forbidden_descendant_ids: list[str] = Field(
        default_factory=list,
        description="canonical_ids finer than target — revealing these = leakage",
    )

    split: str = Field(pattern=r"^(train|val|test)$")
    paraphrase_group_id: str | None = Field(
        default=None,
        description="Groups paraphrases together; train/test paraphrases must be disjoint",
    )
