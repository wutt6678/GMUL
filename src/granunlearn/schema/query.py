"""Query record — a single evaluation prompt derived from an association.

One association generates many queries across different routes and types.

This is the SINGLE query data contract for training and evaluation
(Blocker-1 fix, Iteration 6 review): every query generator emits this
record, and ``PredictionRecord`` scoring consumes its canonical hierarchy
ids (``acceptable_answer_ids`` / ``forbidden_descendant_ids``) directly.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

# Allowed cross-modal routes.
# NOTE: the current smoke 'multimodal_image_text' family names the entity
# in the text, so it exercises image_text_to_text only.  The image_to_text
# route (identity must be recovered FROM the image) is a later iteration;
# current M queries are NOT evidence for image->identity->attribute.
Route = Literal["text_to_text", "image_to_text", "image_text_to_text"]

# Query taxonomy (semantic intent; modality lives in `route`).
QueryType = Literal[
    "fine_direct",            # Ask for the exact fine-grained value
    "target_direct",          # Ask for the target-level value
    "open",                   # Open-ended question about the attribute
    "ancestor",               # Ask about a coarser ancestor
    "negation",               # Target-level answer against broader phrasing
    "retain_same_entity",     # Ask about a *different* attribute of the same entity
    "retain_other_entity",    # Ask about a completely different entity
]

# Fine-grained generation families (Iteration 6 taxonomy + retain roles).
QueryFamily = Literal[
    "fine_direct",
    "fine_descriptive",
    "fine_paraphrase",
    "fine_contrastive",
    "fine_contextual",
    "fine_verification",
    "granular_fine",
    "granular_intermediate",
    "granular_coarse",
    "negation_direct",
    "negation_correction",
    "negation_disambiguation",
    "multimodal_image_text",
    "retain_same_entity",
    "retain_other_entity",
]


class QueryRecord(BaseModel):
    """A single evaluation query linked to an association.

    ``association_id`` references the association being ASKED ABOUT (for
    retain queries this is the retained association, not the unlearning
    target).  Correctness is hierarchy-relative: ``acceptable_answer_ids``
    are the canonical ids that count as correct and
    ``forbidden_descendant_ids`` are all levels strictly finer than the
    expected one — revealing any of them is leakage (this is exactly what
    FILR consumes).
    """

    query_id: str = Field(description="Globally unique query identifier")
    association_id: str = Field(description="FK to the source AssociationRecord")
    route: Route = Field(description="Cross-modal route for this query")
    query_type: QueryType = Field(description="Semantic category of the query")
    family: QueryFamily | None = Field(
        default=None,
        description="Fine-grained generation family (Iteration 6 taxonomy)",
    )

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
    template_id: str | None = Field(
        default=None,
        description="Which paraphrase template produced this prompt",
    )
    expected_answer: str | None = Field(
        default=None,
        description="Convenience/debug: exact expected answer string",
    )
    adversarial: bool = Field(
        default=False,
        description=(
            "True for prompted-recovery probes (e.g. negation_correction, "
            "which quotes the forgotten fine value).  Such queries are "
            "EXCLUDED from the core FILR average: repeating the quoted "
            "value cannot be attributed to model memory."
        ),
    )
