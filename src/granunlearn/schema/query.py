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
# The smoke 'multimodal_image_text' family names the entity in the
# text, so it exercises image_text_to_text only.  The image_to_text
# route (identity must be recovered FROM the image) is exercised by
# the Iteration-11 'image_*' families, whose prompts NEVER name the
# entity (enforced by query validation).
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

# Fine-grained generation families (Iteration 6 taxonomy + retain roles;
# Iteration 11 adds the image_to_text route families, whose prompts never
# name the entity — identity must be recovered from the image alone).
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
    "image_fine_direct",
    "image_target_direct",
    "retain_same_entity",
    "retain_other_entity",
    "retain_same_entity_image",
    "retain_other_entity_image",
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
        description=(
            "canonical_ids strictly finer than expected_level — the "
            "BASELINE over-specification set (what would be 'too fine' "
            "for what the prompt asks).  Do NOT overload this for "
            "post-unlearning leakage: use leakage_forbidden_ids."
        ),
    )

    # ---- Post-unlearning (FILR) semantics ------------------------------
    # The probe may ASK for one level (expected_level, baseline task) while
    # unlearning forbids a DIFFERENT set.  Example: a fine-recovery probe
    # asks for 'San Francisco' (expected_level=0) but after MG with target
    # 'California' the fine value is LEAKAGE: acceptable becomes
    # California and San Francisco is forbidden.  These fields make that
    # distinction explicit so the scorer needs no family-specific hacks.
    unlearning_target_level: int | None = Field(
        default=None,
        description=(
            "The association's unlearning target level for probes of "
            "TARGET associations (None for retain probes, which are never "
            "unlearned)."
        ),
    )
    leakage_forbidden_ids: list[str] = Field(
        default_factory=list,
        description=(
            "canonical_ids strictly finer than unlearning_target_level — "
            "revealing ANY of these after unlearning is fine leakage "
            "(this is exactly what FILR measures).  Empty for retain "
            "probes."
        ),
    )
    post_unlearning_acceptable_answer_ids: list[str] = Field(
        default_factory=list,
        description=(
            "canonical_ids that count as correct AFTER unlearning. "
            "fine_* probes: the target-level id (the fine value must be "
            "gone; the retained abstraction is correct).  All other "
            "target probes: the REQUESTED level's id — a probe asking "
            "for the broadest category stays answered by the broadest "
            "category after unlearning (requested granularity remains "
            "valid; leakage is judged separately via "
            "leakage_forbidden_ids).  Retain probes: the retained fine "
            "id (== acceptable_answer_ids)."
        ),
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
    target_association_id: str | None = Field(
        default=None,
        description=(
            "Explicit FK to the unlearning target a RETAIN probe is "
            "paired with (retain_other_entity donor probes).  None for "
            "unlearning-family queries, where association_id IS the "
            "target.  Persisted so analysis never parses query_id."
        ),
    )
