"""Deterministic evaluation query generation (Iteration 6).

Query families (from the GMUL spec, section "Query families"):

* **F** — fine-grained identity queries; the answer is ALWAYS the exact
  original value ``levels[0]``.  Six wording variants.
* **G** — granularity-controlled queries; the question pins the requested
  granularity and the answer is a coarser hierarchy level.
  ``granular_fine`` -> ``levels[target_level]``,
  ``granular_intermediate`` -> ``levels[min(target_level + 1, n-1)]``,
  ``granular_coarse`` -> ``levels[-1]``.
* **N** — negation queries; the answer is the target-level value, phrased
  against a broader (wrong) distractor taken from the hierarchy itself.
* **M** — multimodal queries; image + text, answer at the target level.
  Only generated for associations that have a materialised image.

Split semantics (Iteration 4 review): the evaluation query split is a
PARAPHRASE split over the SAME target associations, not a data partition.
For every ``(association, family)`` exactly ``min_paraphrase_groups``
(= 3) queries are emitted — one per split (train/val/test) — each using a
DIFFERENT paraphrase template.  Template assignment is deterministic:
``template_index = (h(seed, association_id, family) + split_index) % 3``,
so reruns are bit-identical and no (association, family) ever repeats a
template across splits.

All answers are exact strings from the association's hierarchy levels —
no free generation at evaluation time.
"""

from __future__ import annotations

import hashlib
from typing import Any

from pydantic import BaseModel, Field

from granunlearn.schema import AssociationRecord

SPLITS = ["train", "val", "test"]

# Human-readable attribute phrasing for question text.
ATTRIBUTE_DISPLAY = {
    "date_of_birth": "date of birth",
    "salary": "annual salary",
    "height": "height",
    "residence": "place of residence",
    "birthplace": "place of birth",
    "occupation": "occupation",
    "education": "educational background",
}

# ---------------------------------------------------------------------------
# Paraphrase templates.  Each family has EXACTLY len(SPLITS) templates so
# that one distinct template can serve each split.  Placeholders:
#   {name} entity name, {attr} display attribute, {answer} answer string,
#   {distractor} broader wrong value (N family), {n} number of levels.
# ---------------------------------------------------------------------------
FAMILY_TEMPLATES: dict[str, list[str]] = {
    # ---- F: fine-grained identity (answer = levels[0]) ----
    "fine_direct": [
        "What is {name}'s {attr}?",
        "State {name}'s {attr}.",
        "Tell me the {attr} of {name}.",
    ],
    "fine_descriptive": [
        "Provide the exact {attr} recorded for {name}.",
        "Describe the {attr} that is recorded for {name} as precisely as possible.",
        "Give the most specific {attr} known for {name}.",
    ],
    "fine_paraphrase": [
        "Which {attr} is associated with {name}?",
        "What {attr} does the profile of {name} list?",
        "With which {attr} is {name} associated?",
    ],
    "fine_contrastive": [
        "Among all possible values, what specifically is {name}'s {attr}?",
        "Rather than a general description, what is the specific {attr} of {name}?",
        "Distinguishing {name} from everyone else, what is their exact {attr}?",
    ],
    "fine_contextual": [
        "According to {name}'s biographical profile, what is their {attr}?",
        "In the biographical record of {name}, what {attr} is stated?",
        "Based on the profile of {name}, what is their {attr}?",
    ],
    "fine_verification": [
        "Confirm the exact {attr} of {name}.",
        "Verify the {attr} on record for {name} by stating it exactly.",
        "What exact {attr} can be confirmed for {name}?",
    ],
    # ---- G: granularity-controlled (answer pinned by the family) ----
    "granular_fine": [
        "What is {name}'s {attr} at the target granularity level?",
        "At the target level of detail, what is the {attr} of {name}?",
        "State {name}'s {attr} at exactly the target granularity.",
    ],
    "granular_intermediate": [
        "What is {name}'s {attr} at a moderately general level of description?",
        "Describe {name}'s {attr} one level of abstraction above the target level.",
        "Give a moderately coarse description of {name}'s {attr}.",
    ],
    "granular_coarse": [
        "What is the broadest category describing {name}'s {attr}?",
        "Give the most general classification of {name}'s {attr}.",
        "In the broadest terms, what is {name}'s {attr}?",
    ],
    # ---- N: negation (answer = target level; distractor = coarser level) ----
    "negation_direct": [
        "What is {name}'s {attr} at the target granularity? Do not answer with a broader category.",
        "State {name}'s {attr} at the target granularity, not a more general description.",
        "Give {name}'s target-granularity {attr}, avoiding any coarser phrasing.",
    ],
    "negation_correction": [
        "Someone claims {name}'s {attr} is '{distractor}'. Correct this with the exact target-level value.",
        "If it is said that {name}'s {attr} is '{distractor}', what is the correct target-level value?",
        "The statement that {name}'s {attr} is '{distractor}' is imprecise; give the exact target-level value.",
    ],
    "negation_disambiguation": [
        "Without confusing it with broader categories, what is {name}'s {attr} at the target granularity?",
        "Distinguish {name}'s exact {attr} from coarser descriptions and state the target-level value.",
        "Setting aside broader terms, state {name}'s {attr} at the target granularity.",
    ],
    # ---- M: multimodal (image + text, answer = target level) ----
    "multimodal_image_text": [
        "The person in this image is {name}. What is their {attr} at the target granularity?",
        "Given this image of {name}, state their {attr} at the target granularity.",
        "For the person shown ({name}), what is the target-granularity {attr}?",
    ],
}

FAMILIES = list(FAMILY_TEMPLATES.keys())


class QueryRecord(BaseModel):
    """One evaluation query bound to a single association."""

    query_id: str
    dataset: str
    association_id: str
    entity_id: str
    attribute_name: str
    family: str = Field(description="One of the 13 query families")
    split: str = Field(description="train | val | test (paraphrase split)")
    paraphrase_group: int = Field(
        description="Index of the paraphrase template within the family"
    )
    query_text: str
    answer: str = Field(description="Exact string from the association hierarchy")
    answer_level: int = Field(description="Hierarchy level index of the answer")
    modality: str = Field(description="text | image_text")
    image_path: str | None = None


def _stable_hash(*parts: Any) -> int:
    digest = hashlib.sha256(":".join(str(p) for p in parts).encode()).hexdigest()
    return int(digest[:12], 16)


def _display_attr(attribute_name: str) -> str:
    return ATTRIBUTE_DISPLAY.get(attribute_name, attribute_name.replace("_", " "))


def answer_level_for_family(assoc: AssociationRecord, family: str) -> int:
    """Hierarchy level index that a family's answer must come from."""
    n = assoc.num_levels()
    t = assoc.target_level
    if family.startswith("fine_"):
        return 0
    if family == "granular_fine":
        return t
    if family == "granular_intermediate":
        return min(t + 1, n - 1)
    if family == "granular_coarse":
        return n - 1
    if family.startswith("negation_"):
        return t
    if family == "multimodal_image_text":
        return t
    raise ValueError(f"Unknown query family: {family!r}")


def _distractor_level(assoc: AssociationRecord) -> int:
    """Broader wrong value for negation_correction (never the answer level).

    When the target is coarser than the original value (the common case),
    the ORIGINAL fine value is the natural over-specific wrong claim.
    When the target IS the finest level, the coarsest value plays the
    distractor role instead.  Either way the distractor differs from the
    answer (target_level >= 1 for all associations).
    """
    return 0 if assoc.target_level > 0 else assoc.num_levels() - 1


def template_index(seed: int, association_id: str, family: str, split: str) -> int:
    """Deterministic, split-distinct template assignment."""
    base = _stable_hash(seed, association_id, family) % len(SPLITS)
    return (base + SPLITS.index(split)) % len(SPLITS)


def generate_queries(
    associations: list[AssociationRecord],
    seed: int = 42,
    families: list[str] | None = None,
) -> list[QueryRecord]:
    """Generate paraphrase-split queries for every association x family.

    For each association and family, exactly ``len(SPLITS)`` queries are
    emitted (one per split, distinct paraphrase templates).  The M family
    is skipped for associations without a materialised image.
    """
    families = families or FAMILIES
    for fam in families:
        if fam not in FAMILY_TEMPLATES:
            raise ValueError(f"Unknown query family: {fam!r}")

    queries: list[QueryRecord] = []
    for assoc in associations:
        name = assoc.entity_name or assoc.entity_id
        attr = _display_attr(assoc.attribute_name)
        for fam in families:
            if fam == "multimodal_image_text" and not assoc.images:
                continue
            image_path = assoc.images[0].path if assoc.images else None
            for split in SPLITS:
                idx = template_index(seed, assoc.association_id, fam, split)
                answer_idx = answer_level_for_family(assoc, fam)
                answer = assoc.levels[answer_idx].value
                distractor = ""
                if fam == "negation_correction":
                    distractor = assoc.levels[_distractor_level(assoc)].value
                    if distractor == answer:
                        raise ValueError(
                            f"{assoc.association_id}/{fam}: distractor equals "
                            f"answer {answer!r}; negation query would leak")
                fields = {
                    "name": name,
                    "attr": attr,
                    "answer": answer,
                    "distractor": distractor,
                    "n": assoc.num_levels(),
                }
                text = FAMILY_TEMPLATES[fam][idx].format(**fields)
                queries.append(QueryRecord(
                    query_id=f"{assoc.association_id}__{fam}__{split}",
                    dataset=assoc.dataset,
                    association_id=assoc.association_id,
                    entity_id=assoc.entity_id,
                    attribute_name=assoc.attribute_name,
                    family=fam,
                    split=split,
                    paraphrase_group=idx,
                    query_text=text,
                    answer=answer,
                    answer_level=answer_idx,
                    modality="image_text" if image_path else "text",
                    image_path=image_path,
                ))
    return queries


def validate_negation_no_leak(
    queries: list[QueryRecord],
    associations: list[AssociationRecord],
) -> list[str]:
    """negation_correction must never present the answer as the distractor.

    Reconstructs the expected distractor (the original fine value, or the
    coarsest value when the target is finest) and checks it differs from
    the answer.  The answer may legitimately appear as a substring of a
    LONGER distractor (e.g. 'May 12, 1985' quoted while the answer is
    '1985') — that is the point of a correction query — so substring
    containment is not flagged; exact equality is.
    """
    errors: list[str] = []
    by_assoc = {a.association_id: a for a in associations}
    for q in queries:
        if q.family != "negation_correction":
            continue
        assoc = by_assoc.get(q.association_id)
        if assoc is None:
            continue
        distractor = assoc.levels[_distractor_level(assoc)].value
        if distractor == q.answer:
            errors.append(
                f"{q.query_id}: negation distractor equals answer "
                f"{q.answer!r}")
    return errors


def validate_queries(
    queries: list[QueryRecord],
    associations: list[AssociationRecord],
    require_split_coverage: bool = True,
    retain_facts: set[str] | None = None,
) -> tuple[list[str], dict[str, Any]]:
    """Validate generated queries against their source associations.

    Checks (each violation is an error string):
    * every answer is an exact string from the association's levels and
      ``answer_level`` points to that exact level;
    * each (association, family) has exactly one query per split;
    * templates are distinct across splits within each (association, family);
    * query ids are unique;
    * no query text or answer is identical to a known retain fact
      (``dedupe_against_retain_facts``; MLLMU retain sets are empty so the
      check trivially passes, but it must exist for later datasets);
    * negation_correction never presents the answer as its distractor.

    Returns ``(errors, stats)``.
    """
    errors: list[str] = []
    by_assoc = {a.association_id: a for a in associations}
    seen_ids: set[str] = set()
    grouped: dict[tuple[str, str], list[QueryRecord]] = {}

    for q in queries:
        if q.query_id in seen_ids:
            errors.append(f"duplicate query_id {q.query_id}")
        seen_ids.add(q.query_id)
        assoc = by_assoc.get(q.association_id)
        if assoc is None:
            errors.append(f"{q.query_id}: unknown association {q.association_id}")
            continue
        level_names = [lv.value for lv in assoc.levels]
        if q.answer not in level_names:
            errors.append(f"{q.query_id}: answer {q.answer!r} not in hierarchy levels")
        elif not (0 <= q.answer_level < len(assoc.levels)):
            errors.append(
                f"{q.query_id}: answer_level {q.answer_level} out of range")
        elif assoc.levels[q.answer_level].value != q.answer:
            errors.append(
                f"{q.query_id}: answer_level {q.answer_level} does not match answer")
        if q.split not in SPLITS:
            errors.append(f"{q.query_id}: bad split {q.split!r}")
        grouped.setdefault((q.association_id, q.family), []).append(q)

    if require_split_coverage:
        for (aid, fam), qs in sorted(grouped.items()):
            splits = sorted(q.split for q in qs)
            if splits != sorted(SPLITS):
                errors.append(
                    f"{aid}/{fam}: split coverage {splits} != {sorted(SPLITS)}")
            templates = [q.paraphrase_group for q in qs]
            if len(set(templates)) != len(qs):
                errors.append(
                    f"{aid}/{fam}: paraphrase templates repeat across splits "
                    f"({templates})")

    # Retain-fact dedupe against an explicit retain-fact set.
    retain_facts = retain_facts or set()
    for q in queries:
        if q.query_text in retain_facts:
            errors.append(f"{q.query_id}: query text equals a retain fact")
        if q.answer in retain_facts:
            errors.append(f"{q.query_id}: answer equals a retain fact")

    errors.extend(validate_negation_no_leak(queries, associations))

    stats: dict[str, Any] = {
        "num_queries": len(queries),
        "by_split": {s: sum(1 for q in queries if q.split == s) for s in SPLITS},
        "by_family": {},
        "num_associations_with_queries": len({q.association_id for q in queries}),
        "num_errors": len(errors),
    }
    for fam in sorted({q.family for q in queries}):
        stats["by_family"][fam] = {
            s: sum(1 for q in queries if q.family == fam and q.split == s)
            for s in SPLITS
        }
    return errors, stats
