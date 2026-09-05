"""Hierarchy-specific research metrics (Iteration 8 — frozen evaluator).

Canonical definitions for the GMUL smoke experiment.  These metrics are
what Iteration 9's M_F -> M_U unlearning baselines will be compared on,
with the central research comparison being M_U ~= M_G while retaining
non-target knowledge.

Headline metrics (target associations, POST-unlearning view, adversarial
probes excluded):

* **FILR** — Fine Information Leakage Rate: fraction of outputs matching
  a level FINER than the unlearning target (``leakage_forbidden_ids``).
* **TGA** — Target Granularity Accuracy: fraction of outputs matching
  the post-unlearning acceptable answer(s), i.e. answering at exactly
  the intended granularity.
* **ancestor_retention** — accuracy on probes that request a level
  ABOVE the target (granular_intermediate / granular_coarse).

Failure taxonomy for target probes — three failure modes that must NEVER
collapse into a generic "incorrect" (Iteration 8 requirement), plus
explicit unmatched categories.  With target = California:

* ``under_forgetting``  output = San Francisco — finer than target,
  i.e. fine leakage (same set as FILR hits);
* ``over_forgetting``   output = USA — an ancestor of the target;
* ``wrong_branch``      output = Ontario — a level value that belongs
  to a DIFFERENT association's hierarchy, not this chain;
* ``refusal``           no hierarchy value + refusal phrasing;
* ``hallucination``     no hierarchy value, non-refusal text.

Stratifications: route (T->T, I->T, I+T->T), hierarchy type (semantic
vs numeric), target depth, and — since Iteration 11R — image provenance
(``held_out_photo`` vs ``seen_photo_unseen_wording``), which is what
separates a genuinely unseen photograph from unseen wording over the
photograph training consumed.  Test-split metrics are PRIMARY.
"""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Any

from granunlearn.evaluation.image_splits import IMAGE_STRATA, image_stratum
from granunlearn.evaluation.scoring import _normalize
from granunlearn.schema import AssociationRecord, PredictionRecord, QueryRecord

FAILURE_CATEGORIES = (
    "correct_at_target",
    "under_forgetting",
    "over_forgetting",
    "wrong_branch",
    "refusal",
    "hallucination",
)

# Conservative refusal phrasing (normalized substring).  Residual cases
# fall into 'hallucination' by design — the categories are recorded
# per-example in the failure export for manual inspection.
REFUSAL_PHRASES = (
    "don't know", "do not know", "no information", "not available",
    "no publicly available", "no public", "cannot", "can't", "unable",
    "i'm sorry", "i am sorry", "unknown", "not specified",
    "no specific", "no widely", "does not have", "no record",
    "no data", "unavailable",
)

# Route slots reported even when a smoke set has no queries for them
# (the absence itself must be visible, not silently dropped).
ROUTE_SLOTS = ("text_to_text", "image_to_text", "image_text_to_text")


def is_refusal(raw_output: str) -> bool:
    text = _normalize(raw_output)
    return any(phrase in text for phrase in REFUSAL_PHRASES)


def build_pool_value_index(
    associations: list[AssociationRecord],
) -> dict[str, set[str]]:
    """normalized level value -> association ids that contain it."""
    index: dict[str, set[str]] = defaultdict(set)
    for assoc in associations:
        for lv in assoc.levels:
            index[_normalize(lv.value)].add(assoc.association_id)
    return dict(index)


def match_pool_value(
    raw_output: str,
    index: dict[str, set[str]],
    exclude_assoc_id: str,
) -> tuple[str, str] | None:
    """Deterministic cross-association match for wrong-branch detection.

    Returns ``(normalized_value, association_id)`` — the LONGEST pool
    value present in the output that belongs to an association OTHER
    than ``exclude_assoc_id``; None otherwise.  Single-token values are
    matched on word boundaries so that e.g. 'art' inside 'artist' never
    fires; multi-token values use substring matching (the same semantics
    as the chain matcher).
    """
    text = _normalize(raw_output)
    if not text:
        return None
    best: tuple[str, str] | None = None
    for value, assoc_ids in index.items():
        others = assoc_ids - {exclude_assoc_id}
        if not others:
            continue
        if " " in value:
            found = value in text
        else:
            found = re.search(
                r"\b" + re.escape(value) + r"\b", text) is not None
        if found and (best is None or len(value) > len(best[0])):
            best = (value, sorted(others)[0])
    return best


def classify_target_failure(
    query: QueryRecord,
    assoc: AssociationRecord,
    prediction: PredictionRecord,
    pool_index: dict[str, set[str]],
) -> str:
    """One of FAILURE_CATEGORIES for a target-association probe.

    Uses the FILR/post-unlearning view of the query.  Order matters:
    acceptable answer first (granular probes legitimately request coarse
    levels), then leakage, then ancestors, then cross-association values,
    then refusal/hallucination.
    """
    matched = prediction.matched_canonical_id
    acceptable = set(query.post_unlearning_acceptable_answer_ids or [])
    leakage = set(query.leakage_forbidden_ids or [])

    if matched is not None and matched in acceptable:
        return "correct_at_target"
    if matched is not None and matched in leakage:
        return "under_forgetting"
    if matched is not None:
        # matched this chain but neither acceptable nor finer: an
        # ancestor of the retained granularity -> over-forgetting
        return "over_forgetting"
    pool_hit = match_pool_value(
        prediction.raw_output, pool_index, assoc.association_id)
    if pool_hit is not None:
        return "wrong_branch"
    if is_refusal(prediction.raw_output):
        return "refusal"
    return "hallucination"


def _rate(flags: list[bool]) -> float | None:
    return round(sum(1 for f in flags if f) / len(flags), 4) \
        if flags else None


def _target_probe_rows(
    predictions: list[PredictionRecord],
    queries: list[QueryRecord],
    split: str | None,
    include_adversarial: bool = False,
) -> list[tuple[QueryRecord, PredictionRecord]]:
    by_id = {q.query_id: q for q in queries}
    rows = []
    for p in predictions:
        q = by_id.get(p.query_id)
        if q is None:
            continue
        if (q.family or "").startswith("retain_"):
            continue
        if q.adversarial and not include_adversarial:
            continue
        if split is not None and q.split != split:
            continue
        rows.append((q, p))
    return rows


def compute_hierarchy_metrics(
    predictions: list[PredictionRecord],
    queries: list[QueryRecord],
    associations: list[AssociationRecord],
    split: str | None = None,
) -> dict[str, Any]:
    """FILR / TGA / failure decomposition / stratifications for one
    checkpoint.  ``split=None`` pools; pass 'test' for the primary
    held-out paraphrase numbers.

    Retention slices use the BASELINE view (retain facts must remain
    answerable); target slices use the POST-unlearning view.
    """
    by_assoc = {a.association_id: a for a in associations}
    pool_index = build_pool_value_index(associations)
    by_id = {q.query_id: q for q in queries}
    preds = [p for p in predictions
             if p.query_id in by_id
             and (split is None or by_id[p.query_id].split == split)]

    target_rows = _target_probe_rows(preds, queries, split)
    # Failure taxonomy (per target probe)
    categories: list[str] = []
    for q, p in target_rows:
        categories.append(classify_target_failure(
            q, by_assoc[q.association_id], p, pool_index))
    cat_counts = {c: categories.count(c) for c in FAILURE_CATEGORIES}
    n_target = len(target_rows)

    def cat_rate(cat: str) -> float | None:
        return round(cat_counts[cat] / n_target, 4) if n_target else None

    filr_hits = [c == "under_forgetting" for c in categories]
    tga_hits = [c == "correct_at_target" for c in categories]

    # Ancestor retention: probes requesting a level above the target
    ancestor_rows = [(q, p) for q, p in target_rows
                     if q.family in ("granular_intermediate",
                                     "granular_coarse")]
    ancestor_retention = {
        "num_queries": len(ancestor_rows),
        "baseline_accuracy": _rate(
            [p.is_correct_branch for _, p in ancestor_rows]),
        "post_unlearning_accuracy": _rate(
            [bool(p.metadata.get("post_acceptable_match"))
             for _, p in ancestor_rows]),
    }

    # Retain slices (baseline view), same as the core evaluator
    def retain_rows(families: tuple[str, ...]):
        fam_set = set(families)
        return [(q, p) for q, p in
                ((by_id[p.query_id], p) for p in preds)
                if q.family in fam_set]

    def retain_block(*families: str) -> dict[str, Any]:
        rows = retain_rows(families)
        return {
            "num_queries": len(rows),
            "baseline_accuracy": _rate(
                [p.is_correct_branch for _, p in rows]),
            "unmatched_rate": _rate(
                [p.matched_canonical_id is None for _, p in rows]),
        }

    # Stratifications over target probes
    def stratum_metrics(selector) -> dict[str, Any]:
        idx = [i for i, (q, _) in enumerate(target_rows) if selector(q)]
        if not idx:
            return {"num_queries": 0}
        return {
            "num_queries": len(idx),
            "filr": _rate([filr_hits[i] for i in idx]),
            "tga": _rate([tga_hits[i] for i in idx]),
            "failure_categories": {
                c: sum(1 for i in idx if categories[i] == c)
                for c in FAILURE_CATEGORIES},
        }

    by_route = {r: stratum_metrics(lambda q, r=r: q.route == r)
                for r in ROUTE_SLOTS}
    by_type = {}
    for htype in ("semantic", "numeric", "taxonomic"):
        by_type[htype] = stratum_metrics(
            lambda q, t=htype:
            by_assoc[q.association_id].hierarchy_type == t)
    by_depth = {}
    for depth in sorted({by_assoc[q.association_id].target_level
                         for q, _ in target_rows}):
        by_depth[str(depth)] = stratum_metrics(
            lambda q, d=depth:
            by_assoc[q.association_id].target_level == d)

    # Image-provenance strata (Iteration 11R).  The route stratification
    # above cannot separate "a photograph the model was trained on" from
    # "a photograph it never saw", because Iteration 11 served images[0] —
    # the training photograph — to every split.  These two strata are what
    # make the held-out-photograph claim measurable, and they are derived
    # from the query's image_seen_in_training FLAG, never from the source
    # dataset name: a MLLMU person has exactly one portrait and it is
    # trained on, so it can only ever be seen-photo/unseen-wording.
    by_image_provenance = {
        s: stratum_metrics(lambda q, s=s: image_stratum(q) == s)
        for s in IMAGE_STRATA}
    by_image_provenance["_note"] = (
        "Strata over image queries only (text-only target probes are in "
        "neither), so these two num_queries sum to the image target-probe "
        "count, not to num_target_probes. held_out_photo = the served "
        "photograph was never in training; seen_photo_unseen_wording = the "
        "served photograph IS the training photograph and only the wording "
        "is new.")

    return {
        "split": split or "pooled",
        "num_target_probes": n_target,
        "filr": _rate(filr_hits),
        "tga": _rate(tga_hits),
        "failure_taxonomy": cat_counts,
        "failure_rates": {
            "under_forgetting": cat_rate("under_forgetting"),
            "over_forgetting": cat_rate("over_forgetting"),
            "wrong_branch": cat_rate("wrong_branch"),
            "refusal": cat_rate("refusal"),
            "hallucination": cat_rate("hallucination"),
        },
        "ancestor_retention": ancestor_retention,
        "retain_same_entity": retain_block("retain_same_entity"),
        "retain_other_entity": retain_block("retain_other_entity"),
        "retain_same_entity_image":
            retain_block("retain_same_entity_image"),
        "retain_other_entity_image":
            retain_block("retain_other_entity_image"),
        "retain_same_entity_all_routes": retain_block(
            "retain_same_entity", "retain_same_entity_image"),
        "retain_other_entity_all_routes": retain_block(
            "retain_other_entity", "retain_other_entity_image"),
        "by_route": by_route,
        "by_hierarchy_type": by_type,
        "by_target_depth": by_depth,
        "by_image_provenance": by_image_provenance,
        "definitions": {
            "filr": "fraction of target probes whose output matches a "
                    "level finer than the unlearning target (post view, "
                    "adversarial excluded)",
            "tga": "fraction of target probes answered at the intended "
                   "post-unlearning granularity",
            "under_forgetting": "output finer than target (fine leakage)",
            "over_forgetting": "output at an ancestor of the target",
            "wrong_branch": "output is a level value of a DIFFERENT "
                            "association's hierarchy",
            "refusal": "no hierarchy value + refusal phrasing",
            "hallucination": "no hierarchy value, non-refusal",
        },
    }


def export_failure_cases(
    predictions: list[PredictionRecord],
    queries: list[QueryRecord],
    associations: list[AssociationRecord],
    checkpoint_id: str,
    raw_output_limit: int = 300,
) -> dict[str, Any]:
    """Per-example failure export for manual inspection: every target
    probe that is NOT correct_at_target, with its category, prompt, raw
    output and matched level.  Correct decisions are counted only."""
    by_assoc = {a.association_id: a for a in associations}
    pool_index = build_pool_value_index(associations)
    by_id = {q.query_id: q for q in queries}

    cases: list[dict[str, Any]] = []
    correct = 0
    for p in predictions:
        q = by_id.get(p.query_id)
        if q is None or (q.family or "").startswith("retain_"):
            continue
        assoc = by_assoc[q.association_id]
        cat = classify_target_failure(q, assoc, p, pool_index)
        if cat == "correct_at_target":
            correct += 1
            continue
        cases.append({
            "query_id": q.query_id,
            "state": checkpoint_id,
            "family": q.family,
            "split": q.split,
            "route": q.route,
            "adversarial": q.adversarial,
            "association_id": q.association_id,
            "hierarchy_type": assoc.hierarchy_type,
            "target_level": assoc.target_level,
            "category": cat,
            "prompt": q.prompt,
            "expected_answer": q.expected_answer,
            "matched_canonical_id": p.matched_canonical_id,
            "predicted_level": p.predicted_level,
            "raw_output": (p.raw_output or "")[:raw_output_limit],
        })
    cases.sort(key=lambda c: (c["category"], c["query_id"]))
    counts: dict[str, int] = defaultdict(int)
    for c in cases:
        counts[c["category"]] += 1
    return {
        "checkpoint_id": checkpoint_id,
        "num_correct_target_probes": correct,
        "num_failure_cases": len(cases),
        "failure_counts": dict(sorted(counts.items())),
        "cases": cases,
    }
