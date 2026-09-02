"""Deterministic hierarchical scoring for reference-state evaluation.

Maps a raw model output to a hierarchy level of the queried association
and emits the canonical ``PredictionRecord``.  No LLM judge: matching is
exact-normalized substring matching against the association's level
values, finest match wins (an output containing the fine value reveals
the fine value — that IS leakage, regardless of surrounding text).

Metrics are computed per checkpoint over query groups:
* BASELINE accuracy  — matched id in ``acceptable_answer_ids``
* POST accuracy      — matched id in
                       ``post_unlearning_acceptable_answer_ids``
* LEAKAGE rate       — matched id in ``leakage_forbidden_ids``
* RETAIN accuracy    — baseline accuracy over retain_* probes
Core aggregates EXCLUDE adversarial probes (prompted recovery); the
adversarial slice is reported separately.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import re

from granunlearn.schema import AssociationRecord, PredictionRecord, QueryRecord


def _normalize(s: str) -> str:
    return " ".join(s.lower().replace("\u2013", "-").split())


# Negation cues checked on WORD BOUNDARIES within the 5 tokens preceding a
# value match.  Substring matching is deliberately avoided: it would
# false-positive on tokens like "Nottingham" (Iteration 7 scorer audit).
# Bare "no" is excluded — the audit showed it false-positives on answer
# tokens such as "NSW;".  The cue list is intentionally conservative;
# residual cases are surfaced by the committed scorer-decision audit
# rather than by silently adding more cues.
NEGATION_CUES = (
    "not", "never", "isn't", "wasn't", "aren't", "weren't",
    "doesn't", "didn't", "don't", "can't", "cannot", "won't",
    "is not", "was not", "are not", "does not", "did not",
    "rather than", "instead of",
)


def _prefix_tokens(text: str, match_start: int, n: int = 5) -> list[str]:
    """The last ``n`` tokens before the match within the CURRENT CLAUSE.

    The window never crosses sentence/clause punctuation (. ; ! ?) —
    otherwise 'not Sydney; it is NSW' would flag NSW as denied — and
    leading/trailing punctuation is stripped so 'NSW;' compares as 'nsw'.
    """
    prefix = text[:match_start]
    cut = max(prefix.rfind(sep) for sep in (".", ";", "!", "?"))
    if cut >= 0:
        prefix = prefix[cut + 1:]
    return [t.strip(".,;:!?()[]{}'\"")
            for t in prefix.split()[-n:]]


def _is_negated(text: str, match_start: int) -> bool:
    """True when a negation cue appears in the 5 tokens before the match.

    ALL cues are matched with word boundaries (Iteration 8 hardening):
    uniform semantics for single- and multiword cues, so no cue can
    ever fire inside another token.
    """
    prefix_words = _prefix_tokens(text, match_start)
    tail = " ".join(prefix_words)
    return any(re.search(r"\b" + re.escape(cue) + r"\b", tail)
               for cue in NEGATION_CUES)


def match_answer(
    raw_output: str, assoc: AssociationRecord
) -> tuple[int | None, str | None, list[str]]:
    """Match a raw output to the association's hierarchy.

    Returns ``(level_index, canonical_id, negated_values)`` — the FINEST
    level whose normalized value occurs in the output.  A match directly
    preceded by a negation cue is SKIPPED (the model denied that value)
    and recorded in ``negated_values``; coarser levels are still
    considered.  ``(None, None, [...])`` when no level value is present.
    """
    text = _normalize(raw_output)
    negated: list[str] = []
    if not text:
        return None, None, negated
    for lv in assoc.levels:  # levels ordered finest -> coarsest
        needle = _normalize(lv.value)
        pos = text.find(needle)
        while pos >= 0:
            if not _is_negated(text, pos):
                return lv.level, lv.canonical_id, negated
            negated.append(lv.value)
            pos = text.find(needle, pos + 1)
    return None, None, negated


def score_query(
    query: QueryRecord,
    assoc: AssociationRecord,
    raw_output: str,
    experiment_id: str,
    checkpoint_id: str,
) -> PredictionRecord:
    """Build the canonical PredictionRecord for one generated answer."""
    predicted_level, matched_id, negated = match_answer(raw_output, assoc)
    parsed = (assoc.levels[predicted_level].value
              if predicted_level is not None else None)

    is_correct_branch = (matched_id in query.acceptable_answer_ids
                         if matched_id else False)
    is_leakage = (matched_id in query.leakage_forbidden_ids
                  if matched_id else False)
    is_post_correct = (
        matched_id in query.post_unlearning_acceptable_answer_ids
        if matched_id else False)
    is_coarser = (predicted_level is not None
                  and query.expected_level is not None
                  and predicted_level > query.expected_level)

    return PredictionRecord(
        experiment_id=experiment_id,
        checkpoint_id=checkpoint_id,
        query_id=query.query_id,
        raw_output=raw_output,
        parsed_answer=parsed,
        matched_canonical_id=matched_id,
        predicted_level=predicted_level,
        is_correct_branch=is_correct_branch,
        is_finer_than_target=is_leakage,
        is_coarser_than_target=is_coarser,
        metadata={"post_acceptable_match": is_post_correct,
                  "family": query.family,
                  "adversarial": query.adversarial,
                  "negated_matches": negated},
    )


def _rate(flags: list[bool]) -> float | None:
    return round(sum(1 for f in flags if f) / len(flags), 4) if flags else None


def compute_metrics(
    predictions: list[PredictionRecord],
    queries: list[QueryRecord],
    split: str | None = None,
) -> dict[str, Any]:
    """Aggregate metrics per role/family slice.

    Core slices exclude adversarial probes; the adversarial slice is
    reported separately (prompted-recovery probes quote the forgotten
    fine value and cannot be attributed to model memory).

    ``split=None`` pools train/val/test; pass ``split='test'`` (etc.) for
    the held-out paraphrase metrics, which are reported SEPARATELY from
    the pooled numbers (Iteration 7 review: the paraphrase split exists
    to measure wording generalization, so it must be visible on its own).
    """
    by_id = {q.query_id: q for q in queries}
    rows = [(by_id[p.query_id], p) for p in predictions
            if p.query_id in by_id
            and (split is None or by_id[p.query_id].split == split)]

    def select(pred, role=None, families=None, include_adversarial=False):
        out = []
        for q, p in rows:
            if not include_adversarial and q.adversarial:
                continue
            if role == "target" and (q.family or "").startswith("retain_"):
                continue
            if role == "retain" and not (q.family or "").startswith("retain_"):
                continue
            if families and q.family not in families:
                continue
            out.append((q, p))
        return out

    def block(sel):
        if not sel:
            return {}
        return {
            "num_queries": len(sel),
            "baseline_accuracy": _rate(
                [p.is_correct_branch for _, p in sel]),
            "post_unlearning_accuracy": _rate(
                [bool(p.metadata.get("post_acceptable_match"))
                 for _, p in sel]),
            "leakage_rate": _rate(
                [bool(p.is_finer_than_target) for _, p in sel]),
            "unmatched_rate": _rate(
                [p.matched_canonical_id is None for _, p in sel]),
        }

    fine_families = [f for f in {q.family for q, _ in rows}
                     if f and f.startswith("fine_")]
    metrics: dict[str, Any] = {
        "num_queries": len(rows),
        "all_core": block(select(rows)),
        "target_core": block(select(rows, role="target")),
        "fine_recovery": block(select(rows, families=set(fine_families))),
        "granular": block(select(rows, families={
            "granular_fine", "granular_intermediate", "granular_coarse"})),
        "negation": block(select(rows, families={
            "negation_direct", "negation_disambiguation"})),
        "multimodal": block(
            select(rows, families={"multimodal_image_text"})),
        "image_route": block(
            select(rows, families={"image_fine_direct",
                                   "image_target_direct"})),
        "retain_same_entity": block(
            select(rows, families={"retain_same_entity"})),
        "retain_other_entity": block(
            select(rows, families={"retain_other_entity"})),
        "retain_same_entity_image": block(
            select(rows, families={"retain_same_entity_image"})),
        "retain_other_entity_image": block(
            select(rows, families={"retain_other_entity_image"})),
        "adversarial_only": block(
            [(q, p) for q, p in rows if q.adversarial]),
    }
    return metrics


def separation_gate(
    metrics_by_state: dict[str, dict[str, Any]],
    min_gap: float = 0.15,
    min_retain_accuracy: float = 0.5,
) -> tuple[bool, list[str]]:
    """The Iteration 7 hard gate: MF != MG != MN, behaviorally.

    Requirements (all over CORE slices, adversarial excluded):
    1. fine_recovery baseline accuracy: MF > MG and MF > MN by >= min_gap
       (MF retains the fine facts; MG/MN do not);
    2. target_core post-unlearning accuracy: MG > MN by >= min_gap
       (MG retains the target abstraction; MN has no target knowledge);
    3. retain accuracy (same-entity + other-entity) >= min_retain_accuracy
       for every state (retain facts survive in all reference states).

    Returns ``(passed, reasons)``.
    """
    reasons: list[str] = []
    required = ("MF", "MG", "MN")
    for state in required:
        if state not in metrics_by_state:
            reasons.append(f"missing state metrics: {state}")
    if reasons:
        return False, reasons

    def m(state, slice_, key):
        v = metrics_by_state[state].get(slice_, {}).get(key)
        return v if isinstance(v, (int, float)) else None

    mf_fine, mg_fine, mn_fine = (
        m(s, "fine_recovery", "baseline_accuracy") for s in required)
    if None in (mf_fine, mg_fine, mn_fine):
        reasons.append("fine_recovery baseline_accuracy unavailable")
    else:
        if mf_fine - mg_fine < min_gap:
            reasons.append(
                f"MF fine recovery {mf_fine} not > MG {mg_fine} "
                f"by {min_gap}")
        if mf_fine - mn_fine < min_gap:
            reasons.append(
                f"MF fine recovery {mf_fine} not > MN {mn_fine} "
                f"by {min_gap}")

    mg_post, mn_post = (
        m(s, "target_core", "post_unlearning_accuracy")
        for s in ("MG", "MN"))
    if None in (mg_post, mn_post):
        reasons.append("target_core post_unlearning_accuracy unavailable")
    elif mg_post - mn_post < min_gap:
        reasons.append(
            f"MG post accuracy {mg_post} not > MN {mn_post} by {min_gap}")

    for state in required:
        for slice_ in ("retain_same_entity", "retain_other_entity"):
            acc = m(state, slice_, "baseline_accuracy")
            if acc is None:
                reasons.append(f"{state}/{slice_}: no retain metrics")
            elif acc < min_retain_accuracy:
                reasons.append(
                    f"{state}/{slice_}: retain accuracy {acc} < "
                    f"{min_retain_accuracy}")
    return len(reasons) == 0, reasons
