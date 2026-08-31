"""GMUL proxy metrics + official SALMUBench evaluation (Iteration 10R3).

Two separate metric families:

1. **GMUL proxy metrics** (``compute_gmul_proxy_metrics``):
   In-house CLIP embedding proxies for the SALMUBench concepts.
   These are NOT the official SALMUBench protocol — they use our
   own probe construction and similarity computation.  Prefixed
   ``gmul_proxy_*`` to avoid confusion with the released metrics.

2. **Official SALMUBench metrics** (``compute_official_salmubench``):
   Loads the released evaluation data (forget gallery, holdout
   associations, retain synth, etc.) from the SALMUBench dataset
   and computes the canonical metrics: RetFail (MRR over 2,001-
   caption gallery), CoreAssoc (both name-value token orders),
   ACS (coherence classifier), IntraIdSim / InterIdSim (released
   holdout splits), and general utility (ImageNet / DataComp).
"""

from __future__ import annotations

from typing import Any


# ── GMUL proxy metrics ────────────────────────────────────────────

def compute_gmul_proxy_metrics(
    target_results: list[dict[str, Any]],
    same_entity_results: list[dict[str, Any]],
    other_entity_results: list[dict[str, Any]],
) -> dict[str, float | None]:
    """GMUL in-house CLIP-embedding proxies for SALMUBench concepts.

    All metric names carry the ``gmul_proxy_`` prefix to distinguish
    them from the official SALMUBench protocol.

    10R4: every statistic is association-weighted — variants within a
    (identity_id, attribute) association are macro-averaged first, so
    associations with more images/captions get no extra weight.

    Parameters
    ----------
    target_results : list[dict]
        Per-probe results where ``is_target_attr=True``.
    same_entity_results : list[dict]
        Per-probe results for same-entity retain.
    other_entity_results : list[dict]
        Per-probe results for other-entity retain.

    Returns
    -------
    dict mapping ``gmul_proxy_*`` metric names to float (or None).
    """
    from granunlearn.salmu.embedding_metrics import (
        association_level_results,
        preference_flags,
    )

    # --- Macro-average each role to the association level ---
    target_assoc = association_level_results(target_results)
    same_entity_assoc = association_level_results(same_entity_results)
    other_entity_assoc = association_level_results(other_entity_results)

    # --- Target-attribute metrics ---
    n_target = len(target_results)
    target_flags = [preference_flags(a["sims"]) for a in target_assoc
                    if all(k in a["sims"] for k in
                           ("fine", "target", "sibling"))]
    n_flags = len(target_flags)

    prefers_fine_rate = (sum(1 for f in target_flags if f["prefers_fine"])
                         / n_flags) if n_flags else None
    gmul_proxy_forget = (1 - prefers_fine_rate) \
        if prefers_fine_rate is not None else None

    target_target_sims = [a["sims"]["target"] for a in target_assoc
                          if "target" in a["sims"]]
    gmul_proxy_core_assoc = (
        sum(target_target_sims) / len(target_target_sims)
        if target_target_sims else None)

    # Preference margin: mean(fine - target) per association
    margins = []
    for a in target_assoc:
        if "fine" in a["sims"] and "target" in a["sims"]:
            margins.append(a["sims"]["fine"] - a["sims"]["target"])
    gmul_proxy_preference_margin = (
        sum(margins) / len(margins) if margins else None)

    # --- Same-entity retain metrics ---
    se_flags = [preference_flags(a["sims"]) for a in same_entity_assoc
                if all(k in a["sims"] for k in
                       ("fine", "target", "sibling"))]
    n_se = len(se_flags)
    se_fine_rate = (sum(1 for f in se_flags if f["prefers_fine"]) / n_se
                    if n_se else None)
    gmul_proxy_holdout_association = se_fine_rate

    se_fine_sims = [a["sims"]["fine"] for a in same_entity_assoc
                    if "fine" in a["sims"]]
    gmul_proxy_intra_identity = (
        sum(se_fine_sims) / len(se_fine_sims) if se_fine_sims else None)

    # --- Other-entity retain metrics ---
    oe_fine_sims = [a["sims"]["fine"] for a in other_entity_assoc
                    if "fine" in a["sims"]]
    gmul_proxy_retain_synth = (
        sum(oe_fine_sims) / len(oe_fine_sims) if oe_fine_sims else None)
    gmul_proxy_inter_identity = gmul_proxy_retain_synth

    # --- Identity retention ---
    target_generic_sims = [a["sims"]["generic"] for a in target_assoc
                           if "generic" in a["sims"]]
    all_generic = (target_generic_sims +
                   [a["sims"]["generic"] for a in same_entity_assoc
                    if "generic" in a["sims"]])
    gmul_proxy_holdout_identity = (
        1 - sum(all_generic) / len(all_generic)
        if all_generic else None)

    def _r(v: float | None, digits: int = 4) -> float | None:
        return round(v, digits) if v is not None else None

    return {
        "gmul_proxy_forget": _r(gmul_proxy_forget),
        "gmul_proxy_holdout_association": _r(
            gmul_proxy_holdout_association),
        "gmul_proxy_holdout_identity": _r(
            gmul_proxy_holdout_identity),
        "gmul_proxy_retain_synth": _r(gmul_proxy_retain_synth),
        "gmul_proxy_core_assoc": _r(gmul_proxy_core_assoc),
        "gmul_proxy_intra_identity": _r(gmul_proxy_intra_identity),
        "gmul_proxy_inter_identity": _r(gmul_proxy_inter_identity),
        "gmul_proxy_preference_margin": _r(
            gmul_proxy_preference_margin, 6),
        "weighting": "association_macro_average",
        "n_target_probes": n_target,
        "n_target_associations": len(target_assoc),
        "n_same_entity_retain": len(same_entity_results),
        "n_same_entity_associations": len(same_entity_assoc),
        "n_other_entity_retain": len(other_entity_results),
        "n_other_entity_associations": len(other_entity_assoc),
    }


# Backward-compatible alias (deprecated — use compute_gmul_proxy_metrics)
compute_salmubench_metrics = compute_gmul_proxy_metrics


# ── Official SALMUBench evaluation ────────────────────────────────

def compute_official_salmubench(
    benchmark_dir: str | Any,
) -> dict[str, Any]:
    """Official SALMUBench evaluation status from released data.

    10R4: the released-split CLIP similarity evaluation IS implemented
    — see ``scripts/evaluate_salmu_official_splits.py``, which scores
    the released ``forget`` / ``holdout_association`` /
    ``holdout_identity`` / ``retain_synth`` parquet splits for every
    state and writes ``data/reports/salmu_official_splits.json``
    (identity-clustered bootstrap CIs).  This function only reports
    which released splits are present and links that report.

    Out of scope (require the official SALMUBench codebase,
    github.com/cvc-mmu/salmubench): RetFail (MRR over a 2,001-caption
    gallery), ACS (coherence classifier), IntraIdSim / InterIdSim on
    the released holdout embeddings, and ImageNet / DataComp general
    utility.
    """
    from pathlib import Path
    bench = Path(benchmark_dir) if not isinstance(benchmark_dir, Path) \
        else benchmark_dir

    result: dict[str, Any] = {
        "protocol_note": (
            "Released-split CLIP similarity evaluation is implemented "
            "in scripts/evaluate_salmu_official_splits.py (see "
            "data/reports/salmu_official_splits.json). Paper-protocol "
            "RetFail / ACS / IntraIdSim / InterIdSim / general utility "
            "require the official SALMUBench codebase and are not "
            "reimplemented here."),
        "status": "released_splits_present",
        "metrics": {
            "RetFail": None,
            "CoreAssoc": None,
            "ACS": None,
            "IntraIdSim": None,
            "InterIdSim": None,
            "ImageNet_accuracy": None,
            "DataComp_retrieval": None,
        },
        "benchmark_dir": str(bench),
    }

    # Check for released evaluation splits (parquet shards)
    available = []
    for split in ("forget", "forget_target", "holdout_association",
                  "holdout_identity", "retain_synth"):
        if list((bench / "data").glob(f"{split}-*.parquet")):
            available.append(split)

    result["available_splits"] = available
    if not available:
        result["status"] = "released_splits_not_found"
    return result
