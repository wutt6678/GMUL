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
    from granunlearn.salmu.embedding_metrics import preference_flags

    # --- Target-attribute metrics ---
    n_target = len(target_results)
    target_flags = [preference_flags(r["sims"]) for r in target_results
                    if all(k in r["sims"] for k in
                           ("fine", "target", "sibling"))]
    n_flags = len(target_flags)

    prefers_fine_rate = (sum(1 for f in target_flags if f["prefers_fine"])
                         / n_flags) if n_flags else None
    gmul_proxy_forget = (1 - prefers_fine_rate) \
        if prefers_fine_rate is not None else None

    target_target_sims = [r["sims"]["target"] for r in target_results
                          if "target" in r["sims"]]
    gmul_proxy_core_assoc = (
        sum(target_target_sims) / len(target_target_sims)
        if target_target_sims else None)

    # Preference margin: mean(fine - target)
    margins = []
    for r in target_results:
        if "fine" in r["sims"] and "target" in r["sims"]:
            margins.append(r["sims"]["fine"] - r["sims"]["target"])
    gmul_proxy_preference_margin = (
        sum(margins) / len(margins) if margins else None)

    # --- Same-entity retain metrics ---
    se_flags = [preference_flags(r["sims"]) for r in same_entity_results
                if all(k in r["sims"] for k in
                       ("fine", "target", "sibling"))]
    n_se = len(se_flags)
    se_fine_rate = (sum(1 for f in se_flags if f["prefers_fine"]) / n_se
                    if n_se else None)
    gmul_proxy_holdout_association = se_fine_rate

    se_fine_sims = [r["sims"]["fine"] for r in same_entity_results
                    if "fine" in r["sims"]]
    gmul_proxy_intra_identity = (
        sum(se_fine_sims) / len(se_fine_sims) if se_fine_sims else None)

    # --- Other-entity retain metrics ---
    oe_fine_sims = [r["sims"]["fine"] for r in other_entity_results
                    if "fine" in r["sims"]]
    gmul_proxy_retain_synth = (
        sum(oe_fine_sims) / len(oe_fine_sims) if oe_fine_sims else None)
    gmul_proxy_inter_identity = gmul_proxy_retain_synth

    # --- Identity retention ---
    target_generic_sims = [r["sims"]["generic"] for r in target_results
                           if "generic" in r["sims"]]
    all_generic = (target_generic_sims +
                   [r["sims"].get("generic") for r in same_entity_results
                    if "generic" in r["sims"]])
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
        "n_target_probes": n_target,
        "n_same_entity_retain": len(same_entity_results),
        "n_other_entity_retain": len(other_entity_results),
    }


# Backward-compatible alias (deprecated — use compute_gmul_proxy_metrics)
compute_salmubench_metrics = compute_gmul_proxy_metrics


# ── Official SALMUBench evaluation ────────────────────────────────

def compute_official_salmubench(
    benchmark_dir: str | Any,
) -> dict[str, Any]:
    """Official SALMUBench evaluation from released data.

    Loads the released evaluation splits from the SALMUBench
    benchmark dataset and computes the canonical metrics.

    **Official protocol** (not implemented here — requires released
    data files):

    * **RetFail**: MRR over a 2,001-caption gallery where the target
      caption is ranked among 2,000 negatives.  Measures how well
      the model forgets the target association.
    * **CoreAssoc**: Core association accuracy using BOTH
      name→value AND value→name token orders.
    * **ACS**: Association coherence score from a coherence
      classifier over generated captions.
    * **IntraIdSim / InterIdSim**: Intra- and inter-identity
      similarity on the released holdout splits.
    * **General utility**: ImageNet zero-shot classification
      accuracy and DataComp retrieval metrics.

    Parameters
    ----------
    benchmark_dir : str or Path
        Path to the released SALMUBench benchmark dataset
        (``cvc-mmu/salmubench-512-redistributed``).

    Returns
    -------
    dict with official metric names and values, plus a
    ``protocol_note`` describing what each metric measures.
    """
    from pathlib import Path
    bench = Path(benchmark_dir) if not isinstance(benchmark_dir, Path) \
        else benchmark_dir

    result: dict[str, Any] = {
        "protocol_note": (
            "Official SALMUBench metrics require the released "
            "evaluation splits.  RetFail = MRR over 2,001-caption "
            "gallery; CoreAssoc = both name-value token orders; "
            "ACS = coherence classifier; IntraIdSim/InterIdSim = "
            "released holdout splits; general utility = ImageNet + "
            "DataComp."),
        "status": "not_yet_computed",
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

    # Check for released evaluation files
    forget_file = bench / "forget.jsonl"
    retain_file = bench / "retain_synth.jsonl"
    holdout_file = bench / "holdout_association.jsonl"

    available = []
    if forget_file.exists():
        available.append("forget")
    if retain_file.exists():
        available.append("retain_synth")
    if holdout_file.exists():
        available.append("holdout_association")

    result["available_splits"] = available
    if not available:
        result["status"] = "released_splits_not_found"
    return result
