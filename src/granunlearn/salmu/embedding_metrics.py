"""Hierarchy-aware embedding metrics + reference-state gate (SALMU).

Iteration 10 gate: verify MF_SALMU != MG_SALMU != MN_SALMU BEFORE any
unlearning, on the embedding preferences of TARGET personas:

    sim(I_e, T_fine), sim(I_e, T_target),
    sim(I_e, T_ancestor), sim(I_e, T_sibling), sim(I_e, T_generic)

* T_fine      — a released fine (paraphrased) caption of the persona
* T_target    — controlled caption at the target level
* T_ancestor  — controlled caption at the coarsest level (when it
                differs from the target level; 2-level chains have
                target == ancestor and report ancestor = target)
* T_sibling   — target-level caption of a DIFFERENT persona sharing the
                same ancestor value (same country / sector / ABO) —
                tests branch specificity, not generic topical alignment
* T_generic   — neutral "A photo of a person." reference

Gate (on target personas):
1. MF prefers fine over {target, sibling}: rate_mf >= 0.5 and
   rate_mf >= rate_mg + 0.15 and rate_mf >= rate_mn + 0.15
2. MG prefers target (not fine): prefers_target_not_fine_mg >= 0.5,
   >= prefers_target_not_fine_mn + 0.15, fine preference capped at
   mg_max_fine_preference, and not below its own target preference
3. MN has NO entity-specific association: its mean fine and target
   similarities stay within sim_tol of BASE's.  Preference ORDER is
   deliberately NOT used for MN/BASE: when all entity similarities sit
   near chance, the argmax-style order is noise (Iteration 10 lesson).
"""

from __future__ import annotations

from typing import Any

from granunlearn.logging_utils import setup_logger
from granunlearn.salmu.hierarchy import generalized_caption

log = setup_logger("salmu_embedding_metrics")

GENERIC_CAPTION = "A photo of a person."

PROBE_KINDS = ("fine", "target", "ancestor", "sibling", "generic")


def build_target_probes(
    target_identity_ids: list[str],
    hierarchies: dict[str, dict[str, Any]],
    identities: dict[str, dict[str, Any]],
    fine_captions_by_identity_attr: dict[str, dict[str, list[str]]],
    images_by_identity_attr: dict[str, dict[str, list[str]]],
) -> list[dict[str, Any]]:
    """One probe per (target persona, attribute) that has images.

    Deterministic: first sorted fine caption / first sorted image; the
    sibling is the FIRST other persona (sorted id) sharing the ancestor
    value for that attribute.
    """
    # ancestor value -> persona ids (per attribute), for sibling lookup
    sibling_index: dict[tuple[str, str], list[str]] = {}
    for iid in sorted(hierarchies):
        for attr, hier in hierarchies[iid].items():
            anc = hier["levels"][-1]
            sibling_index.setdefault((attr, anc), []).append(iid)

    probes: list[dict[str, Any]] = []
    for iid in sorted(target_identity_ids):
        name = identities[iid]["name"]
        for attr in sorted(hierarchies[iid]):
            hier = hierarchies[iid][attr]
            images = sorted(images_by_identity_attr.get(
                iid, {}).get(attr, []))
            fines = sorted(fine_captions_by_identity_attr.get(
                iid, {}).get(attr, []))
            if not images or not fines:
                continue
            tgt = hier["target_level"]
            probes.append({
                "identity_id": iid,
                "attribute": attr,
                "image_file": images[0],
                "fine_caption": fines[0],
                "target_caption": generalized_caption(
                    name, attr, tgt, hier["levels"][tgt]),
                "ancestor_caption": generalized_caption(
                    name, attr, len(hier["levels"]) - 1,
                    hier["levels"][-1]),
                "ancestor_is_target": tgt == len(hier["levels"]) - 1,
                "sibling_caption": _sibling_caption(
                    iid, attr, hier, hierarchies, identities,
                    sibling_index),
                "generic_caption": GENERIC_CAPTION,
            })
    return probes


def _sibling_caption(
    iid: str, attr: str, hier: dict,
    hierarchies: dict[str, dict],
    identities: dict[str, dict],
    sibling_index: dict[tuple[str, str], list[str]],
) -> str | None:
    anc = hier["levels"][-1]
    for other in sibling_index.get((attr, anc), []):
        if other == iid:
            continue
        other_hier = hierarchies[other].get(attr)
        if other_hier is None:
            continue
        tgt = other_hier["target_level"]
        return generalized_caption(
            identities[other]["name"], attr, tgt,
            other_hier["levels"][tgt])
    return None


def preference_flags(sims: dict[str, float]) -> dict[str, bool]:
    """Boolean preference summary for one probe's similarity dict."""
    fine, target = sims["fine"], sims["target"]
    sibling = sims["sibling"]
    return {
        "prefers_fine": fine > max(target, sibling),
        "prefers_target": target > max(fine, sibling),
        "prefers_target_not_fine": (target > fine) and (target > sibling),
    }


def aggregate_scores(
    probe_results: list[dict[str, Any]],
) -> dict[str, Any]:
    """Preference rates + mean similarities over target probes.

    Preference rates are computed over probes that HAVE a sibling
    caption (branch-specificity requires one); mean similarities over
    all probes per kind.
    """
    n = len(probe_results)
    if n == 0:
        return {"num_probes": 0}
    flag_probes = [r for r in probe_results
                   if all(k in r["sims"] for k in
                          ("fine", "target", "sibling"))]
    flags = [preference_flags(r["sims"]) for r in flag_probes]
    m = len(flags)

    def rate(key: str) -> float | None:
        return round(sum(1 for f in flags if f[key]) / m, 4) if m else None

    mean_sims = {}
    for kind in PROBE_KINDS:
        vals = [r["sims"][kind] for r in probe_results
                if kind in r["sims"]]
        if vals:
            mean_sims[kind] = round(sum(vals) / len(vals), 4)
    return {
        "num_probes": n,
        "num_preference_probes": m,
        "prefers_fine_rate": rate("prefers_fine"),
        "prefers_target_rate": rate("prefers_target"),
        "prefers_target_not_fine_rate": rate("prefers_target_not_fine"),
        "mean_similarities": mean_sims,
    }


def reference_state_gate(
    scores_by_state: dict[str, dict[str, Any]],
    min_gap: float = 0.15,
    mg_max_fine_preference: float = 0.25,
    mn_sim_tol: float = 0.05,
) -> tuple[bool, list[str]]:
    """MF != MG != MN on embedding preferences (Iteration 10 gate)."""
    reasons: list[str] = []
    required = ("BASE", "MF", "MG", "MN")
    for state in required:
        if state not in scores_by_state:
            reasons.append(f"missing state scores: {state}")
    if reasons:
        return False, reasons
    base, mf, mg, mn = (scores_by_state[s] for s in required)
    for state, s in (("BASE", base), ("MF", mf), ("MG", mg), ("MN", mn)):
        if s.get("prefers_fine_rate") is None or \
                s.get("prefers_target_not_fine_rate") is None:
            reasons.append(f"{state} has no sibling-bearing probes")
    if reasons:
        return False, reasons

    if mf["prefers_fine_rate"] < 0.5:
        reasons.append(
            f"MF prefers_fine_rate {mf['prefers_fine_rate']} < 0.5")
    if mf["prefers_fine_rate"] < mg["prefers_fine_rate"] + min_gap:
        reasons.append("MF does not exceed MG fine-preference by "
                       f"{min_gap}")
    if mf["prefers_fine_rate"] < mn["prefers_fine_rate"] + min_gap:
        reasons.append("MF does not exceed MN fine-preference by "
                       f"{min_gap}")

    if mg["prefers_target_not_fine_rate"] < 0.5:
        reasons.append(
            f"MG prefers_target_not_fine_rate "
            f"{mg['prefers_target_not_fine_rate']} < 0.5")
    if mg["prefers_target_not_fine_rate"] < \
            mn["prefers_target_not_fine_rate"] + min_gap:
        reasons.append("MG does not exceed MN target-preference by "
                       f"{min_gap}")
    if mg["prefers_fine_rate"] > mg["prefers_target_not_fine_rate"]:
        reasons.append("MG prefers fine over its own target level")
    if mg["prefers_fine_rate"] > mg_max_fine_preference:
        reasons.append(
            f"MG prefers_fine_rate {mg['prefers_fine_rate']} > "
            f"{mg_max_fine_preference} (MG must NOT prefer the fine "
            f"caption)")

    # MN: similarity magnitude close to BASE (order is noise at chance)
    base_sims = base.get("mean_similarities") or {}
    mn_sims = mn.get("mean_similarities") or {}
    for kind in ("fine", "target"):
        b, m = base_sims.get(kind), mn_sims.get(kind)
        if b is None or m is None:
            reasons.append(f"missing mean {kind} similarity for "
                           f"BASE/MN")
        elif abs(m - b) > mn_sim_tol:
            reasons.append(
                f"MN mean {kind} similarity {m} deviates from BASE "
                f"{b} by more than {mn_sim_tol}")
    return len(reasons) == 0, reasons
