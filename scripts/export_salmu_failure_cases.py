"""Export compact per-state failure summaries for SALMU states (10R4).

For each (state, role, probe), classify the outcome:

* target_preferred  — sim_target > sim_fine AND sim_target > sim_sibling
                      (desired MG/MU behavior)
* fine_leakage      — sim_fine > sim_target (fine association survives)
* sibling_confusion — sim_sibling > sim_target (branch leakage)
* over_forgotten    — all entity sims within FORGOT_TOL of generic

10R4 changes
------------
* Roles are reported SEPARATELY: target_association probes
  (is_target_attr=True) vs same_entity_retain probes — the previous
  exports mixed both and omitted the role flag entirely.
* Exports are COMPACT: a per-state summary plus TOP_K representative
  cases per (role, category), instead of one row per probe (~18,687
  rows / ~4 MiB per state; ~90 MiB total committed by mistake).
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from granunlearn.config import _find_repo_root
from granunlearn.logging_utils import setup_logger
from granunlearn.salmu.eval_utils import (
    load_probe_cache,
)

log = setup_logger("salmu_failure_export")

FORGOT_TOL = 0.02
TOP_K = 20  # representative cases per (role, category)


def classify_probe(sims: dict[str, float]) -> str:
    """Classify one probe's outcome."""
    fine = sims.get("fine")
    target = sims.get("target")
    sibling = sims.get("sibling")
    generic = sims.get("generic", 0.0)

    if fine is None or target is None:
        return "missing_data"

    # Over-forgotten: all entity sims near generic
    entity_sims = [fine, target]
    if sibling is not None:
        entity_sims.append(sibling)
    if all(abs(s - generic) <= FORGOT_TOL for s in entity_sims):
        return "over_forgotten"

    # Sibling confusion: another persona's caption preferred
    if sibling is not None and sibling > target and sibling > fine:
        return "sibling_confusion"

    # Fine leakage: fine still preferred over target
    if fine > target:
        return "fine_leakage"

    # Target preferred (desired)
    return "target_preferred"


def _severity(r: dict[str, Any], category: str) -> float:
    """Ranking key for picking representative cases (higher = more
    extreme example of the category)."""
    sims = r["sims"]
    fine, target = sims.get("fine", 0.0), sims.get("target", 0.0)
    sibling = sims.get("sibling", 0.0)
    if category == "fine_leakage":
        return fine - target
    if category == "target_preferred":
        return target - max(fine, sibling)
    if category == "sibling_confusion":
        return sibling - max(fine, target)
    if category == "over_forgotten":
        generic = sims.get("generic", 0.0)
        return -max(abs(s - generic) for s in (fine, target, sibling))
    return 0.0


def main() -> None:
    repo_root = _find_repo_root(Path.cwd()) or Path.cwd()
    target_cache_path = repo_root / "data" / "salmu_hierarchical" / \
        "probe_sims_unlearn.json"
    target_cache = load_probe_cache(target_cache_path)
    if target_cache is None:
        raise FileNotFoundError(f"Missing {target_cache_path}")

    out_dir = repo_root / "data" / "salmu_hierarchical" / "failure_exports"
    out_dir.mkdir(parents=True, exist_ok=True)

    summary: dict[str, Any] = {}
    for state, results in sorted(target_cache.items()):
        counts: dict[str, Counter] = {
            "target_association": Counter(),
            "same_entity_retain": Counter(),
        }
        by_bucket: dict[tuple[str, str], list[dict[str, Any]]] = {}
        for r in results:
            role = ("target_association"
                    if r.get("is_target_attr", False)
                    else "same_entity_retain")
            cat = classify_probe(r["sims"])
            counts[role][cat] += 1
            by_bucket.setdefault((role, cat), []).append({
                "identity_id": r["identity_id"],
                "attribute": r["attribute"],
                "is_target_attr": r.get("is_target_attr", False),
                "category": cat,
                "image_file": r.get("image_file"),
                "sims": r["sims"],
            })

        # Top-K most extreme cases per (role, category)
        top_cases: list[dict[str, Any]] = []
        for (role, cat), cases in sorted(by_bucket.items()):
            cases.sort(key=lambda c: _severity(c, cat), reverse=True)
            top_cases.extend(cases[:TOP_K])

        n = len(results)
        state_summary = {
            "num_probes": n,
            "roles": {},
        }
        for role in ("target_association", "same_entity_retain"):
            total = sum(counts[role].values())
            state_summary["roles"][role] = {
                "num_probes": total,
                "counts": dict(counts[role]),
                "rates": {k: round(v / total, 4)
                          for k, v in counts[role].items()}
                if total else {},
            }
        summary[state] = state_summary

        out_path = out_dir / f"failure_cases_{state}.json"
        with open(out_path, "w") as f:
            json.dump({
                "state": state,
                "top_k_per_role_category": TOP_K,
                "summary": state_summary,
                "representative_cases": top_cases,
            }, f, indent=2, ensure_ascii=False)
        tgt = state_summary["roles"]["target_association"]
        log.info("[%s] %d probes (target %d / retain %d): "
                 "target-role rates %s",
                 state, n, tgt["num_probes"],
                 state_summary["roles"]["same_entity_retain"]
                 ["num_probes"],
                 {k: f"{v:.1%}" for k, v in tgt["rates"].items()})

    summary_path = out_dir / "failure_summary.json"
    with open(summary_path, "w") as f:
        json.dump({
            "weighting_note": "probe-level classification; rates are "
                              "per-probe within each role (the "
                              "association-weighted point estimates "
                              "live in the selection/reference "
                              "reports).",
            "top_k_per_role_category": TOP_K,
            "states": summary,
        }, f, indent=2, ensure_ascii=False)
    log.info("SALMU failure exports -> %s (%d states)", out_dir,
             len(summary))


if __name__ == "__main__":
    main()
