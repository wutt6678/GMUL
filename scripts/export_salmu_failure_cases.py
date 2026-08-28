"""Export per-probe failure cases for all SALMU states.

For each (state, target persona, attribute), classify the probe outcome:

* target_preferred  — sim_target > sim_fine AND sim_target > sim_sibling
                      (desired MG/MU behavior)
* fine_leakage      — sim_fine > sim_target (fine association survives)
* sibling_confusion — sim_sibling > sim_target (branch leakage)
* over_forgotten    — all entity sims within 0.02 of generic
                      (complete forgetting)

Writes one JSON per state to data/salmu_hierarchical/failure_exports/.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from granunlearn.config import _find_repo_root
from granunlearn.logging_utils import setup_logger
from granunlearn.salmu.eval_utils import (
    load_probe_cache,
)

log = setup_logger("salmu_failure_export")

FORGOT_TOL = 0.02


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
        cases: list[dict[str, Any]] = []
        counts: Counter[str] = Counter()
        for r in results:
            cat = classify_probe(r["sims"])
            counts[cat] += 1
            cases.append({
                "identity_id": r["identity_id"],
                "attribute": r["attribute"],
                "category": cat,
                "sims": {k: round(v, 4) for k, v in r["sims"].items()},
            })
        # Write per-state export
        out_path = out_dir / f"failure_cases_{state}.json"
        with open(out_path, "w") as f:
            json.dump(cases, f, indent=2, ensure_ascii=False)
        total = len(cases)
        rates = {k: round(v / total, 4) for k, v in counts.items()}
        summary[state] = {"num_probes": total, "counts": dict(counts),
                          "rates": rates}
        log.info("[%s] %d probes: %s", state, total,
                 {k: f"{v:.1%}" for k, v in rates.items()})

    # Write summary
    summary_path = out_dir / "failure_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    log.info("SALMU failure exports -> %s (%d states)", out_dir,
             len(summary))


if __name__ == "__main__":
    main()
