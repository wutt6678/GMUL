"""Evaluate SALMU reference states on hierarchy-aware embedding probes.

    python scripts/evaluate_salmu_reference_states.py --device cuda:0

Loads BASE (released Clean CLIP) + MF/MG/MN checkpoints, encodes the
target-persona probes (fine / target / ancestor / sibling / generic),
applies the Iteration 10 separation gate, and writes
data/reports/salmu_reference_eval.json.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from granunlearn.config import _find_repo_root
from granunlearn.logging_utils import setup_logger
from granunlearn.salmu.adapter import REPOS, locate_repo
from granunlearn.salmu.embedding_metrics import (
    reference_state_gate,
)
from granunlearn.salmu.eval_utils import (
    SalmuImageIndex,
    aggregate_probe_results,
    build_release_probes,
    score_probes,
)

log = setup_logger("evaluate_salmu_reference_states")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate SALMU reference states")
    parser.add_argument("--states", default="BASE,MF,MG,MN")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    repo_root = _find_repo_root(Path.cwd()) or Path.cwd()
    train_ds = locate_repo(REPOS["training_dataset"]["repo_id"], "dataset")

    probes, target_ids = build_release_probes(repo_root)
    log.info("Built %d target probes over %d personas", len(probes),
             len(target_ids))
    image_index = SalmuImageIndex(train_ds / "data")

    scores: dict = {}
    for state in [s.strip().upper() for s in args.states.split(",")]:
        results = score_probes(state, probes, image_index, repo_root,
                               args.device)
        scores[state] = aggregate_probe_results(results)
        s = scores[state]
        log.info("[%s] fine_pref=%s target_not_fine=%s | sims=%s",
                 state, s["prefers_fine_rate"],
                 s["prefers_target_not_fine_rate"],
                 s["mean_similarities"])

    passed, reasons = reference_state_gate(scores)
    report = {
        "experiment_id": "salmu_iter10_reference_states",
        "num_target_personas": len(target_ids),
        "probes_per_kind_note": "one probe per (target persona, core "
                                "attribute) with images",
        "scores_by_state": scores,
        "reference_state_gate": {
            "passed": passed,
            "reasons": reasons,
            "definition": {
                "MF": "prefers fine over {target, sibling}: >= 0.5 "
                      "and >= MG/MN + 0.05",
                "MG": "mean target sim > mean fine sim; fine "
                      "preference capped at 0.50",
                "MN": "mean fine/target similarities below MF's "
                      "(drop >= 0.01; per-attribute removal retains "
                      "most entity info)",
            },
        },
    }
    out = repo_root / "data" / "reports" / "salmu_reference_eval.json"
    with open(out, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    log.info("SALMU reference-state gate: %s%s",
             "PASSED" if passed else "FAILED",
             f" ({'; '.join(reasons)})" if reasons else "")


if __name__ == "__main__":
    main()
