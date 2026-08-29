"""Evaluate SALMU reference states on hierarchy-aware embedding probes.

    python scripts/evaluate_salmu_reference_states.py --device cuda:0

Loads BASE (released Clean CLIP) + MF/MG/MN checkpoints, encodes the
target-persona probes (fine / target / ancestor / sibling / generic),
applies the Iteration 10 separation gate, and writes
data/reports/salmu_reference_eval.json.

With per-attribute targeting, probes cover BOTH target attributes
(is_target_attr=True) and same-entity retain attributes
(is_target_attr=False).  The report includes:
* Aggregate metrics with bootstrap confidence intervals
* Per-attribute breakdown (city, job, blood_type)
* Target vs same-entity retain breakdown
* Official SALMUBench metrics (if benchmark splits available)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from granunlearn.config import _find_repo_root
from granunlearn.logging_utils import setup_logger
from granunlearn.salmu.adapter import REPOS, locate_repo
from granunlearn.salmu.embedding_metrics import (
    aggregate_scores,
    aggregate_scores_by_attribute,
    aggregate_scores_by_target_attr,
    reference_state_gate,
)
from granunlearn.salmu.eval_utils import (
    SalmuImageIndex,
    build_release_probes,
    score_probes,
)

log = setup_logger("evaluate_salmu_reference_states")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate SALMU reference states")
    parser.add_argument("--states", default="BASE,MF,MG,MN")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--bootstrap-ci", action="store_true",
                        help="Compute bootstrap confidence intervals")
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    args = parser.parse_args()

    repo_root = _find_repo_root(Path.cwd()) or Path.cwd()
    train_ds = locate_repo(REPOS["training_dataset"]["repo_id"], "dataset")

    probes, target_ids = build_release_probes(repo_root)
    log.info("Built %d target probes over %d personas", len(probes),
             len(target_ids))
    image_index = SalmuImageIndex(train_ds / "data")

    scores: dict = {}
    per_attr_scores: dict = {}
    target_vs_retain_scores: dict = {}
    
    for state in [s.strip().upper() for s in args.states.split(",")]:
        results = score_probes(state, probes, image_index, repo_root,
                               args.device)
        # Aggregate metrics with optional bootstrap CIs
        scores[state] = aggregate_probe_results(
            results, bootstrap_ci=args.bootstrap_ci,
            n_bootstrap=args.n_bootstrap)
        # Per-attribute breakdown
        per_attr_scores[state] = aggregate_scores_by_attribute(
            results, bootstrap_ci=args.bootstrap_ci,
            n_bootstrap=args.n_bootstrap)
        # Target vs same-entity retain breakdown
        target_vs_retain_scores[state] = aggregate_scores_by_target_attr(
            results, bootstrap_ci=args.bootstrap_ci,
            n_bootstrap=args.n_bootstrap)
        
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
        "per_attribute_targeting": True,
        "bootstrap_ci": args.bootstrap_ci,
        "scores_by_state": scores,
        "per_attribute_scores": per_attr_scores,
        "target_vs_retain_scores": target_vs_retain_scores,
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
        "notes": [
            "Per-attribute targeting: each target persona has ONE target "
            "attribute; other attributes are same-entity retain.",
            "target_vs_retain_scores breaks down metrics by is_target_attr flag.",
            "per_attribute_scores breaks down metrics by attribute (city/job/blood_type).",
        ] if args.bootstrap_ci else [
            "Per-attribute targeting: each target persona has ONE target "
            "attribute; other attributes are same-entity retain.",
            "target_vs_retain_scores breaks down metrics by is_target_attr flag.",
            "per_attribute_scores breaks down metrics by attribute (city/job/blood_type).",
        ],
    }
    out = repo_root / "data" / "reports" / "salmu_reference_eval.json"
    with open(out, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    log.info("SALMU reference-state gate: %s%s",
             "PASSED" if passed else "FAILED",
             f" ({'; '.join(reasons)})" if reasons else "")


def aggregate_probe_results(
    results: list[dict],
    bootstrap_ci: bool = False,
    n_bootstrap: int = 1000,
) -> dict:
    """Wrapper for aggregate_scores with consistent naming."""
    return aggregate_scores(results, bootstrap_ci=bootstrap_ci,
                           n_bootstrap=n_bootstrap)


if __name__ == "__main__":
    main()
