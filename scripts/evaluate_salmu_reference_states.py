"""Evaluate SALMU reference states on hierarchy-aware embedding probes.

    python scripts/evaluate_salmu_reference_states.py --device cuda:0

Loads BASE (released Clean CLIP) + MF/MG/MN checkpoints, encodes the
target-persona probes (fine / target / ancestor / sibling / generic),
applies the Iteration 10 separation gate, and writes
data/reports/salmu_reference_eval.json.

With per-attribute targeting, probes cover BOTH target attributes
(is_target_attr=True) and same-entity retain attributes
(is_target_attr=False).  The gate runs ONLY on the target-attribute
probes (the associations being generalized or removed); same-entity
and other-entity retention are reported separately.
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
    aggregate_scores_by_image,
    aggregate_scores_by_target_attr,
    image_caption_variance,
    reference_state_gate,
)
from granunlearn.salmu.eval_utils import (
    SalmuImageIndex,
    build_release_probes,
    build_retain_probes,
    score_probes,
)
from granunlearn.salmu.salmubench_metrics import (
    compute_gmul_proxy_metrics,
    compute_official_salmubench,
)

log = setup_logger("evaluate_salmu_reference_states")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate SALMU reference states")
    parser.add_argument("--states", default="BASE,MF,MG,MN")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--bootstrap-ci", action=argparse.BooleanOptionalAction,
                        default=True,
                        help="Compute association-clustered bootstrap "
                             "confidence intervals (default: on — the "
                             "committed evidence must carry CIs)")
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--max-images", type=int, default=None,
                        help="Cap images per (persona, attr)")
    parser.add_argument("--max-captions", type=int, default=None,
                        help="Cap captions per (persona, attr)")
    parser.add_argument("--output", default=None,
                        help="Report path (default: "
                             "data/reports/salmu_reference_eval.json); "
                             "use a temp path for single-state "
                             "parallel workers")
    args = parser.parse_args()

    repo_root = _find_repo_root(Path.cwd()) or Path.cwd()
    train_ds = locate_repo(REPOS["training_dataset"]["repo_id"], "dataset")

    # Target-persona probes (cover both target and same-entity retain)
    probes, target_ids = build_release_probes(
        repo_root, max_images=args.max_images,
        max_captions=args.max_captions)
    log.info("Built %d target probes over %d personas", len(probes),
             len(target_ids))

    # Retain-persona probes (other-entity retain)
    retain_probes = build_retain_probes(
        repo_root, max_images=args.max_images,
        max_captions=args.max_captions)
    log.info("Built %d retain probes", len(retain_probes))

    image_index = SalmuImageIndex(train_ds / "data")

    scores: dict = {}
    per_attr_scores: dict = {}
    target_vs_retain_scores: dict = {}
    target_only_scores: dict = {}
    target_only_by_attribute: dict = {}
    same_entity_retain_scores: dict = {}
    other_entity_retain_scores: dict = {}
    gmul_proxy: dict = {}
    # Keep last state's raw results for variance analysis
    _last_results: list = []
    _last_target_results: list = []
    _last_same_entity_results: list = []
    _last_retain_results: list = []

    for state in [s.strip().upper() for s in args.states.split(",")]:
        # Score target-persona probes
        results = score_probes(state, probes, image_index, repo_root,
                               args.device)
        # Score retain-persona probes
        retain_results = score_probes(
            state, retain_probes, image_index, repo_root, args.device)

        # --- Gate metrics: TARGET-ONLY probes ---
        target_results = [r for r in results
                          if r.get("is_target_attr", False)]
        same_entity_results = [r for r in results
                               if not r.get("is_target_attr", False)]

        target_only_scores[state] = aggregate_scores(
            target_results, bootstrap_ci=args.bootstrap_ci,
            n_bootstrap=args.n_bootstrap)
        # Target-only per-attribute breakdown (with CIs, 10R4a)
        target_only_by_attribute[state] = {
            attr: aggregate_scores(
                [r for r in target_results if r["attribute"] == attr],
                bootstrap_ci=args.bootstrap_ci,
                n_bootstrap=args.n_bootstrap)
            for attr in ("city", "job", "blood_type")
        }
        same_entity_retain_scores[state] = aggregate_scores(
            same_entity_results, bootstrap_ci=args.bootstrap_ci,
            n_bootstrap=args.n_bootstrap)
        other_entity_retain_scores[state] = aggregate_scores(
            retain_results, bootstrap_ci=args.bootstrap_ci,
            n_bootstrap=args.n_bootstrap)

        # --- Pooled metrics (all target-persona probes) ---
        scores[state] = aggregate_scores(
            results, bootstrap_ci=args.bootstrap_ci,
            n_bootstrap=args.n_bootstrap)
        per_attr_scores[state] = aggregate_scores_by_attribute(
            results, bootstrap_ci=args.bootstrap_ci,
            n_bootstrap=args.n_bootstrap)
        target_vs_retain_scores[state] = aggregate_scores_by_target_attr(
            results, bootstrap_ci=args.bootstrap_ci,
            n_bootstrap=args.n_bootstrap)

        s = target_only_scores[state]
        log.info("[%s] TARGET fine_pref=%s tnf=%s | sims=%s",
                 state, s.get("prefers_fine_rate"),
                 s.get("prefers_target_not_fine_rate"),
                 s.get("mean_similarities"))
        sr = same_entity_retain_scores[state]
        log.info("[%s] SAME-ENTITY RETAIN mean_fine=%s",
                 state,
                 (sr.get("mean_similarities") or {}).get("fine"))
        oe = other_entity_retain_scores[state]
        log.info("[%s] OTHER-ENTITY RETAIN mean_fine=%s",
                 state,
                 (oe.get("mean_similarities") or {}).get("fine"))

        # GMUL proxy metrics (in-house CLIP-embedding proxies)
        gmul_proxy[state] = compute_gmul_proxy_metrics(
            target_results, same_entity_results, retain_results)
        log.info("[%s] GMUL proxy forget=%s holdout_assoc=%s "
                 "retain_synth=%s",
                 state, gmul_proxy[state]["gmul_proxy_forget"],
                 gmul_proxy[state]["gmul_proxy_holdout_association"],
                 gmul_proxy[state]["gmul_proxy_retain_synth"])

        # Keep last state's raw results for variance analysis
        _last_results = results
        _last_target_results = target_results
        _last_same_entity_results = same_entity_results
        _last_retain_results = retain_results

    # Gate runs ONLY on target-attribute probes
    passed, reasons = reference_state_gate(target_only_scores)

    # Image/caption variance analysis (uses last state's target results)
    img_cap_var = image_caption_variance(_last_results)

    # Official SALMUBench evaluation (from released splits)
    bench = locate_repo(REPOS["benchmark_dataset"]["repo_id"], "dataset")
    official_salmubench = compute_official_salmubench(
        bench, official_splits_report=repo_root / "data" / "reports" /
        "salmu_official_splits.json")

    report = {
        "experiment_id": "salmu_iter10r4a_reference_states",
        "num_target_personas": len(target_ids),
        "num_target_probes": len(target_results),
        "num_same_entity_retain_probes": len(same_entity_results),
        "num_other_entity_retain_probes": len(retain_results),
        "probes_per_kind_note": "one probe per (persona, core "
                                "attribute, image, fine_caption) "
                                "with frozen probe IDs",
        "multi_image": True,
        "multi_image_note": "ALL released image/caption variants per "
                            "association are used (no cap); "
                            "max_images/max_captions null = uncapped.",
        "max_images": args.max_images,
        "max_captions": args.max_captions,
        "per_attribute_targeting": True,
        "weighting": "association_macro_average",
        "weighting_note": "image-caption variants within each "
                          "(identity, attribute) association are "
                          "macro-averaged before rates/similarities, "
                          "so every association counts once.",
        "bootstrap_ci": args.bootstrap_ci,
        "bootstrap_note": "association bootstrap: resamples "
                          "(identity, attribute) units, exact under "
                          "the macro-average."
        if args.bootstrap_ci else None,
        "gate_runs_on": "target_association probes only (is_target_attr=True)",
        "scores_by_state": target_only_scores,
        "target_only_by_attribute": target_only_by_attribute,
        "pooled_scores": scores,
        "per_attribute_scores": per_attr_scores,
        "target_vs_retain_scores": target_vs_retain_scores,
        "same_entity_retain_scores": same_entity_retain_scores,
        "other_entity_retain_scores": other_entity_retain_scores,
        "image_caption_variance": img_cap_var,
        "gmul_proxy_metrics": gmul_proxy,
        "official_salmubench": official_salmubench,
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
            "Gate runs ONLY on target-attribute probes (the "
            "associations being generalized or removed).",
            "same_entity_retain: non-target attributes of target "
            "personas (always fine captions, even in MN).",
            "other_entity_retain: all attributes of non-target "
            "personas (always fine captions).",
            "Per-attribute targeting: each target persona has ONE "
            "target attribute; other attributes are same-entity retain.",
            "Frozen multi-image/caption probes with deterministic "
            "probe IDs for reproducibility.",
        ],
    }
    out = (Path(args.output) if args.output
           else repo_root / "data" / "reports" /
           "salmu_reference_eval.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    tmp.replace(out)
    log.info("SALMU reference-state gate: %s%s",
             "PASSED" if passed else "FAILED",
             f" ({'; '.join(reasons)})" if reasons else "")
    log.info("Wrote report -> %s", out)


if __name__ == "__main__":
    main()
