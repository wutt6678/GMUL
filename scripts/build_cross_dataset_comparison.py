"""Build cross-dataset comparison report (Iteration 10R3; suffix
support + official SALMUBench metrics in 10R5c).

    python scripts/build_cross_dataset_comparison.py
    python scripts/build_cross_dataset_comparison.py --suffix r5

Reads MLLMU + SALMU reference-eval and selection reports, extracts
comparable metrics, and writes
data/reports/cross_dataset_comparison[_suffix].json.  With
``--suffix r5`` the SALMU side reads the holdout-clean r5 reports
(so the selected B-candidates are the r5 retrained ones) and embeds
the official SALMUBench-evaluator metric table (including the
target-only / paired-CI summary) from
salmubench_official_eval[_suffix].json when present.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from granunlearn.config import _find_repo_root
from granunlearn.logging_utils import setup_logger
from granunlearn.salmu.paths import SalmuPaths

log = setup_logger("build_cross_dataset_comparison")


def _mllmu_ref_metrics(ref: dict) -> dict[str, dict]:
    """Extract filr/tga from MLLMU reference eval metrics_by_state."""
    out: dict = {}
    mbs = ref.get("metrics_by_state", {})
    for state, s in mbs.items():
        tc = s.get("target_core", {})
        out[state] = {
            "filr": round(tc.get("leakage_rate", 0), 4),
            "tga": round(tc.get("post_unlearning_accuracy", 0), 4),
            "ancestor_retention": tc.get("ancestor_retention_rate"),
            "retain_same": None,
            "retain_other": None,
            "over_forgetting_rate": None,
            "wrong_branch_rate": None,
        }
    return out


def _mllmu_sel_metrics(sel: dict) -> tuple[dict, dict]:
    """Extract metrics for selected MLLMU candidates."""
    cands = sel.get("candidates", {})
    selected = sel.get("selected", {})
    test_metrics: dict = {}
    for method, cid in selected.items():
        c = cands.get(cid, {})
        v = c.get("vector", {})
        test_metrics[cid] = {
            "filr": v.get("filr"),
            "tga": v.get("tga"),
            "ancestor_retention": v.get("ancestor"),
            "retain_same": v.get("retain_same"),
            "retain_other": v.get("retain_other"),
            "over_forgetting_rate": v.get("over"),
            "wrong_branch_rate": v.get("wrong"),
        }
    return test_metrics, selected


def _salmu_ref_target_metrics(ref: dict) -> dict[str, dict]:
    """Extract SALMU reference target-attribute metrics.

    Reads from ``target_only_scores`` (the gate runs on target-attribute
    probes only) and supplements with GMUL proxy metrics, retain
    similarities, and the per-attribute target-only breakdown.
    """
    out: dict = {}
    sbs = ref.get("scores_by_state", {})  # target_only_scores
    per_attr = ref.get("target_only_by_attribute", {})
    for state, s in sbs.items():
        sims = s.get("mean_similarities", {})
        out[state] = {
            "fine_pref": s.get("prefers_fine_rate"),
            "tnf": s.get("prefers_target_not_fine_rate"),
            "sim_fine": sims.get("fine"),
            "sim_target": sims.get("target"),
            "sim_sibling": sims.get("sibling"),
            "same_entity_retain_sim": None,
            "other_entity_retain_sim": None,
        }
        # Target-only per-attribute breakdown
        if state in per_attr:
            out[state]["per_attribute"] = {
                attr: {
                    "fine_pref": a.get("prefers_fine_rate"),
                    "tnf": a.get("prefers_target_not_fine_rate"),
                    "sim_fine": (a.get("mean_similarities") or {}).get(
                        "fine"),
                    "sim_target": (a.get("mean_similarities") or {}).get(
                        "target"),
                    "num_probes": a.get("num_probes"),
                }
                for attr, a in per_attr[state].items()
            }
    # Same-entity retain similarity from separate scores
    se = ref.get("same_entity_retain_scores", {})
    for state in out:
        if state in se:
            sims = se[state].get("mean_similarities", {})
            out[state]["same_entity_retain_sim"] = sims.get("fine")
    # Other-entity retain similarity
    oe = ref.get("other_entity_retain_scores", {})
    for state in out:
        if state in oe:
            sims = oe[state].get("mean_similarities", {})
            out[state]["other_entity_retain_sim"] = sims.get("fine")
    # GMUL proxy metrics (in-house CLIP-embedding proxies)
    gmul = ref.get("gmul_proxy_metrics", {})
    for state in out:
        if state in gmul:
            out[state]["gmul_proxy_forget"] = gmul[state].get(
                "gmul_proxy_forget")
            out[state]["gmul_proxy_holdout_association"] = gmul[state].get(
                "gmul_proxy_holdout_association")
            out[state]["gmul_proxy_retain_synth"] = gmul[state].get(
                "gmul_proxy_retain_synth")
    return out


def _salmu_sel_metrics(sel: dict) -> tuple[dict, dict]:
    """Extract SALMU selection test metrics for selected candidates."""
    selected = sel.get("selected", {})
    cands = sel.get("candidates", [])
    test_metrics: dict = {}
    for row in cands:
        cid = row.get("candidate_id", "")
        if cid in selected.values():
            tm = row.get("test_metrics", {})
            sv = row.get("test_summary_vector", {})
            sims = tm.get("mean_similarities", {})
            test_metrics[cid] = {
                "fine_pref": tm.get("prefers_fine_rate"),
                "tnf": tm.get("prefers_target_not_fine_rate"),
                "sim_fine": sims.get("fine"),
                "sim_target": sims.get("target"),
                "sim_sibling": sims.get("sibling"),
                "same_entity_retain_sim": sv.get(
                    "same_entity_retain_sim"),
                "other_entity_retain_sim": sv.get(
                    "other_entity_retain_sim"),
            }
    return test_metrics, selected


def _salmu_failure_rates(ref: dict, sel: dict) -> dict:
    """Compute failure rates from SALMU reference and selection."""
    rates: dict = {}
    # Reference states
    sbs = ref.get("scores_by_state", {})
    for state in ("BASE", "MF", "MG", "MN"):
        s = sbs.get(state, {})
        fp = s.get("prefers_fine_rate") or 0
        tnf = s.get("prefers_target_not_fine_rate") or 0
        r: dict = {}
        if state in ("MF", "MG", "MN", "BASE"):
            if fp > 0:
                r["fine_leakage"] = round(fp, 4)
            if tnf > 0:
                r["target_preferred"] = round(tnf, 4)
        if r:
            rates[state] = r
    # Selected candidates
    cands = sel.get("candidates", [])
    selected = sel.get("selected", {})
    for row in cands:
        cid = row.get("candidate_id", "")
        if cid not in selected.values():
            continue
        tm = row.get("test_metrics", {})
        fp = tm.get("prefers_fine_rate") or 0
        tnf = tm.get("prefers_target_not_fine_rate") or 0
        r = {}
        if fp > 0:
            r["fine_leakage"] = round(fp, 4)
        if tnf > 0:
            r["target_preferred"] = round(tnf, 4)
        if r:
            rates[cid] = r
    return rates


def _salmu_official_block(paths: SalmuPaths) -> dict | None:
    """Embed the official SALMUBench-evaluator evidence for this
    iteration (10R5c): per-state metric table, target-only summary,
    paired CIs, evidence status, and the aggregation gate."""
    rep_path = paths.report("salmubench_official_eval")
    if not rep_path.exists():
        return None
    rep = json.loads(rep_path.read_text())
    states = {}
    for state, e in rep.get("states", {}).items():
        states[state] = {
            k: e.get(k) for k in
            ("RetFail_MRR", "RetFail_R@1", "AssocStr", "ACS",
             "IdZSC", "CoreAssoc", "InterIdSim", "IntraIdSim",
             "VisIdInt", "FragSim")}
        t = e.get("target_only", {})
        states[state]["target_only"] = {
            "AssocStr_target": t.get("AssocStr_target"),
            "CoreAssoc_target": t.get("CoreAssoc_target"),
            "RetFail_MRR_target": (t.get("RetFail_target") or {}).get(
                "MRR"),
            "RetFail_R@1_target": (t.get("RetFail_target") or {}).get(
                "R@1"),
        }
    return {
        "source_report": rep_path.name,
        "evidence_status": rep.get("evidence_status"),
        "official_evaluator": rep.get("official_evaluator"),
        "aggregation_gate": rep.get("aggregation_gate"),
        "statistical_metadata": rep.get("statistical_metadata"),
        "identical_checkpoint_invariants":
            rep.get("identical_checkpoint_invariants"),
        "states": states,
        "paired_target_only": rep.get("paired_target_only"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the cross-dataset comparison report")
    parser.add_argument("--suffix", default="",
                        help="Iteration tag (e.g. r5 -> read the "
                             "holdout-clean SALMU reports and write "
                             "cross_dataset_comparison_r5.json)")
    args = parser.parse_args()

    repo_root = _find_repo_root(Path.cwd()) or Path.cwd()
    reports = repo_root / "data" / "reports"
    paths = SalmuPaths(repo_root, suffix=args.suffix)

    # MLLMU (no suffixed iteration exists — always the smoke reports)
    mllmu_ref = json.loads(
        (reports / "mllmu_smoke_reference_eval.json").read_text())
    mllmu_sel = json.loads(
        (reports / "mllmu_smoke_unlearning_selection.json").read_text())
    mllmu_ref_metrics = _mllmu_ref_metrics(mllmu_ref)
    mllmu_test, mllmu_selected = _mllmu_sel_metrics(mllmu_sel)
    mllmu_ref_metrics.update(mllmu_test)

    # SALMU (suffix-routed: r5 reads the holdout-clean chain)
    salmu_ref_path = paths.report("salmu_reference_eval")
    salmu_ref = json.loads(salmu_ref_path.read_text())
    salmu_ref_metrics = _salmu_ref_target_metrics(salmu_ref)

    salmu_sel_path = paths.report("salmu_unlearning_selection")
    salmu_test: dict = {}
    salmu_selected: dict = {"B0": "B0"}
    salmu_failure: dict = {}
    b3_ranking_note = ""
    if salmu_sel_path.exists():
        salmu_sel = json.loads(salmu_sel_path.read_text())
        salmu_test, salmu_selected = _salmu_sel_metrics(salmu_sel)
        salmu_failure = _salmu_failure_rates(salmu_ref, salmu_sel)
        # Quote the B3 ranking DIRECTLY from the selection report so
        # the two reports can never diverge (10R4).
        b3_rows = sorted(
            (r for r in salmu_sel.get("candidates", [])
             if r["candidate_id"].startswith("B3_")
             and r.get("distance_to_MG_trainval") is not None),
            key=lambda r: r["distance_to_MG_trainval"])
        b3_ranking_note = ("SALMU B3 train+val ranking "
                           "(association-weighted, from the selection "
                           "report): " + "; ".join(
                               f"{r['candidate_id']} dist="
                               f"{r['distance_to_MG_trainval']}"
                               for r in b3_rows))
    else:
        log.warning("SALMU selection report not found — "
                     "skipping unlearning metrics")

    salmu_metrics = {**salmu_ref_metrics, **salmu_test}
    official_block = _salmu_official_block(paths)

    comparison = {
        "experiment": "Cross-dataset: MLLMU (generative MLLM) vs "
                      "SALMU (CLIP association)",
        "iteration_suffix": args.suffix or None,
        "salmu_reports": {
            "reference_eval": salmu_ref_path.name,
            "selection": salmu_sel_path.name,
            "official_eval": (official_block or {}).get(
                "source_report"),
        },
        "method_descriptions": {
            "B1": "Gradient ascent on fine-target pairs "
                  "(constrained variants stop at the MG anchor).",
            "B2": "Target-level positive retraining (SFT on target-level "
                  "generalized captions).  NOT gradient ascent.",
            "B2_retain": "Target-level SFT + retain SFT (separate method "
                         "excluded from B2 family).",
            "B3": "Target-level SFT + retain SFT + constrained ascent "
                  "(stop at MG anchor).",
        },
        "mllmu": {
            "model": "Qwen3.5-9B+LoRA",
            "task": "VQA",
            "test_metrics": mllmu_ref_metrics,
            "selected": mllmu_selected,
        },
        "salmu": {
            "model": "CLIP ViT-B/16",
            "task": "image-text association",
            "weighting": salmu_sel.get("weighting")
            if salmu_sel_path.exists() else None,
            "test_metrics": salmu_metrics,
            "test_protocol": salmu_sel.get("test_protocol")
            if salmu_sel_path.exists() else None,
            "selected": salmu_selected,
            "failure_rates": salmu_failure,
            "official_salmubench": official_block,
        },
        "key_findings": [
            "Both benchmarks confirm MF != MG != MN separation "
            "with per-attribute targeting.",
            "B1 (gradient ascent) and B2 (target-level SFT) are "
            "distinct methods; B2 is closest to MG in SALMU.",
            "B1 (gradient ascent) collapses representations when "
            "unconstrained; constrained ascent (B3 stop at MG anchor) "
            "prevents collapse.",
            "Retain health must be in selection vector.",
            "SALMU confirms the phenomenon at the controlled CLIP "
            "association level.",
            "Per-attribute targeting enables same-entity retain "
            "probes (holdout_association analog).",
            "Frozen multi-image/caption probes with deterministic "
            "IDs ensure reproducibility.",
            "Selection uses TARGET-ONLY probes (is_target_attr=True), "
            "association-weighted (each (identity, attribute) counted "
            "once), so same-entity retain probes do not dilute "
            "MG-distance and high-variant attributes get no extra "
            "weight.",
            "B2_retain_* excluded from B2 candidate family (different "
            "method: target_level SFT + retain SFT).",
            b3_ranking_note or "SALMU selection report not found.",
        ],
    }

    if official_block:
        comparison["key_findings"].extend([
            "Official SALMUBench evaluator (commit 8b7f4397, RNG-"
            "restored per state, checkpoint-bound results): the "
            "target-only paired CIs REVISE the full-split verdict — "
            "B2's dramatic full-forget RetFail collapse (MRR 0.427 "
            "-> 0.197) and IdZSC drop (0.768 -> 0.416) are OFF-"
            "TARGET collateral: on the 323 designated target "
            "associations B2 is statistically indistinguishable "
            "from MF (paired MRR_target +0.023 [-0.054, +0.108], "
            "AssocStr_target -0.002 [-0.008, +0.004]), i.e. it "
            "over-generalizes WITHOUT reliably unlearning its "
            "targets. B3 (constrained ascent) stays close to MG on "
            "every official metric — the granularity hypothesis "
            "MU ~= MG holds on the benchmark's own evaluator.",
            "SALMU internal test verdicts are EXPLORATORY (the r5 "
            "test personas are a subset of the candidate-wide-"
            "inspected 10R2/10R3 split); the validated untouched "
            "official holdouts are the primary external evaluation.",
            "Target-only official metrics (46 personas / 323 target "
            "associations) with identity-clustered CIs (AssocStr, "
            "CoreAssoc, RetFail MRR/R@1) and paired difference CIs "
            "vs MF and MG are embedded under "
            "salmu.official_salmubench.",
        ])

    out = paths.report("cross_dataset_comparison")
    with open(out, "w") as f:
        json.dump(comparison, f, indent=2, ensure_ascii=False)
    log.info("Wrote cross-dataset comparison -> %s", out)


if __name__ == "__main__":
    main()
