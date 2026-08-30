"""Build cross-dataset comparison report (Iteration 10R2).

    python scripts/build_cross_dataset_comparison.py

Reads MLLMU + SALMU reference-eval and selection reports, extracts
comparable metrics, and writes
data/reports/cross_dataset_comparison.json.
"""

from __future__ import annotations

import json
from pathlib import Path

from granunlearn.config import _find_repo_root
from granunlearn.logging_utils import setup_logger

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
    """Extract SALMU reference target-attribute metrics."""
    out: dict = {}
    sbs = ref.get("scores_by_state", {})
    for state, s in sbs.items():
        sims = s.get("mean_similarities", {})
        out[state] = {
            "fine_pref": s.get("prefers_fine_rate"),
            "tnf": s.get("prefers_target_not_fine_rate"),
            "sim_fine": sims.get("fine"),
            "sim_target": sims.get("target"),
            "sim_sibling": sims.get("sibling"),
            "retain_sim": None,
        }
    # Add retain similarity from same_entity_retain_scores
    se = ref.get("same_entity_retain_scores", {})
    for state in out:
        if state in se:
            sims = se[state].get("mean_similarities", {})
            out[state]["retain_sim"] = sims.get("fine")
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
                "retain_sim": sv.get("retain_fine_sim"),
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


def main() -> None:
    repo_root = _find_repo_root(Path.cwd()) or Path.cwd()
    reports = repo_root / "data" / "reports"

    # MLLMU
    mllmu_ref = json.loads(
        (reports / "mllmu_smoke_reference_eval.json").read_text())
    mllmu_sel = json.loads(
        (reports / "mllmu_smoke_unlearning_selection.json").read_text())
    mllmu_ref_metrics = _mllmu_ref_metrics(mllmu_ref)
    mllmu_test, mllmu_selected = _mllmu_sel_metrics(mllmu_sel)
    mllmu_ref_metrics.update(mllmu_test)

    # SALMU
    salmu_ref = json.loads(
        (reports / "salmu_reference_eval.json").read_text())
    salmu_ref_metrics = _salmu_ref_target_metrics(salmu_ref)

    salmu_sel_path = reports / "salmu_unlearning_selection.json"
    salmu_test: dict = {}
    salmu_selected: dict = {"B0": "B0"}
    salmu_failure: dict = {}
    if salmu_sel_path.exists():
        salmu_sel = json.loads(salmu_sel_path.read_text())
        salmu_test, salmu_selected = _salmu_sel_metrics(salmu_sel)
        salmu_failure = _salmu_failure_rates(salmu_ref, salmu_sel)
    else:
        log.warning("SALMU selection report not found — "
                     "skipping unlearning metrics")

    salmu_metrics = {**salmu_ref_metrics, **salmu_test}

    comparison = {
        "experiment": "Cross-dataset: MLLMU (generative MLLM) vs "
                      "SALMU (CLIP association)",
        "mllmu": {
            "model": "Qwen3.5-9B+LoRA",
            "task": "VQA",
            "test_metrics": mllmu_ref_metrics,
            "selected": mllmu_selected,
        },
        "salmu": {
            "model": "CLIP ViT-B/16",
            "task": "image-text association",
            "test_metrics": salmu_metrics,
            "selected": salmu_selected,
            "failure_rates": salmu_failure,
        },
        "key_findings": [
            "Both benchmarks confirm MF != MG != MN separation "
            "with per-attribute targeting.",
            "B2 (gradient ascent) closest to MG in SALMU; B2+retain "
            "closest in MLLMU.",
            "Unconstrained gradient ascent collapses representations.",
            "Constrained ascent (stop at MG anchor) prevents collapse.",
            "Retain health must be in selection vector.",
            "SALMU confirms the phenomenon at the controlled CLIP "
            "association level.",
            "Per-attribute targeting enables same-entity retain "
            "probes (holdout_association analog).",
            "Frozen multi-image/caption probes with deterministic "
            "IDs ensure reproducibility.",
        ],
    }

    out = reports / "cross_dataset_comparison.json"
    with open(out, "w") as f:
        json.dump(comparison, f, indent=2, ensure_ascii=False)
    log.info("Wrote cross-dataset comparison -> %s", out)


if __name__ == "__main__":
    main()
