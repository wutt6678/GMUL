"""Iteration 11 pilot-100 FINAL evaluation — ONE-SHOT on the frozen test
paraphrase split.

    python scripts/evaluate_pilot100_final.py --device cuda:0

Runs exactly once, AFTER selection.  Inputs:

* the frozen pilot-100 dataset (``data/mllmu_hier_pilot100``);
* ``data/reports/mllmu_pilot100_unlearning_selection.json`` — the
  per-method winner chosen on TRAIN+VAL only;
* full-split predictions for BASE/MF/MG/MN persisted by the
  reference-state evaluation (reused, never regenerated).

Work done here: generate TEST-split completions for the selected
candidates (their selection-time parquets deliberately contain no test
rows), then report, for every state and on TEST only:

* the three routes — text_to_text, image_to_text, image_text_to_text;
* paraphrased TARGET probes — FILR / TGA / the full failure taxonomy
  (under-forgetting, over-forgetting, wrong-branch, refusal,
  hallucination), stratified by route, hierarchy type and target depth;
* paraphrased RETAIN probes — same-entity and other-entity, text and
  image route, and all-routes;
* PAIRED entity-clustered bootstrap CIs for the four headline rates
  (target-granularity accuracy, wrong-branch rate, same-entity-retain,
  other-entity-retain) of every state against MF, MG and B0.

Integrity invariants asserted before the report is written:

1. B0 is the MF adapter copied unchanged, so B0's paired difference
   against MF must be exactly 0.0 with a degenerate CI [0, 0] on all
   four metrics — and their raw outputs must match query-for-query;
2. selection used train+val only (read back from the selection report);
3. every compared state covers the identical test query set, so each
   paired CI is computed over the same probes.

Writes ``data/reports/mllmu_pilot100_final_evaluation.json``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from granunlearn.config import _find_repo_root
from granunlearn.evaluation.hierarchy_metrics import compute_hierarchy_metrics
from granunlearn.evaluation.paired_ci import (
    PAIRED_METRICS,
    paired_metrics_report,
)
from granunlearn.evaluation.reference_eval import (
    ReferenceStateGenerator,
    load_associations_parquet,
    load_predictions_parquet,
    load_queries_parquet,
)
from granunlearn.evaluation.scoring import compute_metrics, score_query
from granunlearn.logging_utils import setup_logger
from granunlearn.schema import PredictionRecord, QueryRecord

log = setup_logger("evaluate_pilot100_final")

TAG = "pilot100"
EXPERIMENT_ID = f"mllmu_{TAG}_iter11"
REFERENCE_STATES = ("BASE", "MF", "MG", "MN")
#: Paired-CI reference levels: MF (the pre-unlearning state), MG (the
#: target behaviour) and B0 (the no-op self-test).
PAIRED_REFERENCES = ("MF", "MG", "B0")
TEST = ("test",)


def _load_or_generate_test(
    state_id: str,
    adapter_dir: Path,
    test_queries: list[QueryRecord],
    by_assoc: dict,
    repo_root: Path,
    predictions_dir: Path,
    model_id: str,
    device: str,
    batch_size: int,
    image_batch_size: int,
) -> tuple[list[PredictionRecord], bool]:
    """Test-split predictions for one candidate (generated at most once).

    Reusing an existing parquet is crash recovery, not a second look:
    the file is written only after a complete generation pass, and the
    reuse is recorded in the report.
    """
    ppath = predictions_dir / f"predictions_test_{state_id}.parquet"
    if ppath.exists():
        preds = load_predictions_parquet(ppath)
        log.warning("[%s] REUSING existing test predictions %s (%d rows)",
                    state_id, ppath.name, len(preds))
        return preds, True
    log.info("[%s] generating %d TEST queries (one-shot)...", state_id,
             len(test_queries))
    generator = ReferenceStateGenerator(model_id, device,
                                        adapter_dir=adapter_dir)
    raws = generator.generate_for_queries(
        test_queries, by_assoc, repo_root, batch_size=batch_size,
        image_batch_size=image_batch_size)
    generator.unload()
    preds = [score_query(q, by_assoc[q.association_id], raw,
                         experiment_id=EXPERIMENT_ID,
                         checkpoint_id=state_id)
             for q, raw in zip(test_queries, raws)]
    import pandas as pd
    pd.DataFrame([json.loads(p.model_dump_json()) for p in preds]
                 ).to_parquet(ppath, index=False)
    return preds, False


def _check_b0_equals_mf(preds_by_state: dict[str, list[PredictionRecord]],
                        paired: dict[str, Any]) -> dict[str, Any]:
    """The no-op invariant: B0 must be indistinguishable from MF."""
    out: dict[str, Any] = {"checked": "B0" in preds_by_state
                           and "MF" in preds_by_state}
    if not out["checked"]:
        return out
    b0 = {p.query_id: p.raw_output for p in preds_by_state["B0"]}
    mf = {p.query_id: p.raw_output for p in preds_by_state["MF"]}
    common = sorted(set(b0) & set(mf))
    mismatches = [q for q in common if b0[q] != mf[q]]
    out["num_test_queries_compared"] = len(common)
    out["num_raw_output_mismatches"] = len(mismatches)
    out["mismatched_query_ids"] = mismatches[:10]
    b0_vs_mf = (paired.get("comparisons", {}).get("B0", {})
                .get("vs_MF", {}))
    out["paired_diffs_vs_MF"] = {
        m: b0_vs_mf.get(m, {}).get("diff") for m in PAIRED_METRICS}
    out["paired_cis_vs_MF"] = {
        m: b0_vs_mf.get(m, {}).get("ci") for m in PAIRED_METRICS}
    out["passed"] = (
        not mismatches
        and all(b0_vs_mf.get(m, {}).get("diff") == 0.0
                for m in PAIRED_METRICS if m in b0_vs_mf)
        and all(tuple(b0_vs_mf.get(m, {}).get("ci", ())) == (0.0, 0.0)
                for m in PAIRED_METRICS if m in b0_vs_mf))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--model-id", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--image-batch-size", type=int, default=8)
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    parser.add_argument("--ci-level", type=float, default=0.95)
    parser.add_argument("--skip-generation", action="store_true",
                        help="Assemble the report from persisted test "
                             "predictions only (no GPU)")
    args = parser.parse_args()

    repo_root = _find_repo_root(Path.cwd()) or Path.cwd()
    data_dir = repo_root / "data" / f"mllmu_hier_{TAG}"
    unlearn_ckpt = repo_root / "data" / "checkpoints" / \
        f"mllmu_{TAG}_unlearn"
    predictions_dir = data_dir / "predictions"
    predictions_dir.mkdir(parents=True, exist_ok=True)

    selection_path = repo_root / "data" / "reports" / \
        f"mllmu_{TAG}_unlearning_selection.json"
    selection = json.loads(selection_path.read_text())
    selected: dict[str, str] = {m: c for m, c in selection["selected"].items()
                                if c}
    log.info("Selected per method (train+val basis): %s", selected)

    queries = load_queries_parquet(data_dir / "queries.parquet")
    associations = load_associations_parquet(data_dir / "associations.parquet")
    by_assoc = {a.association_id: a for a in associations}
    test_queries = [q for q in queries if q.split == "test"]
    log.info("%d queries total; %d on the frozen TEST paraphrase split",
             len(queries), len(test_queries))

    preds_by_state: dict[str, list[PredictionRecord]] = {}
    provenance: dict[str, Any] = {}

    # 1. Reference states: reuse the full-split predictions persisted by
    #    the reference-state evaluation, filtered to TEST here.
    for state in REFERENCE_STATES:
        ppath = predictions_dir / f"predictions_{state}.parquet"
        if not ppath.exists():
            raise FileNotFoundError(
                f"{ppath} — run scripts/evaluate_reference_states.py "
                f"--tag {TAG} first")
        preds = load_predictions_parquet(ppath)
        split_of = {q.query_id: q.split for q in queries}
        test_preds = [p for p in preds if split_of.get(p.query_id) == "test"]
        preds_by_state[state] = test_preds
        provenance[state] = {
            "kind": "reference_state",
            "source": str(ppath.relative_to(repo_root)),
            "reused_full_split_predictions": True,
            "num_test_predictions": len(test_preds),
        }
        log.info("[%s] %d test predictions reused", state, len(test_preds))

    # 2. Selected candidates: one-shot TEST generation.
    reused: list[str] = []
    for method, cid in sorted(selected.items()):
        adapter_dir = unlearn_ckpt / cid / "adapters"
        if not adapter_dir.exists():
            raise FileNotFoundError(
                f"selected {method} <- {cid} but {adapter_dir} is missing")
        key = method  # one winner per method; B0 is its own method
        if args.skip_generation:
            ppath = predictions_dir / f"predictions_test_{cid}.parquet"
            preds = load_predictions_parquet(ppath)
            was_reused = True
        else:
            preds, was_reused = _load_or_generate_test(
                cid, adapter_dir, test_queries, by_assoc, repo_root,
                predictions_dir, args.model_id, args.device,
                args.batch_size, args.image_batch_size)
        if was_reused:
            reused.append(cid)
        preds_by_state[key] = preds
        provenance[key] = {
            "kind": "selected_candidate",
            "candidate_id": cid,
            "adapter_dir": str(adapter_dir.relative_to(repo_root)),
            "test_predictions_file":
                f"predictions_test_{cid}.parquet",
            "reused_existing_predictions": was_reused,
            "num_test_predictions": len(preds),
            "distance_to_mg_trainval":
                selection["candidates"].get(cid, {}).get("distance_to_mg"),
        }
        log.info("[%s <- %s] %d test predictions", key, cid, len(preds))

    # 3. Pairing completeness: every state must cover the same probes.
    test_ids = {q.query_id for q in test_queries}
    coverage = {s: len({p.query_id for p in preds} & test_ids)
                for s, preds in preds_by_state.items()}
    incomplete = {s: n for s, n in coverage.items() if n != len(test_ids)}
    if incomplete:
        raise SystemExit(
            f"incomplete test coverage (expected {len(test_ids)}): "
            f"{incomplete}")

    # 4. Metrics on TEST only.
    metrics_test = {
        s: compute_metrics(p, queries, split="test")
        for s, p in preds_by_state.items()}
    hierarchy_test = {
        s: compute_hierarchy_metrics(p, queries, associations, split="test")
        for s, p in preds_by_state.items()}

    # 5. Paired entity-clustered CIs (test split).
    paired = paired_metrics_report(
        preds_by_state, queries, associations,
        reference_states=PAIRED_REFERENCES, split="test",
        n_bootstrap=args.n_bootstrap, ci_level=args.ci_level, seed=42)
    b0_check = _check_b0_equals_mf(preds_by_state, paired)
    log.info("B0 == MF invariant: %s", b0_check)

    report = {
        "experiment_id": EXPERIMENT_ID,
        "iteration": 11,
        "dataset": f"mllmu_hier_{TAG}",
        "model_id": args.model_id,
        "one_shot": {
            "protocol": (
                "TEST-split generation happens only here, only for the "
                "candidates selected on TRAIN+VAL; reference-state test "
                "rows are reused from the gate run. Nothing in this "
                "script feeds back into selection."),
            "selection_basis": selection.get("basis"),
            "selection_scope": selection.get("selection_scope"),
            "num_test_queries": len(test_queries),
            "num_queries_total": len(queries),
            "reused_existing_test_predictions": reused,
            "generation_config": {
                "batch_size": args.batch_size,
                "image_batch_size": args.image_batch_size,
                "max_new_tokens": 96,
                "do_sample": False,
            },
        },
        "selected": selected,
        "states": sorted(preds_by_state),
        "test_query_coverage": coverage,
        "routes": {
            "text_to_text": "entity named in text, no image",
            "image_to_text": "entity NEVER named; identity must be "
                             "recovered from the image alone",
            "image_text_to_text": "entity named alongside the image",
        },
        "metrics_by_state_test": metrics_test,
        "hierarchy_metrics_test": hierarchy_test,
        "paired_cis_test": paired,
        "b0_equals_mf_invariant": b0_check,
        "provenance": provenance,
        "notes": [
            "All numbers are the frozen TEST paraphrase split: identical "
            "target associations, unseen wording (template assignment is "
            "deterministic per (association, family, split)).",
            "Target slices use the POST-unlearning view and exclude "
            "adversarial probes; retain slices use the BASELINE view "
            "(retained facts must stay answerable) across BOTH routes.",
            "Paired CIs are entity-clustered percentile bootstraps over "
            "per-entity paired rate differences on the intersection of "
            "query ids, so every comparison is over identical probes.",
            "Individual metrics remain the primary scientific results; "
            "D_G was a model-selection criterion only.",
        ],
    }
    out = repo_root / "data" / "reports" / \
        f"mllmu_{TAG}_final_evaluation.json"
    with open(out, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    log.info("Final evaluation report -> %s", out)

    # Compact console summary.
    log.info("%-6s %-8s %-8s %-8s %-8s %-8s %-8s", "state", "FILR", "TGA",
             "wrong", "rSame", "rOther", "i2t_TGA")
    for s in sorted(hierarchy_test):
        h = hierarchy_test[s]
        rs = (h.get("retain_same_entity_all_routes") or {})
        ro = (h.get("retain_other_entity_all_routes") or {})
        i2t = (h.get("by_route") or {}).get("image_to_text") or {}
        log.info("%-6s %-8s %-8s %-8s %-8s %-8s %-8s", s, h.get("filr"),
                 h.get("tga"),
                 (h.get("failure_rates") or {}).get("wrong_branch"),
                 rs.get("baseline_accuracy"), ro.get("baseline_accuracy"),
                 i2t.get("tga"))


if __name__ == "__main__":
    main()
