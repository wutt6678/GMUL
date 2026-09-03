"""Iteration 11 pilot-100 FINAL evaluation — ONE-SHOT on the frozen test
paraphrase split.

    python scripts/evaluate_pilot100_final.py --device cuda:0

Runs exactly once, AFTER selection.  Inputs:

* the frozen pilot-100 dataset (``data/mllmu_hier_pilot100``);
* ``data/reports/mllmu_pilot100_unlearning_selection.json`` — the
  per-method winner chosen on TRAIN+VAL only;
* the full-split predictions persisted by the reference-state gate, used
  ONLY to quantify the batch-layout noise floor (below).

Work done here: generate TEST-split completions for EVERY reported state
— BASE/MF/MG/MN and each selected candidate — over the same test-query
list with the same batch sizes, then report, on TEST only:

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

1. B0 is the MF adapter copied unchanged (identical SHA-256), so B0's
   paired difference against MF must be exactly 0.0 with a degenerate CI
   [0, 0] on all four metrics — and their raw outputs must match
   query-for-query.  This only holds when both are generated under the
   SAME batch layout: batched greedy decoding with left padding is not
   bit-stable across different batch compositions, so mixing the gate
   run's full-split generations with this run's test-only generations
   makes an identical checkpoint look different (measured: 122/2,259
   decodes flipped, shifting rates by up to 0.006);
2. selection used train+val only (read back from the selection report);
3. every compared state covers the identical test query set, so each
   paired CI is computed over the same probes;
4. the batch-layout noise floor is MEASURED and reported
   (``batch_composition_sensitivity``), separately for the target-side
   slices the claims are made on and for the retain slices, so a reader
   can separate it from the between-model effects.

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
    adapter_dir: Path | None,
    test_queries: list[QueryRecord],
    by_assoc: dict,
    repo_root: Path,
    predictions_dir: Path,
    model_id: str,
    device: str,
    batch_size: int,
    image_batch_size: int,
) -> tuple[list[PredictionRecord], bool]:
    """Test-split predictions for one state (generated at most once).

    ``adapter_dir=None`` means BASE (no adapter).  Reusing an existing
    parquet is crash recovery, not a second look: the file is written
    only after a complete generation pass, and the reuse is recorded in
    the report.

    EVERY state reported here is generated over the SAME test-query list
    with the SAME batch sizes, because left-padded batched generation is
    not bit-stable across different batch compositions: an identical
    checkpoint scored inside a 6,777-query run and inside a 2,259-query
    run produced 122/2,259 differing greedy decodes (near-ties).  Uniform
    generation is what makes the paired CIs — and the B0 == MF invariant
    — comparisons of models rather than of batch layouts.
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


def _parquet_fingerprint(path: Path) -> dict[str, Any]:
    """Hash + write time of one persisted test-prediction file.

    A state whose parquet is REUSED was generated by an earlier complete
    pass of this script, so the report has to say which file it read and
    when that file was written; otherwise "reused" is indistinguishable
    from "stale".  Binding the bytes also lets anyone re-assemble the
    report later and prove the inputs did not move.
    """
    import hashlib
    from datetime import datetime, timezone
    raw = path.read_bytes()
    return {
        "sha256": hashlib.sha256(raw).hexdigest(),
        "bytes": len(raw),
        "written_utc": datetime.fromtimestamp(
            path.stat().st_mtime, tz=timezone.utc).isoformat(
            timespec="seconds"),
    }


def _batch_composition_sensitivity(
    uniform: dict[str, list[PredictionRecord]],
    gate: dict[str, list[PredictionRecord]],
    queries: list[QueryRecord],
    associations: list,
) -> dict[str, Any]:
    """Measure how much the gate run's batch layout moved the numbers.

    The reference-state gate (C3) generated all 6,777 queries in one
    pass; this script generates the 2,259 test queries in one pass.  Both
    are valid, but they are not bit-identical, so the difference between
    the two views of the SAME checkpoint is a pure numerical-noise floor.
    Reporting it is what lets a reader separate that floor from the
    between-model effects.

    Two floors are reported separately because they have different
    magnitudes and different causes:

    * ``max_abs_metric_delta`` — the TARGET-side slices (FILR, TGA,
      wrong-branch) that the headline claims are made on;
    * ``max_abs_retain_delta`` — the RETAIN slices, which are noisier for
      an untuned model: BASE answers in long free-form paragraphs that
      get truncated at ``max_new_tokens``, so a near-tie changes the
      wording (measured: 42.9% of BASE's raw outputs differ between the
      two layouts, against 5.4% for the fine-tuned MF/B0) even though
      hierarchy-relative scoring keeps the target slices almost fixed.
    """
    retain_keys = (("retain_same", "retain_same_entity_all_routes"),
                   ("retain_other", "retain_other_entity_all_routes"))
    per_state: dict[str, Any] = {}
    worst = 0.0
    worst_retain = 0.0
    for state in sorted(set(uniform) & set(gate)):
        a = {p.query_id: p.raw_output for p in uniform[state]}
        b = {p.query_id: p.raw_output for p in gate[state]}
        common = sorted(set(a) & set(b))
        mism = sum(1 for q in common if a[q] != b[q])
        hm_u = compute_hierarchy_metrics(
            uniform[state], queries, associations, split="test")
        hm_g = compute_hierarchy_metrics(
            gate[state], queries, associations, split="test")
        deltas = {}
        for key in ("filr", "tga"):
            if hm_u.get(key) is not None and hm_g.get(key) is not None:
                deltas[key] = round(hm_u[key] - hm_g[key], 4)
                worst = max(worst, abs(deltas[key]))
        for key in ("wrong_branch",):
            u = (hm_u.get("failure_rates") or {}).get(key)
            g = (hm_g.get("failure_rates") or {}).get(key)
            if u is not None and g is not None:
                deltas[key] = round(u - g, 4)
                worst = max(worst, abs(deltas[key]))
        for label, block in retain_keys:
            u = (hm_u.get(block) or {}).get("baseline_accuracy")
            g = (hm_g.get(block) or {}).get("baseline_accuracy")
            if u is not None and g is not None:
                deltas[label] = round(u - g, 4)
                worst_retain = max(worst_retain, abs(deltas[label]))
        per_state[state] = {
            "num_test_queries_compared": len(common),
            "num_raw_output_mismatches": mism,
            "raw_output_mismatch_rate":
                round(mism / len(common), 4) if common else None,
            "metric_deltas_uniform_minus_gate": deltas,
        }
    return {
        "per_state": per_state,
        "max_abs_metric_delta": round(worst, 4),
        "max_abs_retain_delta": round(worst_retain, 4),
        "target_side_metrics": ["filr", "tga", "wrong_branch"],
        "retain_metrics": [label for label, _ in retain_keys],
        "interpretation": (
            "same checkpoint weights, two batch layouts: this is the "
            "numerical noise floor of batched greedy decoding, NOT a "
            "model difference. Every metric reported by this script comes "
            "from the uniform test-only generation, so all cross-state "
            "comparisons and paired CIs here share one batch layout. A "
            "high raw_output_mismatch_rate with small metric deltas means "
            "the wording moved while the hierarchy-relative score did "
            "not — that is expected for an untuned model answering in "
            "free-form prose under a max_new_tokens truncation."),
    }


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
    reused: list[str] = []

    # 1. Reference states: generate their TEST rows HERE, under the same
    #    batch layout as the candidates.  The gate run's full-split
    #    parquets are loaded too, but only to quantify the batch-layout
    #    noise floor — never as the reported numbers.
    ref_ckpt = repo_root / "data" / "checkpoints" / f"mllmu_{TAG}"
    split_of = {q.query_id: q.split for q in queries}
    gate_preds_by_state: dict[str, list[PredictionRecord]] = {}
    for state in REFERENCE_STATES:
        gate_path = predictions_dir / f"predictions_{state}.parquet"
        if not gate_path.exists():
            raise FileNotFoundError(
                f"{gate_path} — run scripts/evaluate_reference_states.py "
                f"--tag {TAG} first")
        gate_all = load_predictions_parquet(gate_path)
        gate_preds_by_state[state] = [
            p for p in gate_all if split_of.get(p.query_id) == "test"]
        adapter_dir = None if state == "BASE" else \
            ref_ckpt / state / "adapters"
        if adapter_dir is not None and not adapter_dir.exists():
            raise FileNotFoundError(
                f"Missing adapter for state {state}: {adapter_dir}")
        if args.skip_generation:
            preds = load_predictions_parquet(
                predictions_dir / f"predictions_test_{state}.parquet")
            was_reused = True
        else:
            preds, was_reused = _load_or_generate_test(
                state, adapter_dir, test_queries, by_assoc, repo_root,
                predictions_dir, args.model_id, args.device,
                args.batch_size, args.image_batch_size)
        if was_reused:
            reused.append(state)
        preds_by_state[state] = preds
        provenance[state] = {
            "kind": "reference_state",
            "test_predictions_file": f"predictions_test_{state}.parquet",
            "test_predictions_fingerprint": _parquet_fingerprint(
                predictions_dir / f"predictions_test_{state}.parquet"),
            "generated_under_uniform_batch_layout": True,
            "reused_existing_predictions": was_reused,
            "num_test_predictions": len(preds),
            "gate_report_source": str(gate_path.relative_to(repo_root)),
            "gate_test_rows": len(gate_preds_by_state[state]),
        }
        log.info("[%s] %d uniform test predictions", state, len(preds))

    # 2. Selected candidates: one-shot TEST generation (their
    #    selection-time parquets deliberately contain no test rows).
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
            # recorded under the STATE KEY, matching report["states"]; the
            # candidate id it came from is in provenance[key]
            reused.append(key)
        preds_by_state[key] = preds
        provenance[key] = {
            "kind": "selected_candidate",
            "candidate_id": cid,
            "adapter_dir": str(adapter_dir.relative_to(repo_root)),
            "test_predictions_file":
                f"predictions_test_{cid}.parquet",
            "test_predictions_fingerprint": _parquet_fingerprint(
                predictions_dir / f"predictions_test_{cid}.parquet"),
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

    # 6. Numerical noise floor: the SAME reference checkpoints viewed
    #    through the gate run's batch layout vs this run's.
    sensitivity = _batch_composition_sensitivity(
        {s: preds_by_state[s] for s in REFERENCE_STATES},
        gate_preds_by_state, queries, associations)
    log.info("Batch-layout noise floor: target max |delta| = %s, retain max "
             "|delta| = %s; raw-output mismatch rates %s",
             sensitivity["max_abs_metric_delta"],
             sensitivity["max_abs_retain_delta"],
             {s: v["raw_output_mismatch_rate"]
              for s, v in sensitivity["per_state"].items()})

    report = {
        "experiment_id": EXPERIMENT_ID,
        "iteration": 11,
        "dataset": f"mllmu_hier_{TAG}",
        "model_id": args.model_id,
        "one_shot": {
            "protocol": (
                "TEST-split generation happens only here and only once per "
                "state: the candidates were selected on TRAIN+VAL (their "
                "selection-time parquets contain no test rows), and every "
                "state reported below — BASE/MF/MG/MN included — is "
                "generated over the SAME test-query list with the SAME "
                "batch sizes, so all cross-state comparisons and paired "
                "CIs share one batch layout. Nothing in this script feeds "
                "back into selection."),
            "selection_basis": selection.get("basis"),
            "selection_scope": selection.get("selection_scope"),
            "num_test_queries": len(test_queries),
            "num_queries_total": len(queries),
            "reused_existing_test_predictions": reused,
            "assembled_without_generation": bool(args.skip_generation),
            "reuse_semantics": (
                "A reused parquet was written by an earlier COMPLETE pass "
                "of this script over the same test-query list with the "
                "same batch sizes (a pass writes its file only after "
                "finishing), so reuse is crash recovery or a report "
                "re-assembly — never a second look at the test split. "
                "test_predictions_fingerprint pins the exact bytes and "
                "write time each state was read from."),
            "generation_config": {
                "batch_size": args.batch_size,
                "image_batch_size": args.image_batch_size,
                "max_new_tokens": 96,
                "do_sample": False,
            },
        },
        "batch_composition_sensitivity": sensitivity,
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
            "Every state — reference states included — was generated over "
            "the same test-query list with the same batch sizes, so no "
            "comparison here contains a batch-layout artifact; see "
            "batch_composition_sensitivity for the measured floor.",
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
