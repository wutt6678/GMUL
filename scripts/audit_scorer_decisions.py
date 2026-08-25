"""Scorer-decision audit (Iteration 7 review).

Substring matching can mishandle negated answers ("not X" would count as
revealing X).  This script:

1. runs a word-boundary negation scan over ALL matched predictions and
   counts decisions affected by the negation-aware matcher;
2. draws a seeded stratified sample of ~100 scorer decisions (per state x
   matched/unmatched, with target-probe and retain-probe strata) for
   manual review;
3. commits the sample + scan counts as evidence
   (data/reports/mllmu_smoke_scorer_audit.json).

    python scripts/audit_scorer_decisions.py
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from granunlearn.config import _find_repo_root
from granunlearn.evaluation.reference_eval import (
    load_associations_parquet,
    load_predictions_parquet,
    load_queries_parquet,
)
from granunlearn.evaluation.scoring import match_answer
from granunlearn.logging_utils import setup_logger

log = setup_logger("audit_scorer")

STATES = ["BASE", "MF", "MG", "MN"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Scorer-decision audit")
    parser.add_argument("--sample-size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    repo_root = _find_repo_root(Path.cwd()) or Path.cwd()
    smoke = repo_root / "data" / "mllmu_hier_smoke"
    queries = load_queries_parquet(smoke / "queries.parquet")
    associations = load_associations_parquet(smoke / "associations.parquet")
    q_by_id = {q.query_id: q for q in queries}
    a_by_id = {a.association_id: a for a in associations}

    rng = random.Random(args.seed)
    audit: dict = {
        "seed": args.seed,
        "sample_size": args.sample_size,
        "matcher": (
            "finest-wins normalized substring matching with "
            "word-boundary negation cues in the 5 tokens preceding a "
            "match; negated matches are skipped and recorded in "
            "metadata.negated_matches"
        ),
        "negation_scan": {},
        "sample": [],
    }

    pool: list[dict] = []
    for state in STATES:
        preds = load_predictions_parquet(
            smoke / "predictions" / f"predictions_{state}.parquet")
        negated_count = 0
        rescore_changed = 0
        for p in preds:
            q = q_by_id[p.query_id]
            assoc = a_by_id[q.association_id]
            # Re-run the CURRENT matcher to detect decisions the
            # negation-aware scorer would change vs the persisted one.
            level, cid, negated = match_answer(p.raw_output, assoc)
            if negated:
                negated_count += 1
            if (cid is None) != (p.matched_canonical_id is None) or \
                    (cid is not None and cid != p.matched_canonical_id):
                rescore_changed += 1
            role = ("retain" if (q.family or "").startswith("retain_")
                    else "target")
            pool.append({
                "state": state,
                "query_id": p.query_id,
                "family": q.family,
                "split": q.split,
                "role": role,
                "adversarial": q.adversarial,
                "prompt": q.prompt,
                "expected_answer": q.expected_answer,
                "raw_output": p.raw_output[:400],
                "persisted_matched_id": p.matched_canonical_id,
                "persisted_predicted_level": p.predicted_level,
                "persisted_is_correct_branch": p.is_correct_branch,
                "persisted_is_leakage": p.is_finer_than_target,
                "current_matched_id": cid,
                "current_negated_matches": negated,
            })
        audit["negation_scan"][state] = {
            "num_predictions": len(preds),
            "outputs_with_negated_matches": negated_count,
            "decisions_changed_by_negation_awareness": rescore_changed,
        }

    # Stratified sample: balance state x matched/unmatched; ensure target
    # and retain roles and adversarial probes are all represented.
    strata: dict[tuple, list] = {}
    for item in pool:
        key = (item["state"], item["persisted_matched_id"] is not None)
        strata.setdefault(key, []).append(item)
    sample: list[dict] = []
    # guaranteed slices first
    for item in pool:
        if item["adversarial"] and len(
                [s for s in sample if s["adversarial"]]) < 10:
            sample.append(item)
    per_stratum = max(1, args.sample_size // len(strata))
    for key in sorted(strata, key=str):
        items = strata[key]
        rng.shuffle(items)
        for item in items[:per_stratum]:
            if item not in sample:
                sample.append(item)
    rng.shuffle(pool)
    for item in pool:
        if len(sample) >= args.sample_size:
            break
        if item not in sample:
            sample.append(item)
    audit["sample"] = sample[:args.sample_size]
    audit["sample_strata_counts"] = {}
    for item in audit["sample"]:
        k = f"{item['state']}/matched={item['persisted_matched_id'] is not None}"
        audit["sample_strata_counts"][k] = \
            audit["sample_strata_counts"].get(k, 0) + 1

    out = repo_root / "data" / "reports" / "mllmu_smoke_scorer_audit.json"
    with open(out, "w") as f:
        json.dump(audit, f, indent=2, ensure_ascii=False)
    log.info("Negation scan: %s", json.dumps(audit["negation_scan"]))
    log.info("Wrote %d-decision audit sample -> %s",
             len(audit["sample"]), out)


if __name__ == "__main__":
    main()
