"""Re-score Stage 1 and Stage 1b on the route-stratified retention floor.

This spends no GPU.  Every prediction it reads was already generated: Stage 1
covered all 4,518 train+val queries for ten states and Stage 1b did the same for
seven, and the image-route retain families are inside those parquets.  What this
script adds is a second instrument on the same never-rehearsed associations.

WHAT IT DOES AND WHAT IT DOES NOT DO
------------------------------------
It reports what an eight-number floor would have said about the candidates and
replicates that already exist, and it preregisters that floor for Stage 2.  It
re-decides nothing: the filed Stage-1 and Stage-1b verdicts stand as filed, no
candidate is selected, and the four text-route numbers are required to come out
exactly as they did before.

THE CONSISTENCY GATE, AND WHY IT RUNS BEFORE ANYTHING IS WRITTEN
----------------------------------------------------------------
An extension of a rule has to agree with the rule it extends on the part they
share, or it is a different rule.  So for every state this script recomputes the
frozen four-number floor and compares it against the FILED Stage-1 and Stage-1b
documents number by number and flag by flag, and refuses to write the report if
any of them differ.  That gate is what makes "the text stratum reproduces the
filed results" a measured property of this run rather than a claim in a commit
message.

WHAT STRATIFYING DOES AND DOES NOT BUY
--------------------------------------
It does NOT enlarge any denominator.  Each of the eight numbers still sits on
408 or 76 queries, exactly as the frozen four did; stratifying adds four
numbers, it does not widen any of them.  The n-doubling that a pooled
``*_all_routes`` rate would give is reported beside the decision and floors
nothing, because pooling would average two instruments that measurably do not
agree.

What it does buy is a second measurement whose effect size is larger.  The
image route loses roughly 2.5x more retention than the text route across the
Stage-1 candidates, so a shortfall that sits inside the between-seed spread on
the text route can sit well outside it on the image route.  Whether that is true
of these replicates is computed in the ``power`` block below rather than
asserted: for each parent and each number it reports the between-seed sd, the
mean's distance from the anchor, and whether the spread is smaller than the
shortfall it would have to adjudicate.

Nothing here reads the sealed confirmation split, and nothing edits a frozen
path.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import build_iter12_route_probe as brp
import freeze_iter12_route_stratification as frs

from granunlearn.config import _find_repo_root
from granunlearn.evaluation import retention_selection as rs
from granunlearn.evaluation import route_stratified_retention as rt
from granunlearn.evaluation.reference_eval import (
    load_associations_parquet,
    load_predictions_parquet,
    load_queries_parquet,
)
from granunlearn.logging_utils import setup_logger
from granunlearn.training.seed_replication import (
    ALL_SEEDS,
    PARENTS,
    aggregate,
    range_classification,
    replicates,
)

log = setup_logger(__name__)

OUT_REPORT = frs.ANALYSIS_REPORT

#: The four numbers the frozen floor reads, as flat stratified keys.  Used for
#: the consistency gate, so the gate compares like with like.
TEXT_KEYS = tuple(k for k in rt.FLOOR_NUMBER_KEYS if k.startswith("text."))
IMAGE_KEYS = tuple(k for k in rt.FLOOR_NUMBER_KEYS if k.startswith("image."))

#: Filed key -> stratified key, for the four frozen numbers.
TEXT_TO_FILED = {k: k[len("text."):] for k in TEXT_KEYS}


def score_state(preds, queries, associations, probe) -> dict[str, Any]:
    """One state's eight numbers, both strata, plus the pooled view."""
    by_route = rt.probe_retention_by_route(preds, queries, associations, probe)
    return {"by_route": by_route, "values": rt.floor_vector(by_route)}


def gate_text_stratum_against_stage1(states: dict[str, dict],
                                     filed: dict) -> dict[str, Any]:
    """Every Stage-1 candidate's four text numbers vs the filed floor.

    Compares the value, the difference and the pass flag, because a value that
    matches while a flag differs would mean the comparison changed rather than
    the measurement.
    """
    problems: list[str] = []
    checked = 0
    anchor_text = {TEXT_TO_FILED[k]: states["B0"]["values"][k] for k in TEXT_KEYS}
    for cid, block in filed["candidates"].items():
        if cid not in states:
            problems.append(f"{cid}: filed as a candidate but not scored here")
            continue
        for filed_key, check in block["floor"]["checks"].items():
            key = f"text.{filed_key}"
            got = states[cid]["values"][key]
            want_diff = round(got - anchor_text[filed_key], 6)
            if got != check["candidate"]:
                problems.append(
                    f"{cid}.{filed_key}: value {got} != filed "
                    f"{check['candidate']}")
            if want_diff != check["difference"]:
                problems.append(
                    f"{cid}.{filed_key}: difference {want_diff} != filed "
                    f"{check['difference']}")
            want_pass = got >= check["b0"] - rs.FLOOR_EPSILON
            if want_pass != check["passes"]:
                problems.append(
                    f"{cid}.{filed_key}: pass flag {want_pass} != filed "
                    f"{check['passes']}")
            checked += 1
    return {"numbers_compared": checked,
            "candidates_compared": len(filed["candidates"]),
            "problems": problems,
            "reproduces_the_filed_stage1_floor_exactly": not problems}


def gate_text_stratum_against_stage1b(per_replicate: dict[str, dict],
                                      anchor_text: dict[str, float],
                                      filed: dict) -> dict[str, Any]:
    """Every replicate's four text numbers vs the filed Stage-1b report."""
    problems: list[str] = []
    checked = 0
    for parent, verdict in filed["verdicts"].items():
        ids = verdict["replicates"]
        for filed_key, block in verdict["per_number"].items():
            key = f"text.{filed_key}"
            got = [per_replicate[rid]["values"][key] for rid in ids]
            if got != block["values"]:
                problems.append(
                    f"{parent}.{filed_key}: replicate values {got} != filed "
                    f"{block['values']}")
            agg = aggregate(got)
            for field in ("mean", "min", "max", "sd"):
                if abs(agg[field] - block[field]) > 1e-12:
                    problems.append(
                        f"{parent}.{filed_key}: {field} {agg[field]} != filed "
                        f"{block[field]}")
            want_diff = agg["mean"] - anchor_text[filed_key]
            if abs(want_diff - block["mean_minus_anchor"]) > 1e-12:
                problems.append(
                    f"{parent}.{filed_key}: mean_minus_anchor {want_diff} != "
                    f"filed {block['mean_minus_anchor']}")
            checked += 1
    return {"numbers_compared": checked,
            "replicates_compared": len(per_replicate),
            "problems": problems,
            "reproduces_the_filed_stage1b_numbers_exactly": not problems}


def power_block(per_replicate: dict[str, dict], anchor: dict[str, float],
                pooled: dict[str, dict]) -> dict[str, Any]:
    """Is the between-seed spread smaller than the shortfall it adjudicates?

    This is the question Stage 1b could not answer, and it is asked per stratum
    rather than assumed.  ``decidable`` means the spread is smaller than the
    distance from the anchor, so the sign of that distance is not an artifact of
    which seeds happened to be run.  A number can be undecidable on one route
    and decidable on the other, and that is the finding -- not a defect of the
    stratification.
    """
    out: dict[str, Any] = {"what_decidable_means": (
        "sd across the four seeds is smaller than |mean - anchor|, so the "
        "direction of the shortfall survives the seed choice. It is not a "
        "significance test; k=4 and no distributional assumption is made."),
        "stratifying_does_not_enlarge_any_denominator": True,
        "parents": {}}
    for parent in PARENTS:
        ids = [r for r in per_replicate if r.startswith(parent)]
        ids = sorted(ids, key=lambda r: ALL_SEEDS.index(int(r.split("__s")[1])))
        per: dict[str, Any] = {}
        for key in rt.FLOOR_NUMBER_KEYS:
            vals = [per_replicate[r]["values"][key] for r in ids]
            agg = aggregate(vals)
            dist = agg["mean"] - anchor[key]
            per[key] = {
                "n_queries": _denominator(per_replicate[ids[0]], key),
                "values": vals,
                "mean": agg["mean"], "sd": agg["sd"],
                "min": agg["min"], "max": agg["max"],
                "mean_minus_anchor": dist,
                "sd_over_abs_shortfall": (abs(agg["sd"] / dist)
                                          if dist else None),
                "decidable": abs(agg["sd"]) < abs(dist) if dist else False,
                "range_classification": range_classification(
                    vals, anchor[key], rs.FLOOR_EPSILON),
            }
        #: The pooled all-routes view, reported beside the floor and flooring
        #: nothing.  This is the only place a denominator actually doubles.
        pooled_per = {}
        for family in rt.POOLED_FAMILIES:
            vals = [pooled[(parent, r)][family]["row_micro"] for r in ids]
            agg = aggregate(vals)
            pooled_per[family] = {
                "n_queries": pooled[(parent, ids[0])][family]["num_queries"],
                "values": vals, "mean": agg["mean"], "sd": agg["sd"],
                "floored": False,
            }
        per["pooled_all_routes_reported_not_floored"] = pooled_per
        out["parents"][parent] = per
    return out


def _denominator(state: dict[str, Any], key: str) -> int | None:
    """The query count behind one of the eight numbers, read not remembered.

    Both estimands share a denominator here; entity-macro's RESOLUTION is
    coarser than 1/n because it averages per-entity means, and the basis report
    carries that per number rather than this one repeating it.
    """
    route, family, _ = key.split(".")
    block = (state["by_route"]["routes"].get(route) or {}).get(family) or {}
    return block.get("num_queries")


def build_report(repo_root: Path) -> dict[str, Any]:
    """Score everything, run every gate, and refuse if a gate fails."""
    reasons = frs.verify_freeze(repo_root)
    if reasons:
        raise SystemExit(
            f"REFUSING to score: the route-stratification freeze does not match "
            f"the repository ({len(reasons)} mismatch(es)), e.g. {reasons[:2]}")
    preds_ok = frs.verify_predictions(repo_root)
    if preds_ok["mismatch"]:
        raise SystemExit(
            f"REFUSING to score: {len(preds_ok['mismatch'])} bound prediction "
            f"parquet(s) have changed since the freeze, e.g. "
            f"{preds_ok['mismatch'][:2]}")

    #: The image stratum compares image-route numbers across BOTH prediction
    #: directories, because each parent's seed-42 replicate is read from Stage 1
    #: and its other three from Stage 1b.  The image route is the one whose
    #: correctness flips when batch composition changes, so one contract across
    #: every bound parquet is a precondition of the comparison rather than a
    #: nicety.  The frozen text-route floor never needed this: all 484 of its
    #: probe queries are text_to_text.
    contract = frs.verify_contract(repo_root)
    if not contract["one_instrument"]:
        raise SystemExit(
            f"REFUSING to score: the bound predictions are not one measurement "
            f"instrument ({len(contract['problems'])} problem(s)), e.g. "
            f"{contract['problems'][:2]}. Comparing image-route retention "
            f"across two generation contracts would put the headline number on "
            f"two different instruments.")

    #: The basis must still re-derive from the dataset alone, with no
    #: prediction read.  Re-checking it here means a stale basis cannot be
    #: scored against.
    fresh_basis = brp.build_report(repo_root)
    committed_basis = json.loads((repo_root / brp.OUT_REPORT).read_text())
    if committed_basis != fresh_basis:
        raise SystemExit(
            f"{brp.OUT_REPORT} does not re-derive from the dataset; refusing "
            f"to score against a basis that has drifted")

    data_dir = repo_root / "data/mllmu_hier_pilot100"
    queries = load_queries_parquet(data_dir / "queries.parquet")
    associations = load_associations_parquet(data_dir / "associations.parquet")
    probe = json.loads(
        (repo_root / brp.PARTITION_REPORT).read_text()
    )["halves"]["probe"]["entities"]

    paths = frs.prediction_paths(repo_root)
    stage1_filed = json.loads((repo_root / frs.STAGE1_SELECTION_REPORT
                               ).read_text())
    stage1b_filed = json.loads((repo_root / frs.STAGE1B_REPORT).read_text())

    def load(key: str):
        rel = paths[key]
        p = repo_root / rel
        if not p.exists():
            raise SystemExit(
                f"{key}: {rel} is bound by the freeze but absent, so the "
                f"re-scoring cannot be completed from existing predictions")
        return load_predictions_parquet(p)

    #: ---- Stage 1: ten states, eight numbers each ----
    stage1_states: dict[str, dict] = {}
    for state in frs.stage1_states():
        stage1_states[state] = score_state(load(f"stage1.{state}"), queries,
                                           associations, probe)
    anchor = stage1_states["B0"]["values"]
    anchor_text = {TEXT_TO_FILED[k]: anchor[k] for k in TEXT_KEYS}

    filed_anchor = stage1b_filed["anchor"]["values"]
    anchor_problems = [
        f"{k}: {anchor_text[k]} != filed {filed_anchor[k]}"
        for k in sorted(filed_anchor) if anchor_text.get(k) != filed_anchor[k]]

    candidates: dict[str, Any] = {}
    for cid in stage1_filed["candidates"]:
        res = rt.floor_check_stratified(stage1_states[cid]["values"], anchor)
        rt.text_stratum_reproduces_the_frozen_floor(res)
        diff = {k: round(stage1_states[cid]["values"][k] - anchor[k], 6)
                for k in rt.FLOOR_NUMBER_KEYS}
        candidates[cid] = {
            "values": stage1_states[cid]["values"],
            "difference_vs_b0": diff,
            "floor": res,
            "eligible_on_the_eight": res["eligible"],
            "eligible_on_the_frozen_four":
                stage1_filed["candidates"][cid]["floor"]["eligible"],
            "per_route": res["per_route"],
            "pooled_all_routes":
                stage1_states[cid]["by_route"]["pooled_all_routes"],
        }

    #: ---- Stage 1b: eight replicates, aggregated per parent ----
    per_replicate: dict[str, dict] = {}
    pooled_by_rep: dict[tuple[str, str], dict] = {}
    for r in replicates():
        key = f"stage1b.{r.replicate_id}"
        sc = score_state(load(key), queries, associations, probe)
        per_replicate[r.replicate_id] = sc
        pooled_by_rep[(r.parent, r.replicate_id)] = \
            sc["by_route"]["pooled_all_routes"]

    gate1b = gate_text_stratum_against_stage1b(per_replicate, anchor_text,
                                               stage1b_filed)
    gate1 = gate_text_stratum_against_stage1(stage1_states, stage1_filed)

    verdicts: dict[str, Any] = {}
    for parent in PARENTS:
        ids = sorted((r for r in per_replicate if r.startswith(parent)),
                     key=lambda r: ALL_SEEDS.index(int(r.split("__s")[1])))
        per_number = {}
        for key in rt.FLOOR_NUMBER_KEYS:
            vals = [per_replicate[r]["values"][key] for r in ids]
            agg = aggregate(vals)
            per_number[key] = {
                **agg,
                "mean_minus_anchor": agg["mean"] - anchor[key],
                "range_classification": range_classification(
                    vals, anchor[key], rs.FLOOR_EPSILON),
            }
        mean_vector = {k: per_number[k]["mean"] for k in rt.FLOOR_NUMBER_KEYS}
        res = rt.floor_check_stratified(mean_vector, anchor)
        rt.text_stratum_reproduces_the_frozen_floor(res)
        filed_v = stage1b_filed["verdicts"][parent]
        verdicts[parent] = {
            "replicates": ids,
            "per_number": per_number,
            "floor_on_the_mean": res,
            "passes_on_the_mean_eight_numbers": res["eligible"],
            "passes_on_the_mean_frozen_four": filed_v["passes_on_the_mean"],
            "per_route": res["per_route"],
            "shortfall_in_queries": {
                k: (round(per_number[k]["mean_minus_anchor"] * n, 4)
                    if (n := _denominator(per_replicate[ids[0]], k)) else None)
                for k in rt.FLOOR_NUMBER_KEYS},
        }

    #: ---- the gates, all of which must pass before anything is written ----
    gates = {
        "freeze_matches_the_repository": not reasons,
        "bound_predictions_unchanged": not preds_ok["mismatch"],
        "predictions_absent_from_this_clone": len(preds_ok["absent"]),
        "one_measurement_contract": contract["one_instrument"],
        "measurement_contract": contract,
        "basis_re_derives_from_the_dataset": committed_basis == fresh_basis,
        "anchor_text_numbers_match_the_filed_anchor": not anchor_problems,
        "anchor_problems": anchor_problems,
        "stage1_text_stratum": gate1,
        "stage1b_text_stratum": gate1b,
    }
    gates["all_passed"] = all([
        gates["freeze_matches_the_repository"],
        gates["bound_predictions_unchanged"],
        gates["one_measurement_contract"],
        gates["basis_re_derives_from_the_dataset"],
        gates["anchor_text_numbers_match_the_filed_anchor"],
        gate1["reproduces_the_filed_stage1_floor_exactly"],
        gate1b["reproduces_the_filed_stage1b_numbers_exactly"],
        gates["predictions_absent_from_this_clone"] == 0,
    ])
    if not gates["all_passed"]:
        raise SystemExit(
            "REFUSING to write: a consistency gate failed. The stratified "
            f"floor must reproduce the filed text-route results exactly; "
            f"gates={json.dumps({k: v for k, v in gates.items() if k != 'all_passed'}, default=str)[:1500]}")

    power = power_block(per_replicate, anchor, pooled_by_rep)

    #: B0 is the anchor, so it is trivially eligible against itself and is
    #: filed as a candidate row for completeness.  Counting it as a candidate
    #: would make every summary number here one too generous, so the summary
    #: counts the eight B4 rows only and says which it counted.
    b4 = [cid for cid in candidates if cid != "B0"]
    worse_on_image = sum(
        1 for cid in b4
        if candidates[cid]["difference_vs_b0"][
            "image.retain_other_entity_image.row_micro"]
        < candidates[cid]["difference_vs_b0"][
            "text.retain_other_entity.row_micro"])

    #: MG is the reference D_G targets, not a candidate, and the floor does not
    #: apply to it.  Its eight numbers are reported because whether the
    #: granularity-controlled reference shows the same route asymmetry is
    #: evidence about the asymmetry's origin, and it is already generated.
    reference = {}
    for state in ("MG",):
        reference[state] = {
            "floored": False,
            "values": stage1_states[state]["values"],
            "difference_vs_b0": {
                k: round(stage1_states[state]["values"][k] - anchor[k], 6)
                for k in rt.FLOOR_NUMBER_KEYS},
            "pooled_all_routes":
                stage1_states[state]["by_route"]["pooled_all_routes"],
        }

    return {
        "iteration": "12",
        "stage": "1c",
        "purpose": (
            "What an eight-number, route-stratified retention floor says about "
            "the candidates and replicates that already exist, and the floor "
            "Stage 2 will be judged on. Re-decides nothing."),
        "re_decides_nothing": True,
        "trains_nothing": True,
        "gpu_hours_spent": 0,
        "generated_from": {
            "freeze": frs.OUT_REPORT,
            "basis": brp.OUT_REPORT,
            "stage1_selection": frs.STAGE1_SELECTION_REPORT,
            "stage1b_report": frs.STAGE1B_REPORT,
            "stage1b_repair_record": frs.STAGE1B_REPAIR,
            "predictions_bound": len(paths),
            "anchor_state": "B0",
            "anchor_source": paths["stage1.B0"],
        },
        "consistency_gates": gates,
        "floor_numbers": list(rt.FLOOR_NUMBER_KEYS),
        "anchor": {
            "state": "B0",
            "values": anchor,
            "text_numbers_equal_the_filed_anchor": not anchor_problems,
            "denominators": {k: _denominator(stage1_states["B0"], k)
                             for k in rt.FLOOR_NUMBER_KEYS},
        },
        "stage1": {
            "candidates": candidates,
            "reference_states_not_floored": reference,
            "summary": {
                "counted": b4,
                "num_candidates": len(b4),
                "b0_excluded_because_it_is_the_anchor": True,
                "eligible_on_the_eight_numbers": sum(
                    1 for cid in b4 if candidates[cid]["eligible_on_the_eight"]),
                "eligible_on_the_frozen_four": sum(
                    1 for cid in b4
                    if candidates[cid]["eligible_on_the_frozen_four"]),
                "worse_on_the_image_route_than_on_text": worse_on_image,
                "verdict_is_unchanged_by_stratifying": all(
                    candidates[cid]["eligible_on_the_eight"]
                    == candidates[cid]["eligible_on_the_frozen_four"]
                    for cid in b4),
            },
        },
        "stage1b": {"verdicts": verdicts, "power": power},
        "disclosure": {
            "basis_identified_after_stage1b_was_scored": True,
            "direction_is_against_interest": (
                "Stratifying adds four conditions and makes the floor harder. "
                f"The image route is worse than the text route for "
                f"{worse_on_image} of {len(b4)} candidates on "
                "retain-other row-micro, so the extension enlarges the "
                "failure it reports."),
            "what_stratifying_does_not_do": (
                "It does not enlarge any denominator. Each of the eight "
                "numbers still sits on 408 or 76 queries. The pooled "
                "all-routes rate, which does double n, is reported beside the "
                "decision and floors nothing."),
            "pooled_was_computed_and_not_adopted": (
                "Pooling, stratifying and dropping the image route were all "
                "computed. Stratifying is frozen because pooling averages two "
                "instruments that measurably disagree; the pooled numbers are "
                "in this report so the choice is visible rather than "
                "asserted."),
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check-only", action="store_true",
                    help="Rebuild the report in memory and compare it with the "
                         "filed one, without writing")
    ap.add_argument("--stdout", action="store_true",
                    help="Print the report instead of writing it")
    ap.add_argument("--repo-root", default=None)
    args = ap.parse_args()
    repo_root = Path(args.repo_root) if args.repo_root \
        else (_find_repo_root(Path.cwd()) or Path.cwd())
    out = repo_root / OUT_REPORT

    report = build_report(repo_root)
    text = json.dumps(report, indent=1, sort_keys=True) + "\n"

    if args.stdout:
        print(text, end="")
        return

    if args.check_only:
        if not out.exists():
            raise SystemExit(f"{OUT_REPORT} is absent -- nothing to check")
        if json.loads(out.read_text()) != report:
            raise SystemExit(
                f"{OUT_REPORT} does not match what the bound predictions "
                f"produce; refusing to overwrite it silently")
        log.info("OK: the filed route-stratified report re-derives exactly "
                 "from the bound predictions")
        return

    #: The gates already ran inside build_report; this is the second half of
    #: the ordering rule.  A report is written once, and a second run that
    #: would change it is a refusal rather than an overwrite.
    if out.exists():
        previous = json.loads(out.read_text())
        if previous != report:
            raise SystemExit(
                f"REFUSING to overwrite {OUT_REPORT}: a fresh scoring differs "
                f"from the filed one. Inspect the difference and decide "
                f"deliberately; a result that can be quietly replaced is not a "
                f"result.")
        log.info("%s already exists and re-derives exactly; nothing written",
                 OUT_REPORT)
        return

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text)
    s1 = report["stage1"]["summary"]
    log.info("Wrote %s", out)
    log.info("Stage 1: %d/%d candidates eligible on eight numbers "
             "(was %d/%d on four); verdict unchanged: %s",
             s1["eligible_on_the_eight_numbers"], s1["num_candidates"],
             s1["eligible_on_the_frozen_four"], s1["num_candidates"],
             s1["verdict_is_unchanged_by_stratifying"])
    log.info("image route worse than text for %d/%d candidates",
             s1["worse_on_the_image_route_than_on_text"],
             s1["num_candidates"])
    for parent, v in report["stage1b"]["verdicts"].items():
        log.info("%s: passes on eight=%s, on the frozen four=%s, per_route=%s",
                 parent, v["passes_on_the_mean_eight_numbers"],
                 v["passes_on_the_mean_frozen_four"], v["per_route"])


if __name__ == "__main__":
    main()
