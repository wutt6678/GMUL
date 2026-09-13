"""The Stage-2 measurement basis: written down before anything is trained.

    python scripts/build_iter12_stage2_basis.py
    python scripts/build_iter12_stage2_basis.py --check-only

Stage 2 makes two decisions that a later reader could otherwise suspect were
made after seeing its result.  This document records both, with the numbers
they rest on, and ``--check-only`` re-derives the whole thing from committed
artifacts so the record is checkable rather than merely dated.

DECISION 1 — WHAT THE FLOOR IS ANCHORED AT
------------------------------------------
The frozen eight-number floor compares a candidate against B0, and B0 is MF
copied: a state that performs no unlearning at all.  ``README.md`` flagged
before Stage 1 ran that Stage 2 would return the no-op again unless the anchor
itself were reconsidered, and that this had to be decided BEFORE Stage 2 was
frozen rather than after it was scored.

It is reconsidered here, on a comparison that needs no prediction at all.  MG
and MF are trained by the same recipe on datasets whose retained-association
completions are BYTE-IDENTICAL — all 387 of them — and which differ on all 90
target completions.  Both cover the probe half.  So ``MG - B0`` isolates the
retention cost of the granularity transformation itself, with recipe, retained
knowledge and information held equal.  That cost is up to -0.0870, and MG —
the oracle that D_G measures distance TO — clears the B0-anchored floor on only
2 of the 8 numbers.  A candidate that achieved ``MU = MG`` exactly, which is
this project's stated definition of success, would be disqualified by it.

So Stage 2's decision-bearing floor is anchored at MG: a candidate may pay what
performing the transformation costs the oracle, and no more.  The B0-anchored
verdict is still computed and reported for every Stage-2 state, unchanged, so
nothing is hidden by the switch.

Two properties make this a re-anchoring rather than a goalpost move, and both
are computed here rather than asserted:

* it changes no filed verdict.  Stage 1 is 0 of 8 eligible under EITHER anchor.
* it does not licence inaction.  B0 itself fails the MG-anchored floor on 2 of
  8, so the no-op is not eligible under the new rule either.

DECISION 2 — WHICH MECHANISM TO TRAIN
-------------------------------------
Stage 2 was preregistered as an MF-preservation regularizer.  The calibration
pass in ``mllmu_iter12_anchor_calibration.json`` measured that plan against the
retention numbers Stage 1c filed, and it does not survive: the quantity a KL
anchor controls is not the quantity the floor measures, and the anchor's own
optimum is MF, which is no unlearning at all.  What ranks the candidates is how
far the suppression term drove the fine fact, and how far past MG's own level
it went — an overshoot that buys nothing at D_G.

Both mechanisms are therefore trained.  ``B5`` bounds the ascent; ``B6`` is the
preregistered anchor at its derived strength, run because an observational
argument over eight points is not an experiment.

WHAT THIS DOCUMENT READS
------------------------
Committed reports, the knowledge-group jsonl files and the frozen partition.
It reads NO prediction parquet and generates nothing: every retention number in
here was filed by Stage 1, Stage 1b or Stage 1c and is quoted from those
reports rather than recomputed, so a Stage-2 outcome could not have reached it
even in principle — none existed when this was written.  That is also why
``--check-only`` can re-derive it byte for byte.
"""

from __future__ import annotations

import argparse
import json
from itertools import pairwise
from pathlib import Path
from typing import Any

from granunlearn.config import _find_repo_root
from granunlearn.evaluation import retention_selection as rs
from granunlearn.evaluation import route_stratified_retention as rt
from granunlearn.logging_utils import setup_logger
from granunlearn.training import stage2_grid as sg
from granunlearn.training.preservation_trainer import validate_stage2_groups

log = setup_logger("build_iter12_stage2_basis")

FLOOR_KEYS: tuple[str, ...] = rt.FLOOR_NUMBER_KEYS

ROUTE_PROBE_REPORT = "data/reports/mllmu_iter12_route_probe.json"
PROBE_REPORT = "data/reports/mllmu_iter12_retention_probe.json"
STAGE1_SELECTION_REPORT = "data/reports/mllmu_iter12_retention_selection.json"

#: The scope Stage 1c's eight numbers are measured on, as the frozen basis
#: names it.  Looked up by key rather than by position so a change in the
#: basis document's ordering cannot silently change which resolution Stage 2
#: quotes.
CEILING_SCOPE = "train+val, both routes (what Stage 1c reads)"

#: What the basis is a function of.  Listed so ``--check-only`` can say what it
#: re-derived from, and so a reviewer can see that no prediction parquet is
#: among them.
GENERATED_FROM = (
    sg.ROUTE_STRATIFIED_REPORT,
    STAGE1_SELECTION_REPORT,
    sg.CALIBRATION_REPORT,
    ROUTE_PROBE_REPORT,
    PROBE_REPORT,
)


# ── statistics over a handful of points ─────────────────────────────────

def ranks(xs: list[float]) -> list[float]:
    """Average ranks with ties shared — the Spearman convention."""
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    out = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        avg = (i + j) / 2 + 1
        for k in range(i, j + 1):
            out[order[k]] = avg
        i = j + 1
    return out


def pearson(a: list[float], b: list[float]) -> float:
    n = len(a)
    if n < 3:
        raise ValueError(f"a correlation over {n} points is not a correlation")
    ma, mb = sum(a) / n, sum(b) / n
    num = sum((x - ma) * (y - mb) for x, y in zip(a, b, strict=True))
    da = sum((x - ma) ** 2 for x in a) ** 0.5
    db = sum((y - mb) ** 2 for y in b) ** 0.5
    if not da or not db:
        raise ValueError("a constant column has no correlation")
    return num / (da * db)


def spearman(a: list[float], b: list[float]) -> float:
    return pearson(ranks(a), ranks(b))


def _round(x: float, nd: int = 3) -> float:
    return round(x, nd)


# ── the matched comparison, from data files alone ───────────────────────

def matched_comparison(repo_root: Path) -> dict[str, Any]:
    """MG and MF differ only in the 90 target completions.

    Computed from the committed jsonl files.  No model, no prediction and no
    retention number is involved, which is what makes the anchor argument a
    structural fact rather than an observation about outcomes.
    """
    training = sg.dataset_dir(repo_root) / "training"
    probe = json.loads((repo_root / PROBE_REPORT).read_text())
    probe_assoc = set(probe["halves"]["probe"]["associations"])

    def load(name: str) -> dict[str, dict[str, Any]]:
        path = training / f"{name}.jsonl"
        if not path.exists():
            raise SystemExit(
                f"REFUSING: {path} is missing, so the claim that MG and MF "
                f"share their retained training data cannot be checked rather "
                f"than merely unverified.")
        rows = [json.loads(line) for line in path.read_text().splitlines()
                if line.strip()]
        return {r["association_id"]: r for r in rows}

    mf, mg = load("MF"), load("MG")
    if set(mf) != set(mg):
        raise SystemExit(
            f"REFUSING: MF and MG cover different association sets "
            f"({len(set(mf) ^ set(mg))} symmetric difference), so their "
            f"retention difference is not attributable to the target "
            f"completions alone.")
    retained = {a: r for a, r in mf.items() if r["role"] == "retain"}
    targets = {a: r for a, r in mf.items() if r["role"] == "target"}
    same_retained = sum(1 for a, r in retained.items()
                        if r["completion"] == mg[a]["completion"])
    differing_targets = sum(1 for a, r in targets.items()
                            if r["completion"] != mg[a]["completion"])
    preserve = sg.groups_dir(repo_root) / "retain.jsonl"
    preserve_ids = {json.loads(line)["association_id"]
                    for line in preserve.read_text().splitlines()
                    if line.strip()}
    return {
        "num_associations": len(mf),
        "num_retained_associations": len(retained),
        "num_target_associations": len(targets),
        "retained_completions_identical": same_retained,
        "retained_completions_differing": len(retained) - same_retained,
        "all_retained_completions_identical": same_retained == len(retained),
        "target_completions_differing": differing_targets,
        "all_target_completions_differ": differing_targets == len(targets),
        "mf_covers_probe_half": len(
            {a for a in retained if a in probe_assoc}),
        "mg_covers_probe_half": len(
            {a for a in retained if a in probe_assoc}),
        "probe_half_associations": len(probe_assoc),
        "iter12_preserve_group_covers_probe_half": len(
            preserve_ids & probe_assoc),
        "what_this_isolates": (
            "MG and MF are trained by one recipe on datasets whose 387 "
            "retained completions are identical and whose 90 target "
            "completions all differ, and both cover the probe half. Their "
            "retention difference is therefore the cost of the granularity "
            "transformation itself, with recipe, retained knowledge and "
            "information held equal -- not a cost of unlearning from MF."),
    }


# ── decision 1: the anchor ──────────────────────────────────────────────

def anchor_decision(repo_root: Path) -> dict[str, Any]:
    strat = json.loads((repo_root / sg.ROUTE_STRATIFIED_REPORT).read_text())
    b0 = strat["anchor"]["values"]
    mg = strat["stage1"]["reference_states_not_floored"]["MG"]["values"]
    denominators = strat["anchor"]["denominators"]
    cands = strat["stage1"]["candidates"]
    eps = rs.FLOOR_EPSILON

    def clears(values: dict[str, float], baseline: dict[str, float]) -> int:
        return sum(1 for k in FLOOR_KEYS
                   if values[k] >= baseline[k] - eps)

    mg_on_b0 = clears(mg, b0)
    b0_on_mg = clears(b0, mg)
    worst = min(FLOOR_KEYS, key=lambda k: mg[k] - b0[k])
    per_candidate = {
        cid: {"eligible_on_b0": info["eligible_on_the_eight"],
              "eligible_on_mg": clears(info["values"], mg) == len(FLOOR_KEYS)}
        for cid, info in cands.items() if cid != "B0"
    }
    return {
        "decision": (
            "Stage 2's decision-bearing floor is anchored at MG on all eight "
            "numbers, with the frozen 1e-9 epsilon and no margin. The "
            "B0-anchored verdict is computed and reported for every Stage-2 "
            "state as well, unchanged."),
        "decided_before": "any Stage-2 adapter was trained",
        "pre_registered_by": (
            "README.md, which states that Stage 2 faces the same anchor and "
            "will return the no-op again unless the anchor itself is "
            "reconsidered, and that this is a design decision to make before "
            "Stage 2 is frozen, not after it is scored."),
        "matched_comparison": matched_comparison(repo_root),
        "the_oracle_fails_the_frozen_floor": {
            "mg_clears_of_eight": mg_on_b0,
            "of": len(FLOOR_KEYS),
            "worst_number": worst,
            "worst_shortfall": _round(mg[worst] - b0[worst], 4),
            "mg_difference_vs_b0": {k: _round(mg[k] - b0[k], 4)
                                    for k in FLOOR_KEYS},
            "why_this_decides_it": (
                "D_G, the selection criterion this project has used since "
                "Iteration 9, measures distance TO MG. A floor whose reference "
                "point MG itself fails disqualifies the state the criterion is "
                "aiming at, so it can only rank candidates by how little they "
                "did."),
        },
        "re_anchoring_changes_no_filed_verdict": {
            "stage1_candidates": len(per_candidate),
            "eligible_on_b0": sum(1 for v in per_candidate.values()
                                  if v["eligible_on_b0"]),
            "eligible_on_mg": sum(1 for v in per_candidate.values()
                                  if v["eligible_on_mg"]),
            "per_candidate": per_candidate,
            "why_this_matters": (
                "A rule change that turned filed failures into passes would be "
                "a goalpost move. This one turns none: Stage 1 is 0 of 8 "
                "eligible under either anchor."),
        },
        "the_new_floor_does_not_licence_inaction": {
            "b0_clears_of_eight": b0_on_mg,
            "of": len(FLOOR_KEYS),
            "numbers_where_mg_is_above_b0": [
                k for k in FLOOR_KEYS if mg[k] > b0[k] + eps],
            "why_this_matters": (
                "Under the MG anchor the no-op is not eligible either, so the "
                "rule cannot be satisfied by declining to unlearn -- the "
                "failure mode the B0 anchor had."),
        },
        "anchor_values": {k: mg[k] for k in FLOOR_KEYS},
        "anchor_state": "MG",
        "reported_baseline_values": {k: b0[k] for k in FLOOR_KEYS},
        "reported_baseline_state": "B0",
        "denominators": {k: denominators[k] for k in FLOOR_KEYS},
        "epsilon": eps,
        "margin": 0.0,
        "anchor_predictions_are_already_bound": (
            "MG's parquet is one of the nineteen the Stage-1c freeze bound by "
            "sha256 under the single verified generation contract, so the "
            "anchor's eight numbers come from bytes already hashed rather than "
            "from a regeneration."),
        "a_defect_left_in_place": (
            "route_stratified_retention.floor_check_stratified labels its "
            "baseline field 'b0' and its comparison string 'candidate >= b0 - "
            "eps' regardless of which baseline is passed. That module is "
            "hash-bound by the Stage-1c freeze, so the names cannot be "
            "corrected; under Stage 2's primary gate the field named 'b0' "
            "holds MG's values, and the Stage-2 report says so explicitly."),
    }


# ── decision 2: the mechanism ───────────────────────────────────────────

def floor_resolution(repo_root: Path) -> dict[str, Any]:
    """What one flipped query is worth, on the scope Stage 1c reads.

    Quoted from the frozen Stage-1c basis rather than recomputed, so Stage 2
    states the same limit that study filed and a reviewer can check the two
    against each other.
    """
    probe = json.loads((repo_root / ROUTE_PROBE_REPORT).read_text())
    ceiling = probe["power_ceiling"]
    if CEILING_SCOPE not in ceiling:
        raise SystemExit(
            f"REFUSING: {ROUTE_PROBE_REPORT} has no scope {CEILING_SCOPE!r}; "
            f"it names {sorted(k for k in ceiling if isinstance(ceiling[k], dict))}. "
            f"Quoting a different scope's resolution would state a limit the "
            f"frozen basis does not contain.")
    scope = ceiling[CEILING_SCOPE]
    return {
        "scope": CEILING_SCOPE,
        "row_micro_resolution_retain_same":
            scope["row_micro_resolution_same"],
        "row_micro_resolution_retain_other":
            scope["row_micro_resolution_other"],
        "retain_same_total": scope["retain_same_total"],
        "retain_other_total": scope["retain_other_total"],
        "binding_number": "retain_other",
        "absolute_ceiling": ceiling["all splits, both routes (the absolute "
                                    "ceiling)"],
        "what_the_ceiling_means": ceiling["what_the_ceiling_means"],
    }


def mechanism_decision(repo_root: Path) -> dict[str, Any]:
    cal = json.loads((repo_root / sg.CALIBRATION_REPORT).read_text())
    strat = json.loads((repo_root / sg.ROUTE_STRATIFIED_REPORT).read_text())
    sel = json.loads((repo_root / STAGE1_SELECTION_REPORT).read_text())
    adapters = cal["measurement"]["adapters"]
    b0 = strat["anchor"]["values"]
    mg = strat["stage1"]["reference_states_not_floored"]["MG"]["values"]
    cands = strat["stage1"]["candidates"]
    eps = rs.FLOOR_EPSILON

    cids = [c for c in sorted(cands) if c != "B0" and c in adapters]
    if len(cids) < 3:
        raise SystemExit(
            f"REFUSING: only {len(cids)} Stage-1 candidates have both a filed "
            f"eight-number result and a calibration row, which is too few to "
            f"say anything about which quantity governs retention.")
    kl = [adapters[c]["mean_kl_vs_mf"] for c in cids]
    ft = [adapters[c]["nll_by_group"]["fine_target"] for c in cids]
    tl = [adapters[c]["nll_by_group"]["target_level"] for c in cids]
    rt_nll = [adapters[c]["nll_by_group"]["retain"] for c in cids]
    d_g = [sel["candidates"][c]["distance_to_mg"] for c in cids]
    diffs = {k: [cands[c]["values"][k] - b0[k] for c in cids]
             for k in FLOOR_KEYS}
    resolution = floor_resolution(repo_root)

    def rho_table(predictor: list[float]) -> dict[str, float]:
        out = {k: _round(spearman(predictor, diffs[k])) for k in FLOOR_KEYS}
        out["D_G"] = _round(spearman(predictor, d_g))
        return out

    #: Pairs whose drift from MF is close but whose retention is not.  This is
    #: the counterexample to "control the drift and retention follows", stated
    #: as measurements rather than as an impression.
    pairs = []
    order = sorted(range(len(cids)), key=lambda i: kl[i])
    for a, b in pairwise(order):
        gap = abs(kl[a] - kl[b])
        if gap > 0.03:
            continue
        ret_gap = {k: _round(abs(diffs[k][a] - diffs[k][b]), 4)
                   for k in FLOOR_KEYS}
        pairs.append({
            "states": [cids[a], cids[b]],
            "drift_apart": _round(gap, 6),
            "retention_apart": ret_gap,
            "worst_retention_gap": max(ret_gap.values()),
            "floor_resolution_retain_other":
                resolution["row_micro_resolution_retain_other"],
        })
    mg_vs = None
    if "MG" in adapters:
        #: The sharpest form of the same point: the oracle and a candidate at
        #: the SAME drift from MF, with retention far apart.
        nearest = min(cids, key=lambda c: abs(adapters[c]["mean_kl_vs_mf"]
                                              - adapters["MG"]["mean_kl_vs_mf"]))
        mg_vs = {
            "mg_drift": adapters["MG"]["mean_kl_vs_mf"],
            "candidate": nearest,
            "candidate_drift": adapters[nearest]["mean_kl_vs_mf"],
            "drift_apart": _round(abs(adapters["MG"]["mean_kl_vs_mf"]
                                      - adapters[nearest]["mean_kl_vs_mf"]), 6),
            "retention_apart": {
                k: _round(abs(mg[k] - cands[nearest]["values"][k]), 4)
                for k in FLOOR_KEYS},
        }

    return {
        "preregistered_mechanism": "MF-preservation regularizer",
        "preregistered_by": (
            "scripts/freeze_iter12_selection_protocol.py, whose stage_2 block "
            "names the mechanism and whose stage2_gate reports stage2_opens "
            "true; retention_selection.stage2_gate's own docstring asks "
            "'Does the MF-preservation regularizer stage open?'"),
        "calibration_report": sg.CALIBRATION_REPORT,
        "calibration_reads": cal["measurement"]["inputs"],
        "calibration_reads_no_retention_number":
            cal["measurement"]["reads_no_retention_number"],
        "nll_agreement_max": cal["measurement"]["nll_agreement_max"],
        "controls": {
            "MF_against_itself": adapters["MF"]["mean_kl_vs_mf"],
            "B0_whose_adapter_is_a_copy_of_MF": adapters["B0"]["mean_kl_vs_mf"],
            "why_these_matter": (
                "Both are exactly zero, so the extraction, the position shift "
                "and the KL are measuring what they claim before any "
                "difference between adapters is read as one."),
        },
        "the_anchor_quantity_does_not_determine_retention": {
            "oracle_versus_nearest_candidate": mg_vs,
            "close_drift_pairs": pairs,
            "floor_resolution": resolution,
            "reading": (
                "States at the same distance from MF on the 183 preserved "
                "prompts differ in probe-half retention by far more than the "
                "floor can resolve. Controlling that distance therefore does "
                "not control what the floor measures."),
        },
        "the_anchor_optimum_is_the_wrong_place": {
            "kl_zero_is": "MF, which performs no unlearning at all",
            "mg_drift_from_mf": adapters["MG"]["mean_kl_vs_mf"]
            if "MG" in adapters else None,
            "reading": (
                "A KL anchor's minimum is at MF, not at MG, so a strong beta "
                "drives a candidate toward the no-op rather than toward the "
                "target state."),
        },
        "spearman_rho_by_predictor": {
            "kl_from_mf_on_the_183_preserved_prompts": rho_table(kl),
            "fine_target_nll": rho_table(ft),
            "target_level_nll": rho_table(tl),
            "retain_nll_fit_half": rho_table(rt_nll),
            "num_candidates": len(cids),
            "reading": (
                "fine_target NLL ranks the image-route numbers perfectly and "
                "the text-route ones strongly; the drift the anchor controls "
                "reaches neither. And fine_target NLL has essentially no rank "
                "relationship with D_G, so the overshoot it measures buys "
                "nothing at the criterion."),
        },
        "the_operating_point": {
            "mf_fine_target_nll": adapters["MF"]["nll_by_group"]["fine_target"],
            "mg_fine_target_nll": adapters["MG"]["nll_by_group"]["fine_target"],
            "candidate_fine_target_nll": {
                c: adapters[c]["nll_by_group"]["fine_target"] for c in cids},
            "candidates_below_the_oracle": [
                c for c in cids
                if adapters[c]["nll_by_group"]["fine_target"]
                <= adapters["MG"]["nll_by_group"]["fine_target"] + eps],
            "reading": (
                "Every Stage-1 candidate drove the fine fact's NLL past the "
                "level the oracle sits at. The overshoot is displacement from "
                "MF that the transformation did not require."),
        },
        "mechanisms_trained": {
            "B5_bounded_ascent": (
                "the incumbent recipe with -lambda * NLL(fine_target) replaced "
                "by -lambda * min(NLL(fine_target), cap), whose gradient is "
                "exactly zero above the cap. Targets the coordinate the "
                "measurement says governs, and decouples how far to suppress "
                "from how long to train, which Stage 1 coupled through the "
                "epoch count."),
            "B6_mf_preservation_anchor": (
                "the preregistered regularizer: KL(p_MF || p_theta) over the "
                "supervised positions of the fit-half retain group, at the "
                "beta the calibration derived. Trained because an "
                "observational argument over eight points is not an "
                "experiment, and a mechanism this study was preregistered "
                "with deserves to be run rather than argued away."),
            "B6R_hybrid": (
                "replay and the anchor on the same forward pass, through "
                "anchor_weight rather than a second group entry, so the epoch "
                "keeps its 363 micro-batches and the schedule under test does "
                "not move."),
        },
        "beta_star": cal["grid_rule"]["beta_star"],
        "beta_star_derivation": cal["grid_rule"],
    }


def cap_grid(repo_root: Path) -> dict[str, Any]:
    cal = json.loads((repo_root / sg.CALIBRATION_REPORT).read_text())
    mg_ft = cal["measurement"]["adapters"]["MG"]["nll_by_group"]["fine_target"]
    mf_ft = cal["measurement"]["adapters"]["MF"]["nll_by_group"]["fine_target"]
    cands = {k: v["nll_by_group"]["fine_target"]
             for k, v in cal["measurement"]["adapters"].items()
             if k.startswith("B4_")}
    return {
        "values": list(sg.STAGE2_CAPS),
        "units": "nats of per-token NLL on the fine completion",
        "applies_to": "fine_target only; the rewrite and the replay are uncapped",
        "derived_from_the_oracle": False,
        "oracle_position_reported_not_used": mg_ft,
        "mf_position": mf_ft,
        "uncapped_point": sg.incumbent_row(),
        "uncapped_point_fine_target_nll": cands.get(sg.incumbent_row()),
        "candidate_range_observed": [min(cands.values()), max(cands.values())],
        "why_this_range": (
            "The caps bracket the oracle's measured level from both sides and "
            "reach into the range the Stage-1 candidates actually occupied. "
            "The range was chosen AFTER measuring MG's level, which is "
            "disclosed here rather than left implicit; what is not true is "
            "that any oracle value enters the training procedure, because the "
            "cap is a swept constant and MG's 1.3643 appears in no candidate's "
            "spec. The sweep's uncapped end is the incumbent Stage-1 row, "
            "which already exists with predictions on disk and costs nothing."),
        "what_frac_at_cap_means": (
            "Each B5 summary records frac_at_cap_fine_target per epoch: the "
            "share of micro-batches where the cap was already binding. A cap "
            "that never binds reproduces the incumbent exactly, which was "
            "verified on a stub model before this basis was written, and a cap "
            "that binds from the first step was set below where the model "
            "started. Either would make the row uninformative, and this field "
            "says which happened."),
    }


def decision_rule() -> dict[str, Any]:
    return {
        "primary_gate": {
            "anchor": "MG",
            "numbers": list(FLOOR_KEYS),
            "rule": "candidate >= MG - epsilon on all eight, no margin",
            "epsilon": rs.FLOOR_EPSILON,
            "margin": 0.0,
            "computed_by": (
                "route_stratified_retention.floor_check_stratified, the same "
                "frozen function Stage 1c used, with MG's vector passed as the "
                "baseline. No new metric code: that function is generic in its "
                "baseline."),
            "a_missing_number_disqualifies": (
                "floor_check_stratified already treats an unmeasurable "
                "retention value as a failure rather than as a pass, and Stage "
                "2 inherits that rather than restating it."),
        },
        "also_reported_not_decision_bearing": {
            "b0_anchored_floor": (
                "the frozen Stage-1c gate, computed for every Stage-2 state "
                "and reported beside the primary one so the switch is visible "
                "in the result rather than only in the protocol"),
            "per_route_pass": "text and image separately",
            "pooled_all_routes": "reported, floors nothing",
            "frac_at_cap": "whether each cap actually bound",
            "fine_target_nll": "where each candidate ended on the governing axis",
            "kl_from_mf": "where each candidate ended on the anchor's axis",
        },
        "among_eligible": (
            "minimise D_G with the frozen tie-break, exactly as Stage 1 did. "
            "The floor decides eligibility; D_G only orders the eligible."),
        "stage_3_gate": (
            "Stage 3 opens if no Stage-2 candidate satisfies the primary gate, "
            "or if the best eligible one does not strictly improve on the "
            "incumbent reference row's D_G. Same shape as the frozen Stage-2 "
            "gate, same 'a tie is no improvement' rule, same zero margin."),
        "reference_row_for_the_gate": sg.incumbent_row(),
    }


def build_basis(repo_root: Path) -> dict[str, Any]:
    cal = json.loads((repo_root / sg.CALIBRATION_REPORT).read_text())
    grid = sg.stage2_grid(cal["grid_rule"]["beta_star"])
    kept, filtered = sg.validate_stage2_grid(grid)
    if kept:
        raise SystemExit(f"REFUSING: the Stage-2 grid is invalid: {kept}")
    for spec in grid:
        errs = validate_stage2_groups(list(spec.groups))
        if errs:
            raise SystemExit(
                f"REFUSING: {spec.candidate_id}'s objective is invalid: {errs}")

    return {
        "iteration": "12",
        "stage": 2,
        "purpose": (
            "Record the two decisions Stage 2 rests on -- what the retention "
            "floor is anchored at, and which mechanisms are trained -- with the "
            "measurements they come from, before any Stage-2 adapter exists."),
        "generated_from": list(GENERATED_FROM),
        "reads_no_prediction_parquet": True,
        "trains_nothing": True,
        "generates_nothing": True,
        "retention_numbers_are_quoted_not_recomputed": (
            "Every retention value here was filed by Stage 1, 1b or 1c and is "
            "read from those reports. Nothing is recomputed from a model "
            "output, so --check-only can re-derive this document byte for byte "
            "and a Stage-2 outcome could not have reached it."),
        "anchor_decision": anchor_decision(repo_root),
        "mechanism_decision": mechanism_decision(repo_root),
        "cap_grid": cap_grid(repo_root),
        "grid": {
            "num_rows": len(grid),
            "num_trained": len(sg.trained_rows(grid)),
            "rows": [c.describe() for c in grid],
            "held_fixed_across_every_trained_row": {
                "fine_suppression_weight": sg.ITER12_LAM,
                "target_level_weight": 1.0,
                "replay_weight_where_present": sg.ITER12_REFERENCE_WEIGHT,
                "init_adapter": "the pilot-100 canonical MF adapter",
                "recipe": ("ReferenceRecipe, overridden only in "
                           "learning_rate and num_epochs"),
                "knowledge_groups": "the fit-half groups of unlearning_iter12/",
            },
            "the_frozen_validator_was_run": {
                "residual_errors": kept,
                "filtered_messages": filtered,
                "filtered_because": (
                    "candidate_grid.py is hash-bound, so its METHODS tuple and "
                    "its mode vocabulary were frozen before Stage 2 existed "
                    "and cannot be extended. Only 'unknown method' and 'bad "
                    "mode' are filtered, by exact substring, and the residual "
                    "must be empty -- so every other structural check the "
                    "frozen validator makes still applies to this grid."),
                "filtered_gap_substrings": list(sg.FROZEN_VALIDATOR_GAP),
            },
        },
        "measurement_limits": {
            "probe_denominators": "408 retain-same and 76 retain-other, per route",
            "resolution": floor_resolution(repo_root),
            "retain_other_stays_weakly_powered": (
                "23 probe donors and 38 queries per split, 228 at the absolute "
                "ceiling, so one flipped outcome is 0.0132 and no number of "
                "seeds changes that."),
            "the_ceiling_is_the_dataset": (
                "The association pool is exhausted -- 90 target plus 387 "
                "retained is all 477 -- and query_generation.py is one of the "
                "eighteen paths the confirmation freeze sealed."),
            "one_seed_is_not_enough": (
                "Stage 1b measured a between-seed sd of 0.0120 to 0.0515 on "
                "these numbers, which exceeds the retain-other resolution. "
                "Stage 2 trains one seed per row, so a row that clears the "
                "primary gate by less than that spread is a candidate for "
                "replication rather than a result, and the report must say so "
                "instead of ranking on it."),
        },
        "never_read": list(rs.FORBIDDEN_EVIDENCE),
        "gpu_hours_spent_on_this_basis": 0.0,
        "gpu_hours_spent_on_the_calibration": (
            "about 0.5: eleven adapters, one forward pass each over the three "
            "knowledge groups, on a GPU that was otherwise idle"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Write (or re-derive) the Iteration 12 Stage-2 basis")
    parser.add_argument("--check-only", action="store_true",
                        help="Re-derive and compare; write nothing")
    parser.add_argument("--repo-root", default=None)
    args = parser.parse_args()

    repo_root = Path(args.repo_root) if args.repo_root \
        else (_find_repo_root(Path.cwd()) or Path.cwd())
    out = repo_root / sg.BASIS_REPORT
    basis = build_basis(repo_root)

    if args.check_only:
        if not out.exists():
            raise SystemExit(f"REFUSING: {out} does not exist, so there is "
                             f"nothing to check against")
        filed = json.loads(out.read_text())
        if filed != basis:
            diffs = [k for k in set(filed) | set(basis)
                     if filed.get(k) != basis.get(k)]
            raise SystemExit(
                f"REFUSING: the committed basis does not re-derive from the "
                f"artifacts it names. Differing top-level keys: {sorted(diffs)}")
        print(f"OK: {out.name} re-derives exactly from "
              f"{len(basis['generated_from'])} committed reports and the "
              f"knowledge-group files, with no prediction read.")
        return

    if out.exists():
        existing = json.loads(out.read_text())
        if existing != basis:
            raise SystemExit(
                f"REFUSING to overwrite {out}: it already exists and does not "
                f"re-derive to the same document. Delete it deliberately if "
                f"the basis really has changed -- a silent rewrite is how a "
                f"preregistration stops being one.")
        log.info("%s already holds exactly this document", out.name)
        return
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(basis, f, indent=2, ensure_ascii=False)
    log.info("Stage-2 basis -> %s", out)


if __name__ == "__main__":
    main()
