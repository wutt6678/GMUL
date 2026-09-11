"""Iteration 12 selection protocol — frozen before training, tested on CPU.

These tests have one job that the rest of the suite does not: they check that
a protocol written BEFORE any candidate existed actually binds the run that
comes after it.  Three things are frozen and each gets its own regression,
because each can silently change which candidate wins:

* the DIRECTION of D_G — it is a distance and is minimised, and it does not
  reward low leakage;
* the numerical TOLERANCE — distances at the 6 decimals
  ``distance_to_reference`` already rounds to, and a floor epsilon of 1e-9
  that sits six orders of magnitude below the smallest difference the probe
  measurement can express, so it absorbs representation error and cannot
  decide a real comparison.  A +0.02 retention margin was rejected and one
  test here fails if it is reintroduced;
* the TIE-BREAK — ``(distance_to_mg, candidate_id)``, which carries no
  outcome information and does not depend on grid order or on which adapters
  happen to be on disk.

Nothing here needs a GPU, an adapter or a photograph, and nothing reads a
prediction parquet: the evidence is the committed association pool, the
committed group files, the committed pilot-100 selection report and the two
reports this protocol files.  So it all runs in CI and in a fresh clone.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import types
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import build_iter12_retention_probe as bip
import freeze_iter12_selection_protocol as fip
import select_iter12_retention_checkpoints as sic

from granunlearn.evaluation import retention_selection as rs
from granunlearn.evaluation import selection as sel
from granunlearn.training.candidate_grid import (
    ITER12_PILOT_EQUIVALENT_WEIGHT,
    ITER12_REFERENCE_ROW,
    dataset_dir_for_tag,
    grid_for_tag,
    groups_subdir_for_tag,
    pilot100_grid,
    smoke_grid,
    validate_grid,
)

PROBE_REPORT = REPO_ROOT / bip.OUT_REPORT
FREEZE_REPORT = REPO_ROOT / fip.OUT_REPORT
CONFIRM_FREEZE = REPO_ROOT / \
    "data/reports/mllmu_pilot100_confirmation_freeze.json"
PILOT_GROUPS = REPO_ROOT / bip.PILOT_GROUPS
ITER12_GROUPS = REPO_ROOT / bip.ITER12_GROUPS


@pytest.fixture(scope="module")
def probe() -> dict:
    assert PROBE_REPORT.exists(), f"{PROBE_REPORT} is absent"
    return json.loads(PROBE_REPORT.read_text())


@pytest.fixture(scope="module")
def freeze() -> dict:
    assert FREEZE_REPORT.exists(), f"{FREEZE_REPORT} is absent"
    return json.loads(FREEZE_REPORT.read_text())


@pytest.fixture(scope="module")
def pool():
    """The committed association pool and queries (no photographs needed)."""
    from granunlearn.evaluation.reference_eval import (
        load_associations_parquet,
        load_queries_parquet,
    )
    data = REPO_ROOT / bip.DATA_DIR
    return (load_queries_parquet(data / "queries.parquet"),
            load_associations_parquet(data / "associations.parquet"))


def _pred(flag: bool):
    """A stand-in prediction.  The rate helpers read one attribute."""
    return types.SimpleNamespace(is_correct_branch=flag)


def _rows(flags_by_entity: dict[str, list[bool]]):
    return [(ent, _pred(f)) for ent, fs in flags_by_entity.items()
            for f in fs]


def _vector(filr, tga, ancestor=0.25, retain_same=0.55, retain_other=0.55,
            over=0.03, wrong=0.22) -> dict:
    """A hierarchy-metrics block shaped like the sealed one returns."""
    return {
        "filr": filr, "tga": tga,
        "ancestor_retention": {"post_unlearning_accuracy": ancestor},
        "retain_same_entity": {"baseline_accuracy": retain_same},
        "retain_other_entity": {"baseline_accuracy": retain_other},
        "failure_rates": {"over_forgetting": over, "wrong_branch": wrong},
    }


def _probe_block(ret_s_micro, ret_s_macro, ret_o_micro, ret_o_macro) -> dict:
    return {
        "scope": ["train", "val"], "probe_entities": 35,
        "retain_same_entity": {
            "num_queries": 408, "num_entities": 35,
            "row_micro": ret_s_micro, "entity_macro": ret_s_macro},
        "retain_other_entity": {
            "num_queries": 76, "num_entities": 23,
            "row_micro": ret_o_micro, "entity_macro": ret_o_macro},
    }


# ── 1. the partition ────────────────────────────────────────────────────────

class TestThePartitionIsDisjointAndCoversTheRetainedKnowledge:
    def test_the_halves_are_disjoint_and_cover_the_retain_set(self, probe):
        partition = json.loads(
            (REPO_ROOT / bip.PARTITION_REPORT).read_text())
        halves = probe["halves"]
        p_ent, f_ent = halves["probe"], halves["fit"]
        assert set(p_ent["entities"]) & set(f_ent["entities"]) == set()
        assert (set(p_ent["associations"]) | set(f_ent["associations"])
                == set(partition["retain_association_ids"]))
        assert set(p_ent["associations"]) & set(f_ent["associations"]) == set()
        assert p_ent["num_entities"] == f_ent["num_entities"] == 35
        assert p_ent["num_associations"] == 204
        assert f_ent["num_associations"] == 183

    def test_no_probe_association_reaches_the_committed_replay_group(self,
                                                                     probe):
        """The load-bearing claim: the probe half was never rehearsed."""
        replay = {json.loads(line)["association_id"]
                  for line in (ITER12_GROUPS / "retain.jsonl").read_text()
                  .splitlines() if line.strip()}
        probe_ids = set(probe["halves"]["probe"]["associations"])
        assert replay & probe_ids == set(), (
            "a probe association is in the replay group, so retention "
            "measured on the probe half would be in-sample after all")
        assert replay == set(probe["halves"]["fit"]["associations"])
        assert len(replay) == 183

    def test_the_target_side_groups_are_byte_identical_to_the_pilots(self):
        for name in bip.COPIED_GROUPS:
            assert hashlib.sha256(
                (ITER12_GROUPS / name).read_bytes()).hexdigest() == \
                hashlib.sha256((PILOT_GROUPS / name).read_bytes()
                               ).hexdigest(), (
                f"{name} moved: the target-side objective must be the same "
                f"bytes the incumbent was trained on, or the comparison is "
                f"not about retention")

    def test_the_partition_is_reproducible_from_the_hash_rule_alone(self,
                                                                    probe,
                                                                    pool):
        """Re-derived here, not trusted from the report."""
        _, associations = pool
        partition = json.loads(
            (REPO_ROOT / bip.PARTITION_REPORT).read_text())
        retain = {a.association_id for a in associations} & \
            set(partition["retain_association_ids"])
        entities = sorted({a.entity_id for a in associations
                           if a.association_id in retain})
        ordered = sorted(entities, key=bip.hash_rule)
        cut = len(ordered) // 2
        assert ordered[:cut] == probe["halves"]["probe"]["entities"]
        assert ordered[cut:] == probe["halves"]["fit"]["entities"]

    def test_the_committed_report_matches_a_fresh_derivation(self):
        assert json.loads(PROBE_REPORT.read_text()) == \
            bip.build_report(REPO_ROOT, write=False), (
            "the committed partition is not what the association pool "
            "produces, so it was edited by hand or the pool moved")

    def test_no_probe_entity_is_a_target(self, probe, pool):
        _, associations = pool
        partition = json.loads(
            (REPO_ROOT / bip.PARTITION_REPORT).read_text())
        targets = set(partition["target_association_ids"])
        probe_ids = set(probe["halves"]["probe"]["associations"])
        assert probe_ids & targets == set()
        assert len({a.entity_id for a in associations
                    if a.association_id in probe_ids}) == 35


# ── 2. the tolerance ────────────────────────────────────────────────────────

class TestTheToleranceCannotDecideARealComparison:
    def test_the_epsilon_is_below_every_resolvable_gap(self, probe):
        gaps = probe["probe_measurement_basis"]["per_family"]
        every = [block["min_resolvable_gap"]
                 for family in gaps.values()
                 for block in (family["row_micro"], family["entity_macro"])]
        assert len(every) == 4, "four numbers are frozen: 2 families x 2 " \
                                "estimands"
        assert rs.FLOOR_EPSILON < min(every)
        assert probe["numerical_tolerance"][
            "epsilon_is_below_every_resolvable_gap"] is True
        #: Six orders of magnitude is the difference between absorbing a
        #: rounding artefact and inventing a threshold.
        assert min(every) / rs.FLOOR_EPSILON > 10 ** 5

    def test_the_gaps_recompute_from_the_committed_partition(self, probe,
                                                             pool):
        queries, associations = pool
        fresh = bip.resolvable_gaps(
            set(probe["halves"]["probe"]["entities"]), queries, associations)
        assert fresh == probe["probe_measurement_basis"]["per_family"]
        assert fresh["retain_same_entity"]["row_micro"][
            "min_resolvable_gap"] == round(1 / 408, 10)
        assert fresh["retain_same_entity"]["entity_macro"][
            "min_resolvable_gap"] == round(1 / (35 * 14), 10)

    def test_an_exact_tie_passes_the_floor(self):
        same = _probe_block(0.5294, 0.5256, 0.4737, 0.5072)
        check = rs.floor_check(same, same)
        assert check["eligible"] is True, (
            "a candidate exactly at B0's level must be eligible; this is the "
            "only thing the epsilon is for")
        assert all(c["difference"] == 0.0 for c in check["checks"].values())

    def test_a_shortfall_of_one_resolvable_gap_fails(self, probe):
        gap = probe["numerical_tolerance"][
            "smallest_resolvable_gap_over_the_four_frozen_numbers"]
        b0 = _probe_block(0.5294, 0.5256, 0.4737, 0.5072)
        worse = _probe_block(0.5294 - gap, 0.5256, 0.4737, 0.5072)
        check = rs.floor_check(worse, b0)
        assert check["eligible"] is False
        assert check["checks"][
            "retain_same_entity.row_micro"]["passes"] is False

    def test_the_rate_rounding_cannot_merge_two_values(self, probe):
        gap = probe["numerical_tolerance"][
            "smallest_resolvable_gap_over_the_four_frozen_numbers"]
        assert gap > 10 ** -rs.RATE_DECIMALS, (
            "rates are reported at 4 decimals; if a resolvable gap were "
            "smaller than that quantum, rounding could make two different "
            "retention levels look tied")

    def test_no_margin_is_required_anywhere(self, freeze):
        floor = freeze["floor"]
        assert floor["epsilon"] == 1e-9
        assert floor["no_retention_margin_is_required"] is True
        assert floor["epsilon_is_a_tolerance_not_a_margin"] is True
        assert "+0.02" in floor["rejected_alternative"]
        assert freeze["stage_2"]["margin_required"] == 0.0
        assert freeze["stage_2"]["no_margin_is_used"] is True

    def test_reintroducing_a_margin_breaks_the_freeze(self, monkeypatch):
        """The rejected margin, caught by the check the selector runs."""
        assert fip.verify_freeze(REPO_ROOT) == []
        monkeypatch.setattr(rs, "FLOOR_EPSILON", 0.02)
        reasons = fip.verify_freeze(REPO_ROOT)
        assert reasons, "a +0.02 margin did not change the freeze, so nothing " \
                        "would stop it being reintroduced"
        assert any("floor.epsilon" in r for r in reasons)


# ── 3. the floor needs both estimands ───────────────────────────────────────

class TestTheFloorRequiresBothEstimands:
    def test_passing_macro_and_failing_micro_is_ineligible(self):
        b0 = _probe_block(0.5294, 0.5256, 0.4737, 0.5072)
        #: The incumbent's actual pilot-100 shape on retain_other: above B0
        #: entity-macro, below it row-micro.
        cand = _probe_block(0.5931, 0.5862, 0.4600, 0.5942)
        check = rs.floor_check(cand, b0)
        assert check["eligible"] is False
        assert check["checks"][
            "retain_other_entity.row_micro"]["passes"] is False
        assert check["checks"][
            "retain_other_entity.entity_macro"]["passes"] is True

    def test_passing_micro_and_failing_macro_is_ineligible(self):
        b0 = _probe_block(0.5294, 0.5256, 0.4737, 0.5072)
        cand = _probe_block(0.5400, 0.5100, 0.4800, 0.5100)
        check = rs.floor_check(cand, b0)
        assert check["eligible"] is False
        assert check["checks"][
            "retain_same_entity.entity_macro"]["passes"] is False
        assert check["checks"][
            "retain_same_entity.row_micro"]["passes"] is True

    def test_an_unmeasurable_value_disqualifies_rather_than_passing(self):
        b0 = _probe_block(0.5294, 0.5256, 0.4737, 0.5072)
        cand = _probe_block(0.5294, None, 0.4737, 0.5072)
        check = rs.floor_check(cand, b0)
        assert check["eligible"] is False
        assert "not measurable" in check["checks"][
            "retain_same_entity.entity_macro"]["why"]

    def test_the_floor_binds_on_the_committed_pilot_evidence(self):
        """The constraint is not vacuous, and it disqualifies the incumbent.

        Read from the committed pilot-100 selection report rather than from
        the gitignored prediction parquets, so this holds in a clone.
        """
        report = json.loads(
            (REPO_ROOT / "data/reports/"
             "mllmu_pilot100_unlearning_selection.json").read_text())
        rows = report["candidates"]
        b0 = rows["B0"]["vector"]
        incumbent = "B3_lam0.5_lr2e-05_ep5"
        assert report["selected"]["B3"] == incumbent, (
            "the checkpoint the 11C confirmation scored is the one this "
            "protocol succeeds")
        #: The confirmed checkpoint sits BELOW the no-op on retain_other, so
        #: the floor the user asked for disqualifies it.
        assert rows[incumbent]["vector"]["retain_other"] < \
            b0["retain_other"]
        assert b0["retain_same"] == 0.5594
        assert b0["retain_other"] == 0.6056

        trained = {c: r for c, r in rows.items() if c != "B0"}
        passing = sorted(c for c, r in trained.items()
                         if r["vector"]["retain_same"] >= b0["retain_same"]
                         and r["vector"]["retain_other"] >= b0["retain_other"])
        assert 0 < len(passing) < len(trained), (
            "a floor that everything passes, or nothing passes, constrains "
            "nothing")
        #: And this is the contamination the probe partition exists to fix:
        #: measured in-sample, the floor passes exactly the candidates that
        #: carry a replay group and fails every candidate that does not.
        grid = {c.candidate_id: c for c in pilot100_grid()}
        has_replay = {c for c in passing
                      if any(g.name == "retain" for g in grid[c].groups)}
        assert has_replay == set(passing), (
            "on in-sample retention the floor selects FOR replay instead of "
            "testing it, which is why the protocol measures the floor on the "
            "probe half")

    def test_the_sign_disagreement_is_recorded_in_the_protocol(self, freeze):
        """Why both estimands are required is written down, with its numbers."""
        why = freeze["floor"]["why_both_estimands"]
        assert "-0.1056" in why and "+0.0093" in why
        assert "SIGN" in why
        assert freeze["floor"]["estimands_required"] == \
            list(rs.RETENTION_ESTIMANDS) == ["entity_macro", "row_micro"]

    def test_the_probe_restriction_is_cross_checked_on_every_call(self,
                                                                  monkeypatch):
        """Regression for a bug this suite caught while being written.

        Filtering the sealed metric call by SPLIT alone yields the pooled
        train+val rate — the in-sample number the partition exists to avoid —
        and it looks entirely plausible.  ``probe_retention`` compares the
        sealed value against a direct computation over the rows it was handed
        and raises when they differ.
        """
        probe_ents = json.loads(PROBE_REPORT.read_text())["halves"]["probe"][
            "entities"]
        monkeypatch.setattr(rs, "compute_hierarchy_metrics",
                            lambda *a, **k: {
                                "retain_same_entity": {
                                    #: what a wider, unfiltered input would
                                    #: have produced
                                    "baseline_accuracy": 0.9999},
                                "retain_other_entity": {
                                    "baseline_accuracy": None}})
        pred = types.SimpleNamespace(query_id="q", is_correct_branch=True)
        query = types.SimpleNamespace(
            query_id="q", family="retain_same_entity", split="train",
            association_id="a")
        assoc = types.SimpleNamespace(association_id="a",
                                      entity_id=probe_ents[0])
        with pytest.raises(AssertionError, match="probe restriction"):
            rs.probe_retention([pred], [query], [assoc], probe_ents)

    def test_the_two_estimands_really_do_differ(self):
        """Non-vacuity: requiring both must cost something.

        One entity gets nine of ten right and another gets none.  Row-micro
        reads 0.45; entity-macro reads 0.45 too only if the entities are the
        same size, and here they are not, which is the whole point — the
        macro value weights the single-query entity as heavily as the
        ten-query one.
        """
        rows = _rows({"big": [True] * 9 + [False], "small": [False]})
        assert rs.row_micro_rate(rows) == round(9 / 11, 4)
        assert rs.entity_macro_rate(rows) == round((0.9 + 0.0) / 2, 4)
        assert rs.entity_macro_rate(rows) < rs.row_micro_rate(rows)

    def test_a_retention_rate_is_none_when_there_are_no_rows(self):
        assert rs.row_micro_rate([]) is None
        assert rs.entity_macro_rate([]) is None


# ── 4. direction, tolerance and tie-break ───────────────────────────────────

class TestTheDirectionToleranceAndTieBreakAreFrozen:
    def test_the_direction_is_minimise(self, freeze):
        crit = freeze["criterion"]
        assert crit["direction"] == "minimize" == rs.D_G_DIRECTION
        assert "MINIMUM" in crit["direction_spelled_out"]
        assert "does NOT reward low leakage" in crit["direction_is_not"]

    def test_d_g_penalises_undershooting_mg_as_well_as_overshooting(self):
        """The direction note is not prose: it is a property of the formula."""
        mg = sel.summary_vector(_vector(0.1416, 0.4284))
        at_mg = sel.summary_vector(_vector(0.1416, 0.4284))
        below = sel.summary_vector(_vector(0.0, 0.4284))
        above = sel.summary_vector(_vector(0.2832, 0.4284))
        d_at, _ = rs.distance_to_mg(at_mg, mg)
        d_below, _ = rs.distance_to_mg(below, mg)
        d_above, _ = rs.distance_to_mg(above, mg)
        assert d_at == 0.0
        assert d_below == pytest.approx(d_above, abs=1e-6), (
            "leakage suppressed BELOW MG's level must cost exactly what "
            "leakage left above it costs, or D_G is secretly rewarding "
            "complete forgetting")
        assert d_below > d_at

    def test_a_smaller_distance_wins(self):
        mg = sel.summary_vector(_vector(0.1416, 0.4284))
        b0 = _probe_block(0.5294, 0.5256, 0.4737, 0.5072)
        good = _probe_block(0.5294, 0.5256, 0.4737, 0.5072)
        cands = {
            "B4_w2.0_far": {"method": "B4", "probe_retention": good,
                            "trainval_metrics": _vector(0.40, 0.20)},
            "B4_w0.5_near": {"method": "B4", "probe_retention": good,
                             "trainval_metrics": _vector(0.15, 0.42)},
        }
        report = rs.select_successor(cands, mg, b0)
        assert report["selected"] == "B4_w0.5_near"
        assert report["ranking_of_eligible"][0] == "B4_w0.5_near"

    def test_an_exact_tie_is_broken_by_id_not_by_insertion_order(self):
        mg = sel.summary_vector(_vector(0.1416, 0.4284))
        b0 = _probe_block(0.5294, 0.5256, 0.4737, 0.5072)
        same = {"method": "B4", "probe_retention": b0,
                "trainval_metrics": _vector(0.20, 0.40)}
        forward = {"B4_w2.0_b": dict(same), "B4_w0.5_a": dict(same)}
        backward = {"B4_w0.5_a": dict(same), "B4_w2.0_b": dict(same)}
        first = rs.select_successor(forward, mg, b0)
        second = rs.select_successor(backward, mg, b0)
        assert first["candidates"]["B4_w2.0_b"]["distance_to_mg"] == \
            first["candidates"]["B4_w0.5_a"]["distance_to_mg"]
        assert first["selected"] == second["selected"] == "B4_w0.5_a", (
            "the tie-break must be the frozen id order, not whichever "
            "candidate the dict happened to hold first")
        assert first["tied_with"] == ["B4_w2.0_b"]
        assert first["a_tie_means"] is not None

    def test_d_g_is_delegated_not_reimplemented(self, freeze):
        vec = sel.summary_vector(_vector(0.2, 0.4))
        mg = sel.summary_vector(_vector(0.1416, 0.4284))
        assert rs.distance_to_mg(vec, mg) == \
            sel.distance_to_reference(vec, mg)
        assert freeze["criterion"]["computed_by"].endswith(
            "distance_to_reference")
        assert freeze["criterion"]["reimplemented_in_this_iteration"] is False
        assert freeze["criterion"]["changed_from_pilot100"] is False

    def test_the_tolerance_is_the_rounding_already_in_place(self, freeze):
        tol = freeze["criterion"]["numerical_tolerance"]
        assert tol["distance_decimals"] == rs.DISTANCE_DECIMALS == 6
        assert tol["additional_epsilon_on_distances"] == 0.0
        assert tol["retention_rate_decimals"] == rs.RATE_DECIMALS == 4
        assert tol["tied_iff"] == "the two rounded distances are exactly equal"

    def test_the_tie_break_is_written_down(self, freeze):
        tb = freeze["criterion"]["tie_break"]
        assert tb["rule"] == "(distance_to_mg, candidate_id) ascending"
        assert tb["id_ordering"] == "lexicographic"
        assert tb["carries_no_outcome_information"] is True
        for dependency in ("grid order", "filesystem order",
                           "which adapters exist on disk"):
            assert dependency in tb["does_not_depend_on"]


# ── 5. the Stage-2 gate ─────────────────────────────────────────────────────

class TestTheStageTwoGateUsesNoMargin:
    def _report(self, ref_dist, best_dist, eligible=True):
        mg = sel.summary_vector(_vector(0.1416, 0.4284))
        b0 = _probe_block(0.5294, 0.5256, 0.4737, 0.5072)
        cands = {
            ITER12_REFERENCE_ROW: {
                "method": "B4", "probe_retention": b0,
                "trainval_metrics": _vector(*ref_dist)},
            "B4_w2.0_lam0.5_lr2e-05_ep5": {
                "method": "B4", "probe_retention": b0,
                "trainval_metrics": _vector(*best_dist)},
        }
        report = rs.select_successor(cands, mg, b0)
        if not eligible:
            for row in report["candidates"].values():
                row["eligible"] = False
            report["eligible"] = []
            report["selected"] = None
        return report

    def test_it_opens_when_nothing_is_eligible(self):
        report = self._report((0.2, 0.4), (0.15, 0.42), eligible=False)
        gate = rs.stage2_gate(report, ITER12_REFERENCE_ROW)
        assert gate["stage2_opens"] is True
        assert "no candidate satisfied the retention floor" in gate["reason"]
        assert gate["margin_required"] == 0.0

    def test_it_opens_on_an_exact_tie_with_the_reference_row(self):
        report = self._report((0.2, 0.4), (0.2, 0.4))
        gate = rs.stage2_gate(report, ITER12_REFERENCE_ROW)
        assert gate["stage2_opens"] is True, (
            "a mechanism that changes nothing has not earned the claim that "
            "replay was enough")
        assert gate["margin_required"] == 0.0

    def test_it_stays_closed_on_a_strict_improvement(self):
        report = self._report((0.40, 0.20), (0.15, 0.42))
        gate = rs.stage2_gate(report, ITER12_REFERENCE_ROW)
        assert gate["stage2_opens"] is False
        assert gate["best_eligible_distance_to_mg"] < \
            gate["reference_distance_to_mg"]
        assert gate["no_margin_is_used"] is True

    def test_it_refuses_without_a_reference_row(self):
        report = self._report((0.40, 0.20), (0.15, 0.42))
        with pytest.raises(ValueError, match="reference row"):
            rs.stage2_gate(report, "B4_w9.9_does_not_exist")

    def test_the_gate_rule_is_frozen_in_the_document(self, freeze):
        s2 = freeze["stage_2"]
        assert s2["mechanism"] == "MF-preservation regularizer"
        assert len(s2["opens_iff"]) == 2
        assert s2["margin_required"] == 0.0
        assert "STRICTLY" in s2["opens_iff"][1]


# ── 6. the protocol binds the run ───────────────────────────────────────────

class TestTheProtocolIsBoundNotJustWritten:
    def test_check_only_passes_against_the_repository(self):
        assert fip.verify_freeze(REPO_ROOT) == []

    def test_freezing_is_refused_once_a_candidate_exists(self, tmp_path):
        """The check the confirmation freeze did not have."""
        cid = grid_for_tag("iter12")[1].candidate_id
        (tmp_path / fip.CANDIDATE_ROOT / cid / "adapters").mkdir(parents=True)
        assert fip.trained_candidates(tmp_path) == [cid]
        out = subprocess.run(
            (sys.executable, "scripts/freeze_iter12_selection_protocol.py",
             "--repo-root", str(tmp_path)),
            cwd=REPO_ROOT, capture_output=True, text=True, check=False)
        assert out.returncode != 0
        assert "REFUSING to freeze" in out.stdout + out.stderr
        assert "must precede the candidates" in out.stdout + out.stderr

    def test_no_flag_reaches_past_the_post_training_refusal(self):
        """``--refreeze`` is permitted only BEFORE training.

        This is the difference from the confirmation protocol, whose
        ``--allow-refreeze`` flag was the only gate and therefore a request
        rather than a check -- the finding the 11C-5R3 audit filed.  Asserted
        on the order of the two refusals in the source, which is the property
        that matters: the unconditional one must be reached first.
        """
        source = Path(fip.__file__).read_text()
        unconditional = source.index("REFUSING to freeze:")
        flag_gated = source.index("if not args.refreeze")
        assert unconditional < flag_gated, (
            "the post-training refusal must be evaluated before any flag can "
            "bypass an overwrite check, or the flag becomes the only gate "
            "again")
        assert "trained = trained_candidates(repo_root)" in source
        assert source.index("trained = trained_candidates(repo_root)") < \
            unconditional

    def test_the_refreeze_flag_cannot_reach_past_a_trained_candidate(self,
                                                                    tmp_path):
        """Behavioural, not textual: pass the flag and it must still refuse.

        A test that greps the help text would pass on a promise.  This runs
        the script with ``--refreeze --reason`` against a tree that already
        holds a candidate adapter, which is the exact situation the
        confirmation protocol's ``--allow-refreeze`` flag allowed.
        """
        cid = grid_for_tag("iter12")[1].candidate_id
        (tmp_path / fip.CANDIDATE_ROOT / cid / "adapters").mkdir(parents=True)
        out = subprocess.run(
            (sys.executable, "scripts/freeze_iter12_selection_protocol.py",
             "--repo-root", str(tmp_path), "--refreeze",
             "--reason", "trying to re-freeze after training"),
            cwd=REPO_ROOT, capture_output=True, text=True, check=False)
        assert out.returncode != 0
        assert "REFUSING to freeze" in out.stdout + out.stderr
        assert "must precede the candidates" in out.stdout + out.stderr

    def test_the_freeze_is_not_one_of_the_sealed_analysis_scripts(self):
        confirm = json.loads(CONFIRM_FREEZE.read_text())
        sealed = set(confirm["code"]["analysis_scripts_sha256"])
        sealed |= set(confirm["code"]["fingerprinted_modules"])
        for rel in fip.PROTOCOL_PATHS:
            assert rel not in sealed, (
                f"{rel} is sealed by the confirmation freeze; adding it to "
                f"the Iteration-12 protocol set would bind bytes that must "
                f"not move")
        assert fip.OUT_REPORT not in sealed

    def test_the_sealed_dependencies_really_are_unchanged(self, freeze):
        """Cross-checked against the confirmation freeze, not asserted.

        The confirmation seals eighteen paths, but they are recorded in three
        places: ``analysis_scripts_sha256`` (7), ``fingerprinted_modules``
        (10), and ``primary_test.implementation`` -- ``paired_ci.py``, which is
        hashed separately precisely because adding it to the fingerprint set
        would have invalidated thirty committed sidecars.  A cross-check that
        read only the two dicts would silently skip it.
        """
        confirm = json.loads(CONFIRM_FREEZE.read_text())
        known = dict(confirm["code"]["analysis_scripts_sha256"])
        known.update(confirm["code"]["fingerprinted_modules"])
        impl = confirm["primary_test"]["implementation"]
        assert impl["in_the_code_fingerprint"] is False
        known[impl["module"]] = impl["sha256"]
        assert len(known) == 18, "the confirmation seals eighteen paths"

        checked = 0
        for rel, digest in freeze["hashes"]["depends_on_unchanged"].items():
            if rel in known:
                assert known[rel] == digest, (
                    f"{rel} is sealed by the confirmation and its bytes "
                    f"moved")
                checked += 1
        assert checked == 5, (
            "five of the sealed paths are dependencies of this protocol; if "
            "that count changed, either a dependency was dropped or a sealed "
            "path was touched")
        assert set(freeze["hashes"]["protocol_paths"]) & set(known) == set(), (
            "a protocol path must not also be a sealed one: Iteration 12 may "
            "not write to any of the eighteen")

    def test_the_amendment_records_what_it_superseded(self, freeze):
        for entry in freeze["amendments"]:
            assert entry["reason"]
            assert len(entry["supersedes_sha256"]) == 64
            assert entry["no_candidate_had_been_trained"] is True
            assert entry["fields_that_changed"]

    def test_the_selector_verifies_the_freeze_before_generating(self,
                                                                tmp_path):
        """End-to-end, and it needs no GPU: the check precedes generation."""
        out = subprocess.run(
            (sys.executable,
             "scripts/select_iter12_retention_checkpoints.py",
             "--repo-root", str(tmp_path)),
            cwd=REPO_ROOT, capture_output=True, text=True, check=False)
        assert out.returncode != 0
        assert "unfrozen criterion" in out.stdout + out.stderr

    def test_the_frozen_grid_matches_the_grid_in_code(self, freeze):
        grid = grid_for_tag("iter12")
        assert validate_grid(grid) == []
        assert [c.describe() for c in grid] == \
            freeze["grid"]["candidates"]
        assert freeze["grid"]["num_candidates"] == len(grid) == 9
        assert freeze["grid"]["reference_row"] == ITER12_REFERENCE_ROW


# ── 7. the confirmation is out of bounds ────────────────────────────────────

class TestTheStudyNeverReadsTheConfirmation:
    def test_a_confirmation_path_is_refused(self):
        bad = REPO_ROOT / "data/mllmu_hier_confirm100/predictions"
        with pytest.raises(SystemExit, match="retrospective"):
            sic.assert_no_forbidden_evidence([bad], REPO_ROOT)

    def test_a_traversal_path_into_the_confirmation_is_refused(self):
        """Checked on resolved paths, so ``..`` does not walk past it."""
        sneaky = REPO_ROOT / "data/mllmu_hier_pilot100" / \
            "../mllmu_hier_confirm100/predictions"
        with pytest.raises(SystemExit, match="REFUSING"):
            sic.assert_no_forbidden_evidence([sneaky], REPO_ROOT)

    def test_a_clean_path_is_allowed(self):
        sic.assert_no_forbidden_evidence(
            [REPO_ROOT / "data/mllmu_hier_pilot100/predictions_iter12",
             REPO_ROOT / fip.OUT_REPORT], REPO_ROOT)

    def test_the_predictions_directory_is_not_the_pilot_one(self):
        out = sic.predictions_dir_for(REPO_ROOT)
        assert out.name == "predictions_iter12"
        assert out != REPO_ROOT / "data/mllmu_hier_pilot100/predictions"

    def test_both_grids_contain_b0_which_is_why_the_dirs_are_separate(self):
        """The collision the separate directory exists to prevent."""
        assert "B0" in {c.candidate_id for c in grid_for_tag("iter12")}
        assert "B0" in {c.candidate_id for c in pilot100_grid()}
        assert sic.EXPERIMENT_ID != "mllmu_pilot100_iter11", (
            "a shared experiment id would make the pilot-100 B0 sidecar "
            "verify, and a different one would make it refuse and regenerate "
            "over committed evidence -- so the files must not share a "
            "directory either")

    def test_the_freeze_records_the_exclusion(self, freeze):
        base = freeze["evidence_base"]
        assert "data/mllmu_hier_confirm100" in base["never_read"]
        assert base["pilot100_selection_report"]["never_transcribed_into"]


# ── 8. the grid ─────────────────────────────────────────────────────────────

class TestTheGridIsWhatTheProtocolSays:
    def test_the_pilot_equivalent_weight_rederives_from_the_group_counts(self,
                                                                         probe):
        """Derived from committed files, not from the constant."""
        def count(path):
            return sum(1 for line in path.read_text().splitlines()
                       if line.strip())
        pilot = count(PILOT_GROUPS / "retain.jsonl")
        fit = count(ITER12_GROUPS / "retain.jsonl")
        target = count(PILOT_GROUPS / "fine_target.jsonl")
        #: Parenthesised on purpose.  ``assert pilot, fit, target == 387,
        #: 183, 90`` parses as a four-element tuple, which is always truthy,
        #: so it would have checked nothing at all.
        assert (pilot, fit, target) == (387, 183, 90)
        share_pilot = pilot / (2 * target + pilot)
        share_fit = fit / (2 * target + fit)
        assert round(share_pilot / share_fit, 4) == \
            ITER12_PILOT_EQUIVALENT_WEIGHT
        assert probe["replay_strength"][
            "weight_reproducing_the_pilot_influence"] == \
            ITER12_PILOT_EQUIVALENT_WEIGHT
        assert ITER12_PILOT_EQUIVALENT_WEIGHT in \
            {g.weight for c in grid_for_tag("iter12") for g in c.groups}

    def test_only_replay_strength_and_budget_are_swept(self):
        grid = [c for c in grid_for_tag("iter12") if not c.noop]
        assert all(c.method == "B4" for c in grid)
        for spec in grid:
            by_name = {g.name: g for g in spec.groups}
            assert by_name["fine_target"].mode == "gd"
            assert by_name["fine_target"].weight == 0.5, (
                "fine suppression must stay at the incumbent's weight, or a "
                "difference between rows is not attributable to replay")
            assert by_name["target_level"].weight == 1.0
            assert by_name["retain"].mode == "sft"
            assert by_name["retain"].weight > 0
        assert len({g.weight for c in grid
                    for g in c.groups if g.name == "retain"}) == 5

    def test_the_reference_row_is_weight_one(self):
        grid = {c.candidate_id: c for c in grid_for_tag("iter12")}
        ref = grid[ITER12_REFERENCE_ROW]
        weights = {g.name: g.weight for g in ref.groups}
        assert weights["retain"] == 1.0
        assert ref.overrides == {"learning_rate": 2e-5, "num_epochs": 5}

    def test_the_existing_grids_did_not_move(self):
        assert len(smoke_grid()) == 6
        assert len(pilot100_grid()) == 16
        assert validate_grid(smoke_grid()) == []
        assert validate_grid(pilot100_grid()) == []
        assert [c.candidate_id for c in pilot100_grid()][1] == \
            "B1_lr1e-05_ep10"

    def test_iter12_reuses_the_pilot_dataset_but_its_own_groups(self):
        assert dataset_dir_for_tag("iter12") == "data/mllmu_hier_pilot100"
        assert groups_subdir_for_tag("iter12") == "unlearning_iter12"
        assert groups_subdir_for_tag("pilot100") == "unlearning"
        assert groups_subdir_for_tag("smoke") == "unlearning"

    def test_the_trainer_resolves_the_fit_half_groups(self):
        from train_unlearning_baselines import resolve_paths
        data, mf, out_root, groups = resolve_paths("iter12", REPO_ROOT)
        assert data == REPO_ROOT / "data/mllmu_hier_pilot100"
        assert mf == REPO_ROOT / \
            "data/checkpoints/mllmu_pilot100/MF/adapters", (
            "Iteration 12 continues from the pilot-100 MF adapter; there is "
            "no separate reference-state training for this tag")
        assert out_root.name == "mllmu_iter12_unlearn"
        assert groups == REPO_ROOT / bip.ITER12_GROUPS
        assert (groups / "retain.jsonl").exists()

    def test_the_pilot_tag_still_resolves_unchanged(self):
        from train_unlearning_baselines import resolve_paths
        data, mf, out_root, groups = resolve_paths("pilot100", REPO_ROOT)
        assert data == REPO_ROOT / "data/mllmu_hier_pilot100"
        assert mf == REPO_ROOT / \
            "data/checkpoints/mllmu_pilot100/MF/adapters"
        assert out_root.name == "mllmu_pilot100_unlearn"
        assert groups == PILOT_GROUPS
