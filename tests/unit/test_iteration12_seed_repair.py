"""Stage-1b scoring repair — tested on CPU, with no adapter and no GPU.

The frozen analyzer crashed after scoring all eight replicates and before
writing its report, on a field that is not decision-bearing.  Both files on
that path are ``PROTOCOL_PATHS`` in a freeze that refuses amendment now that
the replicates exist, so the repair is a separate script that runs the frozen
code with one call unwrapped.

These tests have two jobs.  The first is the ordinary one: that the repair's
guards do what they say.  The second is specific to a repair, and matters more:
that the frozen bytes are still the frozen bytes, and that the verdict in the
report is the one the frozen rule gives when it is recomputed from the report's
own per-replicate numbers rather than trusted from it.
"""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "src"))

import analyze_iter12_seed_replication as aisr
import freeze_iter12_seed_replication as fisr
import repair_iter12_seed_replication_scoring as rep

from granunlearn.evaluation import retention_selection as rs
from granunlearn.training.seed_replication import (
    ALL_SEEDS,
    FLOOR_NUMBERS,
    mean_candidate,
)

REPORT = REPO_ROOT / aisr.OUT_REPORT
REPAIR = REPO_ROOT / rep.REPAIR_RECORD
FREEZE = REPO_ROOT / fisr.OUT_REPORT
STAGE1 = REPO_ROOT / aisr.STAGE1_SELECTION_REPORT


def report() -> dict:
    return json.loads(REPORT.read_text())


def repair_record() -> dict:
    return json.loads(REPAIR.read_text())


def freeze_doc() -> dict:
    return json.loads(FREEZE.read_text())


def reference_vector() -> dict:
    """The MG vector the frozen analyzer passes to ``distance_to_mg``.

    Read from the filed Stage-1 report rather than constructed here: D_G's
    components are a fixed list, and a hand-written vector with one missing
    would return None for a reason that has nothing to do with what is being
    tested.
    """
    return json.loads(STAGE1.read_text())["reference"]["vector"]


# ──────────────────────────────────────────────────────────────────────
class TestTheFrozenBytesAreStillTheFrozenBytes:
    """The repair's whole licence to exist.

    If a frozen file had been edited, the result could have been fitted to the
    replicates and the preregistration would be worthless — so this is checked
    against the freeze's own record rather than against a constant here.
    """

    def test_every_protocol_path_hashes_to_what_the_freeze_records(self):
        recorded = freeze_doc()["hashes"]["protocol_paths"]
        assert len(recorded) == 4, recorded
        for rel, want in recorded.items():
            got = rep._sha256(REPO_ROOT / rel)
            assert got == want, f"{rel} was edited after the freeze"

    def test_the_guard_agrees_and_is_not_reading_itself(self):
        out = rep.frozen_paths_match_the_freeze()
        assert out["all_match"] is True
        assert out["num_protocol_paths"] == 4
        #: Derived from the committed freeze, so the two cannot agree by
        #: construction — one is the file, the other is the working tree.
        for rel, v in out["per_path"].items():
            assert v["freeze_records"] == v["now"], rel

    def test_the_guard_refuses_on_drift(self, monkeypatch):
        real = rep._sha256
        calls = {"n": 0}

        def drifted(path):
            calls["n"] += 1
            return "0" * 64 if calls["n"] == 1 else real(path)

        monkeypatch.setattr(rep, "_sha256", drifted)
        with pytest.raises(SystemExit) as exc:
            rep.frozen_paths_match_the_freeze()
        assert "drifted" in str(exc.value)

    def test_the_repair_record_claims_no_frozen_file_was_edited(self):
        assert repair_record()["frozen_files_edited"] == []


# ──────────────────────────────────────────────────────────────────────
class TestThePatchCannotReachTheDecision:
    def test_no_decision_path_function_reaches_distance_to_mg(self):
        out = rep.decision_path_cannot_reach_the_patched_function()
        assert out["functions_reaching_distance_to_mg"] == []
        #: And the walk really covered the functions that produce the verdict.
        walked = out["walked"]
        assert "floor_check" in walked[
            "src/granunlearn/evaluation/retention_selection.py"]
        assert "mean_candidate" in walked[
            "src/granunlearn/training/seed_replication.py"]

    def test_it_refuses_when_a_decision_function_does_reach_it(self,
                                                               monkeypatch):
        """Not a hypothetical: ``select_successor`` is the Stage-1 function that
        really does call ``distance_to_mg``, and Stage 1b never runs it.  Adding
        it to the decision path must stop the repair."""
        rel = "src/granunlearn/evaluation/retention_selection.py"
        monkeypatch.setitem(rep.DECISION_PATH, rel,
                            (*rep.DECISION_PATH[rel], "select_successor"))
        with pytest.raises(SystemExit) as exc:
            rep.decision_path_cannot_reach_the_patched_function()
        assert "reach distance_to_mg" in str(exc.value)

    def test_it_refuses_when_a_named_function_does_not_exist(self, monkeypatch):
        rel = "src/granunlearn/training/seed_replication.py"
        monkeypatch.setitem(rep.DECISION_PATH, rel, ("no_such_function",))
        with pytest.raises(SystemExit) as exc:
            rep.decision_path_cannot_reach_the_patched_function()
        assert "has no function" in str(exc.value)


# ──────────────────────────────────────────────────────────────────────
class TestThePatchIsConfinedToOneCallSite:
    def test_the_analyzer_has_exactly_one_call_site(self):
        sites = rep.analyzer_call_sites()
        assert len(sites) == 1
        line = next(iter(sites))
        source = (REPO_ROOT / rep.ANALYZER_REL).read_text().splitlines()
        #: Checked against the file's own text: an AST line number that did not
        #: land on the call would make every later claim about "that one site"
        #: a claim about some other line.
        assert "rs.distance_to_mg" in source[line - 1], source[line - 1]

    def test_it_refuses_when_there_is_more_than_one(self, tmp_path,
                                                    monkeypatch):
        two = tmp_path / "two_sites.py"
        two.write_text(
            "import x as rs\n"
            "a = rs.distance_to_mg(1)\n"
            "b = rs.distance_to_mg(2)\n")
        monkeypatch.setattr(rep, "ANALYZER_REL", str(two))
        with pytest.raises(SystemExit) as exc:
            rep.analyzer_call_sites()
        assert "not just the one" in str(exc.value)

    def test_the_run_measured_eight_calls_all_from_that_site(self):
        proof = repair_record()["the_patch"]["proof_only_one_site_saw_it"]
        assert proof["calls_served"] == 8
        assert proof["calls_from_anywhere_else"] == []
        assert proof["call_sites_in_the_analyzer"] == \
            sorted(rep.analyzer_call_sites())
        #: Eight replicates, one call each — a count that would be different if
        #: anything else in the process had consumed the patched behaviour.
        assert len(report()["replicates"]) == 8

    def test_the_function_is_unpatched_in_this_process(self):
        """The ``finally`` restored it, and it still returns the tuple its own
        module's callers unpack."""
        ref = reference_vector()
        out = rs.distance_to_mg(ref, ref)
        assert isinstance(out, tuple) and len(out) == 2
        assert out[0] == pytest.approx(0.0)

    def test_the_wrapper_returns_the_distance_and_logs_the_caller(self):
        log: list = []
        ref = reference_vector()
        patched = rep.unwrap_distance_to_mg(log)
        got = patched(ref, ref)
        assert isinstance(got, float)
        assert got == pytest.approx(0.0)
        assert len(log) == 1
        assert log[0]["caller_file"] == Path(__file__).name
        assert log[0]["caller_line"] > 0

    def test_the_wrapper_refuses_a_none_distance(self, monkeypatch):
        monkeypatch.setattr(rs, "distance_to_mg",
                            lambda *a, **k: (None, ["missing_component"]))
        patched = rep.unwrap_distance_to_mg([])
        with pytest.raises(SystemExit) as exc:
            patched({"a": None}, {"a": 1.0})
        assert "returned None" in str(exc.value)


# ──────────────────────────────────────────────────────────────────────
class TestTheVerdictIsTheFrozenRuleRecomputed:
    """The result is re-derived here from the report's own per-replicate
    numbers, not read off the field that states it."""

    def test_the_bug_is_gone_from_the_report(self):
        for rid, r in report()["replicates"].items():
            assert isinstance(r["distance_to_mg"], float), \
                f"{rid} still carries the tuple the crash came from"

    def test_recomputing_the_floor_from_the_replicates_gives_the_same_verdict(
            self):
        doc = report()
        b0 = json.loads(STAGE1.read_text())["floor"]["b0"]
        for parent, v in doc["verdicts"].items():
            kids = v["replicates"]
            means = {(f, e): [doc["replicates"][k]["probe_retention"][f][e]
                              for k in kids]
                     for f, e in FLOOR_NUMBERS}
            floor = rs.floor_check(mean_candidate(means), b0, rs.FLOOR_EPSILON)
            assert floor == v["floor_on_the_mean"], parent
            assert v["passes_on_the_mean"] == floor["eligible"]

    def test_the_means_in_the_report_are_the_means_of_the_replicates(self):
        doc = report()
        for v in doc["verdicts"].values():
            for key, pn in v["per_number"].items():
                family, estimand = key.split(".")
                vals = [doc["replicates"][k]["probe_retention"][family][estimand]
                        for k in v["replicates"]]
                assert pn["values"] == vals, key
                assert pn["mean"] == pytest.approx(sum(vals) / len(vals))
                assert pn["k"] == len(ALL_SEEDS)
                #: The comparison is unrounded, so a mean that only clears the
                #: anchor after rounding to four decimals must not be reported
                #: as clearing it.  Both sides parenthesised: ``A == B >= C``
                #: chains in Python and would test something else entirely.
                clears = pn["mean_minus_anchor"] >= -rs.FLOOR_EPSILON
                assert clears == (pn["mean"] >= pn["anchor"] - rs.FLOOR_EPSILON)

    def test_the_anchor_is_stage1s_b0_and_not_something_chosen_here(self):
        b0 = json.loads(STAGE1.read_text())["floor"]["b0"]
        values = report()["anchor"]["values"]
        for f, e in FLOOR_NUMBERS:
            assert values[f"{f}.{e}"] == b0[f][e]


# ──────────────────────────────────────────────────────────────────────
class TestTheResult:
    """Pinned so a later change to the rule or the evidence cannot quietly
    move a filed result."""

    def test_neither_parent_passes_on_the_mean(self):
        verdicts = report()["verdicts"]
        assert set(verdicts) == {"B4_w4.0_lam0.5_lr2e-05_ep5",
                                 "B4_w2.0_lam0.5_lr2e-05_ep3"}
        assert all(v["passes_on_the_mean"] is False for v in verdicts.values())

    def test_how_many_of_the_four_numbers_each_parent_clears(self):
        per = {p: sum(1 for pn in v["per_number"].values()
                      if pn["mean_minus_anchor"] >= -rs.FLOOR_EPSILON)
               for p, v in report()["verdicts"].items()}
        #: Replication moved the w2.0/ep3 parent onto two of the four and left
        #: the w4.0/ep5 parent on none.  Neither is enough: the rule is all
        #: four, with no margin.
        assert per["B4_w2.0_lam0.5_lr2e-05_ep3"] == 2
        assert per["B4_w4.0_lam0.5_lr2e-05_ep5"] == 0

    def test_every_number_straddles_the_anchor(self):
        classes = {pn["range_classification"]
                   for v in report()["verdicts"].values()
                   for pn in v["per_number"].values()}
        assert classes == {"straddles"}, classes

    def test_seed_variance_is_larger_than_the_binding_shortfall(self):
        """The finding that decides what to do next.

        The shortfalls are one to two queries on a 76-query measure, and the
        seed-to-seed spread on that same measure is larger than the shortfall
        it is being compared against — so the floor cannot separate these
        candidates from the anchor even in principle at this sample size.
        """
        worst = {}
        for parent, v in report()["verdicts"].items():
            pn = v["per_number"]["retain_other_entity.row_micro"]
            worst[parent] = {"mean_minus_anchor": pn["mean_minus_anchor"],
                             "sd": pn["sd"],
                             "range": pn["max"] - pn["min"]}
            assert pn["sd"] > abs(pn["mean_minus_anchor"]), parent
        #: retain-other row-micro resolves in units of 1/76.
        one_query = 1 / 76
        for parent, w in worst.items():
            assert abs(w["mean_minus_anchor"]) < 3 * one_query, parent

    def test_replication_did_not_reorder_the_two_parents_on_d_g(self):
        d = {p: v["distance_to_mg"]["mean"]
             for p, v in report()["verdicts"].items()}
        #: D_G is reported and not decision-bearing, but the seed spread on it
        #: is the reason the Stage-1 tie-break's sixth decimal is not physical.
        assert d["B4_w4.0_lam0.5_lr2e-05_ep5"] < d["B4_w2.0_lam0.5_lr2e-05_ep3"]
        for p, v in report()["verdicts"].items():
            assert v["distance_to_mg"]["sd"] > 0.005, p

    def test_the_control_passed_on_every_field_not_just_the_four(self):
        ctrl = report()["generation_determinism_control"]
        assert ctrl["floor_is_reproducible"] is True
        assert ctrl["num_queries"] == 4518
        assert set(ctrl["rows_differing_by_field"].values()) == {0}


# ──────────────────────────────────────────────────────────────────────
class TestTheRepairDisclosesItself:
    def test_the_record_says_the_preregistered_code_did_not_finish(self):
        r = repair_record()
        assert "did not" in r["what_still_cannot_be_claimed"]
        assert r["crash_reproduced_first"]["reproduced"] is True
        assert "not 'tuple'" in r["crash_reproduced_first"]["exception"]

    def test_the_crashed_run_neither_wrote_nor_changed_the_report(self):
        #: Otherwise the filed result could have come from the unpatched run and
        #: the patch would be unverifiable.  Compared by hash either side rather
        #: than by existence: the report exists now because the PATCHED run
        #: wrote it, so "does the file exist" answers the wrong question — and
        #: did, on the first version of this check.
        crash = repair_record()["crash_reproduced_first"]
        assert crash["the_crashed_run_wrote_or_changed_the_report"] is False
        assert crash["report_sha256_before_the_crashed_run"] == \
            crash["report_sha256_after_the_crashed_run"]

    def test_the_bug_is_recorded_as_not_decision_bearing(self):
        bug = repair_record()["the_bug"]
        assert bug["decision_bearing"] is False
        assert bug["site"].endswith(":325")
        assert bug["site"] == \
            f"{rep.ANALYZER_REL}:{next(iter(rep.analyzer_call_sites()))}"

    def test_the_report_itself_says_d_g_does_not_decide(self):
        stated = report()["reported_but_not_decision_bearing"]["distance_to_mg"]
        assert "the floor decides eligibility" in stated

    def test_both_new_reports_are_valid_json_and_committed(self):
        for p in (REPORT, REPAIR):
            assert p.exists(), p
            json.loads(p.read_text())
            tracked = (REPO_ROOT / ".gitignore").read_text()
            assert f"!{p.relative_to(REPO_ROOT)}" in tracked, \
                f"{p.name} is gitignored, so it would never be committed"

    def test_the_confidential_split_was_never_read(self):
        never = report()["never_read"]
        assert "data/mllmu_hier_confirm100" in never
        assert "data/reports/mllmu_confirm100_final_analysis.json" in never


# ──────────────────────────────────────────────────────────────────────
class TestTheRepairScriptIsNotItselfABackDoor:
    def test_it_is_not_a_protocol_path_and_not_hash_bound(self):
        rel = "scripts/repair_iter12_seed_replication_scoring.py"
        assert rel not in freeze_doc()["hashes"]["protocol_paths"]
        assert rel not in freeze_doc()["hashes"]["depends_on_unchanged"]

    def test_it_cannot_write_the_freeze(self):
        """The repair edits nothing frozen, so it must not touch the freeze."""
        src = (REPO_ROOT / rep.__file__).read_text()
        tree = ast.parse(src)
        written = [n for n in ast.walk(tree)
                   if isinstance(n, ast.Call)
                   and isinstance(n.func, ast.Attribute)
                   and n.func.attr in ("write_text", "write_bytes")]
        #: Exactly one write: the repair record.  A second would mean this
        #: script can produce or alter an artifact it is supposed only to
        #: describe.
        assert len(written) == 1, len(written)
        #: And it is the record, not the report and not the freeze: one write
        #: call in the whole file, on the line that writes REPAIR_RECORD.
        line = src.splitlines()[written[0].lineno - 1]
        assert "write_text" in line, line
        assert src.count("write_text") == 1, \
            "the repair writes more than its own record"
        assert "freeze" not in src.split("out = REPO_ROOT / REPAIR_RECORD")[1]

    def test_it_never_invokes_the_freeze_writer(self):
        src = (REPO_ROOT / rep.__file__).read_text()
        assert "fisr.main(" not in src
        assert "--refreeze" not in src
