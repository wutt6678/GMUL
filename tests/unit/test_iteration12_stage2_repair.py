"""The Stage-2 selection repair — tested on CPU, with no adapter and no GPU.

The frozen selector crashed twice: once on a missing argument to a field that
floors nothing, and once because the anchor MG won a ranking the same file says
it is excluded from.  Both files on that path are bound by a freeze that refuses
amendment now that seven adapters exist, and the selector verifies that freeze as
its own Step 1, so the repair is a separate script that patches two names for the
duration of one ``main()`` call.

These tests have three jobs.  The ordinary one: that the repair's guards do what
they say.  The one specific to a repair: that the frozen bytes are still the
frozen bytes, and that the filed verdict is the one the frozen rule gives when it
is recomputed from the report's own per-candidate numbers rather than trusted from
the field that states it.  And a third that only exists because the SECOND patch
changes an outcome: that the record says so plainly, in a field a reader will find,
rather than leaving them to infer it from a diff.

What is pinned and what is not follows a rule worth stating, because a test that
pins the wrong thing breaks for a reason that has nothing to do with the invariant
it names.  Line numbers inside ``scripts/select_iter12_stage2.py`` and inside the
two freeze scripts ARE pinned as literals: every one of those files is bound by
sha256, so the number cannot move unless the freeze moves, and a test that refused
to name it would not catch the regression it exists for.  Values that record WHEN
or FROM WHERE this particular run happened — the record's own ``git_commit``, the
reconstructed HEAD, file sizes, mtime nanoseconds, the number of invocations — are
never pinned, because they are circumstances of a run and not properties of the
protocol.
"""

from __future__ import annotations

import ast
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "src"))

import freeze_iter12_stage2 as fis2
import repair_iter12_stage2_selection as rep
import select_iter12_stage2 as sis2

from granunlearn.evaluation import retention_selection as rs
from granunlearn.evaluation import route_stratified_retention as rt
from granunlearn.training import stage2_grid as sg

REPORT = REPO_ROOT / sg.OUT_REPORT
REPAIR = REPO_ROOT / rep.REPAIR_RECORD
FREEZE = REPO_ROOT / fis2.OUT_REPORT
SELECTOR = REPO_ROOT / rep.SELECTOR_REL
LANE_SCRIPT = REPO_ROOT / "scripts" / "lanes" / "iter12_stage2.sh"

#: The two lines the first defect is on, and the four the selector already filters
#: the anchor at.  Literals on purpose: see the module docstring.
BROKEN_SITES = [550, 579]
ANCHOR_FILTER_SITES = [294, 299, 304, 331]
RANKING_BINDINGS = {"eligible": 232, "ranked": 233}
CRASH_LINE_BUG_2 = 630


def report() -> dict:
    return json.loads(REPORT.read_text())


def record() -> dict:
    return json.loads(REPAIR.read_text())


def freeze_doc() -> dict:
    return json.loads(FREEZE.read_text())


def _history_can_reconstruct(frozen_at_utc: str) -> bool:
    """Whether THIS checkout can turn that one timestamp into a commit.

    The same query the script runs, asked of git directly rather than inferred
    from what the script returned -- a test that branched on the returned value
    would also pass against a script that disclosed unconditionally and measured
    nothing.

    Asked PER FREEZE, because the answer is a property of one timestamp against
    one checkout and not of the checkout alone.  ``git rev-parse
    --is-shallow-repository`` is not that question and must not be used as a
    proxy for it: a depth-8 clone reports ``true`` and still contains both
    commits these two freezes need, so branching on it demands a disclosure from
    a checkout that could have measured, and fails on the measurement it should
    have required instead.  Shallow says history was TRUNCATED; it does not say
    how much, or whether what is needed survived the truncation.
    """
    return bool(subprocess.run(
        ("git", "rev-list", "-1", f"--before={frozen_at_utc}", "HEAD"),
        cwd=REPO_ROOT, capture_output=True, text=True,
        check=False).stdout.strip())


def selector_lines() -> list[str]:
    return SELECTOR.read_text().splitlines()


def repo_env() -> dict:
    """An environment in which ``granunlearn`` resolves to THIS repository.

    A subprocess inherits whatever editable install happens to be present, and an
    editable install is a path, not a promise: a verification clone made elsewhere
    and never cleaned up will serve its own ``granunlearn`` to a child process
    started from here.  That is not hypothetical — it made a freeze ``--check-only``
    fail on an import this repository satisfies, in a suite whose in-process tests
    all passed, because which tree wins depended on collection order.  In CI the
    install is the checkout and this is a no-op; it is what makes the subprocess
    checks mean the same thing in both places.
    """
    return dict(os.environ, PYTHONPATH=os.pathsep.join(
        [str(REPO_ROOT / "src"), str(REPO_ROOT / "scripts"),
         os.environ.get("PYTHONPATH", "")]))


def _crash_line_number(traceback_line: str) -> int:
    """The line number out of a recorded traceback frame.

    The same pattern the repair uses to parse the chain log, repeated here rather
    than imported: a test that reused the code's own parser would agree with it by
    construction and could not catch the parser being wrong.  It was wrong once —
    it split on a quoted ``'"line '`` that does not occur in a traceback, whose
    format is ``, line 630, in main``, and the resulting silent None made the
    record claim the crashed run HAD filed a report, which is the opposite of the
    truth and the one thing that field exists to establish.
    """
    m = re.search(r"\bline (\d+), in ", traceback_line)
    assert m, f"not a traceback frame: {traceback_line!r}"
    return int(m.group(1))


# ──────────────────────────────────────────────────────────────────────
class TestTheFrozenBytesAreStillTheFrozenBytes:
    """The repair's whole licence to exist.

    If a frozen file had been edited the result could have been fitted to the
    candidates, and the preregistration would be worth nothing — so this is
    checked against the freeze's own record rather than against a constant here.
    """

    def test_every_bound_path_hashes_to_what_the_freeze_records(self):
        hashes = freeze_doc()["hashes"]
        groups = ("protocol_paths", "depends_on_unchanged", "data_paths")
        total = 0
        for group in groups:
            for rel, want in hashes[group].items():
                path = REPO_ROOT / rel
                if not path.exists():
                    #: A data path may legitimately be absent from a clone; the
                    #: freeze's own --check-only reports those as "fewer bytes to
                    #: check".  A protocol path may not.
                    assert group == "data_paths", f"{rel} is bound and missing"
                    continue
                got = rep._stamp(path)[0]
                assert got == want, f"{rel} was edited after the freeze"
                total += 1
        assert total > 0

    def test_the_selector_and_the_callee_are_both_bound(self):
        out = rep.frozen_paths_match_the_freeze()
        assert out["the_selector_is_one_of_them"] is True
        assert out["the_callee_is_one_of_them"] is True
        assert out["verified"] is True

    def test_the_guard_refuses_when_a_bound_path_drifts(self, monkeypatch):
        real = fis2.verify_freeze
        monkeypatch.setattr(fis2, "verify_freeze",
                            lambda root: ["scripts/x.py drifted"])
        with pytest.raises(SystemExit) as exc:
            rep.frozen_paths_match_the_freeze()
        assert "unfrozen criterion" in str(exc.value)
        monkeypatch.setattr(fis2, "verify_freeze", real)

    def test_the_record_claims_no_frozen_file_was_edited(self):
        assert record()["frozen_files_edited"] == []

    def test_the_freeze_still_verifies_from_its_own_entry_point(self):
        #: Not the repair's wrapper — the freeze's own check, so a repair that had
        #: quietly weakened the guard would still be caught by the thing it guards.
        p = subprocess.run(
            (sys.executable, "scripts/freeze_iter12_stage2.py", "--check-only"),
            cwd=REPO_ROOT, capture_output=True, text=True, check=False,
            env=repo_env())
        assert p.returncode == 0, p.stderr[-2000:]


# ──────────────────────────────────────────────────────────────────────
class TestTheFirstDefectIsExactlyTheTwoSitesItClaims:
    """The arity pin.

    This is the test that would have prevented the crash: 1944 unit tests passed
    over a file that could not run, because both broken calls are inside
    ``main()``, which needs torch and the real parquets.  Auditing the frozen
    selector statically is what turns "we found one" into "we found all of them".
    """

    def test_the_selector_calls_the_callee_at_exactly_those_two_lines(self):
        sites = rep.selector_call_sites()["sites"]
        #: Sorted, because the derivation walks the AST and ast.walk is not source
        #: order; the claim is which lines, not what order a traversal found them.
        assert sorted(s["line"] for s in sites) == BROKEN_SITES
        assert all(s["num_positional"] == 3 for s in sites), sites
        assert all(s["num_keyword"] == 0 for s in sites), sites

    def test_the_ast_line_numbers_land_on_the_call_in_the_file(self):
        #: An AST line number that did not land on the call would make every later
        #: claim about "those two sites" a claim about some other line.
        src = selector_lines()
        for line in BROKEN_SITES:
            assert "rt.pooled_all_routes(" in src[line - 1], src[line - 1]

    def test_the_callee_really_requires_four_arguments_and_names_the_fourth(self):
        args = rt.pooled_all_routes.__code__.co_varnames[
            :rt.pooled_all_routes.__code__.co_argcount]
        assert list(args) == ["predictions", "queries", "associations",
                              "probe_entities"]
        assert rep.REQUIRED_ARGS == len(args)

    def test_the_missing_argument_is_the_one_the_selector_already_binds(self):
        #: The repair does not choose a value; the selector reads it at line 499
        #: from a sha-bound report and the sibling calls two lines away pass it.
        src = selector_lines()
        assert "probe_entities" in src[498], src[498]
        assert rep.probe_entities() == json.loads(
            (REPO_ROOT / rep.PROBE_REPORT).read_text()
        )["halves"]["probe"]["entities"]

    def test_the_audit_covers_every_call_and_tolerates_only_those_two(self):
        audit = rep.arity_audit()
        assert audit["known_broken_sites"] == BROKEN_SITES
        assert audit["any_other_arity_problem"] == []
        assert audit["unresolved_calls"] == []
        #: An equality, not a floor.  Every file the audit reads is sha-bound, so
        #: the count cannot move unless the freeze moves — and pinning it is the
        #: only thing that catches the regression this audit was rewritten for: a
        #: first version resolved half the call sites, reported no problems, and
        #: was worse than no audit at all because it looked like one.
        assert audit["calls_with_decidable_arity"] == 27

    def test_the_audit_follows_a_re_export_rather_than_giving_up_on_it(self):
        #: ``rs.summary_vector`` is imported INTO retention_selection.py, not
        #: defined there, and the calls to it sit at lines 207, 547 and 575 —
        #: either side of the bug and on the path that produces the decision
        #: vector.  Treating them as unresolvable is exactly the half-audit above.
        found = rep._find_def(
            REPO_ROOT / "src/granunlearn/evaluation/retention_selection.py",
            "summary_vector")
        assert found is not None
        assert isinstance(found, ast.FunctionDef)

    def test_it_refuses_when_a_third_site_is_broken(self, tmp_path, monkeypatch):
        three = tmp_path / "three_sites.py"
        three.write_text(
            "import x as rt\n"
            "a = rt.pooled_all_routes(1, 2, 3)\n"
            "b = rt.pooled_all_routes(1, 2, 3)\n"
            "c = rt.pooled_all_routes(1, 2, 3)\n")
        monkeypatch.setattr(rep, "SELECTOR_REL", str(three))
        with pytest.raises(SystemExit) as exc:
            rep.selector_call_sites()
        assert "exactly two calls" in str(exc.value)

    def test_it_refuses_when_a_call_it_cannot_resolve_sits_beside_the_bug(
            self, tmp_path, monkeypatch):
        #: An unresolvable call is reported as a refusal rather than skipped, so a
        #: bound module losing a function stops the repair instead of narrowing it.
        #: The alias has to resolve to a real file for the call to be audited at
        #: all — an import of a module that does not exist is not an alias the
        #: audit ever sees, which is why this points at the real callee.
        src = tmp_path / "sel.py"
        src.write_text(
            "import granunlearn.evaluation.route_stratified_retention as rt\n"
            "a = rt.pooled_all_routes(1, 2, 3)\n"
            "b = rt.pooled_all_routes(1, 2, 3)\n"
            "c = rt.a_function_that_is_not_there(1)\n")
        monkeypatch.setattr(rep, "SELECTOR_REL", str(src))
        with pytest.raises(SystemExit) as exc:
            rep.arity_audit()
        assert "unresolved" in str(exc.value)


# ──────────────────────────────────────────────────────────────────────
class TestTheFirstPatchCannotReachTheDecision:
    def test_the_floor_reads_the_routes_and_never_the_patched_field(self):
        out = rep.the_patched_field_floors_nothing()
        assert out["floor_vector_reads"]["routes"] is True
        assert out["floor_vector_reads"][rep.PATCHED] is False

    def test_no_decision_bearing_call_is_passed_the_patched_field(self):
        out = rep.the_patched_field_floors_nothing()
        assert out["any_of_them_passed_the_patched_field"] == []
        #: And the walk really covered the calls that produce the verdict, so the
        #: empty list above is not the empty list of a walk that found nothing.
        assert out["decision_bearing_calls_in_the_selector"] > 0
        names = {c["function"] for c in _decision_calls()}
        assert {"summary_vector", "distance_to_mg", "floor_check_stratified",
                "rank_key"} <= names

    def test_the_callee_documents_the_field_as_floored_by_nothing(self):
        src = (REPO_ROOT / rep.RT_REL).read_text()
        assert "floored" in src.casefold()
        assert "POOLED_FAMILIES" in src

    def test_the_selector_copies_the_field_beside_the_decision_not_into_it(self):
        src = selector_lines()
        assert '"pooled_all_routes": info.get(' in src[226], src[226]
        assert 'rs.summary_vector(info["trainval_metrics"])' in src[206], src[206]

    def test_it_refuses_when_the_floor_does_read_the_patched_field(
            self, tmp_path, monkeypatch):
        bad = tmp_path / "rt.py"
        bad.write_text(
            "def floor_vector(by_route):\n"
            "    return by_route['pooled_all_routes']\n")
        monkeypatch.setattr(rep, "RT_REL", str(bad))
        with pytest.raises(SystemExit) as exc:
            rep.the_patched_field_floors_nothing()
        assert "not the shape the proof needs" in str(exc.value)

    def test_the_wrapper_supplies_the_argument_only_when_a_caller_omits_it(self):
        log: list = []
        calls: list = []
        original = rt.pooled_all_routes

        def spy(predictions, queries, associations, probe_entities):
            calls.append(probe_entities)
            return "sentinel"

        rt.pooled_all_routes = spy
        try:
            patched = rep.supply_probe_entities(log, ["e1", "e2"])
            assert patched(1, 2, 3) == "sentinel"
            assert patched(1, 2, 3, ["other"]) == "sentinel"
        finally:
            rt.pooled_all_routes = original

        #: The decision path's own call always passes four arguments, so it is
        #: served nothing and sees the original function's behaviour exactly.
        assert calls == [["e1", "e2"], ["other"]]
        assert [c["argument_supplied_by_the_repair"] for c in log] == [True, False]
        assert all(c["caller_file"] == Path(__file__).name for c in log)
        assert all(c["caller_line"] > 0 for c in log)

    def test_the_callee_is_unpatched_in_this_process(self):
        assert rt.pooled_all_routes.__name__ == "pooled_all_routes"
        assert rt.pooled_all_routes.__code__.co_argcount == 4
        assert sis2.build_report.__name__ == "build_report"


def _decision_calls() -> list[dict]:
    tree = ast.parse(SELECTOR.read_text())
    return [{"line": c.lineno, "function": c.func.attr}
            for c in ast.walk(tree)
            if isinstance(c, ast.Call) and isinstance(c.func, ast.Attribute)
            and c.func.attr in rep.DECISION_CALLS]


# ──────────────────────────────────────────────────────────────────────
class TestTheSecondDefectIsTheAnchorWinningItsOwnSelection:
    """The patch that DOES change an outcome, so the bar is different.

    What has to be shown is not that it changes nothing but that excluding the
    anchor is what the frozen protocol says, in several independent places, and
    not a preference held by a script written after the protocol was frozen.
    """

    def test_the_freeze_names_the_anchor_and_states_its_rule_over_candidates(self):
        out = rep.the_anchor_is_not_a_candidate()
        assert out["the_freeze_names_it_as_the_anchor"] == rep.ANCHOR
        assert "candidate" in out["the_frozen_rule_is_stated_over_candidates"]
        frozen = freeze_doc()["what_is_frozen"]
        assert frozen["primary_anchor"] == rep.ANCHOR

    def test_the_anchor_is_not_a_row_of_the_frozen_grid(self):
        out = rep.the_anchor_is_not_a_candidate()
        assert out["the_anchor_is_a_grid_row"] is False
        assert out["grid_rows"] == 8
        cal = json.loads((REPO_ROOT / sg.CALIBRATION_REPORT).read_text())
        ids = [c.candidate_id
               for c in sg.stage2_grid(cal["grid_rule"]["beta_star"])]
        assert rep.ANCHOR not in ids
        assert len(ids) == 8

    def test_build_report_takes_the_anchor_through_parameters_of_its_own(self):
        #: This is what makes it safe to filter the ROWS: the floor, the anchor
        #: values and the reported MG block are all computed from inputs the patch
        #: does not touch.  The recorded list is every parameter EXCEPT `generated`,
        #: which is the rows being filtered, so its absence is the point.
        out = rep.the_anchor_is_not_a_candidate()
        params = out["build_report_receives_the_anchor_separately_from_the_rows"]
        for needed in ("reused_eight", "reused_vec", "mg_vec"):
            assert needed in params, params
        assert "generated" not in params, params
        #: And `generated` really is a parameter of the frozen function — the one
        #: the patch filters — rather than something this test invented.
        tree = ast.parse(SELECTOR.read_text())
        br = next(n for n in tree.body
                  if isinstance(n, ast.FunctionDef) and n.name == "build_report")
        positional = [a.arg for a in br.args.posonlyargs + br.args.args]
        assert positional[0] == "generated"
        assert positional[1:] == params

    def test_the_selector_already_filters_the_anchor_in_four_places(self):
        out = rep.the_anchor_is_not_a_candidate()
        assert out["places_the_selector_already_filters_the_anchor"] == \
            ANCHOR_FILTER_SITES
        src = selector_lines()
        for line in ANCHOR_FILTER_SITES:
            assert '"MG"' in src[line - 1], (line, src[line - 1])

    def test_and_omits_the_filter_from_the_two_bindings_that_rank(self):
        out = rep.the_anchor_is_not_a_candidate()
        assert {b["binding"]: b["line"]
                for b in out["the_two_bindings_that_rank_and_do_not"]} == \
            RANKING_BINDINGS
        src = selector_lines()
        for binding, line in RANKING_BINDINGS.items():
            assert binding in src[line - 1], src[line - 1]
            assert '"MG"' not in src[line - 1], src[line - 1]
        #: Without that omission the frozen code would be doing the exclusion
        #: itself and the patch would only be interfering with it.
        assert src[RANKING_BINDINGS["eligible"] - 1].lstrip().startswith("eligible")
        assert src[RANKING_BINDINGS["ranked"] - 1].lstrip().startswith("ranked")

    def test_the_selector_declares_the_anchor_excluded_from_the_ranking(self):
        out = rep.the_anchor_is_not_a_candidate()
        line = out["the_selector_declares_it_excluded_at_line"]
        src = selector_lines()
        assert "excluded_from_the_ranking" in src[line - 1], src[line - 1]
        #: The reason is the selector's own sentence, not this test's opinion.
        window = "\n".join(src[line - 1:line + 4])
        assert "win its own selection" in window

    def test_the_crash_site_looks_the_winner_up_in_a_dict_that_excludes_it(self):
        src = selector_lines()
        assert 'report["candidates"][cid]' in src[CRASH_LINE_BUG_2 - 1], \
            src[CRASH_LINE_BUG_2 - 1]
        bug2 = record()["the_bugs"][1]
        assert bug2["exception"] == "KeyError: 'MG'"
        assert f"line {CRASH_LINE_BUG_2}" in bug2["where_it_surfaced"]
        #: The record does not merely name the line; it says the dictionary it
        #: looked in is one the same function built with an explicit exclusion.
        assert "candidates" in bug2["where_it_surfaced"]
        #: A set, not a list: the record's order comes from ast.walk, which is not
        #: source order, and pinning it would test the traversal rather than the
        #: finding.
        assert set(bug2["sites"]) == {
            f"{rep.SELECTOR_REL}:{line} ({name})"
            for name, line in RANKING_BINDINGS.items()}

    def test_dropping_one_key_from_a_sorts_input_cannot_reorder_the_rest(self):
        #: The claim the patch rests on, checked rather than asserted: with the
        #: anchor's distance 0 by construction it always sorted first, so removing
        #: it can only expose the order that was already behind it.
        rows = {"MG": 0.0, "a": 0.31, "b": 0.07, "c": 0.22}
        full = sorted(rows, key=lambda k: rs.rank_key(rows[k], k))
        kept = {k: v for k, v in rows.items() if k != "MG"}
        assert full[0] == "MG"
        assert full[1:] == sorted(kept, key=lambda k: rs.rank_key(kept[k], k))

    def test_the_wrapper_refuses_rows_that_are_not_the_anchor_being_dropped(self):
        log: list = []
        original = sis2.build_report
        sis2.build_report = lambda generated, *a, **k: sorted(generated)
        try:
            patched = rep.drop_the_anchor(log)
            assert patched({"MG": 1, "B0": 2}) == ["B0"]
            with pytest.raises(SystemExit) as exc:
                patched({"B0": 2, "B5": 3})
            assert "REFUSING" in str(exc.value)
        finally:
            sis2.build_report = original
        assert log[0]["rows_dropped"] == [rep.ANCHOR]
        assert log[0]["rows_kept"] == 1

    def test_it_refuses_if_the_frozen_code_ever_starts_filtering_itself(
            self, monkeypatch):
        #: Then the patch would be interfering with the protocol rather than
        #: applying it, and the repair has to stop saying so.
        monkeypatch.setattr(rep, "ANCHOR", "B0")
        with pytest.raises(SystemExit) as exc:
            rep.the_anchor_is_not_a_candidate()
        assert "REFUSING" in str(exc.value)


# ──────────────────────────────────────────────────────────────────────
class TestTheVerdictIsTheFrozenRuleRecomputed:
    """Re-derived from the report's own per-candidate numbers, not read off the
    field that states it."""

    def test_the_anchor_is_out_of_the_ranking_and_still_anchoring_the_floor(self):
        rep_doc = report()
        assert rep_doc["anchor_states"][rep.ANCHOR]["values"] == \
            rep_doc["criterion"]["anchor_values"]
        assert rep_doc["anchor_states"][rep.ANCHOR]["is_the_anchor"] is True
        assert rep.ANCHOR not in rep_doc["candidates"]
        assert rep.ANCHOR not in rep_doc["ranking_of_eligible"]
        assert rep_doc["selected"] != rep.ANCHOR

    def test_recomputing_eligibility_from_the_values_gives_the_filed_verdict(self):
        doc = report()
        anchor = doc["criterion"]["anchor_values"]
        baseline = doc["criterion"]["reported_baseline_values"]
        eps = doc["criterion"]["epsilon"]
        assert eps == rs.FLOOR_EPSILON
        for cid, row in doc["candidates"].items():
            primary = rt.floor_check_stratified(row["values"], anchor, eps)
            reported = rt.floor_check_stratified(row["values"], baseline, eps)
            assert primary == row["primary_floor"], cid
            assert reported == row["reported_b0_floor"], cid
            assert row["eligible"] == (primary["eligible"]
                                       and row["distance_to_mg"] is not None), cid

    def test_the_two_anchors_disagree_and_the_report_says_which_one_decides(self):
        doc = report()
        both = doc["eligibility_under_both_anchors"]
        assert sum(1 for v in both.values() if v["on_mg"]) == 0
        assert sum(1 for v in both.values() if v["on_b0"]) == 1
        assert doc["criterion"]["primary"].casefold().find("mg") >= 0
        #: The one candidate that clears the reported B0 floor is B0 itself, which
        #: is why the primary anchor was moved: the baseline passes its own floor
        #: trivially and would have disqualified the oracle state.
        passing = [c for c, v in both.items() if v["on_b0"]]
        assert passing == ["B0"], passing

    def test_the_ranking_is_empty_because_nothing_is_eligible(self):
        doc = report()
        assert doc["ranking_of_eligible"] == []
        assert doc["eligible"] == {} or len(doc["eligible"]) == 0
        assert doc["selected"] is None
        assert doc["tied_with"] == []

    def test_every_candidate_row_carries_the_eight_numbers_the_floor_reads(self):
        doc = report()
        keys = set(doc["criterion"]["anchor_values"])
        assert len(keys) == 8
        for cid, row in doc["candidates"].items():
            assert set(row["values"]) == keys, cid
            assert all(isinstance(v, float) for v in row["values"].values()), cid
            #: The patched field is present and is beside the decision, not in it.
            assert "pooled_all_routes" in row, cid

    def test_the_stage_3_gate_is_the_frozen_consequence_of_an_empty_ranking(self):
        gate = report()["stage_3_gate"]
        assert gate["stage3_opens"] is True
        assert gate["best_eligible"] is None
        assert gate["margin_required"] == 0.0
        assert gate["no_margin_is_used"] is True

    def test_the_confidential_split_was_never_read(self):
        never = report()["never_read"]
        assert "data/mllmu_hier_confirm100" in never
        assert "data/reports/mllmu_confirm100_final_analysis.json" in never


# ──────────────────────────────────────────────────────────────────────
class TestTheThirdDefectIsDisclosedRatherThanPatched:
    """Neither the Stage-1c nor the Stage-2 freeze can be attributed to a commit.

    Found while fixing this script's own ``git_commit`` field, which called
    ``_git(REPO_ROOT)`` and died on ``AttributeError``.  It is not patched: both
    fields are volatile, so no verdict depends on them, and each freeze script
    binds itself, so editing either would make the selector refuse to run.
    """

    def test_the_helper_takes_the_repo_root_first(self):
        out = rep.the_freeze_records_no_git_provenance()["the_helper"]
        assert out["first_parameter_is_the_repo_root_not_a_subcommand"] is True
        assert out["signature"].startswith("(repo_root")

    def test_called_as_the_freeze_calls_it_the_helper_says_nothing(self):
        out = rep.the_freeze_records_no_git_provenance()["the_helper"]
        assert out["called_as_the_freeze_calls_it"] is None
        #: The helper is not broken — the call is.  Without this half the finding
        #: would read as "git was unavailable" rather than as a mis-bound argument.
        assert out["called_correctly_it_returns_a_commit"] is True
        assert re.fullmatch(r"[0-9a-f]{40}",
                            fis2._git(REPO_ROOT, "rev-parse", "HEAD"))

    def test_the_mis_bound_calls_are_parsed_out_of_both_freeze_scripts(self):
        out = rep.the_freeze_records_no_git_provenance()
        #: Sorted per file, for the ast.walk reason given above.
        got = {k: sorted(v, key=lambda c: c["line"])
               for k, v in out["the_mis_bound_calls"].items()}
        assert got == {
            "scripts/freeze_iter12_route_stratification.py": [
                {"line": 294, "first_argument": "'rev-parse'",
                 "binds_a_subcommand_to_repo_root": True,
                 "the_git_args_it_therefore_passed": ["'HEAD'"]},
                {"line": 295, "first_argument": "'status'",
                 "binds_a_subcommand_to_repo_root": True,
                 "the_git_args_it_therefore_passed": ["'--porcelain'"]},
            ],
            "scripts/freeze_iter12_stage2.py": [
                {"line": 305, "first_argument": "'rev-parse'",
                 "binds_a_subcommand_to_repo_root": True,
                 "the_git_args_it_therefore_passed": ["'HEAD'"]},
                {"line": 306, "first_argument": "'status'",
                 "binds_a_subcommand_to_repo_root": True,
                 "the_git_args_it_therefore_passed": ["'--porcelain'"]},
            ],
        }

    def test_exactly_two_filed_freezes_record_no_commit_and_two_do(self):
        out = rep.the_freeze_records_no_git_provenance()
        silent = sorted(k for k, v in out["per_freeze"].items()
                        if v["git_commit"] is None)
        assert silent == [
            "data/reports/mllmu_iter12_route_stratification_freeze.json",
            "data/reports/mllmu_iter12_stage2_freeze.json",
        ]
        assert out["the_freezes_that_recorded_it_properly"] == [
            "data/reports/mllmu_iter12_seed_replication_freeze.json",
            "data/reports/mllmu_iter12_selection_protocol_freeze.json",
            "data/reports/mllmu_pilot100_confirmation_freeze.json",
        ]
        #: The two that worked record a porcelain listing; the two that did not
        #: record a bool, which is how the same mistake shows up in the artifact.
        for rel in silent:
            assert out["per_freeze"][rel][
                "git_dirty_is_a_bool_not_a_porcelain_listing"] is True

    def test_no_verdict_depends_on_the_two_fields(self):
        out = rep.the_freeze_records_no_git_provenance()
        assert "git_commit" in out["not_decision_bearing_because"]["volatile_fields"]
        assert "git_dirty" in out["not_decision_bearing_because"]["volatile_fields"]
        assert out["not_decision_bearing_because"][
            "verify_freeze_reads_neither_field"] is True
        #: Read out of the function's own source rather than out of a claim about
        #: it, and the source is the Stage-2 freeze's, which is the one that runs.
        import inspect
        src = inspect.getsource(fis2.verify_freeze)
        assert "git_commit" not in src and "git_dirty" not in src

    def test_each_broken_freeze_script_binds_itself_so_it_cannot_be_edited(self):
        out = rep.the_freeze_records_no_git_provenance()
        binds = out["cannot_be_fixed_in_place_because"]["each_freeze_script_binds_itself"]
        assert binds == {"scripts/freeze_iter12_route_stratification.py": True,
                         "scripts/freeze_iter12_stage2.py": True}
        hashes = freeze_doc()["hashes"]["protocol_paths"]
        assert "scripts/freeze_iter12_stage2.py" in hashes

    def test_the_reconstruction_shows_the_tree_was_dirty_whatever_git_dirty_says(
            self):
        out = rep.the_freeze_records_no_git_provenance()
        for rel, entry in out["per_freeze"].items():
            if entry["git_commit"] is not None:
                continue
            #: Read out of the FILED freeze, not out of the proof's copy of it, so
            #: the branch is chosen by git and the artifact rather than by anything
            #: the function under test produced.
            frozen_at = json.loads((REPO_ROOT / rel).read_text())["frozen_at_utc"]
            if not _history_can_reconstruct(frozen_at):
                #: This checkout does not contain the commit that preceded the
                #: freeze, so there is nothing to reconstruct FROM.  Asserted
                #: rather than skipped, and the assertion is about the shape: the
                #: disclosure must carry no field a reader could mistake for a
                #: measurement, which is what makes it a disclosure.
                assert entry["reconstructed_head"] is None, rel
                assert entry["so_no_bound_path_was_compared"] is True, rel
                assert "the_tree_was_therefore_dirty" not in entry, rel
                continue
            assert re.fullmatch(r"[0-9a-f]{40}", entry["reconstructed_head"]), rel
            assert entry["the_tree_was_therefore_dirty"] is True, rel
            assert entry["which_contradicts_git_dirty"] is False, rel
            #: The reconstruction is checked against the freeze's own hashes rather
            #: than offered, and every path it cannot account for is explained: a
            #: differing one is a documented amendment, an absent one is a file the
            #: freeze introduced and the commit that filed it contains.
            accounted = entry["bound_paths_matching_that_commit"]
            unaccounted = (len(entry["bound_paths_differing_there"])
                           + len(entry["bound_paths_absent_there"]))
            assert accounted + unaccounted > 0, rel
            if entry["bound_paths_differing_there"]:
                assert entry[
                    "every_differing_path_is_a_documented_amendment"] is True, rel
            else:
                #: Tri-state: "nothing differs" is not the same claim as "every
                #: difference is documented", and reporting False here would read
                #: as a failed cross-check where there was nothing to cross-check.
                assert entry[
                    "every_differing_path_is_a_documented_amendment"] is None, rel
            if entry["bound_paths_absent_there"]:
                assert entry["every_absent_path_exists_there"] is True, rel

    def test_a_checkout_without_the_history_says_so_rather_than_measuring_nothing(
            self):
        """The disclosure branch, driven deterministically rather than hoped for.

        CI clones with ``fetch-depth: 0``, so a shallow checkout is not something
        the suite would otherwise ever see -- and a branch no test reaches is a
        branch a mutation can rewrite unnoticed.  A timestamp before the first
        commit produces the same "no commit precedes this" condition in ANY clone,
        full or shallow, so the branch is exercised everywhere.

        What it must not do is return the successful shape with empty lists.
        ``the_tree_was_therefore_dirty`` is ``bool(differing or absent)``, so
        measuring nothing would compute to False and file as a measurement that
        found the tree clean -- the opposite of the finding this disclosure exists
        to make.
        """
        doc = freeze_doc()
        doc["frozen_at_utc"] = "1970-01-01T00:00:00+00:00"
        entry = rep._reconstruct_the_commit(fis2.OUT_REPORT, doc)
        assert entry["reconstructed_head"] is None
        assert entry["so_no_bound_path_was_compared"] is True
        assert "1970-01-01" in entry["history_insufficient_in_this_clone"]
        #: It names what it would have measured, so the absence is informative
        #: rather than a hole.
        assert "hashes.protocol_paths" in entry[
            "what_would_be_measured_with_history"]
        #: And it carries none of the fields whose values would read as findings.
        for measured in ("the_tree_was_therefore_dirty",
                         "bound_paths_matching_that_commit",
                         "bound_paths_differing_there",
                         "bound_paths_absent_there",
                         "which_contradicts_git_dirty"):
            assert measured not in entry, measured

    def test_the_record_does_not_repeat_the_mistake_it_discloses(self):
        r = record()
        assert re.fullmatch(r"[0-9a-f]{40}", r["git_commit"]), r["git_commit"]
        #: A porcelain listing, not a bool — the shape the two working freezes use.
        assert isinstance(r["git_dirty_at_filing"], str)

    def test_it_refuses_if_the_defect_is_ever_fixed(self, tmp_path, monkeypatch):
        #: Then the disclosure would be describing a file that no longer exists in
        #: that state, and --check-only has to say so rather than pass silently.
        #: Driven on a stand-in repository holding one freeze that DOES record a
        #: commit: the refusal fires before any reconstruction, so no git history
        #: is needed and nothing here depends on this clone's past.
        reports = tmp_path / "data" / "reports"
        reports.mkdir(parents=True)
        (reports / Path(fis2.OUT_REPORT).name).write_text(json.dumps(
            {"git_commit": "0" * 40, "git_dirty": "",
             "frozen_at_utc": "2026-01-01T00:00:00+00:00"}))
        monkeypatch.setattr(rep, "REPO_ROOT", tmp_path)
        with pytest.raises(SystemExit) as exc:
            rep.the_freeze_records_no_git_provenance()
        assert "expected" in str(exc.value)
        assert fis2.OUT_REPORT in str(exc.value)


# ──────────────────────────────────────────────────────────────────────
class TestTheRepairDisclosesItself:
    def test_the_record_says_the_preregistered_selector_did_not_finish(self):
        r = record()
        claim = r["what_still_cannot_be_claimed"]
        assert "did not" in claim
        assert r["crash_reproduced_first"]["reproduced"] is True
        assert "missing 1 required positional argument" in \
            r["crash_reproduced_first"]["exception"]

    def test_it_says_plainly_that_the_second_patch_changes_the_outcome(self):
        #: The one disclosure that must not be left to inference.  A reader who
        #: never opens the diffs has to be told that a script written after the
        #: freeze decided which name lands in `selected`.
        r = record()
        patches = {p["for_bug"]: p for p in r["the_patches"]}
        assert patches[1]["changes_the_outcome"] is False
        assert patches[2]["changes_the_outcome"] is True
        assert "selected" in patches[2]["what_it_changes"]
        assert r["the_bugs"][0]["decision_bearing"] is False
        assert r["the_bugs"][1]["decision_bearing"] is True
        assert "judgement" in r["what_still_cannot_be_claimed"]
        assert "judgement" in r["what_this_is"] or \
            "changes which candidate" in r["what_this_is"]

    def test_the_crashed_run_neither_wrote_nor_changed_the_report(self):
        #: Hash AND mtime, because the report has no timestamp: a hash-only check
        #: cannot tell "never reached the write" from "rewrote identical bytes".
        crash = record()["crash_reproduced_first"]
        assert crash["the_crashed_run_wrote_or_changed_the_report"] is False
        assert crash["selection_report_sha256_before"] == \
            crash["selection_report_sha256_after"]
        assert crash["selection_report_mtime_ns_before"] == \
            crash["selection_report_mtime_ns_after"]
        #: It crashed at the FIRST defect, not the second: the unpatched run never
        #: reached build_report, so the crash proof is about the missing argument
        #: and says nothing about the anchor.
        assert _crash_line_number(crash["crash_line"]) in BROKEN_SITES, \
            crash["crash_line"]
        assert _crash_line_number(crash["crash_line"]) != CRASH_LINE_BUG_2

    def test_the_second_crash_is_copied_out_of_a_log_that_is_not_committed(self):
        second = record()["the_second_crash_as_it_happened"]
        assert second is not None
        assert second["crash_line_in_the_selector"] == CRASH_LINE_BUG_2
        assert second["traceback"][-1] == "KeyError: 'MG'"
        #: The point of recording where the write is: the crash is BEFORE it, so
        #: the run that died filed nothing and the report on disk came from the
        #: patched run rather than from the one that crashed.
        assert second["the_line_that_writes_the_report"] > CRASH_LINE_BUG_2
        assert second["so_the_crashed_run_filed_nothing"] is True
        assert second["found_in"].startswith("outputs/lanes/")
        #: Re-derived rather than re-read: the log is committed, so the block in the
        #: record has to be what the function still produces from it.
        assert second == rep.the_second_crash_as_it_happened()

    def test_a_missing_chain_log_is_refused_rather_than_reported_as_nothing(
            self, monkeypatch):
        #: Returning None here is how the traceback parser once reported
        #: `so_the_crashed_run_filed_nothing: false` — the opposite of the truth —
        #: because "could not parse" was indistinguishable from "the log says
        #: otherwise".  Both refusals are driven, one in each of the two tests here,
        #: because a refusal nobody exercises is only a comment.
        monkeypatch.setattr(rep, "SECOND_CRASH_LOG", "outputs/lanes/not-written.log")
        with pytest.raises(SystemExit) as exc:
            rep.the_second_crash_as_it_happened()
        assert "is absent" in str(exc.value)

    def test_a_chain_log_without_the_crash_is_refused_too(self, tmp_path,
                                                           monkeypatch):
        log = tmp_path / "chain.log"
        log.write_text("all lanes finished\nscore: OK\n")
        monkeypatch.setattr(rep, "SECOND_CRASH_LOG", str(log))
        with pytest.raises(SystemExit) as exc:
            rep.the_second_crash_as_it_happened()
        assert "no line starting" in str(exc.value)

    def test_every_recorded_invocation_succeeded_and_stayed_on_the_two_sites(self):
        r = record()
        ev = r["invocations"]
        assert len(ev) >= 8, "seven generation lanes and at least one scoring run"
        assert all(e["failure"] is None for e in ev)
        expected = {f"{Path(rep.SELECTOR_REL).name}:{ln}" for ln in BROKEN_SITES}
        served = {s for e in ev for s in e["served_call_sites"]}
        assert served <= expected, served
        assert served == expected, "both sites were served at least once"
        #: The callee's own internal call — which IS on the decision path, because
        #: it produces the eight numbers — always passes four arguments and is
        #: served nothing.
        assert sum(e["passthrough_calls_from_the_callee"] for e in ev) > 0

    def test_a_scoring_run_reproduced_the_filed_report_byte_identically(self):
        r = record()
        proof = r["the_patches"][1]["proof_the_anchor_still_anchors_the_floor"]
        assert proof["checked_in_the_report_it_wrote"] is not None
        assert proof["checked_in_the_report_it_wrote"][
            "this_run_reproduced_the_filed_report_byte_identically"] is True
        assert proof["reproduced_by_a_later_run"][
            "runs_that_rewrote_the_report_byte_identically"] >= 1
        #: Why an identical hash is a reproduction and not a no-op, measured.
        assert proof["checked_in_the_report_it_wrote"][
            "the_report_carries_no_timestamp_of_its_own"] is True
        assert rep._keys_that_record_a_time(report()) == []

    def test_every_way_a_run_can_touch_the_report_is_told_apart(self):
        #: The gate that decides whether the anchor proof is filed or discarded,
        #: driven through all of its cases rather than only the one a real run
        #: happens to produce.  Two of these never occur and are here because a
        #: function that is only ever exercised by one input is not tested.
        f = rep.how_the_run_touched_the_report
        same, other, m1, m2 = "a" * 64, "b" * 64, 100, 200

        nothing = f(same, same, m1, m1)
        assert nothing["never_reached_the_write"] is True
        assert nothing["wrote_or_changed"] is False
        assert nothing["reproduced_byte_identically"] is False

        repro = f(same, same, m1, m2)
        assert repro["reproduced_byte_identically"] is True
        assert repro["wrote_or_changed"] is True
        assert repro["changed"] is False

        moved = f(same, other, m1, m2)
        assert moved["changed"] is True
        assert moved["reproduced_byte_identically"] is False
        assert moved["wrote_or_changed"] is True

        appeared = f(None, other, None, m2)
        assert appeared["the_file_appeared_in_this_run"] is True
        assert appeared["wrote_or_changed"] is True

        #: A hash that moves while the mtime does not is impossible for a file this
        #: process wrote, and is reported as changed rather than as a reproduction.
        impossible = f(same, other, m1, m1)
        assert impossible["changed"] is True
        assert impossible["reproduced_byte_identically"] is False
        assert impossible["never_reached_the_write"] is False

    def test_the_record_embeds_the_evidence_log_and_not_a_copy_of_itself(self):
        #: --check-only reads the log for every other claim, so without a prefix
        #: check the invocations embedded in the record would be checked by nothing
        #: and could be edited to hide a failure or to claim a site never served.
        #: Mutating them in the filed record is what makes --check-only refuse.
        r = record()
        log = [json.loads(ln) for ln in
               (REPO_ROOT / rep.EVIDENCE_LOG).read_text().splitlines() if ln.strip()]
        assert log[:len(r["invocations"])] == r["invocations"]

    def test_the_key_tokeniser_does_not_match_substrings(self):
        #: "candidates" contains "date" and "runtime" contains "time".  A substring
        #: test reported a timestamp in a report that has none and refused a run
        #: that had just reproduced it perfectly.
        assert rep._keys_that_record_a_time({"candidates": {}, "runtime_s": 1}) == []
        assert rep._keys_that_record_a_time({"a": {"frozen_at_utc": "x"}}) == \
            ["a.frozen_at_utc"]

    def test_the_anchor_was_dropped_and_only_the_anchor(self):
        r = record()
        proof = r["the_patches"][1]["proof_the_anchor_still_anchors_the_floor"]
        assert proof["rows_dropped_across_all_invocations"] == [rep.ANCHOR]
        scoring = [e for e in r["invocations"]
                   if e.get("build_report_patched_calls") == 1]
        assert scoring, "no scoring run was recorded"
        assert all(e["rows_dropped_from_scoring"] == [rep.ANCHOR] for e in scoring)
        assert all(e["rows_scored"] == len(report()["candidates"]) for e in scoring)

    def test_the_evidence_log_spans_two_versions_of_the_repair_and_says_so(self):
        #: The first seven generation lanes ran before the anchor patch existed, so
        #: their rows have no anchor fields at all.  That is a fact about the log
        #: rather than a gap in it — the second defect could not be found until the
        #: first was out of the way — but a reader who indexed the fields directly
        #: would hit a KeyError and conclude the log was corrupt.
        ev = record()["invocations"]
        with_anchor = [e for e in ev if "build_report_patched_calls" in e]
        without = [e for e in ev if "build_report_patched_calls" not in e]
        assert without and with_anchor
        #: The ones without are all generation lanes, and all of them predate the
        #: first one that has it: the log is append-only and in time order.
        assert all("--generate-only" in e["forwarded_argv"] for e in without)
        last_without = max(e["recorded_utc"] for e in without)
        first_with = min(e["recorded_utc"] for e in with_anchor)
        assert last_without < first_with, (last_without, first_with)
        #: Every row, old schema or new, records the fields the safety argument
        #: rests on — which sites were served and whether the run failed.
        for e in ev:
            assert "served_call_sites" in e and "failure" in e

    def test_earlier_filings_are_listed_with_a_measured_difference(self):
        r = record()
        block = r["this_record_supersedes_earlier_filings_of_itself"]
        filings = block["in_the_order_they_were_filed"]
        assert len(filings) == len(rep.SUPERSEDED_RECORDS)
        for entry in filings:
            path = REPO_ROOT / entry["preserved_at"]
            assert path.exists(), entry["preserved_at"]
            assert rep._stamp(path)[0] == entry["preserved_sha256"]
            assert entry["preserved_bytes"] == path.stat().st_size
            assert entry["why_it_was_superseded"]
            assert entry["what_did_not_change"]["the_two_bugs_and_the_arity_audit"]
        assert filings[0]["measured_difference"]["its_anchor_proof_was_null"] is True
        assert filings[1]["measured_difference"]["its_anchor_proof_was_null"] is False

    def test_all_five_artifacts_are_committed_and_not_merely_present(self):
        tracked = (REPO_ROOT / ".gitignore").read_text()
        paths = [REPORT, REPAIR] + [REPO_ROOT / e["path"]
                                    for e in rep.SUPERSEDED_RECORDS]
        for p in paths:
            assert p.exists(), p
            json.loads(p.read_text())
            rel = str(p.relative_to(REPO_ROOT))
            assert f"!{rel}" in tracked, f"{rel} is gitignored, so never committed"
        required = (REPO_ROOT / ".github" / "workflows" / "tests.yml").read_text()
        for p in paths:
            rel = str(p.relative_to(REPO_ROOT))
            assert f'"{rel}"' in required, f"{rel} is absent from the required list"

    def test_check_only_re_derives_the_record_and_passes(self):
        #: Torch-free and offline apart from git, so this is what CI actually runs.
        p = subprocess.run(
            (sys.executable, "scripts/repair_iter12_stage2_selection.py",
             "--check-only"),
            cwd=REPO_ROOT, capture_output=True, text=True, check=False,
            env=repo_env())
        assert p.returncode == 0, p.stdout + p.stderr
        assert "no frozen file edited" in p.stdout


# ──────────────────────────────────────────────────────────────────────
class TestTheRepairScriptIsNotItselfABackDoor:
    def test_it_is_not_a_protocol_path_and_not_a_bound_dependency(self):
        rel = "scripts/repair_iter12_stage2_selection.py"
        hashes = freeze_doc()["hashes"]
        assert rel not in hashes["protocol_paths"]
        assert rel not in hashes["depends_on_unchanged"]
        assert rel not in hashes["data_paths"]

    def test_it_writes_its_own_record_and_its_evidence_log_and_nothing_else(self):
        src = (REPO_ROOT / rep.__file__).read_text()
        tree = ast.parse(src)
        opened = []
        for n in ast.walk(tree):
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) \
                    and n.func.id == "open" and len(n.args) >= 2:
                mode = n.args[1]
                if isinstance(mode, ast.Constant) and mode.value in ("w", "a"):
                    opened.append(n.lineno)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) \
                    and n.func.attr in ("write_text", "write_bytes"):
                opened.append(n.lineno)
        assert len(opened) == 2, opened
        lines = src.splitlines()
        #: One write of the record, one append of the evidence row — and neither is
        #: the report, the freeze, or anything the freeze binds.
        assert any("record_path_" in lines[ln - 1] for ln in opened)
        assert any("path" in lines[ln - 1] for ln in opened)
        assert "OUT_REPORT" not in "\n".join(lines[ln - 1] for ln in opened)

    def test_it_never_invokes_a_freeze_writer_or_asks_for_an_amendment(self):
        #: The repair's licence is that it edits nothing frozen.  Writing the freeze
        #: document, or asking for it to be amended, would be the one way to make a
        #: crashed selector run by changing the criterion instead of the run -- and
        #: ``--refreeze``/``--reason`` are exactly the flags that do it.
        src = (REPO_ROOT / rep.__file__).read_text()
        for forbidden in ("fis2.main(", "build_freeze(", "--refreeze", "--reason"):
            assert forbidden not in src, forbidden
        #: verify_freeze is called, and only ever for its verdict: it appears as a
        #: call whose result is tested, never as something written to.
        assert "fis2.verify_freeze(REPO_ROOT)" in src

    def test_both_patches_are_restored_in_a_finally(self):
        #: The whole claim that nothing was edited rests on this: two module
        #: attributes are rebound, and one ``finally`` puts both back.
        tree = ast.parse((REPO_ROOT / rep.__file__).read_text())
        main = next(n for n in tree.body
                    if isinstance(n, ast.FunctionDef) and n.name == "main")
        tries = [n for n in ast.walk(main) if isinstance(n, ast.Try)
                 and n.finalbody]
        restored = set()
        for t in tries:
            for stmt in t.finalbody:
                if isinstance(stmt, ast.Assign):
                    for target in stmt.targets:
                        if isinstance(target, ast.Attribute):
                            restored.add(target.attr)
            body = "\n".join(ast.unparse(s) for s in t.body)
            if "sis2.main()" in body:
                break
        else:
            pytest.fail("no try/finally wraps the selector's main()")
        assert {"pooled_all_routes", "build_report"} <= restored, restored

    def test_the_patched_names_are_rebound_and_not_the_functions_edited(self):
        #: Monkeypatching a module attribute leaves the def'd function object
        #: alone, which is what "no frozen file was edited" has to mean at
        #: runtime as well as on disk.
        src = (REPO_ROOT / rep.__file__).read_text()
        assert "rt.pooled_all_routes = " in src
        assert "sis2.build_report = " in src
        assert "def pooled_all_routes" not in src
        assert "def build_report" not in src


# ──────────────────────────────────────────────────────────────────────
class TestTheLaneRoutesThroughTheRepair:
    """The chain must not be able to reach the frozen selector directly.

    If it could, a relaunch would crash the same way at the same line and burn
    seven lane attempts doing it — ``wait_for_gpu.sh`` treats any non-zero exit as
    transient and re-queues up to eight times.
    """

    def _phase(self, start: str, end: str) -> str:
        text = LANE_SCRIPT.read_text()
        i = text.index(start)
        j = text.index(end, i)
        return text[i:j]

    def test_the_gen_phase_goes_through_the_repair_generate_only(self):
        gen = self._phase("# ── gen", "# ── score")
        assert "repair_iter12_stage2_selection.py" in gen
        assert "--generate-only" in gen
        assert "--skip-crash-proof" in gen
        assert "select_iter12_stage2.py" not in gen

    def test_the_score_phase_goes_through_the_repair_and_files_the_record(self):
        score = self._phase("# ── score", "chain complete")
        assert "repair_iter12_stage2_selection.py" in score
        assert "--file-record" in score
        #: The scoring run is the one that reproduces the crash unpatched, so it
        #: must NOT skip the proof; the seven gen lanes must, because each would
        #: otherwise re-run the whole selector first.
        assert "--skip-crash-proof" not in score
        assert "select_iter12_stage2.py" not in score

    def test_no_phase_invokes_the_frozen_selector_directly(self):
        text = LANE_SCRIPT.read_text()
        #: Mentioned in prose is fine; invoked is not.
        invocations = [ln for ln in text.splitlines()
                       if "select_iter12_stage2.py" in ln
                       and not ln.lstrip().startswith("#")]
        assert invocations == [], invocations

    def test_the_repair_is_not_what_gates_the_grid_that_gating_is_still_the_file(
            self):
        #: The check phase keys on the adapter FILE over the whole grid and exits
        #: non-zero, which is what stops a relaunch after an interrupted row.  The
        #: repair is downstream of it and must not have replaced it.
        check = self._phase("# ── check", "# ── gen")
        assert '-f "$STAGE2_CKPT/$cid/adapters/adapter_model.safetensors"' in check
        assert "exit 4" in check
        assert "repair_iter12_stage2_selection.py" not in check
