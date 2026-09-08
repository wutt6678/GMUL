"""Iteration 11C-5 — the confirmation scorer and its analysis, tested on CPU.

Nothing here touches a GPU and nothing here writes to the real confirmation
dataset.  The predictions the analysis is exercised on are FABRICATED: for each
query a raw output is chosen that the frozen scorer classifies into a known
category, so a state can be made to have any TGA/FILR/retention profile wanted
without decoding a single token.

Why a symlink farm rather than ``tmp_path``
-------------------------------------------
``PredictionFingerprint.build`` hashes the adapter bytes, the dataset artifacts
and the ten fingerprinted modules, all resolved relative to a repo root.  A
bare temp directory has none of them, and copying them is 300 MB of adapters.
So the farm symlinks the repository and makes exactly ONE directory real:
``data/mllmu_hier_confirm100/predictions/``, which is where the fabricated
parquets go.  The real dataset's ``predictions/`` stays absent throughout, and
a test asserting that is included below, because the whole point of stage 5 not
having run yet is that it has not run.

What runs where
---------------
Fabricating a prediction file that VERIFIES needs the real adapter bytes, and
the adapters are gitignored.  So the tests that fabricate a state depend on the
``adapters`` fixture below and SKIP on a checkout without them — CI, and any
fresh clone.  The tests that do not fabricate anything (the source-level
boundary checks, the protocol refusals, the completeness gate over an empty
farm, and the assertion that the real repository is unscored) run everywhere.
That split is deliberate and it is the repository's existing convention, but it
is a real coverage boundary and is stated here rather than left to be inferred
from a skip count.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "src"))

import analyze_confirmation_split as acs  # noqa: E402
import evaluate_confirmation_split as ecs  # noqa: E402

from granunlearn.evaluation.prediction_provenance import (  # noqa: E402
    write_sidecar,
)
from granunlearn.evaluation.reference_eval import (  # noqa: E402
    load_associations_parquet,
)
from granunlearn.evaluation.scoring import score_query  # noqa: E402

REAL_PREDICTIONS = REPO_ROOT / "data" / "mllmu_hier_confirm100" / "predictions"
REAL_ANALYSIS = REPO_ROOT / "data" / "reports" / \
    "mllmu_confirm100_final_analysis.json"


def _build_farm(root: Path) -> Path:
    """A repo that is the real one by symlink, except for ``predictions/``."""
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)
    for entry in REPO_ROOT.iterdir():
        (root / entry.name).symlink_to(entry)
    #: ``data/`` and the confirmation dataset have to be real directories, or
    #: the one path that must not be shared with the real repository — its
    #: ``predictions/`` — would be a symlink into it.
    (root / "data").unlink()
    (root / "data").mkdir()
    for entry in (REPO_ROOT / "data").iterdir():
        (root / "data" / entry.name).symlink_to(entry)
    ds = root / "data" / "mllmu_hier_confirm100"
    ds.unlink()
    ds.mkdir()
    for entry in (REPO_ROOT / "data" / "mllmu_hier_confirm100").iterdir():
        (ds / entry.name).symlink_to(entry)
    (ds / "predictions").mkdir()
    assert not (ds / "predictions").is_symlink()
    return root


@pytest.fixture(scope="module")
def farm(tmp_path_factory) -> Path:
    root = _build_farm(tmp_path_factory.mktemp("farm") / "repo")
    yield root


@pytest.fixture(scope="module")
def protocol(farm) -> dict:
    freeze = ecs.load_frozen_protocol(farm)
    queries = ecs.confirmation_queries(farm, freeze)
    associations = load_associations_parquet(
        farm / ecs.CONFIRM_DATASET_DIR / "associations.parquet")
    return {
        "freeze": freeze,
        "queries": queries,
        "associations": associations,
        "by_assoc": {a.association_id: a for a in associations},
        "config": ecs.frozen_generation_config(freeze),
        "states": ecs.scored_states(freeze),
        "order": ecs.generation_order_sha256(queries),
        "model_id": freeze["checkpoints"]["B3"]["recipe"]["model_id"],
    }


def _absent_adapters(freeze: dict) -> list[str]:
    """The scored states whose gitignored adapter directory is not on disk."""
    out = []
    for state in ecs.scored_states(freeze):
        rel = ((freeze.get("checkpoints") or {}).get(state) or {}).get(
            "adapter_dir")
        if not rel or not (REPO_ROOT / rel).exists():
            out.append(f"{state} -> {rel}")
    return out


@pytest.fixture(scope="module")
def adapters(protocol) -> dict:
    """SKIP, not error, when the adapter bytes are absent.

    ``PredictionFingerprint.build`` hashes the adapter files, so fabricating a
    state that VERIFIES needs the real adapter directory to exist — a made-up
    one would not match the contract hash the freeze pins, and patching the pin
    would replace the committed freeze with a fabricated one, which is the thing
    most of these tests exist to check.

    The adapters are gitignored, so a fresh checkout and CI both lack them.  The
    repository's convention for a test whose inputs are gitignored bytes is to
    skip and say why; erroring instead would turn CI red on a checkout that is
    missing nothing it was supposed to have, and a red suite gets overridden.
    These tests therefore run on the artifact machine — which is the machine
    that will run the three GPU passes, so the scorer is exercised where it
    matters.  Everything that does NOT need adapter bytes (the source-level
    checks, the protocol refusals, the completeness gate over an empty farm) is
    above this fixture and runs everywhere.
    """
    absent = _absent_adapters(protocol["freeze"])
    if absent:
        pytest.skip(
            f"{len(absent)} adapter directory(ies) absent and gitignored: "
            f"{absent}; fabricating a verified prediction file needs the real "
            "bytes, so this runs only where the adapters were trained")
    return protocol


def _raw_for(query, assoc, kind: str) -> str:
    """A raw output the frozen scorer classifies into ``kind``.

    Chosen from the association's own hierarchy, so the classification is the
    scorer's and not a guess about it: the target level's value is
    ``correct_at_target``, the finest level's value is ``under_forgetting``
    (which IS the FILR numerator), the coarsest is ``over_forgetting``, and for
    a retain probe the finest value is the retained fact still being produced.
    """
    levels = {lv.level: lv.value for lv in assoc.levels}
    target = query.unlearning_target_level
    if target is None:
        target = assoc.target_level
    return str(levels[{"tga": target, "filr": min(levels),
                       "over": max(levels)}[kind]])


def _write_state(farm: Path, protocol: dict, state: str, kind: str,
                 order: str | None = None, config: dict | None = None) -> Path:
    """Fabricate one state's predictions and seal them with a real sidecar."""
    preds = [score_query(q, protocol["by_assoc"][q.association_id],
                         _raw_for(q, protocol["by_assoc"][q.association_id],
                                  kind),
                         experiment_id=ecs.EXPERIMENT_ID, checkpoint_id=state)
             for q in protocol["queries"]]
    adapter_dir = ecs.adapter_for(state, protocol["freeze"], farm)
    cfg = config if config is not None else protocol["config"]
    fingerprint = ecs.expected_fingerprint(
        state, adapter_dir, farm, protocol["model_id"], cfg, len(preds))
    path = ecs.predictions_path(farm, state)
    path.parent.mkdir(parents=True, exist_ok=True)
    import pandas as pd
    pd.DataFrame([json.loads(p.model_dump_json()) for p in preds]
                 ).to_parquet(path, index=False)
    write_sidecar(path, fingerprint)
    (path.parent / f"predictions_{state}.generation_order.json").write_text(
        json.dumps({
            "state": state,
            "experiment_id": ecs.EXPERIMENT_ID,
            "num_queries": len(preds),
            "generation_order_sha256": order or protocol["order"],
            "generation_config": cfg,
        }, indent=1, sort_keys=True))
    return path


@pytest.fixture(scope="module")
def scored(farm, protocol, adapters) -> Path:
    """The farm with all three states present.

    B0 is the no-op, so it still leaks the fine value; B3 is the
    granularity-aware unlearning, so it answers at the target level; MG is the
    oracle and answers at the target level too.  That makes B3 − B0 maximal in
    both directions the primary claims point — TGA up, FILR down — and B3 − MG
    zero, which is the shape the real result is expected to have and the one
    the descriptive block has to decline to call equivalence.
    """
    for state, kind in (("B3", "tga"), ("B0", "filr"), ("MG", "tga")):
        _write_state(farm, protocol, state, kind)
    return farm


class TestNothingHasBeenScoredInTheRealRepository:
    """The precondition every other test in this file depends on.

    Stage 5 has not run.  If it had, the confirmation would already be scored
    and these tests would be exercising a protocol whose one-shot property was
    spent — so this is asserted first and against the REAL paths, not the farm.
    """

    def test_the_real_dataset_has_no_predictions_directory(self):
        assert not REAL_PREDICTIONS.exists(), \
            f"{REAL_PREDICTIONS} exists: stage 5 has already run"

    def test_no_analysis_report_has_been_written(self):
        assert not REAL_ANALYSIS.exists()

    def test_the_farm_does_not_write_through_to_the_real_dataset(self, farm):
        """The farm shares ``data/mllmu_hier_confirm100`` by symlink for every
        file EXCEPT ``predictions/``, so a fabricated parquet cannot land in
        the real dataset.  Asserted rather than assumed, because the failure
        mode is silent and permanent."""
        ds = farm / "data" / "mllmu_hier_confirm100"
        assert (ds / "queries.parquet").is_symlink()
        assert not (ds / "predictions").is_symlink()
        assert not REAL_PREDICTIONS.exists()


class TestTheEvaluatorReadsTheProtocolInsteadOfDefaultingIt:
    def test_the_three_frozen_states_and_no_others(self, protocol):
        assert protocol["states"] == ("B3", "B0", "MG")

    def test_the_generation_config_is_the_frozen_one(self, protocol):
        assert protocol["config"] == {
            "batch_size": 8, "image_batch_size": 8, "max_new_tokens": 96,
            "do_sample": False, "max_image_pixels": 147456, "max_length": 1536}

    def test_a_sampled_decode_refuses(self, protocol):
        """Greedy is what makes a prediction reproducible, and a prediction
        that cannot be reproduced cannot be verified and reused — which is the
        entire resumption contract."""
        freeze = dict(protocol["freeze"])
        freeze["generation"] = dict(freeze["generation"], do_sample=True)
        with pytest.raises(SystemExit, match="greedy"):
            ecs.frozen_generation_config(freeze)

    def test_a_config_that_drifts_from_the_module_defaults_refuses(self,
                                                                  protocol):
        """``max_new_tokens`` truncates a decoded answer, so a frozen value the
        code no longer produces changes what is being measured."""
        freeze = dict(protocol["freeze"])
        freeze["generation"] = dict(freeze["generation"], max_new_tokens=64)
        with pytest.raises(SystemExit, match="disagrees with"):
            ecs.frozen_generation_config(freeze)

    def test_an_unstated_generation_field_refuses(self, protocol):
        freeze = dict(protocol["freeze"])
        gen = dict(freeze["generation"])
        gen.pop("image_batch_size")
        freeze["generation"] = gen
        with pytest.raises(SystemExit, match="image_batch_size"):
            ecs.frozen_generation_config(freeze)

    def test_a_fourth_scored_state_refuses(self, protocol):
        freeze = dict(protocol["freeze"])
        freeze["states"] = dict(freeze["states"],
                                scored=["B3", "B0", "MG", "MF"])
        with pytest.raises(SystemExit, match="written for"):
            ecs.scored_states(freeze)

    def test_a_state_that_is_not_recorded_as_excluded_refuses(self, protocol):
        """MF is excluded because B0 is a byte-identical copy of it.  A freeze
        that stopped saying so would leave this script unable to know whether
        generating MF was inside the protocol."""
        freeze = dict(protocol["freeze"])
        excluded = dict(freeze["states"]["excluded"])
        excluded.pop("MF")
        freeze["states"] = dict(freeze["states"], excluded=excluded)
        with pytest.raises(SystemExit, match="MF"):
            ecs.scored_states(freeze)

    def test_a_split_that_is_not_the_sealed_one_refuses(self, farm, protocol):
        freeze = dict(protocol["freeze"])
        size = dict(freeze["confirmation_size"])
        seal = dict(size["frozen_at_stage_3_before_any_scoring"],
                    confirmation_query_id_list_sha256="0" * 64)
        size["frozen_at_stage_3_before_any_scoring"] = seal
        freeze["confirmation_size"] = size
        with pytest.raises(SystemExit, match="not the split"):
            ecs.confirmation_queries(farm, freeze)

    def test_an_unsealed_split_refuses(self, farm, protocol):
        freeze = dict(protocol["freeze"])
        size = dict(freeze["confirmation_size"])
        size["frozen_at_stage_3_before_any_scoring"] = dict(
            size["frozen_at_stage_3_before_any_scoring"], sealed=False)
        freeze["confirmation_size"] = size
        with pytest.raises(SystemExit, match="not record the split as sealed"):
            ecs.confirmation_queries(farm, freeze)

    def test_a_moved_adapter_refuses(self, farm, protocol, adapters,
                                     monkeypatch):
        """The freeze pins the adapter contract's roll-up hash, so a checkpoint
        retrained between the freeze and this run is refused rather than scored
        and reported as the frozen state.

        Needs the real bytes: with the adapter directory absent the existence
        check fires first and reports a different, also correct, refusal — so
        without ``adapters`` this would test the wrong branch and pass for the
        wrong reason.
        """
        freeze = dict(protocol["freeze"])
        ckpts = dict(freeze["checkpoints"])
        ckpts["B3"] = dict(ckpts["B3"], adapter_contract=dict(
            ckpts["B3"]["adapter_contract"], sha256="0" * 64))
        freeze["checkpoints"] = ckpts
        with pytest.raises(SystemExit, match="has moved"):
            ecs.adapter_for("B3", freeze, farm)

    def test_a_freeze_carrying_refusals_refuses(self, farm, protocol):
        freeze = dict(protocol["freeze"], refusals=["something is wrong"])
        path = farm / ecs.FREEZE_REPORT
        original = path.read_bytes()
        try:
            #: The freeze is a symlink into the real report, so write through a
            #: replacement rather than into the committed artifact.
            path.unlink()
            path.write_text(json.dumps(freeze))
            with pytest.raises(SystemExit, match="carries 1 refusal"):
                ecs.load_frozen_protocol(farm)
        finally:
            path.unlink()
            path.write_bytes(original)
            assert path.read_bytes() == original


class TestTheEvaluatorComputesNoMetricAtAll:
    """Sealed-split invariant 6, enforced by what the module imports.

    A rule that says "do not look at partial results" is a rule about
    discipline.  A generation script with no aggregation code in it cannot be
    broken by indiscipline, so the check is on the source.
    """

    @staticmethod
    def _code_references(path: Path) -> set[str]:
        """Every module the file imports and every string literal in its CODE.

        Parsed rather than substring-searched, because both scripts explain in
        their docstrings exactly which modules and reports they avoid and why.
        A whole-file scan finds that prose and reports the explanation as the
        offence, so the test would fail on the documentation of the boundary
        instead of on a crossing of it.  Docstrings and comments are excluded;
        imports and real string literals are what can actually reach a file or
        a function.
        """
        import ast
        tree = ast.parse(path.read_text())
        found: set[str] = set()
        docstrings = set()
        for node in ast.walk(tree):
            if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef,
                                 ast.AsyncFunctionDef)):
                body = getattr(node, "body", None)
                if body and isinstance(body[0], ast.Expr) and isinstance(
                        body[0].value, ast.Constant) and isinstance(
                            body[0].value.value, str):
                    docstrings.add(id(body[0].value))
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                if isinstance(node, ast.Import):
                    found.update(a.name for a in node.names)
                else:
                    found.add(node.module or "")
                    found.update(a.name for a in node.names)
            elif (isinstance(node, ast.Constant)
                  and isinstance(node.value, str)
                  and id(node) not in docstrings):
                found.add(node.value)
        return found

    def test_the_gpu_script_imports_no_aggregation_module(self):
        refs = self._code_references(
            REPO_ROOT / "scripts" / "evaluate_confirmation_split.py")
        for banned in ("paired_ci", "hierarchy_metrics",
                       "granunlearn.evaluation.paired_ci",
                       "granunlearn.evaluation.hierarchy_metrics",
                       "compute_metrics", "paired_rate_diff_ci",
                       "one_sided_permutation_pvalue", "holm_family",
                       "row_flags", "compute_hierarchy_metrics",
                       "paired_metrics_report"):
            assert banned not in refs, banned
        #: And the scan is not vacuous: the analysis script DOES import them,
        #: so the same helper pointed one file over finds what this one must
        #: not contain.
        analysis_refs = self._code_references(
            REPO_ROOT / "scripts" / "analyze_confirmation_split.py")
        assert "row_flags" in analysis_refs
        assert "holm_family" in analysis_refs

    def test_it_names_neither_the_gate_nor_the_selection_report(self):
        """Invariants 4 and 5: the confirmation split never enters the
        reference-state gate and never enters candidate selection.  Not reading
        those files is the enforcement, and a script cannot open a path that is
        nowhere in its code."""
        refs = self._code_references(
            REPO_ROOT / "scripts" / "evaluate_confirmation_split.py")
        joined = "\n".join(refs)
        for banned in ("unlearning_selection", "evaluate_reference_states",
                       "select_unlearning_checkpoints", "predictions_BASE",
                       "predictions_MF", "predictions_MN",
                       "predictions_tv_"):
            assert banned not in joined, banned

    def test_the_analysis_script_names_neither_either(self):
        joined = "\n".join(self._code_references(
            REPO_ROOT / "scripts" / "analyze_confirmation_split.py"))
        for banned in ("unlearning_selection", "evaluate_reference_states",
                       "select_unlearning_checkpoints", "predictions_BASE",
                       "predictions_MF", "predictions_MN",
                       "predictions_tv_"):
            assert banned not in joined, banned

    def test_the_analysis_runs_no_equivalence_or_non_inferiority_test(self):
        """Both were demoted by the 11C-2a decisions.  The freeze records the
        0.05 margin as a reporting yardstick, so the analysis must not import a
        TOST."""
        refs = self._code_references(
            REPO_ROOT / "scripts" / "analyze_confirmation_split.py")
        for banned in ("tost", "equivalence_test", "non_inferiority_test",
                       "tost_equivalence"):
            assert banned not in refs, banned
        assert not any("tost" in r.lower() for r in refs)

    def test_verify_only_reports_completeness_and_no_rate(self, farm,
                                                          protocol, capsys):
        """Run against a farm with nothing generated: the refusal has to name
        all three states, and the output has to contain no number that could be
        read as a partial result."""
        for state in protocol["states"]:
            preds, reasons = ecs.verify_state(
                state, farm, protocol["freeze"], protocol["queries"],
                protocol["by_assoc"], protocol["model_id"], protocol["config"])
            assert preds is None
            assert reasons and "does not exist" in reasons[0]
        captured = capsys.readouterr()
        assert "diff" not in captured.out and "p_value" not in captured.out


class TestTheAnalysisRefusesUntilEveryStateIsComplete:
    def test_one_state_missing_refuses_and_names_it(self, farm, protocol,
                                                    adapters):
        _write_state(farm, protocol, "B3", "tga")
        _write_state(farm, protocol, "B0", "filr")
        try:
            with pytest.raises(SystemExit) as exc:
                acs.analyze(farm)
            message = str(exc.value)
            assert "2 of 3" in message
            assert "MG" in message
        finally:
            for state in ("B3", "B0"):
                ecs.predictions_path(farm, state).unlink(missing_ok=True)
                (ecs.predictions_path(farm, state).parent /
                 f"predictions_{state}.generation_order.json"
                 ).unlink(missing_ok=True)

    def test_a_report_is_not_written_from_a_subset(self, farm, protocol):
        """The refusal has to happen before anything is computed, so no partial
        interval exists to be tempted by — and no file is left behind."""
        with pytest.raises(SystemExit):
            acs.analyze(farm)
        assert not (farm / acs.ANALYSIS_REPORT).exists()

    def test_a_sidecar_that_does_not_verify_refuses(self, scored, protocol):
        """Provenance-verified reuse is a verified decision, not a filename
        match: a parquet whose sidecar was written for another dataset version
        is refused, which is what stops an exploratory prediction being scored
        as a confirmation one."""
        path = ecs.predictions_path(scored, "MG")
        sidecar = path.parent / f"{path.name}.provenance.json"
        record = json.loads(sidecar.read_text())
        original = sidecar.read_bytes()
        try:
            record["dataset"] = dict(record["dataset"],
                                     version="pilot100_v2")
            sidecar.write_text(json.dumps(record))
            with pytest.raises(SystemExit, match="MG"):
                acs.analyze(scored)
        finally:
            sidecar.write_bytes(original)

    def test_a_different_generation_order_refuses(self, scored, protocol):
        """Batched greedy decoding is not bit-stable across batch
        compositions, so two states generated over different orderings are not
        comparable and the paired difference would contain decoding noise."""
        order_file = (ecs.predictions_path(scored, "MG").parent /
                      "predictions_MG.generation_order.json")
        original = order_file.read_bytes()
        try:
            order_file.write_text(json.dumps(
                json.loads(original), indent=1, sort_keys=True).replace(
                    protocol["order"], "0" * 64))
            with pytest.raises(SystemExit, match="query order"):
                acs.analyze(scored)
        finally:
            order_file.write_bytes(original)

    def test_a_row_invalid_parquet_refuses(self, scored, protocol):
        """Coverage is exact, not an intersection size: a dropped row would
        make the paired comparison cover fewer probes than it claims."""
        import pandas as pd
        path = ecs.predictions_path(scored, "B0")
        original = path.read_bytes()
        try:
            frame = pd.read_parquet(path).iloc[:-1]
            frame.to_parquet(path, index=False)
            with pytest.raises(SystemExit, match="B0"):
                acs.analyze(scored)
        finally:
            path.write_bytes(original)


class TestThePrimaryIsTheFrozenClaimOverAllTargetEntities:
    def test_it_averages_over_exactly_the_72_frozen_target_entities(self,
                                                                   scored):
        report = acs.analyze(scored)
        assert report["refusals"] == []
        te = report["primary"]["target_entities"]
        assert te["target_entity_ids"] == 72
        assert te["persons"] == 42 and te["species"] == 30
        assert te["target_entity_ids_match_the_freeze"] is True
        assert te["target_association_ids_match_the_freeze"] is True
        for claim in report["primary"]["claims"].values():
            assert claim["num_entity_clusters"] == 72
            assert claim["entity_macro"]["num_entity_clusters"] == 72

    def test_the_primary_family_is_exactly_the_two_frozen_claims(self, scored):
        report = acs.analyze(scored)
        assert sorted(report["primary"]["claims"]) == [
            "B3_minus_B0:filr", "B3_minus_B0:tga"]

    def test_the_directions_are_the_frozen_ones_and_they_differ(self, scored):
        """TGA is an accuracy and FILR a leakage rate, so the two claims point
        in OPPOSITE directions on the same difference; one sign convention would
        test one of them backwards."""
        report = acs.analyze(scored)
        claims = report["primary"]["claims"]
        assert claims["B3_minus_B0:tga"]["direction"] == "greater"
        assert claims["B3_minus_B0:filr"]["direction"] == "less"

    def test_both_claims_are_rejected_when_b3_beats_b0_on_every_entity(self,
                                                                      scored):
        report = acs.analyze(scored)
        claims = report["primary"]["claims"]
        assert claims["B3_minus_B0:tga"]["entity_macro"]["diff"] == 1.0
        assert claims["B3_minus_B0:filr"]["entity_macro"]["diff"] == -1.0
        holm = report["primary"]["multiplicity"]["holm"]
        assert holm["k"] == 2
        assert sorted(holm["rejected"]) == [
            "B3_minus_B0:filr", "B3_minus_B0:tga"]
        assert holm["retained"] == []
        assert report["primary"]["verdict"]["all_rejected"] is True

    def test_holm_uses_the_frozen_thresholds_and_steps_down(self, scored):
        report = acs.analyze(scored)
        holm = report["primary"]["multiplicity"]["holm"]
        assert [s["threshold"] for s in holm["steps"]] == [0.025, 0.05]
        assert report["primary"]["multiplicity"]["familywise_alpha"] == 0.05
        assert report["primary"]["multiplicity"]["k"] == 2

    def test_the_seeds_and_draw_counts_are_the_frozen_ones(self, scored):
        """Read from the freeze, not defaulted: sharing a seed between the
        interval and the p-value would make their Monte Carlo errors dependent,
        and a reader could not tell whether they agreed because the data said so
        or because they were drawn from the same stream."""
        report = acs.analyze(scored)
        params = report["analysis_parameters_read_from_the_freeze"]
        assert params["n_permutations"] == 10000
        assert params["permutation_seed"] == 20260908
        assert params["n_bootstrap"] == 1000
        assert params["bootstrap_seed"] == 42
        assert params["ci_level"] == 0.95
        for claim in report["primary"]["claims"].values():
            assert claim["test"]["n_permutations"] == 10000
            assert claim["test"]["seed"] == 20260908

    def test_the_smallest_reportable_p_is_not_zero(self, scored):
        """The observed vector is included in the null, so an exact zero is
        never returned: p = 0.0 would claim a resolution the draw count does
        not have."""
        report = acs.analyze(scored)
        for claim in report["primary"]["claims"].values():
            test = claim["test"]
            assert test["p_value_one_sided"] >= test["p_smallest_reportable"]
            assert test["p_value_one_sided"] > 0.0
            assert test["observed_vector_included_in_the_null"] is True
            #: Every entity differs, so all 72 clusters are flippable and the
            #: test has the resolution its cluster count suggests.
            assert test["num_flippable_clusters"] == 72
            assert test["all_differences_zero"] is False

    def test_the_interval_covers_the_statistic_the_test_uses(self, scored):
        """The p-value and the CI have to be built for the SAME quantity, or one
        number serves two estimands."""
        report = acs.analyze(scored)
        for claim in report["primary"]["claims"].values():
            assert claim["test"]["statistic"] == claim["entity_macro"]["diff"]
            lo, hi = claim["entity_macro"]["ci"]
            assert lo <= claim["entity_macro"]["diff"] <= hi
            assert claim["test"]["num_clusters"] == \
                claim["entity_macro"]["num_entity_clusters"]


class TestBothAveragingUnitsAreReportedAndNotBlended:
    def test_row_micro_is_reported_beside_every_entity_macro(self, scored):
        report = acs.analyze(scored)
        for claim in report["primary"]["claims"].values():
            assert "row_micro" in claim and "entity_macro" in claim
            micro, macro = claim["row_micro"], claim["entity_macro"]
            assert micro["num_paired_rows"] == macro["num_paired_rows"] == 864
            assert "why_it_differs_from_the_macro" in micro

    def test_the_two_units_are_different_quantities_and_the_report_says_so(
            self, scored):
        """With every entity contributing 12 target probes the two happen to
        coincide here, so the test that they are NOT conflated is on the
        descriptive retention block, where entities contribute 3 probes to
        unequal numbers of entities."""
        report = acs.analyze(scored)
        assert report["primary"]["averaging_units_are_reported_separately"] \
            is True
        retention = report["descriptive"]["retention"]["metrics"]
        assert sorted(retention) == ["B3_minus_B0:retain_other",
                                     "B3_minus_B0:retain_same"]
        same = retention["B3_minus_B0:retain_same"]
        #: 70 entities carry retain_same and 45 carry retain_other, so the
        #: cluster counts differ from each other and from the 72 of the primary.
        assert same["num_entity_clusters"] == 70
        assert retention["B3_minus_B0:retain_other"][
            "num_entity_clusters"] == 45


class TestTheStrataAreSecondaryDiagnosticsOnly:
    def test_the_two_image_strata_are_reported_with_their_own_sizes(self,
                                                                   scored):
        report = acs.analyze(scored)
        strata = report["secondary_strata"]["strata"]
        assert sorted(strata) == ["held_out_photo", "seen_photo_unseen_wording"]
        person = strata["seen_photo_unseen_wording"]
        photo = strata["held_out_photo"]
        assert (person["num_probes"], person["num_entities"]) == (504, 42)
        assert (photo["num_probes"], photo["num_entities"]) == (360, 30)

    def test_no_stratum_carries_a_p_value(self, scored):
        """A stratum is where the exploratory effect was seen.  Testing it here
        would be a second family nobody adjusted for, and the person stratum is
        the one a reader would most want to see a p-value for."""
        report = acs.analyze(scored)
        for name, stratum in report["secondary_strata"]["strata"].items():
            assert "SECONDARY DIAGNOSTIC" in stratum["status"], name
            for metric, comp in stratum["metrics"].items():
                assert comp["test"] is None, (name, metric)
                assert comp["claim_kind"] == "descriptive", (name, metric)
                assert "why_there_is_no_test" in comp

    def test_the_strata_are_not_in_the_holm_family(self, scored):
        report = acs.analyze(scored)
        holmed = {s["claim"]
                  for s in report["primary"]["multiplicity"]["holm"]["steps"]}
        assert holmed == {"B3_minus_B0:tga", "B3_minus_B0:filr"}
        assert not any("seen_photo" in c or "held_out" in c for c in holmed)


class TestTheDescriptiveBlocksConcludeNothing:
    def test_retention_is_descriptive_with_no_declared_margin(self, scored):
        report = acs.analyze(scored)
        retention = report["descriptive"]["retention"]
        assert retention["declared_margin"] is None
        assert "no margin" in retention["status"].lower()
        assert "0.0556" in retention["why"]
        for comp in retention["metrics"].values():
            assert comp["test"] is None
            assert comp["claim_kind"] == "descriptive"

    def test_retention_probes_are_the_text_route_ones(self, scored):
        """The image route was omitted from retention: 64 of the 70 retention
        entities are persons no new photograph can be sourced for."""
        report = acs.analyze(scored)
        assert report["descriptive"]["retention"]["route"] == \
            "text_to_text only"
        states = report["inputs_bound"]["states"]
        for audit in states.values():
            assert audit["retain_probe_rows_scored"] == 345
            assert audit["target_probe_rows_scored"] == 864
            assert audit["retain_probe_rows_scored"] + \
                audit["target_probe_rows_scored"] == 1209

    def test_b3_minus_mg_publishes_an_interval_and_concludes_no_equivalence(
            self, scored):
        report = acs.analyze(scored)
        mg = report["descriptive"]["b3_minus_mg"]
        assert mg["equivalence_concluded"] is False
        assert mg["equivalence_test_run"] is False
        assert "absence of evidence" in mg["why_no_equivalence_conclusion"]
        assert mg["margin_that_is_not_tested"] == 0.05
        assert "YARDSTICK" in mg["margin_role"]
        for comp in mg["metrics"].values():
            assert comp["test"] is None
            assert comp["states_compared"] == ["B3", "MG"]

    def test_a_zero_difference_from_the_oracle_is_still_not_equivalence(
            self, scored):
        """MG was fabricated to answer at the target level, exactly as B3 does,
        so every B3 − MG difference here is zero with a degenerate interval.
        That is the strongest possible-looking case for equivalence and the
        report still concludes none, which is the point: the claim was never
        licensed by the data, only by a test that was not run."""
        report = acs.analyze(scored)
        mg = report["descriptive"]["b3_minus_mg"]
        assert mg["metrics"]["B3_minus_MG:tga"]["entity_macro"]["diff"] == 0.0
        assert mg["metrics"]["B3_minus_MG:tga"]["entity_macro"]["ci"] == \
            [0.0, 0.0]
        assert mg["equivalence_concluded"] is False


class TestTheReportBindsWhatItRead:
    def test_every_input_is_bound_by_hash(self, scored):
        report = acs.analyze(scored)
        bound = report["inputs_bound"]
        assert bound["dataset_version"] == "confirm100_v1"
        assert bound["experiment_id"] == "mllmu_confirm100_iter11c"
        assert len(bound["freeze_sha256"]) == 64
        assert len(bound["power_sha256"]) == 64
        assert len(bound["queries_parquet_sha256"]) == 64
        assert len(bound["states"]) == 3
        for state, audit in bound["states"].items():
            assert audit["complete"] is True
            assert audit["sidecar_verified"] is True
            assert audit["num_predictions"] == 1209
            assert audit["generation_order_matches_the_other_states"] is True
            assert len(audit["parquet_sha256"]) == 64, state

    def test_compliance_is_stated_as_measurements_not_prose(self, scored):
        report = acs.analyze(scored)
        compliance = report["protocol_compliance"]
        assert compliance["reference_state_gate_invoked"] is False
        assert compliance["reference_state_gate_parquets_read"] == []
        assert compliance["checkpoint_selection_invoked"] is False
        assert compliance["selection_report_read"] is False
        assert compliance["equivalence_test_run"] is False
        assert compliance["non_inferiority_test_run"] is False
        assert compliance["query_ordering_identical_across_states"] is True
        assert compliance["generation_config"] == {
            "batch_size": 8, "image_batch_size": 8, "max_new_tokens": 96,
            "do_sample": False, "max_image_pixels": 147456, "max_length": 1536}
        assert sorted(compliance["states_excluded"]) == [
            "B1", "B2", "B2R", "BASE", "MF", "MN"]

    def test_the_report_is_json_serialisable_and_sorted(self, scored):
        report = acs.analyze(scored)
        text = json.dumps(report, indent=1, sort_keys=True)
        assert json.loads(text) == json.loads(
            json.dumps(json.loads(text), sort_keys=True))


class TestADegenerateResultIsReportedAsADegenerateResult:
    """Invariant 7: a failed primary claim is reported as failed.

    Fabricating B3 identical to B0 makes every entity's paired difference
    exactly zero, which is the shape a null confirmation result has.  The
    report has to say "not rejected" and not soften it, and the test has to say
    what the test's own resolution is when nothing can flip.
    """

    def test_nothing_is_rejected_when_b3_equals_b0(self, farm, protocol,
                                                   adapters):
        _write_state(farm, protocol, "B3", "filr")
        _write_state(farm, protocol, "B0", "filr")
        _write_state(farm, protocol, "MG", "tga")
        try:
            report = acs.analyze(farm)
            assert report["refusals"] == []
            holm = report["primary"]["multiplicity"]["holm"]
            assert holm["rejected"] == []
            assert sorted(holm["retained"]) == [
                "B3_minus_B0:filr", "B3_minus_B0:tga"]
            assert report["primary"]["verdict"]["all_rejected"] is False
            for claim in report["primary"]["claims"].values():
                assert claim["entity_macro"]["diff"] == 0.0
                assert claim["test"]["p_value_one_sided"] == 1.0
                assert claim["test"]["all_differences_zero"] is True
                assert claim["test"]["num_flippable_clusters"] == 0
            for name, verdict in report["primary"]["verdict"][
                    "per_claim"].items():
                assert verdict["rejected"] is False
                assert "NOT rejected" in verdict["statement"], name
                assert "no significance is claimed" in verdict["statement"]
        finally:
            for state in protocol["states"]:
                ecs.predictions_path(farm, state).unlink(missing_ok=True)
                (ecs.predictions_path(farm, state).parent /
                 f"predictions_{state}.generation_order.json"
                 ).unlink(missing_ok=True)
