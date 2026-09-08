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

import io
import json
import logging
import shutil
import sys
from contextlib import contextmanager
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
    """A repo that is the real one by symlink, except where a test must write.

    TWO directories are made real with symlinked ENTRIES instead of being
    symlinked wholesale, because writing through a symlinked directory writes
    into the committed repository:

    * ``data/mllmu_hier_confirm100`` — its ``predictions/`` is where the
      fabricated parquets go, and it must not be a link into the real dataset,
      or a test would leave predictions behind in a dataset whose whole
      property is that stage 5 has not run;
    * ``data/reports`` — several tests below replace the freeze to check a
      refusal, and an ``unlink`` through a symlinked directory would delete the
      COMMITTED freeze rather than a link to it.
    """
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)
    for entry in REPO_ROOT.iterdir():
        (root / entry.name).symlink_to(entry)
    (root / "data").unlink()
    (root / "data").mkdir()
    for entry in (REPO_ROOT / "data").iterdir():
        (root / "data" / entry.name).symlink_to(entry)
    for real_dir in (("data", "mllmu_hier_confirm100"), ("data", "reports")):
        target = REPO_ROOT.joinpath(*real_dir)
        local = root.joinpath(*real_dir)
        local.unlink()
        local.mkdir()
        for entry in target.iterdir():
            (local / entry.name).symlink_to(entry)
    ds = root / "data" / "mllmu_hier_confirm100"
    (ds / "predictions").mkdir()
    assert not (ds / "predictions").is_symlink()
    assert not (root / "data" / "reports").is_symlink()
    return root


@contextmanager
def _mutated_freeze(farm: Path, mutate):
    """Replace the farm's freeze for the duration of one test, then restore it.

    ``mutate`` takes the committed freeze as a dict and returns the variant to
    test against.  The path is a symlink into ``data/reports``, which the farm
    makes a REAL directory, so unlinking it removes the link and the committed
    artifact is never written to — an earlier version of this file replaced the
    freeze through a symlinked directory, which transiently deleted the real
    one and relied on a ``finally`` block to put it back.
    """
    path = farm / ecs.FREEZE_REPORT
    original = path.read_bytes()
    assert path.is_symlink(), (
        f"{path} is not a symlink, so replacing it would edit the committed "
        "freeze rather than the farm's view of it")
    variant = mutate(json.loads(original))
    try:
        path.unlink()
        path.write_text(json.dumps(variant))
        yield variant
    finally:
        #: Restore the LINK, not just the bytes: the farm is module-scoped, so
        #: writing a regular file back would leave every later ``_mutated_freeze``
        #: facing a non-symlink and failing its own safety assertion.
        path.unlink(missing_ok=True)
        path.symlink_to(REPO_ROOT / ecs.FREEZE_REPORT)
        assert path.is_symlink() and path.read_bytes() == original


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


@contextmanager
def _mutated_module(farm: Path, rel: str, suffix: bytes = b"\n# mutated\n"):
    """Replace ONE module in the farm with an edited copy, then put it back.

    ``farm/src`` is a symlink into the repository, so writing through it would
    edit the COMMITTED module and every test that ran after this one.  Each
    parent of the target is therefore turned into a real directory holding links
    to the same entries — the trick ``_build_farm`` uses for ``data/`` — the leaf
    is written as a real file, and the whole chain is restored afterwards.
    """
    parts = Path(rel).parts
    original = (farm / rel).read_bytes()
    replaced: list[tuple[Path, Path]] = []
    current = farm
    for part in parts[:-1]:
        child = current / part
        if child.is_symlink():
            target = child.resolve()
            replaced.append((child, target))
            child.unlink()
            child.mkdir()
            for entry in target.iterdir():
                (child / entry.name).symlink_to(entry)
        current = child
    leaf = current / parts[-1]
    assert leaf.is_symlink(), (
        f"{leaf} is not a symlink, so writing it would edit the committed "
        "module rather than the farm's view of it")
    leaf.unlink()
    leaf.write_bytes(original + suffix)
    try:
        yield leaf
    finally:
        #: Innermost first: each directory has to be a link again before the one
        #: containing it is replaced by its own.
        for link, target in reversed(replaced):
            shutil.rmtree(link)
            link.symlink_to(target)
        assert (farm / rel).read_bytes() == original


@contextmanager
def _mutated_dataset_file(farm: Path, name: str, edit):
    """Replace one confirmation-dataset artifact with an edited copy.

    ``_build_farm`` makes ``data/mllmu_hier_confirm100`` a REAL directory whose
    entries are links, so unlinking an entry removes the link and the committed
    dataset is never written to — the property ``_mutated_freeze`` relies on.
    ``edit`` takes the artifact's bytes and returns the replacement.
    """
    path = farm / ecs.CONFIRM_DATASET_DIR / name
    assert path.is_symlink(), (
        f"{path} is not a symlink, so writing it would edit the committed "
        "dataset rather than the farm's view of it")
    target = path.resolve()
    original = path.read_bytes()
    path.unlink()
    path.write_bytes(edit(original))
    try:
        yield path
    finally:
        path.unlink(missing_ok=True)
        path.symlink_to(target)
        assert path.read_bytes() == original


def _edited_parquet(raw: bytes, mutate) -> bytes:
    """``mutate`` a DataFrame, then return the rewritten file's bytes.

    Rewritten rather than patched byte-wise, because the claim under test is
    that the freeze binds the FILE: a change to one cell that leaves every
    query id and every count intact still has to be visible.
    """
    import pandas as pd
    frame = pd.read_parquet(io.BytesIO(raw))
    mutate(frame)
    out = io.BytesIO()
    frame.to_parquet(out, index=False)
    return out.getvalue()


def _absent_photographs(repo_root: Path) -> list[str]:
    """The sealed confirmation photographs that are not on disk.

    ``image_manifest.json`` IS committed — 402 paths with their sha256 — but the
    bytes behind them are not: the 360 iNaturalist photographs are gitignored and
    re-fetchable through ``confirm_v1/PROVENANCE.json``, and the 42 exempted
    target-person portraits sit under the gitignored exploratory media.  So "does
    the frozen pool verify" is a question a bare clone cannot answer.
    """
    data_dir = repo_root / ecs.CONFIRM_DATASET_DIR
    return [str(path) for path in ecs.referenced_image_paths(data_dir)
            if ecs.resolve_image_path(path, data_dir, repo_root) is None]


def _skip_without_photographs(repo_root: Path) -> None:
    """Skip, naming the absent bytes, when a test needs the frozen photographs.

    The committed HALF of this evidence — that the manifest pins 402 paths and
    hashes, that the pool is disjoint from exploratory media, that the fetch
    provenance accounts for every file — is tested in
    ``test_iteration11c_probes.py`` and needs no bytes at all.  What is skipped
    here is only the re-hash of the photographs themselves, which is a check on
    this machine's pool and not a check on the repository.
    """
    absent = _absent_photographs(repo_root)
    if absent:
        pytest.skip(
            f"{len(absent)} of the sealed confirmation photographs are absent "
            f"and gitignored (first: {absent[0]}); this test asserts something "
            "about the frozen image BYTES, so it runs where the pool was "
            "fetched, and the committed manifest that pins them is tested "
            "without bytes in test_iteration11c_probes.py")


@pytest.fixture(scope="module")
def photographs(farm) -> Path:
    """SKIP, not error, when the sealed photograph bytes are absent.

    The same convention as ``adapters`` and for the same reason: these are
    gitignored inputs, so a fresh checkout is missing nothing it was supposed to
    have, and a red suite gets overridden.  Unlike ``adapters`` this guard is
    also called directly from ``_run_main``, because ``main()`` runs the
    preflight and a test that reaches it must not be able to bypass the guard by
    calling the helper instead of requesting the fixture.
    """
    _skip_without_photographs(farm)
    return farm


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
    ecs.generation_order_path(farm, state).write_text(
        json.dumps({
            "state": state,
            "experiment_id": ecs.EXPERIMENT_ID,
            "num_queries": len(preds),
            "generation_order_sha256": order or protocol["order"],
            "generation_config": cfg,
        }, indent=1, sort_keys=True))
    return path


#: The canonical fabricated result: B3 answers at the target level, B0 leaks the
#: finest, MG answers at the target level too.
CANONICAL_STATES = (("B3", "tga"), ("B0", "filr"), ("MG", "tga"))


def _sidecar_for(farm: Path, state: str) -> Path:
    path = ecs.predictions_path(farm, state).parent / \
        f"predictions_{state}.parquet.provenance.json"
    assert path.exists(), f"the sidecar naming changed: {path} is absent"
    return path


def _clear(farm: Path, *states: str) -> None:
    """Delete fabricated states, leaving the farm with none of them."""
    for state in states:
        ecs.predictions_path(farm, state).unlink(missing_ok=True)
        ecs.generation_order_path(farm, state).unlink(missing_ok=True)


def _restore_scored(farm: Path, protocol: dict) -> None:
    """Put the farm back into the state the ``scored`` fixture establishes.

    ``scored`` is MODULE-scoped, so it is built once and every later test that
    requests it receives the same farm.  A test that overwrites or deletes those
    parquets therefore has to put them back, or the next test silently finds an
    empty dataset and fails on a fixture that pytest still reports as set up.
    """
    _clear(farm, *(state for state, _ in CANONICAL_STATES))
    for state, kind in CANONICAL_STATES:
        _write_state(farm, protocol, state, kind)


@pytest.fixture
def unscored(farm, protocol) -> Path:
    """The farm with NOTHING generated, and the canonical three restored after.

    Several refusals can only be observed on a farm that has no predictions on
    it, and ``scored`` is module-scoped — so a test needing the empty state has
    to establish it and put it back rather than depend on where in the file it
    happens to sit.  Three of these tests were written against ambient state
    first and passed or failed according to what the preceding test had left
    behind, which is not a property a regression test should have.

    The CLEAR needs no adapter bytes, only the RESTORE does, so ``adapters`` is
    deliberately not a dependency: requiring it would drag tests that run fine
    without the adapters — ``analyze`` refusing an empty farm is one — into the
    clean-clone skip set.  Where the adapters are absent nothing was ever
    written, so there is nothing to put back.
    """
    _clear(farm, *(state for state, _ in CANONICAL_STATES))
    yield farm
    if not _absent_adapters(protocol["freeze"]):
        _restore_scored(farm, protocol)


@pytest.fixture(scope="module")
def scored(farm, protocol, adapters, photographs) -> Path:
    """The farm with all three states present.

    B0 is the no-op, so it still leaks the fine value; B3 is the
    granularity-aware unlearning, so it answers at the target level; MG is the
    oracle and answers at the target level too.  That makes B3 − B0 maximal in
    both directions the primary claims point — TGA up, FILR down — and B3 − MG
    zero, which is the shape the real result is expected to have and the one
    the descriptive block has to decline to call equivalence.
    """
    for state, kind in CANONICAL_STATES:
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
        with (_mutated_freeze(farm, lambda f: dict(
                f, refusals=["something is wrong"])),
              pytest.raises(SystemExit, match="carries 1 refusal")):
            ecs.load_frozen_protocol(farm)


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
                                                    adapters, unscored):
        _write_state(farm, protocol, "B3", "tga")
        _write_state(farm, protocol, "B0", "filr")
        with pytest.raises(SystemExit) as exc:
            acs.analyze(farm)
        message = str(exc.value)
        assert "2 of 3" in message
        assert "MG" in message

    def test_a_report_is_not_written_from_a_subset(self, farm, protocol,
                                                   unscored):
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
            #: Restore the canonical three rather than merely deleting the
            #: degenerate ones — ``scored`` is module-scoped and later tests
            #: request it expecting the farm to still hold its files.
            _restore_scored(farm, protocol)


class TestTheFrozenCodeHashesAreComparedAndNotJustRecorded:
    """11C-5R finding 4: the pins were recorded and never compared.

    The freeze bound the evaluator, the analyzer and ``paired_ci.py``, but no
    runtime code read those hashes back.  A recorded hash is a DESCRIPTION of
    the protocol; comparing it is what makes sealed-split invariant 7
    enforceable.  Without the comparison, the module implementing the sign-flip
    test, the paired CI and the Holm step-down could be edited after the freeze
    and the confirmation would still report itself as produced under the frozen
    protocol — and ``paired_ci.py`` is the one module the primary claims depend
    on that ``CODE_FINGERPRINT_MODULES`` deliberately does not cover.
    """

    def test_the_committed_freeze_matches_the_committed_code(self, farm,
                                                            protocol):
        assert ecs.verify_frozen_code(farm, protocol["freeze"]) == []

    def test_every_fingerprinted_module_is_compared_not_just_recorded(
            self, farm, protocol):
        """11C-5R2 finding 1, and THE regression for it.

        ``code.fingerprinted_modules`` held ten hashes that nothing read back:
        the reviewer edited one in a temporary copy and the preflight still
        returned no problems.  The sidecar cannot carry this check —
        ``PredictionFingerprint.build`` hashes these modules into every
        prediction file, but ``expected_fingerprint`` re-hashes them from the
        same disk when that file is verified, so an edit moves the recorded
        value and the expected value together and the sidecar still verifies.
        """
        victim = "src/granunlearn/evaluation/scoring.py"
        assert victim in protocol["freeze"]["code"]["fingerprinted_modules"]
        with _mutated_module(farm, victim):
            problems = ecs.verify_frozen_code(farm, protocol["freeze"])
        assert len(problems) == 1
        assert victim in problems[0]
        #: It has to say why a sidecar full of these hashes did not notice.
        assert "CODE_FINGERPRINT_MODULES" in problems[0]
        assert "no sidecar comparison can notice" in problems[0]

    def test_every_one_of_the_ten_modules_is_covered(self, farm, protocol):
        """Not just the one above: the check has to reach the modules that
        define the hierarchy, the image splits and the persisted row schemas
        too, or an edit to one of those is still invisible."""
        bound = protocol["freeze"]["code"]["fingerprinted_modules"]
        assert set(bound) == set(
            protocol["freeze"]["code"]["fingerprinted_module_list"])
        for rel in ("src/granunlearn/evaluation/hierarchy_metrics.py",
                    "src/granunlearn/evaluation/image_splits.py",
                    "src/granunlearn/schema/prediction.py"):
            assert rel in bound, rel
            with _mutated_module(farm, rel):
                problems = ecs.verify_frozen_code(farm, protocol["freeze"])
            assert len(problems) == 1, rel
            assert rel in problems[0]

    def test_a_fingerprinted_module_that_is_absent_is_reported(self, farm):
        def mutate(f):
            code = dict(f["code"])
            modules = dict(code["fingerprinted_modules"], **{
                "src/granunlearn/evaluation/never_written.py": "0" * 64})
            return dict(f, code=dict(code, fingerprinted_modules=modules))
        with _mutated_freeze(farm, mutate):
            problems = ecs.verify_frozen_code(farm, ecs.load_frozen_protocol(farm))
        assert any("absent from the repository" in p for p in problems)

    def test_a_module_listed_but_not_hashed_is_reported(self, farm):
        """The freeze records the module LIST beside the hashes, so a dict
        missing an entry is detectable rather than silently shorter: hashing nine
        of the ten would still leave every sidecar verifiable."""
        def mutate(f):
            code = dict(f["code"])
            modules = dict(code["fingerprinted_modules"])
            modules.pop("src/granunlearn/evaluation/scoring.py")
            return dict(f, code=dict(code, fingerprinted_modules=modules))
        with _mutated_freeze(farm, mutate):
            problems = ecs.verify_frozen_code(farm, ecs.load_frozen_protocol(farm))
        assert any("pins no sha256 for" in p and "scoring.py" in p
                   for p in problems)

    def test_a_freeze_that_pins_no_fingerprinted_modules_is_a_refusal(
            self, farm):
        def mutate(f):
            return dict(f, code=dict(f["code"], fingerprinted_modules={}))
        with _mutated_freeze(farm, mutate):
            problems = ecs.verify_frozen_code(farm, ecs.load_frozen_protocol(farm))
        assert any("binds no code.fingerprinted_modules" in p
                   for p in problems)

    def test_both_confirmation_scripts_and_paired_ci_are_in_the_bound_set(
            self, protocol):
        bound = protocol["freeze"]["code"]["analysis_scripts_sha256"]
        for rel in ("scripts/evaluate_confirmation_split.py",
                    "scripts/analyze_confirmation_split.py"):
            assert rel in bound, rel
        impl = protocol["freeze"]["primary_test"]["implementation"]
        assert impl["module"] == "src/granunlearn/evaluation/paired_ci.py"
        assert impl["in_the_code_fingerprint"] is False

    def test_a_drifted_analysis_script_is_reported(self, farm):
        def mutate(f):
            code = dict(f["code"])
            scripts = dict(code["analysis_scripts_sha256"],
                           **{"scripts/analyze_confirmation_split.py": "0" * 64})
            return dict(f, code=dict(code, analysis_scripts_sha256=scripts))
        with _mutated_freeze(farm, mutate):
            problems = ecs.verify_frozen_code(farm, ecs.load_frozen_protocol(farm))
        assert len(problems) == 1
        assert "analyze_confirmation_split.py" in problems[0]
        assert "re-freeze" in problems[0]

    def test_a_drifted_paired_ci_is_reported_and_says_what_it_implements(
            self, farm):
        """The most consequential drift, and the one nothing else pins."""
        def mutate(f):
            pt = dict(f["primary_test"])
            pt["implementation"] = dict(pt["implementation"], sha256="0" * 64)
            return dict(f, primary_test=pt)
        with _mutated_freeze(farm, mutate):
            problems = ecs.verify_frozen_code(farm, ecs.load_frozen_protocol(farm))
        assert len(problems) == 1
        assert "paired_ci.py" in problems[0]
        #: It has to say what is at stake, not merely that a hash differs.
        assert "one_sided_permutation_pvalue" in problems[0]
        assert "CODE_FINGERPRINT_MODULES" in problems[0]

    def test_a_bound_script_that_is_absent_is_reported(self, farm):
        def mutate(f):
            code = dict(f["code"])
            scripts = dict(code["analysis_scripts_sha256"],
                           **{"scripts/never_written.py": "0" * 64})
            return dict(f, code=dict(code, analysis_scripts_sha256=scripts))
        with _mutated_freeze(farm, mutate):
            problems = ecs.verify_frozen_code(farm, ecs.load_frozen_protocol(farm))
        assert any("absent from the repository" in p for p in problems)

    def test_a_freeze_that_pins_no_code_is_itself_a_refusal(self, farm):
        """An empty pin set must not read as "nothing to check, all clear"."""
        def mutate(f):
            code = dict(f["code"], analysis_scripts_sha256={})
            pt = dict(f["primary_test"], implementation={})
            return dict(f, code=code, primary_test=pt)
        with _mutated_freeze(farm, mutate):
            problems = ecs.verify_frozen_code(farm, ecs.load_frozen_protocol(farm))
        assert any("binds no analysis_scripts_sha256" in p for p in problems)
        assert any("names no module or no sha256" in p for p in problems)

    def test_the_analysis_refuses_when_a_bound_hash_drifts(self, scored):
        """End to end: a drifted pin stops the report, not just the check."""
        def mutate(f):
            code = dict(f["code"])
            scripts = dict(code["analysis_scripts_sha256"],
                           **{"scripts/evaluate_confirmation_split.py": "0" * 64})
            return dict(f, code=dict(code, analysis_scripts_sha256=scripts))
        with (_mutated_freeze(scored, mutate),
              pytest.raises(SystemExit, match="has changed since the "
                                              "protocol was frozen")):
            acs.analyze(scored)

    def test_the_report_records_that_the_check_ran(self, scored):
        report = acs.analyze(scored)
        compliance = report["protocol_compliance"]
        assert compliance["frozen_code_hashes_verified_at_runtime"] is True
        assert "analysis_scripts_sha256" in \
            compliance["what_the_runtime_verifications_are"]

    def test_the_analysis_refuses_an_edited_fingerprinted_module(self, scored):
        """End to end: no report is assembled under an edited scorer.

        The analysis imports the module at process start, so the edit does not
        change what THIS process computes — it changes what the repository says
        the protocol was, which is the thing a preregistered verdict has to be
        produced under.
        """
        with (_mutated_module(scored,
                              "src/granunlearn/evaluation/scoring.py"),
              pytest.raises(SystemExit, match="CODE_FINGERPRINT_MODULES")):
            acs.analyze(scored)


class TestTheFrozenDatasetIsBoundByItsBytes:
    """11C-5R2 finding 2: the frozen query semantics could change undetected.

    ``confirmation_queries`` binds the SORTED query-id list and the row count.
    That identifies which probes were selected — the right thing to seal at
    stage 3 — and says nothing about what they ask.  A prompt with an extra
    instruction appended behind an unchanged ``query_id`` was accepted as the
    frozen split, and so was an association whose target level had been moved,
    which changes what every TGA and FILR judgement on that association means.

    The sidecars do not close the gap for the same reason they do not close it
    for the code hashes: ``expected_fingerprint`` re-derives
    ``dataset_fingerprint`` from the same directory, so both sides move together.
    And ``manifest.json``'s own ``frozen_artifact_sha256`` cannot either, because
    it sits inside the dataset it describes — one test below rewrites a parquet
    and updates the manifest to agree with it, and the refusal still fires.
    """

    def test_the_committed_dataset_matches_the_committed_freeze(self, farm,
                                                               protocol):
        assert ecs.verify_confirmation_dataset(farm, protocol["freeze"]) == []

    def test_all_three_artifacts_and_the_image_manifest_are_pinned(self,
                                                                  protocol):
        bound = protocol["freeze"]["confirmation_dataset"]
        assert set(bound["artifacts_sha256"]) == set(
            ecs.CONFIRM_DATASET_ARTIFACTS)
        assert bound["artifacts_the_hashes_cover"] == sorted(
            ecs.CONFIRM_DATASET_ARTIFACTS)
        assert bound["built_at_freeze_time"] is True
        assert bound["version"] == "confirm100_v1"
        assert bound["data_dir"] == ecs.CONFIRM_DATASET_DIR
        assert len(bound["image_manifest_sha256"]) == 64
        #: Every pinned value is a hash of a file that is actually there.
        for name, sha in bound["artifacts_sha256"].items():
            assert len(sha) == 64, name

    def test_a_prompt_rewritten_behind_its_own_id_is_refused(self, farm,
                                                             protocol):
        """THE regression.  The seal passes and the dataset check does not, which
        is the whole point: the ids and the count are not the semantics."""
        def edit(raw):
            def mutate(frame):
                frame.loc[0, "prompt"] = (
                    frame.loc[0, "prompt"] + " Answer with one word only.")
            return _edited_parquet(raw, mutate)
        with _mutated_dataset_file(farm, "queries.parquet", edit):
            #: The stage-3 seal is satisfied by the edit — same ids, same count.
            assert len(ecs.confirmation_queries(
                farm, protocol["freeze"])) == len(protocol["queries"])
            problems = ecs.verify_confirmation_dataset(farm, protocol["freeze"])
        assert len(problems) == 1
        assert "queries.parquet" in problems[0]
        assert "not the dataset the protocol was frozen over" in problems[0]

    def test_an_answer_set_edited_behind_its_own_id_is_refused(self, farm,
                                                              protocol):
        """The mutation that changes a verdict rather than a wording: one query's
        acceptable answers widened, so a leak would have been scored correct."""
        def edit(raw):
            def mutate(frame):
                frame.loc[0, "expected_answer"] = "something else entirely"
            return _edited_parquet(raw, mutate)
        with _mutated_dataset_file(farm, "queries.parquet", edit):
            problems = ecs.verify_confirmation_dataset(farm, protocol["freeze"])
        assert any("queries.parquet" in p for p in problems)

    def test_an_association_hierarchy_edited_behind_its_id_is_refused(
            self, farm, protocol):
        """``target_level`` decides which rung of the hierarchy counts as a
        correct answer, so moving it moves TGA and FILR for every query on that
        association while leaving all 317 ids in place."""
        def edit(raw):
            def mutate(frame):
                frame.loc[0, "target_level"] = frame.loc[0, "target_level"] + 1
            return _edited_parquet(raw, mutate)
        with _mutated_dataset_file(farm, "associations.parquet", edit):
            problems = ecs.verify_confirmation_dataset(farm, protocol["freeze"])
        assert any("associations.parquet" in p for p in problems)

    def test_a_dataset_that_agrees_with_itself_is_still_refused(self, farm,
                                                               protocol):
        """``manifest.json`` declares ``frozen_artifact_sha256`` for the two
        parquets, so an edit that updates those too leaves the dataset
        self-consistent.  Only a pin from OUTSIDE the dataset can see it."""
        import hashlib

        def edit_queries(raw):
            def mutate(frame):
                frame.loc[0, "prompt"] = frame.loc[0, "prompt"] + " Briefly."
            return _edited_parquet(raw, mutate)

        with _mutated_dataset_file(farm, "queries.parquet",
                                   edit_queries) as edited:
            new_sha = hashlib.sha256(edited.read_bytes()).hexdigest()

            def edit_manifest(raw):
                doc = json.loads(raw)
                doc["frozen_artifact_sha256"]["queries.parquet"] = new_sha
                return json.dumps(doc, indent=1).encode()

            with _mutated_dataset_file(farm, "manifest.json", edit_manifest):
                manifest = json.loads(
                    (farm / ecs.CONFIRM_DATASET_DIR / "manifest.json"
                     ).read_text())
                #: The dataset now agrees with itself about the edit.
                assert manifest["frozen_artifact_sha256"][
                    "queries.parquet"] == new_sha
                problems = ecs.verify_confirmation_dataset(
                    farm, protocol["freeze"])
        assert any("queries.parquet" in p for p in problems), problems
        assert any("manifest.json" in p for p in problems), problems

    def test_a_version_that_moved_is_reported(self, farm, protocol):
        """The version is a string someone can forget to bump, so it is compared
        rather than trusted — and a bump nobody authorised is reported too."""
        def mutate(f):
            return dict(f, confirmation_dataset=dict(
                f["confirmation_dataset"], version="confirm100_v2"))
        with _mutated_freeze(farm, mutate):
            problems = ecs.verify_confirmation_dataset(
                farm, ecs.load_frozen_protocol(farm))
        assert any("declares version" in p for p in problems)

    def test_a_rewritten_image_manifest_is_reported(self, farm, protocol):
        """``verify_image_manifest`` compares 402 photographs AGAINST this
        roll-up, so a manifest rewritten to describe swapped photographs would
        agree with the swaps and pass that check."""
        def mutate(f):
            return dict(f, confirmation_dataset=dict(
                f["confirmation_dataset"], image_manifest_sha256="0" * 64))
        with _mutated_freeze(farm, mutate):
            problems = ecs.verify_confirmation_dataset(
                farm, ecs.load_frozen_protocol(farm))
        assert any("rolls up to" in p for p in problems)

    def test_a_pin_set_missing_one_of_the_three_is_reported(self, farm):
        """An empty pin set must not read as "nothing to check, all clear", and
        neither must a shorter one."""
        def mutate(f):
            cd = dict(f["confirmation_dataset"])
            arts = dict(cd["artifacts_sha256"])
            arts.pop("associations.parquet")
            return dict(f, confirmation_dataset=dict(
                cd, artifacts_sha256=arts))
        with _mutated_freeze(farm, mutate):
            problems = ecs.verify_confirmation_dataset(
                farm, ecs.load_frozen_protocol(farm))
        assert any("pins" in p and "associations.parquet" in p
                   for p in problems)

    def test_an_artifact_bound_but_absent_is_reported(self, farm):
        def mutate(f):
            cd = dict(f["confirmation_dataset"])
            arts = dict(cd["artifacts_sha256"],
                        **{"never_written.parquet": "0" * 64})
            return dict(f, confirmation_dataset=dict(cd,
                                                     artifacts_sha256=arts))
        with _mutated_freeze(farm, mutate):
            problems = ecs.verify_confirmation_dataset(
                farm, ecs.load_frozen_protocol(farm))
        assert any("absent from the dataset" in p for p in problems)

    def test_a_freeze_derived_before_the_split_was_built_refuses(self, farm):
        """The split is built AFTER the protocol is frozen — the builder reads the
        freeze and refuses when it refuses — so an honest freeze can have no
        dataset hashes yet.  What is not honest is scoring under it, and the
        refusal has to say what to do instead."""
        def mutate(f):
            return dict(f, confirmation_dataset=dict(
                f["confirmation_dataset"], artifacts_sha256={},
                built_at_freeze_time=False,
                status="the split had NOT been built when this freeze was "
                       "derived"))
        with _mutated_freeze(farm, mutate):
            problems = ecs.verify_confirmation_dataset(
                farm, ecs.load_frozen_protocol(farm))
        assert len(problems) == 1
        assert "binds no confirmation_dataset" in problems[0]
        assert "had NOT been built" in problems[0]
        assert "--allow-refreeze" in problems[0]

    def test_a_freeze_with_no_dataset_block_at_all_refuses(self, farm):
        """The freeze this repair was written against: the key simply was not
        there, and a ``.get`` chain turned that into an empty pin set."""
        def mutate(f):
            out = dict(f)
            out.pop("confirmation_dataset")
            return out
        with _mutated_freeze(farm, mutate):
            problems = ecs.verify_confirmation_dataset(
                farm, ecs.load_frozen_protocol(farm))
        assert any("binds no confirmation_dataset" in p for p in problems)

    def test_the_analysis_refuses_a_rewritten_prompt(self, scored):
        """End to end: no report over a dataset the freeze does not bind."""
        def edit(raw):
            def mutate(frame):
                frame.loc[0, "prompt"] = frame.loc[0, "prompt"] + " Briefly."
            return _edited_parquet(raw, mutate)
        with (_mutated_dataset_file(scored, "queries.parquet", edit),
              pytest.raises(SystemExit, match="not the dataset the protocol "
                                              "was frozen over")):
            acs.analyze(scored)

    def test_the_report_says_the_dataset_hashes_were_compared(self, scored):
        compliance = acs.analyze(scored)["protocol_compliance"]
        assert compliance["frozen_dataset_hashes_verified_at_runtime"] is True
        assert "confirmation_dataset.artifacts_sha256" in \
            compliance["what_the_runtime_verifications_are"]
        #: And why the sidecars could not carry it alone.
        assert "RE-DERIVED" in compliance[
            "why_the_code_and_dataset_hashes_are_not_left_to_the_sidecars"]


class TestTheOutstandingAnswerCannotBePolluted:
    """11C-5R2 finding 3, in the scorer: the chain's half is in
    ``test_iteration11c5_chain.py``.

    ``--list-outstanding`` made stdout the data channel and silenced THIS
    module's logger.  ``setup_logger`` attaches a ``StreamHandler(sys.stdout)``
    to every logger it creates, and verification reaches into
    ``prediction_provenance``, whose ``read_sidecar`` logs one ERROR line when a
    sidecar is truncated.  That line arrived where a state name was expected,
    reproduced here with a sidecar cut off inside a string.
    """

    TRUNCATED = '{"contract_version": 2, "experiment_id": "mll'

    def test_a_truncated_sidecar_puts_nothing_into_the_channel(self, farm,
                                                              protocol,
                                                              adapters,
                                                              unscored,
                                                              monkeypatch,
                                                              capsys):
        """THE regression, on the exact artifact the reviewer truncated.

        ``unscored`` rather than a local ``_clear``: ``scored`` is
        module-scoped, so a test that writes one state and then deletes it
        leaves the farm missing a state every later test assumes is there.
        """
        _write_state(farm, protocol, "B3", "tga")
        _sidecar_for(farm, "B3").write_text(self.TRUNCATED)
        _run_main(farm, monkeypatch, "--list-outstanding")
        out = capsys.readouterr().out
        assert out.split() == list(protocol["states"]), out

    def test_a_foreign_logger_cannot_reach_the_channel(self, farm, protocol,
                                                       monkeypatch, capsys):
        """The general form of the finding: the leak was not this module's
        logger, so clearing this module's handlers could not have fixed it.  Any
        logger in the process, at any level, is muted for this mode."""
        real = ecs.verify_state

        def noisy(*args, **kwargs):
            logging.getLogger("prediction_provenance").error(
                "unreadable sidecar %s: %s", "/x/sidecar.json", "truncated")
            logging.getLogger("some.other.library").warning("a warning")
            logging.getLogger().critical("a root-logger critical")
            return real(*args, **kwargs)

        monkeypatch.setattr(ecs, "verify_state", noisy)
        _run_main(farm, monkeypatch, "--list-outstanding")
        out = capsys.readouterr().out
        assert set(out.split()) <= set(protocol["states"]), out

    def test_the_structured_report_names_the_states_and_their_reasons(
            self, farm, protocol, adapters, unscored, monkeypatch, tmp_path,
            capsys):
        """What the chain actually reads, and why it is worth more than a list of
        names: the reasons are the only diagnosis an unattended run leaves."""
        _write_state(farm, protocol, "B3", "tga")
        dest = tmp_path / "outstanding.json"
        _sidecar_for(farm, "B3").write_text(self.TRUNCATED)
        _run_main(farm, monkeypatch, "--list-outstanding",
                  "--outstanding-report", str(dest))
        captured = capsys.readouterr()
        doc = json.loads(dest.read_text())
        assert doc["outstanding_states"] == list(protocol["states"])
        assert doc["verified_states"] == []
        assert set(doc["scored_states"]) == set(protocol["states"])
        assert doc["mode"] == "list-outstanding"
        assert any("provenance sidecar" in r
                   for r in doc["reasons"]["B3"]), doc["reasons"]
        #: stdout and the file agree, so neither can be the wrong one.
        assert captured.out.split() == doc["outstanding_states"]
        assert "structured answer written to" in captured.err
        assert "provenance sidecar" in captured.err

    def test_a_verified_state_is_not_listed_in_the_report(self, scored,
                                                          protocol,
                                                          monkeypatch,
                                                          tmp_path):
        """The converse: a report that named every state would be safe and
        useless, and would queue three lanes on a complete run."""
        dest = tmp_path / "outstanding.json"
        _run_main(scored, monkeypatch, "--list-outstanding",
                  "--outstanding-report", str(dest))
        doc = json.loads(dest.read_text())
        assert doc["outstanding_states"] == []
        assert doc["verified_states"] == list(protocol["states"])
        assert doc["reasons"] == {}


class TestTheBaseModelIsPinnedToTheFrozenRevision:
    """11C-5R finding 3: the pinned revision was recorded and never used.

    ``ReferenceStateGenerator`` calls ``from_pretrained(model_id)`` with no
    revision, so the repo id resolves through
    ``~/.cache/huggingface/hub/models--<id>/refs/main`` — a MUTABLE pointer.
    A moved ref silently changes the base model under all three states.  It is
    worse than merely loading the wrong weights: ``PredictionFingerprint.build``
    records ``base_model_revision(model_id)``, which reads that same live ref,
    and ``verify_sidecar`` compares the sidecar against an expectation built
    from it — so both sides move together and the sidecar still verifies.
    """

    PIN = "c202236235762e1c871ad0ccb60c8ee5ba337b9a"
    MODEL_ID = "Qwen/Qwen3.5-9B"

    def test_the_freeze_pins_one_revision_and_every_state_agrees(self,
                                                                protocol):
        assert ecs.frozen_base_model_revision(protocol["freeze"]) == self.PIN

    def test_a_freeze_that_pins_no_revision_refuses(self, protocol):
        freeze = dict(protocol["freeze"])
        gen = dict(freeze["generation"])
        gen.pop("base_model_revision")
        with pytest.raises(SystemExit, match="no pinned base model"):
            ecs.frozen_base_model_revision(dict(freeze, generation=gen))

    def test_one_state_disagreeing_about_the_base_model_refuses(self, protocol):
        """Every state is the same base model plus an adapter, so a freeze whose
        checkpoints disagree describes two base models — and then the difference
        between states would not be a difference between adapters."""
        freeze = dict(protocol["freeze"])
        ck = dict(freeze["checkpoints"])
        ck["MG"] = dict(ck["MG"], base_model_revision="f" * 40)
        with pytest.raises(SystemExit, match="two base models"):
            ecs.frozen_base_model_revision(dict(freeze, checkpoints=ck))

    def test_a_moved_cache_ref_refuses(self, monkeypatch):
        monkeypatch.setattr(ecs, "base_model_revision", lambda _: "f" * 40)
        with pytest.raises(SystemExit, match="points at"):
            ecs.pinned_model_source(self.MODEL_ID, self.PIN)

    def test_an_unresolvable_cache_ref_refuses(self, monkeypatch):
        """Not the same as a matching one: an unresolvable ref would be recorded
        in the sidecar as null, so the pin could not be compared at all."""
        monkeypatch.setattr(ecs, "base_model_revision", lambda _: None)
        with pytest.raises(SystemExit, match="unresolvable"):
            ecs.pinned_model_source(self.MODEL_ID, self.PIN)

    def test_the_snapshot_is_requested_at_the_pinned_revision(
            self, monkeypatch, tmp_path):
        import types
        calls = {}

        def fake_snapshot_download(**kwargs):
            calls.update(kwargs)
            return str(tmp_path)

        hub = types.ModuleType("huggingface_hub")
        hub.snapshot_download = fake_snapshot_download
        monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
        monkeypatch.setattr(ecs, "base_model_revision", lambda _: self.PIN)
        got = ecs.pinned_model_source(self.MODEL_ID, self.PIN)
        assert calls["repo_id"] == self.MODEL_ID
        assert calls["revision"] == self.PIN
        #: No network: a confirmation run must not silently substitute a freshly
        #: downloaded snapshot for the one the protocol was frozen over.
        assert calls["local_files_only"] is True
        assert got == str(tmp_path)

    def test_a_pinned_snapshot_that_is_not_a_directory_refuses(
            self, monkeypatch, tmp_path):
        import types
        hub = types.ModuleType("huggingface_hub")
        hub.snapshot_download = lambda **kw: str(tmp_path / "absent")
        monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
        monkeypatch.setattr(ecs, "base_model_revision", lambda _: self.PIN)
        with pytest.raises(SystemExit, match="not a directory"):
            ecs.pinned_model_source(self.MODEL_ID, self.PIN)

    def test_a_snapshot_that_cannot_be_resolved_locally_refuses(self,
                                                               monkeypatch):
        import types

        def boom(**kwargs):
            raise OSError("no such revision in the local cache")

        hub = types.ModuleType("huggingface_hub")
        hub.snapshot_download = boom
        monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
        monkeypatch.setattr(ecs, "base_model_revision", lambda _: self.PIN)
        with pytest.raises(SystemExit, match="no local snapshot"):
            ecs.pinned_model_source(self.MODEL_ID, self.PIN)

    def test_the_analysis_refuses_on_a_moved_ref(self, scored, monkeypatch):
        monkeypatch.setattr(acs, "base_model_revision", lambda _: "f" * 40)
        with pytest.raises(SystemExit, match="cannot be shown to be the "
                                             "frozen one"):
            acs.analyze(scored)

    def test_the_analysis_refuses_on_an_unresolvable_ref(self, scored,
                                                         monkeypatch):
        """An absent ref is not a matching one, and this script WRITES A REPORT
        claiming ``base_model_revision_verified_at_runtime`` — a comparison that
        was skipped cannot make that claim true.

        ``verify_sidecar`` happens to refuse the same state, because with no
        live ref the expectation records null while the sidecar records the pin,
        but it reports a disagreement between two values rather than the fact
        that matters here: the base model cannot be verified on this machine.
        The evaluator's two READ-ONLY modes still tolerate an absent ref, since
        they write nothing and the sidecar comparison carries them.
        """
        monkeypatch.setattr(acs, "base_model_revision", lambda _: None)
        with pytest.raises(SystemExit, match="no local HF cache ref"):
            acs.analyze(scored)

    def test_the_report_binds_the_revision_it_used(self, scored):
        bound = acs.analyze(scored)["inputs_bound"]
        assert bound["base_model_revision"] == self.PIN
        assert bound["base_model_id"] == self.MODEL_ID


class TestThePhotographsAreVerifiedBeforeAnythingIsScored:
    """11C-5R finding 2: the image bytes were never checked, and a missing one
    degrades silently.

    ``verify_image_manifest`` existed and three other scripts called it, but the
    confirmation scorer did not.  Worse, ``ReferenceStateGenerator._render_prompt``
    appends ``{"type": "image"}`` only inside ``if p.exists()``, and only when the
    query's ``image_ids`` match one of ITS OWN association's images — so a
    missing or unresolvable photograph produces a text-only generation that is
    still decoded, still scored and still sealed with a provenance-valid sidecar.
    The evidence would be invalid while every hash check passed.
    """

    def test_the_committed_dataset_verifies(self, farm, protocol, photographs):
        assert ecs.verify_confirmation_images(
            farm, protocol["queries"], protocol["by_assoc"]) == []

    def test_all_the_image_route_queries_actually_resolve(self, protocol):
        """The check is only worth requiring if it currently holds: 864 of the
        1,209 queries are image-route and every one must reach a real byte."""
        image_route = [q for q in protocol["queries"] if q.image_ids]
        assert len(image_route) == 864
        assert len(protocol["queries"]) - len(image_route) == 345

    def test_an_image_id_absent_from_its_own_association_is_reported(
            self, farm, protocol, photographs):
        """The exact case that degrades silently: ``img_ref`` comes back None,
        so no image is appended and nothing says so.

        ``len(problems) == 1`` is the assertion that needs the pool: with a
        photograph missing, the manifest check contributes 402 problems of its
        own and the one this test is about is no longer countable.  The tests
        below it assert ``any(...)`` and so stay portable.
        """
        target = next(q for q in protocol["queries"] if q.image_ids)
        broken = target.model_copy(update={"image_ids": ["no_such_image_id"]})
        problems = ecs.verify_confirmation_images(
            farm, [broken], protocol["by_assoc"])
        assert len(problems) == 1
        assert "match none of" in problems[0]
        assert "text-only" in problems[0]

    def test_an_image_route_query_with_no_image_ids_is_reported(self, farm,
                                                               protocol):
        target = next(q for q in protocol["queries"] if q.image_ids)
        broken = target.model_copy(update={"image_ids": []})
        problems = ecs.verify_confirmation_images(
            farm, [broken], protocol["by_assoc"])
        assert any("names no image_ids" in p for p in problems)

    def test_a_text_route_query_naming_an_image_is_reported(self, farm,
                                                            protocol):
        target = next(q for q in protocol["queries"] if not q.image_ids)
        broken = target.model_copy(update={"image_ids": ["unexpected"]})
        problems = ecs.verify_confirmation_images(
            farm, [broken], protocol["by_assoc"])
        assert any("is a text route" in p for p in problems)

    def test_an_unrecognised_route_is_reported_rather_than_assumed_text(
            self, farm, protocol):
        """A route this script has never seen must not be silently treated as a
        text probe, or a new route would arrive with no image obligation."""
        target = next(q for q in protocol["queries"] if q.image_ids)
        broken = target.model_copy(update={"route": "audio_to_text"})
        problems = ecs.verify_confirmation_images(
            farm, [broken], protocol["by_assoc"])
        assert any("neither a known image route" in p for p in problems)

    def test_a_photograph_absent_on_disk_is_reported(self, farm, protocol,
                                                     monkeypatch):
        monkeypatch.setattr(ecs, "resolve_image_path", lambda *a, **k: None)
        target = next(q for q in protocol["queries"] if q.image_ids)
        problems = ecs.verify_confirmation_images(
            farm, [target], protocol["by_assoc"])
        assert any("absent on disk" in p and "text-only" in p for p in problems)

    def test_a_swapped_photograph_is_reported(self, farm, protocol, monkeypatch):
        """The mutation the manifest's own hash cannot see: the bytes behind a
        correctly spelled path are not the frozen ones."""
        monkeypatch.setattr(
            ecs, "verify_image_manifest",
            lambda *a, **k: [("x.jpg: image bytes differ from the frozen "
                              "manifest")])
        problems = ecs.verify_confirmation_images(
            farm, protocol["queries"], protocol["by_assoc"])
        assert any("image manifest:" in p for p in problems)

    def test_the_analysis_refuses_a_swapped_photograph(self, scored,
                                                       monkeypatch):
        monkeypatch.setattr(
            ecs, "verify_image_manifest",
            lambda *a, **k: ["x.jpg: image bytes differ"])
        with pytest.raises(SystemExit, match="image manifest"):
            acs.analyze(scored)

    def test_the_report_says_the_images_were_re_hashed(self, scored):
        compliance = acs.analyze(scored)["protocol_compliance"]
        assert compliance["image_manifest_verified_at_runtime"] is True
        assert "TEXT-ONLY" in compliance[
            "why_the_photographs_are_checked_per_query_and_not_only_by_hash"]


class TestTheGenerationOrderIsPartOfVerification:
    """11C-5R finding 6: the order record was checked only at assembly.

    ``generate_state`` decides whether to reuse a parquet by calling
    ``verify_state``, and ``verify_state`` did not look at the generation-order
    record — so the reuse path could accept a parquet produced over a different
    query order.  Batched greedy decoding is not bit-stable across batch
    compositions, so the three states would then be compared across two layouts
    while every sidecar verified and nothing refused.
    """

    def _verify(self, farm, protocol, state):
        return ecs.verify_state(state, farm, protocol["freeze"],
                                protocol["queries"], protocol["by_assoc"],
                                protocol["model_id"], protocol["config"])

    def test_a_verifying_sidecar_with_no_order_record_is_not_reusable(
            self, farm, protocol, adapters):
        _write_state(farm, protocol, "B3", "tga")
        try:
            ecs.generation_order_path(farm, "B3").unlink()
            preds, reasons = self._verify(farm, protocol, "B3")
            assert preds is None
            assert any("no predictions_B3.generation_order.json" in r
                       for r in reasons), reasons
        finally:
            ecs.predictions_path(farm, "B3").unlink(missing_ok=True)

    def test_a_wrong_order_hash_is_not_reusable(self, farm, protocol, adapters):
        _write_state(farm, protocol, "B3", "tga", order="0" * 64)
        try:
            preds, reasons = self._verify(farm, protocol, "B3")
            assert preds is None
            assert any("records query order" in r for r in reasons), reasons
            assert any("not bit-stable" in r for r in reasons), reasons
        finally:
            ecs.predictions_path(farm, "B3").unlink(missing_ok=True)
            ecs.generation_order_path(farm, "B3").unlink(missing_ok=True)

    def test_an_order_record_describing_another_generation_is_not_reusable(
            self, farm, protocol, adapters):
        """Every field of the record is compared, not just the order hash: a
        record that names a different state or configuration is not evidence
        about this one."""
        _write_state(farm, protocol, "B3", "tga")
        try:
            opath = ecs.generation_order_path(farm, "B3")
            record = json.loads(opath.read_text())
            opath.write_text(json.dumps(
                dict(record, generation_config=dict(
                    record["generation_config"], batch_size=16))))
            preds, reasons = self._verify(farm, protocol, "B3")
            assert preds is None
            assert any("generation_config" in r for r in reasons), reasons
        finally:
            ecs.predictions_path(farm, "B3").unlink(missing_ok=True)
            ecs.generation_order_path(farm, "B3").unlink(missing_ok=True)

    def test_an_unreadable_order_record_is_not_reusable(self, farm, protocol,
                                                        adapters):
        _write_state(farm, protocol, "B3", "tga")
        try:
            ecs.generation_order_path(farm, "B3").write_text("{not json")
            preds, reasons = self._verify(farm, protocol, "B3")
            assert preds is None
            assert any("unreadable" in r for r in reasons), reasons
        finally:
            ecs.predictions_path(farm, "B3").unlink(missing_ok=True)
            ecs.generation_order_path(farm, "B3").unlink(missing_ok=True)

    def test_the_reuse_path_reaches_the_model_rather_than_reusing_a_wrong_order(
            self, farm, protocol, adapters, monkeypatch):
        """THE regression.  ``generate_state`` asks ``verify_state`` first and
        reuses whatever it returns; with the order unchecked it returned the
        rows and reused them.  A stub generator that refuses to exist proves the
        reuse did NOT happen — under the old code this test would never reach
        the stub, because the parquet would have been accepted.
        """
        _write_state(farm, protocol, "B3", "tga", order="0" * 64)

        class _NoGPU:
            def __init__(self, *args, **kwargs):
                raise RuntimeError("generation was attempted")

        monkeypatch.setattr(ecs, "ReferenceStateGenerator", _NoGPU)
        try:
            with pytest.raises(RuntimeError, match="generation was attempted"):
                ecs.generate_state(
                    "B3", ecs.adapter_for("B3", protocol["freeze"], farm), farm,
                    protocol["freeze"], protocol["queries"],
                    protocol["by_assoc"], protocol["model_id"],
                    "/pinned/snapshot", "cpu", protocol["config"],
                    protocol["order"])
        finally:
            ecs.predictions_path(farm, "B3").unlink(missing_ok=True)
            ecs.generation_order_path(farm, "B3").unlink(missing_ok=True)

    def test_a_state_written_over_the_right_order_is_reusable(self, farm,
                                                              protocol,
                                                              adapters):
        """The converse, so the tests above cannot pass by refusing everything."""
        _write_state(farm, protocol, "B3", "tga")
        try:
            preds, reasons = self._verify(farm, protocol, "B3")
            assert reasons == []
            assert preds is not None and len(preds) == len(protocol["queries"])
        finally:
            ecs.predictions_path(farm, "B3").unlink(missing_ok=True)
            ecs.generation_order_path(farm, "B3").unlink(missing_ok=True)


def _run_main(farm: Path, monkeypatch, *argv: str) -> None:
    """Run the evaluator's ``main()`` against the farm.

    ``main()`` resolves the repository from the process cwd, which under pytest
    is the REAL repository, so ``_find_repo_root`` is pointed at the farm —
    otherwise these tests would generate into, or refuse over, the committed
    dataset instead of the throwaway one.

    It also runs the preflight in every mode, and the preflight re-hashes the
    402 sealed photographs and resolves every image-route query to one.  On a
    checkout that never fetched the pool that refusal arrives BEFORE whatever the
    test asked about, so the guard is here rather than on each test: a test that
    reaches ``main()`` cannot bypass it, and a refusal for the wrong reason is
    reported as a skip that names the missing bytes instead of a failure that
    looks like a broken check.
    """
    _skip_without_photographs(farm)
    monkeypatch.setattr(ecs, "_find_repo_root", lambda _cwd: farm)
    monkeypatch.setattr(
        sys, "argv", ["evaluate_confirmation_split.py", *argv])
    try:
        ecs.main()
    finally:
        #: ``--list-outstanding`` disables logging PROCESS-WIDE, because every
        #: logger in the import graph writes to stdout through ``setup_logger``
        #: and stdout is what that mode used to be read from.  In the real script
        #: that is the rest of the process's life; here it is one call among
        #: thousands, and leaving the interpreter muted would switch off every
        #: later test's logging, including pytest's own capture.
        logging.disable(logging.NOTSET)


class TestTheLaneAsksWhatVerifiesNotWhatExists:
    """11C-5R finding 5: the lane skipped any state with a parquet FILENAME.

    A stale sidecar or a wrong generation-order record left the chain with no
    way forward: the lane skipped the state because the file was there, the gate
    then refused it because the file was invalid, and the chain stopped with the
    offending parquet still in place and no step that would ever rewrite it.
    ``--list-outstanding`` asks ``verify_state`` instead, so "needs work" and
    "needs regeneration" are the same question.
    """

    def test_with_nothing_generated_all_three_are_outstanding(self, farm,
                                                             protocol,
                                                             unscored,
                                                             monkeypatch,
                                                             capsys):
        _run_main(farm, monkeypatch, "--list-outstanding")
        assert capsys.readouterr().out.split() == ["B3", "B0", "MG"]

    def test_a_validly_written_state_is_not_outstanding(self, farm, protocol,
                                                        adapters, unscored,
                                                        monkeypatch, capsys):
        _write_state(farm, protocol, "B3", "tga")
        try:
            _run_main(farm, monkeypatch, "--list-outstanding")
            assert capsys.readouterr().out.split() == ["B0", "MG"]
        finally:
            _clear(farm, "B3")

    def test_a_parquet_with_a_stale_order_record_IS_outstanding(self, farm,
                                                                protocol,
                                                                adapters,
                                                                monkeypatch,
                                                                capsys):
        """THE regression.  The file exists, so a filename test skipped it; it
        does not verify, so it has to be regenerated."""
        _write_state(farm, protocol, "B3", "tga", order="0" * 64)
        try:
            assert ecs.predictions_path(farm, "B3").exists()
            _run_main(farm, monkeypatch, "--list-outstanding")
            assert "B3" in capsys.readouterr().out.split()
        finally:
            _clear(farm, "B3")

    def test_a_parquet_with_a_stale_sidecar_IS_outstanding(self, farm, protocol,
                                                           adapters,
                                                           monkeypatch,
                                                           capsys):
        _write_state(farm, protocol, "B0", "filr")
        try:
            sidecar = _sidecar_for(farm, "B0")
            record = json.loads(sidecar.read_text())
            sidecar.write_text(json.dumps(
                dict(record, num_rows=len(protocol["queries"]) + 1)))
            _run_main(farm, monkeypatch, "--list-outstanding")
            assert "B0" in capsys.readouterr().out.split()
        finally:
            _clear(farm, "B0")

    def test_stdout_is_only_state_names_so_a_human_can_read_it(self, farm,
                                                               protocol,
                                                               monkeypatch,
                                                               capsys):
        """The lane reads a STRUCTURED FILE now, because every logger in the
        process writes to stdout through ``setup_logger`` and one truncated
        sidecar was enough to put an error line where a state name was expected.
        stdout still has to be clean: it is what a person reads, and a mode that
        prints log lines among its answer is unreadable either way."""
        _run_main(farm, monkeypatch, "--list-outstanding")
        captured = capsys.readouterr()
        assert set(captured.out.split()) <= set(protocol["states"])
        assert "still need generation" in captured.err

    def test_nothing_outstanding_exits_zero_and_prints_nothing(self, farm,
                                                               protocol,
                                                               adapters,
                                                               monkeypatch,
                                                               capsys):
        """An empty list is a complete answer, not a failure: a lane that treated
        it as one would never reach the gate on a resumed run."""
        for state, kind in (("B3", "tga"), ("B0", "filr"), ("MG", "tga")):
            _write_state(farm, protocol, state, kind)
        try:
            _run_main(farm, monkeypatch, "--list-outstanding")
            assert capsys.readouterr().out.split() == []
        finally:
            _clear(farm, *protocol["states"])


class TestAShardedLaneSucceedsOnItsOwnState:
    """11C-5R finding 1: ``--state X`` required all three states to be complete.

    Each lane generated one state and then fell through to the global
    completeness check, so whichever finished FIRST exited nonzero merely
    because the other two were still running.  The chain recorded ``gen_rc=1``
    and stopped before its own gate, however healthy that gate would have been —
    the advertised single invocation of the chain could not succeed.
    """

    def _stub_generation(self, monkeypatch, farm, protocol, seen=None):
        def fake_generate(state, adapter_dir, repo_root, freeze, queries,
                          by_assoc, model_id, model_source, device,
                          generation_config, order_sha):
            if seen is not None:
                seen.update(state=state, model_id=model_id,
                            model_source=model_source)
            _write_state(repo_root, protocol, state, "tga")
            return False

        monkeypatch.setattr(ecs, "generate_state", fake_generate)
        monkeypatch.setattr(ecs, "pinned_model_source",
                            lambda model_id, revision: "/pinned/snapshot")

    def test_state_b3_exits_zero_while_the_other_two_are_absent(
            self, farm, protocol, adapters, unscored, monkeypatch, capsys):
        self._stub_generation(monkeypatch, farm, protocol)
        try:
            #: No SystemExit IS the assertion: B0 and MG do not exist, and that
            #: must not fail a lane that was only asked for B3.
            _run_main(farm, monkeypatch, "--state", "B3", "--device", "cpu")
            assert "B3" in capsys.readouterr().out
        finally:
            _clear(farm, "B3")

    def test_the_per_state_exit_disclaims_global_completeness(
            self, farm, protocol, adapters, unscored, monkeypatch, capsys):
        """A lane that succeeded on one state must not be readable as a claim
        that the confirmation is complete."""
        self._stub_generation(monkeypatch, farm, protocol)
        try:
            _run_main(farm, monkeypatch, "--state", "B3", "--device", "cpu")
            out = capsys.readouterr().out
            assert "says nothing about the other states" in out
            assert "--verify-only" in out
        finally:
            _clear(farm, "B3")

    def test_the_generation_loads_the_pinned_snapshot_not_the_repo_id(
            self, farm, protocol, adapters, unscored, monkeypatch):
        """Finding 3 end to end: the model SOURCE handed to the generator is the
        revision-pinned snapshot, while the model ID recorded in the sidecar
        stays the logical name the freeze binds."""
        seen = {}
        self._stub_generation(monkeypatch, farm, protocol, seen)
        try:
            _run_main(farm, monkeypatch, "--state", "B3", "--device", "cpu")
            assert seen["state"] == "B3"
            assert seen["model_source"] == "/pinned/snapshot"
            assert seen["model_id"] == "Qwen/Qwen3.5-9B"
        finally:
            _clear(farm, "B3")

    def test_verify_only_still_refuses_when_a_state_is_missing(self, farm,
                                                              protocol,
                                                              adapters,
                                                              unscored,
                                                              monkeypatch):
        """The global gate keeps its teeth: moving it out of ``--state`` must not
        move it out of existence."""
        _write_state(farm, protocol, "B3", "tga")
        try:
            with pytest.raises(SystemExit, match="2 of 3"):
                _run_main(farm, monkeypatch, "--verify-only")
        finally:
            _clear(farm, "B3")

    def test_a_state_outside_the_frozen_set_still_refuses(self, farm, protocol,
                                                          monkeypatch):
        with pytest.raises(SystemExit, match="not one of the frozen"):
            _run_main(farm, monkeypatch, "--state", "MF")
