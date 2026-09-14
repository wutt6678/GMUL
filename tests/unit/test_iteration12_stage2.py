"""Iteration 12 Stage 2: bounded suppression and the MF-preservation anchor.

Stage 1 swept how hard to rehearse the fit half, Stage 1b replicated the two
near-misses at four seeds, and Stage 1c found the failure on the image route the
frozen floor had never read.  Stage 2 changes the OBJECTIVE: ``B5`` bounds the
gradient ascent on ``fine_target``, ``B6`` pins ``KL(p_MF || p_theta)`` on the
fit-half retained prompts.

These tests hold Stage 2 to what it claims, from committed artifacts and CPU
stubs only -- no adapter, no prediction parquet, no GPU.  They exist BEFORE the
first Stage-2 adapter is trained for a reason that is specific to this study:
``freeze_iter12_stage2.py`` refuses to amend the protocol once any adapter
exists, and ``select_iter12_stage2.py`` refuses to run when the freeze does not
match.  So the moment training starts, a bug found in one of the eight bound
protocol paths can no longer be fixed at all.  This file is the last point at
which one still can be.

What is checked:

* the grid is the incumbent recipe with ONE coordinate changed per row, and the
  frozen Stage-1 validator still governs it -- with exactly two complaint
  classes filtered, pinned by message so a third cannot slip through;
* both training loops are the SAME loop: driven against one stub model on CPU
  with only ``sft``/``gd`` groups they produce bit-identical parameters and
  bit-identical epoch summaries, which is what ``--control`` then confirms on
  the real model -- against a noise floor it MEASURES, by running the frozen
  loop twice, rather than against the filed Stage-1 adapter, whose bytes no
  later process can reproduce;
* the control's gate is pinned STRUCTURALLY, by parsing the script that holds
  it: the hash seed is checked before anything trains, bitwise equality is
  demanded when the measured floor is zero, ``passed`` is never assigned a bare
  ``True``, the filed adapter never feeds it, and ``main`` refuses to train when
  it did not pass.  These exist because an amendment widened what the freeze
  may change to include ``faithfulness_control.*``, and a seal that can be
  widened needs something holding the other side;
* the cap binds exactly and only when it should -- zero gradient below the
  bound, and bit-identical parameters to plain ascent above it;
* the anchor is zero at theta = MF, its gradient there is numerically zero, and
  the logits/labels shift is checked rather than assumed;
* the reference cache round-trips, and refuses to be read when its bytes, its
  version, its alignment or its provenance do not match;
* the freeze binds what the executor reads, and its amendment path cannot be
  reached once an adapter exists;
* the selector refuses the sealed confirmation evidence, including by ``..``
  traversal, and refuses to apply a floor whose anchor does not reproduce the
  value Stage 1c filed.
"""

from __future__ import annotations

import json
import random
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import freeze_iter12_stage2 as fis2
import select_iter12_stage2 as sis2

from granunlearn.evaluation import route_stratified_retention as rt
from granunlearn.training import preservation_anchor as pa
from granunlearn.training import preservation_trainer as pt
from granunlearn.training import stage2_grid as sg
from granunlearn.training import unlearning_trainer as ut
from granunlearn.training.candidate_grid import (
    ITER12_LAM,
    ITER12_REFERENCE_ROW,
    ITER12_REFERENCE_WEIGHT,
    iter12_grid,
)
from granunlearn.training.reference_trainer import ReferenceRecipe

PY = sys.executable
BASIS = REPO_ROOT / sg.BASIS_REPORT
CALIBRATION = REPO_ROOT / sg.CALIBRATION_REPORT
FREEZE = REPO_ROOT / sg.FREEZE_REPORT
ROUTE_STRATIFIED = REPO_ROOT / sg.ROUTE_STRATIFIED_REPORT
CACHE_SIDECAR = REPO_ROOT / sg.reference_cache_dir(REPO_ROOT) / pa.CACHE_SIDECAR


def calibration() -> dict:
    return json.loads(CALIBRATION.read_text())


def freeze() -> dict:
    return json.loads(FREEZE.read_text())


def beta_star() -> float:
    """The anchor strength the calibration derived, read not retyped."""
    return calibration()["grid_rule"]["beta_star"]


def grid() -> list:
    return sg.stage2_grid(beta_star())


def run(script: str, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run([PY, str(REPO_ROOT / "scripts" / script), *args],
                          cwd=REPO_ROOT, capture_output=True, text=True,
                          check=False)


def _importable(name: str) -> bool:
    """Whether ``name`` can be imported, WITHOUT importing it.

    ``find_spec`` rather than a try/except import, because the check has to be
    cheaper than the thing it checks for: CI installs the torch-free closure in
    ``requirements/ci-unit.txt`` and the suite must still COLLECT, so touching
    torch here would turn a skip into an error at collection and report nothing
    else.
    """
    import importlib.util
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


#: The half of this file that drives the two training loops against a stub model
#: needs torch, and the cache half needs it to save and load the tensor.  The
#: other half -- the grid, the objective validator, the path resolvers, the
#: freeze and the selector's guards -- reads committed artifacts and runs
#: anywhere, which is why the file is not skipped as a whole.
TORCH_AVAILABLE = all(_importable(m) for m in ("torch", "transformers", "peft"))

requires_torch = pytest.mark.skipif(
    not TORCH_AVAILABLE,
    reason="this test drives real tensor arithmetic, which needs "
           "torch/transformers/peft; CI installs a torch-free closure, so it "
           "runs where those exist and is reported as skipped where they do "
           "not -- the grid, validator, resolver, freeze and selector tests "
           "beside it still check the protocol from committed files alone")


# ── the stub world ──────────────────────────────────────────────────────
def T():
    """``torch``, imported here rather than at module level.

    ``tests/unit/test_ci_dependency_closure.py`` enforces that: CI installs a
    torch-free closure, so a module-level ``import torch`` in a unit test fails
    at COLLECTION and reports nothing else -- the whole suite looks broken
    instead of the one file that reached for a dependency CI does not have.
    Every function below that needs torch calls this first.
    """
    import torch
    return torch


#: A vocabulary and sequence length big enough that the log-softmax is a real
#: reduction and small enough that a full-vocabulary KL costs nothing.
VOCAB = 29
SEQ = 7
INIT_SEED = 1234

#: Unequal group sizes, so the round-robin interleave has a ragged tail and the
#: accumulation normalisation is exercised on a trailing group SHORTER than the
#: accumulation window -- the branch both loops have to agree on.
GROUP_SIZES = {"fine_target": 5, "target_level": 4, "retain": 6}


class _Out:
    def __init__(self, loss, logits):
        self.loss = loss
        self.logits = logits


def StubLM(seed: int = INIT_SEED):
    """Build a stub causal LM, following HF's shift convention exactly.

    A factory rather than a class, because ``torch.nn.Module`` cannot be named
    as a base class without importing torch at module level -- which is the one
    thing :func:`T` exists to avoid.  Call sites are unchanged: it is only ever
    constructed, never subclassed.

    ``logits[:, :-1]`` predicts ``labels[:, 1:]``, which is the convention
    ``supervised_positions`` assumes when it returns the LOGIT index ``i - 1``
    for a label at ``i``.  Getting this wrong in the stub would let the anchor's
    alignment guard pass against a model that disagrees with the real one, so
    the stub is deliberately built to the same rule rather than to a convenient
    one.
    """
    import torch
    import torch.nn.functional as F

    class _StubLM(torch.nn.Module):
        def __init__(self, seed: int) -> None:
            super().__init__()
            g = torch.Generator().manual_seed(seed)
            self.embed = torch.nn.Embedding(VOCAB, 8)
            self.head = torch.nn.Linear(8, VOCAB)
            with torch.no_grad():
                self.embed.weight.copy_(
                    torch.randn(VOCAB, 8, generator=g) * 0.4)
                self.head.weight.copy_(
                    torch.randn(VOCAB, 8, generator=g) * 0.4)
                self.head.bias.copy_(torch.randn(VOCAB, generator=g) * 0.2)
            self.seen: list[tuple] = []

        def gradient_checkpointing_enable(self) -> None:
            pass

        def enable_input_require_grads(self) -> None:
            pass

        def print_trainable_parameters(self) -> None:
            pass

        def forward(self, input_ids=None, labels=None, **kw):
            self.seen.append(tuple(input_ids.reshape(-1).tolist()))
            logits = self.head(self.embed(input_ids))
            shift_logits = logits[:, :-1, :].reshape(-1, VOCAB).float()
            shift_labels = labels[:, 1:].reshape(-1)
            loss = F.cross_entropy(shift_logits, shift_labels,
                                   ignore_index=-100)
            return _Out(loss, logits)

        def save_pretrained(self, path, *a, **k) -> None:
            path = Path(path)
            path.mkdir(parents=True, exist_ok=True)
            torch.save({n: p.detach().clone()
                        for n, p in self.state_dict().items()},
                       path / "stub_state.pt")

    return _StubLM(seed)



class StubProcessor:
    def __init__(self) -> None:
        #: Not "left", so the branch that reassigns it is exercised too.
        self.tokenizer = SimpleNamespace(padding_side="right")

    def save_pretrained(self, path, *a, **k) -> None:
        Path(path).mkdir(parents=True, exist_ok=True)


def ids_for(example_id: str) -> list[int]:
    rnd = random.Random(example_id)
    return [rnd.randrange(1, VOCAB) for _ in range(SEQ)]


def stub_encode(example, processor, max_length, max_image_pixels):
    """The same encoding both loops see, so a difference is the loop's."""
    torch = T()
    ids = ids_for(example.example_id)
    labels = [-100, -100, -100] + ids[3:]
    return {"input_ids": torch.tensor([ids], dtype=torch.long),
            "labels": torch.tensor([labels], dtype=torch.long),
            "attention_mask": torch.ones(1, SEQ, dtype=torch.long)}


def stub_load(path, repo_root=None):
    name = Path(str(path)).stem
    return [SimpleNamespace(example_id=f"{name}__ex{i}")
            for i in range(GROUP_SIZES[name])]


def stub_recipe(**over) -> ReferenceRecipe:
    base = dict(num_epochs=2, learning_rate=1e-2, gradient_accumulation_steps=4,
                seed=7, bf16=False, gradient_checkpointing=False)
    base.update(over)
    return ReferenceRecipe(**base)


@pytest.fixture
def stubbed(monkeypatch, tmp_path):
    """Patch every heavy dependency BOTH trainers reach, to one shared stub.

    ``_encode_example`` and ``load_state_examples`` are imported at module level
    by each trainer, so they are patched in each namespace; patching only one
    would let the two loops see different data and the comparison would measure
    the patch rather than the code.  ``transformers`` and ``peft`` are imported
    inside the function bodies, so patching the attributes on those modules
    reaches both.
    """
    if not TORCH_AVAILABLE:
        #: Before the imports, not after: in a torch-free CI these two would
        #: raise at the import and the test would be reported as an ERROR
        #: rather than as the skip it is.
        pytest.skip(requires_torch.kwargs["reason"])
    import peft
    import transformers

    for mod in (ut, pt):
        monkeypatch.setattr(mod, "_encode_example", stub_encode)
        monkeypatch.setattr(mod, "load_state_examples", stub_load)
    monkeypatch.setattr(transformers.AutoProcessor, "from_pretrained",
                        staticmethod(lambda *a, **k: StubProcessor()))
    monkeypatch.setattr(transformers.AutoModelForImageTextToText,
                        "from_pretrained",
                        staticmethod(lambda *a, **k: StubLM()))
    monkeypatch.setattr(peft, "get_peft_model", lambda model, config: model)
    monkeypatch.setattr(peft.PeftModel, "from_pretrained",
                        staticmethod(lambda model, *a, **k: model))
    return tmp_path


def saved_state(out_dir: Path) -> dict:
    torch = T()
    return torch.load(out_dir / "adapters" / "stub_state.pt", weights_only=True)


def initial_state() -> dict:
    return {n: p.detach().clone() for n, p in StubLM().state_dict().items()}


def states_equal(a: dict, b: dict) -> bool:
    torch = T()
    return set(a) == set(b) and all(torch.equal(a[k], b[k]) for k in a)


THREE_GROUPS = (("fine_target", "gd", ITER12_LAM),
                ("target_level", "sft", 1.0),
                ("retain", "sft", ITER12_REFERENCE_WEIGHT))


def incumbent_objective(tmp_path: Path, mode_for=None):
    """The incumbent's three components, as both loops express them."""
    frozen, stage2 = [], []
    for name, mode, weight in THREE_GROUPS:
        m = mode if mode_for is None else mode_for(name, mode)
        path = tmp_path / f"{name}.jsonl"
        frozen.append(ut.GroupSpec(name, path, m, weight))
        cap = 1e-9 if m == pt.CAPPED_MODE else None
        stage2.append(pt.PreservationGroupSpec(name, path, m, weight, cap=cap))
    return frozen, stage2


# ── the grid ────────────────────────────────────────────────────────────

def test_grid_is_b0_plus_seven_trained_rows():
    g = grid()
    assert len(g) == 8
    assert [c.candidate_id for c in g][0] == "B0"
    assert g[0].noop and not any(c.noop for c in g[1:])
    assert len(sg.trained_rows(g)) == 7
    ids = [c.candidate_id for c in g]
    assert len(ids) == len(set(ids))


def test_grid_is_the_four_caps_the_decoupling_row_and_two_anchor_rows():
    ids = [c.candidate_id for c in grid()]
    assert ids[1:5] == [sg.bounded_id(c, sg.STAGE2_LR, sg.STAGE2_EPOCHS)
                        for c in sg.STAGE2_CAPS]
    assert ids[5] == sg.bounded_id(sg.STAGE2_DECOUPLING_CAP, sg.STAGE2_LR,
                                   sg.STAGE2_DECOUPLING_EPOCHS)
    assert ids[6].startswith("B6_beta") and not ids[6].startswith("B6R")
    assert ids[7].startswith("B6R_beta")


def test_the_frozen_stage1_validator_governs_the_grid_with_two_pinned_gaps():
    """The residual is empty and the filtered list is EXACTLY the known gaps.

    ``candidate_grid.validate_grid`` is hash-bound and cannot learn Stage 2's
    method labels or modes, so those two complaint classes are filtered.  A
    filter that quietly swallowed a real structural complaint would show up as a
    filtered message that is neither of the two, which is what the second
    assertion pins.
    """
    errors, filtered = sg.validate_stage2_grid(grid())
    assert errors == []
    assert filtered, "the frozen validator should still object to B5/B6"
    for msg in filtered:
        assert any(gap in msg for gap in sg.FROZEN_VALIDATOR_GAP), msg
    #: 5 capped rows object on both method and mode; B6 on both; B6R on method
    #: only, because the hybrid keeps the frozen ``sft`` mode and spends the
    #: anchor through ``anchor_weight`` instead of a new mode.
    assert len(filtered) == 13
    assert sum("B6R" in m for m in filtered) == 1


def test_the_filter_is_selective_rather_than_a_general_mute():
    """Why this needs a BROKEN grid to mean anything.

    On a well-formed grid ``validate_grid`` raises only the two known gaps, so a
    filter that routed EVERY complaint to ``filtered`` would return exactly the
    same pair and no assertion on the real grid could tell the difference -- a
    mutation run found precisely that.  Selectivity is observable only where a
    real structural defect exists to be swallowed, so these grids carry one each:
    a duplicated no-op row, a zero weight, and an override outside the
    swept-knob allowlist.  None of the three is about a method label or a mode,
    and all three must come back as errors.
    """
    import dataclasses

    g = grid()
    errors, filtered = sg.validate_stage2_grid(g)
    assert errors == [] and len(filtered) == 13

    #: A second no-op row: a duplicate id AND two B0 candidates.
    errors, filtered = sg.validate_stage2_grid(
        list(g) + [sg.Stage2CandidateSpec("B0", "B0", noop=True)])
    assert any("duplicate candidate ids" in e for e in errors)
    assert any("exactly one B0" in e for e in errors)
    assert len(filtered) == 13

    #: A zero weight on the capped row.  The weight complaint must survive the
    #: filter even though the SAME row's method and mode complaints do not --
    #: which is the whole distinction the filter is supposed to preserve.
    row = g[1]
    zeroed = dataclasses.replace(
        row, groups=tuple(
            dataclasses.replace(gr, weight=0.0) if gr.name == "fine_target"
            else gr for gr in row.groups))
    errors, filtered = sg.validate_stage2_grid([g[0], zeroed])
    assert any("weight must be > 0" in e for e in errors)
    assert not any("unknown method" in e for e in errors)
    assert filtered == [f"{row.candidate_id}: unknown method B5",
                        f"{row.candidate_id}: bad mode gd_capped"]

    #: An override the frozen recipe does not accept.
    widened = dataclasses.replace(row, overrides=dict(row.overrides,
                                                      batch_hack=4))
    errors, _filtered = sg.validate_stage2_grid([g[0], widened])
    assert any("swept-knob allowlist" in e for e in errors)


def test_every_b5_row_is_the_incumbent_with_one_coordinate_changed():
    incumbent = {c.candidate_id: c for c in iter12_grid()}[ITER12_REFERENCE_ROW]
    want_groups = {g.name: g for g in incumbent.groups}
    for c in [c for c in grid() if c.method == sg.METHOD_BOUNDED]:
        assert set(c.overrides) == {"learning_rate", "num_epochs"}
        assert c.overrides["learning_rate"] == incumbent.overrides["learning_rate"]
        assert {g.name for g in c.groups} == set(want_groups)
        for g in c.groups:
            ref = want_groups[g.name]
            assert g.weight == ref.weight, c.candidate_id
            if g.name == "fine_target":
                assert g.mode == pt.CAPPED_MODE and g.cap == c.cap
                assert ref.mode == "gd"
            else:
                assert g.mode == ref.mode and g.cap is None


def test_the_decoupling_row_differs_from_its_cap_sibling_only_in_budget():
    g = {c.candidate_id: c for c in grid()}
    short = g[sg.bounded_id(sg.STAGE2_DECOUPLING_CAP, sg.STAGE2_LR,
                            sg.STAGE2_EPOCHS)]
    long = g[sg.bounded_id(sg.STAGE2_DECOUPLING_CAP, sg.STAGE2_LR,
                           sg.STAGE2_DECOUPLING_EPOCHS)]
    assert short.cap == long.cap == sg.STAGE2_DECOUPLING_CAP
    assert [x.describe() for x in short.groups] == \
           [x.describe() for x in long.groups]
    assert short.overrides["num_epochs"] != long.overrides["num_epochs"]


def test_the_hybrid_row_keeps_three_groups_so_the_schedule_does_not_move():
    """``anchor_weight`` rather than a second entry on the same file.

    Two group entries pointing at ``retain.jsonl`` would lengthen every epoch by
    183 micro-batches and quietly change the schedule under test, which is the
    defect this row exists to avoid.
    """
    hybrid = [c for c in grid() if c.candidate_id.startswith("B6R")][0]
    alone = [c for c in grid()
             if c.candidate_id.startswith("B6_beta")][0]
    assert len(hybrid.groups) == len(alone.groups) == 3
    retain = [g for g in hybrid.groups if g.name == "retain"][0]
    assert retain.mode == "sft"
    assert retain.weight == ITER12_REFERENCE_WEIGHT
    assert retain.anchor_weight == hybrid.beta
    #: The anchor-alone row REPLACES the supervised term, so its weight IS beta.
    solo = [g for g in alone.groups if g.name == "retain"][0]
    assert solo.mode == pa.ANCHOR_MODE and solo.weight == alone.beta
    assert solo.anchor_weight is None


def test_beta_is_read_from_the_calibration_and_rounded_once():
    b = round(beta_star(), 4)
    assert b > 0
    for c in grid():
        if c.method == sg.METHOD_ANCHOR:
            assert c.beta == b
            assert f"beta{b}" in c.candidate_id
        else:
            assert c.beta is None


def test_a_non_positive_beta_star_is_refused():
    for bad in (0.0, -1.0):
        with pytest.raises(ValueError, match="positive"):
            sg.stage2_grid(bad)


def test_the_cap_sweep_spans_the_oracle_without_reading_it():
    """MG's own level is reported, never used to pick the range.

    The training procedure never reads MG, so the caps have to bracket its level
    from both sides on their own.  This is the check that the claim in the
    module docstring is true of the constants rather than merely asserted.
    """
    mg = calibration()["measurement"]["adapters"]["MG"]["nll_by_group"]
    level = mg["fine_target"]
    assert level == pytest.approx(sg.MG_FINE_TARGET_NLL_REPORTED, abs=5e-5)
    caps = set(sg.STAGE2_CAPS) | {sg.STAGE2_DECOUPLING_CAP}
    assert min(caps) < level < max(caps)


# ── the objective validator ─────────────────────────────────────────────

def valid_groups(**over):
    base = [pt.PreservationGroupSpec("fine_target", "f.jsonl", "gd",
                                     ITER12_LAM),
            pt.PreservationGroupSpec("target_level", "t.jsonl", "sft", 1.0),
            pt.PreservationGroupSpec("retain", "r.jsonl", "sft",
                                     ITER12_REFERENCE_WEIGHT)]
    if over:
        base = [over.get(g.name, g) for g in base]
    return base


def test_the_incumbent_objective_is_valid_and_every_grid_row_is_too():
    assert pt.validate_stage2_groups(valid_groups()) == []
    for c in grid():
        if not c.noop:
            assert pt.validate_stage2_groups(list(c.groups)) == [], \
                c.candidate_id


@pytest.mark.parametrize("groups,match", [
    ([pt.PreservationGroupSpec("retain", "r", pt.CAPPED_MODE, 1.0, cap=2.0)],
     "belongs on a target group"),
    ([pt.PreservationGroupSpec("fine_target", "f", "sft", 1.0, cap=2.0)],
     "meaningless in mode"),
    ([pt.PreservationGroupSpec("fine_target", "f", pt.CAPPED_MODE, 1.0)],
     "needs a cap"),
    ([pt.PreservationGroupSpec("fine_target", "f", pt.CAPPED_MODE, 1.0,
                               cap=0.0)], "must be > 0 nats"),
    ([pt.PreservationGroupSpec("fine_target", "f", pa.ANCHOR_MODE, 1.0)],
     "only the 'retain' group"),
    ([pt.PreservationGroupSpec("target_level", "t", pa.ANCHOR_MODE, 1.0)],
     "only the 'retain' group"),
    ([pt.PreservationGroupSpec("retain", "r", pa.ANCHOR_MODE, 1.0,
                               anchor_weight=2.0)], "apply it twice"),
    ([pt.PreservationGroupSpec("retain", "r", "gd", 1.0, anchor_weight=2.0)],
     "augments an sft term"),
    ([pt.PreservationGroupSpec("retain", "r", "sft", 1.0, anchor_weight=0.0)],
     "anchor_weight must be > 0"),
    ([pt.PreservationGroupSpec("fine_target", "f", "nope", 1.0)],
     "unknown mode"),
    ([pt.PreservationGroupSpec("retain", "r", "sft", 0.0)], "weight must be"),
    ([pt.PreservationGroupSpec("retain", "r", "sft", 1.0),
      pt.PreservationGroupSpec("retain", "r2", "sft", 1.0)], "repeated group"),
    ([pt.PreservationGroupSpec("fine_target", "f", pt.CAPPED_MODE, 1.0,
                               cap=1.0),
      pt.PreservationGroupSpec("target_level", "t", pt.CAPPED_MODE, 1.0,
                               cap=1.0)], "at most one capped"),
    ([pt.PreservationGroupSpec("target_level", "t", "sft", 1.0,
                               anchor_weight=2.0)],
     "only the 'retain' group"),
])
def test_an_invalid_objective_is_refused_with_the_reason_named(groups, match):
    assert any(match in e for e in pt.validate_stage2_groups(groups)), \
        pt.validate_stage2_groups(groups)


def test_two_anchored_groups_are_refused_even_when_both_are_retain_shaped():
    groups = [pt.PreservationGroupSpec("retain", "r", pa.ANCHOR_MODE, 1.0),
              pt.PreservationGroupSpec("retain2", "r2", "sft", 1.0,
                                       anchor_weight=1.0)]
    assert any("at most one anchored" in e
               for e in pt.validate_stage2_groups(groups))



# ── the anchor's mathematics ────────────────────────────────────────────

@requires_torch
def test_supervised_positions_returns_the_logit_index_not_the_label_index():
    """``labels[i]`` is predicted by ``logits[i-1]``.

    Returning ``i`` instead would score the distribution that predicts the NEXT
    token, one position late, and every KL in the study would be measured on the
    wrong row while still looking plausible.
    """
    torch = T()
    labels = torch.tensor([[-100, -100, -100, 5, 6, 7, 8]])
    assert pa.supervised_positions(labels) == [(2, 5), (3, 6), (4, 7), (5, 8)]


@requires_torch
def test_a_supervised_label_at_position_zero_is_refused():
    torch = T()
    with pytest.raises(ValueError, match="no logit predicts it"):
        pa.supervised_positions(torch.tensor([[7, -100, 3]]))


@requires_torch
def test_supervised_logprobs_agrees_with_the_models_own_loss():
    torch = T()
    F = torch.nn.functional
    torch.manual_seed(0)
    logits = torch.randn(1, SEQ, VOCAB)
    labels = torch.tensor([[-100, -100, -100] + ids_for("x")[3:]])
    logp, targets = pa.supervised_logprobs(logits, labels)
    assert list(logp.shape) == [4, VOCAB] and logp.dtype == torch.float32
    model_loss = F.cross_entropy(logits[:, :-1].reshape(-1, VOCAB).float(),
                                 labels[:, 1:].reshape(-1),
                                 ignore_index=-100).item()
    #: The guard the anchor runs at every training step, on a model built to the
    #: same convention: it has to pass, and by far more than the tolerance.
    assert pa.assert_nll_matches_model(logp, targets, model_loss, "t") < 1e-6


@requires_torch
def test_the_nll_guard_catches_an_off_by_one_shift():
    """The mutation this guard exists for, applied directly.

    Indexing ``logits[i]`` against ``labels[i]`` is the off-by-one; it produces
    a perfectly plausible small number and no error, so the guard is the only
    thing that would notice.
    """
    torch = T()
    F = torch.nn.functional
    torch.manual_seed(0)
    logits = torch.randn(1, SEQ, VOCAB)
    labels = torch.tensor([[-100, -100, -100] + ids_for("x")[3:]])
    logp, targets = pa.supervised_logprobs(logits, labels)
    unshifted = F.cross_entropy(logits.reshape(-1, VOCAB).float(),
                                labels.reshape(-1), ignore_index=-100).item()
    with pytest.raises(AssertionError, match="shift is wrong"):
        pa.assert_nll_matches_model(logp, targets, unshifted, "t")


@requires_torch
def test_the_anchor_is_exactly_zero_at_the_reference():
    torch = T()
    logp = torch.log_softmax(torch.randn(4, VOCAB), dim=-1)
    assert pa.anchor_loss(logp, logp).item() == 0.0
    assert pa.forward_kl(logp, logp).abs().max().item() == 0.0


@requires_torch
def test_the_anchor_gradient_vanishes_at_the_reference_and_not_away_from_it():
    """A trust region around the initialisation, not a second objective.

    Analytically ``d KL / d z_i = -(p_ref_i - p_theta_i)``, which is zero at
    theta = MF.  The bound is a float-noise bound rather than an exact zero
    because ``sum(p_ref)`` is 1.0 only to within rounding, and AdamW would
    amplify an exact-but-tiny gradient to a full step -- which is why the
    comparison against a perturbed model is the informative half.
    """
    torch = T()
    torch.manual_seed(0)
    #: ``ref`` must be the log-softmax OF ``z`` -- theta = MF means the current
    #: model is the model the cache was built from.  Drawing a second independent
    #: tensor would compare two different distributions and measure a gradient
    #: that has no reason to vanish.
    z = torch.randn(4, VOCAB, requires_grad=True)
    ref = torch.log_softmax(z.detach(), dim=-1)
    pa.anchor_loss(ref, torch.log_softmax(z, dim=-1)).backward()
    at_reference = z.grad.abs().max().item()
    assert at_reference < 1e-6

    other = torch.randn(4, VOCAB) * 2.0
    z2 = other.clone().requires_grad_(True)
    pa.anchor_loss(ref, torch.log_softmax(z2, dim=-1)).backward()
    assert z2.grad.abs().max().item() > 100 * at_reference


@requires_torch
def test_the_reference_operand_is_a_constant_so_the_gradient_only_flows_one_way():
    torch = T()
    ref = torch.randn(3, VOCAB)
    cur = torch.randn(3, VOCAB, requires_grad=True)
    pa.anchor_loss(ref, torch.log_softmax(cur, dim=-1)).backward()
    assert cur.grad is not None and not ref.requires_grad


# ── the two loops are one loop ──────────────────────────────────────────

def test_the_new_loop_reproduces_the_frozen_one_bit_for_bit(stubbed):
    """The CPU half of the faithfulness control.

    ``--control`` compares real adapter bytes on a GPU; this compares the same
    two loops on a stub, so a divergence is caught without one.  Both the
    parameters AND the epoch summaries must match: the summaries are what the
    basis report reads ``avg_loss_fine_target`` from, so a loop that trained
    identically but booked differently would still break the calibration.
    """
    tmp = stubbed
    frozen, stage2 = incumbent_objective(tmp)
    recipe = stub_recipe()

    a = tmp / "frozen"
    b = tmp / "stage2"
    sa = ut.train_unlearning("frozen", frozen, a, device="cpu", recipe=recipe)
    sb = pt.train_with_preservation("stage2", stage2, b, device="cpu",
                                    recipe=recipe)

    assert states_equal(saved_state(a), saved_state(b))
    assert sa["epochs"] == sb["epochs"]
    assert sa["num_optimizer_steps"] == sb["num_optimizer_steps"] > 0
    #: With no capped and no anchored group the new loop books nothing extra, so
    #: the epoch entries are the same KEYS and not merely equal values.
    assert set(sb["epochs"][0]) == set(sa["epochs"][0])


def test_the_stub_actually_trains_so_the_comparison_is_not_vacuous(stubbed):
    """A guard on the guard: equal parameters mean nothing if none moved."""
    tmp = stubbed
    frozen, stage2 = incumbent_objective(tmp)
    pt.train_with_preservation("x", stage2, tmp / "moved", device="cpu",
                               recipe=stub_recipe())
    assert not states_equal(saved_state(tmp / "moved"), initial_state())


def test_the_ragged_accumulation_tail_is_exercised(stubbed):
    """15 micro-batches at accum=4 -> a trailing group of 3, not 4.

    Both loops normalise by the ACTUAL trailing group size.  If the stub stream
    divided evenly the branch would never run and the equivalence above would
    have proved less than it claims.
    """
    total = sum(GROUP_SIZES.values())
    accum = stub_recipe().gradient_accumulation_steps
    assert total % accum != 0
    steps = total // accum + 1
    tmp = stubbed
    _frozen, stage2 = incumbent_objective(tmp)
    s = pt.train_with_preservation("x", stage2, tmp / "tail", device="cpu",
                                   recipe=stub_recipe(num_epochs=1))
    assert s["num_optimizer_steps"] == steps


# ── the cap ─────────────────────────────────────────────────────────────

def capped_objective(tmp_path: Path, cap: float):
    return [pt.PreservationGroupSpec("fine_target", tmp_path / "fine_target.jsonl",
                                     pt.CAPPED_MODE, ITER12_LAM, cap=cap)]


def test_a_binding_cap_produces_exactly_zero_parameter_change(stubbed):
    """``clamp`` passes gradient 0 above the bound, so nothing moves at all.

    ``weight_decay`` is 0.0 by default in ``ReferenceRecipe``; with a non-zero
    decay AdamW would shrink the parameters even on a zero gradient, which is
    why the recipe is left at its default rather than set here -- the claim is
    about the default recipe the grid actually trains under.
    """
    tmp = stubbed
    s = pt.train_with_preservation(
        "capped", capped_objective(tmp, 1e-9), tmp / "capped", device="cpu",
        recipe=stub_recipe(num_epochs=1))
    assert states_equal(saved_state(tmp / "capped"), initial_state())
    assert s["epochs"][0][f"frac_at_cap_fine_target"] == 1.0


def test_a_cap_above_the_loss_is_bit_identical_to_plain_ascent(stubbed):
    """The cap changes nothing when it does not bind -- so a difference between
    a ``B5`` row and the incumbent is attributable to the binding, not to the
    clamp having been in the graph."""
    tmp = stubbed
    recipe = stub_recipe(num_epochs=1)
    plain = [pt.PreservationGroupSpec("fine_target",
                                      tmp / "fine_target.jsonl", "gd",
                                      ITER12_LAM)]
    pt.train_with_preservation("plain", plain, tmp / "plain", device="cpu",
                               recipe=recipe)
    s = pt.train_with_preservation(
        "loose", capped_objective(tmp, 1e9), tmp / "loose", device="cpu",
        recipe=recipe)
    assert states_equal(saved_state(tmp / "plain"), saved_state(tmp / "loose"))
    assert s["epochs"][0]["frac_at_cap_fine_target"] == 0.0
    assert not states_equal(saved_state(tmp / "loose"), initial_state())


def test_a_cap_that_binds_partway_reports_the_fraction_that_bound(stubbed):
    """Neither extreme: a cap that never binds is the incumbent with extra
    bookkeeping, and one that binds from step 0 was set below where the model
    started.  ``frac_at_cap`` is what says which happened."""
    torch = T()
    tmp = stubbed
    losses = []
    for name in [f"fine_target__ex{i}" for i in range(GROUP_SIZES["fine_target"])]:
        enc = stub_encode(SimpleNamespace(example_id=name), None, None, None)
        with torch.no_grad():
            losses.append(StubLM()(input_ids=enc["input_ids"],
                                   labels=enc["labels"]).loss.item())
    mid = sorted(losses)[len(losses) // 2]
    s = pt.train_with_preservation(
        "mid", capped_objective(tmp, mid), tmp / "mid", device="cpu",
        recipe=stub_recipe(num_epochs=1))
    frac = s["epochs"][0]["frac_at_cap_fine_target"]
    assert 0.0 < frac < 1.0


def test_the_uncapped_nll_is_still_booked_under_the_stage1_key(stubbed):
    """``avg_loss_fine_target`` is the coordinate the calibration found governs
    retention damage, and Stage 1 watched it under this exact key.  Recording
    the CAPPED value there instead would make the two studies' trajectories
    incomparable on the one quantity that matters.

    So the two booked quantities have to disagree in the direction the cap
    implies: ``avg_loss_*`` carries the raw NLL, unbounded, while
    ``avg_objective`` carries what was actually optimised, which the cap bounds.
    A fully binding cap drives the second to zero and leaves the first large --
    that asymmetry is the whole content of the mechanism.
    """
    tmp = stubbed
    cap = 1e-9
    s = pt.train_with_preservation(
        "capped", capped_objective(tmp, cap), tmp / "capped", device="cpu",
        recipe=stub_recipe(num_epochs=1))
    epoch = s["epochs"][0]
    assert epoch["avg_loss_fine_target"] > cap
    assert abs(epoch["avg_objective"]) <= ITER12_LAM * cap + 1e-4


# ── the reference cache ─────────────────────────────────────────────────

def build_stub_cache(directory: Path, example_ids, vocab: int = VOCAB):
    """A real cache on disk, built from the stub at its initial weights.

    Built through ``save`` and read back through ``load``, so the round-trip,
    the tensor hash and the row alignment are all exercised rather than
    bypassed by constructing the object in memory.
    """
    torch = T()
    model = StubLM()
    logps, targets, rows, start = [], [], [], 0
    for eid in example_ids:
        enc = stub_encode(SimpleNamespace(example_id=eid), None, None, None)
        with torch.no_grad():
            out = model(input_ids=enc["input_ids"], labels=enc["labels"])
        lp, tg = pa.supervised_logprobs(out.logits, enc["labels"])
        logps.append(lp)
        targets.append(tg)
        rows.append({"example_id": eid, "association_id": eid, "entity_id": eid,
                     "start": start, "stop": start + lp.shape[0],
                     "targets": tg.tolist()})
        start += lp.shape[0]
    sidecar = {"cache_version": pa.CACHE_VERSION, "num_positions": start,
               "vocab_size": vocab, "num_examples": len(example_ids),
               "kl_direction": "KL(p_MF || p_theta)",
               "mf_adapter_sha256": "0" * 64,
               "base_model_revision": "stub"}
    cache = pa.ReferenceCache.from_parts(sidecar, torch.cat(logps),
                                         torch.cat(targets), rows)
    cache.save(directory)
    return cache


@requires_torch
def test_the_cache_round_trips_through_disk(tmp_path):
    torch = T()
    ids = [f"retain__ex{i}" for i in range(3)]
    built = build_stub_cache(tmp_path / "cache", ids)
    loaded = pa.ReferenceCache.load(tmp_path / "cache")
    assert set(loaded.rows) == set(ids)
    assert torch.equal(loaded.logp, built.logp)
    assert torch.equal(loaded.targets, built.targets)
    for eid in ids:
        a, b = built.slice_for(eid), loaded.slice_for(eid)
        assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])
    #: What the trainer records in its summary has to survive the trip.
    for key in ("kl_direction", "mf_adapter_sha256", "base_model_revision",
                "num_examples", "num_positions", "vocab_size", "tensor_sha256"):
        assert key in loaded.sidecar


@requires_torch
def test_a_cache_whose_tensor_moved_is_refused(tmp_path):
    build_stub_cache(tmp_path / "cache", ["retain__ex0"])
    blob = tmp_path / "cache" / pa.CACHE_TENSOR
    raw = bytearray(blob.read_bytes())
    raw[-1] ^= 0xFF
    blob.write_bytes(bytes(raw))
    with pytest.raises(ValueError, match="do not match the MF adapter"):
        pa.ReferenceCache.load(tmp_path / "cache")


@requires_torch
def test_a_cache_from_another_layout_version_is_refused(tmp_path):
    build_stub_cache(tmp_path / "cache", ["retain__ex0"])
    p = tmp_path / "cache" / pa.CACHE_SIDECAR
    s = json.loads(p.read_text())
    s["cache_version"] = pa.CACHE_VERSION + 1
    p.write_text(json.dumps(s))
    with pytest.raises(ValueError, match="rebuilt, not reinterpreted"):
        pa.ReferenceCache.load(tmp_path / "cache")


@requires_torch
def test_a_cache_whose_shape_disagrees_with_its_sidecar_is_refused(tmp_path):
    build_stub_cache(tmp_path / "cache", ["retain__ex0"])
    p = tmp_path / "cache" / pa.CACHE_SIDECAR
    s = json.loads(p.read_text())
    s["num_positions"] = s["num_positions"] + 1
    p.write_text(json.dumps(s))
    #: The tensor hash is recomputed into the sidecar's favour, so the shape
    #: check is what fires rather than the hash check.
    s = json.loads(p.read_text())
    s["tensor_sha256"] = pa.sha256_file(tmp_path / "cache" / pa.CACHE_TENSOR)
    p.write_text(json.dumps(s))
    with pytest.raises(ValueError, match="holds"):
        pa.ReferenceCache.load(tmp_path / "cache")


@requires_torch
def test_an_absent_cache_names_the_builder(tmp_path):
    with pytest.raises(FileNotFoundError, match="build_mf_reference_logprobs"):
        pa.ReferenceCache.load(tmp_path / "nope")


@requires_torch
def test_an_unknown_example_raises_rather_than_returning_a_neighbour(tmp_path):
    d = tmp_path / "cache"
    build_stub_cache(d, ["retain__ex0", "retain__ex1"])
    cache = pa.ReferenceCache.load(d)
    with pytest.raises(KeyError, match="no preserved distribution"):
        cache.row("retain__ex2")


@requires_torch
def test_moved_supervised_tokens_are_refused(tmp_path):
    """The alignment guard: anchoring position ``k`` of one encoding against
    position ``k`` of another compares two different distributions and still
    produces a plausible small KL."""
    d = tmp_path / "cache"
    build_stub_cache(d, ["retain__ex0"])
    cache = pa.ReferenceCache.load(d)
    _logp, targets = cache.slice_for("retain__ex0")
    assert cache.assert_aligned("retain__ex0", targets) == len(targets)
    with pytest.raises(AssertionError, match="no longer row-aligned"):
        cache.assert_aligned("retain__ex0", targets.flip(0))


@requires_torch
def test_the_cache_provenance_check_compares_the_adapter_it_came_from(tmp_path):
    d = tmp_path / "cache"
    build_stub_cache(d, ["retain__ex0"])
    cache = pa.ReferenceCache.load(d)

    good = tmp_path / "mf"
    good.mkdir()
    (good / "adapter_model.safetensors").write_bytes(b"weights")
    sidecar = dict(cache.sidecar, mf_adapter_sha256=pa.adapter_digest(good))
    cache2 = pa.ReferenceCache(sidecar, cache.logp, cache.targets, cache.rows)
    cache2.assert_built_from(good)

    other = tmp_path / "other"
    other.mkdir()
    (other / "adapter_model.safetensors").write_bytes(b"different")
    with pytest.raises(AssertionError, match="preserves a different function"):
        cache2.assert_built_from(other)
    with pytest.raises(AssertionError, match="base_model_revision"):
        cache2.assert_built_from(good, base_model_revision="not-the-stub")


def test_the_adapter_digest_is_order_independent_but_name_sensitive(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    for d in (a, b):
        d.mkdir()
    (a / "one.bin").write_bytes(b"111")
    (a / "two.json").write_bytes(b"222")
    (b / "two.json").write_bytes(b"222")
    (b / "one.bin").write_bytes(b"111")
    assert pa.adapter_digest(a) == pa.adapter_digest(b)

    c = tmp_path / "c"
    c.mkdir()
    (c / "renamed.bin").write_bytes(b"111")
    (c / "two.json").write_bytes(b"222")
    assert pa.adapter_digest(c) != pa.adapter_digest(a)

    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(FileNotFoundError):
        pa.adapter_digest(empty)


# ── the anchor path end to end, on the stub ─────────────────────────────

def test_an_anchored_run_starts_at_exactly_zero_kl(stubbed, monkeypatch):
    """``theta = MF`` at step 0, so the KL there is exactly 0.0.

    This is the property the whole mechanism rests on -- that the anchor is a
    trust region and not a second supervised objective -- measured through the
    trainer rather than asserted about the loss function.

    One example per epoch, so the single forward pass happens at theta = MF.  A
    six-example epoch cannot show this: its first optimizer step moves the model,
    and every example after that is legitimately at a small nonzero KL.
    """
    tmp = stubbed
    monkeypatch.setitem(GROUP_SIZES, "retain", 1)
    build_stub_cache(tmp / "cache", ["retain__ex0"])
    groups = [pt.PreservationGroupSpec("retain", tmp / "retain.jsonl",
                                       pa.ANCHOR_MODE, 13.642)]
    s = pt.train_with_preservation("anchored", groups, tmp / "out",
                                   device="cpu", cache_dir=tmp / "cache",
                                   recipe=stub_recipe(num_epochs=1))
    assert s["num_optimizer_steps"] == 1
    assert s["epochs"][0]["avg_kl_retain"] == 0.0
    assert s["preservation_anchor"]["anchor_weight"] == {"retain": 13.642}
    assert s["preservation_anchor"]["anchor_alongside_supervised_term"] == \
        {"retain": False}
    assert s["preservation_anchor"]["nll_agreement_max"] < \
        pa.NLL_AGREEMENT_TOLERANCE


def test_the_anchor_holds_the_model_near_mf_across_a_whole_epoch(stubbed):
    """The same run with the group's real size: the KL is no longer exactly
    zero, because the first step moved theta, but it stays orders of magnitude
    below the drift the Stage-1 candidates reached (0.063 to 0.849 nats)."""
    tmp = stubbed
    ids = [f"retain__ex{i}" for i in range(GROUP_SIZES["retain"])]
    build_stub_cache(tmp / "cache", ids)
    groups = [pt.PreservationGroupSpec("retain", tmp / "retain.jsonl",
                                       pa.ANCHOR_MODE, 13.642)]
    s = pt.train_with_preservation("anchored", groups, tmp / "out",
                                   device="cpu", cache_dir=tmp / "cache",
                                   recipe=stub_recipe(num_epochs=1))
    kl = s["epochs"][0]["avg_kl_retain"]
    assert 0.0 < kl < 1e-5
    assert kl < 0.063 / 1000


def test_the_hybrid_reports_its_anchor_strength_and_not_its_replay_weight(stubbed):
    """Recording ``weight`` for both anchor paths would misreport the hybrid's
    anchor strength as 1.0, which is the replay weight and not beta."""
    tmp = stubbed
    ids = [f"retain__ex{i}" for i in range(GROUP_SIZES["retain"])]
    build_stub_cache(tmp / "cache", ids)
    groups = [pt.PreservationGroupSpec("fine_target", tmp / "fine_target.jsonl",
                                       "gd", ITER12_LAM),
              pt.PreservationGroupSpec("retain", tmp / "retain.jsonl", "sft",
                                       ITER12_REFERENCE_WEIGHT,
                                       anchor_weight=13.642)]
    s = pt.train_with_preservation("hybrid", groups, tmp / "out", device="cpu",
                                   cache_dir=tmp / "cache",
                                   recipe=stub_recipe(num_epochs=1))
    assert s["preservation_anchor"]["anchor_weight"] == {"retain": 13.642}
    assert s["preservation_anchor"]["anchor_alongside_supervised_term"] == \
        {"retain": True}
    assert "avg_kl_retain" in s["epochs"][0]
    assert "avg_loss_retain" in s["epochs"][0]


def test_an_anchored_run_without_a_cache_is_refused(stubbed):
    groups = [pt.PreservationGroupSpec("retain", stubbed / "retain.jsonl",
                                       pa.ANCHOR_MODE, 1.0)]
    with pytest.raises(ValueError, match="no cache_dir"):
        pt.train_with_preservation("x", groups, stubbed / "out", device="cpu",
                                   recipe=stub_recipe(num_epochs=1))


def test_a_misaligned_cache_stops_the_run(stubbed):
    """The cache is built for the right example IDS but the wrong tokens, which
    is what a changed processor or truncation length would produce."""
    tmp = stubbed
    ids = [f"retain__ex{i}" for i in range(GROUP_SIZES["retain"])]
    cache = build_stub_cache(tmp / "cache", ids)
    p = tmp / "cache" / pa.CACHE_SIDECAR
    s = json.loads(p.read_text())
    for row in s["rows"]:
        row["targets"] = list(reversed(row["targets"]))
    p.write_text(json.dumps(s))
    assert pa.ReferenceCache.load(tmp / "cache").rows  # hash still valid
    del cache
    groups = [pt.PreservationGroupSpec("retain", tmp / "retain.jsonl",
                                       pa.ANCHOR_MODE, 1.0)]
    with pytest.raises(AssertionError, match="no longer row-aligned"):
        pt.train_with_preservation("x", groups, tmp / "out", device="cpu",
                                   cache_dir=tmp / "cache",
                                   recipe=stub_recipe(num_epochs=1))


# ── the docstring is checked against the artifact it describes ──────────

def prose(text: str) -> str:
    """Docstring text with the line wrapping removed.

    Searched rather than matched verbatim because a docstring is rewrapped by
    hand and a figure can land on either side of a newline; the point is that
    the number appears, not that the line break does.
    """
    return " ".join(text.split())


def test_the_anchor_docstring_quotes_the_cache_that_was_actually_built():
    """The figures in ``preservation_anchor``'s docstring are read back from the
    committed sidecar.

    This test exists because those figures were wrong.  The paragraph was
    written before the cache existed and estimated 1,434 rows of 248,077
    log-probabilities at 1.42 GB; the built cache holds 1,489 rows of 248,320
    at 1.479 GB.  Every one of the four was wrong and nothing else in the
    repository would have noticed, because an estimate in a docstring is not
    compared against anything.
    """
    if not CACHE_SIDECAR.exists():
        pytest.skip(f"{CACHE_SIDECAR} is absent from this clone")
    s = json.loads(CACHE_SIDECAR.read_text())
    doc = prose(pa.__doc__)
    gb = s["num_positions"] * s["vocab_size"] * 4 / 1e9
    assert f"{s['num_positions']:,} rows" in doc
    assert f"{s['vocab_size']:,} log-probabilities" in doc
    assert f"{gb:.3f} GB" in doc
    assert f"{s['num_examples']} examples" in doc
    per = s["num_positions"] / s["num_examples"]
    assert f"{per:.2f} each" in doc


def test_the_mechanism_docstrings_quote_the_calibration_they_cite():
    """The rho values and drift figures that justify training both mechanisms.

    These are the numbers the decision rests on, quoted in three places; if the
    calibration ever changes and the prose does not, the prose is arguing from
    a measurement nobody made.
    """
    b = json.loads(BASIS.read_text())
    rho = b["mechanism_decision"]["spearman_rho_by_predictor"]
    fine = rho["fine_target_nll"]
    kl = rho["kl_from_mf_on_the_183_preserved_prompts"]
    #: The rho table is keyed by exactly the eight frozen floor numbers plus
    #: D_G -- so the correlation is computed on the quantities the floor reads,
    #: and on nothing else.
    assert set(fine) == set(kl) == set(rt.FLOOR_NUMBER_KEYS) | {"D_G"}
    assert rho["num_candidates"] == 8
    assert fine["image.retain_same_entity_image.row_micro"] == -0.976
    assert fine["image.retain_other_entity_image.row_micro"] == -1.0
    assert fine["D_G"] == -0.024
    drift = [kl[k] for k in
             ("image.retain_same_entity_image.entity_macro",
              "image.retain_same_entity_image.row_micro",
              "image.retain_other_entity_image.entity_macro",
              "image.retain_other_entity_image.row_micro")]
    assert min(drift) == -0.619 and max(drift) == -0.571

    md = b["mechanism_decision"]["the_anchor_quantity_does_not_determine_retention"]
    oracle = md["oracle_versus_nearest_candidate"]
    key = "image.retain_same_entity_image.row_micro"
    assert oracle["mg_drift"] == pytest.approx(0.0296, abs=5e-5)
    assert oracle["candidate_drift"] == pytest.approx(0.0308, abs=5e-5)
    assert oracle["retention_apart"][key] == pytest.approx(0.105, abs=5e-4)
    pairs = md["close_drift_pairs"]
    assert all(p["drift_apart"] < 0.015 for p in pairs)
    quoted = sorted(p["retention_apart"][key] for p in pairs[2:4])
    assert quoted[0] == pytest.approx(0.150, abs=5e-4)
    assert quoted[1] == pytest.approx(0.169, abs=5e-4)

    for mod in (pt, sg):
        doc = prose(mod.__doc__)
        for figure in ("-0.976", "-1.000", "-0.024", "0.0296", "0.0308",
                       "0.105", "0.150", "0.169"):
            assert figure in doc, (mod.__name__, figure)



# ── path resolution ─────────────────────────────────────────────────────

def test_no_resolver_produces_a_doubled_data_prefix():
    """The defect that killed the Stage-1b determinism control in two seconds.

    ``dataset_dir_for_tag`` already returns a ``data/``-prefixed relative path,
    so a second ``"data"`` join is invisible in the source and only appears when
    the file is opened.  One owner for every path is the fix; this is the check
    that the owner holds.
    """
    for fn in (sg.dataset_dir, sg.groups_dir, sg.stage2_ckpt_root,
               sg.stage1_ckpt_root, sg.control_dir, sg.stage2_predictions_dir,
               sg.stage1_predictions_dir, sg.mf_adapters, sg.mg_adapters,
               sg.reference_cache_dir):
        resolved = str(fn(REPO_ROOT))
        assert "data/data/" not in resolved, fn.__name__
        assert "//" not in resolved, fn.__name__


def test_stage2_never_writes_where_stage1_filed_its_evidence():
    """Both grids contain a row called B0, so a shared root would let a Stage-2
    write land on committed Stage-1 evidence.

    Compared as resolved paths, not as strings: ``predictions_iter12`` is a
    prefix of ``predictions_iter12_stage2``, so a substring test would either
    fail on a correct pair or pass on a genuinely shared one.
    """
    assert sg.stage2_ckpt_root(REPO_ROOT) != sg.stage1_ckpt_root(REPO_ROOT)
    stage2_preds = sg.stage2_predictions_dir(REPO_ROOT).resolve()
    stage1_preds = sg.stage1_predictions_dir(REPO_ROOT).resolve()
    assert stage2_preds != stage1_preds
    #: Neither nested inside the other, in either direction: Stage 2 READS two
    #: parquets from Stage 1's directory, so a Stage-2 write landing anywhere
    #: under it would overwrite evidence the Stage-1c freeze bound by sha256.
    assert stage1_preds not in stage2_preds.parents
    assert stage2_preds not in stage1_preds.parents


def test_the_control_directory_cannot_collide_with_a_candidate_id():
    control = sg.control_dir(REPO_ROOT)
    assert control.parent == sg.stage2_ckpt_root(REPO_ROOT)
    assert control.name.startswith("_")
    assert control.name not in {c.candidate_id for c in grid()}


def test_the_incumbent_row_is_the_one_stage1_filed():
    assert sg.incumbent_row() == ITER12_REFERENCE_ROW
    assert ITER12_REFERENCE_ROW in {c.candidate_id for c in iter12_grid()}


# ── the freeze ──────────────────────────────────────────────────────────

def test_the_freeze_verifies_against_the_repository():
    res = run("freeze_iter12_stage2.py", "--check-only")
    assert res.returncode == 0, res.stdout + res.stderr
    assert "OK" in res.stdout


def test_the_freeze_binds_the_code_that_implements_stage2():
    doc = freeze()
    assert len(doc["hashes"]["protocol_paths"]) == len(fis2.PROTOCOL_PATHS) == 8
    assert len(doc["hashes"]["depends_on_unchanged"]) == \
        len(fis2.DEPENDS_ON_UNCHANGED)
    #: Every protocol path is a file this repository actually has, so a bound
    #: hash is a hash of something rather than of an absence.
    for rel in fis2.PROTOCOL_PATHS:
        assert (REPO_ROOT / rel).exists(), rel
    #: The sealed confirmation selector is DEPENDS ON, never a protocol path:
    #: Stage 2 imports it and must not edit it.
    assert "scripts/select_unlearning_checkpoints.py" in \
        fis2.DEPENDS_ON_UNCHANGED
    assert "scripts/select_unlearning_checkpoints.py" not in fis2.PROTOCOL_PATHS


def test_the_frozen_grid_is_the_one_the_code_builds():
    """A freeze that recorded a grid the code no longer produces would govern
    one study and train another."""
    doc = freeze()
    rebuilt = sg.stage2_grid(beta_star())
    assert doc["grid"]["num_rows"] == len(rebuilt)
    assert doc["grid"]["num_trained"] == len(sg.trained_rows(rebuilt)) == 7
    frozen_ids = [r["candidate_id"] for r in doc["grid"]["rows"]]
    assert frozen_ids == [c.candidate_id for c in rebuilt]


def test_the_freeze_anchors_on_mg_and_reports_against_b0():
    doc = freeze()
    frozen = doc["what_is_frozen"]
    assert frozen["primary_anchor"] == "MG"
    assert frozen["num_numbers"] == len(rt.FLOOR_NUMBER_KEYS) == 8
    strat = json.loads(ROUTE_STRATIFIED.read_text())
    assert frozen["anchor_values"] == \
        strat["stage1"]["reference_states_not_floored"]["MG"]["values"]
    assert frozen["reported_baseline"] == "B0"
    assert frozen["reported_baseline_values"] == strat["anchor"]["values"]
    assert frozen["reported_baseline_is_not_decision_bearing"] is True


def test_the_generation_contract_is_inherited_and_not_restated():
    """Stage 2 generates for its own candidates and reads two of Stage 1's; if
    those came from different instruments the comparison would span two
    measurements, and the image route is the one whose correctness flips with
    batch composition."""
    doc = freeze()
    inherited = doc[fis2.CONTRACT_FIELD]
    stage1 = json.loads(
        (REPO_ROOT / fis2.STAGE1_FREEZE_REPORT).read_text())
    assert inherited["read_from"] == fis2.STAGE1_FREEZE_REPORT
    for key in fis2.CONTRACT_MUST_MATCH:
        assert inherited["contract"][key] == \
            stage1["generation_contract"][key], key
    assert inherited["agrees_with_the_stage1c_verified_contract"] is True


def test_the_reused_states_are_shared_between_the_freeze_and_the_selector():
    """Imported, not copied: the two cannot disagree about how many bytes the
    floor rests on."""
    assert sis2.REUSED_STATES is fis2.REUSED_STATES
    assert set(fis2.REUSED_STATES) == {"B0", "MG", sg.incumbent_row()}
    assert set(freeze()[fis2.PREDICTIONS_FIELD]["sha256"]) == \
        {f"stage1.{s}" for s in fis2.REUSED_STATES}


def test_the_selector_imports_the_sealed_generation_route():
    """A reimplementation of the batch layout would drift from the one that
    produced the anchor's numbers, silently."""
    from select_unlearning_checkpoints import _generate_state
    assert sis2._generate_state is _generate_state


def test_adapters_exist_sees_the_control_one_level_deeper(tmp_path):
    """The control's adapter is the one that most needs to lock the freeze: a
    control run under an unfrozen loop proves nothing about the frozen one."""
    assert fis2.adapters_exist(tmp_path) == []
    assert fis2.trained_yet(tmp_path) is False

    (tmp_path / sg.STAGE2_CKPT_ROOT / "B5_cap2.0" / "adapters").mkdir(
        parents=True)
    assert len(fis2.adapters_exist(tmp_path)) == 1

    deeper = tmp_path / sg.STAGE2_CKPT_ROOT / sg.STAGE2_CONTROL_SUBDIR / "c"
    (deeper / "adapters").mkdir(parents=True)
    found = fis2.adapters_exist(tmp_path)
    assert len(found) == 2
    assert any(sg.STAGE2_CONTROL_SUBDIR in f for f in found)
    assert fis2.trained_yet(tmp_path) is True


def test_an_amendment_without_a_reason_is_refused_and_writes_nothing(monkeypatch):
    """An amendment whose reason is not recorded cannot be distinguished from a
    rule changed to fit its own result -- and the refusal has to happen before
    the write, not after it.

    ``adapters_exist`` is patched to empty rather than left alone, and that is
    deliberate rather than a convenience.  The adapter refusal is evaluated
    FIRST and unconditionally, so once the control has written an adapter this
    test would be exercising that refusal instead of the one it is about, and
    its assertion on the message would fail -- a transient-precondition test
    that breaks because the study made progress.  The adapter refusal has its
    own test below; this one holds the reason check.
    """
    before = FREEZE.read_bytes()
    monkeypatch.setattr(fis2, "adapters_exist", lambda root: [])
    monkeypatch.setattr(sys, "argv", ["freeze", "--refreeze",
                                      "--repo-root", str(REPO_ROOT)])
    try:
        with pytest.raises(SystemExit, match="--reason"):
            fis2.main()
    finally:
        #: Restored rather than merely asserted, so a regression here cannot
        #: leave the repository holding an unreasoned amendment.
        FREEZE.write_bytes(before)
    assert FREEZE.read_bytes() == before


def test_every_amendment_records_that_no_adapter_existed_yet():
    doc = freeze()
    assert doc["amendments"], "the freeze was amended once and says so"
    for a in doc["amendments"]:
        assert a["no_adapter_existed_at_amendment_time"] is True
        assert a["reason"].strip()
        assert a["num_fields_changed"] == len(a["fields_changed"])
        assert len(a["supersedes_sha256"]) == 64


def test_no_amendment_has_ever_touched_the_criterion():
    """What an amendment may and may not move.

    A protocol path's own hash necessarily changes whenever the script or a
    module it binds is edited, and a self-describing field may be added; those
    are amendments a reader can audit.  What an amendment must NEVER do is move
    the rule -- the anchor, its values, the eight numbers, the epsilon, the
    tie-break or the grid -- because that is precisely the change a
    preregistration exists to make impossible after the fact.

    ``faithfulness_control.*`` is amendable and was amended, and that is a
    concession worth stating plainly: the control's comparison basis was
    replaced after it ran and refused.  It is APPARATUS, not criterion -- it
    decides nothing about which candidate wins, and the run it retired produced
    no candidate, no prediction, no retention number and no score, so there was
    no result for the anchor, the epsilon, the tie-break or the grid to have
    been fitted to.  What the criterion is stays sealed.  Because a seal that
    can be widened is only as good as what replaces it, the tests below pin
    what the amended control must still DO, structurally, in the script.

    Written as an exclusion rather than as a list of the fields changed so far,
    so the test still means something after a fourth amendment.
    """
    sealed_prefixes = ("what_is_frozen.", "grid.", "anchor_change.anchor_values",
                       "anchor_change.reported_baseline",
                       "reused_predictions_bound.", "reference_cache_bound.",
                       "generation_contract_inherited.contract.")
    #: The only prefixes an amendment may touch, besides the hashes it must.
    amendable = ("hashes.", "faithfulness_control.",
                 "anchor_change.enforced_by")
    changed = [k for a in freeze()["amendments"] for k in a["fields_changed"]]
    assert changed, "an amendment that changed nothing would be a strange one"
    for key in changed:
        assert not key.startswith(sealed_prefixes), key
        assert key == "anchor_change.enforced_by" or key.startswith(
            tuple(p for p in amendable if p != "anchor_change.enforced_by")), key


TRAIN_SCRIPT = REPO_ROOT / "scripts" / "train_iter12_stage2.py"


def _script_tree():
    """The training script as a syntax tree.

    Parsed rather than imported.  The script pulls torch in through the
    trainers, and CI installs a closure without it -- but that is a convenient
    reason, not the real one.  What these tests assert is WHICH BRANCH assigns
    the control's gate, and that is a property of the script's structure, so a
    parse is the instrument that actually answers the question.  Importing it
    and running it would need a GPU and half an hour.
    """
    import ast
    return ast.parse(TRAIN_SCRIPT.read_text())


def _function(tree, name):
    import ast
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"{name} is no longer a top-level function of "
                         f"scripts/train_iter12_stage2.py")


def _statements(fn) -> list:
    """A function's body with its docstring dropped."""
    import ast
    body = list(fn.body)
    if body and isinstance(body[0], ast.Expr) \
            and isinstance(body[0].value, ast.Constant) \
            and isinstance(body[0].value.value, str):
        body = body[1:]
    return body


def _gate_assignments(fn) -> list:
    """Every ``passed = ...`` in ``fn``, in source order.

    Collected rather than assumed, so a test that counts them fails loudly if
    the gate grows a fourth branch instead of failing silently because it was
    looking at the wrong one.
    """
    import ast
    out = []
    for node in ast.walk(fn):
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id == "passed":
                out.append(node)
    return sorted(out, key=lambda n: n.lineno)


def test_the_control_pins_the_hash_seed_before_it_trains_anything():
    """The environment is held still BEFORE a GPU-minute is spent, not after.

    Checked as the first statement of ``run_control`` rather than merely as a
    call somewhere inside it: a check that ran after the three trainings would
    report a refusal nobody could act on, having already produced adapters
    whose ``target_modules`` order no later run could match.
    """
    import ast
    first = _statements(_function(_script_tree(), "run_control"))[0]
    assert isinstance(first, ast.Assign), (
        "run_control's first statement is no longer the hash-seed check")
    assert isinstance(first.value, ast.Call)
    assert first.value.func.id == "assert_hash_seed_pinned", (
        "run_control's first statement no longer pins PYTHONHASHSEED")


def test_an_unpinned_hash_seed_refuses_rather_than_measuring_the_interpreter(monkeypatch):
    """Executed, not merely present.

    The function is lifted out of the script by ``ast.get_source_segment`` and
    run, because the behaviour that matters is that it RAISES when the variable
    is absent -- and a structural check for "there is a Raise node somewhere in
    here" passes just as happily for ``if False: raise``, which is the mistake
    worth catching.  Lifting one function out keeps this torch-free: the script
    imports the trainers at module level and CI installs a closure without it,
    but this function needs only ``os``.
    """
    import ast
    src = ast.get_source_segment(
        TRAIN_SCRIPT.read_text(),
        _function(_script_tree(), "assert_hash_seed_pinned"))
    ns: dict = {}
    exec(compile(src, "<assert_hash_seed_pinned>", "exec"), ns)

    monkeypatch.delenv("PYTHONHASHSEED", raising=False)
    with pytest.raises(SystemExit, match="PYTHONHASHSEED"):
        ns["assert_hash_seed_pinned"]()

    #: And it returns the value when it is set, so the control can record which
    #: seed it ran under rather than merely that one was present.
    monkeypatch.setenv("PYTHONHASHSEED", "7")
    assert ns["assert_hash_seed_pinned"]() == "7"


def test_the_control_runs_the_frozen_loop_twice_and_the_new_loop_once():
    """The noise floor is measured, not assumed.

    ``A2`` is the whole point of the redesign: without a second run of the SAME
    loop there is no number to compare the across-loop gap to, and any
    threshold would be a guess.  Two of the three runs must therefore be the
    frozen loop, and the third the new one with no cap and no anchor.
    """
    import ast
    fn = _function(_script_tree(), "run_control")
    loops = [n for n in ast.walk(fn) if isinstance(n, ast.For)
             and isinstance(n.iter, ast.Tuple)]
    assert loops, "run_control no longer drives its runs from one tuple"
    entries = loops[0].iter.elts
    assert len(entries) == 3, (
        f"the control runs {len(entries)} trainings; the design needs exactly "
        f"three -- frozen, frozen again, new")
    called = [e.elts[1] for e in entries]

    def name_of(node):
        return node.attr if isinstance(node, ast.Attribute) else node.id

    assert [name_of(c) for c in called] == [
        "train_unlearning", "train_unlearning", "train_with_preservation"], (
        "the control no longer runs the frozen loop twice before the new one")
    assert entries[2].elts[0].value == "B_new_loop_no_cap_no_anchor", (
        "the third run is the one that must carry no cap and no anchor")


def test_the_gate_never_passes_unconditionally():
    """The gate must be able to fail.

    An amendment may widen what ``faithfulness_control`` says, so the script
    itself is what has to hold: if ``passed`` could be assigned a bare ``True``,
    or if the across-loop gap were never compared against the floor, the control
    would be a formality that reported success whatever the two loops did.
    """
    import ast
    fn = _function(_script_tree(), "run_control")
    gates = _gate_assignments(fn)
    assert len(gates) == 3, (
        f"the gate has {len(gates)} branches; the design has three -- not "
        f"comparable, floor is zero, floor is not zero")
    for node in gates:
        v = node.value
        assert not (isinstance(v, ast.Constant) and v.value is True), (
            "the gate assigns passed = True unconditionally, so it can no "
            "longer fail")
    dumped = [ast.dump(n.value) for n in gates]
    assert any("bitwise_identical" in d for d in dumped), (
        "no branch demands bitwise equality, so the strongest form of the "
        "claim -- the two loops are one loop -- is no longer reachable")
    assert any(d.startswith("Compare") for d in dumped), (
        "no branch compares the across-loop gap against the measured floor, so "
        "the gate would have no magnitude to read")
    assert any("max_abs_gap" in d for d in dumped), (
        "the gate no longer reads the per-tensor gap; a digest alone cannot say "
        "whether two adapters differ by float noise or by a different objective")

    #: The bitwise demand has to be CONDITIONED ON THE FLOOR being zero.  A
    #: branch that demanded bitwise equality unconditionally would fail on any
    #: non-deterministic stack, and one that demanded it under some other
    #: condition would be demanding it for a reason the report does not state.
    #: Checking the assignment alone cannot see the condition, so the If chain
    #: is walked instead.
    conditional = [n for n in ast.walk(fn) if isinstance(n, ast.If)
                   and "floor" in ast.dump(n.test)
                   and "bitwise_identical" in ast.dump(n.test)]
    assert conditional, (
        "no branch conditions on the noise floor being bitwise identical, so "
        "the gate cannot be telling 'the floor is zero, demand exact equality' "
        "apart from 'the floor is not zero, compare magnitudes'")


def test_the_gate_compares_the_right_pair_of_runs_on_each_side():
    """Which two runs each side of the gate is built from.

    ``floor`` must be A against A2 -- one loop versus itself -- and ``across``
    must be A against B -- the two loops versus each other.  Naming the wrong
    pair is the subtlest way to break this control, and the test above cannot
    see it: the gate would still read ``across`` and ``floor``, still branch
    three ways, still never assign a bare ``True``, and the report would still
    be complete.  But if ``across`` were built from the FILED adapter the gate
    would be the unsatisfiable criterion this design replaced, and if ``floor``
    were built from A against B the gate would compare a quantity against
    itself and pass always.

    Indirection is why this reads the two bindings rather than the gate.
    """
    import ast
    fn = _function(_script_tree(), "run_control")

    def keys_of(value):
        return sorted(n.value for n in ast.walk(value)
                      if isinstance(n, ast.Constant) and isinstance(n.value, str))

    bound = {}
    for node in ast.walk(fn):
        if isinstance(node, ast.Assign) and len(node.targets) == 1 \
                and isinstance(node.targets[0], ast.Name):
            bound[node.targets[0].id] = node.value
    for name in ("floor", "across"):
        assert name in bound, f"run_control no longer binds {name}"
        assert isinstance(bound[name], ast.Call) \
            and bound[name].func.id == "compare_adapters", (
            f"{name} is no longer a compare_adapters call")
    assert keys_of(bound["floor"]) == ["A2_frozen_loop_again", "A_frozen_loop"], (
        "the noise floor is not the frozen loop against itself, so the gate "
        "would have no run-to-run spread to compare against")
    assert keys_of(bound["across"]) == [
        "A_frozen_loop", "B_new_loop_no_cap_no_anchor"], (
        "the across-loop gap is not the frozen loop against the new one, so the "
        "gate is not measuring the thing the control exists to measure")


def test_the_filed_adapter_is_reported_and_never_gates():
    """The bytes that cannot be reproduced must not decide anything.

    The incumbent's adapter is still loaded and its gap still written into the
    report, so a reader can see how far today's frozen loop lands from the one
    Stage 1 filed.  It must not reach ``passed``: gating on it is the defect
    this control was redesigned to remove.
    """
    import ast
    fn = _function(_script_tree(), "run_control")
    assert any(isinstance(n, ast.Assign)
               and any(isinstance(t, ast.Name) and t.id == "vs_filed"
                       for t in n.targets) for n in ast.walk(fn)), (
        "the filed adapter is no longer compared at all, so its gap is no "
        "longer visible in the report")
    #: Compared by NAME rather than by substring, because the gate could reach
    #: the filed adapter through either binding: ``vs_filed``, the comparison,
    #: or ``filed``, the path it was loaded from.
    forbidden = {"vs_filed", "filed"}
    for node in _gate_assignments(fn):
        reached = {n.id for n in ast.walk(node.value)
                   if isinstance(n, ast.Name)} & forbidden
        assert not reached, (
            f"the gate reads {sorted(reached)}, the filed Stage-1 adapter, "
            f"whose target_modules order was written under a randomised "
            f"PYTHONHASHSEED and cannot be reproduced by any later run")
    assert "filed_stage1_adapter_reported_not_gated" in ast.dump(fn), (
        "the report no longer says the filed comparison is not a gate, so a "
        "reader cannot tell what the gate read")


def test_main_refuses_to_train_when_the_control_did_not_pass():
    """The gate has to be a gate.

    ``--control`` writes a marker; ``main`` reads it and refuses.  If that
    refusal disappeared, a failed control would become a warning and seven
    candidates would be trained against a baseline produced by code nobody had
    shown to be the same code.
    """
    import ast
    fn = _function(_script_tree(), "main")
    guarding = [n for n in ast.walk(fn) if isinstance(n, ast.If)
                and "gate_passed" in ast.dump(n.test)
                and any(isinstance(b, ast.Raise) for b in ast.walk(n))]
    assert guarding, (
        "main no longer raises when the control's gate_passed is false")
    #: ``ast.dump`` renders ``marker.exists()`` as an Attribute node, so the
    #: source text "marker.exists" never appears in it; match on the two names.
    assert any(isinstance(n, ast.If) and "id='marker'" in ast.dump(n.test)
               and "attr='exists'" in ast.dump(n.test)
               and any(isinstance(b, ast.Raise) for b in ast.walk(n))
               for n in ast.walk(fn)), (
        "main no longer refuses when the control has not been run at all")


def test_the_freeze_discloses_the_criterion_the_control_replaced():
    """A criterion changed after it ran is only auditable if the record says so.

    The amendment log names the fields; this is what a reader needs beyond it --
    that the control still names the incumbent row, still claims to be a gate,
    and carries the retired criterion, the evidence against it and the reason
    replacing it was legitimate, in its own words rather than leaving the change
    to be reconstructed from a diff.

    Each field is held to what it is FOR rather than to the same list of
    strings: the mechanism has to be named precisely once, where the mechanism
    is stated, and the justification has to say the criterion could not be met
    and point at evidence a reader can actually open.
    """
    fc = freeze()["faithfulness_control"]
    assert fc["reproduces_the_objective_of"] == sg.incumbent_row()
    assert fc["it_is_a_gate_not_a_report"].strip()
    assert fc["the_gate"].strip()
    for key in ("a_criterion_this_replaced",
                "the_evidence_that_it_was_the_stack_not_the_new_loop",
                "why_replacing_it_is_not_moving_the_goalposts"):
        assert key in fc, f"the freeze no longer discloses {key}"
        assert fc[key].strip(), f"{key} is present but empty"

    #: The mechanism, named precisely and where the mechanism is stated.  Both
    #: spellings are required: the environment variable is what an operator
    #: would set, the PEFT attribute is what makes it matter.
    mechanism = fc["a_criterion_this_replaced"]
    assert "PYTHONHASHSEED" in mechanism and "target_modules" in mechanism, (
        "the disclosure does not name the mechanism precisely, so a reader "
        "cannot reproduce the finding or check it")

    #: The justification has to say the criterion COULD NOT be met -- that is
    #: what separates replacing a broken gate from moving one that was working.
    justification = fc["why_replacing_it_is_not_moving_the_goalposts"]
    assert "unsatisfiable" in justification, (
        "the disclosure does not claim the retired criterion was unsatisfiable, "
        "which is the only thing that would make replacing it legitimate")
    assert "hash seed" in justification, (
        "the justification does not say what the retired criterion actually "
        "tested instead of the loops")

    #: And a disclosure that names no inspectable artifact is an assertion.
    #: Both fields must point at the kept runs and quote a measured magnitude,
    #: so the claim can be checked against the files rather than believed.
    evidence = fc["the_evidence_that_it_was_the_stack_not_the_new_loop"]
    assert "outputs/superseded/iter12_stage2_control_v1" in evidence, (
        "the disclosure no longer says where the two runs are kept")
    assert (REPO_ROOT / "outputs/superseded/iter12_stage2_control_v1"
            / "README.md").exists(), (
        "the freeze points at a superseded-evidence README that is not there")
    assert "e-03" in evidence, (
        "the evidence field quotes no measured magnitude, so 'the same order' "
        "would be unverifiable")
    #: The whole point of the re-run: the FROZEN loop missed the filed bytes
    #: too.  Without that, the disclosure would be an argument, not evidence.
    assert "FROZEN loop" in evidence


def test_the_freeze_names_the_tests_that_hold_the_widened_seal():
    """The freeze lists those tests by name, so the list has to be true.

    Naming them is what makes a later amendment that drops one visible as a
    change to that field -- but only if something checks the names against this
    file.  A list of tests that do not exist would be a worse disclosure than no
    list at all, because it would look like a guarantee.

    The keyword set below is the definition of "a structural test of the
    control", and it is enforced here rather than left in a throwaway checking
    script so the count the README and the freeze quote is verifiable from the
    committed repository alone.
    """
    import ast
    named = freeze()["faithfulness_control"]["what_pins_this_in_the_suite"]
    tree = ast.parse(Path(__file__).read_text())
    present = {n.name for n in tree.body
               if isinstance(n, ast.FunctionDef) and n.name.startswith("test_")}
    missing = [n for n in named if n not in present]
    assert not missing, (
        f"the freeze names tests this file does not define: {missing}")
    assert len(named) == len(set(named)), "the freeze lists a test twice"

    keywords = ("hash_seed", "frozen_loop_twice", "never_passes",
                "right_pair_of_runs", "never_gates", "main_refuses",
                "discloses_the_criterion")
    structural = {n for n in present if any(k in n for k in keywords)}
    assert structural == set(named), (
        "the freeze's list and this file's structural control tests differ: "
        f"unlisted {sorted(structural - set(named))}, "
        f"named-but-absent {sorted(set(named) - structural)}")


def test_the_freeze_still_refuses_once_an_adapter_exists(tmp_path, monkeypatch):
    """The refusal is keyed on adapters, evaluated first and unconditionally, so
    ``--refreeze`` cannot reach past it."""
    (tmp_path / sg.STAGE2_CKPT_ROOT / "x" / "adapters").mkdir(parents=True)
    monkeypatch.setattr(sys, "argv",
                        ["freeze", "--refreeze", "--reason",
                         "fitted to the result", "--repo-root", str(tmp_path)])
    with pytest.raises(SystemExit, match="already exist"):
        fis2.main()


# ── the selector's guards ───────────────────────────────────────────────

def test_the_sealed_confirmation_is_refused_in_either_direction():
    from granunlearn.evaluation import retention_selection as rs
    sealed = REPO_ROOT / rs.FORBIDDEN_EVIDENCE[0]
    with pytest.raises(SystemExit, match="sealed"):
        sis2.assert_no_forbidden_evidence([sealed / "predictions.parquet"],
                                          REPO_ROOT)
    with pytest.raises(SystemExit, match="sealed"):
        sis2.assert_no_forbidden_evidence([sealed], REPO_ROOT)


def test_a_traversal_path_cannot_walk_past_the_seal():
    """Checked on RESOLVED paths: a relative path with ``..`` in it, or a
    symlink, would otherwise walk straight past a string comparison."""
    sneaky = REPO_ROOT / "data" / "mllmu_hier_pilot100" / ".." / \
        "mllmu_hier_confirm100" / "predictions.parquet"
    assert "mllmu_hier_confirm100" not in str(sneaky).split("/..")[0]
    with pytest.raises(SystemExit, match="sealed"):
        sis2.assert_no_forbidden_evidence([sneaky], REPO_ROOT)


def test_stage2s_own_paths_are_allowed():
    sis2.assert_no_forbidden_evidence(
        [sg.stage2_predictions_dir(REPO_ROOT) / "B5_cap2.0.parquet",
         sg.stage2_ckpt_root(REPO_ROOT) / "B5_cap2.0",
         REPO_ROOT / sg.OUT_REPORT], REPO_ROOT)


def filed_eight_numbers() -> dict:
    strat = json.loads(ROUTE_STRATIFIED.read_text())
    return {
        "B0": dict(strat["anchor"]["values"]),
        "MG": dict(strat["stage1"]["reference_states_not_floored"]["MG"]
                   ["values"]),
        sg.incumbent_row(): dict(
            strat["stage1"]["candidates"][sg.incumbent_row()]["values"]),
    }


def test_the_floor_gate_holds_on_the_values_stage1c_filed():
    """The same discipline Stage 1c applied to its own text stratum: an
    extension that disagrees with the rule it extends, on the numbers they
    share, is a different rule wearing its name."""
    res = sis2.gate_reused_against_the_filed_values(filed_eight_numbers(),
                                                    REPO_ROOT)
    assert res["holds"] is True, res["problems"]
    assert res["problems"] == []
    assert res["numbers_compared"] == 3 * len(rt.FLOOR_NUMBER_KEYS) == 24
    assert sorted(res["states_gated"]) == sorted(fis2.REUSED_STATES)


@pytest.mark.parametrize("state", sorted(fis2.REUSED_STATES))
def test_one_perturbed_number_fails_the_gate_and_names_it(state):
    """A floor whose anchor does not reproduce its own filed value is not the
    frozen floor -- and the report has to say WHICH of the 24 moved."""
    values = filed_eight_numbers()
    key = rt.FLOOR_NUMBER_KEYS[0]
    values[state][key] = round(values[state][key] + 0.001, 6)
    res = sis2.gate_reused_against_the_filed_values(values, REPO_ROOT)
    assert res["holds"] is False
    assert len(res["problems"]) == 1
    assert state in res["problems"][0] and key in res["problems"][0]


def test_a_state_the_gate_was_not_given_is_reported_rather_than_passed():
    values = filed_eight_numbers()
    del values["MG"]
    res = sis2.gate_reused_against_the_filed_values(values, REPO_ROOT)
    assert res["holds"] is False
    assert any("MG: not recomputed" in p for p in res["problems"])


def test_the_reused_bytes_are_refused_when_they_are_not_there(tmp_path):
    """Stage 2 reads the anchor's predictions rather than regenerating them, so
    without those bytes there is no floor to apply -- and the refusal has to
    name the parquet, not merely fail."""
    (tmp_path / Path(fis2.OUT_REPORT).parent).mkdir(parents=True)
    (tmp_path / fis2.OUT_REPORT).write_text(FREEZE.read_text())
    assert sis2.reused_paths(tmp_path)
    with pytest.raises(SystemExit, match="is missing"):
        sis2.verify_reused_bytes(tmp_path)


def test_a_reused_parquet_whose_hash_moved_is_refused(tmp_path, monkeypatch):
    (tmp_path / Path(fis2.OUT_REPORT).parent).mkdir(parents=True)
    doc = freeze()
    key = f"stage1.{fis2.REUSED_STATES[0]}"
    doc[fis2.PREDICTIONS_FIELD]["sha256"][key] = "f" * 64
    (tmp_path / fis2.OUT_REPORT).write_text(json.dumps(doc))
    for state, rel in sis2.reused_paths(tmp_path).items():
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"not the parquet stage 1c measured")
    with pytest.raises(SystemExit, match="hashes to"):
        sis2.verify_reused_bytes(tmp_path)


def test_a_parquet_the_freeze_never_bound_is_refused(tmp_path):
    """Certifying a file no freeze hashed would be certifying nothing."""
    (tmp_path / Path(fis2.OUT_REPORT).parent).mkdir(parents=True)
    doc = freeze()
    doc[fis2.PREDICTIONS_FIELD]["sha256"] = {}
    (tmp_path / fis2.OUT_REPORT).write_text(json.dumps(doc))
    for _state, rel in sis2.reused_paths(tmp_path).items():
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"x")
    with pytest.raises(SystemExit, match="binds no sha256"):
        sis2.verify_reused_bytes(tmp_path)
