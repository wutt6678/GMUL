"""Iteration 12 Stage 2 — the bounded-suppression and anchor grids.

Stage 1 swept how hard to rehearse the fit half and Stage 1b replicated the two
near-misses at four seeds; Stage 1c read the image route the frozen floor had
never looked at and found the failure was real rather than noise.  Stage 2
changes the OBJECTIVE instead of its strength, and it does so on a measurement
rather than on a hunch.

``scripts/build_mf_reference_logprobs.py --calibrate`` loaded all eleven
adapters this repository already had — MF, MG, B0 and the eight Stage-1
candidates — and measured, on the 183 fit-half prompts the anchor would pin,
how far each one had drifted from MF, together with each one's NLL on all three
knowledge groups.  Joined against the eight retention numbers Stage 1c filed:

* drift does not determine retention.  MG sits 0.0296 nats from MF on those
  prompts and ``B4_w4.0_lam0.5_lr2e-05_ep5`` sits 0.0308, and their image
  retain-same differs by 0.105.  Two further pairs within 0.015 of each other
  in drift differ by 0.150 and 0.169 in retention, against a floor that
  resolves 0.0044.
* the anchor's optimum is the wrong place.  ``KL = 0`` is MF, which performs no
  unlearning at all; the oracle sits at 0.0296.  A strong beta drives a
  candidate toward MF rather than toward MG.
* what DOES rank them is how far the suppression term drove the fine fact.
  Spearman rho between ``fine_target`` NLL and image retention is -0.976 and
  -1.000, against -0.571 to -0.619 for the drift.
* and the overshoot buys nothing.  rho between ``fine_target`` NLL and D_G is
  -0.024.  MG, the target, sits at 1.3643 nats; the nearest candidate is
  1.7083 and the rest run to 12.777.  Every candidate pushed the fine fact
  further than the state they are scored by distance to.

So Stage 2 trains BOTH mechanisms and lets the frozen floor decide.  ``B5``
bounds the ascent, which is the coordinate the measurement says governs.  ``B6``
is the MF-preservation regularizer Stage 2 was preregistered with, trained at
the beta the calibration derived, because an eight-point observational argument
is not an experiment.

WHAT IS HELD FIXED
------------------
Everything but the objective.  ``B5_cap*`` is the incumbent Stage-1 row
``B4_w1.0_lam0.5_lr2e-05_ep5`` with the ascent capped and nothing else changed:
same lambda 0.5, same target-level weight 1.0, same replay weight 1.0 on the
fit half, same lr, same budget.  That row already exists with predictions on
disk, so it is the sweep's ``cap = infinity`` point for free, and a difference
between it and a ``B5`` row is attributable to the cap alone.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from granunlearn.training.candidate_grid import (
    ITER12_LAM,
    ITER12_REFERENCE_ROW,
    ITER12_REFERENCE_WEIGHT,
    dataset_dir_for_tag,
    groups_subdir_for_tag,
    validate_grid,
)
from granunlearn.training.preservation_anchor import ANCHOR_MODE
from granunlearn.training.preservation_trainer import (
    CAPPED_MODE,
    PreservationGroupSpec,
)

TAG = "iter12"

#: Method labels Stage 2 adds.  ``candidate_grid.METHODS`` is hash-bound and
#: cannot grow, so the frozen validator reports these as unknown; that is
#: filtered in :func:`validate_stage2_grid` below and nowhere else, and the
#: residual must be empty.
METHOD_BOUNDED = "B5"
METHOD_ANCHOR = "B6"
STAGE2_METHODS: tuple[str, ...] = (METHOD_BOUNDED, METHOD_ANCHOR)

#: The ascent caps, in nats of per-token NLL on the fine completion.  Swept,
#: NOT derived from the oracle: the training procedure never reads MG.  MG's
#: own level (1.3643) is reported afterwards as where the target state happens
#: to sit, and this range brackets it from both sides.  The uncapped end of the
#: sweep is the incumbent Stage-1 row, which already exists.
STAGE2_CAPS: tuple[float, ...] = (0.5, 1.0, 2.0, 4.0)

#: The budget the cap is tested at, and the one it is tested AGAINST.  Stage 1
#: coupled "how far to suppress" to "how long to train" through the epoch count,
#: which is why its best retainer (ep3, 1.7083 nats, image retain-same -0.0073)
#: was also its worst forgetter (D_G 0.0619) and its ep8 row reached 11.8286
#: nats.  A cap decouples the two, so the decoupling gets its own row: the same
#: cap at a longer budget.  If the cap is what preserves retention, the longer
#: budget should stop costing it.
STAGE2_LR = 2e-5
STAGE2_EPOCHS = 5
STAGE2_DECOUPLING_EPOCHS = 8
STAGE2_DECOUPLING_CAP = 2.0

#: Reported, never used to choose anything: where the oracle sits on the same
#: axis the cap bounds.  It is a property of the committed MG adapter and is
#: re-measured by the basis builder rather than trusted from this constant.
MG_FINE_TARGET_NLL_REPORTED = 1.3643


@dataclass(frozen=True)
class Stage2CandidateSpec:
    """A single Stage-2 training job.

    Attribute-compatible with ``candidate_grid.CandidateSpec`` so the FROZEN
    ``validate_grid`` can be run over a Stage-2 grid: writing a second
    structural validator for a grid that differs from the frozen one only in
    its method labels and mode vocabulary would mean Stage 2 is checked by a
    rule nobody froze.
    """

    candidate_id: str
    method: str
    groups: tuple[PreservationGroupSpec, ...] = ()
    overrides: dict[str, Any] = field(default_factory=dict)
    noop: bool = False

    def describe(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "method": self.method,
            "noop": self.noop,
            "groups": [g.describe() for g in self.groups],
            "overrides": dict(self.overrides),
        }

    @property
    def cap(self) -> float | None:
        """The row's ascent cap, or None if the ascent is unbounded."""
        for g in self.groups:
            if g.mode == CAPPED_MODE:
                return g.cap
        return None

    @property
    def beta(self) -> float | None:
        """The row's anchor strength, however the anchor is applied."""
        for g in self.groups:
            if g.mode == ANCHOR_MODE:
                return g.weight
            if g.anchor_weight is not None:
                return g.anchor_weight
        return None


def bounded_id(cap: float, lr: float, epochs: int) -> str:
    return f"B5_cap{cap}_lam{ITER12_LAM}_lr{lr:g}_ep{epochs}"


def anchor_id(beta: float, lr: float, epochs: int, replay: float | None) -> str:
    """``replay=None`` is the anchor ALONE; a float is the hybrid row."""
    tag = "B6R" if replay is not None else "B6"
    extra = f"_w{replay}" if replay is not None else ""
    return f"{tag}_beta{beta}{extra}_lam{ITER12_LAM}_lr{lr:g}_ep{epochs}"


def stage2_grid(beta_star: float) -> list[Stage2CandidateSpec]:
    """B0 + 7 trained rows: 4 caps, 1 decoupling row, 2 anchor rows.

    ``beta_star`` comes from the calibration report, where it is derived as the
    strength at which the anchor term equals the suppression term's magnitude at
    the end of the incumbent row's training.  It is a parameter rather than a
    constant so that the grid cannot be built without the measurement that
    justifies it.
    """
    if beta_star <= 0:
        raise ValueError(
            f"beta_star must be positive, got {beta_star}; it is the anchor "
            f"strength derived by the calibration and a non-positive value "
            f"means the calibration did not produce one")
    gdir = f"{dataset_dir_for_tag(TAG)}/{groups_subdir_for_tag(TAG)}"
    ft, tl, rt = f"{gdir}/fine_target.jsonl", f"{gdir}/target_level.jsonl", \
        f"{gdir}/retain.jsonl"
    beta = round(beta_star, 4)

    #: B0 is in the grid because the frozen validator requires exactly one
    #: no-op row, and because the selector's report is easier to read with the
    #: floor's own reference point listed beside the candidates.  Its
    #: predictions are REUSED from the parquet the Stage-1c freeze bound, not
    #: regenerated: regenerating would spend 40 minutes to produce a second
    #: draw from identical bytes.
    grid: list[Stage2CandidateSpec] = [
        Stage2CandidateSpec("B0", "B0", noop=True)]

    #: The cap sweep at the incumbent budget.  Every row is the incumbent
    #: recipe with the ascent bounded and nothing else touched.
    for cap in STAGE2_CAPS:
        grid.append(Stage2CandidateSpec(
            bounded_id(cap, STAGE2_LR, STAGE2_EPOCHS), METHOD_BOUNDED,
            (PreservationGroupSpec("fine_target", ft, CAPPED_MODE,
                                   ITER12_LAM, cap=cap),
             PreservationGroupSpec("target_level", tl, "sft", 1.0),
             PreservationGroupSpec("retain", rt, "sft",
                                   ITER12_REFERENCE_WEIGHT)),
            {"learning_rate": STAGE2_LR, "num_epochs": STAGE2_EPOCHS}))

    #: The decoupling row: the same cap at a longer budget.
    grid.append(Stage2CandidateSpec(
        bounded_id(STAGE2_DECOUPLING_CAP, STAGE2_LR,
                   STAGE2_DECOUPLING_EPOCHS), METHOD_BOUNDED,
        (PreservationGroupSpec("fine_target", ft, CAPPED_MODE, ITER12_LAM,
                               cap=STAGE2_DECOUPLING_CAP),
         PreservationGroupSpec("target_level", tl, "sft", 1.0),
         PreservationGroupSpec("retain", rt, "sft", ITER12_REFERENCE_WEIGHT)),
        {"learning_rate": STAGE2_LR,
         "num_epochs": STAGE2_DECOUPLING_EPOCHS}))

    #: The preregistered MF-preservation regularizer, replay REPLACED by the
    #: KL anchor.  Same three components and the same stream length as the
    #: incumbent, so the only difference is what the retained group's forward
    #: pass is asked to do.
    grid.append(Stage2CandidateSpec(
        anchor_id(beta, STAGE2_LR, STAGE2_EPOCHS, None), METHOD_ANCHOR,
        (PreservationGroupSpec("fine_target", ft, "gd", ITER12_LAM),
         PreservationGroupSpec("target_level", tl, "sft", 1.0),
         PreservationGroupSpec("retain", rt, ANCHOR_MODE, beta)),
        {"learning_rate": STAGE2_LR, "num_epochs": STAGE2_EPOCHS}))

    #: The hybrid: replay AND anchor on the same forward pass, through
    #: ``anchor_weight`` rather than a second group entry, so the epoch keeps
    #: 363 micro-batches and the schedule under test does not move.
    grid.append(Stage2CandidateSpec(
        anchor_id(beta, STAGE2_LR, STAGE2_EPOCHS,
                  ITER12_REFERENCE_WEIGHT), METHOD_ANCHOR,
        (PreservationGroupSpec("fine_target", ft, "gd", ITER12_LAM),
         PreservationGroupSpec("target_level", tl, "sft", 1.0),
         PreservationGroupSpec("retain", rt, "sft", ITER12_REFERENCE_WEIGHT,
                               anchor_weight=beta)),
        {"learning_rate": STAGE2_LR, "num_epochs": STAGE2_EPOCHS}))
    return grid


#: The two error classes the FROZEN validator cannot know about, because
#: ``candidate_grid.py`` is hash-bound and its ``METHODS`` tuple and mode
#: vocabulary were frozen before Stage 2 existed.  Filtered EXPLICITLY, by exact
#: message shape, and only here: everything else that validator checks —
#: duplicate ids, exactly one no-op row, positive weights, repeated groups, the
#: override allowlist — still applies to Stage 2 unchanged.
FROZEN_VALIDATOR_GAP = ("unknown method", "bad mode")


def validate_stage2_grid(grid: list[Stage2CandidateSpec],
                         ) -> tuple[list[str], list[str]]:
    """``(errors, filtered)`` — the residual must be empty.

    Returns the filtered messages alongside the errors so a test can pin them:
    a filter that quietly swallowed a real structural complaint would show up
    as a filtered list that is not exactly the two known gaps.
    """
    raw = validate_grid(grid)  # type: ignore[arg-type]
    kept, filtered = [], []
    for msg in raw:
        if any(gap in msg for gap in FROZEN_VALIDATOR_GAP):
            filtered.append(msg)
        else:
            kept.append(msg)

    #: What the frozen validator cannot check, checked here instead.
    ids = [c.candidate_id for c in grid]
    for c in grid:
        if c.noop:
            continue
        if c.method not in STAGE2_METHODS:
            kept.append(f"{c.candidate_id}: Stage-2 method {c.method!r} is not "
                        f"one of {list(STAGE2_METHODS)}")
        if not c.groups:
            kept.append(f"{c.candidate_id}: no knowledge groups")
        modes = {g.mode for g in c.groups}
        if CAPPED_MODE in modes and c.cap is None:
            kept.append(f"{c.candidate_id}: capped mode with no cap")
        if c.cap is not None and c.cap <= 0:
            kept.append(f"{c.candidate_id}: cap must be > 0 nats")
        is_anchored = ANCHOR_MODE in modes or any(
            g.anchor_weight is not None for g in c.groups)
        if is_anchored and (c.beta is None or c.beta <= 0):
            kept.append(f"{c.candidate_id}: anchored row needs a positive "
                        f"beta")
        names = [g.name for g in c.groups]
        if len(names) != len(set(names)):
            kept.append(f"{c.candidate_id}: repeated group {names}")
    if len(ids) != len(set(ids)):
        kept.append(f"duplicate candidate ids: "
                    f"{sorted({i for i in ids if ids.count(i) > 1})}")
    return kept, filtered


# ── path resolution ───────────────────────────────────────────────────
#: Every path this study touches is resolved HERE and nowhere else.  Two call
#: sites building the same path independently is how ``data/data/...`` reached
#: the Stage-1b determinism control: ``dataset_dir_for_tag`` already returns a
#: ``data/``-prefixed relative path, and prepending ``"data"`` again is invisible
#: in the source and only appears when the file is opened.  A single owner makes
#: that disagreement impossible rather than unlikely.

STAGE2_CKPT_ROOT = "data/checkpoints/mllmu_iter12_stage2"
STAGE1_CKPT_ROOT = "data/checkpoints/mllmu_iter12_unlearn"
STAGE2_PREDICTIONS_SUBDIR = "predictions_iter12_stage2"
STAGE2_CONTROL_SUBDIR = "_control"

BASIS_REPORT = "data/reports/mllmu_iter12_stage2_basis.json"
CALIBRATION_REPORT = "data/reports/mllmu_iter12_anchor_calibration.json"
FREEZE_REPORT = "data/reports/mllmu_iter12_stage2_freeze.json"
OUT_REPORT = "data/reports/mllmu_iter12_stage2_selection.json"

#: Read, never written.  Stage 2 reuses the anchor's and B0's predictions from
#: the parquet files the Stage-1c freeze already bound by sha256, so the floor's
#: reference values are byte-identical to the ones already filed.
STAGE1_PREDICTIONS_SUBDIR = "predictions_iter12"
ROUTE_STRATIFIED_REPORT = "data/reports/mllmu_iter12_route_stratified.json"
STAGE1C_FREEZE_REPORT = \
    "data/reports/mllmu_iter12_route_stratification_freeze.json"


def _guard(rel: str) -> str:
    if rel.startswith("data/data/") or "//" in rel:
        raise ValueError(
            f"{rel!r} has a doubled path component. dataset_dir_for_tag and "
            f"the checkpoint roots already carry their own prefix; join them "
            f"once, through these resolvers.")
    return rel


def dataset_dir(repo_root: Path) -> Path:
    """The pilot-100 dataset Stage 2 reuses."""
    return repo_root / _guard(dataset_dir_for_tag(TAG))


def groups_dir(repo_root: Path) -> Path:
    """The fit-half knowledge groups.  The probe half is not here at all."""
    return dataset_dir(repo_root) / groups_subdir_for_tag(TAG)


def stage2_ckpt_root(repo_root: Path) -> Path:
    """Stage 2's own checkpoint root, never Stage 1's."""
    out = repo_root / _guard(STAGE2_CKPT_ROOT)
    stage1 = stage1_ckpt_root(repo_root)
    if out.resolve() == stage1.resolve():
        raise ValueError(
            f"{out} is the Stage-1 checkpoint root. Both grids contain a row "
            f"called B0, so sharing a root would let a Stage-2 write land on "
            f"committed Stage-1 evidence.")
    return out


def stage1_ckpt_root(repo_root: Path) -> Path:
    """Where Stage 1's adapters are.  READ ONLY: Stage 2 writes none of them.

    The incumbent row lives here, and so does the adapter the faithfulness
    control must reproduce byte for byte.
    """
    return repo_root / _guard(STAGE1_CKPT_ROOT)


def control_dir(repo_root: Path) -> Path:
    """Where the faithfulness control writes, apart from the candidates.

    Kept under the Stage-2 root but outside the grid's id namespace, and
    prefixed with an underscore so no candidate id can ever collide with it.
    """
    return stage2_ckpt_root(repo_root) / STAGE2_CONTROL_SUBDIR


def stage2_predictions_dir(repo_root: Path) -> Path:
    """Stage 2's own predictions directory, never Stage 1's or the pilot's."""
    out = dataset_dir(repo_root) / STAGE2_PREDICTIONS_SUBDIR
    for other in (STAGE1_PREDICTIONS_SUBDIR, "predictions",
                  "predictions_iter12_seeds"):
        if out.resolve() == (dataset_dir(repo_root) / other).resolve():
            raise ValueError(
                f"{out} is the {other} directory. Stage 2 reuses two parquets "
                f"from Stage 1 by READING them; writing into that directory "
                f"would put regenerated bytes over evidence the Stage-1c "
                f"freeze bound by sha256.")
    return out


def stage1_predictions_dir(repo_root: Path) -> Path:
    """Where the anchor's and B0's bound predictions already are."""
    return dataset_dir(repo_root) / STAGE1_PREDICTIONS_SUBDIR


def mf_adapters(repo_root: Path) -> Path:
    return repo_root / "data" / "checkpoints" / "mllmu_pilot100" / "MF" \
        / "adapters"


def mg_adapters(repo_root: Path) -> Path:
    return repo_root / "data" / "checkpoints" / "mllmu_pilot100" / "MG" \
        / "adapters"


def reference_cache_dir(repo_root: Path) -> Path:
    """The cached MF distributions the anchor rows read."""
    from granunlearn.training.preservation_anchor import CACHE_DIRNAME
    return dataset_dir(repo_root) / CACHE_DIRNAME


def incumbent_row() -> str:
    """The Stage-1 row every B5 row is the incumbent of, and the cap sweep's
    ``cap = infinity`` point."""
    return ITER12_REFERENCE_ROW


def trained_rows(grid: list[Stage2CandidateSpec],
                 ) -> list[Stage2CandidateSpec]:
    """The rows that need a GPU.  B0 needs none: it is a copy of MF."""
    return [c for c in grid if not c.noop]
