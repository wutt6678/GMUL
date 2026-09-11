"""MF -> MU candidate grids (Iteration 9 smoke, Iteration 11 pilot-100).

A candidate is a fully-specified training job: which knowledge groups it
consumes, how each group enters the objective (``sft`` = minimise the
loss, ``gd`` = gradient ASCENT), and which recipe knobs it overrides.
Everything not listed in ``overrides`` is inherited verbatim from
:class:`~granunlearn.training.reference_trainer.ReferenceRecipe`, so a
candidate can only differ from the reference recipe in the swept knobs —
the counterfactual stays clean.

The grid is DATA (group names + modes + weights + overrides), never
paths: the launching script resolves ``<group>.jsonl`` inside the tagged
dataset directory.  That keeps the grid unit-testable on CPU and makes
the smoke and pilot-100 runs share one code path.

Methods
-------
* ``B0``  no-op — the MF adapter copied unchanged.  Must reproduce MF's
  metrics exactly; it is the pipeline's self-test and the reference
  level for every paired CI.
* ``B1``  complete-forget — gradient ascent on ``fine_target``.
* ``B2``  generalized SFT — SFT on ``target_level`` completions
  (the coarse-positive rewrite; the config's ``generalized_sft``).
* ``B2R`` B2 plus an explicit ``retain`` SFT group.  Iteration 10
  showed the retain group is what closes B2's retain gap, so it is
  swept as its own method rather than folded into B3.
* ``B3``  granularity-aware — ``gd`` fine suppression (weight lambda)
  + ``sft`` target level + ``sft`` retain.
* ``B4``  Iteration 12 retention-aware B3 successor — the same three
  components as B3, but the ``retain`` group is the FIT HALF of the
  retained entities (see ``build_iter12_retention_probe.py``) and its
  weight is the swept knob.  The probe half is never rehearsed, so
  retention can be measured out-of-sample against B0.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

Method = Literal["B0", "B1", "B2", "B2R", "B3", "B4"]
Mode = Literal["sft", "gd"]

METHODS: tuple[str, ...] = ("B0", "B1", "B2", "B2R", "B3", "B4")


@dataclass(frozen=True)
class GroupUse:
    """One objective component, path-free."""

    name: str
    mode: Mode
    weight: float = 1.0


@dataclass(frozen=True)
class CandidateSpec:
    """A single trained candidate."""

    candidate_id: str
    method: Method
    groups: tuple[GroupUse, ...] = ()
    overrides: dict[str, Any] = field(default_factory=dict)
    noop: bool = False

    def describe(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "method": self.method,
            "noop": self.noop,
            "groups": [{"name": g.name, "mode": g.mode,
                        "weight": g.weight} for g in self.groups],
            "overrides": dict(self.overrides),
        }


def _lr_str(lr: float) -> str:
    """Compact, filesystem-safe learning-rate token (``%g``)."""
    return f"{lr:g}"


# ---------------------------------------------------------------------------
# Iteration 9 smoke grid — FROZEN: these ids are referenced by the
# committed selection report and by select_unlearning_checkpoints'
# historical candidate registry, so they must never change.
# ---------------------------------------------------------------------------

def smoke_grid() -> list[CandidateSpec]:
    """The deliberately small Iteration 9 sweep (5 trained + B0)."""
    ft, tl, rt = (GroupUse("fine_target", "gd", 1.0),
                  GroupUse("target_level", "sft", 1.0),
                  GroupUse("retain", "sft", 1.0))
    grid: list[CandidateSpec] = [
        CandidateSpec("B0", "B0", noop=True),
    ]
    for lr in (2e-5, 1e-4):
        grid.append(CandidateSpec(
            f"B1_lr{_lr_str(lr)}", "B1", (ft,),
            {"learning_rate": lr}))
    grid.append(CandidateSpec(
        "B2_lr1e-04", "B2", (tl,), {"learning_rate": 1e-4}))
    for lam in (1.0, 0.5):
        grid.append(CandidateSpec(
            f"B3_lam{lam}", "B3",
            (GroupUse("fine_target", "gd", lam), tl, rt),
            {"learning_rate": 1e-4}))
    return grid


# ---------------------------------------------------------------------------
# Iteration 11 pilot-100 grid — the WIDE B0-B3 sweep.  Every candidate
# starts from the SAME MF checkpoint; only the swept knobs differ.
# Epoch budgets are part of the sweep because the pilot-100 retain group
# is 387 examples (vs 48 in the smoke set), so a fixed 10-epoch budget
# would make B2R/B3 an order of magnitude more expensive than B1/B2
# without being a fairer comparison.
# ---------------------------------------------------------------------------

def pilot100_grid() -> list[CandidateSpec]:
    """16 candidates: B0 + 4 B1 + 5 B2 + 2 B2R + 4 B3."""
    tl = GroupUse("target_level", "sft", 1.0)
    rt = GroupUse("retain", "sft", 1.0)
    grid: list[CandidateSpec] = [CandidateSpec("B0", "B0", noop=True)]

    # B1 complete-forget: gradient ascent on the fine target facts.
    for lr in (1e-5, 2e-5, 5e-5, 1e-4):
        grid.append(CandidateSpec(
            f"B1_lr{_lr_str(lr)}_ep10", "B1",
            (GroupUse("fine_target", "gd", 1.0),),
            {"learning_rate": lr, "num_epochs": 10}))

    # B2 generalized SFT: rewrite the target facts at the retained level.
    for lr, ep in ((2e-5, 10), (5e-5, 10), (1e-4, 10),
                   (2e-5, 5), (1e-4, 5)):
        grid.append(CandidateSpec(
            f"B2_lr{_lr_str(lr)}_ep{ep}", "B2", (tl,),
            {"learning_rate": lr, "num_epochs": ep}))

    # B2R: B2 plus the explicit retain group (Iteration 10 finding).
    for lr in (5e-6, 2e-5):
        grid.append(CandidateSpec(
            f"B2R_lr{_lr_str(lr)}_ep5", "B2R", (tl, rt),
            {"learning_rate": lr, "num_epochs": 5}))

    # B3 granularity-aware: fine suppression + coarse rewrite + retain.
    for lam, lr, ep in ((0.5, 2e-5, 5), (1.0, 2e-5, 5),
                        (0.25, 2e-5, 5), (0.5, 5e-5, 3)):
        grid.append(CandidateSpec(
            f"B3_lam{lam}_lr{_lr_str(lr)}_ep{ep}", "B3",
            (GroupUse("fine_target", "gd", lam), tl, rt),
            {"learning_rate": lr, "num_epochs": ep}))
    return grid


# ---------------------------------------------------------------------------
# Iteration 12 Stage 1 — the retention-aware B3 successor (method ``B4``).
#
# B4 is B3 with one difference: the ``retain`` group is the FIT HALF of the
# retained entities, so the probe half is never rehearsed by any candidate and
# the retention floor can be measured against B0 on knowledge neither model
# was trained on.  Its weight is the swept knob.
#
# Stage 1 holds everything else at the incumbent's values — lambda 0.5, and
# lr/epochs at 2e-5/5 except where a row exists specifically to test the
# budget — so that a difference between rows is attributable to replay
# strength rather than to three knobs moving at once.
# ---------------------------------------------------------------------------

#: Fine suppression is held at the incumbent's weight across every Stage-1
#: row.  Sweeping it too would confound the mechanism under test.
ITER12_LAM = 0.5

#: Weight 1.0 on the fit half IS the incumbent recipe retrained under the
#: Iteration-12 partition, so it doubles as the reference row.
ITER12_REFERENCE_WEIGHT = 1.0

#: Halving the replay group also cuts its share of every epoch's interleaved
#: gradient stream — 387/(2*90+387) = 0.6825 for the pilot-100 group,
#: 183/(2*90+183) = 0.5041 for the fit half — so weight 1.0 does NOT
#: reproduce the incumbent's effective retain influence.  The ratio of the two
#: shares does, and it is DERIVED from the committed partition rather than
#: chosen: ``build_iter12_retention_probe.py`` records it as
#: ``replay_strength.weight_reproducing_the_pilot_influence`` and a test
#: recomputes it from the group counts.  Including it makes the sweep span the
#: pilot's operating point instead of guessing where that point landed.
ITER12_PILOT_EQUIVALENT_WEIGHT = 1.3539

#: The replay weights swept at the incumbent budget.
ITER12_WEIGHTS = (0.5, ITER12_REFERENCE_WEIGHT,
                  ITER12_PILOT_EQUIVALENT_WEIGHT, 2.0, 4.0)


def _b4_id(weight: float, lr: float, epochs: int) -> str:
    return f"B4_w{weight}_lam{ITER12_LAM}_lr{_lr_str(lr)}_ep{epochs}"


#: The row the Stage-2 gate compares the winner against: the incumbent recipe
#: under the Iteration-12 partition, i.e. replay weight 1.0.
ITER12_REFERENCE_ROW = _b4_id(ITER12_REFERENCE_WEIGHT, 2e-5, 5)


def iter12_grid() -> list[CandidateSpec]:
    """9 rows: B0 (the floor) + 8 B4 replay-strength candidates."""
    ft = GroupUse("fine_target", "gd", ITER12_LAM)
    tl = GroupUse("target_level", "sft", 1.0)
    grid: list[CandidateSpec] = [CandidateSpec("B0", "B0", noop=True)]

    #: The weight sweep at the incumbent budget.  0.5 and 4.0 bracket the
    #: pilot's effective influence from below and above; 1.3539 lands on it.
    for w in ITER12_WEIGHTS:
        grid.append(CandidateSpec(
            _b4_id(w, 2e-5, 5), "B4",
            (ft, tl, GroupUse("retain", "sft", w)),
            {"learning_rate": 2e-5, "num_epochs": 5}))

    #: Budget sensitivity at a mid weight.  A heavier replay term may need a
    #: shorter schedule to avoid overwriting the target-side objective, or a
    #: longer one to converge; both directions are swept rather than assumed.
    for ep in (3, 8):
        grid.append(CandidateSpec(
            _b4_id(2.0, 2e-5, ep), "B4",
            (ft, tl, GroupUse("retain", "sft", 2.0)),
            {"learning_rate": 2e-5, "num_epochs": ep}))

    #: One learning-rate step, so a null result cannot be blamed entirely on
    #: the schedule the incumbent happened to use.
    grid.append(CandidateSpec(
        _b4_id(2.0, 5e-5, 5), "B4",
        (ft, tl, GroupUse("retain", "sft", 2.0)),
        {"learning_rate": 5e-5, "num_epochs": 5}))
    return grid


GRIDS = {"smoke": smoke_grid, "pilot100": pilot100_grid,
         "iter12": iter12_grid}


def grid_for_tag(tag: str) -> list[CandidateSpec]:
    if tag not in GRIDS:
        raise ValueError(f"unknown grid tag {tag!r}; expected one of "
                         f"{sorted(GRIDS)}")
    return GRIDS[tag]()


def dataset_dir_for_tag(tag: str) -> str:
    """Tagged dataset directory, repo-relative.

    ``iter12`` deliberately resolves to the pilot-100 dataset: Iteration 12
    changes which knowledge is rehearsed, not which queries are asked, so it
    keeps pilot-100's queries, associations and ``pilot100_v2`` version and
    its candidates stay directly comparable with the selection they succeed.
    It also leaves the frozen confirmation dataset fingerprint untouched.
    """
    if tag == "iter12":
        return "data/mllmu_hier_pilot100"
    return f"data/mllmu_hier_{tag}"


def groups_subdir_for_tag(tag: str) -> str:
    """Which unlearning-group directory a tag's candidates are fitted on.

    ``iter12`` trains on ``unlearning_iter12/``, whose ``fine_target`` and
    ``target_level`` are byte-identical copies of the pilot-100 ones and
    whose ``retain`` holds only the FIT half of the retained entities.  The
    separation is physical rather than a flag, so a candidate cannot rehearse
    the probe half by accident.
    """
    return "unlearning_iter12" if tag == "iter12" else "unlearning"


def validate_grid(grid: list[CandidateSpec]) -> list[str]:
    """Structural invariants every grid must satisfy."""
    errors: list[str] = []
    ids = [c.candidate_id for c in grid]
    if len(ids) != len(set(ids)):
        dupes = sorted({i for i in ids if ids.count(i) > 1})
        errors.append(f"duplicate candidate ids: {dupes}")
    if sum(1 for c in grid if c.method == "B0") != 1:
        errors.append("exactly one B0 (no-op) candidate is required")
    for c in grid:
        if c.method not in METHODS:
            errors.append(f"{c.candidate_id}: unknown method {c.method}")
        if c.noop:
            if c.groups or c.overrides:
                errors.append(
                    f"{c.candidate_id}: a no-op candidate takes no groups "
                    f"and no overrides")
            continue
        if not c.groups:
            errors.append(f"{c.candidate_id}: no knowledge groups")
        for g in c.groups:
            if g.mode not in ("sft", "gd"):
                errors.append(f"{c.candidate_id}: bad mode {g.mode}")
            if g.weight <= 0:
                errors.append(
                    f"{c.candidate_id}/{g.name}: weight must be > 0 (use "
                    f"a smaller positive weight, not 0 — a 0-weight group "
                    f"still consumes the interleaved stream)")
        names = [g.name for g in c.groups]
        if len(names) != len(set(names)):
            errors.append(f"{c.candidate_id}: repeated group {names}")
        allowed = set(
            {"learning_rate", "num_epochs", "lora_r", "lora_alpha",
             "lora_dropout", "weight_decay", "seed"}
        )
        unknown = set(c.overrides) - allowed
        if unknown:
            errors.append(
                f"{c.candidate_id}: overrides outside the swept-knob "
                f"allowlist: {sorted(unknown)}")
    return errors
