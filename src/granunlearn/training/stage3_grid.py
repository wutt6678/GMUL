"""Iteration 12 Stage 3 — the route-aware successor to B6.

Stage 2's result is narrow enough to act on without re-opening the whole
mechanism space.  ``B6_beta13.642_lam0.5_lr2e-05_ep5``:

* clears MG on all four text-route retention numbers;
* fails exactly three image-route numbers;
* is short by five of 408 image retain-same rows and two of 76 image
  retain-other rows (equivalently, ten of 816 and four of 152 rows across the
  two routes' train+val denominators, fourteen rows in total); and
* keeps the fine-target objective less suppressed than every bounded-ascent
  row (0.6857 nats against 0.7832-2.9924), so the target transformation is
  preserved rather than overshot.

Stage 3 therefore changes ONE coordinate: how strongly MF is preserved on the
image-conditioned fit-half retention stream.  It retains B6's fine-target
ascent, target-level objective, learning rate, epoch budget, seed, cache and
starting adapter unchanged.

WHAT "IMAGE-CONDITIONED" MEANS HERE
-----------------------------------
This is measured, not assumed.  Every one of the 183 examples in the frozen
fit-half ``retain.jsonl`` has ``modality == "image_text"`` and a non-empty
``image_path``, and the MF reference cache bound by Stage 2 covers exactly
those 183 examples.  There is no text-only fit-half retention example to
exclude.  The route-aware term is consequently an ADDITIONAL weight on the
image-conditioned stream, not a restriction to a smaller subset: a row with
additional weight ``gamma`` anchors that stream at total weight
``beta + gamma``.  The protocol says this plainly rather than letting the word
"subset" imply a selection that the dataset does not provide.

At ``gamma = 0`` the objective is byte-for-byte B6's objective: the same three
groups, modes, weights, paths, overrides, cache and recipe.  B6 is therefore
reused as the anchor-disabled control rather than retrained.  A same-seed
re-run would measure GPU nondeterminism; the useful control is that the
Stage-3 row maps exactly to B6 and that its bound prediction bytes reproduce
B6's filed eight numbers and D_G within the tolerances below.

Nothing in this module edits a frozen path.  ``stage2_grid`` supplies the B6
row it succeeds, ``preservation_trainer`` supplies the objective vocabulary,
and both are imported only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from granunlearn.training import stage2_grid as s2g
from granunlearn.training.candidate_grid import (
    ITER12_LAM,
    dataset_dir_for_tag,
    groups_subdir_for_tag,
    validate_grid,
)
from granunlearn.training.preservation_trainer import PreservationGroupSpec

TAG = "iter12"

#: Method label for the route-aware B6 successor.  B6 remains a method in its
#: own right because the zero-weight control is the filed Stage-2 row reused.
METHOD_ROUTE_ANCHOR = "B7"
STAGE3_METHODS: tuple[str, ...] = ("B0", s2g.METHOD_ANCHOR,
                                   METHOD_ROUTE_ANCHOR)

#: Additional image-conditioned anchor weight, as fractions of B6's beta.
#: Four settings, geometrically spaced from one eighth of beta to beta itself.
#: The sweep stays LOCAL to B6: it asks whether a modest increase on the image
#: route repairs a fourteen-query shortfall, rather than re-running Stage 2's
#: full mechanism search with a larger anchor.
IMAGE_WEIGHT_MULTIPLIERS: tuple[float, ...] = (0.125, 0.25, 0.5, 1.0)

#: Numerical tolerances for the zero-weight control.  The eight retention
#: values are filed at four decimals and must reproduce to float precision;
#: D_G is filed at six decimals, so one unit in its last place is allowed.
CONTROL_FLOOR_TOLERANCE = 1e-9
CONTROL_DG_TOLERANCE = 1e-6

#: Reports and artifact roots.  Stage 3 never writes into Stage 1's or
#: Stage 2's checkpoint or prediction namespaces.
STAGE3_CKPT_ROOT = "data/checkpoints/mllmu_iter12_stage3"
STAGE3_PREDICTIONS_SUBDIR = "predictions_iter12_stage3"
BASIS_REPORT = "data/reports/mllmu_iter12_stage3_basis.json"
CONTROL_REPORT = "data/reports/mllmu_iter12_stage3_control.json"
FREEZE_REPORT = "data/reports/mllmu_iter12_stage3_freeze.json"
OUT_REPORT = "data/reports/mllmu_iter12_stage3_selection.json"


@dataclass(frozen=True)
class Stage3CandidateSpec:
    """One Stage-3 row.

    ``image_anchor_weight`` is the additional route-aware weight ``gamma``.
    ``effective_anchor_weight`` is the coefficient actually passed to the
    frozen preservation trainer on the image-conditioned retain stream,
    ``beta + gamma``.  Keeping both in the spec is what makes the zero-weight
    control checkable: at ``gamma == 0`` the effective coefficient is B6's
    beta and the row must be indistinguishable from the B6 parent.
    """

    candidate_id: str
    method: str
    groups: tuple[PreservationGroupSpec, ...] = ()
    overrides: dict[str, Any] = field(default_factory=dict)
    noop: bool = False
    image_anchor_weight: float | None = None
    effective_anchor_weight: float | None = None
    reused_from: str | None = None

    def describe(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "method": self.method,
            "noop": self.noop,
            "groups": [g.describe() for g in self.groups],
            "overrides": dict(self.overrides),
            "image_anchor_weight": self.image_anchor_weight,
            "effective_anchor_weight": self.effective_anchor_weight,
            "reused_from": self.reused_from,
        }

    @property
    def beta(self) -> float | None:
        """The coefficient passed to the anchor, matching Stage 2's name."""
        return self.effective_anchor_weight


def _weight_token(x: float) -> str:
    """Filesystem-safe weight text, with trailing zeros removed."""
    return f"{x:.4f}".rstrip("0").rstrip(".")


def route_anchor_id(image_weight: float, beta: float, lr: float,
                    epochs: int) -> str:
    return (f"B7_imgw{_weight_token(image_weight)}_beta{_weight_token(beta)}"
            f"_lam{ITER12_LAM}_lr{lr:g}_ep{epochs}")


def b6_parent(beta_star: float) -> s2g.Stage2CandidateSpec:
    """The filed Stage-2 B6 row, read from the frozen Stage-2 grid."""
    matches = [c for c in s2g.stage2_grid(beta_star)
               if c.method == s2g.METHOD_ANCHOR
               and c.candidate_id.startswith("B6_beta")]
    if len(matches) != 1:
        raise LookupError(
            f"expected exactly one B6 parent in the Stage-2 grid, got "
            f"{[c.candidate_id for c in matches]}")
    return matches[0]


def b6_control_row(beta_star: float) -> Stage3CandidateSpec:
    """The zero-additional-weight control: B6 reused, not retrained."""
    parent = b6_parent(beta_star)
    return Stage3CandidateSpec(
        candidate_id=parent.candidate_id,
        method=parent.method,
        groups=parent.groups,
        overrides=dict(parent.overrides),
        image_anchor_weight=0.0,
        effective_anchor_weight=parent.beta,
        reused_from="stage2")


def same_objective_as_b6(spec: Stage3CandidateSpec,
                         parent: s2g.Stage2CandidateSpec) -> bool:
    """Whether a Stage-3 row is exactly B6's objective.

    The comparison is over every decision-bearing field, not over the id: the
    three groups with their paths, modes, weights, cap and anchor fields, and
    the recipe overrides.  This is the structural half of the zero-weight
    control.
    """
    if spec.image_anchor_weight != 0.0:
        return False
    if spec.overrides != parent.overrides:
        return False
    if len(spec.groups) != len(parent.groups):
        return False
    for got, want in zip(spec.groups, parent.groups, strict=True):
        if (got.name, str(got.path), got.mode, got.weight, got.cap,
                got.anchor_weight) != (want.name, str(want.path), want.mode,
                                       want.weight, want.cap,
                                       want.anchor_weight):
            return False
    return True


def stage3_grid(beta_star: float) -> list[Stage3CandidateSpec]:
    """B0 + the B6 zero-weight control + four trained B7 rows."""
    if beta_star <= 0:
        raise ValueError(
            f"beta_star must be positive, got {beta_star}; Stage 3 derives "
            f"its image-anchor sweep from B6's calibrated strength")
    beta = round(beta_star, 4)
    parent = b6_parent(beta_star)
    control = b6_control_row(beta_star)
    if not same_objective_as_b6(control, parent):
        raise AssertionError(
            "the zero-weight Stage-3 control does not map exactly to B6; "
            "training against it would not be a B6 reproduction control")

    grid: list[Stage3CandidateSpec] = [
        Stage3CandidateSpec("B0", "B0", noop=True, reused_from="stage1"),
        control,
    ]
    for multiplier in IMAGE_WEIGHT_MULTIPLIERS:
        gamma = round(beta * multiplier, 4)
        effective = round(beta + gamma, 4)
        groups = tuple(
            PreservationGroupSpec(
                g.name, g.path, g.mode,
                effective if g.name == "retain" else g.weight,
                cap=g.cap, anchor_weight=g.anchor_weight)
            for g in parent.groups)
        grid.append(Stage3CandidateSpec(
            route_anchor_id(gamma, beta, s2g.STAGE2_LR, s2g.STAGE2_EPOCHS),
            METHOD_ROUTE_ANCHOR,
            groups,
            dict(parent.overrides),
            image_anchor_weight=gamma,
            effective_anchor_weight=effective))
    return grid


#: The two error classes the Stage-1 validator cannot know about: B6/B7 were
#: not in its frozen method tuple and ``anchor`` was not in its frozen mode
#: vocabulary.  Only those exact gaps may be filtered.
FROZEN_VALIDATOR_GAP = ("unknown method", "bad mode")


def validate_stage3_grid(grid: list[Stage3CandidateSpec],
                         beta_star: float) -> tuple[list[str], list[str]]:
    """``(errors, filtered)`` — the residual must be empty.

    The frozen validator still checks duplicate ids, exactly one no-op row,
    positive weights, repeated groups and the override allowlist.  Stage 3 adds
    the checks that define its own design: one B6 zero-weight control, four
    B7 rows at the frozen weights, and no drift from B6's objective.
    """
    raw = validate_grid(grid)  # type: ignore[arg-type]
    kept, filtered = [], []
    for msg in raw:
        if any(gap in msg for gap in FROZEN_VALIDATOR_GAP):
            filtered.append(msg)
        else:
            kept.append(msg)

    parent = b6_parent(beta_star)
    controls = [c for c in grid if c.reused_from == "stage2"]
    b7 = [c for c in grid if c.method == METHOD_ROUTE_ANCHOR]
    if len(controls) != 1:
        kept.append(f"expected exactly one B6 zero-weight control, got "
                    f"{len(controls)}")
    elif not same_objective_as_b6(controls[0], parent):
        kept.append(f"{controls[0].candidate_id}: the zero-weight control "
                    f"does not match B6's objective")
    if len(b7) != len(IMAGE_WEIGHT_MULTIPLIERS):
        kept.append(f"expected {len(IMAGE_WEIGHT_MULTIPLIERS)} B7 rows, got "
                    f"{len(b7)}")
    expected_gammas = [round(round(beta_star, 4) * m, 4)
                       for m in IMAGE_WEIGHT_MULTIPLIERS]
    if sorted(c.image_anchor_weight for c in b7) != sorted(expected_gammas):
        kept.append(f"B7 image weights {sorted(c.image_anchor_weight for c in b7)} "
                    f"!= frozen {sorted(expected_gammas)}")
    for c in grid:
        if c.noop:
            continue
        if c.method not in STAGE3_METHODS:
            kept.append(f"{c.candidate_id}: Stage-3 method {c.method!r} is "
                        f"not one of {list(STAGE3_METHODS)}")
        names = [g.name for g in c.groups]
        if names != ["fine_target", "target_level", "retain"]:
            kept.append(f"{c.candidate_id}: B6's three groups must remain "
                        f"fine_target, target_level, retain, got {names}")
        modes = {g.name: g.mode for g in c.groups}
        if modes.get("fine_target") != "gd" or \
                modes.get("target_level") != "sft" or \
                modes.get("retain") != "anchor":
            kept.append(f"{c.candidate_id}: B6's modes changed: {modes}")
        weights = {g.name: g.weight for g in c.groups}
        if weights.get("fine_target") != ITER12_LAM or \
                weights.get("target_level") != 1.0:
            kept.append(f"{c.candidate_id}: B6's non-anchor weights changed: "
                        f"{weights}")
        if c.effective_anchor_weight is None or c.effective_anchor_weight <= 0:
            kept.append(f"{c.candidate_id}: anchored row needs a positive "
                        f"effective anchor weight")
    return kept, filtered


# ── path resolution ───────────────────────────────────────────────────
#: Every path this study touches is resolved HERE and nowhere else.  This is
#: the same single-owner rule as Stage 2, for the same reason: a doubled
#: ``data/`` component is invisible in source and only fails inside an
#: unattended GPU lane.


def _guard(rel: str) -> str:
    if rel.startswith("data/data/") or "//" in rel:
        raise ValueError(
            f"{rel!r} has a doubled path component. dataset_dir_for_tag and "
            f"the checkpoint roots already carry their own prefix; join them "
            f"once, through these resolvers.")
    return rel


def dataset_dir(repo_root: Path) -> Path:
    return repo_root / _guard(dataset_dir_for_tag(TAG))


def groups_dir(repo_root: Path) -> Path:
    """The fit-half groups B6 used.  The probe half is not here at all."""
    return dataset_dir(repo_root) / groups_subdir_for_tag(TAG)


def retain_group_path(repo_root: Path) -> Path:
    return groups_dir(repo_root) / "retain.jsonl"


def stage3_ckpt_root(repo_root: Path) -> Path:
    out = repo_root / _guard(STAGE3_CKPT_ROOT)
    for other in (s2g.stage2_ckpt_root(repo_root),
                  s2g.stage1_ckpt_root(repo_root)):
        if out.resolve() == other.resolve():
            raise ValueError(
                f"{out} is an earlier stage's checkpoint root. Stage 3 writes "
                f"nowhere that filed adapters already live.")
    return out


def stage3_predictions_dir(repo_root: Path) -> Path:
    out = dataset_dir(repo_root) / STAGE3_PREDICTIONS_SUBDIR
    for other in (s2g.stage1_predictions_dir(repo_root),
                  s2g.stage2_predictions_dir(repo_root),
                  dataset_dir(repo_root) / "predictions",
                  dataset_dir(repo_root) / "predictions_iter12_seeds"):
        if out.resolve() == other.resolve():
            raise ValueError(
                f"{out} is an earlier study's predictions directory. Stage 3 "
                f"reuses B0, MG and B6 by READING them; it never writes over "
                f"their evidence.")
    return out


def mf_adapters(repo_root: Path) -> Path:
    return s2g.mf_adapters(repo_root)


def reference_cache_dir(repo_root: Path) -> Path:
    """The image-conditioned MF cache B6 also used."""
    return s2g.reference_cache_dir(repo_root)


def stage1_predictions_dir(repo_root: Path) -> Path:
    return s2g.stage1_predictions_dir(repo_root)


def stage2_predictions_dir(repo_root: Path) -> Path:
    return s2g.stage2_predictions_dir(repo_root)


def trained_rows(grid: list[Stage3CandidateSpec]) -> list[Stage3CandidateSpec]:
    """Rows needing a GPU: B0 and the B6 control are reused, never trained."""
    return [c for c in grid if not c.noop and c.reused_from is None]


def b7_rows(grid: list[Stage3CandidateSpec]) -> list[Stage3CandidateSpec]:
    return [c for c in grid if c.method == METHOD_ROUTE_ANCHOR]


def control_row(grid: list[Stage3CandidateSpec]) -> Stage3CandidateSpec:
    matches = [c for c in grid if c.reused_from == "stage2"]
    if len(matches) != 1:
        raise LookupError(f"expected one B6 control row, got {len(matches)}")
    return matches[0]


def image_conditioned(example: dict[str, Any]) -> bool:
    """The frozen image-anchor predicate over a retain-group jsonl row."""
    return example.get("modality") == "image_text" and \
        bool(example.get("image_path"))
