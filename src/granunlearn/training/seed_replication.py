"""Iteration 12 Stage 1b: seed replicates of the two near-miss candidates.

Stage 1 disqualified all eight B4 replay candidates against a retention floor
measured on the probe half of the retained entities.  Two came close enough that
a single training seed cannot tell a real shortfall from an unlucky one:

    B4_w4.0_lam0.5_lr2e-05_ep5   retain-same row-micro 0.5221 vs B0 0.5294
                                 -- 3 queries out of 408
    B4_w2.0_lam0.5_lr2e-05_ep3   retain-same row-micro 0.5196 vs B0 0.5294
                                 -- 4 queries out of 408

This module defines the replicate set for those two, so the floor can be asked
of a MEAN over independent training seeds instead of one draw.

WHY THIS IS A NEW MODULE AND NOT A WIDER GRID

``candidate_grid.py`` is one of the eight paths hash-bound by the Stage-1
protocol freeze, and that freeze now refuses to be rewritten because nine
candidates have adapters on disk.  Adding replicate rows to ``iter12_grid()``
would therefore permanently break ``--check-only`` and the selector's own
pre-generation freeze verification.  So the replicates live here instead.
Nothing frozen is edited and nothing here is imported by a frozen file; the
replicates reuse the frozen trainer, the frozen retention arithmetic and the
sealed generation path BY IMPORT, which is how the Stage-1 selector already
reuses the sealed pilot-100 selector.

WHAT A SEED ACTUALLY VARIES, MEASURED NOT ASSUMED

``recipe.seed`` is consumed in two places in the frozen trainer:
``set_recipe_seeds(recipe.seed)`` (LoRA initialisation and dropout) and
``random.Random(f"{recipe.seed}:{epoch}:{gi}").shuffle(order)`` (the per-group,
per-epoch order of the interleaved stream).  A seed override therefore changes
initialisation, dropout and data order -- and NOT which examples are trained on,
so the fit/probe partition is untouched by replication.

WHAT A SEED DOES NOT VARY

The floor's four numbers are computed over 408 + 76 probe queries that are
100% ``text_to_text`` route.  Comparing two existing B0 parquets generated from
IDENTICAL adapter bytes under different ``image_batch_size`` (8 in the pilot
run, 1 in the Stage-1 run) gives 15 correctness flips out of 4,518 rows -- all
of them on image-route queries, and NONE inside either probe retention subset.
Generation instability is therefore real, confined to the image route, and does
not reach the floor.  Replicate retention differences are attributable to the
training seed alone, which is what makes this a one-factor design.

The same comparison moved the full train+val vector's ``filr`` by -0.0004 and
``wrong`` by +0.0008, so D_G -- which does read image-route components --
carries generation noise of order 2e-4.  That is ~17x smaller than the gap
between Stage 1's two best candidates (0.027343 vs 0.030686), so it does not
reorder them, but it does mean D_G's sixth decimal is not physically
meaningful.  Recorded here because the frozen tie-break compares distances at
exactly six decimals.
"""

from __future__ import annotations

import statistics
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from granunlearn.training.candidate_grid import (
    CandidateSpec,
    dataset_dir_for_tag,
    grid_for_tag,
)

#: The two Stage-1 rows replicated.  Chosen BEFORE any replicate was trained,
#: from the filed Stage-1 report, as the two candidates whose retain-same
#: shortfall against B0 is smallest in query counts (3 and 4 of 408).  They are
#: not the two best on D_G -- that would be selecting on the criterion rather
#: than on the reason a single seed cannot decide them.
PARENTS: tuple[str, ...] = (
    "B4_w4.0_lam0.5_lr2e-05_ep5",
    "B4_w2.0_lam0.5_lr2e-05_ep3",
)

#: The seed every Stage-1 candidate was trained with.  Its adapter and
#: predictions already exist and are sidecar-verified, so it is REUSED as the
#: first replicate rather than retrained: re-running it would produce a second
#: draw from the same seed and tell us nothing about between-seed variance.
BASE_SEED = 42

#: The three additional seeds.  Consecutive integers after the recipe default,
#: recorded here before training so they cannot be chosen after seeing results.
NEW_SEEDS: tuple[int, ...] = (43, 44, 45)

ALL_SEEDS: tuple[int, ...] = (BASE_SEED, *NEW_SEEDS)

#: Where the reused seed-42 evidence lives, and where replicate evidence goes.
#: Separate roots so a replicate can never overwrite a Stage-1 artifact.
STAGE1_CKPT_ROOT = "data/checkpoints/mllmu_iter12_unlearn"
STAGE1_PREDICTIONS_SUBDIR = "predictions_iter12"
SEED_CKPT_ROOT = "data/checkpoints/mllmu_iter12_seeds"
SEED_PREDICTIONS_SUBDIR = "predictions_iter12_seeds"

#: The generation contract is inherited from the Stage-1 freeze, not restated
#: here, so a replicate cannot be generated under a different batch layout than
#: the anchor it is compared against.
STAGE1_FREEZE_REPORT = "data/reports/mllmu_iter12_selection_protocol_freeze.json"
STAGE1_SELECTION_REPORT = "data/reports/mllmu_iter12_retention_selection.json"
PROBE_REPORT = "data/reports/mllmu_iter12_retention_probe.json"

#: Two reports, and they must not share a path.  The freeze is the record that
#: this protocol predates its own replicates; the analysis report is the result.
#: Pointing both at one file would let the analyzer overwrite the freeze with
#: the result it was frozen to produce -- which is the exact failure the
#: overwrite refusal exists to prevent, arrived at from the other direction.
FREEZE_REPORT = "data/reports/mllmu_iter12_seed_replication_freeze.json"
OUT_REPORT = "data/reports/mllmu_iter12_seed_replication.json"

#: The four frozen numbers the floor reads, as (family, estimand) pairs.
FLOOR_NUMBERS: tuple[tuple[str, str], ...] = (
    ("retain_same_entity", "row_micro"),
    ("retain_same_entity", "entity_macro"),
    ("retain_other_entity", "row_micro"),
    ("retain_other_entity", "entity_macro"),
)


@dataclass(frozen=True)
class Replicate:
    """One (parent, seed) training run.

    ``reused`` marks the seed-42 replicate whose adapter and predictions were
    produced by Stage 1 and are read from there rather than regenerated.
    """

    replicate_id: str
    parent: str
    seed: int
    spec: CandidateSpec
    reused: bool

    @property
    def overrides(self) -> dict[str, Any]:
        """The parent's frozen overrides with this replicate's seed.

        The parent's learning rate, epoch budget and group weights are read
        from the frozen grid, never retyped here, so a replicate cannot drift
        from the candidate it replicates.
        """
        out = dict(self.spec.overrides)
        out["seed"] = self.seed
        return out


def parent_specs() -> dict[str, CandidateSpec]:
    """The two parents, looked up in the frozen Stage-1 grid by id."""
    grid = {c.candidate_id: c for c in grid_for_tag("iter12")}
    missing = [p for p in PARENTS if p not in grid]
    if missing:
        raise LookupError(
            f"replicate parents {missing} are not in the frozen iter12 grid; "
            f"a replicate of a candidate that was never frozen is not a "
            f"replicate")
    return {p: grid[p] for p in PARENTS}


def replicate_id(parent: str, seed: int) -> str:
    return f"{parent}__s{seed}"


# ── path resolution ───────────────────────────────────────────────────
#: Every path this study touches is resolved HERE and nowhere else.  Two call
#: sites building the same path independently is how ``data/data/...`` got into
#: the determinism control: ``dataset_dir_for_tag`` already returns a
#: ``data/``-prefixed relative path, and prepending ``"data"`` again is invisible
#: in the source and only appears when the file is opened.  A single owner makes
#: that disagreement impossible rather than unlikely.


def dataset_dir(repo_root):
    """The pilot-100 dataset this study reuses (already ``data/``-prefixed)."""
    return repo_root / dataset_dir_for_tag("iter12")


def stage1_predictions_dir(repo_root):
    """Where Stage 1 filed its predictions, including the B0 anchor."""
    return dataset_dir(repo_root) / STAGE1_PREDICTIONS_SUBDIR


def seed_predictions_dir(repo_root):
    """This study's own predictions directory."""
    return dataset_dir(repo_root) / SEED_PREDICTIONS_SUBDIR


def stage1_ckpt_root(repo_root):
    return repo_root / STAGE1_CKPT_ROOT


def seed_ckpt_root(repo_root):
    return repo_root / SEED_CKPT_ROOT


def replicates() -> list[Replicate]:
    """All (parent, seed) replicates, parents in frozen order, seeds ascending.

    Order is fixed by construction rather than by discovery, so a report's row
    order cannot depend on filesystem enumeration.
    """
    specs = parent_specs()
    out: list[Replicate] = []
    for parent in PARENTS:
        for seed in ALL_SEEDS:
            out.append(Replicate(
                replicate_id=replicate_id(parent, seed),
                parent=parent,
                seed=seed,
                spec=specs[parent],
                reused=(seed == BASE_SEED),
            ))
    return out


def to_train() -> list[Replicate]:
    """The replicates that need a GPU: everything except the reused seed."""
    return [r for r in replicates() if not r.reused]


def aggregate(values: Sequence[float]) -> dict[str, Any]:
    """Mean, dispersion and range of one number across replicates.

    The mean is reported and compared UNROUNDED.  Rounding it to the four
    decimals its inputs carry would let a rounding step decide a comparison the
    inputs did not decide -- the same reason the Stage-1 tolerance is derived
    from resolvable gaps rather than chosen.
    """
    vals = [float(v) for v in values]
    if not vals:
        raise ValueError("cannot aggregate zero replicates")
    return {
        "k": len(vals),
        "values": vals,
        "mean": statistics.fmean(vals),
        "min": min(vals),
        "max": max(vals),
        "sd": statistics.stdev(vals) if len(vals) > 1 else 0.0,
    }


def range_classification(values: Sequence[float], anchor: float,
                         epsilon: float) -> str:
    """Where the replicates sit relative to the anchor, non-parametrically.

    Deliberately not a confidence-interval rule: with k=4 replicates an
    interval would import a distributional assumption the design does not need.
    The observed range answers the question actually being asked -- did every
    independent training seed fall below the anchor, or do they straddle it?
    """
    vals = [float(v) for v in values]
    if max(vals) < anchor - epsilon:
        return "every_replicate_below"
    if min(vals) >= anchor - epsilon:
        return "every_replicate_at_or_above"
    return "straddles"


def mean_candidate(values_by_number: dict[tuple[str, str], Sequence[float]]
                   ) -> dict[str, Any]:
    """A per-seed retention dict reduced to its means, in ``probe_retention``'s
    own shape, so the FROZEN ``floor_check`` can score it unchanged.

    Building the dict the frozen function already expects -- rather than writing
    a new comparison -- is what keeps the replicate verdict arithmetically
    identical to the Stage-1 verdict it revisits.
    """
    out: dict[str, Any] = {}
    for (family, estimand), vals in values_by_number.items():
        out.setdefault(family, {})[estimand] = aggregate(vals)["mean"]
    return out
