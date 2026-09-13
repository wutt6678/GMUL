"""Train the Iteration 12 Stage-1b seed replicates.

    python scripts/train_iter12_seed_replicates.py --device cuda:0
    python scripts/train_iter12_seed_replicates.py \
        --candidates B4_w4.0_lam0.5_lr2e-05_ep5__s43

Each replicate is the parent candidate's FROZEN recipe with one field changed:
``seed``.  Learning rate, epoch budget and group weights are read from the
Stage-1 grid rather than restated, and the trained group files are the fit-half
ones, so a replicate differs from its parent in initialisation, dropout and
stream order and in nothing else.

The seed-42 replicate is NOT trained here: its adapter was produced by Stage 1
and is reused, because re-running the same seed would produce a second draw
from an identical RNG state and say nothing about between-seed variance.

Nothing frozen is edited.  ``train_unlearning`` and ``validate_grid`` are
imported, and the seed-replication freeze is verified before any GPU work, so a
replicate cannot be trained under a protocol that has since drifted.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import freeze_iter12_seed_replication as fisr

from granunlearn.config import _find_repo_root
from granunlearn.logging_utils import setup_logger
from granunlearn.training.candidate_grid import (
    CandidateSpec,
    dataset_dir_for_tag,
    groups_subdir_for_tag,
    validate_grid,
)
from granunlearn.training.reference_trainer import ReferenceRecipe
from granunlearn.training.seed_replication import (
    BASE_SEED,
    SEED_CKPT_ROOT,
    replicate_id,
    replicates,
    to_train,
)
from granunlearn.training.unlearning_trainer import (
    GroupSpec,
    train_unlearning,
)

log = setup_logger("train_iter12_seeds")


def as_candidate_specs(reps) -> list[CandidateSpec]:
    """Replicates expressed as ``CandidateSpec``, so the FROZEN ``validate_grid``
    checks them.

    Writing a second validator for a grid that differs from the frozen one by a
    single field would mean the replicates are checked by a rule nobody froze.
    ``seed`` is already on that validator's override allowlist.
    """
    return [CandidateSpec(candidate_id=r.replicate_id,
                          method=r.spec.method,
                          groups=tuple(r.spec.groups),
                          overrides=r.overrides,
                          noop=False) for r in reps]


def resolve_paths(repo_root: Path) -> tuple[Path, Path, Path, Path]:
    """(dataset_dir, mf_adapters, out_root, groups_dir).

    The tag is Stage 1's ``iter12``: replicates train on the pilot-100 dataset
    with the fit-half groups and continue from the pilot-100 MF adapter.  Only
    the output root differs, and it differs so that a replicate can never
    overwrite a Stage-1 adapter.
    """
    dataset_dir = repo_root / dataset_dir_for_tag("iter12")
    mf_adapters = repo_root / "data" / "checkpoints" / \
        "mllmu_pilot100" / "MF" / "adapters"
    out_root = repo_root / SEED_CKPT_ROOT
    groups_dir = dataset_dir / groups_subdir_for_tag("iter12")
    return dataset_dir, mf_adapters, out_root, groups_dir


def group_sizes(groups_dir: Path) -> dict[str, int]:
    return {p.stem: sum(1 for line in p.read_text().splitlines() if line.strip())
            for p in sorted(groups_dir.glob("*.jsonl"))}


def replicate_cost(rep, sizes: dict[str, int]) -> int:
    """Micro-batches: one per group example per epoch.

    The same cost model the Stage-1 planner uses, applied to the group files
    these replicates actually train on.  It puts the three ep5 replicates at
    5x363 and the three ep3 ones at 3x363, which is the 1609s / ~965s split
    Stage 1 measured.
    """
    epochs = int(rep.overrides.get("num_epochs", 10))
    return epochs * sum(sizes[g.name] for g in rep.spec.groups)


def plan_lanes(reps, sizes: dict[str, int], n_lanes: int) -> list[list]:
    """Longest-processing-time-first bin packing over the replicates.

    Order is LPT over a deterministic tie-break of replicate id, so the plan
    does not depend on dictionary or filesystem enumeration order.
    """
    if n_lanes < 1:
        raise ValueError("n_lanes must be >= 1")
    lanes: list[list] = [[] for _ in range(n_lanes)]
    loads = [0] * n_lanes
    for rep in sorted(reps, key=lambda r: (-replicate_cost(r, sizes),
                                           r.replicate_id)):
        i = min(range(n_lanes), key=lambda k: (loads[k], k))
        lanes[i].append(rep)
        loads[i] += replicate_cost(rep, sizes)
    return [lane for lane in lanes if lane]


def emit_plan(args, repo_root: Path) -> None:
    _, _, _, groups_dir = resolve_paths(repo_root)
    sizes = group_sizes(groups_dir)
    lanes = plan_lanes(to_train(), sizes, args.plan_lanes)
    total = sum(replicate_cost(r, sizes) for r in to_train())
    print(f"# group sizes: {json.dumps(sizes)}")
    print(f"# {total} micro-batches over {len(to_train())} replicate(s) "
          f"in {len(lanes)} lane(s)")
    for lane in lanes:
        ids = ",".join(r.replicate_id for r in lane)
        if args.emit_sh:
            print(ids)
        else:
            cost = sum(replicate_cost(r, sizes) for r in lane)
            print(f"lane: cost={cost} ({cost / max(total, 1):.0%}) "
                  f"n={len(lane)} -> {ids}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--candidates", default=None,
                    help="Comma-separated replicate ids (default: all that "
                         "still need training)")
    ap.add_argument("--repo-root", default=None)
    ap.add_argument("--force", action="store_true",
                    help="Retrain a replicate whose adapter already exists")
    ap.add_argument("--plan-lanes", type=int, default=None,
                    metavar="N", help="Print an N-lane plan and exit; no GPU")
    ap.add_argument("--emit-sh", action="store_true",
                    help="With --plan-lanes, print one --candidates value "
                         "per line")
    args = ap.parse_args()

    repo_root = Path(args.repo_root) if args.repo_root \
        else (_find_repo_root(Path.cwd()) or Path.cwd())

    if args.plan_lanes is not None:
        #: No freeze verification here: planning reads group files and the
        #: frozen grid and touches no GPU, so it is safe -- and useful -- to
        #: inspect a plan before the protocol is frozen.
        emit_plan(args, repo_root)
        return

    #: Verified BEFORE any GPU work, exactly as the Stage-1 selector verifies
    #: the Stage-1 freeze: a replicate trained under a protocol that has since
    #: drifted cannot be scored by it afterwards.
    reasons = fisr.verify_freeze(repo_root)
    if reasons:
        raise SystemExit(
            "REFUSING to train: the seed-replication freeze does not match the "
            "repository:\n  " + "\n  ".join(reasons))

    dataset_dir, mf_adapters, out_root, groups_dir = resolve_paths(repo_root)
    if not mf_adapters.exists():
        raise FileNotFoundError(f"canonical MF adapter missing: {mf_adapters}")
    log.info("dataset %s | groups %s | MF %s | out %s",
             dataset_dir, groups_dir, mf_adapters, out_root)

    reps = to_train()
    if args.candidates:
        wanted = {s.strip() for s in args.candidates.split(",") if s.strip()}
        known = {r.replicate_id for r in reps}
        reused = {replicate_id(p, BASE_SEED) for p in
                  (r.parent for r in replicates())}
        unknown = wanted - known - reused
        if unknown:
            raise SystemExit(
                f"--candidates {sorted(unknown)} matches no replicate that "
                "needs training; trainable ids: "
                f"{sorted(known)}")
        #: A reused replicate named explicitly is a no-op with a log line, not
        #: an error: a lane plan that lists it should say why it was skipped.
        reps = [r for r in reps if r.replicate_id in wanted]

    #: ``validate_grid`` is written for a complete candidate grid, so one of its
    #: checks is grid-structural rather than per-candidate: a grid must contain
    #: exactly one no-op baseline.  A set of replicates of two already-frozen
    #: candidates has no baseline of its own -- the anchor is read from the
    #: filed Stage-1 report -- so that one check does not apply and is the only
    #: error tolerated here.  Every PER-CANDIDATE check still runs under the
    #: frozen validator: group modes, positive weights, no repeated group, and
    #: overrides inside the swept-knob allowlist (which is where ``seed`` lives).
    all_errors = validate_grid(as_candidate_specs(reps))
    errors = [e for e in all_errors if "B0 (no-op)" not in e]
    if errors:
        raise ValueError(f"replicate specs are invalid: {errors}")
    unexpected = [e for e in all_errors if e not in errors]
    if unexpected != ["exactly one B0 (no-op) candidate is required"]:
        #: Guards the filter itself: if the frozen validator ever changes that
        #: message, or starts reporting a second structural error, this stops
        #: rather than quietly widening what is tolerated.
        raise ValueError(
            f"the frozen validator reported structural errors this script does "
            f"not recognise and will not filter: {unexpected}")

    log.info("%d replicate(s) on %s: %s", len(reps), args.device,
             [r.replicate_id for r in reps])
    for r in reps:
        out_dir = out_root / r.replicate_id
        existing = out_dir / "adapters" / "adapter_model.safetensors"
        if existing.exists() and not args.force:
            log.info("[%s] adapter exists — skipping (re-run with --force to "
                     "retrain)", r.replicate_id)
            continue
        groups = [GroupSpec(g.name, groups_dir / f"{g.name}.jsonl",
                            g.mode, g.weight) for g in r.spec.groups]
        for g in groups:
            if not Path(g.path).exists():
                raise FileNotFoundError(
                    f"{g.path} — run scripts/build_iter12_retention_probe.py "
                    f"first")
        recipe = ReferenceRecipe(**r.overrides)
        if recipe.seed != r.seed:
            #: The whole point of a replicate is that the seed reached the
            #: recipe, so this is asserted rather than assumed.
            raise AssertionError(
                f"{r.replicate_id}: recipe.seed is {recipe.seed}, not {r.seed}")
        log.info("[%s] parent %s seed %d lr %s epochs %s groups %s",
                 r.replicate_id, r.parent, r.seed, recipe.learning_rate,
                 recipe.num_epochs,
                 json.dumps([g.name for g in groups]))
        train_unlearning(
            method_id=r.replicate_id,
            groups=groups,
            output_dir=out_dir,
            device=args.device,
            recipe=recipe,
            init_adapter_dir=mf_adapters,
        )
    log.info("requested replicates trained -> %s", out_root)


if __name__ == "__main__":
    main()
