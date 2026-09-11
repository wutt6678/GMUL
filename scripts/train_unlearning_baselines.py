"""Train the MF -> MU unlearning candidates (Iteration 9 / Iteration 11).

    python scripts/train_unlearning_baselines.py --tag pilot100 \
        --device cuda:0 --candidates B0,B1_lr1e-05_ep10

Every candidate continues from the SAME canonical MF adapter of the
tagged dataset, so the only difference between candidates is the swept
knob set declared in :mod:`granunlearn.training.candidate_grid`.

Methods:
* B0  no-op:                MF adapter copied unchanged (sanity baseline)
* B1  complete-forget:      gradient ascent on fine_target
* B2  coarse-positive SFT:  SFT on target_level completions
* B2R B2 + explicit retain SFT group (Iteration 10 finding)
* B3  granularity-aware:    gd fine_target + sft target_level + sft retain

Sweep knobs are tuned on train/val probes ONLY by
scripts/select_unlearning_checkpoints.py; the frozen test split is a
genuine held-out evaluation.

``--candidates`` takes candidate ids (comma-separated) or method letters,
so several GPU lanes can split one grid without touching it.  Lanes must
never share a candidate id.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from granunlearn.config import _find_repo_root
from granunlearn.logging_utils import setup_logger
from granunlearn.training.candidate_grid import (
    CandidateSpec,
    dataset_dir_for_tag,
    grid_for_tag,
    groups_subdir_for_tag,
    validate_grid,
)
from granunlearn.training.reference_trainer import ReferenceRecipe
from granunlearn.training.unlearning_trainer import (
    GroupSpec,
    make_noop_checkpoint,
    train_unlearning,
)

log = setup_logger("train_unlearning_baselines")


def resolve_paths(tag: str, repo_root: Path) -> tuple[Path, Path, Path, Path]:
    """(dataset_dir, mf_adapter_dir, candidate_output_root, groups_dir).

    ``groups_dir`` is separate from ``dataset_dir`` because Iteration 12
    reuses the pilot-100 dataset while training on a different set of
    knowledge groups: its ``retain`` group is the fit half of the retained
    entities, so the probe half is never rehearsed.  Resolving the groups
    through the tag rather than hardcoding ``<dataset>/unlearning`` is what
    keeps that separation out of the caller's hands.
    """
    dataset_dir = repo_root / dataset_dir_for_tag(tag)
    #: Iteration 12 continues from the pilot-100 MF adapter: there is no
    #: separate ``mllmu_iter12`` reference-state training, and inventing one
    #: would make its candidates incomparable with the incumbent's.
    mf_tag = "pilot100" if tag == "iter12" else tag
    mf_adapters = repo_root / "data" / "checkpoints" / \
        f"mllmu_{mf_tag}" / "MF" / "adapters"
    out_root = repo_root / "data" / "checkpoints" / f"mllmu_{tag}_unlearn"
    groups_dir = dataset_dir / groups_subdir_for_tag(tag)
    return dataset_dir, mf_adapters, out_root, groups_dir


def groups_builder_hint(tag: str) -> str:
    """Which script produces the group files a tag trains on."""
    if tag == "iter12":
        return "scripts/build_iter12_retention_probe.py"
    return f"scripts/build_unlearning_groups.py --tag {tag}"


def select_candidates(grid: list[CandidateSpec],
                      selector: str | None) -> list[CandidateSpec]:
    """Filter a grid by candidate id or method letter.  ``None``/``all``
    keeps the whole grid; order follows the grid, never the selector."""
    if not selector or selector.strip().lower() in ("all", "*"):
        return list(grid)
    tokens = [s.strip() for s in selector.split(",") if s.strip()]
    ids = {c.candidate_id for c in grid}
    methods = {c.method for c in grid}
    wanted_ids: set[str] = set()
    wanted_methods: set[str] = set()
    for token in tokens:
        if token in ids:
            wanted_ids.add(token)
        elif token.upper() in methods:
            wanted_methods.add(token.upper())
        else:
            raise SystemExit(
                f"--candidates token {token!r} matches neither a candidate "
                f"id nor a method letter; grid ids: {sorted(ids)}")
    return [c for c in grid
            if c.candidate_id in wanted_ids or c.method in wanted_methods]


def mf_inherited_recipe(mf_adapters: Path) -> ReferenceRecipe:
    """The recipe MF was actually trained with, read back from its summary.

    B0 copies MF's adapter and applies no updates of its own, so the
    recipe it reports is an INHERITED one rather than a swept one.  Read
    it from the checkpoint instead of assuming the dataclass default, so
    a recipe change in a later iteration cannot silently desynchronise
    B0's provenance from the weights it carries.
    """
    summary = mf_adapters.parent / "training_summary.json"
    if summary.exists():
        recorded = json.loads(summary.read_text()).get("recipe")
        if recorded:
            return ReferenceRecipe.from_dict(recorded)
    log.warning("MF summary has no recipe (%s) — falling back to the "
                "ReferenceRecipe default", summary)
    return ReferenceRecipe()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train MF->MU baseline candidates")
    parser.add_argument("--tag", default="smoke",
                        choices=("smoke", "pilot100", "iter12"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--methods", default=None,
                        help="DEPRECATED alias for --candidates")
    parser.add_argument("--candidates", default=None,
                        help="Comma-separated candidate ids and/or method "
                             "letters (default: the whole grid)")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override epoch budget for all candidates")
    parser.add_argument("--repo-root", default=None)
    args = parser.parse_args()

    repo_root = Path(args.repo_root) if args.repo_root \
        else (_find_repo_root(Path.cwd()) or Path.cwd())
    dataset_dir, mf_adapters, out_root, groups_dir = resolve_paths(
        args.tag, repo_root)
    if not mf_adapters.exists():
        raise FileNotFoundError(f"Canonical MF adapter missing: {mf_adapters}")
    #: Logged because for ``iter12`` these four paths deliberately do not all
    #: carry the same tag: the dataset and the MF adapter are pilot-100's, the
    #: groups are the fit-half ones, and only the output root is iter12's.  A
    #: run log that did not say so could not be checked afterwards.
    log.info("[%s] dataset %s | groups %s | MF %s | out %s", args.tag,
             dataset_dir, groups_dir, mf_adapters, out_root)

    grid = grid_for_tag(args.tag)
    errors = validate_grid(grid)
    if errors:
        raise ValueError(f"{args.tag} grid is invalid: {errors}")
    selector = args.candidates or args.methods
    chosen = select_candidates(grid, selector)
    log.info("[%s] %d/%d candidates on %s: %s", args.tag, len(chosen),
             len(grid), args.device, [c.candidate_id for c in chosen])

    for spec in chosen:
        out_dir = out_root / spec.candidate_id
        if spec.noop:
            make_noop_checkpoint(spec.candidate_id, mf_adapters, out_dir,
                                 recipe=mf_inherited_recipe(mf_adapters))
            continue
        groups = [
            GroupSpec(g.name, groups_dir / f"{g.name}.jsonl",
                      g.mode, g.weight)
            for g in spec.groups
        ]
        for g in groups:
            if not Path(g.path).exists():
                raise FileNotFoundError(
                    f"{g.path} — run {groups_builder_hint(args.tag)} first")
        overrides = dict(spec.overrides)
        if args.epochs is not None:
            overrides["num_epochs"] = args.epochs
        train_unlearning(
            method_id=spec.candidate_id,
            groups=groups,
            output_dir=out_dir,
            device=args.device,
            recipe=ReferenceRecipe(**overrides),
            init_adapter_dir=mf_adapters,
        )
    log.info("All requested candidates trained -> %s", out_root)


if __name__ == "__main__":
    main()
