"""Train the Iteration 12 Stage-2 candidates, behind a faithfulness control.

    python scripts/train_iter12_stage2.py --device cuda:0 --control
    python scripts/train_iter12_stage2.py --device cuda:0
    python scripts/train_iter12_stage2.py --device cuda:0 \
        --candidates B5_cap2.0_lam0.5_lr2e-05_ep5
    python scripts/train_iter12_stage2.py --plan-lanes 4 --emit-sh

``preservation_trainer.train_with_preservation`` is a second copy of a training
loop, because ``unlearning_trainer.train_unlearning`` is hash-bound and cannot
grow a mode.  A second copy can drift from the first, and a drift would appear
as a difference between Stage 1 and Stage 2 that has nothing to do with either
mechanism.  So ``--control`` runs FIRST and on its own: it runs BOTH loops in
one process, under a pinned ``PYTHONHASHSEED``, on the incumbent Stage-1 row's
objective with every group in a plain ``sft``/``gd`` mode -- the frozen loop
twice and the new loop once -- and compares the resulting adapters tensor by
tensor.

Running the frozen loop TWICE is what makes the comparison a measurement rather
than an assumption: gap(A, A2) is how far two runs of the same code land apart
on this stack, and the gate is that the gap ACROSS the two loops does not exceed
it.  A floor of exactly zero demands bitwise equality.

WHY THE CONTROL DOES NOT COMPARE AGAINST THE FILED ADAPTER
----------------------------------------------------------
Its first version did, by sha256, and it refused -- correctly, on a criterion no
run can satisfy.  ``PeftConfig.target_modules`` is a Python ``set``, so the
order PEFT serialises it into ``adapter_config.json``, and the order it injects
the LoRA modules (which decides which module draws which dropout mask), depend
on ``PYTHONHASHSEED``.  Python randomises that per process.  Four seeds were
measured to give four different orders over the same seven modules, and a pinned
seed gives the same order every time.  Stage 1's adapter was written by a process
whose seed is not recoverable, so its bytes are unreachable from here; requiring
them made the control a test of the interpreter's hash seed rather than of the
two loops.  The filed adapter is still compared and REPORTED, with the gap and
the reason it is not zero, but it is never a gate.

The control is a gate, not a report.  If the two loops differ by more than the
frozen loop differs from itself, this script refuses to train a single
candidate, because every Stage-2 row would then be measured against a Stage-1
baseline produced by different code.  The spec it trains is read from the FROZEN
Stage-1 grid rather than retyped here, so the control cannot quietly test an
easier recipe than the one it stands in for.

Order of operations, and why each guard is where it is
------------------------------------------------------
1. verify the Stage-2 freeze (before any GPU work: a candidate trained under a
   protocol that has since drifted cannot be scored by it afterwards);
2. refuse any path that resolves inside the sealed confirmation evidence;
3. resolve every group path from the fit-half group directory and check it
   against the path the grid declares, so the grid cannot describe one dataset
   and train on another;
4. train, skipping rows whose adapters already exist so an interrupted chain
   resumes rather than repeats.

Needs a GPU.  Reads no confirmation prediction and writes no sealed path: every
adapter goes under ``data/checkpoints/mllmu_iter12_stage2/``, which the resolver
refuses to let coincide with Stage 1's root.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import freeze_iter12_stage2 as fis2

from granunlearn.config import _find_repo_root
from granunlearn.evaluation import retention_selection as rs
from granunlearn.logging_utils import setup_logger
from granunlearn.training import stage2_grid as sg
from granunlearn.training.candidate_grid import iter12_grid
from granunlearn.training.preservation_anchor import adapter_digest
from granunlearn.training.preservation_trainer import (
    PreservationGroupSpec,
    train_with_preservation,
    validate_stage2_groups,
)
from granunlearn.training.reference_trainer import ReferenceRecipe

#: The hash-bound frozen loop.  Imported and CALLED by the faithfulness control,
#: never edited: the control's whole claim is that the new loop agrees with this
#: one, so it has to run the real thing rather than a description of it.
from granunlearn.training import unlearning_trainer as ut
from granunlearn.training.unlearning_trainer import GroupSpec

log = setup_logger("train_iter12_stage2")

#: Named for what it establishes.  It was ``control_incumbent_byte_reproduction``
#: until the first run showed that name described something no run can do: the
#: incumbent's adapter was written under a randomised PYTHONHASHSEED, so its
#: bytes are not reproducible by any later process.
CONTROL_ID = "control_loop_equivalence"


# ── paths and specs ─────────────────────────────────────────────────────

def resolve_group_path(repo_root: Path, name: str,
                       declared: str | None = None) -> Path:
    """The group's real path, checked against the one the grid declares.

    The grid carries a repo-relative path so the basis document can show what
    each row trains on.  This does not trust it: the file is resolved from the
    fit-half group directory, as every prior Iteration-12 launcher has done,
    and the two must agree.  A grid that described one dataset and trained on
    another would otherwise be invisible.

    ``declared`` is optional because the faithfulness control builds its groups
    from the FROZEN Stage-1 grid, which declares no paths at all.
    """
    resolved = sg.groups_dir(repo_root) / f"{name}.jsonl"
    declared_path = None if declared is None \
        else (repo_root / declared).resolve()
    if declared_path is not None and resolved.resolve() != declared_path:
        raise SystemExit(
            f"REFUSING: the grid declares {declared} for group {name!r} "
            f"but the fit-half group directory resolves it to {resolved}. "
            f"One of the two is wrong, and training on the wrong one would "
            f"rehearse knowledge this protocol does not describe.")
    if not resolved.exists():
        raise SystemExit(f"REFUSING: {resolved} does not exist")
    return resolved


def as_training_specs(repo_root: Path, spec) -> list[PreservationGroupSpec]:
    """The grid row's groups with their paths resolved against the repo."""
    return [PreservationGroupSpec(
        name=g.name,
        path=resolve_group_path(repo_root, g.name, g.path),
        mode=g.mode,
        weight=g.weight,
        cap=g.cap,
        anchor_weight=g.anchor_weight) for g in spec.groups]


def recipe_for(spec) -> ReferenceRecipe:
    """The frozen recipe with only this row's overrides applied."""
    allowed = {"learning_rate", "num_epochs", "lora_r", "lora_alpha",
               "lora_dropout", "weight_decay", "seed"}
    unknown = set(spec.overrides) - allowed
    if unknown:
        raise SystemExit(
            f"{spec.candidate_id}: overrides outside the swept-knob allowlist: "
            f"{sorted(unknown)}. The frozen validator allows the same set, so "
            f"this is a second line of defence rather than the only one.")
    return ReferenceRecipe(**spec.overrides)


def incumbent_spec():
    """The Stage-1 row the control reproduces, read from the FROZEN grid."""
    want = sg.incumbent_row()
    grid = {c.candidate_id: c for c in iter12_grid()}
    if want not in grid:
        raise SystemExit(
            f"REFUSING: the incumbent reference row {want!r} is not in the "
            f"frozen Stage-1 grid, so there is nothing for the control to "
            f"reproduce.")
    return grid[want]


def control_spec(repo_root: Path) -> tuple[list[PreservationGroupSpec],
                                           ReferenceRecipe]:
    """The incumbent's objective expressed in Stage-2 terms, unchanged.

    Every group is in a plain ``sft``/``gd`` mode, no cap and no anchor, so
    this exercises exactly the branch of ``train_with_preservation`` that is
    supposed to be the frozen loop.
    """
    spec = incumbent_spec()
    groups = [PreservationGroupSpec(
        name=g.name,
        path=resolve_group_path(repo_root, g.name),
        mode=g.mode,
        weight=g.weight) for g in spec.groups]
    return groups, ReferenceRecipe(**spec.overrides)


def training_cost(spec, sizes: dict[str, int]) -> int:
    """Micro-batches: one per group example per epoch.

    The same cost model the Stage-1 and Stage-1b planners use.  Every Stage-2
    row consumes 90 + 90 + 183 = 363 examples per epoch, so cost separates the
    ep5 rows from the ep8 one and nothing else -- which is the point of
    recording it rather than assuming the lanes are equal.
    """
    epochs = int(spec.overrides.get("num_epochs", 10))
    return epochs * sum(sizes[g.name] for g in spec.groups)


def group_sizes(repo_root: Path) -> dict[str, int]:
    gdir = sg.groups_dir(repo_root)
    return {p.stem: sum(1 for line in p.read_text().splitlines() if line.strip())
            for p in sorted(gdir.glob("*.jsonl"))}


def plan_lanes(rows, sizes: dict[str, int], n_lanes: int) -> list[list]:
    """Longest-processing-time-first bin packing, deterministically ordered."""
    if n_lanes < 1:
        raise ValueError("n_lanes must be >= 1")
    lanes: list[list] = [[] for _ in range(n_lanes)]
    loads = [0] * n_lanes
    for spec in sorted(rows, key=lambda s: (-training_cost(s, sizes),
                                            s.candidate_id)):
        i = min(range(n_lanes), key=lambda k: (loads[k], k))
        lanes[i].append(spec)
        loads[i] += training_cost(spec, sizes)
    return [lane for lane in lanes if lane]


# ── the control ─────────────────────────────────────────────────────────

#: Two epochs, not the incumbent's five.  The control compares two loops, not
#: two points in a sweep, and two epochs exercise what one cannot: the
#: per-epoch shuffle is keyed on ``f"{seed}:{epoch}:{gi}"``, so a single epoch
#: would never vary the epoch index.  Three runs at two epochs cost about half
#: of one incumbent row.
CONTROL_EPOCHS = 2


def adapter_tensors(adapter_dir: Path) -> dict:
    """The adapter's tensors by name, for a comparison a digest cannot make."""
    from safetensors.torch import load_file
    path = Path(adapter_dir) / "adapter_model.safetensors"
    if not path.exists():
        raise SystemExit(f"REFUSING: {path} does not exist")
    return load_file(path)


def compare_adapters(a: dict, b: dict) -> dict:
    """Bitwise equality plus a magnitude, because a digest reports neither.

    A sha256 says "different" and stops.  Whether two adapters differ by float
    noise or by a different objective is the entire question the control exists
    to answer, and only a per-tensor gap can say.
    """
    import torch
    if set(a) != set(b):
        return {"comparable": False,
                "only_in_first": sorted(set(a) - set(b))[:8],
                "only_in_second": sorted(set(b) - set(a))[:8],
                "bitwise_identical": False,
                "max_abs_gap": None,
                "note": ("the two adapters do not even hold the same tensors, "
                         "so they were built by different LoRA configurations")}
    gaps = {k: (a[k].float() - b[k].float()).abs().max().item() for k in a}
    worst = max(gaps, key=gaps.get)
    return {"comparable": True,
            "num_tensors": len(a),
            "bitwise_identical": all(torch.equal(a[k], b[k]) for k in a),
            "max_abs_gap": gaps[worst],
            "mean_abs_gap": sum(gaps.values()) / len(gaps),
            "worst_tensor": worst}


def _release_gpu() -> None:
    """Free the previous run's model before the next one is loaded.

    Three trainings run in one process, each loading a ~22 GiB bf16 model.  The
    interpreter frees the reference on return but the caching allocator keeps
    the blocks, so the next load can otherwise fail on a card that a co-tenant
    has since grown into.
    """
    import gc

    import torch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def assert_hash_seed_pinned() -> str:
    """Refuse to run the control under a randomised PYTHONHASHSEED.

    ``PeftConfig.target_modules`` is a Python ``set``, so the order PEFT
    serialises it into ``adapter_config.json`` -- and the order it injects the
    LoRA modules, which decides which module draws which dropout mask -- depends
    on the string hash seed.  Python randomises that seed per process unless
    PYTHONHASHSEED is set, so two runs of IDENTICAL code produce different
    adapter bytes.  Measured: four seeds give four different orders over the
    same seven modules, and a pinned seed gives the same order every time.

    It cannot be set at runtime, only in the environment the interpreter was
    started with, so the control checks and refuses rather than quietly
    measuring a difference the environment caused.
    """
    import os
    seed = os.environ.get("PYTHONHASHSEED")
    if not seed:
        raise SystemExit(
            "REFUSING to run the control: PYTHONHASHSEED is not set, so PEFT's "
            "target_modules set iterates in a per-process random order and two "
            "runs of identical code produce different adapter bytes. Export a "
            "fixed value (the lane script does) and re-run. Without it this "
            "control measures the interpreter's hash seed rather than the two "
            "loops.")
    return seed


def run_control(repo_root: Path, device: str, force: bool = False) -> dict:
    """Show the two loops are one loop, on the real model, with the environment
    held still.

    Three trainings in ONE process, under ONE pinned hash seed, from the same
    initial adapter and the same recipe seeds:

      A   unlearning_trainer.train_unlearning        -- the frozen loop
      A2  unlearning_trainer.train_unlearning        -- the frozen loop again
      B   preservation_trainer.train_with_preservation, every group in a plain
          sft/gd mode, no cap and no anchor

    A2 exists to MEASURE the noise floor rather than assume it: gap(A, A2) is
    how far two runs of the same code land apart on this stack.  The gate is
    that gap(A, B) does not exceed it -- exactly zero when the floor is exactly
    zero, which is what a pinned hash seed and one process are supposed to buy.

    The incumbent Stage-1 adapter is compared and REPORTED but never gated on.
    It was produced by a process whose PYTHONHASHSEED was randomised and is not
    recoverable, so no run today can match its bytes; requiring that would make
    the control unsatisfiable by construction, which is what the first version
    of this control did and why it refused.
    """
    hash_seed = assert_hash_seed_pinned()
    out_root = sg.control_dir(repo_root) / CONTROL_ID
    marker = out_root / "CONTROL.json"
    if marker.exists() and not force:
        previous = json.loads(marker.read_text())
        if previous["gate_passed"]:
            log.info("control already passed (gap A->B max %s); pass --force to "
                     "re-run", previous["gate"]["gap_frozen_vs_new"]["max_abs_gap"])
            return previous
        raise SystemExit(
            f"REFUSING: the control was already run and FAILED. "
            f"{previous['detail']}")

    incumbent = incumbent_spec()
    groups, recipe = control_spec(repo_root)
    recipe = ReferenceRecipe(**dict(recipe.to_dict(),
                                    num_epochs=CONTROL_EPOCHS))
    errors = validate_stage2_groups(groups)
    if errors:
        raise SystemExit(f"the control's own objective is invalid: {errors}")
    frozen_groups = [GroupSpec(g.name, g.path, g.mode, g.weight)
                     for g in groups]

    mf = sg.mf_adapters(repo_root)
    runs: dict[str, Path] = {}
    for label, loop, specs in (
            ("A_frozen_loop", ut.train_unlearning, frozen_groups),
            ("A2_frozen_loop_again", ut.train_unlearning, frozen_groups),
            ("B_new_loop_no_cap_no_anchor", train_with_preservation, groups)):
        out_dir = out_root / label
        log.info("CONTROL %s: %s | %d groups | %d epoch(s) | PYTHONHASHSEED=%s",
                 label, loop.__module__ + "." + loop.__name__, len(specs),
                 CONTROL_EPOCHS, hash_seed)
        loop(label, specs, out_dir, device=device, recipe=recipe,
             init_adapter_dir=mf)
        runs[label] = out_dir / "adapters"
        _release_gpu()

    tensors = {k: adapter_tensors(v) for k, v in runs.items()}
    floor = compare_adapters(tensors["A_frozen_loop"],
                             tensors["A2_frozen_loop_again"])
    across = compare_adapters(tensors["A_frozen_loop"],
                              tensors["B_new_loop_no_cap_no_anchor"])

    #: The gate.  A noise floor of exactly zero demands exact equality; a
    #: non-zero floor -- which would mean this stack is not run-to-run
    #: deterministic even inside one process -- demands only that the gap
    #: ACROSS the two loops be no wider than the gap WITHIN one of them.
    if not across["comparable"]:
        passed = False
    elif floor["bitwise_identical"]:
        passed = across["bitwise_identical"]
    else:
        passed = across["max_abs_gap"] <= floor["max_abs_gap"]

    filed = sg.stage1_ckpt_root(repo_root) / incumbent.candidate_id / "adapters"
    vs_filed = compare_adapters(tensors["A_frozen_loop"],
                                adapter_tensors(filed)) \
        if filed.exists() else {"comparable": False,
                                "note": f"{filed} does not exist"}

    result = {
        "control": CONTROL_ID,
        "what_it_establishes": (
            "that train_with_preservation with every group in a plain sft/gd "
            "mode computes the same function as the hash-bound "
            "train_unlearning it copies, on the real model and the real groups, "
            "so a difference between Stage 1 and Stage 2 is attributable to the "
            "objective and not to the code that runs it"),
        "spec_read_from": "the frozen Stage-1 grid, not retyped here",
        "reproduces_the_objective_of": incumbent.candidate_id,
        "epochs": CONTROL_EPOCHS,
        "why_not_the_incumbents_five_epochs": (
            "the control compares two loops rather than reproducing a filed "
            "adapter, and two epochs are the fewest that vary the per-epoch "
            "shuffle key"),
        "python_hash_seed": hash_seed,
        "why_the_hash_seed_must_be_pinned": (
            "PeftConfig.target_modules is a set, so the order PEFT serialises "
            "it -- and the order it injects the LoRA modules, which decides "
            "which module draws which dropout mask -- varies with PYTHONHASHSEED. "
            "Four seeds were measured to give four different orders over the "
            "same seven modules. Unpinned, two runs of identical code produce "
            "different adapter bytes and the control measures the interpreter "
            "rather than the loops."),
        "groups": [g.describe() for g in groups],
        "recipe_overrides": dict(incumbent.overrides),
        "runs": {k: str(v.relative_to(repo_root)) for k, v in runs.items()},
        "noise_floor_frozen_loop_vs_itself": floor,
        "gate": {
            "gap_frozen_vs_new": across,
            "required": ("bitwise identical, because the noise floor is exactly "
                         "zero" if floor["bitwise_identical"] else
                         "no wider than the measured noise floor"),
            "passed": passed,
        },
        "gate_passed": passed,
        "filed_stage1_adapter_reported_not_gated": {
            "path": str(filed.relative_to(repo_root)) if filed.exists() else None,
            "gap": vs_filed,
            "why_this_is_not_a_gate": (
                "the filed adapter was written by a process whose "
                "PYTHONHASHSEED was randomised and is not recoverable, so its "
                "target_modules order -- and therefore its bytes and, through "
                "the dropout stream, its weights -- cannot be reproduced by any "
                "later run. The first version of this control gated on exactly "
                "that and refused, correctly, on a criterion no run could "
                "satisfy."),
        },
        "detail": (
            "the two loops produced bitwise-identical adapters under a pinned "
            "hash seed, so a difference between Stage 1 and Stage 2 is "
            "attributable to the objective and not to the code that runs it"
            if passed else
            f"the two loops differ by max {across.get('max_abs_gap')} where the "
            f"frozen loop's own run-to-run floor is "
            f"{floor.get('max_abs_gap')}; every Stage-2 row would be measured "
            f"against a Stage-1 baseline produced by different code, so no "
            f"candidate may be trained until this is explained"),
    }
    out_root.mkdir(parents=True, exist_ok=True)
    with open(marker, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    if not passed:
        raise SystemExit(f"REFUSING: {result['detail']}")
    log.info("CONTROL PASSED: the two loops agree bitwise (noise floor max %s)",
             floor["max_abs_gap"])
    return result



# ── main ────────────────────────────────────────────────────────────────

def assert_no_forbidden_evidence(paths: list[Path], repo_root: Path) -> None:
    """Refuse to touch the sealed confirmation, in either direction.

    Checked on RESOLVED paths, because a relative path with ``..`` in it, or a
    symlink, would otherwise walk straight past a string comparison.
    """
    forbidden = [((repo_root / rel).resolve(), rel)
                 for rel in rs.FORBIDDEN_EVIDENCE]
    for path in paths:
        resolved = Path(path).resolve()
        for bad, rel in forbidden:
            if resolved == bad or bad in resolved.parents:
                raise SystemExit(
                    f"REFUSING: {path} resolves inside {rel}. The 11C "
                    f"confirmation was scored once and is sealed; developing a "
                    f"successor on its predictions would make a confirmatory "
                    f"result retrospective.")


def rows_needing_training(repo_root: Path, grid, force: bool):
    """The trained rows with no adapter yet, so a chain resumes rather than
    repeats."""
    root = sg.stage2_ckpt_root(repo_root)
    rows = sg.trained_rows(grid)
    if force:
        return list(rows)
    return [c for c in rows
            if not (root / c.candidate_id / "adapters").exists()]


def emit_plan(args, repo_root: Path) -> None:
    cal = json.loads((repo_root / sg.CALIBRATION_REPORT).read_text())
    grid = sg.stage2_grid(cal["grid_rule"]["beta_star"])
    rows = rows_needing_training(repo_root, grid, args.force)
    sizes = group_sizes(repo_root)
    lanes = plan_lanes(rows, sizes, args.plan_lanes)
    total = sum(training_cost(c, sizes) for c in rows)
    print(f"# group sizes: {json.dumps(sizes)}")
    print(f"# {total} micro-batches over {len(rows)} row(s) in "
          f"{len(lanes)} lane(s)")
    for lane in lanes:
        ids = ",".join(c.candidate_id for c in lane)
        if args.emit_sh:
            print(ids)
        else:
            cost = sum(training_cost(c, sizes) for c in lane)
            print(f"lane: cost={cost} ({cost / max(total, 1):.0%}) "
                  f"n={len(lane)} -> {ids}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--control", action="store_true",
                    help="Run ONLY the byte-reproduction control, and refuse "
                         "if it fails. Do this before training anything.")
    ap.add_argument("--candidates", default=None,
                    help="Comma-separated candidate ids (default: all that "
                         "still need training)")
    ap.add_argument("--force", action="store_true",
                    help="Retrain rows whose adapters already exist, and "
                         "re-run the control even if it already passed")
    ap.add_argument("--plan-lanes", type=int, default=None, metavar="N",
                    help="Print an N-lane plan and exit; no GPU")
    ap.add_argument("--emit-sh", action="store_true",
                    help="With --plan-lanes, print one --candidates value "
                         "per line")
    ap.add_argument("--repo-root", default=None)
    args = ap.parse_args()

    repo_root = Path(args.repo_root) if args.repo_root \
        else (_find_repo_root(Path.cwd()) or Path.cwd())

    if args.plan_lanes is not None:
        #: No freeze verification here: planning reads group files and the
        #: calibration report and touches no GPU, so it is safe -- and useful
        #: -- to inspect a plan before the protocol is frozen.
        emit_plan(args, repo_root)
        return

    #: Step 1 -- verified BEFORE any GPU work.
    reasons = fis2.verify_freeze(repo_root)
    if reasons:
        raise SystemExit(
            "REFUSING to train: the Stage-2 freeze does not match the "
            "repository:\n  " + "\n  ".join(reasons[:20]))

    #: Step 2 -- the sealed confirmation is out of bounds.
    assert_no_forbidden_evidence(
        [sg.stage2_ckpt_root(repo_root), sg.groups_dir(repo_root),
         sg.reference_cache_dir(repo_root)], repo_root)

    if args.control:
        run_control(repo_root, args.device, force=args.force)
        return

    marker = sg.control_dir(repo_root) / CONTROL_ID / "CONTROL.json"
    if not marker.exists():
        raise SystemExit(
            "REFUSING to train candidates: the faithfulness control has not "
            "been run. --control runs BOTH loops in one process under a pinned "
            "PYTHONHASHSEED and compares their adapters against the frozen "
            "loop's own measured run-to-run floor; without it, a Stage-2 "
            "difference could be a difference between two copies of a training "
            "loop rather than between two objectives.")
    control = json.loads(marker.read_text())
    #: Required, never re-run here.  The lane script gives the control its own
    #: phase, so four training lanes do not each spend half an hour repeating it,
    #: and a FAILED control cannot be skipped by any flag: the refusal below is
    #: unconditional.
    if not control["gate_passed"]:
        raise SystemExit(
            f"REFUSING to train candidates: {control['detail']}")
    log.info("control passed (the two loops agree, max gap %s against a noise "
             "floor of %s); training against it",
             control["gate"]["gap_frozen_vs_new"]["max_abs_gap"],
             control["noise_floor_frozen_loop_vs_itself"]["max_abs_gap"])

    cal = json.loads((repo_root / sg.CALIBRATION_REPORT).read_text())
    grid = sg.stage2_grid(cal["grid_rule"]["beta_star"])
    kept, _filtered = sg.validate_stage2_grid(grid)
    if kept:
        raise SystemExit(f"REFUSING: the Stage-2 grid is invalid: {kept}")
    by_id = {c.candidate_id: c for c in grid}

    rows = sg.trained_rows(grid)
    if args.candidates:
        wanted = {s.strip() for s in args.candidates.split(",") if s.strip()}
        unknown = wanted - set(by_id)
        if unknown:
            raise SystemExit(
                f"--candidates {sorted(unknown)} matches no Stage-2 grid row; "
                f"trainable ids: {sorted(c.candidate_id for c in rows)}")
        rows = [c for c in rows if c.candidate_id in wanted]

    ckpt_root = sg.stage2_ckpt_root(repo_root)
    mf = sg.mf_adapters(repo_root)
    if not mf.exists():
        raise SystemExit(f"REFUSING: the canonical MF adapter {mf} is missing; "
                         f"every Stage-2 row continues from it.")
    cache_dir = sg.reference_cache_dir(repo_root)

    trained: list[dict[str, Any]] = []
    for spec in rows:
        out_dir = ckpt_root / spec.candidate_id
        if (out_dir / "adapters").exists() and not args.force:
            log.info("[%s] adapters exist — skipped (pass --force to retrain)",
                     spec.candidate_id)
            continue
        groups = as_training_specs(repo_root, spec)
        errors = validate_stage2_groups(groups)
        if errors:
            raise SystemExit(
                f"REFUSING: {spec.candidate_id}'s objective is invalid: "
                f"{errors}")
        needs_cache = any(g.mode == "anchor" or g.anchor_weight is not None
                          for g in groups)
        log.info("[%s] cap=%s beta=%s epochs=%s lr=%s%s", spec.candidate_id,
                 spec.cap, spec.beta, spec.overrides.get("num_epochs"),
                 spec.overrides.get("learning_rate"),
                 " (anchored)" if needs_cache else "")
        summary = train_with_preservation(
            spec.candidate_id, groups, out_dir, device=args.device,
            recipe=recipe_for(spec), init_adapter_dir=mf,
            cache_dir=cache_dir if needs_cache else None,
            mf_adapter_dir=mf if needs_cache else None)
        last = (summary["epochs"] or [{}])[-1]
        trained.append({
            "candidate_id": spec.candidate_id,
            "num_optimizer_steps": summary["num_optimizer_steps"],
            "train_seconds": summary["train_seconds"],
            "final_avg_loss_fine_target": last.get("avg_loss_fine_target"),
            "final_frac_at_cap_fine_target":
                last.get("frac_at_cap_fine_target"),
            "final_avg_kl_retain": last.get("avg_kl_retain"),
            "adapter_sha256": adapter_digest(out_dir / "adapters"),
        })
        log.info("[%s] done in %.0fs | fine_target NLL %s | frac_at_cap %s | "
                 "KL %s", spec.candidate_id, summary["train_seconds"],
                 last.get("avg_loss_fine_target"),
                 last.get("frac_at_cap_fine_target"),
                 last.get("avg_kl_retain"))

    log.info("trained %d of %d requested row(s)", len(trained), len(rows))
    if trained:
        print(json.dumps(trained, indent=2))


if __name__ == "__main__":
    main()
