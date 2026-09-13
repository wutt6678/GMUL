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
mechanism.  So ``--control`` runs FIRST and on its own: it trains the incumbent
Stage-1 row ``B4_w1.0_lam0.5_lr2e-05_ep5`` through the new loop with every
group in a plain ``sft``/``gd`` mode, and compares the resulting adapter to the
one Stage 1 filed, by sha256 over the whole adapter directory.

The control is a gate, not a report.  If the digests differ, this script
refuses to train a single candidate, because every Stage-2 row would then be
measured against a Stage-1 baseline produced by different code.  The spec it
trains is read from the FROZEN Stage-1 grid rather than retyped here, so the
control cannot quietly test an easier recipe than the one it stands in for.

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

log = setup_logger("train_iter12_stage2")

CONTROL_ID = "control_incumbent_byte_reproduction"


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

def run_control(repo_root: Path, device: str, force: bool = False) -> dict:
    """Reproduce the incumbent Stage-1 adapter through the new loop."""
    out_dir = sg.control_dir(repo_root) / CONTROL_ID
    incumbent = incumbent_spec()
    filed = sg.stage1_ckpt_root(repo_root) / incumbent.candidate_id / "adapters"
    if not filed.exists():
        raise SystemExit(
            f"REFUSING: the incumbent adapter {filed} does not exist, so there "
            f"is nothing for the control to reproduce. Train Stage 1 first, or "
            f"do not claim the new loop is faithful to it.")
    marker = out_dir / "CONTROL.json"
    if marker.exists() and not force:
        previous = json.loads(marker.read_text())
        if previous["reproduced_byte_for_byte"]:
            log.info("control already passed (%s); pass --force to re-run",
                     previous["digest"][:16])
            return previous
        raise SystemExit(
            f"REFUSING: the control was already run and FAILED. "
            f"{previous['detail']}")

    groups, recipe = control_spec(repo_root)
    errors = validate_stage2_groups(groups)
    if errors:
        raise SystemExit(f"the control's own objective is invalid: {errors}")
    log.info("CONTROL: training %s through train_with_preservation with no cap "
             "and no anchor | %d groups | %d epochs",
             incumbent.candidate_id, len(groups), recipe.num_epochs)
    summary = train_with_preservation(
        CONTROL_ID, groups, out_dir, device=device, recipe=recipe,
        init_adapter_dir=sg.mf_adapters(repo_root))

    got = adapter_digest(out_dir / "adapters")
    want = adapter_digest(filed)
    reproduced = got == want
    result = {
        "control": CONTROL_ID,
        "reproduces": incumbent.candidate_id,
        "spec_read_from": "the frozen Stage-1 grid, not retyped here",
        "groups": [g.describe() for g in groups],
        "recipe_overrides": dict(incumbent.overrides),
        "num_optimizer_steps": summary["num_optimizer_steps"],
        "train_seconds": summary["train_seconds"],
        "produced_adapter_dir": str((out_dir / "adapters").relative_to(
            repo_root)),
        "filed_adapter_dir": str(filed.relative_to(repo_root)),
        "digest": got,
        "filed_digest": want,
        "reproduced_byte_for_byte": reproduced,
        "epochs_match": summary["epochs"] == json.loads(
            (filed.parent / "training_summary.json").read_text())["epochs"],
        "detail": (
            "the new loop reproduced the incumbent adapter exactly, so a "
            "difference between Stage 1 and Stage 2 is attributable to the "
            "objective and not to the code that runs it" if reproduced else
            f"the new loop produced {got[:16]} where Stage 1 filed "
            f"{want[:16]}; every Stage-2 row would be measured against a "
            f"baseline produced by different code, so no candidate may be "
            f"trained until this is explained"),
    }
    with open(marker, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    if not reproduced:
        raise SystemExit(f"REFUSING: {result['detail']}")
    log.info("CONTROL PASSED: %s reproduced byte for byte (%s)",
             incumbent.candidate_id, got[:16])
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
            "been run. --control trains the incumbent Stage-1 row through the "
            "new loop and compares adapters by sha256; without it, a Stage-2 "
            "difference could be a difference between two copies of a training "
            "loop rather than between two objectives.")
    control = json.loads(marker.read_text())
    #: Required, never re-run here.  The lane script gives the control its own
    #: phase, so four training lanes do not each spend 28 minutes repeating it,
    #: and a FAILED control cannot be skipped by any flag: the refusal below is
    #: unconditional.
    if not control["reproduced_byte_for_byte"]:
        raise SystemExit(
            f"REFUSING to train candidates: {control['detail']}")
    log.info("control passed (%s reproduces %s); training against it",
             control["digest"][:16], control["reproduces"])

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
