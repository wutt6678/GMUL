"""Plan balanced GPU lanes for a candidate grid.

    python scripts/plan_candidate_lanes.py --tag pilot100 --lanes 3
    python scripts/plan_candidate_lanes.py --tag pilot100 --lanes 3 --emit-sh

Cost model: the trainer runs one micro-batch per knowledge-group example
per epoch, so

    cost(candidate) = num_epochs * sum(len(group) for group in groups)

is proportional to its GPU time (every candidate uses the same recipe,
the same base model and the same multimodal formatting).  Lanes are
packed longest-processing-time-first, which keeps the makespan close to
max(total/lanes, largest_single_candidate) — important because B3
candidates carry the 387-example retain group while B1/B2 carry only the
90-example target groups.

B0 (no-op) costs nothing and is pinned to lane 0 so it always exists
before any lane tries to score it.  Candidates whose adapters are already
trained are skipped unless --include-trained is given.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from granunlearn.config import _find_repo_root
from granunlearn.training.candidate_grid import grid_for_tag


def group_sizes(dataset_dir: Path) -> dict[str, int]:
    sizes: dict[str, int] = {}
    for p in sorted((dataset_dir / "unlearning").glob("*.jsonl")):
        sizes[p.stem] = sum(1 for line in p.read_text().splitlines()
                            if line.strip())
    return sizes


def candidate_cost(spec, sizes: dict[str, int],
                   default_epochs: int) -> int:
    if spec.noop:
        return 0
    epochs = int(spec.overrides.get("num_epochs", default_epochs))
    per_epoch = sum(sizes.get(g.name, 0) for g in spec.groups)
    missing = [g.name for g in spec.groups if g.name not in sizes]
    if missing:
        raise SystemExit(
            f"{spec.candidate_id}: unknown knowledge groups {missing} — "
            f"run scripts/build_unlearning_groups.py first")
    return epochs * per_epoch


def plan_lanes(grid, sizes: dict[str, int], n_lanes: int,
               default_epochs: int) -> list[list]:
    """LPT bin packing; returns one candidate list per lane."""
    if n_lanes < 1:
        raise ValueError("n_lanes must be >= 1")
    lanes: list[list] = [[] for _ in range(n_lanes)]
    loads = [0] * n_lanes
    noop = [c for c in grid if c.noop]
    rest = [c for c in grid if not c.noop]
    if noop:
        lanes[0].extend(noop)  # free, and must exist before scoring
    for spec in sorted(rest, key=lambda c: -candidate_cost(
            c, sizes, default_epochs)):
        i = min(range(n_lanes), key=lambda k: (loads[k], k))
        lanes[i].append(spec)
        loads[i] += candidate_cost(spec, sizes, default_epochs)
    return lanes


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tag", default="pilot100",
                    choices=("smoke", "pilot100"))
    ap.add_argument("--lanes", type=int, default=3)
    ap.add_argument("--default-epochs", type=int, default=10)
    ap.add_argument("--include-trained", action="store_true",
                    help="Also plan candidates whose adapters exist")
    ap.add_argument("--emit-sh", action="store_true",
                    help="Print one ready-to-run --candidates value per line")
    args = ap.parse_args()

    repo_root = _find_repo_root(Path.cwd()) or Path.cwd()
    dataset_dir = repo_root / "data" / f"mllmu_hier_{args.tag}"
    ckpt_root = repo_root / "data" / "checkpoints" / \
        f"mllmu_{args.tag}_unlearn"
    sizes = group_sizes(dataset_dir)
    grid = grid_for_tag(args.tag)
    if not args.include_trained:
        todo = [c for c in grid
                if c.noop or not (ckpt_root / c.candidate_id / "adapters"
                                  / "adapter_model.safetensors").exists()]
        skipped = len(grid) - len(todo)
        if skipped:
            print(f"# {skipped} candidate(s) already trained — excluded")
        grid = todo
    if not grid:
        print("# nothing left to train")
        return

    lanes = plan_lanes(grid, sizes, args.lanes, args.default_epochs)
    total = sum(candidate_cost(c, sizes, args.default_epochs) for c in grid)
    print(f"# group sizes: {json.dumps(sizes)}")
    print(f"# total cost {total} micro-batches over {len(grid)} candidates")
    for i, lane in enumerate(lanes):
        if not lane:
            continue
        cost = sum(candidate_cost(c, sizes, args.default_epochs)
                   for c in lane)
        ids = ",".join(c.candidate_id for c in lane)
        if args.emit_sh:
            print(ids)
        else:
            print(f"lane {i}: cost={cost} ({cost / max(total, 1):.0%}) "
                  f"n={len(lane)} -> {ids}")


if __name__ == "__main__":
    main()
