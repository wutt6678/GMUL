"""Evaluate reference states and apply the MF != MG != MN gate.

    python scripts/evaluate_reference_states.py --states BASE,MF,MG,MN \
        --device cuda:0

Generates greedy completions for every committed smoke query (all
families/splits — paraphrase split is evaluation-only), scores them
against the canonical hierarchy, writes predictions + metrics, and
applies the Iteration 7 hard gate.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from granunlearn.config import _find_repo_root
from granunlearn.evaluation.reference_eval import run_reference_evaluation
from granunlearn.logging_utils import setup_logger

log = setup_logger("evaluate_reference_states")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate reference states + separation gate")
    parser.add_argument("--states", default="BASE,MF,MG,MN")
    parser.add_argument("--smoke-dir", default="data/mllmu_hier_smoke")
    parser.add_argument("--checkpoints-dir",
                        default="data/checkpoints/mllmu_smoke")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--rescore", action="store_true",
                        help="Re-score persisted prediction parquets "
                             "instead of running the model (CPU-only)")
    args = parser.parse_args()

    repo_root = _find_repo_root(Path.cwd()) or Path.cwd()
    smoke_dir = Path(args.smoke_dir)
    ckpt_dir = Path(args.checkpoints_dir)
    smoke_dir = smoke_dir if smoke_dir.is_absolute() else repo_root / smoke_dir
    ckpt_dir = ckpt_dir if ckpt_dir.is_absolute() else repo_root / ckpt_dir

    report = run_reference_evaluation(
        smoke_dir=smoke_dir,
        checkpoints_dir=ckpt_dir,
        report_path=repo_root / "data" / "reports" /
        "mllmu_smoke_reference_eval.json",
        device=args.device,
        states=[s.strip().upper() for s in args.states.split(",")],
        predictions_dir=smoke_dir / "predictions",
        batch_size=args.batch_size,
        rescore=args.rescore,
    )
    gate = report["separation_gate"]
    log.info("GATE %s | reasons: %s",
             "PASSED" if gate["passed"] else "FAILED", gate["reasons"])


if __name__ == "__main__":
    main()
