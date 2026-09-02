"""Evaluate reference states and apply the MF != MG != MN gate.

    python scripts/evaluate_reference_states.py --tag pilot100 \
        --states BASE,MF,MG,MN --device cuda:0

Generates greedy completions for every committed query of the tagged
dataset (all families/splits — the paraphrase split is evaluation-only),
scores them against the canonical hierarchy, writes predictions +
metrics, and applies the hard gate on the POOLED set and on the TEST
paraphrase split separately.
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
    parser.add_argument("--tag", default="smoke",
                        choices=("smoke", "pilot100"))
    parser.add_argument("--states", default="BASE,MF,MG,MN")
    parser.add_argument("--smoke-dir", default=None,
                        help="Override the dataset directory "
                             "(default: data/mllmu_hier_<tag>)")
    parser.add_argument("--checkpoints-dir", default=None,
                        help="Override the checkpoint directory "
                             "(default: data/checkpoints/mllmu_<tag>)")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--image-batch-size", type=int, default=1,
                        help="Batch size for image probes (pilot100 uses "
                             "8; 1 reproduces the Iteration 7/9 path)")
    parser.add_argument("--rescore", action="store_true",
                        help="Re-score persisted prediction parquets "
                             "instead of running the model (CPU-only)")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Reuse persisted predictions for states "
                             "that already have them")
    args = parser.parse_args()

    repo_root = _find_repo_root(Path.cwd()) or Path.cwd()
    smoke_dir = Path(args.smoke_dir or f"data/mllmu_hier_{args.tag}")
    ckpt_dir = Path(args.checkpoints_dir or
                    f"data/checkpoints/mllmu_{args.tag}")
    smoke_dir = smoke_dir if smoke_dir.is_absolute() else repo_root / smoke_dir
    ckpt_dir = ckpt_dir if ckpt_dir.is_absolute() else repo_root / ckpt_dir
    experiment_id = f"mllmu_{args.tag}_iter11" if args.tag == "pilot100" \
        else "mllmu_smoke_iter7"

    report = run_reference_evaluation(
        smoke_dir=smoke_dir,
        checkpoints_dir=ckpt_dir,
        report_path=repo_root / "data" / "reports" /
        f"mllmu_{args.tag}_reference_eval.json",
        device=args.device,
        states=[s.strip().upper() for s in args.states.split(",")],
        predictions_dir=smoke_dir / "predictions",
        batch_size=args.batch_size,
        image_batch_size=args.image_batch_size,
        rescore=args.rescore,
        failure_export_dir=smoke_dir / "failure_exports",
        skip_existing=args.skip_existing,
        experiment_id=experiment_id,
    )
    gate = report["separation_gate"]
    log.info("GATE %s | reasons: %s",
             "PASSED" if gate["passed"] else "FAILED", gate["reasons"])
    # Frozen Iteration 8 headline metrics, TEST split primary
    for state, hm in report["hierarchy_metrics"]["test"].items():
        log.info("[%s/TEST] FILR=%s TGA=%s | under=%s over=%s wrong=%s "
                 "refusal=%s halluc=%s | ancestor_post=%s",
                 state, hm["filr"], hm["tga"],
                 hm["failure_rates"]["under_forgetting"],
                 hm["failure_rates"]["over_forgetting"],
                 hm["failure_rates"]["wrong_branch"],
                 hm["failure_rates"]["refusal"],
                 hm["failure_rates"]["hallucination"],
                 hm["ancestor_retention"]["post_unlearning_accuracy"])


if __name__ == "__main__":
    main()
