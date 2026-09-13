"""Score the Iteration 12 Stage-1b seed replicates against the frozen floor.

    python scripts/analyze_iter12_seed_replication.py --control-only
    python scripts/analyze_iter12_seed_replication.py --generate-only \
        --candidates B4_w4.0_lam0.5_lr2e-05_ep5__s43
    python scripts/analyze_iter12_seed_replication.py

Three phases, in this order, because each one can invalidate the next.

CONTROL.  Regenerate B0 -- whose adapter is a byte copy of MF and which takes
zero training steps -- into this study's own predictions directory, under the
Stage-1 generation contract, and compare it row by row with Stage 1's B0
parquet.  Two existing B0 parquets already show that generation is NOT
bit-stable when ``image_batch_size`` changes (15 correctness flips in 4,518
rows, all of them image-route).  What that comparison cannot tell us is whether
generation is deterministic under a FIXED contract, which is the assumption
every replicate comparison rests on.  If the four floor numbers move, this stops
here: a point floor over a non-reproducible anchor is not interpretable and no
amount of seed replication would make it so.

GENERATE.  One replicate per lane.  The seed-42 replicate is READ from Stage 1's
predictions rather than regenerated, because its adapter is the same bytes
Stage 1 scored and regenerating it would replace filed evidence with a second
draw.

SCORE.  Per parent, per frozen number: mean, min, max, sd and the count of
replicates individually at or above the anchor.  The PRIMARY RULE is the mean,
scored by the frozen ``floor_check`` against the Stage-1 B0 anchor with the
Stage-1 epsilon, so the arithmetic is identical to the verdict being revisited.
The range classification is reported alongside and is NOT decision-bearing:
with k=4 an interval would import a distributional assumption the design does
not need, while "did every independent seed fall below the anchor" is the
question actually being asked.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import freeze_iter12_seed_replication as fisr

#: The sealed pilot-100 selector supplies generation, exactly as the Stage-1
#: selector imports it: batch layout, sidecar contract and coverage validation
#: have to be IDENTICAL to the ones that produced the anchor, and a
#: reimplementation would drift from them silently.
from select_unlearning_checkpoints import _generate_state

from granunlearn.config import _find_repo_root
from granunlearn.evaluation import retention_selection as rs
from granunlearn.evaluation.reference_eval import (
    DEFAULT_MAX_IMAGE_PIXELS,
    DEFAULT_MAX_LENGTH,
    load_associations_parquet,
    load_predictions_parquet,
    load_queries_parquet,
)
from granunlearn.logging_utils import setup_logger
from granunlearn.training.candidate_grid import dataset_dir_for_tag
from granunlearn.training.seed_replication import (
    FLOOR_NUMBERS,
    OUT_REPORT,
    SEED_CKPT_ROOT,
    SEED_PREDICTIONS_SUBDIR,
    STAGE1_CKPT_ROOT,
    STAGE1_FREEZE_REPORT,
    STAGE1_PREDICTIONS_SUBDIR,
    STAGE1_SELECTION_REPORT,
    aggregate,
    mean_candidate,
    range_classification,
    replicates,
)

log = setup_logger("iter12_seed_replication")

EXPERIMENT_ID = "mllmu_iter12_seeds"
SPLITS = ("train", "val")

#: Keys that must MATCH Stage 1, because they are inside the measurement.  Batch
#: layout is inside D_G, and the image batch is what the measured instability
#: was sensitive to.
MEASUREMENT_CONTRACT_KEYS = ("batch_size", "image_batch_size",
                             "max_new_tokens", "max_length",
                             "max_image_pixels", "do_sample", "model_id")
#: Keys that must DIFFER, because they identify the study rather than the
#: measurement.  Sharing Stage 1's directory would put a second
#: ``predictions_tv_B0.parquet`` under a different ``experiment_id``, and the
#: sidecar would refuse reuse and REGENERATE over filed Stage-1 evidence.
STUDY_IDENTITY_KEYS = ("predictions_dir", "experiment_id")

#: Fields compared by the determinism control, in the order that matters: the
#: scored outcomes first, because they are what the floor reads.
CONTROL_FIELDS = ("is_correct_branch", "parsed_answer",
                  "matched_canonical_id", "predicted_level",
                  "is_finer_than_target", "is_coarser_than_target",
                  "raw_output")


def assert_no_forbidden_evidence(paths, repo_root: Path) -> None:
    """The sealed confirmation split stays out of bounds, as in Stage 1."""
    for forbidden in rs.FORBIDDEN_EVIDENCE:
        bad = (repo_root / forbidden).resolve()
        for p in paths:
            resolved = Path(p).resolve()
            if bad == resolved or bad in resolved.parents:
                raise SystemExit(
                    f"REFUSING: {resolved} is inside {bad}. This study reads "
                    f"exploratory train+val evidence only; the confirmation "
                    f"predictions are one-shot sealed evidence and tuning "
                    f"against them would spend them.")


def prediction_filename(state_id: str) -> str:
    return f"predictions_{''.join(s[0] for s in SPLITS)}_{state_id}.parquet"


def enforce_contract(args, repo_root: Path, predictions_dir: Path) -> dict:
    """Match the measurement, differ on the identity, and say both out loud."""
    generation_config = {
        "batch_size": args.batch_size,
        "image_batch_size": args.image_batch_size,
        "max_new_tokens": args.max_new_tokens,
        "do_sample": False,
        "max_image_pixels": DEFAULT_MAX_IMAGE_PIXELS,
        "max_length": DEFAULT_MAX_LENGTH,
    }
    stage1 = json.loads(
        (repo_root / STAGE1_FREEZE_REPORT).read_text())["generation_contract"]
    actual = dict(generation_config)
    actual["model_id"] = args.model_id
    actual["experiment_id"] = EXPERIMENT_ID
    actual["predictions_dir"] = str(predictions_dir.relative_to(repo_root))
    for key in MEASUREMENT_CONTRACT_KEYS:
        if stage1[key] != actual.get(key):
            raise SystemExit(
                f"REFUSING: the measurement contract is frozen at {key}="
                f"{stage1[key]!r} but this run would use {actual.get(key)!r}. "
                f"Batch layout is inside D_G, and the image batch is what the "
                f"measured generation instability is sensitive to.")
    same = [k for k in STUDY_IDENTITY_KEYS if stage1[k] == actual.get(k)]
    if same:
        raise SystemExit(
            f"REFUSING: {same} must DIFFER from Stage 1's. Sharing them would "
            f"put this study's files beside filed Stage-1 evidence under a "
            f"different experiment_id, and the sidecar would regenerate over "
            f"it rather than reuse it.")
    return generation_config


def compare_floor_numbers(before: dict, after: dict) -> dict[str, Any]:
    """Which of the four frozen numbers moved between two retentions dicts.

    Factored out of ``control_b0`` so the control's DECISION is testable without
    a parquet: the study stops iff this returns anything, and a test that cannot
    reach the decision cannot show that it stops.
    """
    return {f"{fam}.{est}": {"stage1": before[fam][est], "control":
                             after[fam][est]}
            for fam, est in FLOOR_NUMBERS
            if before[fam][est] != after[fam][est]}


def control_b0(args, repo_root: Path, data_dir: Path, predictions_dir: Path,
               generation_config: dict, queries, by_assoc) -> dict[str, Any]:
    """Is generation deterministic under a FIXED contract?

    B0 takes zero training steps and its adapter is a byte copy of MF, so any
    difference between this parquet and Stage 1's is generation noise and
    nothing else.  Measured rather than assumed, because the answer decides
    whether the anchor is a point or a distribution.
    """
    stage1_dir = repo_root / "data" / dataset_dir_for_tag("iter12") \
        / STAGE1_PREDICTIONS_SUBDIR
    before = load_predictions_parquet(
        stage1_dir / prediction_filename("B0"))
    adapter = repo_root / STAGE1_CKPT_ROOT / "B0" / "adapters"
    after = _generate_state(
        "B0", adapter, queries, by_assoc, repo_root, args.device,
        predictions_dir, data_dir, args.model_id, generation_config,
        EXPERIMENT_ID, SPLITS)
    fa = {r.query_id: r for r in before}
    fb = {r.query_id: r for r in after}
    if set(fa) != set(fb):
        raise SystemExit(
            f"REFUSING: the control covers {len(fa)} queries and the "
            f"regeneration {len(fb)}; they are not the same measurement")
    common = sorted(fa)
    differing = {f: sum(1 for k in common
                        if getattr(fa[k], f) != getattr(fb[k], f))
                  for f in CONTROL_FIELDS}
    probe = json.loads((repo_root / fisr.PROBE_REPORT).read_text())
    entities = probe["halves"]["probe"]["entities"]
    #: No entity map built here: ``probe_retention`` derives its own from the
    #: associations it is handed, and a second copy would be a second place for
    #: the two to disagree.
    ra = rs.probe_retention(before, queries, list(by_assoc.values()), entities)
    rb = rs.probe_retention(after, queries, list(by_assoc.values()), entities)
    moved = compare_floor_numbers(ra, rb)
    out = {
        "question": "is generation deterministic under a FIXED contract?",
        "why_it_matters": ("every replicate comparison assumes the anchor is a "
                           "point; if it is not, a point floor is not "
                           "interpretable and replication cannot repair it"),
        "adapter_is_a_byte_copy_of_mf_and_trains_zero_steps": True,
        "num_queries": len(common),
        "rows_differing_by_field": differing,
        "floor_numbers_that_moved": moved,
        "floor_is_reproducible": not moved,
        "already_known": {
            "measurement": ("two existing B0 parquets from identical adapter "
                            "bytes under image_batch_size 8 and 1"),
            "correctness_flips": 15,
            "of_which_inside_either_probe_subset": 0,
            "why": ("all 15 flips are image-route; the floor's 408 + 76 probe "
                    "queries are 100% text_to_text"),
            "d_g_components_that_moved": {"filr": -0.0004, "wrong": 0.0008},
        },
    }
    if moved:
        out["consequence"] = (
            "STOP. The four floor numbers are not reproducible from identical "
            "adapter bytes under an identical contract, so the Stage-1 anchor "
            "is a draw rather than a point. Replicating training seeds would "
            "not repair that, and no replicate verdict is reported.")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--image-batch-size", type=int, default=1)
    ap.add_argument("--max-new-tokens", type=int, default=96)
    ap.add_argument("--model-id", default="Qwen/Qwen3.5-9B")
    ap.add_argument("--candidates", default=None,
                    help="Comma-separated replicate ids (generate phase)")
    ap.add_argument("--control-only", action="store_true",
                    help="Run only the generation-determinism control")
    ap.add_argument("--generate-only", action="store_true",
                    help="Generate predictions and exit without scoring")
    ap.add_argument("--repo-root", default=None)
    args = ap.parse_args()

    repo_root = Path(args.repo_root) if args.repo_root \
        else (_find_repo_root(Path.cwd()) or Path.cwd())

    #: Step 1 -- the protocol this study is scored by must still be the one
    #: committed before its replicates were trained.
    reasons = fisr.verify_freeze(repo_root)
    if reasons:
        raise SystemExit("REFUSING: the seed-replication freeze does not match "
                         "the repository:\n  " + "\n  ".join(reasons))

    data_dir = repo_root / dataset_dir_for_tag("iter12")
    predictions_dir = data_dir / SEED_PREDICTIONS_SUBDIR
    out_report = repo_root / OUT_REPORT
    assert_no_forbidden_evidence(
        [data_dir, predictions_dir, out_report, repo_root / SEED_CKPT_ROOT],
        repo_root)
    generation_config = enforce_contract(args, repo_root, predictions_dir)
    predictions_dir.mkdir(parents=True, exist_ok=True)

    queries = load_queries_parquet(data_dir / "queries.parquet")
    associations = load_associations_parquet(data_dir / "associations.parquet")
    by_assoc = {a.association_id: a for a in associations}
    probe = json.loads((repo_root / fisr.PROBE_REPORT).read_text())
    entities = probe["halves"]["probe"]["entities"]

    if args.control_only:
        ctrl = control_b0(args, repo_root, data_dir, predictions_dir,
                          generation_config, queries, by_assoc)
        log.info("control: floor reproducible=%s | rows differing %s",
                 ctrl["floor_is_reproducible"],
                 json.dumps(ctrl["rows_differing_by_field"]))
        if not ctrl["floor_is_reproducible"]:
            raise SystemExit(json.dumps(ctrl["floor_numbers_that_moved"]))
        return

    reps = replicates()
    if args.candidates:
        wanted = {s.strip() for s in args.candidates.split(",") if s.strip()}
        unknown = wanted - {r.replicate_id for r in reps}
        if unknown:
            raise SystemExit(f"replicates outside the frozen set: "
                             f"{sorted(unknown)}")
        reps = [r for r in reps if r.replicate_id in wanted]

    stage1_dir = data_dir / STAGE1_PREDICTIONS_SUBDIR
    ckpt_root = repo_root / SEED_CKPT_ROOT
    per_replicate: dict[str, dict[str, Any]] = {}
    missing: list[str] = []
    for r in reps:
        if r.reused:
            #: Read, never regenerated: this is the parquet Stage 1 scored.
            ppath = stage1_dir / prediction_filename(r.parent)
            if not ppath.exists():
                missing.append(r.replicate_id)
                continue
            preds = load_predictions_parquet(ppath)
            source = str(ppath.relative_to(repo_root))
            adapter = repo_root / STAGE1_CKPT_ROOT / r.parent / "adapters"
        else:
            adapter = ckpt_root / r.replicate_id / "adapters"
            if not adapter.exists():
                missing.append(r.replicate_id)
                log.warning("[%s] adapters missing — skipped", r.replicate_id)
                continue
            preds = _generate_state(
                r.replicate_id, adapter, queries, by_assoc, repo_root,
                args.device, predictions_dir, data_dir, args.model_id,
                generation_config, EXPERIMENT_ID, SPLITS)
            source = str((predictions_dir / prediction_filename(
                r.replicate_id)).relative_to(repo_root))
        ret = rs.probe_retention(preds, queries, associations, entities)
        vec = rs.trainval_vector(preds, queries, associations)
        per_replicate[r.replicate_id] = {
            "parent": r.parent,
            "seed": r.seed,
            "reused_from_stage1": r.reused,
            "predictions": source,
            "probe_retention": ret,
            "trainval_vector": vec,
            "distance_to_mg": rs.distance_to_mg(
                vec, json.loads((repo_root / STAGE1_SELECTION_REPORT)
                                .read_text())["reference"]["vector"]),
        }
        log.info("[%s] seed %d %s", r.replicate_id, r.seed,
                 json.dumps({f"{f}.{e}": ret[f][e] for f, e in FLOOR_NUMBERS}))

    if args.generate_only:
        log.info("generate-only: %d replicate(s) have predictions, %d missing",
                 len(per_replicate), len(missing))
        if missing:
            raise SystemExit(f"missing: {missing}")
        return

    #: Step 2 -- the control must have run and passed before a verdict is
    #: reported, and its result is filed in the report rather than trusted from
    #: a previous invocation.
    ctrl_path = predictions_dir / prediction_filename("B0")
    if not ctrl_path.exists():
        raise SystemExit(
            "REFUSING to score: the generation-determinism control has not "
            "run. Run with --control-only first; if the anchor is not "
            "reproducible the replicate verdicts mean nothing.")

    if missing:
        log.error("missing adapters/predictions for %s — no report written",
                  missing)
        raise SystemExit(
            f"REFUSING to write a report over an incomplete replicate set: "
            f"{missing}. A mean over whichever replicates finished would look "
            f"like the mean over all of them.")

    stage1 = json.loads((repo_root / STAGE1_SELECTION_REPORT).read_text())
    anchor = stage1["floor"]["b0"]
    anchor_flat = {(f, e): anchor[f][e] for f, e in FLOOR_NUMBERS}
    ctrl = control_b0(args, repo_root, data_dir, predictions_dir,
                      generation_config, queries, by_assoc)

    verdicts: dict[str, Any] = {}
    for parent in sorted({r.parent for r in replicates()}):
        kids = [rid for rid, v in per_replicate.items()
                if v["parent"] == parent]
        kids.sort(key=lambda rid: per_replicate[rid]["seed"])
        agg: dict[str, Any] = {}
        for family, estimand in FLOOR_NUMBERS:
            vals = [per_replicate[k]["probe_retention"][family][estimand]
                    for k in kids]
            a = aggregate(vals)
            a["anchor"] = anchor_flat[(family, estimand)]
            a["mean_minus_anchor"] = a["mean"] - a["anchor"]
            a["replicates_at_or_above_anchor"] = sum(
                1 for v in vals
                if v >= anchor_flat[(family, estimand)] - rs.FLOOR_EPSILON)
            a["range_classification"] = range_classification(
                vals, anchor_flat[(family, estimand)], rs.FLOOR_EPSILON)
            agg[f"{family}.{estimand}"] = a
        #: Scored by the FROZEN floor_check on a dict of means, so the
        #: comparison is arithmetically the one Stage 1 used.
        floor = rs.floor_check(mean_candidate(
            {(f, e): [per_replicate[k]["probe_retention"][f][e] for k in kids]
             for f, e in FLOOR_NUMBERS}), anchor, rs.FLOOR_EPSILON)
        dists = [per_replicate[k]["distance_to_mg"] for k in kids]
        verdicts[parent] = {
            "replicates": kids,
            "seeds": [per_replicate[k]["seed"] for k in kids],
            "seed_42_reused_from_stage1": True,
            "per_number": agg,
            "floor_on_the_mean": floor,
            "passes_on_the_mean": floor["eligible"],
            "distance_to_mg": {**aggregate(dists),
                               "stage1_single_seed":
                                   stage1["candidates"][parent]
                                   ["distance_to_mg"]},
            "stage1_verdict": "disqualified",
            "stage1_worst_shortfall": min(
                stage1["candidates"][parent]["floor"]["checks"]
                [f"{f}.{e}"]["difference"] for f, e in FLOOR_NUMBERS),
        }
        log.info("[%s] mean-floor %s | %s", parent,
                 "PASS" if floor["eligible"] else "fail",
                 json.dumps({k: round(v["mean_minus_anchor"], 4)
                             for k, v in agg.items()}))

    report = {
        "iteration": 12,
        "stage": "1b — seed replication of the two near-miss candidates",
        "exploratory": True,
        "question": ("are the two smallest Stage-1 retain-same shortfalls "
                     "stable across independent training seeds, or one draw?"),
        "primary_rule": {
            "statement": ("for each of the four frozen numbers, the MEAN over "
                          "replicates must be at or above the Stage-1 B0 "
                          "anchor"),
            "scored_by": "granunlearn.evaluation.retention_selection.floor_check",
            "epsilon": rs.FLOOR_EPSILON,
            "mean_is_compared_unrounded": True,
            "why_unrounded": ("rounding the mean to the four decimals its "
                              "inputs carry would let a rounding step decide a "
                              "comparison the inputs did not"),
            "no_margin": True,
            "margin_required": 0.0,
        },
        "reported_but_not_decision_bearing": {
            "range_classification": ("every_replicate_below / straddles / "
                                     "every_replicate_at_or_above"),
            "why_not_an_interval": ("with k=4 an interval imports a "
                                    "distributional assumption the design does "
                                    "not need; the observed range answers "
                                    "whether every independent seed fell below "
                                    "the anchor"),
            "distance_to_mg": ("reported per replicate and as a mean, but the "
                               "floor decides eligibility, exactly as in "
                               "Stage 1"),
        },
        "anchor": {"source": STAGE1_SELECTION_REPORT,
                   "values": {f"{f}.{e}": anchor_flat[(f, e)]
                              for f, e in FLOOR_NUMBERS},
                   "why_a_point": ("B0 is the no-op copy of MF and takes zero "
                                   "training steps, so it has no seed "
                                   "variance; the control below tests whether "
                                   "it has generation variance")},
        "generation_determinism_control": ctrl,
        "measurement_contract": {k: generation_config.get(k, args.model_id)
                                 for k in MEASUREMENT_CONTRACT_KEYS},
        "study_identity": {k: (EXPERIMENT_ID if k == "experiment_id" else
                               str(predictions_dir.relative_to(repo_root)))
                           for k in STUDY_IDENTITY_KEYS},
        "replicates": per_replicate,
        "verdicts": verdicts,
        "never_read": list(rs.FORBIDDEN_EVIDENCE),
        "limitations": [
            "Replication does not increase the query count. The probe half is "
            "35 entities / 408 retain-same and 23 donors / 76 retain-other "
            "queries, fixed by the association pool, so retain-other remains "
            "resolvable only in units of 1/76 = 0.0132 however many seeds are "
            "run.",
            "k=4 replicates per parent (seed 42 reused from Stage 1 plus 43, "
            "44, 45). A mean over four draws bounds training-seed variance "
            "loosely; it does not license an interval claim.",
            "Two parents only, chosen as the two smallest retain-same "
            "shortfalls in the filed Stage-1 report. This says nothing about "
            "the other six candidates.",
            "Exploratory: no hypothesis is tested and no error rate is "
            "controlled. A pass here is a reason to design a confirmatory "
            "protocol, not a claim.",
        ],
    }
    out_report.parent.mkdir(parents=True, exist_ok=True)
    out_report.write_text(json.dumps(report, indent=2, sort_keys=False) + "\n")
    log.info("Seed-replication report -> %s", out_report)


if __name__ == "__main__":
    main()
