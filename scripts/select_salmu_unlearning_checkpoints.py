"""Select SALMU unlearning checkpoints by distance to MG.

    python scripts/select_salmu_unlearning_checkpoints.py --device cuda:3

Protocol (mirrors Iteration 9):
* summary vector v(M) = [prefers_fine, prefers_target_not_fine,
  sim_fine, sim_target, sim_sibling] over TARGET-persona probes
* hyperparameter selection uses TRAIN+VAL probe personas ONLY
* the TEST probe personas are a frozen held-out evaluation, reported
  exactly once after selection
* individual metrics stay primary; the distance is the compact
  selection criterion only
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

from granunlearn.config import _find_repo_root
from granunlearn.logging_utils import setup_logger
from granunlearn.salmu.adapter import REPOS, locate_repo
from granunlearn.salmu.embedding_metrics import aggregate_scores
from granunlearn.salmu.paths import SalmuPaths
from granunlearn.salmu.eval_utils import (
    SalmuImageIndex,
    aggregate_probe_results,
    build_release_probes,
    build_retain_probes,
    save_probe_cache,
    score_probes,
)
from granunlearn.salmu.unlearning import split_target_personas

log = setup_logger("select_salmu_unlearning_checkpoints")

SUMMARY_COMPONENTS = ("prefers_fine_rate", "prefers_target_not_fine_rate",
                      "sim_fine", "sim_target", "sim_sibling",
                      "same_entity_retain_sim",
                      "other_entity_retain_sim")


def summary_vector(agg: dict,
                   same_entity_retain_sim: float | None = None,
                   other_entity_retain_sim: float | None = None,
                   ) -> dict:
    """Build the selection summary vector from TARGET-ONLY aggregates.

    ``agg`` must already be filtered to ``is_target_attr=True`` probes
    (so that same-entity retain probes do not dilute the MG-distance
    signal).  Retain similarities are passed separately.
    """
    sims = agg.get("mean_similarities") or {}
    vec = {
        "prefers_fine_rate": agg.get("prefers_fine_rate"),
        "prefers_target_not_fine_rate":
            agg.get("prefers_target_not_fine_rate"),
        "sim_fine": sims.get("fine"),
        "sim_target": sims.get("target"),
        "sim_sibling": sims.get("sibling"),
        "same_entity_retain_sim": same_entity_retain_sim,
        "other_entity_retain_sim": other_entity_retain_sim,
    }
    return {k: v for k, v in vec.items() if v is not None}


def distance_to_mg(vec: dict[str, float],
                   ref_vec: dict[str, float]) -> tuple[float | None,
                                                       list[str]]:
    """Weighted-L1 distance on SALMU summary components.

    (The MLLMU selection.distance_to_reference iterates its OWN
    frozen component list — filr/tga/... — so SALMU uses a parallel
    implementation with the same missing-component renormalization.)
    """
    used: list[str] = []
    total = 0.0
    for comp in SUMMARY_COMPONENTS:
        a, b = vec.get(comp), ref_vec.get(comp)
        if a is None or b is None:
            continue
        total += abs(a - b)
        used.append(comp)
    if not used:
        return None, []
    return round(total / len(used), 6), used


# ── Per-state shard caching (parallel / resumable scoring) ────────

def _shard_dir(repo_root: Path, suffix: str = "") -> Path:
    return SalmuPaths(repo_root, suffix=suffix).shard_dir


def _shard_path(repo_root: Path, state: str, kind: str,
                suffix: str = "") -> Path:
    return _shard_dir(repo_root, suffix) / f"{state}.{kind}.json"


def load_shard(repo_root: Path, state: str, kind: str,
               suffix: str = ""):
    """Load one per-state shard (``kind`` in {'target','retain'}).

    Returns None if the shard is absent so callers score it.
    """
    p = _shard_path(repo_root, state, kind, suffix)
    if not p.exists():
        return None
    return json.loads(p.read_text())


def save_shard(repo_root: Path, state: str, kind: str, results,
               suffix: str = "") -> None:
    p = _shard_path(repo_root, state, kind, suffix)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(results, f)
    tmp.replace(p)  # atomic — safe across parallel workers


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Select SALMU unlearning checkpoints")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-images", type=int, default=None,
                        help="Cap images per (persona, attr)")
    parser.add_argument("--max-captions", type=int, default=None,
                        help="Cap captions per (persona, attr)")
    parser.add_argument("--subset", default=None,
                        help="Comma-separated state ids to SCORE only "
                             "(parallel worker). Skips selection.")
    parser.add_argument("--suffix", default="",
                        help="Iteration tag (e.g. r5 -> holdout-clean "
                             "artifacts under *_r5 paths)")
    args = parser.parse_args()

    repo_root = _find_repo_root(Path.cwd()) or Path.cwd()
    paths = SalmuPaths(repo_root, suffix=args.suffix)
    unlearn_root = paths.unlearn_root
    train_ds = locate_repo(REPOS["training_dataset"]["repo_id"], "dataset")
    sys.path.insert(0, str(repo_root / "scripts"))
    import train_salmu_unlearning_baselines as tmod  # registry source

    probes, target_ids = build_release_probes(
        repo_root, max_images=args.max_images,
        max_captions=args.max_captions, suffix=args.suffix)
    split = split_target_personas(target_ids)
    trainval = set(split["train"]) | set(split["val"])
    test = set(split["test"])
    log.info("Probe persona split: train %d / val %d / test %d",
             len(split["train"]), len(split["val"]), len(split["test"]))

    states = ["MG", "B0"] + [tmod.candidate_id(m, lr, lam, c)
                             for m, lr, _, _, lam, c in tmod.CANDIDATES]
    # Filter out states with missing checkpoints
    available_states = []
    for state in states:
        ckpt_path = unlearn_root / state / "pytorch_model.bin"
        if state in ("MG", "B0") or ckpt_path.exists():
            available_states.append(state)
        else:
            log.warning("[%s] skipping — checkpoint missing", state)
    states = available_states
    # Parallel-worker mode: restrict scoring to a subset of states.
    score_states = states
    worker_mode = False
    if args.subset:
        wanted = {s.strip() for s in args.subset.split(",") if s.strip()}
        score_states = [s for s in states if s in wanted]
        missing = wanted - set(states)
        if missing:
            log.warning("subset contains unknown states: %s",
                        sorted(missing))
        worker_mode = True
        log.info("WORKER mode: scoring %d/%d states on %s: %s",
                 len(score_states), len(states), args.device,
                 score_states)

    cache_path = paths.target_cache_path
    retain_cache_path = paths.retain_cache_path
    retain_probes = build_retain_probes(
        repo_root, max_images=args.max_images,
        max_captions=args.max_captions, suffix=args.suffix)
    log.info("Built %d retain probes", len(retain_probes))
    image_index = None
    for state in score_states:
        # Per-state shards: load if present, else score + save shard.
        tgt = load_shard(repo_root, state, "target", args.suffix)
        ret = load_shard(repo_root, state, "retain", args.suffix)
        if tgt is not None and ret is not None:
            log.info("[%s] reusing shard probe sims", state)
            continue
        if image_index is None:
            image_index = SalmuImageIndex(train_ds / "data")
        if tgt is None:
            tgt = score_probes(state, probes, image_index,
                               repo_root, args.device,
                               unlearn_root=unlearn_root,
                               ref_root=paths.ref_ckpt_root)
            save_shard(repo_root, state, "target", tgt, args.suffix)
        if ret is None:
            ret = score_probes(state, retain_probes, image_index,
                               repo_root, args.device,
                               unlearn_root=unlearn_root,
                               ref_root=paths.ref_ckpt_root)
            save_shard(repo_root, state, "retain", ret, args.suffix)
    if worker_mode:
        log.info("WORKER mode: shard scoring complete — selection "
                 "deferred to the aggregation run.")
        return

    # Aggregation run: assemble full caches from ALL state shards.
    cache, retain_cache = {}, {}
    incomplete = []
    for state in states:
        tgt = load_shard(repo_root, state, "target", args.suffix)
        ret = load_shard(repo_root, state, "retain", args.suffix)
        if tgt is None or ret is None:
            incomplete.append(state)
            continue
        cache[state], retain_cache[state] = tgt, ret
    if incomplete:
        raise RuntimeError(
            "Cannot run selection — shards missing for states: "
            f"{incomplete}. Score them first (e.g. with --subset).")
    save_probe_cache(cache_path, cache)
    save_probe_cache(retain_cache_path, retain_cache)

    def same_entity_retain_sim(state: str,
                               personas: set[str]) -> float | None:
        """Mean fine sim on same-entity retain probes (target personas,
        non-target attributes) restricted to ``personas`` — the
        summary vector must use the SAME identities as the component
        it accompanies (trainval for selection, test for the final
        verdict; 10R4)."""
        results = cache[state]
        se = [r for r in results if not r.get("is_target_attr", False)]
        agg = aggregate_probe_results(se, personas)
        return (agg.get("mean_similarities") or {}).get("fine")

    def other_entity_retain_sim(state: str) -> float | None:
        """Mean fine sim on other-entity retain probes (retain personas)."""
        agg = aggregate_probe_results(retain_cache[state])
        return (agg.get("mean_similarities") or {}).get("fine")

    # Selection on TARGET-ONLY train+val probes so that same-entity
    # retain probes do not dilute the MG-distance signal.
    # 10R4: all aggregates are association-weighted (macro-average of
    # image-caption variants) and carry association bootstrap CIs.
    mg_vec = summary_vector(
        aggregate_probe_results(cache["MG"], trainval,
                                target_attr_only=True,
                                bootstrap_ci=True),
        same_entity_retain_sim("MG", trainval),
        other_entity_retain_sim("MG"))
    report_rows = []
    for state in states:
        agg_tv = aggregate_probe_results(
            cache[state], trainval, target_attr_only=True,
            bootstrap_ci=True)
        se_sim = same_entity_retain_sim(state, trainval)
        oe_sim = other_entity_retain_sim(state)
        vec = summary_vector(agg_tv, se_sim, oe_sim)
        dist, used = distance_to_mg(vec, mg_vec)
        report_rows.append({
            "candidate_id": state,
            "trainval_metrics": agg_tv,
            "same_entity_retain_sim": se_sim,
            "other_entity_retain_sim": oe_sim,
            "trainval_summary_vector": vec,
            "distance_to_MG_trainval": dist,
            "components_used": used,
        })

    selected: dict[str, str] = {"B0": "B0"}
    for method in ("B1", "B2", "B3"):
        # Exclude B2_retain_* from the B2 family — they are a
        # different method (target_level SFT + retain SFT) and must
        # not compete with pure B2 (target_level SFT only).
        if method == "B2":
            rows = [r for r in report_rows
                    if r["candidate_id"].startswith("B2_")
                    and not r["candidate_id"].startswith("B2_retain_")]
        else:
            rows = [r for r in report_rows
                    if r["candidate_id"].startswith(method + "_")]
        if any(r["distance_to_MG_trainval"] is None for r in rows):
            raise RuntimeError(
                f"distance undefined for {method} — summary vectors "
                f"share no components")
        best = min(rows, key=lambda r: r["distance_to_MG_trainval"])
        selected[method] = best["candidate_id"]
    log.info("Selected on train+val: %s", selected)

    # Stage selected checkpoints
    sel_root = unlearn_root / "selected"
    for method, cid in selected.items():
        dst = sel_root / method
        dst.mkdir(parents=True, exist_ok=True)
        src = unlearn_root / cid
        for fname in ("pytorch_model.bin", "training_summary.json"):
            s, d = src / fname, dst / fname
            if s.exists() and s.resolve() != d.resolve():
                shutil.copy2(s, d)

    # Frozen held-out test verdict: computed ONCE, for the SELECTED
    # checkpoints ONLY (10R4).  Evaluating every candidate on the
    # test split would turn it into a second selection set; the
    # untouched-test protocol requires selected-checkpoint-only
    # reporting.  Same-entity retention here uses the TEST
    # identities, matching the target-component identities.
    for row in report_rows:
        cid = row["candidate_id"]
        if cid not in selected.values():
            continue
        agg_test = aggregate_probe_results(
            cache[cid], test, target_attr_only=True,
            bootstrap_ci=True)
        row["test_metrics"] = agg_test
        # Target-only per-attribute breakdown on the frozen test split
        test_results = [r for r in cache[cid]
                        if r["identity_id"] in test
                        and r.get("is_target_attr", False)]
        row["test_per_attribute"] = {
            attr: aggregate_scores(
                [r for r in test_results if r["attribute"] == attr],
                bootstrap_ci=True)
            for attr in ("city", "job", "blood_type")
        }
        row["test_summary_vector"] = summary_vector(
            agg_test,
            same_entity_retain_sim(cid, test),
            other_entity_retain_sim(cid))

    # Document the B3 near-tie transparently (computed, not hardcoded)
    b3_constrained = sorted(
        (r for r in report_rows
         if r["candidate_id"].startswith("B3_")
         and r["distance_to_MG_trainval"] is not None),
        key=lambda r: r["distance_to_MG_trainval"])
    b3_note = ("10R4 B3 train+val ranking (target-only, "
               "association-weighted): "
               + "; ".join(f"{r['candidate_id']} "
                           f"dist={r['distance_to_MG_trainval']}"
                           for r in b3_constrained))
    # Association / variant counts for honest metadata
    tv_target_assoc = len({(r["identity_id"], r["attribute"])
                           for r in cache["MG"]
                           if r["identity_id"] in trainval
                           and r.get("is_target_attr", False)})
    tv_target_probes = sum(1 for r in cache["MG"]
                           if r["identity_id"] in trainval
                           and r.get("is_target_attr", False))
    # 10R5a: the test-protocol text is DERIVED from the actual persona
    # split and iteration — never hardcoded stale counts.
    n_train, n_val, n_test = (len(split["train"]), len(split["val"]),
                              len(split["test"]))
    n_total = len(target_ids)
    split_desc = (f"train/val/test = {n_train}/{n_val}/{n_test} of "
                  f"{n_total} target personas")
    if args.suffix == "r5":
        test_protocol = (
            f"held-out internal test identities ({split_desc}). Test "
            "metrics are computed for the SELECTED checkpoints only. "
            "This iteration's checkpoints and personas were never "
            "scored on the test split before selection, so these "
            "internal test numbers are a genuine held-out verdict "
            "for the 10R5 chain. The untouched EXTERNAL evaluation "
            "is the validated holdout-clean official-split report "
            "(salmu_official_splits_r5.json).")
    else:
        test_protocol = (
            f"held-out internal test identities ({split_desc}). "
            "Since 10R4, test metrics are computed for the SELECTED "
            "checkpoints only. NOTE: this split was scored "
            "candidate-wide in Iterations 10R2/10R3, so it is NOT "
            "genuinely untouched — treat these test numbers as "
            "exploratory. The released official splits are ALSO only "
            "transfer diagnostics for this chain: their holdout "
            "pairs were consumed by GMUL training. A genuinely "
            "untouched evaluation requires the Iteration 10R5 "
            "holdout-clean retrain.")
    report = {
        "experiment_id": (
            "salmu_iter10r5_unlearning_selection"
            if args.suffix == "r5"
            else "salmu_iter10r4_unlearning_selection"),
        "iteration_suffix": args.suffix or None,
        "persona_split": {"train": split["train"], "val": split["val"],
                          "test": split["test"]},
        "multi_image": True,
        "multi_image_note": (
            "ALL released image/caption variants per association are "
            "used (no cap); max_images/max_captions null = uncapped."
            if args.max_images is None and args.max_captions is None
            else "Probe variants are CAPPED per (persona, attribute) "
                 "by max_images/max_captions; association-weighted "
                 "aggregation makes the cap a variance control, not "
                 "a weighting change."),
        "max_images": args.max_images,
        "max_captions": args.max_captions,
        "weighting": "association_macro_average",
        "trainval_target_associations": tv_target_assoc,
        "trainval_target_probe_combinations": tv_target_probes,
        "bootstrap_ci": True,
        "summary_components": list(SUMMARY_COMPONENTS),
        "target_only_selection": True,
        "b2_retain_excluded": True,
        "test_protocol": test_protocol,
        "mg_trainval_vector": mg_vec,
        "selected": selected,
        "candidates": report_rows,
        "selection_rule": "min distance_to_MG over TARGET-ONLY "
                          "TRAIN+VAL probes (is_target_attr=True), "
                          "association-weighted (each (identity, "
                          "attribute) counted once); "
                          "same_entity_retain_sim and "
                          "other_entity_retain_sim are separate "
                          "components; B2_retain_* excluded from B2 "
                          "family; test personas frozen.",
        "notes": ([
            "10R5 holdout-clean protocol: the pair universe is the "
            "official forget split ONLY; no released "
            "holdout_association/holdout_identity pair enters ANY "
            "training group; target associations are selected "
            "exclusively from forget. Holdout cleanliness is "
            "VALIDATED (manifest allowed_split == forget and zero "
            "exact holdout overlap) in "
            "salmu_official_splits_r5.json:holdout_clean_validation; "
            "released-split evaluation on this chain is therefore an "
            "untouched external evaluation.",
            "10R5a test-protocol honesty: this iteration's "
            f"personas/split ({split_desc}) were built fresh from "
            "the holdout-clean pair set and never scored before "
            "selection, so the internal test verdict is genuine for "
            "these checkpoints.",
        ] if args.suffix == "r5" else []) + [
            "10R4 corrected selection: association-weighted — "
            f"{tv_target_assoc} train+val target associations "
            f"({tv_target_probes} image-caption combinations "
            "macro-averaged); retain similarities are separate "
            "summary components; association-bootstrap 95% CIs on "
            "every aggregate.",
            "same_entity_retain_sim uses the same identities as the "
            "target component it accompanies (train+val for "
            "selection, test for the final verdict).",
        ] + ([] if args.suffix == "r5" else [
            "10R4a test-protocol honesty: the internal test split "
            "was inspected candidate-wide in 10R2/10R3; results on "
            "it are exploratory, not an untouched final verdict.",
        ]) + [
            b3_note,
        ],
    }
    out = paths.report("salmu_unlearning_selection")
    with open(out, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    log.info("Wrote selection report -> %s", out)
    for row in report_rows:
        if row["candidate_id"] in selected.values():
            tm = row["test_metrics"]
            log.info("[TEST %s <- %s] fine_pref=%s tnf=%s sims=%s",
                     [m for m, c in selected.items()
                      if c == row["candidate_id"]][0],
                     row["candidate_id"],
                     tm.get("prefers_fine_rate"),
                     tm.get("prefers_target_not_fine_rate"),
                     tm.get("mean_similarities"))


if __name__ == "__main__":
    main()
