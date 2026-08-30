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
from granunlearn.salmu.eval_utils import (
    SalmuImageIndex,
    aggregate_probe_results,
    build_release_probes,
    build_retain_probes,
    load_probe_cache,
    save_probe_cache,
    score_probes,
)
from granunlearn.salmu.unlearning import split_target_personas

log = setup_logger("select_salmu_unlearning_checkpoints")

SUMMARY_COMPONENTS = ("prefers_fine_rate", "prefers_target_not_fine_rate",
                      "sim_fine", "sim_target", "sim_sibling",
                      "retain_fine_sim")


def summary_vector(agg: dict, retain_fine_sim: float | None) -> dict:
    sims = agg.get("mean_similarities") or {}
    vec = {
        "prefers_fine_rate": agg.get("prefers_fine_rate"),
        "prefers_target_not_fine_rate":
            agg.get("prefers_target_not_fine_rate"),
        "sim_fine": sims.get("fine"),
        "sim_target": sims.get("target"),
        "sim_sibling": sims.get("sibling"),
        "retain_fine_sim": retain_fine_sim,
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Select SALMU unlearning checkpoints")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--skip-existing", action="store_true",
                        help="reuse cached per-probe sims")
    parser.add_argument("--max-images", type=int, default=None,
                        help="Cap images per (persona, attr)")
    parser.add_argument("--max-captions", type=int, default=None,
                        help="Cap captions per (persona, attr)")
    args = parser.parse_args()

    repo_root = _find_repo_root(Path.cwd()) or Path.cwd()
    unlearn_root = repo_root / "data" / "checkpoints" / "salmu_unlearn"
    train_ds = locate_repo(REPOS["training_dataset"]["repo_id"], "dataset")
    sys.path.insert(0, str(repo_root / "scripts"))
    import train_salmu_unlearning_baselines as tmod  # registry source

    probes, target_ids = build_release_probes(
        repo_root, max_images=args.max_images,
        max_captions=args.max_captions)
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
    cache_path = repo_root / "data" / "salmu_hierarchical" / \
        "probe_sims_unlearn.json"
    cache = load_probe_cache(cache_path) \
        if args.skip_existing else None
    cache = cache or {}
    # Retain collateral-damage probes (same states; separate cache)
    retain_cache_path = repo_root / "data" / "salmu_hierarchical" / \
        "probe_sims_retain.json"
    retain_cache = load_probe_cache(retain_cache_path) \
        if args.skip_existing else None
    retain_cache = retain_cache or {}
    retain_probes = build_retain_probes(
        repo_root, max_images=args.max_images,
        max_captions=args.max_captions)
    log.info("Built %d retain probes", len(retain_probes))
    image_index = None
    for state in states:
        need_target = state not in cache
        need_retain = state not in retain_cache
        if not need_target and not need_retain:
            log.info("[%s] reusing cached probe sims", state)
            continue
        if image_index is None:
            image_index = SalmuImageIndex(train_ds / "data")
        if need_target:
            cache[state] = score_probes(state, probes, image_index,
                                        repo_root, args.device,
                                        unlearn_root=unlearn_root)
        if need_retain:
            retain_cache[state] = score_probes(
                state, retain_probes, image_index, repo_root,
                args.device, unlearn_root=unlearn_root)
    save_probe_cache(cache_path, cache)
    save_probe_cache(retain_cache_path, retain_cache)

    def retain_sim(state: str) -> float | None:
        agg = aggregate_probe_results(retain_cache[state])
        return (agg.get("mean_similarities") or {}).get("fine")

    # Selection on train+val personas only (retain probes are global:
    # they measure collateral damage, not the forgetting target)
    mg_vec = summary_vector(aggregate_probe_results(
        cache["MG"], trainval), retain_sim("MG"))
    report_rows = []
    for state in states:
        agg_tv = aggregate_probe_results(cache[state], trainval)
        vec = summary_vector(agg_tv, retain_sim(state))
        dist, used = distance_to_mg(vec, mg_vec)
        report_rows.append({
            "candidate_id": state,
            "trainval_metrics": agg_tv,
            "retain_fine_sim": retain_sim(state),
            "trainval_summary_vector": vec,
            "distance_to_MG_trainval": dist,
            "components_used": used,
        })

    selected: dict[str, str] = {"B0": "B0"}
    for method in ("B1", "B2", "B3"):
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

    # Frozen held-out test verdict (computed once, never used to choose)
    for row in report_rows:
        cid = row["candidate_id"]
        agg_test = aggregate_probe_results(cache[cid], test)
        row["test_metrics"] = agg_test
        row["test_summary_vector"] = summary_vector(
            agg_test, retain_sim(cid))

    report = {
        "experiment_id": "salmu_iter10r2_unlearning_selection",
        "persona_split": {"train": split["train"], "val": split["val"],
                          "test": split["test"]},
        "multi_image": args.max_images is not None or
                       args.max_captions is not None,
        "max_images": args.max_images,
        "max_captions": args.max_captions,
        "summary_components": list(SUMMARY_COMPONENTS),
        "mg_trainval_vector": mg_vec,
        "selected": selected,
        "candidates": report_rows,
        "selection_rule": "min distance_to_MG over TRAIN+VAL probe "
                          "personas only; test personas frozen and "
                          "evaluated exactly once after selection",
    }
    out = repo_root / "data" / "reports" / \
        "salmu_unlearning_selection.json"
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
