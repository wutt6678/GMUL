"""Run the OFFICIAL SALMUBench evaluator on GMUL iteration states.

    python scripts/run_official_salmubench_eval.py --suffix r5 \
        --device cuda:1

Uses the official evaluation code from github.com/cvc-mmu/salmubench
(``evaluation/evaluation.py``, pinned at commit
``8b7f439746862f09b7f0f44ae5004f297cc2dec9``) VERBATIM — only the
DATA SOURCE is redirected from the Hub repo id to our pinned local
snapshot (same content, same benchmark revision), so the run is
offline-reproducible against the committed provenance.

Official metrics produced per state:
* RetFail   (1.1) — R@1 / MRR over a 2,000-distractor caption gallery
* AssocStr  (1.2) — mean cos-sim on forget
* ACS       (1.3) — member-vs-shuffled logistic-probe accuracy
* IdZSC     (1.4) — zero-shot identity-name classification
* CoreAssoc (1.5) — max sim of "{name} {value}" / "{value} {name}"
* GenKnow   (2.1) — ImageNet-1k zero-shot (SKIPPED: no local
  ImageNet webdataset)
* InterIdSim / IntraIdSim / VisIdInt / FragSim (2.2-2.5) — utility
  similarities on holdout_identity / holdout_association /
  retain_joint / fragile_set (retain utility)

Outputs:
* raw per-state JSONs (official format) under
  ``data/salmu_hierarchical/official_salmubench_results[_suffix]/``
* aggregated report ``data/reports/salmubench_official_eval[_suffix].json``
  with provenance (official repo commit, benchmark revision,
  checkpoint SHA-256s) and a cross-check against our own
  released-split evaluator.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from granunlearn.config import _find_repo_root
from granunlearn.logging_utils import setup_logger
from granunlearn.salmu.adapter import REPOS, locate_repo
from granunlearn.salmu.paths import SalmuPaths

log = setup_logger("run_official_salmubench_eval")

DEFAULT_SALMUBENCH_REPO = "/scratch/wutiantong/GMUL/salmubench"
OFFICIAL_REPO_COMMIT = "8b7f439746862f09b7f0f44ae5004f297cc2dec9"
OFFICIAL_REPO_URL = "https://github.com/cvc-mmu/salmubench"

# (report key, official section, official metric key, statistic key)
METRIC_MAP = [
    ("RetFail_R@1", "efficacy", "1.1_RetFail", "R@1"),
    ("RetFail_MRR", "efficacy", "1.1_RetFail", "MRR"),
    ("AssocStr", "efficacy", "1.2_AssocStr", "mean"),
    ("ACS", "efficacy", "1.3_ACS", "acs_accuracy"),
    ("IdZSC", "efficacy", "1.4_IdZSC",
     "identity_classification_accuracy"),
    ("CoreAssoc", "efficacy", "1.5_CoreAssoc", "mean"),
    ("GenKnow", "utility", "2.1_GenKnow", "ImageNet1k_Accuracy"),
    ("InterIdSim", "utility", "2.2_InterIdSim", "mean"),
    ("IntraIdSim", "utility", "2.3_IntraIdSim", "mean"),
    ("VisIdInt", "utility", "2.4_VisIdInt", "mean"),
    ("FragSim", "utility", "2.5_FragSim", "mean"),
]


def sha256_file(path: Path) -> str | None:
    if not path.exists():
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _snapshot_revision(path: Path) -> str | None:
    parts = Path(path).parts
    if "snapshots" in parts:
        idx = parts.index("snapshots")
        if idx + 1 < len(parts):
            return parts[idx + 1]
    return None


def stage_checkpoints(paths: SalmuPaths, staging: Path,
                      sel_report: Path) -> dict[str, Path]:
    """Stage every state as a ``.pth`` file the official loader can
    torch.load: symlinks for our raw state-dict checkpoints and one
    safetensors->pth conversion for BASE (Clean CLIP)."""
    import torch
    staging.mkdir(parents=True, exist_ok=True)
    tag = paths.tag or "orig"
    staged: dict[str, Path] = {}

    # BASE: released Clean CLIP (safetensors -> pth conversion)
    clean = locate_repo(REPOS["clean_model"]["repo_id"], "model")
    base_pth = staging / f"BASE_{tag}.pth"
    if not base_pth.exists():
        from safetensors.torch import load_file
        torch.save(load_file(str(clean / "open_clip_model.safetensors")),
                   base_pth)
        log.info("[BASE] converted safetensors -> %s", base_pth)
    staged["BASE"] = base_pth

    # COMPROMISED: released benchmark checkpoint (raw open_clip dict)
    comp = locate_repo(REPOS["compromised_model"]["repo_id"], "model")
    comp_pth = staging / f"COMPROMISED_{tag}.pth"
    if not comp_pth.exists():
        comp_pth.symlink_to(comp / "open_clip_pytorch_model.bin")
    staged["COMPROMISED"] = comp_pth

    states = ["MF", "MG", "MN", "B0"]
    if sel_report.exists():
        sel = json.loads(sel_report.read_text())
        for method in ("B1", "B2", "B3"):
            cid = sel.get("selected", {}).get(method)
            if cid:
                states.append(cid)
    for state in states:
        ckpt = paths.ref_checkpoint(state)
        if not ckpt.exists():
            ckpt = paths.candidate_checkpoint(state)
        if not ckpt.exists():
            raise FileNotFoundError(f"Missing checkpoint for {state}")
        pth = staging / f"{state}_{tag}.pth"
        if not pth.exists():
            pth.symlink_to(ckpt.resolve())
        staged[state] = pth
    return staged


def make_local_evaluator_class(salmubench_repo: Path,
                               snapshot_dir: Path):
    """Import the official evaluator and subclass it with a
    local-snapshot data loader (official logic replicated verbatim;
    only the Hub fetch is replaced)."""
    eval_dir = salmubench_repo / "evaluation"
    if not (eval_dir / "evaluation.py").exists():
        raise FileNotFoundError(
            f"Official evaluator not found under {eval_dir}")
    sys.path.insert(0, str(eval_dir))
    import evaluation as official  # noqa: PLC0415

    class LocalSALMUBenchEvaluation(official.SALMUBenchEvaluation):
        """Same protocol as the official runner; data comes from the
        PINNED LOCAL snapshot instead of a Hub download."""

        def load_benchmark_data(self):
            import numpy as np
            from datasets import load_dataset

            print(f"[INFO] Loading benchmark data from the pinned "
                  f"local snapshot: {snapshot_dir}")
            self.dataset = load_dataset(
                str(snapshot_dir), trust_remote_code=True)
            print("[SUCCESS] Benchmark dataset loaded.")

            self.captions_metadata = json.loads(
                (snapshot_dir
                 / "sensitive_set_captions_metadata.json").read_text())
            fragile_set_ids = set(json.loads(
                (snapshot_dir / "fragile_set_ids.json").read_text()))
            print("[SUCCESS] All metadata loaded.")

            # Derived splits — replicated VERBATIM from the official
            # load_benchmark_data (commit 8b7f439).
            print("\n[INFO] Creating derived dataset splits for "
                  "evaluation...")
            all_ids_forget = set(self.dataset["forget"]["identity_id"])
            self.dataset["retain_joint"] = self.dataset[
                "retain_synth"].filter(
                lambda x: (x["identity_id"] is not None)
                and (x["identity_id"] in all_ids_forget),
                num_proc=self.num_workers)
            print(f"[INFO] -> 'retain_joint': "
                  f"{len(self.dataset['retain_joint'])} samples.")
            self.dataset["retain_disjoint"] = self.dataset[
                "retain_synth"].filter(
                lambda x: x["identity_id"] is None,
                num_proc=self.num_workers)
            print(f"[INFO] -> 'retain_disjoint': "
                  f"{len(self.dataset['retain_disjoint'])} samples.")
            self.dataset["fragile_set"] = self.dataset[
                "retain_disjoint"].filter(
                lambda x: x["file_name"] in fragile_set_ids,
                num_proc=self.num_workers)
            print(f"[INFO] -> 'fragile_set': "
                  f"{len(self.dataset['fragile_set'])} samples.")
            ids_in_holdout_association = set(
                self.dataset["holdout_association"]["identity_id"])
            self.dataset["forget_identity"] = self.dataset[
                "forget"].filter(
                lambda x: x["identity_id"]
                not in ids_in_holdout_association,
                num_proc=self.num_workers)
            print(f"[INFO] -> 'forget_identity': "
                  f"{len(self.dataset['forget_identity'])} samples "
                  "for IdZSC.")

            print("\n[INFO] Creating distractor text pools for "
                  "Retrieval Failure metric...")
            self.distractors_forget_text = np.random.choice(
                self.dataset["forget"]["text"], 1000, replace=False)
            self.distractors_retain_text = np.random.choice(
                self.dataset["retain_disjoint"]["text"], 1000,
                replace=False)
            print("[SUCCESS] Data loading and preparation complete.")

        def _load_json_from_hub(self, filename):
            return json.loads(
                (snapshot_dir / filename).read_text())

    return LocalSALMUBenchEvaluation


def aggregate(results_dir: Path, staged: dict[str, Path],
              paths: SalmuPaths, bench_revision: str | None,
              ours_report: Path) -> dict:
    """Compact per-state metric table + provenance + cross-check
    against our own released-split evaluator."""
    by_state: dict[str, dict] = {}
    hashes = {state: sha256_file(p.resolve())
              for state, p in staged.items()}
    for state, pth in staged.items():
        # Official output naming (evaluation.py evaluate_model):
        # slug = Path(identifier with '/' and ':' -> '_').stem, then
        # "evaluation_<slug>.json".replace("__","_").replace("_._","_")
        slug = Path(str(pth).replace("/", "_").replace(":", "_")).stem
        name = (f"evaluation_{slug}.json"
                .replace("__", "_").replace("_._", "_"))
        out = results_dir / name
        if not out.exists():
            log.warning("[%s] official results missing (%s)", state,
                        name)
            continue
        raw = json.loads(out.read_text())
        entry: dict = {"results_file": out.name}
        for name, section, key, stat in METRIC_MAP:
            val = raw.get(section, {}).get(key, {}).get(stat)
            if name == "GenKnow" and val is not None and val < 0:
                val = None  # -1.0 = skipped by the official code
            entry[name] = val
        by_state[state] = entry

    crosscheck: dict = {}
    if ours_report.exists():
        ours = json.loads(ours_report.read_text())
        ours_states = ours.get("states", {})
        pairs = [("AssocStr", "forget"),
                 ("IntraIdSim", "holdout_association"),
                 ("InterIdSim", "holdout_identity")]
        for state in by_state:
            if state not in ours_states:
                continue
            diffs = {}
            for metric, split in pairs:
                a = by_state[state].get(metric)
                b = ours_states[state].get(split, {}).get(
                    "mean_assoc_sim")
                if a is not None and b is not None:
                    diffs[metric] = {
                        "official": a, "ours": b,
                        "abs_diff": round(abs(a - b), 4)}
            if diffs:
                crosscheck[state] = diffs

    evidence = None
    if ours_report.exists():
        evidence = json.loads(
            ours_report.read_text()).get("evidence_status")

    return {
        "experiment_id": (
            f"salmu_iter{paths.suffix or 'orig'}"
            "_official_salmubench_evaluator"),
        "official_evaluator": {
            "repo": OFFICIAL_REPO_URL,
            "commit": OFFICIAL_REPO_COMMIT,
            "entry_point": "evaluation/evaluation.py",
            "invocation": "SALMUBenchEvaluation verbatim; ONLY the "
                          "data source is redirected to the pinned "
                          "local snapshot (same benchmark revision); "
                          "derived splits replicated verbatim.",
            "seed": 42,
        },
        "benchmark": REPOS["benchmark_dataset"]["repo_id"],
        "benchmark_revision": bench_revision,
        "evidence_status": evidence,
        "skipped": {
            "GenKnow": "requires datacomp + a local ImageNet-1k "
                       "webdataset; not available on this machine "
                       "(official code returns -1 -> null).",
        },
        "checkpoint_sha256": hashes,
        "metric_definitions": {
            "RetFail_R@1/RetFail_MRR": "rank of the true association "
                "caption in a 2,001-caption gallery over forget",
            "AssocStr": "mean cos-sim on forget",
            "ACS": "logistic-probe accuracy separating member vs "
                   "shuffled non-member captions",
            "IdZSC": "zero-shot identity-name classification on "
                     "forget_identity",
            "CoreAssoc": "mean max-sim of '{name} {value}' / "
                         "'{value} {name}' captions on forget",
            "VisIdInt": "mean cos-sim on retain_joint (retain "
                        "utility, identities also in forget)",
            "FragSim": "mean cos-sim on the fragile subset of "
                       "retain_disjoint (retain utility)",
        },
        "states": by_state,
        "crosscheck_vs_our_evaluator": crosscheck,
        "crosscheck_note": "official pipeline uses torch.amp "
                           "autocast + DataLoader batching; ours uses "
                           "fp32 single-pass encoding — small "
                           "numerical differences are expected and "
                           "reported as abs_diff.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the official SALMUBench evaluator on GMUL "
                    "iteration checkpoints")
    parser.add_argument("--suffix", default="r5")
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--salmubench-repo",
                        default=DEFAULT_SALMUBENCH_REPO)
    parser.add_argument("--states", default=None,
                        help="Comma-separated subset of staged states")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--aggregate-only", action="store_true",
                        help="Skip evaluation; re-aggregate existing "
                             "official result JSONs")
    args = parser.parse_args()

    repo_root = _find_repo_root(Path.cwd()) or Path.cwd()
    paths = SalmuPaths(repo_root, suffix=args.suffix)
    bench = locate_repo(REPOS["benchmark_dataset"]["repo_id"],
                        "dataset")
    bench_revision = _snapshot_revision(bench)

    staging = paths.hier_dir / f"official_eval_models{paths.tag}"
    results_dir = paths.hier_dir / \
        f"official_salmubench_results{paths.tag}"
    results_dir.mkdir(parents=True, exist_ok=True)
    staged = stage_checkpoints(
        paths, staging, paths.report("salmu_unlearning_selection"))
    if args.states:
        wanted = {s.strip() for s in args.states.split(",")}
        staged = {k: v for k, v in staged.items() if k in wanted}
    log.info("Staged %d states: %s", len(staged), sorted(staged))

    if not args.aggregate_only:
        cls = make_local_evaluator_class(
            Path(args.salmubench_repo), bench)
        ev = cls(output_dir=str(results_dir), device=args.device,
                 batch_size=args.batch_size,
                 num_workers=args.num_workers, imagenet_path=None)
        ev.load_benchmark_data()
        for state, pth in staged.items():
            log.info("=== evaluating %s (%s) ===", state, pth.name)
            ev.evaluate_model(pth)

    report = aggregate(results_dir, staged, paths, bench_revision,
                       paths.report("salmu_official_splits"))
    out = paths.report("salmubench_official_eval")
    with open(out, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    log.info("Wrote official-evaluator report -> %s", out)


if __name__ == "__main__":
    main()
