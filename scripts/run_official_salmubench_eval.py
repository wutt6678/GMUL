"""Run the OFFICIAL SALMUBench evaluator on GMUL iteration states.

    python scripts/run_official_salmubench_eval.py --suffix r5 \
        --device cuda:1
    python scripts/run_official_salmubench_eval.py --suffix r5 \
        --aggregate-only          # re-aggregate existing raw results

Uses the official evaluation code from github.com/cvc-mmu/salmubench
(``evaluation/evaluation.py``, pinned at commit
``8b7f439746862f09b7f0f44ae5004f297cc2dec9``) VERBATIM — only the
DATA SOURCE is redirected from the Hub repo id to our pinned local
snapshot (same content, same benchmark revision), so the run is
offline-reproducible against the committed provenance.

10R5b evaluation-integrity protections
--------------------------------------
* OFFICIAL REPO VERIFICATION: the supplied repository's actual git
  HEAD must equal the pinned commit and the worktree must be clean;
  execution aborts otherwise.
* PER-STATE RNG RESET: NumPy's global random state is captured after
  benchmark-data loading and RESTORED before every model evaluation,
  so the ACS shuffled negatives (and every other draw) are identical
  across states — identical checkpoints produce identical metrics
  (MF == B0 invariant, checked in the aggregate report).
* CHECKPOINT-SAFE REUSE: every raw result carries a sidecar
  provenance record (checkpoint SHA-256, result-file SHA-256,
  official repo commit, benchmark revision).  A result is reused ONLY
  if its sidecar matches the CURRENT checkpoint hash and evaluator
  commit; stale results are quarantined under ``stale/`` and the
  state is rescored.
* TARGET-ONLY + PAIRED CIs: the aggregate report adds the official
  metrics restricted to the GMUL target associations (identity-
  clustered CIs) and paired identity-clustered difference CIs vs the
  MF and MG reference states (see
  ``granunlearn.salmu.official_eval_analysis``).

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
* raw per-state JSONs (official format) + sidecars under
  ``data/salmu_hierarchical/official_salmubench_results[_suffix]/``
* aggregated report
  ``data/reports/salmubench_official_eval[_suffix].json``
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
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

RNG_PROTOCOL = (
    "numpy global RNG state captured after load_benchmark_data() and "
    "restored before EVERY evaluate_model() call: the ACS shuffled "
    "negatives (np.random.permutation) are therefore identical "
    "across states and identical checkpoints yield identical "
    "metrics.")
# The rng_protocol string IS the RNG schema marker: bumping
# RNG_PROTOCOL invalidates every previously scored raw result.
RESULT_BINDING_FIELDS = ("checkpoint_sha256", "official_repo_commit",
                         "benchmark_revision", "rng_protocol",
                         "result_sha256")


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


def verify_official_repo(repo_dir: Path,
                         expected_commit: str) -> dict:
    """Verify the supplied official repository's ACTUAL git HEAD and
    worktree cleanliness before execution (10R5b)."""
    repo_dir = Path(repo_dir)
    if not (repo_dir / "evaluation" / "evaluation.py").exists():
        raise SystemExit(
            f"Not an official SALMUBench checkout: {repo_dir}")

    def git(*args):
        return subprocess.run(
            ["git", "-C", str(repo_dir), *args],
            capture_output=True, text=True, check=True
        ).stdout.strip()

    head = git("rev-parse", "HEAD")
    if head != expected_commit:
        raise SystemExit(
            f"Official repo HEAD {head} != pinned commit "
            f"{expected_commit}. Update OFFICIAL_REPO_COMMIT "
            "deliberately (and re-validate) to proceed.")
    # Tracked modifications always mean the evaluation code may
    # differ from the pinned commit -> fatal.  Untracked files are
    # fatal too, EXCEPT Python bytecode caches (__pycache__), which
    # importing the official module inevitably creates and which
    # cannot alter its source.
    dirty_tracked = git("status", "--porcelain",
                        "--untracked-files=no")
    untracked = [ln for ln in
                 git("status", "--porcelain").splitlines()
                 if ln.startswith("??")
                 and "__pycache__" not in ln]
    if dirty_tracked or untracked:
        raise SystemExit(
            "Official repo worktree is DIRTY — evaluation code may "
            "differ from the pinned commit:\n"
            f"{dirty_tracked}\n{chr(10).join(untracked)}")
    return {"repo": str(repo_dir), "commit": head, "clean": True,
            "verified_at": datetime.now(timezone.utc).isoformat()}


def official_output_path(results_dir: Path, pth: Path) -> Path:
    """Replicates the official evaluate_model output naming exactly:
    slug = Path(identifier with '/' and ':' -> '_').stem, then
    "evaluation_<slug>.json".replace("__", "_").replace("_._", "_")."""
    slug = Path(str(pth).replace("/", "_").replace(":", "_")).stem
    name = (f"evaluation_{slug}.json"
            .replace("__", "_").replace("_._", "_"))
    return Path(results_dir) / name


def sidecar_path(result_path: Path) -> Path:
    return result_path.parent / (result_path.name + ".provenance.json")


def binding_mismatches(result_path: Path, *,
                        checkpoint_sha256: str,
                        official_commit: str,
                        benchmark_revision: str | None,
                        rng_protocol: str = RNG_PROTOCOL
                        ) -> list[str]:
    """Every field on which a raw result's validity depends.

    Returns the list of binding violations (empty = valid): missing
    result/sidecar, corrupt sidecar, or any mismatch among the
    CURRENT checkpoint SHA-256, the pinned evaluator commit, the
    current benchmark revision, the RNG-protocol/schema marker, and
    the result file's own SHA-256 (10R5c).
    """
    side = sidecar_path(result_path)
    if not result_path.exists():
        return ["missing_result"]
    if not side.exists():
        return ["missing_sidecar"]
    try:
        prov = json.loads(side.read_text())
    except (json.JSONDecodeError, OSError):
        return ["corrupt_sidecar"]
    bad = []
    expected = {
        "checkpoint_sha256": checkpoint_sha256,
        "official_repo_commit": official_commit,
        "benchmark_revision": benchmark_revision,
        "rng_protocol": rng_protocol,
        "result_sha256": sha256_file(result_path),
    }
    for field in RESULT_BINDING_FIELDS:
        if prov.get(field) != expected[field]:
            bad.append(field)
    return bad


def reuse_is_valid(result_path: Path, **expected) -> bool:
    """A raw result is reused ONLY if its sidecar binds ALL current
    provenance dimensions (see binding_mismatches)."""
    return not binding_mismatches(result_path, **expected)


def quarantine_stale(result_path: Path) -> None:
    """Move a stale result (+ sidecar) aside instead of deleting, so
    the evidence trail survives."""
    stale = result_path.parent / "stale"
    stale.mkdir(exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    for p in (result_path, sidecar_path(result_path)):
        if p.exists():
            p.rename(stale / f"{p.name}.{stamp}")
    log.warning("[%s] quarantined stale result (checkpoint/evaluator "
                "mismatch)", result_path.name)


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


def _invariant_groups(staged: dict[str, Path]) -> dict[str, list[str]]:
    """States sharing an identical checkpoint SHA-256 must produce
    identical metrics (MF == B0)."""
    by_sha: dict[str, list[str]] = {}
    for state, pth in staged.items():
        sha = sha256_file(pth.resolve())
        if sha:
            by_sha.setdefault(sha, []).append(state)
    return {sha: sorted(v) for sha, v in by_sha.items()
            if len(v) > 1}


def aggregate(results_dir: Path, staged: dict[str, Path],
              paths: SalmuPaths, bench: Path,
              bench_revision: str | None, ours_report: Path,
              repo_info: dict | None) -> dict:
    """Compact per-state metric table + target-only metrics + paired
    CIs + checkpoint-bound provenance."""
    from granunlearn.salmu import official_eval_analysis as analysis

    manifest = json.loads(paths.manifest_path.read_text())
    target_ids = set(manifest["partition"]["target_identity_ids"])
    target_attr_map = manifest.get("target_attr_map") or {}
    cap_meta = json.loads(
        (bench / "sensitive_set_captions_metadata.json").read_text())
    attr_of = {f: m.get("data_field") for f, m in cap_meta.items()}
    forget_ids, forget_files = analysis.split_row_order(bench,
                                                        "forget")
    tmask = analysis.target_attr_mask(
        forget_ids, forget_files, target_ids, target_attr_map,
        attr_of)
    log.info("Target-only mask: %d of %d forget rows",
             sum(tmask), len(tmask))

    by_state: dict[str, dict] = {}
    raw_by_state: dict[str, dict] = {}
    hashes = {state: sha256_file(p.resolve())
              for state, p in staged.items()}
    invalid: dict[str, list[str]] = {}
    for state, pth in staged.items():
        out = official_output_path(results_dir, pth)
        # 10R5c: aggregation NEVER accepts unvalidated evidence —
        # every raw result must bind the current checkpoint hash,
        # pinned evaluator commit, current benchmark revision, RNG
        # protocol/schema marker, and its own file hash.
        bad = binding_mismatches(
            out, checkpoint_sha256=hashes.get(state),
            official_commit=OFFICIAL_REPO_COMMIT,
            benchmark_revision=bench_revision)
        if bad:
            invalid[state] = bad
            continue
        raw = json.loads(out.read_text())
        raw_by_state[state] = raw
        entry: dict = {
            "results_file": out.name,
            "results_file_sha256": sha256_file(out),
            "checkpoint_sha256": hashes.get(state),
            "binding_validated": True,
        }
        side = sidecar_path(out)
        if side.exists():
            prov = json.loads(side.read_text())
            entry["result_provenance"] = {
                k: prov.get(k) for k in
                ("official_repo_commit", "benchmark_revision",
                 "rng_protocol", "scored_at")}
        for name, section, key, stat in METRIC_MAP:
            val = raw.get(section, {}).get(key, {}).get(stat)
            if name == "GenKnow" and val is not None and val < 0:
                val = None  # -1.0 = skipped by the official code
            entry[name] = val
        entry["target_only"] = analysis.target_only_from_raw(
            raw, forget_ids, forget_files, tmask)
        by_state[state] = entry

    if invalid:
        raise SystemExit(
            "Refusing to aggregate UNVALIDATED official results "
            "(10R5c binding gate — every raw result must bind the "
            "current checkpoint hash, pinned evaluator commit, "
            "current benchmark revision, RNG protocol, and its own "
            "file hash):\n" + "\n".join(
                f"  {state}: {bad}"
                for state, bad in sorted(invalid.items())))

    # Paired identity-clustered difference CIs vs MF and MG on the
    # target associations.
    paired: dict[str, dict] = {}
    for state, raw in raw_by_state.items():
        for ref in ("MF", "MG"):
            if ref not in raw_by_state or state == ref:
                continue
            d = analysis.paired_target_only(
                raw, raw_by_state[ref], forget_ids, tmask)
            if d:
                paired.setdefault(state, {})[f"vs_{ref}"] = d

    # Identical-checkpoint invariant (MF == B0): every metric must be
    # bit-identical across states sharing a checkpoint SHA-256.
    invariants: dict[str, Any] = {}
    for sha, group in _invariant_groups(staged).items():
        present = [s for s in group if s in by_state]
        if len(present) < 2:
            continue
        base = {k: v for k, v in by_state[present[0]].items()
                if k in {m[0] for m in METRIC_MAP}}
        ok = all(
            {k: v for k, v in by_state[s].items()
             if k in {m[0] for m in METRIC_MAP}} == base
            for s in present[1:])
        invariants[" == ".join(present)] = {
            "identical_checkpoint_sha256": sha[:16] + "...",
            "all_metrics_identical": ok}
        if not ok:
            log.error("INVARIANT VIOLATED: %s share a checkpoint but "
                      "differ in metrics", present)

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
            "pinned_commit": OFFICIAL_REPO_COMMIT,
            "verified_head": (repo_info or {}).get("commit"),
            "worktree_clean": (repo_info or {}).get("clean"),
            "verified_at": (repo_info or {}).get("verified_at"),
            "entry_point": "evaluation/evaluation.py",
            "invocation": "SALMUBenchEvaluation verbatim; ONLY the "
                          "data source is redirected to the pinned "
                          "local snapshot (same benchmark revision); "
                          "derived splits replicated verbatim.",
            "seed": 42,
            "rng_protocol": RNG_PROTOCOL,
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
        "aggregation_gate": {
            "all_results_binding_validated": True,
            "binding_fields": list(RESULT_BINDING_FIELDS),
            "note": "aggregate() (including --aggregate-only) "
                    "refuses with a non-zero exit unless EVERY raw "
                    "result's sidecar matches the current checkpoint "
                    "SHA-256, the pinned evaluator commit, the "
                    "current benchmark revision, the RNG-protocol/"
                    "schema marker, and the result file's own "
                    "SHA-256.",
        },
        "statistical_metadata": {
            "clustering_unit": "identity_id",
            "bootstrap": "percentile bootstrap over identity-level "
                         "unit means (macro); paired differences "
                         "resample the SAME identities for both "
                         "states",
            "n_bootstrap": 1000,
            "ci_level": 0.95,
            "seed": 42,
            "ci_coverage": "target-only CIs cover AssocStr, "
                           "CoreAssoc, and RetFail (MRR and R@1 via "
                           "per-row reciprocal-rank / hit means); "
                           "IdZSC and ACS target-only CIs are NOT "
                           "computed (their row spaces are filtered/"
                           "stratified subsets and are out of "
                           "scope).",
        },
        "identical_checkpoint_invariants": invariants,
        "metric_definitions": {
            "RetFail_R@1/RetFail_MRR": "rank of the true association "
                "caption in a 2,001-caption gallery over forget",
            "AssocStr": "mean cos-sim on forget",
            "ACS": "logistic-probe accuracy separating member vs "
                   "shuffled non-member captions (RNG-restored: "
                   "identical shuffle across states)",
            "IdZSC": "zero-shot identity-name classification on "
                     "forget_identity",
            "CoreAssoc": "mean max-sim of '{name} {value}' / "
                         "'{value} {name}' captions on forget",
            "VisIdInt": "mean cos-sim on retain_joint (retain "
                        "utility, identities also in forget)",
            "FragSim": "mean cos-sim on the fragile subset of "
                       "retain_disjoint (retain utility)",
            "target_only": "official metrics restricted to the GMUL "
                           "target associations (designated target "
                           "attribute rows of the forget split); "
                           "identity-clustered bootstrap CIs on "
                           "AssocStr, CoreAssoc, and RetFail "
                           "(MRR/R@1) — see statistical_metadata",
            "paired_target_only": "paired identity-clustered "
                                  "bootstrap CIs of the target-only "
                                  "difference (state - reference) "
                                  "over the SAME identities, for "
                                  "AssocStr, CoreAssoc, and RetFail "
                                  "(MRR/R@1)",
        },
        "states": by_state,
        "paired_target_only": paired,
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

    repo_info = verify_official_repo(Path(args.salmubench_repo),
                                     OFFICIAL_REPO_COMMIT)
    log.info("Official repo verified at %s (clean)",
             repo_info["commit"][:12])

    if not args.aggregate_only:
        cls = make_local_evaluator_class(
            Path(args.salmubench_repo), bench)
        ev = cls(output_dir=str(results_dir), device=args.device,
                 batch_size=args.batch_size,
                 num_workers=args.num_workers, imagenet_path=None)
        ev.load_benchmark_data()

        # 10R5b: capture the RNG state AFTER data loading and restore
        # it before EVERY model so each state sees identical random
        # draws (ACS shuffled negatives).
        import numpy as np
        rng_state = np.random.get_state()

        for state, pth in staged.items():
            result_path = official_output_path(results_dir, pth)
            sha = sha256_file(pth.resolve())
            if result_path.exists():
                if reuse_is_valid(
                        result_path, checkpoint_sha256=sha,
                        official_commit=OFFICIAL_REPO_COMMIT,
                        benchmark_revision=bench_revision):
                    log.info("[%s] reusing official result (sidecar "
                             "binds checkpoint %s, evaluator commit, "
                             "benchmark revision, RNG protocol, and "
                             "result hash)", state, sha[:12])
                    continue
                quarantine_stale(result_path)
            log.info("=== evaluating %s (%s) ===", state, pth.name)
            np.random.set_state(rng_state)
            ev.evaluate_model(pth)
            # Bind the raw result to its checkpoint + evaluator.
            side = sidecar_path(result_path)
            side.write_text(json.dumps({
                "state": state,
                "model_identifier": str(pth),
                "checkpoint_sha256": sha,
                "result_sha256": sha256_file(result_path),
                "official_repo_commit": repo_info["commit"],
                "benchmark_revision": bench_revision,
                "rng_protocol": RNG_PROTOCOL,
                "scored_at": datetime.now(timezone.utc).isoformat(),
            }, indent=2))
            log.info("[%s] sidecar written -> %s", state, side.name)

    report = aggregate(results_dir, staged, paths, bench,
                       bench_revision,
                       paths.report("salmu_official_splits"),
                       repo_info)
    out = paths.report("salmubench_official_eval")
    with open(out, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    log.info("Wrote official-evaluator report -> %s", out)


if __name__ == "__main__":
    main()
