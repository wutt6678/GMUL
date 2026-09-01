"""Evaluate released SALMUBench evaluation splits (Iteration 10R4a).

    python scripts/evaluate_salmu_official_splits.py --device cuda:0
    python scripts/evaluate_salmu_official_splits.py --device cuda:1 \
        --subset MF,MG          # parallel worker (sharded caching)

Scores the RELEASED official evaluation splits
(``cvc-mmu/salmubench-512-redistributed``):

* ``forget``               — sensitive associations to remove
* ``holdout_association``  — partially forgotten identities
* ``holdout_identity``     — unseen identities (collateral damage)
* ``retain_synth``         — synthetic non-sensitive utility pairs

Per split and state we report (summarization lives in
``granunlearn.salmu.official_metrics`` and is unit-tested):

* ``mean_assoc_sim``              — mean cos-sim between the released
                                    image and its association caption
* ``unit_macro_assoc_sim`` (+CI)  — each clustering unit counts once
                                    (identity, or pair where forced)
* ``leakage_rate`` (+CI)          — PAIR-level fraction of rows where
                                    the association caption outscores
                                    the released per-image generic
                                    caption; pair-level bootstrap CI
* ``identity_leakage_rate`` (+CI) — fraction of UNITS whose MACRO
                                    assoc sim exceeds their MACRO
                                    generic sim; the CI resamples
                                    exactly those unit flags (10R4a
                                    estimand/bootstrap correspondence)

``retain_synth`` clustering is FORCED pair-level for every row (the
released synthetic split defines no identity units; 10R4a).

GMUL target subsets (10R4a): every split additionally reports
``gmul_target_subset`` (rows on GMUL target-persona identities) and
``gmul_target_attr_subset`` (rows on each target persona's designated
target attribute, via the committed manifest + released caption
metadata), so the official splits can be read on exactly the
associations GMUL unlearns.

Every shard carries provenance (benchmark revision, complete-file
checkpoint SHA-256, aggregation schema; 10R4a) and is REUSED only if
ALL of schema + benchmark repo + revision + state + current checkpoint
SHA-256 still match (10R4b); aggregation applies the same check.

IMPORTANT — holdout contamination of the current chain (10R4b):
the released sensitive TRAINING dataset is the union of the forget
and holdout splits, so the current MF/MG/MN pair sets and every
unlearning group consume released holdout pairs (see
``holdout_consumption`` in the report).  The released-split results
are therefore TRANSFER DIAGNOSTICS, NOT untouched external
evaluation; a holdout-clean retrain (Iteration 10R5) is required for
the latter.

Scope note: this implements the released-split CLIP similarity
evaluation.  The paper's full protocol additionally defines RetFail
(MRR over a 2,001-caption gallery), ACS (coherence classifier),
IdZSC, CoreAssoc, GenKnow, VisIdInt and FragSim, which require the
official SALMUBench codebase (github.com/cvc-mmu/salmubench) and are
out of scope here.  AssocStr / IntraIdSim / InterIdSim ARE the
mean-cosine statistics this script computes (see
``official_metric_map``).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

# This box is shared (load >> nproc): cap per-process thread pools so
# the CPU-side image decode/transform does not oversubscribe cores.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
             "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "4")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from granunlearn.config import _find_repo_root
from granunlearn.logging_utils import setup_logger
from granunlearn.salmu.adapter import REPOS, locate_repo
from granunlearn.salmu.eval_utils import load_clip
from granunlearn.salmu.official_metrics import (
    AGGREGATION_SCHEMA,
    summarize_state,
)
from granunlearn.salmu.paths import SalmuPaths

log = setup_logger("evaluate_salmu_official_splits")

SPLITS = ("forget", "holdout_association", "holdout_identity",
          "retain_synth")
GENERIC_SPLITS = ("forget", "holdout_association", "holdout_identity")
BATCH = 128  # ViT-B-16 @ 224px: large batches amortize kernel-launch
             # overhead on shared GPUs without memory pressure


def _shard_dir(repo_root: Path, suffix: str = "") -> Path:
    return SalmuPaths(repo_root, suffix=suffix).official_shard_dir


def _snapshot_revision(path: Path) -> str | None:
    """HF snapshot revision from a cache path
    (``.../snapshots/<revision>``)."""
    parts = Path(path).parts
    if "snapshots" in parts:
        idx = parts.index("snapshots")
        if idx + 1 < len(parts):
            return parts[idx + 1]
    return None


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def state_checkpoint_sha256(state: str, repo_root: Path,
                            suffix: str = "") -> str | None:
    """Complete-file SHA-256 of the checkpoint actually loaded for
    ``state`` (shard versioning, 10R4a; suffix-aware for 10R5)."""
    paths = SalmuPaths(repo_root, suffix=suffix)
    if state == "BASE":
        clean = locate_repo(REPOS["clean_model"]["repo_id"], "model")
        ckpt = clean / "open_clip_model.safetensors"
    elif state == "COMPROMISED":
        comp = locate_repo(REPOS["compromised_model"]["repo_id"],
                           "model")
        ckpt = comp / "open_clip_pytorch_model.bin"
    elif state in ("MF", "MG", "MN"):
        ckpt = paths.ref_checkpoint(state)
    else:
        ckpt = paths.candidate_checkpoint(state)
    return _sha256_file(ckpt) if ckpt.exists() else None


def expected_provenance(state: str, repo_root: Path,
                        bench_revision: str | None,
                        suffix: str = "") -> dict:
    """The provenance a CURRENT shard for ``state`` must carry
    (10R4b): reuse is refused unless every field matches."""
    return {
        "aggregation_schema": AGGREGATION_SCHEMA,
        "benchmark_repo_id": REPOS["benchmark_dataset"]["repo_id"],
        "benchmark_revision": bench_revision,
        "state": state,
        "checkpoint_sha256": state_checkpoint_sha256(
            state, repo_root, suffix=suffix),
    }


def shard_matches_provenance(shard_data: dict, expected: dict) -> bool:
    prov = shard_data.get("_provenance", {})
    return all(prov.get(k) == v for k, v in expected.items())


def holdout_consumption_stats(repo_root: Path, bench: Path,
                            suffix: str = "") -> dict:
    """Quantify how much released HOLDOUT content the current GMUL
    training chain consumed (10R4b evidence).

    The released sensitive training dataset is the union of the
    forget + holdout splits; this counts, over the committed MF pair
    set, exact released pairs (by file_name) and role breakdowns.
    """
    import pandas as pd
    pairs = []
    mf_path = SalmuPaths(repo_root, suffix=suffix).mf_pairs_path
    if not mf_path.exists():
        return {"error": "MF.jsonl not found"}
    with open(mf_path) as f:
        for line in f:
            pairs.append(json.loads(line))
    stats: dict = {"num_mf_pairs": len(pairs)}
    for split in ("forget", "holdout_association", "holdout_identity"):
        ids: set[str] = set()
        files: set[str] = set()
        for pq in sorted((bench / "data").glob(f"{split}-*.parquet")):
            col = pd.read_parquet(pq,
                                  columns=["identity_id", "file_name"])
            ids.update(col["identity_id"].dropna())
            files.update(col["file_name"].dropna())
        on_ids = [p for p in pairs if p["identity_id"] in ids]
        exact = [p for p in pairs if p["image_file"] in files]
        stats[split] = {
            "pairs_on_split_identities": len(on_ids),
            "identities": len({p["identity_id"] for p in on_ids}),
            "exact_released_pairs": len(exact),
            "exact_released_pairs_by_role": {
                role: sum(1 for p in exact if p["role"] == role)
                for role in sorted({p["role"] for p in exact})
            },
        }
    if suffix:
        stats["note"] = (
            f"Holdout-clean iteration (suffix={suffix}): the MF pair "
            "set is restricted to the official forget split, so the "
            "exact_released_pairs counts on the holdout splits are 0 "
            "— released holdout pairs are NOT consumed by training, "
            "and released-split results are an untouched external "
            "evaluation.")
    else:
        stats["note"] = (
            "Counts over the committed MF pair set (all current states "
            "share these pairs; B3's retain group additionally trains on "
            "the non-target roles). Released holdout pairs consumed by "
            "training make released-split results transfer diagnostics, "
            "not untouched external evaluation — corrected by the "
            "holdout-clean retrain of Iteration 10R5.")
    return stats


def holdout_clean_validation(manifest: dict, consumption: dict) -> dict:
    """VALIDATE holdout cleanliness — never infer it from the suffix.

    An iteration is holdout-clean only when ALL THREE conditions
    hold (10R5a):

    1. the pair-set manifest was built with ``allowed_split ==
       "forget"``;
    2. ZERO exact released ``holdout_association`` pairs in the MF
       pair set;
    3. ZERO exact released ``holdout_identity`` pairs in the MF pair
       set.

    Pure over (manifest, consumption) so it is unit-testable.
    """
    protocol = manifest.get("protocol") or {}
    allowed = protocol.get("allowed_split")
    exact = lambda split: (consumption.get(split) or {}).get(
        "exact_released_pairs")
    checks = {
        "allowed_split_is_forget": allowed == "forget",
        "zero_exact_holdout_association_pairs":
            exact("holdout_association") == 0,
        "zero_exact_holdout_identity_pairs":
            exact("holdout_identity") == 0,
    }
    return {
        "allowed_split": allowed,
        "checks": checks,
        "validated": all(checks.values()),
        "note": "holdout cleanliness is validated from the manifest "
                "and the exact released-pair overlap — it is NEVER "
                "inferred from the iteration suffix.",
    }


def load_split(bench: Path, split: str):
    """Load one released split as rows of (file_name, text,
    identity_id) without touching the image column until encoding."""
    import glob

    from datasets import concatenate_datasets, load_dataset
    files = sorted(glob.glob(str(bench / "data" / f"{split}-*.parquet")))
    if not files:
        raise FileNotFoundError(f"No parquet shards for split {split}")
    ds = concatenate_datasets(
        [load_dataset("parquet", data_files=f, split="train")
         for f in files])
    ds = ds.remove_columns(
        [c for c in ds.column_names
         if c not in ("image", "text", "identity_id", "file_name")])
    return ds


def preprocess_split(ds, preprocess, generic_caps: dict,
                     split_name: str,
                     target_ids: set[str] | None = None,
                     target_attr_map: dict[str, str] | None = None,
                     attr_of: dict[str, str] | None = None) -> dict:
    """Decode + preprocess a released split ONCE into a CPU tensor.

    The box is shared (load >> nproc) and parquet image decode +
    CLIP preprocessing dominate wall time; doing it once per worker
    and reusing the tensors across all checkpoint states makes the
    GPU encoding the only per-state cost.  All states share the SAME
    deterministic CLIP preprocessing, so similarities stay exactly
    comparable across states.

    Also attaches the exact GMUL target-subset membership masks
    (10R4a): ``gmul_target_mask`` (row's identity is a GMUL target
    persona) and ``gmul_target_attr_mask`` (row is additionally on
    that persona's DESIGNATED target attribute, resolved through the
    released caption metadata's file_name -> data_field map).
    """
    import torch
    target_ids = target_ids or set()
    target_attr_map = target_attr_map or {}
    attr_of = attr_of or {}
    n = len(ds)
    tensors = []
    ids: list = []
    texts: list[str] = []
    file_names: list = []
    generic_texts: list[str | None] = []
    gmul_target_mask: list[bool] = []
    gmul_target_attr_mask: list[bool] = []
    for start in range(0, n, BATCH):
        rows = ds[start:start + BATCH]
        tensors.append(torch.stack([preprocess(
            im.convert("RGB")) for im in rows["image"]]))
        ids.extend(rows["identity_id"])
        texts.extend(list(rows["text"]))
        file_names.extend(list(rows["file_name"]))
        if split_name in GENERIC_SPLITS:
            generic_texts.extend(
                [generic_caps.get(f) for f in rows["file_name"]])
        else:
            generic_texts.extend([None] * len(rows["text"]))
        for iid, fname in zip(rows["identity_id"],
                              rows["file_name"]):
            is_target = iid in target_ids
            gmul_target_mask.append(is_target)
            gmul_target_attr_mask.append(
                bool(is_target
                     and attr_of.get(fname)
                     == target_attr_map.get(iid)))
        if start % (BATCH * 10) == 0:
            log.info("[preprocess/%s] %d/%d pairs", split_name,
                     start, n)
    return {
        "images": torch.cat(tensors, dim=0),
        "identity_id": ids,
        "text": texts,
        "generic_text": generic_texts,
        "gmul_target_mask": gmul_target_mask,
        "gmul_target_attr_mask": gmul_target_attr_mask,
    }


def _encode_texts(model, tokenizer, texts: list[str], device: str,
                  batch: int = 512):
    """Encode texts in batches (tokenizing 16k+ strings in one call
    is slow and memory-hungry on a shared box)."""
    import torch
    feats = []
    with torch.no_grad():
        for start in range(0, len(texts), batch):
            tok = tokenizer(list(texts[start:start + batch])).to(device)
            f = model.encode_text(tok)
            f = f / f.norm(dim=-1, keepdim=True)
            feats.append(f.cpu())
    return torch.cat(feats, dim=0)


def score_state(state: str, splits: dict, repo_root: Path,
                device: str, unlearn_root: Path,
                ref_root: Path | None = None) -> dict:
    """Score every released split for one checkpoint.

    ``splits`` holds pre-preprocessed CPU tensors (see
    ``preprocess_split``); per-state cost is GPU encoding only.
    ``ref_root`` locates the MF/MG/MN reference checkpoints for
    suffixed iterations (e.g. salmu_r5/ for 10R5).
    """
    import torch
    model, _, tokenizer = load_clip(
        state, repo_root, device, unlearn_root, ref_root)
    out: dict = {}
    with torch.no_grad():
        for split_name, data in splits.items():
            img_all = data["images"]
            n = img_all.shape[0]
            txt_feat = _encode_texts(
                model, tokenizer, data["text"], device).to(device)
            assoc_sims: list[float] = []
            generic_sims: list[float | None] = []
            gtexts = data["generic_text"]
            gidx = [i for i, g in enumerate(gtexts)
                    if g is not None]
            if gidx:
                gfeat = _encode_texts(
                    model, tokenizer,
                    [gtexts[i] for i in gidx], device).to(device)
            else:
                gfeat = None
            for start in range(0, n, BATCH):
                end = min(start + BATCH, n)
                img_feat = model.encode_image(
                    img_all[start:end].to(device))
                img_feat = img_feat / img_feat.norm(
                    dim=-1, keepdim=True)
                sims = (img_feat * txt_feat[start:end]).sum(dim=-1)
                assoc_sims.extend(sims.cpu().tolist())
                if gfeat is not None:
                    # Align generic sims with the SAME row's image
                    # feature (rows without a released generic caption
                    # stay None).
                    per_row: list[float | None] = \
                        [None] * (end - start)
                    local: list[int] = []
                    gpos: list[int] = []
                    for j, gi in enumerate(gidx):
                        if start <= gi < end:
                            local.append(gi - start)
                            gpos.append(j)
                        elif gi >= end:
                            break
                    if local:
                        gs = (img_feat[torch.tensor(
                                  local, device=device)]
                              * gfeat[torch.tensor(
                                  gpos, device=device)]).sum(dim=-1)
                        for li, val in zip(local,
                                           gs.cpu().tolist()):
                            per_row[li] = val
                    generic_sims.extend(per_row)
                else:
                    generic_sims.extend([None] * (end - start))
                if start % (BATCH * 10) == 0:
                    log.info("[%s/%s] %d/%d pairs", state, split_name,
                             start, n)
            out[split_name] = {
                "identity_id": data["identity_id"],
                "assoc_sim": assoc_sims,
                "generic_sim": generic_sims,
                "gmul_target_mask": data["gmul_target_mask"],
                "gmul_target_attr_mask":
                    data["gmul_target_attr_mask"],
            }
    del model
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    return out


def _get_preprocess():
    """CLIP ViT-B/16 image transform WITHOUT loading a checkpoint
    (preprocessing is state-independent and shared by all states)."""
    import open_clip
    from granunlearn.salmu.clip_trainer import ClipRecipe
    recipe = ClipRecipe()
    _, _, preprocess = open_clip.create_model_and_transforms(
        recipe.arch)
    return preprocess


def default_states(repo_root: Path, suffix: str = "") -> list[str]:
    """COMPROMISED (SALMUBench's unlearning start point) +
    BASE/MF/MG/MN/B0 + the selected unlearning checkpoints."""
    states = ["COMPROMISED", "BASE", "MF", "MG", "MN", "B0"]
    sel_path = SalmuPaths(repo_root, suffix=suffix).report(
        "salmu_unlearning_selection")
    if sel_path.exists():
        sel = json.loads(sel_path.read_text())
        for method in ("B1", "B2", "B3"):
            cid = sel.get("selected", {}).get(method)
            if cid and cid not in states:
                states.append(cid)
    return states


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate released SALMUBench splits")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--states", default=None,
                        help="Comma-separated states (default: "
                             "reference states + selected)")
    parser.add_argument("--subset", default=None,
                        help="Worker mode: score only these states, "
                             "write shards, skip aggregation")
    parser.add_argument("--suffix", default="",
                        help="Iteration tag (e.g. r5 -> holdout-clean "
                             "shards/reports under *_r5 paths)")
    args = parser.parse_args()

    repo_root = _find_repo_root(Path.cwd()) or Path.cwd()
    paths = SalmuPaths(repo_root, suffix=args.suffix)
    unlearn_root = paths.unlearn_root
    ref_root = paths.ref_ckpt_root
    bench = locate_repo(REPOS["benchmark_dataset"]["repo_id"], "dataset")
    generic_caps = json.loads(
        (bench / "sensitive_set_generic_captions.json").read_text())
    bench_revision = _snapshot_revision(bench)

    # Exact GMUL target subset (10R4a): target identities + per-
    # persona designated target attribute from the committed manifest,
    # and the released caption metadata's file_name -> attribute map.
    manifest = json.loads(paths.manifest_path.read_text())
    target_ids = set(manifest["partition"]["target_identity_ids"])
    target_attr_map = manifest.get("target_attr_map") or {}
    cap_meta = json.loads(
        (bench / "sensitive_set_captions_metadata.json").read_text())
    attr_of = {fname: meta.get("data_field")
               for fname, meta in cap_meta.items()}

    states = ([s.strip() for s in args.states.split(",") if s.strip()]
              if args.states else default_states(repo_root,
                                                 suffix=args.suffix))
    if args.subset:
        wanted = {s.strip() for s in args.subset.split(",") if s.strip()}
        states = [s for s in states if s in wanted]

    shard_dir = _shard_dir(repo_root, suffix=args.suffix)
    shard_dir.mkdir(parents=True, exist_ok=True)

    splits = None
    todo = []
    expected_prov: dict[str, dict] = {}
    for state in states:
        shard = shard_dir / f"{state}.json"
        expected = expected_provenance(state, repo_root, bench_revision,
                                       suffix=args.suffix)
        expected_prov[state] = expected
        if shard.exists():
            data = json.loads(shard.read_text())
            if shard_matches_provenance(data, expected):
                log.info("[%s] reusing official-split shard "
                         "(provenance matched)", state)
            else:
                log.warning("[%s] shard provenance OUTDATED "
                            "(schema/revision/state/checkpoint "
                            "mismatch) — re-scoring", state)
                todo.append(state)
        else:
            todo.append(state)
    if todo:
        # Decode + preprocess each released split ONCE; every state
        # then pays GPU encoding only.
        log.info("Loading released splits...")
        raw = {s: load_split(bench, s) for s in SPLITS}
        log.info("Preprocessing released splits (shared across all "
                 "states)...")
        splits = {s: preprocess_split(raw[s], _get_preprocess(),
                                      generic_caps, s,
                                      target_ids=target_ids,
                                      target_attr_map=target_attr_map,
                                      attr_of=attr_of)
                  for s in SPLITS}
        del raw
    for state in todo:
        shard = shard_dir / f"{state}.json"
        log.info("[%s] scoring %d released pairs", state,
                 sum(d["images"].shape[0] for d in splits.values()))
        per_split = score_state(state, splits, repo_root,
                                args.device, unlearn_root, ref_root)
        summary = summarize_state(per_split)
        summary["_provenance"] = expected_prov[state]
        tmp = shard.with_suffix(".tmp")
        with open(tmp, "w") as f:
            json.dump(summary, f, indent=2)
        tmp.replace(shard)
        log.info("[%s] shard written -> %s", state, shard)

    if args.subset:
        log.info("Worker mode done — aggregation deferred.")
        return

    # Aggregation: merge all available shards (full provenance check:
    # schema + benchmark repo + revision + state + checkpoint SHA-256;
    # 10R4b)
    by_state: dict = {}
    missing = []
    for state in default_states(repo_root, suffix=args.suffix):
        shard = shard_dir / f"{state}.json"
        if shard.exists():
            data = json.loads(shard.read_text())
            expected = expected_provenance(state, repo_root,
                                           bench_revision,
                                           suffix=args.suffix)
            if not shard_matches_provenance(data, expected):
                raise RuntimeError(
                    f"Shard {shard.name} is outdated — provenance "
                    "mismatch (schema/revision/state/checkpoint). "
                    "Delete it and re-score.")
            prov = data.pop("_provenance")
            prov.pop("aggregation_schema", None)
            by_state[state] = {"_provenance": prov, **data}
        else:
            missing.append(state)
    if missing:
        raise RuntimeError(
            f"Missing official-split shards for: {missing}. Score them "
            "first (e.g. with --subset).")
    log.info("Computing holdout-consumption statistics...")
    consumption = holdout_consumption_stats(repo_root, bench,
                                          suffix=args.suffix)
    # 10R5a: holdout cleanliness is VALIDATED (manifest allowed split
    # + zero exact holdout overlap), never inferred from the suffix.
    cleanliness = holdout_clean_validation(manifest, consumption)
    is_holdout_clean = cleanliness["validated"]
    if args.suffix == "r5":
        experiment_id = "salmu_iter10r5_official_splits"
    elif args.suffix:
        experiment_id = f"salmu_iter{args.suffix}_official_splits"
    else:
        experiment_id = "salmu_iter10r4b_official_splits"
    if is_holdout_clean:
        evidence_status = (
            "UNTOUCHED EXTERNAL EVALUATION — this iteration is "
            "VALIDATED holdout-clean (see holdout_clean_validation): "
            "targets come exclusively from the official forget split "
            "and no holdout_identity/holdout_association pair "
            "enters any training group (holdout_consumption holdout "
            "counts are 0). All GMUL-chain states (BASE, "
            "MF, MG, MN, B0-B3) were retrained holdout-clean for this "
            "iteration, so their released-holdout numbers are the "
            "protocol-compliant external evaluation. EXCEPTION: "
            "COMPROMISED is the benchmark's published starting "
            "checkpoint, fine-tuned by SALMUBench's authors on the "
            "released sensitive set (forget + holdouts); its holdout "
            "numbers are in-sample and shown only as the memorization "
            "upper bound.")
    elif args.suffix:
        evidence_status = (
            "TRANSFER DIAGNOSTIC — this suffixed iteration FAILED "
            "the holdout-clean validation (see "
            "holdout_clean_validation); its released-split numbers "
            "are NOT an untouched external evaluation.")
    else:
        evidence_status = (
            "TRANSFER DIAGNOSTIC — the current GMUL "
            "training chain consumed released holdout "
            "pairs (see holdout_consumption); these "
            "numbers are NOT an untouched external "
            "evaluation. Iteration 10R5 retrains "
            "holdout-clean for the latter.")
    report = {
        "experiment_id": experiment_id,
        "aggregation_schema": AGGREGATION_SCHEMA,
        "benchmark": REPOS["benchmark_dataset"]["repo_id"],
        "benchmark_revision": bench_revision,
        "splits": list(SPLITS),
        "evidence_status": evidence_status,
        "holdout_consumption": consumption,
        "holdout_clean_validation": cleanliness,
        "official_metric_map": {
            "AssocStr": "forget.mean_assoc_sim",
            "IntraIdSim": "holdout_association.mean_assoc_sim",
            "InterIdSim": "holdout_identity.mean_assoc_sim",
            "note": "AssocStr/IntraIdSim/InterIdSim are defined by "
                    "the paper as mean cosine similarity on these "
                    "splits — exactly what this report computes "
                    "(unit-macro variants and CIs also provided).",
        },
        "protocol": "Released (image, association-caption) pairs "
                    "encoded per checkpoint; cos-similarity point "
                    "estimates with clustering-correspondent "
                    "bootstrap CIs: every CI resamples the same unit "
                    "its point estimate averages over. Leakage = "
                    "association caption outscores the released "
                    "per-image generic caption (pair-level rate with "
                    "pair-level CI; unit-level rate with unit-level "
                    "CI).",
        "clustering": "identity-clustered on released splits with "
                      "identity units; retain_synth is FORCED "
                      "pair-level for all rows.",
        "gmul_target_subsets": "Per split, gmul_target_subset "
                               "restricts to GMUL target-persona "
                               "identities and gmul_target_attr_"
                               "subset further to each persona's "
                               "designated target attribute "
                               "(committed manifest + released "
                               "caption metadata).",
        "shard_provenance": "Each state entry carries "
                              "_provenance: benchmark revision + "
                              "complete-file checkpoint SHA-256 + "
                              "aggregation schema; reuse requires an "
                              "exact match on all fields.",
        "scope_note": "Implemented here: AssocStr, IntraIdSim, "
                      "InterIdSim (mean cos-sim on the released "
                      "splits). NOT reimplemented (require the "
                      "official SALMUBench codebase): RetFail "
                      "(2,001-caption gallery MRR), ACS (coherence "
                      "classifier), IdZSC, CoreAssoc, GenKnow, "
                      "VisIdInt, FragSim.",
        "weighting": "unit-macro for CIs; pair-level means also "
                     "reported",
        "states": by_state,
    }
    out = paths.report("salmu_official_splits")
    with open(out, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    log.info("Wrote official-split report -> %s", out)


if __name__ == "__main__":
    main()
