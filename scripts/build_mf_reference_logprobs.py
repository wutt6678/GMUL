"""Cache MF's preserved distributions, then calibrate the anchor strength.

    python scripts/build_mf_reference_logprobs.py --device cuda:0
    python scripts/build_mf_reference_logprobs.py --device cuda:0 \
        --calibrate --calibration-report data/reports/mllmu_iter12_anchor_calibration.json
    python scripts/build_mf_reference_logprobs.py --check-only

Two jobs, one script, because they must share the position extraction.

BUILD writes ``data/mllmu_hier_pilot100/mf_reference_logprobs/``: the exact
full-vocabulary log-probabilities of the frozen MF adapter at every supervised
position of the 183 fit-half ``retain`` examples, plus a sidecar recording the
sha256 of the MF adapter, the base-model revision and the encoding contract.
MF does not change during Stage 2, so this is a constant computed once; no
second model is resident at training time.

CALIBRATE measures, for every adapter Stage 1 already produced, the KL that
anchor would have seen at the end of that training.  It exists so the strength
grid is DERIVED rather than guessed.  It reads no evaluation query, no
prediction parquet and no retention number -- only adapters and the training
summaries already committed -- so it cannot be fitted to the outcome it is
meant to govern.

THE GUARD THAT MAKES THE POSITIONS CHECKABLE
--------------------------------------------
For every example, the NLL recomputed from the extracted log-probabilities must
equal the loss the model itself returned.  A causal LM predicts ``labels[i]``
from ``logits[i-1]``; indexing at ``i`` instead would compare MF's distribution
at one position against the candidate's at its neighbour and still yield a
plausible small KL.  The largest observed disagreement across all rows is
recorded in the sidecar and in the calibration report, so the agreement is a
measured quantity rather than an assumption.

Needs a GPU for BUILD and CALIBRATE.  ``--check-only`` is CPU-only and verifies
a cache already on disk against its sidecar and against the live MF adapter.
Nothing here reads the sealed confirmation split and nothing edits a frozen
path: ``unlearning_datasets``, ``reference_trainer`` and
``prediction_provenance`` are imported, never modified.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from granunlearn.config import _find_repo_root
from granunlearn.evaluation.prediction_provenance import base_model_revision
from granunlearn.logging_utils import setup_logger
from granunlearn.training.candidate_grid import (
    ITER12_LAM,
    ITER12_REFERENCE_ROW,
    dataset_dir_for_tag,
    groups_subdir_for_tag,
    iter12_grid,
)
from granunlearn.training.preservation_anchor import (
    CACHE_DIRNAME,
    CACHE_MAX_IMAGE_PIXELS,
    CACHE_MAX_LENGTH,
    CACHE_VERSION,
    NLL_AGREEMENT_TOLERANCE,
    PRESERVE_GROUP,
    ReferenceCache,
    adapter_digest,
    anchor_loss,
    assert_nll_matches_model,
    mean_nll,
    sha256_file,
    supervised_logprobs,
)
from granunlearn.training.reference_trainer import ReferenceRecipe, _encode_example
from granunlearn.training.state_datasets import load_state_examples

log = setup_logger("build_mf_reference_logprobs")

TAG = "iter12"
MF_ADAPTERS = "data/checkpoints/mllmu_pilot100/MF/adapters"
STAGE1_CKPT_ROOT = "data/checkpoints/mllmu_iter12_unlearn"
DEFAULT_CALIBRATION_REPORT = \
    "data/reports/mllmu_iter12_anchor_calibration.json"


def cache_dir_for(repo_root: Path) -> Path:
    """The cache lives beside the dataset it preserves knowledge for.

    Built from ``dataset_dir_for_tag`` rather than a literal, because that
    helper already carries the ``data/`` prefix: joining another ``data``
    component here would produce the double prefix that killed the Stage-1b
    control in two seconds.
    """
    return repo_root / dataset_dir_for_tag(TAG) / CACHE_DIRNAME


def preserve_group_path(repo_root: Path) -> Path:
    return repo_root / dataset_dir_for_tag(TAG) / groups_subdir_for_tag(TAG) \
        / f"{PRESERVE_GROUP}.jsonl"


def _load_model(adapter_dir: Path, device: str, model_id: str,
                recipe: ReferenceRecipe):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForImageTextToText

    model = AutoModelForImageTextToText.from_pretrained(
        model_id, device_map={"": device},
        torch_dtype=torch.bfloat16 if recipe.bf16 else torch.float32)
    model = PeftModel.from_pretrained(model, str(adapter_dir))
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def _forward_rows(model, processor, examples, device: str,
                  recipe: ReferenceRecipe, label: str):
    """``(logp_list, targets_list, spans, max_nll_disagreement)``."""
    import torch

    logp_rows: list[Any] = []
    target_rows: list[Any] = []
    spans: list[dict[str, Any]] = []
    cursor = 0
    worst = 0.0
    for ex in examples:
        enc = _encode_example(ex, processor, recipe.max_length,
                              recipe.max_image_pixels)
        enc = {k: v.to(device) for k, v in enc.items() if torch.is_tensor(v)}
        with torch.no_grad():
            out = model(**enc)
        logp, targets = supervised_logprobs(out.logits, enc["labels"])
        #: The guard.  Without it a wrong shift is invisible: the numbers stay
        #: small and plausible, and every KL in the study is measured on the
        #: wrong row.
        worst = max(worst, assert_nll_matches_model(
            logp, targets, out.loss.item(), f"{label}/{ex.example_id}"))
        logp_rows.append(logp.cpu())
        target_rows.append(targets.cpu())
        spans.append({
            "example_id": ex.example_id,
            "association_id": ex.association_id,
            "entity_id": ex.entity_id,
            "start": cursor,
            "stop": cursor + len(targets),
            "targets": [int(t) for t in targets.tolist()],
        })
        cursor += len(targets)
    return logp_rows, target_rows, spans, worst


def build_cache(repo_root: Path, device: str, model_id: str,
                recipe: ReferenceRecipe, force: bool = False) -> dict[str, Any]:
    """Compute and write the exact reference cache.  Returns the sidecar."""
    import torch

    group_path = preserve_group_path(repo_root)
    if not group_path.exists():
        raise SystemExit(f"REFUSING: the preservation group {group_path} does "
                         f"not exist. Stage 2 anchors on the FIT half of the "
                         f"retained entities; there is nothing to anchor "
                         f"without it.")
    examples = load_state_examples(group_path, repo_root=repo_root)
    out_dir = cache_dir_for(repo_root)
    if (out_dir / "sidecar.json").exists() and not force:
        existing = ReferenceCache.load(out_dir)
        existing.assert_built_from(repo_root / MF_ADAPTERS,
                                   base_model_revision(model_id))
        log.info("cache already present and verified against the live MF "
                 "adapter (%d rows); pass --force to rebuild",
                 len(existing.rows))
        return existing.sidecar

    adapter_dir = repo_root / MF_ADAPTERS
    if not adapter_dir.exists():
        raise SystemExit(f"REFUSING: no MF adapter at {adapter_dir}. The "
                         f"anchor preserves MF; without those bytes there is "
                         f"no reference to preserve.")
    revision = base_model_revision(model_id)

    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(model_id)
    if processor.tokenizer.padding_side != "left":
        processor.tokenizer.padding_side = "left"

    log.info("loading MF from %s on %s", adapter_dir, device)
    model = _load_model(adapter_dir, device, model_id, recipe)
    logp_rows, target_rows, spans, worst = _forward_rows(
        model, processor, examples, device, recipe, "MF")
    del model
    torch.cuda.empty_cache()

    logp = torch.cat(logp_rows, dim=0).to(torch.float32).contiguous()
    targets = torch.cat(target_rows, dim=0).to(torch.int64).contiguous()
    vocab = int(logp.shape[1])
    sidecar: dict[str, Any] = {
        "cache_version": CACHE_VERSION,
        "purpose": (
            "Exact full-vocabulary log-probabilities of the frozen MF adapter "
            "at every supervised position of the fit-half retain group. The "
            "Stage-2 anchor penalises KL(p_MF || p_theta) on these rows, so "
            "this file IS the reference the regularizer preserves."),
        "model_id": model_id,
        "base_model_revision": revision,
        "mf_adapter_dir": MF_ADAPTERS,
        "mf_adapter_sha256": adapter_digest(adapter_dir),
        "preserve_group_path": str(group_path.relative_to(repo_root)),
        "preserve_group_sha256": sha256_file(group_path),
        "max_length": recipe.max_length,
        "max_image_pixels": recipe.max_image_pixels,
        "dtype": "float32",
        "vocab_size": vocab,
        "num_examples": len(examples),
        "num_positions": int(logp.shape[0]),
        "kl_direction": "KL(p_MF || p_theta), exact over the full vocabulary",
        "nll_agreement_max": worst,
        "nll_agreement_tolerance": NLL_AGREEMENT_TOLERANCE,
        "probe_half_excluded": True,
        "why_the_probe_half_is_excluded": (
            "A preservation term fitted on probe-half prompts would rehearse "
            "the quantity the retention floor measures, which is the leak the "
            "fit/probe split exists to prevent."),
        "target_groups_excluded": ["fine_target", "target_level"],
        "why_target_groups_are_excluded": (
            "Anchoring MF's distribution on fine_target would oppose the "
            "suppression term on the very prompts the method exists to change, "
            "and on target_level it would oppose the rewrite."),
    }
    if recipe.max_length != CACHE_MAX_LENGTH or \
            recipe.max_image_pixels != CACHE_MAX_IMAGE_PIXELS:
        raise SystemExit(
            f"REFUSING: the recipe encodes at max_length="
            f"{recipe.max_length}, max_image_pixels="
            f"{recipe.max_image_pixels} but this cache module declares "
            f"{CACHE_MAX_LENGTH}/{CACHE_MAX_IMAGE_PIXELS}. The supervised "
            f"positions depend on both, so the declared contract would be "
            f"false.")
    cache = ReferenceCache.from_parts(sidecar, logp, targets, spans)
    written = cache.save(out_dir)
    log.info("cached %d positions over %d examples | vocab %d | %.2f GB | "
             "max NLL disagreement %.2e",
             written["num_positions"], written["num_examples"], vocab,
             (out_dir / "logprobs.pt").stat().st_size / 1e9, worst)
    return written


# --------------------------------------------------------------------------
# calibration
# --------------------------------------------------------------------------

def calibration_adapters(repo_root: Path) -> dict[str, Path]:
    """Every Stage-1 adapter that exists, plus MF and MG.

    Taken from the FROZEN grid rather than from a directory listing, so a
    stray directory cannot enter the calibration and a missing one is visible
    as absent rather than silently skipped.

    MG is included because it is the state the selection criterion targets.
    Its NLL on ``fine_target`` is the level a correctly transformed model sits
    at without ever having learned the fine facts, and its KL against MF on the
    preservation set is how far the ORACLE drifts from MF on the very prompts
    an anchor would pin.  Both are design measurements; neither is a retention
    number.
    """
    root = repo_root / STAGE1_CKPT_ROOT
    reference = repo_root / "data" / "checkpoints" / "mllmu_pilot100"
    found: dict[str, Path] = {"MF": reference / "MF" / "adapters"}
    mg = reference / "MG" / "adapters"
    if mg.exists():
        found["MG"] = mg
    for spec in iter12_grid():
        adir = root / spec.candidate_id / "adapters"
        if adir.exists():
            found[spec.candidate_id] = adir
    return found


def group_paths(repo_root: Path) -> dict[str, Path]:
    """All three knowledge groups, for the per-group NLL measurement."""
    gdir = repo_root / dataset_dir_for_tag(TAG) / groups_subdir_for_tag(TAG)
    return {name: gdir / f"{name}.jsonl"
            for name in ("fine_target", "target_level", PRESERVE_GROUP)}


def calibrate(repo_root: Path, cache: ReferenceCache, device: str,
              model_id: str, recipe: ReferenceRecipe,
              out_report: Path) -> dict[str, Any]:
    """Measure the KL each Stage-1 adapter ended training at.

    No retention number, prediction parquet or evaluation query is read.  The
    inputs are adapters and the training summaries Stage 1 already committed,
    so the strength grid this calibrates cannot have been fitted to the
    outcome it governs.
    """
    import torch
    from transformers import AutoProcessor

    group_path = repo_root / cache.sidecar["preserve_group_path"]
    examples = load_state_examples(group_path, repo_root=repo_root)
    #: Only the two TARGET groups are looped separately.  The preserved group's
    #: NLL falls out of the KL loop below, where the extraction is already
    #: checked against the model's own loss, so measuring it twice would cost
    #: 183 forwards per adapter to recompute a number already in hand.
    target_groups = {n: load_state_examples(p, repo_root=repo_root)
                     for n, p in group_paths(repo_root).items()
                     if n != PRESERVE_GROUP}
    processor = AutoProcessor.from_pretrained(model_id)
    if processor.tokenizer.padding_side != "left":
        processor.tokenizer.padding_side = "left"

    adapters = calibration_adapters(repo_root)
    log.info("calibrating against %d adapter(s): %s", len(adapters),
             ", ".join(sorted(adapters)))
    rows: dict[str, Any] = {}
    worst = cache.sidecar["nll_agreement_max"]
    for name in sorted(adapters):
        adir = adapters[name]
        model = _load_model(adir, device, model_id, recipe)

        #: ``out.loss`` is the model's own mean NLL, so a per-group NLL needs
        #: no position extraction and is directly comparable with the
        #: ``avg_loss_*`` keys the Stage-1 summaries already committed.  Each
        #: example is weighted by its supervised-token count, because a group
        #: mean over unweighted per-example means would let a two-token
        #: completion count as much as a nine-token one.
        nll: dict[str, float | None] = {}
        for gname, gexamples in target_groups.items():
            tot, n = 0.0, 0
            for ex in gexamples:
                enc = _encode_example(ex, processor, recipe.max_length,
                                      recipe.max_image_pixels)
                enc = {k: v.to(device) for k, v in enc.items()
                       if torch.is_tensor(v)}
                with torch.no_grad():
                    out = model(**enc)
                n_sup = int((enc["labels"] != -100).sum().item())
                tot += float(out.loss.item()) * n_sup
                n += n_sup
            nll[gname] = round(tot / n, 6) if n else None

        total_kl, total_nll, n_pos = 0.0, 0.0, 0
        for ex in examples:
            enc = _encode_example(ex, processor, recipe.max_length,
                                  recipe.max_image_pixels)
            enc = {k: v.to(device) for k, v in enc.items()
                   if torch.is_tensor(v)}
            with torch.no_grad():
                out = model(**enc)
            logp, targets = supervised_logprobs(out.logits, enc["labels"])
            worst = max(worst, assert_nll_matches_model(
                logp, targets, out.loss.item(), f"{name}/{ex.example_id}"))
            ref_logp, ref_targets = cache.slice_for(ex.example_id)
            #: Alignment, checked per adapter rather than assumed once: a
            #: candidate whose encoding moved would otherwise be compared
            #: against the wrong reference rows.
            cache.assert_aligned(ex.example_id, targets)
            if not torch.equal(ref_targets.cpu(), targets.cpu()):
                raise AssertionError(f"{name}/{ex.example_id}: target mismatch")
            kl = float(anchor_loss(ref_logp.to(device), logp).item())
            total_kl += kl * len(targets)
            total_nll += mean_nll(logp, targets) * len(targets)
            n_pos += len(targets)
        del model
        torch.cuda.empty_cache()
        nll[PRESERVE_GROUP] = round(total_nll / n_pos, 6)
        summary_path = adir.parent / "training_summary.json"
        summary = json.loads(summary_path.read_text()) \
            if summary_path.exists() else {}
        last_epoch = (summary.get("epochs") or [{}])[-1]
        rows[name] = {
            "adapter_dir": str(adir.relative_to(repo_root)),
            "adapter_sha256": adapter_digest(adir),
            "mean_kl_vs_mf": round(total_kl / n_pos, 6),
            "mean_nll": round(total_nll / n_pos, 6),
            "num_positions": n_pos,
            "nll_by_group": nll,
            "committed_objective": {
                k: last_epoch.get(k) for k in
                ("avg_loss_fine_target", "avg_loss_retain",
                 "avg_loss_target_level", "avg_objective")
            },
        }
        log.info("[%s] KL vs MF %.6f | nll fine_target %.4f target_level "
                 "%.4f retain %.4f", name, rows[name]["mean_kl_vs_mf"],
                 nll["fine_target"], nll["target_level"],
                 nll[PRESERVE_GROUP])

    report = derive_grid(rows, cache, worst)
    report["measurement"] = {
        "adapters": rows,
        "reads_no_retention_number": True,
        "reads_no_prediction_parquet": True,
        "reads_no_evaluation_query": True,
        "inputs": "Stage-1 adapters and the training summaries they committed",
        "nll_agreement_max": worst,
        "device": device,
        "model_id": model_id,
    }
    out_report.parent.mkdir(parents=True, exist_ok=True)
    with open(out_report, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    log.info("calibration -> %s", out_report)
    return report


def derive_grid(rows: dict[str, Any], cache: ReferenceCache,
                worst: float) -> dict[str, Any]:
    """Derive the strength grid from the incumbent's own operating point.

    The rule is stated before the numbers are looked at: beta-star is the
    strength at which the anchor term equals the suppression term's magnitude
    at the end of the INCUMBENT reference row's training, so the sweep spans
    the incumbent's operating point by a factor of ten in each direction
    instead of bracketing a guess.  ``B4_w1.0_lam0.5_lr2e-05_ep5`` is the
    incumbent: the Stage-1 freeze names it ``reference_row`` and the Stage-2
    gate compares the winner against it.

    Every input is a committed training summary or a measured KL.  No retention
    number appears, so the grid cannot be fitted to the floor it will be judged
    on.
    """
    ref = rows.get(ITER12_REFERENCE_ROW)
    if ref is None:
        raise SystemExit(
            f"REFUSING to derive the grid: the incumbent reference row "
            f"{ITER12_REFERENCE_ROW!r} was not measured, so there is no "
            f"operating point to span. Measured: {sorted(rows)}.")
    fine_nll = ref["committed_objective"]["avg_loss_fine_target"]
    if fine_nll is None:
        raise SystemExit(
            f"REFUSING: {ITER12_REFERENCE_ROW}'s training summary records no "
            f"avg_loss_fine_target, so the suppression term's magnitude is "
            f"unknown and beta cannot be derived from it.")
    kl = ref["mean_kl_vs_mf"]
    if kl <= 0:
        raise SystemExit(
            f"REFUSING: the incumbent's mean KL against MF is {kl}, so no "
            f"positive beta can be derived from it. A zero KL would mean the "
            f"incumbent never drifted from MF, which its own training summary "
            f"contradicts.")
    suppression = ITER12_LAM * fine_nll
    beta_star = suppression / kl
    #: Four values, not five: the low end is dropped deliberately.  At one
    #: tenth of the operating point the anchor term is a tenth the size of the
    #: term it is meant to bound, and Stage 1 already swept the analogous weak
    #: end of the replay knob (weight 0.5) and found it the WORST row on every
    #: route.  Spending a training on "a very weak regularizer does very
    #: little" is the least informative row available.
    multipliers = (1 / 3, 1.0, 3.0, 10.0)
    grid = sorted({round(beta_star * m, 4) for m in multipliers})
    return {
        "grid_rule": {
            "beta_star_is": (
                "the strength at which beta * KL(p_MF || p_candidate) equals "
                "lambda * avg_loss_fine_target at the END of the incumbent "
                "reference row's training, i.e. the anchor term is exactly as "
                "large as the term it is meant to bound"),
            "incumbent_reference_row": ITER12_REFERENCE_ROW,
            "lambda_fine_suppression": ITER12_LAM,
            "incumbent_avg_loss_fine_target": fine_nll,
            "incumbent_suppression_magnitude": round(suppression, 6),
            "incumbent_mean_kl_vs_mf": kl,
            "beta_star": round(beta_star, 6),
            "multipliers": list(multipliers),
            "why_these_multipliers": (
                "1.5 orders of magnitude centred on the incumbent's operating "
                "point. A sweep narrower than that could sit entirely on one "
                "side of the transition between 'the anchor does nothing' and "
                "'the anchor prevents the transformation', and neither end "
                "would be measured."),
            "beta_grid": grid,
            "derived_from": (
                "committed training summaries and measured KL only; no "
                "retention number, prediction parquet or evaluation query"),
            "nll_agreement_max": worst,
        },
        "cache_provenance": {
            "mf_adapter_sha256": cache.sidecar["mf_adapter_sha256"],
            "base_model_revision": cache.sidecar["base_model_revision"],
            "num_examples": cache.sidecar["num_examples"],
            "num_positions": cache.sidecar["num_positions"],
            "vocab_size": cache.sidecar["vocab_size"],
            "preserve_group_sha256": cache.sidecar["preserve_group_sha256"],
        },
    }


def check_only(repo_root: Path, model_id: str) -> int:
    """Verify a cache on disk.  CPU-only; loads no model."""
    out_dir = cache_dir_for(repo_root)
    cache = ReferenceCache.load(out_dir)
    cache.assert_built_from(repo_root / MF_ADAPTERS,
                            base_model_revision(model_id))
    group_path = repo_root / cache.sidecar["preserve_group_path"]
    live = sha256_file(group_path)
    if live != cache.sidecar["preserve_group_sha256"]:
        raise SystemExit(
            f"REFUSING: the preservation group hashes to {live} but the cache "
            f"records {cache.sidecar['preserve_group_sha256']}; the examples "
            f"the reference was computed on are not the ones on disk.")
    print(f"OK: {len(cache.rows)} examples, "
          f"{cache.sidecar['num_positions']} positions, vocab "
          f"{cache.sidecar['vocab_size']}, MF adapter "
          f"{cache.sidecar['mf_adapter_sha256'][:16]}, base revision "
          f"{cache.sidecar['base_model_revision']}, max NLL disagreement "
          f"{cache.sidecar['nll_agreement_max']:.2e}")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cache MF's preserved distributions and calibrate the "
                    "Stage-2 anchor strength")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--model-id", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--calibrate", action="store_true",
                        help="Also measure the KL every Stage-1 adapter ended "
                             "training at, and derive the strength grid")
    parser.add_argument("--calibration-report", default=None)
    parser.add_argument("--force", action="store_true",
                        help="Rebuild an existing cache instead of verifying it")
    parser.add_argument("--check-only", action="store_true",
                        help="Verify the cache on disk and exit (CPU-only)")
    parser.add_argument("--repo-root", default=None)
    args = parser.parse_args()

    repo_root = Path(args.repo_root) if args.repo_root \
        else (_find_repo_root(Path.cwd()) or Path.cwd())

    if args.check_only:
        raise SystemExit(check_only(repo_root, args.model_id))

    recipe = ReferenceRecipe()
    sidecar = build_cache(repo_root, args.device, args.model_id, recipe,
                          force=args.force)
    if not args.calibrate:
        print(json.dumps({"cache_dir": str(cache_dir_for(repo_root)),
                          "num_positions": sidecar["num_positions"],
                          "nll_agreement_max": sidecar["nll_agreement_max"]},
                         indent=2))
        return
    cache = ReferenceCache.load(cache_dir_for(repo_root))
    out = Path(args.calibration_report or DEFAULT_CALIBRATION_REPORT)
    calibrate(repo_root, cache, args.device, args.model_id, recipe, out)


if __name__ == "__main__":
    main()
