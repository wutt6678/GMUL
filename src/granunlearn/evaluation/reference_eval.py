"""Reference-state evaluation (Iteration 7).

Loads the base Qwen3.5-9B (optionally + a state's LoRA adapter), runs the
committed smoke queries (all families, all splits — the paraphrase split
is an EVALUATION split; nothing here trains), scores outputs against the
canonical hierarchy, and applies the hard gate:

    MF != MG != MN, behaviorally, on the committed smoke set.

The gate precedes any unlearning method: only once the reference states
separate is MF -> MU worth running.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from granunlearn.evaluation.prediction_provenance import (
    PredictionFingerprint,
    dataset_version,
    validate_prediction_coverage,
    verify_sidecar,
    write_sidecar,
)
from granunlearn.evaluation.scoring import (
    compute_metrics,
    score_query,
    separation_gate,
)
from granunlearn.evaluation.hierarchy_metrics import (
    compute_hierarchy_metrics,
    export_failure_cases,
)
from granunlearn.imaging import image_size_kwargs
from granunlearn.logging_utils import setup_logger
from granunlearn.schema import AssociationRecord, PredictionRecord, QueryRecord

log = setup_logger("reference_eval")

#: Generation defaults, single-sourced so a prediction fingerprint can
#: record exactly what the generator will do.  Changing either changes the
#: decoded bytes, which is why both are part of the reuse contract in
#: :mod:`granunlearn.evaluation.prediction_provenance`.
DEFAULT_MAX_NEW_TOKENS = 96
DEFAULT_MAX_LENGTH = 1536
DEFAULT_MAX_IMAGE_PIXELS = 384 * 384


def load_queries_parquet(path: str | Path) -> list[QueryRecord]:
    """NaN-safe parquet -> canonical QueryRecord loader (parquet stores
    None int columns as float NaN)."""
    import pandas as pd
    df = pd.read_parquet(path)
    df = df.astype(object).where(pd.notnull(df), None)
    return [QueryRecord.model_validate(r)
            for r in df.to_dict(orient="records")]


def load_predictions_parquet(path: str | Path) -> list[PredictionRecord]:
    """NaN-safe parquet -> canonical PredictionRecord loader (rescore
    path: regenerate metrics from persisted raw outputs without
    re-running the model)."""
    import pandas as pd
    df = pd.read_parquet(path)
    df = df.astype(object).where(pd.notnull(df), None)
    return [PredictionRecord.model_validate(r)
            for r in df.to_dict(orient="records")]


def load_associations_parquet(path: str | Path) -> list[AssociationRecord]:
    import pandas as pd
    df = pd.read_parquet(path)
    df = df.astype(object).where(pd.notnull(df), None)
    return [AssociationRecord.model_validate(r)
            for r in df.to_dict(orient="records")]


class ReferenceStateGenerator:
    """Greedy generator for one checkpoint (BASE = no adapter)."""

    def __init__(self, model_id: str, device: str,
                 adapter_dir: str | Path | None = None,
                 max_image_pixels: int = DEFAULT_MAX_IMAGE_PIXELS):
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self.device = device
        self.max_image_pixels = max_image_pixels
        self.processor = AutoProcessor.from_pretrained(model_id)
        if self.processor.tokenizer.padding_side != "left":
            self.processor.tokenizer.padding_side = "left"
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_id, device_map={"": device}, torch_dtype=torch.bfloat16,
        )
        if adapter_dir is not None:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(
                self.model, str(adapter_dir))
            log.info("Loaded LoRA adapter from %s", adapter_dir)
        self.model.eval()

    def _render_prompt(self, q: QueryRecord,
                       assoc: AssociationRecord,
                       repo_root: str | Path):
        """Chat-template prompt + optional image for one query."""
        from PIL import Image
        content: list[dict[str, Any]] = []
        image = None
        if q.image_ids:
            img_ref = next(
                (i for i in assoc.images
                 if i.image_id in q.image_ids), None)
            if img_ref is not None:
                p = Path(img_ref.path)
                if not p.is_absolute():
                    p = Path(repo_root) / p
                if p.exists():
                    image = Image.open(p).convert("RGB")
                    content.append({"type": "image"})
        content.append({"type": "text", "text": q.prompt})
        try:
            text = self.processor.apply_chat_template(
                [{"role": "user", "content": content}],
                tokenize=False, add_generation_prompt=True,
                enable_thinking=False)
        except TypeError:
            text = self.processor.apply_chat_template(
                [{"role": "user", "content": content}],
                tokenize=False, add_generation_prompt=True)
        return text, image

    def _generate_batch(self, texts: list[str], images_per_sample,
                        max_new_tokens: int) -> list[str]:
        import torch
        kwargs: dict[str, Any] = {
            "text": texts, "return_tensors": "pt", "padding": True,
            "truncation": True, "max_length": DEFAULT_MAX_LENGTH,
        }
        if images_per_sample is not None:
            kwargs["images"] = images_per_sample
            kwargs.update(image_size_kwargs(self.max_image_pixels))
        inputs = self.processor(**kwargs).to(self.device)
        with torch.no_grad():
            gen = self.model.generate(
                **inputs, max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.processor.tokenizer.pad_token_id)
        trimmed = gen[:, inputs["input_ids"].shape[1]:]
        return self.processor.batch_decode(
            trimmed, skip_special_tokens=True)

    def generate_for_queries(
        self,
        queries: list[QueryRecord],
        associations: dict[str, AssociationRecord],
        repo_root: str | Path,
        batch_size: int = 8,
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
        image_batch_size: int = 1,
    ) -> list[str]:
        """Greedy completions; multimodal queries include their image.

        Text-only queries are batched.  Image queries run in groups of
        ``image_batch_size`` (default 1 = the Iteration 7/9 one-at-a-time
        path): the processor rejects ``None`` entries in ``images``, so a
        batch may never MIX image and text samples — but an all-image
        batch is legal, and the pilot-100 set has 2,241 image probes
        where one-at-a-time generation dominates the wall clock.  A
        query whose image file is missing degrades to a single-sample
        text-only call.  Formatting is identical across checkpoints, and
        every checkpoint is generated with the SAME batch sizes, so
        cross-state comparability is preserved.
        """
        outputs: list[str] = []
        i = 0
        total = len(queries)
        # A full generation pass is the longest silent phase in the
        # pipeline (tens of minutes on a shared GPU), so emit a progress
        # mark with a throughput-based ETA.  Logging only: it must never
        # influence how the queries are grouped into batches, because
        # cross-state comparability depends on an identical batch layout.
        step = max(256, total // 20)
        next_mark = step
        started = time.time()
        while i < len(queries):
            if len(outputs) >= next_mark:
                elapsed = time.time() - started
                rate = len(outputs) / elapsed if elapsed > 0 else 0.0
                eta = ((total - len(outputs)) / rate) if rate > 0 else 0.0
                log.info("generation progress: %d/%d (%.1f%%) in %.0fs — "
                         "%.2f q/s, ETA %.0fs", len(outputs), total,
                         100.0 * len(outputs) / total, elapsed, rate, eta)
                next_mark += step
            if queries[i].image_ids:
                # maximal image run, capped at image_batch_size
                j = i
                while (j < len(queries) and queries[j].image_ids
                       and j - i < max(1, image_batch_size)):
                    j += 1
                texts, images_per_sample, has_image = [], [], False
                for k in range(i, j):
                    q = queries[k]
                    text, image = self._render_prompt(
                        q, associations[q.association_id], repo_root)
                    texts.append(text)
                    images_per_sample.append(
                        [image] if image is not None else None)
                    has_image = has_image or image is not None
                if has_image and any(im is None
                                      for im in images_per_sample):
                    # a missing image file would poison the whole batch:
                    # fall back to per-sample calls for this run
                    for k in range(i, j):
                        q = queries[k]
                        text, image = self._render_prompt(
                            q, associations[q.association_id], repo_root)
                        outputs.extend(self._generate_batch(
                            [text],
                            [[image]] if image is not None else None,
                            max_new_tokens))
                else:
                    outputs.extend(self._generate_batch(
                        texts,
                        images_per_sample if has_image else None,
                        max_new_tokens))
                i = j
                continue
            # maximal text-only run, capped at batch_size
            j = i
            while (j < len(queries) and not queries[j].image_ids
                   and j - i < batch_size):
                j += 1
            texts = [self._render_prompt(
                        queries[k], associations[queries[k].association_id],
                        repo_root)[0]
                     for k in range(i, j)]
            outputs.extend(self._generate_batch(
                texts, None, max_new_tokens))
            i = j
        return outputs

    def unload(self) -> None:
        import torch
        del self.model
        torch.cuda.empty_cache()


def evaluate_state(
    checkpoint_id: str,
    queries: list[QueryRecord],
    associations: list[AssociationRecord],
    repo_root: str | Path,
    model_id: str,
    device: str,
    adapter_dir: str | Path | None,
    experiment_id: str = "mllmu_smoke_iter7",
    batch_size: int = 2,
    image_batch_size: int = 1,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
) -> tuple[list[PredictionRecord], dict[str, Any]]:
    """Generate + score one checkpoint over the full query set."""
    by_assoc = {a.association_id: a for a in associations}
    generator = ReferenceStateGenerator(
        model_id, device, adapter_dir=adapter_dir)
    t0 = time.time()
    raw_outputs = generator.generate_for_queries(
        queries, by_assoc, repo_root, batch_size=batch_size,
        image_batch_size=image_batch_size, max_new_tokens=max_new_tokens)
    generator.unload()

    predictions = [
        score_query(q, by_assoc[q.association_id], raw,
                    experiment_id=experiment_id,
                    checkpoint_id=checkpoint_id)
        for q, raw in zip(queries, raw_outputs)
    ]
    return predictions, {"generation_seconds": round(time.time() - t0, 1)}


def metrics_for_predictions(
    predictions: list[PredictionRecord],
    queries: list[QueryRecord],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Pooled metrics + per-split metrics (test-paraphrase SEPARATE from
    pooled train/val/test, Iteration 7 review)."""
    pooled = compute_metrics(predictions, queries)
    by_split = {s: compute_metrics(predictions, queries, split=s)
                for s in ("train", "val", "test")}
    return pooled, by_split


def run_reference_evaluation(
    smoke_dir: str | Path,
    checkpoints_dir: str | Path,
    report_path: str | Path,
    model_id: str = "Qwen/Qwen3.5-9B",
    device: str = "cuda:0",
    states: list[str] | None = None,
    predictions_dir: str | Path | None = None,
    batch_size: int = 2,
    rescore: bool = False,
    failure_export_dir: str | Path | None = None,
    skip_existing: bool = False,
    experiment_id: str = "mllmu_smoke_iter7",
    image_batch_size: int = 1,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
) -> dict[str, Any]:
    """Evaluate BASE + MF/MG/MN and apply the separation gate.

    With ``rescore=True`` the persisted prediction parquets are re-scored
    (no model loading) — used to regenerate metrics after scorer changes.
    With ``skip_existing=True`` states whose prediction parquet already
    exists AND whose provenance sidecar matches this run are loaded
    instead of re-generated; a parquet that does not match (or predates
    sidecars entirely) is regenerated, never silently trusted.
    Hierarchy metrics (FILR/TGA/failure taxonomy/strata) are reported
    per split with TEST as the primary basis; ``failure_export_dir``
    receives per-example failure exports for inspection.
    ``experiment_id`` is stamped into every PredictionRecord and the
    report (Iteration 11: the pilot-100 run uses its own id so its
    artifacts can never be confused with the Iteration 7 smoke run).
    """
    smoke_dir = Path(smoke_dir)
    checkpoints_dir = Path(checkpoints_dir)
    queries = load_queries_parquet(smoke_dir / "queries.parquet")
    associations = load_associations_parquet(
        smoke_dir / "associations.parquet")
    repo_root = smoke_dir.parent.parent  # data/.. -> repo root
    states = states or ["BASE", "MF", "MG", "MN"]

    predictions_dir = Path(predictions_dir) if predictions_dir else None
    if predictions_dir:
        predictions_dir.mkdir(parents=True, exist_ok=True)

    generation_config = {
        "batch_size": batch_size,
        "image_batch_size": image_batch_size,
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
        "max_image_pixels": DEFAULT_MAX_IMAGE_PIXELS,
        "max_length": DEFAULT_MAX_LENGTH,
    }
    query_ids = [q.query_id for q in queries]

    def fingerprint_for(state: str, adapter_dir: Path | None):
        return PredictionFingerprint.build(
            experiment_id=experiment_id, checkpoint_id=state,
            repo_root=repo_root, data_dir=smoke_dir, model_id=model_id,
            adapter_dir=adapter_dir,
            generation_config=generation_config, num_rows=len(queries))

    metrics_by_state: dict[str, Any] = {}
    metrics_by_split: dict[str, Any] = {
        s: {} for s in ("train", "val", "test")}
    # Iteration 8 frozen hierarchy metrics: TEST split first (primary)
    hierarchy_by_split: dict[str, Any] = {
        s: {} for s in ("test", "train", "val", "pooled")}
    failure_export_dir = (Path(failure_export_dir)
                          if failure_export_dir else None)
    if failure_export_dir:
        failure_export_dir.mkdir(parents=True, exist_ok=True)
    for state in states:
        adapter_dir = None if state == "BASE" else \
            checkpoints_dir / state / "adapters"
        ppath = (predictions_dir / f"predictions_{state}.parquet"
                 if predictions_dir else None)
        expected = fingerprint_for(state, adapter_dir)
        preds: list[PredictionRecord] | None = None
        gen_info: dict[str, Any] = {}
        loaded_existing = False

        if rescore:
            if ppath is None:
                raise ValueError("rescore requires predictions_dir")
            log.info("Rescoring %s from %s...", state, ppath)
            preds = load_predictions_parquet(ppath)
            # Rescoring deliberately applies CURRENT scoring code to
            # previously generated bytes, so a code-fingerprint mismatch is
            # expected here.  It is recorded rather than refused, so the
            # report states exactly which generations were rescored.
            gen_info = {"rescored": True,
                        "provenance_mismatches": verify_sidecar(ppath,
                                                                expected)}
            loaded_existing = True
        elif skip_existing and ppath is not None and ppath.exists():
            reasons = verify_sidecar(ppath, expected)
            if reasons:
                log.warning("[%s] REFUSING to reuse %s — %d provenance "
                            "mismatch(es); regenerating instead:",
                            state, ppath.name, len(reasons))
                for r in reasons:
                    log.warning("    - %s", r)
            else:
                preds = load_predictions_parquet(ppath)
                problems = validate_prediction_coverage(
                    preds, query_ids, experiment_id, state)
                if problems:
                    raise SystemExit(
                        f"[{state}] {ppath.name} is provenance-valid but "
                        f"row-invalid:\n  " + "\n  ".join(problems))
                log.info("[%s] reusing provenance-validated predictions %s",
                         state, ppath)
                gen_info = {"reused_predictions": True}
                loaded_existing = True

        if preds is None:
            loaded_existing = False
            if adapter_dir is not None and not adapter_dir.exists():
                raise FileNotFoundError(
                    f"Missing adapter for state {state}: {adapter_dir}")
            log.info("Evaluating state %s...", state)
            preds, gen_info = evaluate_state(
                state, queries, associations, repo_root, model_id, device,
                adapter_dir, experiment_id=experiment_id,
                batch_size=batch_size, image_batch_size=image_batch_size,
                max_new_tokens=max_new_tokens)
            problems = validate_prediction_coverage(
                preds, query_ids, experiment_id, state)
            if problems:
                raise SystemExit(
                    f"[{state}] freshly generated predictions are "
                    f"row-invalid:\n  " + "\n  ".join(problems))

        pooled, per_split = metrics_for_predictions(preds, queries)
        pooled.update(gen_info)
        metrics_by_state[state] = pooled
        for s, m in per_split.items():
            metrics_by_split[s][state] = m
        for s in ("test", "train", "val"):
            hierarchy_by_split[s][state] = compute_hierarchy_metrics(
                preds, queries, associations, split=s)
        hierarchy_by_split["pooled"][state] = compute_hierarchy_metrics(
            preds, queries, associations, split=None)
        if failure_export_dir:
            export = export_failure_cases(preds, queries, associations,
                                          checkpoint_id=state)
            with open(failure_export_dir /
                      f"failure_cases_{state}.json", "w") as f:
                json.dump(export, f, indent=2, ensure_ascii=False)
            log.info("[%s] failure export: %d cases (%s)", state,
                     export["num_failure_cases"],
                     export["failure_counts"])
        if predictions_dir and not loaded_existing:
            import pandas as pd
            pd.DataFrame(
                [json.loads(p.model_dump_json()) for p in preds]
            ).to_parquet(ppath, index=False)
            # The sidecar is written with the parquet, never afterwards by
            # hand: downstream scripts refuse a parquet that has no sidecar,
            # so a gate run that skipped this step would force every
            # candidate-selection and final-evaluation pass to regenerate.
            write_sidecar(ppath, expected)
            log.info("[%s] wrote %s + provenance sidecar", state, ppath.name)
        log.info("[%s] fine_recovery=%.3f target_post=%.3f "
                 "retain_same=%.3f retain_other=%.3f leakage=%.3f",
                 state,
                 (pooled.get("fine_recovery") or {}).get("baseline_accuracy"),
                 (pooled.get("target_core") or {}).get("post_unlearning_accuracy"),
                 (pooled.get("retain_same_entity") or {}).get("baseline_accuracy"),
                 (pooled.get("retain_other_entity") or {}).get("baseline_accuracy"),
                 (pooled.get("target_core") or {}).get("leakage_rate"))

    gate_states = {s: m for s, m in metrics_by_state.items()
                   if s in ("MF", "MG", "MN")}
    passed, reasons = separation_gate(gate_states)
    test_gate_states = {s: metrics_by_split["test"][s]
                        for s in ("MF", "MG", "MN")
                        if s in metrics_by_split["test"]}
    test_passed, test_reasons = separation_gate(test_gate_states)

    report = {
        "experiment_id": experiment_id,
        "model_id": model_id,
        "states": states,
        "num_queries": len(queries),
        # Read from the frozen manifest, never assumed: this gate report is
        # the artifact that says WHICH dataset the separation was measured
        # on, and Iteration 11R re-froze the dataset underneath it.
        "dataset_version": dataset_version(smoke_dir),
        "generation_config": {
            **generation_config,
            "note": "identical for every state in this report",
        },
        "test_split_exposure": (
            "This gate generates every query in every split, TEST "
            "included, and reports a separation gate on the test split as "
            "well as pooled. It runs BEFORE candidate selection, so the "
            "test split informs a go/no-go decision here and is not an "
            "untouched hold-out. Candidates are still never ranked on "
            "their own test predictions — selection is train+val only."),
        "metrics_split_semantics": (
            "metrics_by_state pools train/val/test; metrics_by_split "
            "reports each paraphrase split separately — the TEST numbers "
            "measure generalization to unseen wording and are the "
            "authoritative held-out view."
        ),
        "metrics_by_state": metrics_by_state,
        "metrics_by_split": metrics_by_split,
        "hierarchy_metrics": hierarchy_by_split,
        "hierarchy_metrics_note": (
            "Frozen Iteration 8 research metrics (FILR, TGA, ancestor "
            "retention, under/over-forgetting, wrong-branch, refusal, "
            "hallucination, route/type/depth strata). TEST split is the "
            "PRIMARY basis; 'pooled' is secondary. These are the metrics "
            "Iteration 9's MF->MU baselines will be compared on "
            "(central comparison: MU ~= MG with retained non-target "
            "knowledge)."
        ),
        "separation_gate": {
            "passed": passed,
            "reasons": reasons,
            "basis": "pooled train/val/test",
            "definition": {
                "fine_recovery": "MF > MG and MF > MN by >= 0.15 "
                                 "(baseline accuracy, adversarial excluded)",
                "target_abstraction": "MG > MN by >= 0.15 on target_core "
                                      "post-unlearning accuracy",
                "retain": "all states >= 0.5 on same-entity and "
                          "other-entity retain accuracy",
            },
        },
        "separation_gate_test_split": {
            "passed": test_passed,
            "reasons": test_reasons,
            "basis": "test paraphrase split only",
        },
    }
    Path(report_path).parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    log.info("Separation gate (pooled): %s%s", "PASSED" if passed else "FAILED",
             f" ({'; '.join(reasons)})" if reasons else "")
    log.info("Separation gate (test split): %s%s",
             "PASSED" if test_passed else "FAILED",
             f" ({'; '.join(test_reasons)})" if test_reasons else "")
    return report
