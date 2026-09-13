"""MF -> MU trainer with bounded suppression and an MF anchor (Stage 2).

``unlearning_trainer.train_unlearning`` is one of the eight paths the Stage-1
freeze hash-bound, so Stage 2 cannot add a mode to it.  This module re-runs
that loop with two additions, and it is worth being clear about why there are
two rather than one.

Stage 2 was preregistered as an MF-preservation regularizer: pin
``KL(p_MF || p_theta)`` on the fit-half retained prompts, so the model stays
where MF was on knowledge it should keep.  A calibration pass over the eight
adapters Stage 1 already produced measured that plan against the numbers Stage
1c filed, and it does not survive:

* MG, the oracle the selection criterion targets, sits 0.0296 nats from MF on
  those prompts while ``B4_w4.0_lam0.5_lr2e-05_ep5`` sits 0.0308 -- the same
  drift -- and their image retain-same differs by 0.105.  Two further pairs
  within 0.015 of each other in drift differ by 0.150 and 0.169 in retention,
  against a floor that resolves 0.0044.
* The anchor's own optimum is the wrong place.  KL = 0 is MF, which is no
  unlearning at all; the oracle sits at 0.0296, not at 0.  A strong beta
  therefore drives a candidate toward MF rather than toward MG.
* What DOES rank the candidates is how far the suppression term drove the fine
  fact: Spearman rho between ``fine_target`` NLL and image retention is -0.976
  and -1.000, against -0.571 to -0.619 for the KL.  And rho between
  ``fine_target`` NLL and D_G is -0.024, so the overshoot past MG's own level
  of 1.3643 buys nothing at the criterion and costs up to 0.30 of retention.

So both mechanisms are implemented.  ``gd_capped`` bounds the ascent -- the
coordinate the measurement says governs -- and ``anchor`` is trained too, at
the beta the calibration derived, because an eight-point observational
argument is not an experiment and the mechanism Stage 2 was preregistered with
deserves to be run rather than argued away.

    gd_capped   -weight * min(NLL, cap)   gradient EXACTLY zero above the cap
    anchor      +weight * KL(p_MF || p_theta) over supervised positions

Everything else -- LoRA config, optimizer, seeds, the per-group shuffle, the
round-robin interleave, the accumulation normalization by the ACTUAL trailing
group size -- is the frozen recipe, so a Stage-2 row differs from its Stage-1
counterpart in the objective and in nothing else.

WHY THE CAP IS AN ABSOLUTE NUMBER OF NATS
-----------------------------------------
The alternative is to cap at ``NLL(target_level) + m``, so the bound tightens
as the coarse answer is learned.  That was rejected on a measured behaviour of
``torch.clamp``: with a tensor cap the gradient flows into the cap as well, so
a non-detached ``NLL(target_level)`` cap contributed -0.5 to the coarse term's
gradient -- the ascent term pushing AGAINST the rewrite it is supposed to
coexist with.  Detaching fixes it, but the result is a moving target coupled to
a second loss.  A plain float cap has no such path: ``clamp(x, max=2.0)``
returns gradient -weight at or below 2.0 and exactly 0.0 above it, which was
verified before this module was written.

FAITHFULNESS IS MEASURED, NOT ASSERTED
--------------------------------------
A second copy of a training loop can drift from the first, and a drift would
show up as a difference between Stage 1 and Stage 2 that has nothing to do with
the mechanism.  Two independent checks bound that:

* with every group in a plain ``sft``/``gd`` mode this trainer must reproduce
  the incumbent Stage-1 adapter BYTE FOR BYTE, which
  ``scripts/train_iter12_stage2.py --control`` runs and compares by sha256; and
* ``tests/unit/test_iteration12_stage2.py`` drives both loops against one stub
  model on CPU and asserts identical parameter updates, so a divergence is
  caught without a GPU.

The anchor's own correctness rests on a third check that runs at every step:
the NLL recomputed from the extracted log-probabilities must equal the loss the
model returned.  A causal LM predicts ``labels[i]`` from ``logits[i-1]``, and an
off-by-one there is invisible in the numbers -- it just anchors every position
against its neighbour's reference distribution.
"""

from __future__ import annotations

import json
import random
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from granunlearn.logging_utils import setup_logger
from granunlearn.training.preservation_anchor import (
    ANCHOR_MODE,
    PRESERVE_GROUP,
    ReferenceCache,
    anchor_loss,
    assert_nll_matches_model,
    supervised_logprobs,
)
from granunlearn.training.reference_trainer import (
    ReferenceRecipe,
    _encode_example,
    set_recipe_seeds,
)
from granunlearn.training.state_datasets import load_state_examples

log = setup_logger("preservation_trainer")

#: Gradient ascent, bounded: ``-weight * min(NLL, cap)``.
CAPPED_MODE = "gd_capped"

#: The mode vocabulary this trainer accepts.  ``GroupSpec.mode`` in the frozen
#: trainer is annotated ``Literal["sft", "gd"]`` and nothing enforces it at
#: runtime, but reusing that dataclass would mean expressing Stage-2 knobs in a
#: type that does not declare them.  So Stage 2 has its own spec, and the
#: vocabulary is checked explicitly below rather than left to an annotation
#: Python never evaluates.
STAGE2_MODES: tuple[str, ...] = ("sft", "gd", CAPPED_MODE, ANCHOR_MODE)

#: Groups whose objective is the transformation rather than its restraint.
TARGET_GROUPS: tuple[str, ...] = ("fine_target", "target_level")


@dataclass(frozen=True)
class PreservationGroupSpec:
    """One Stage-2 objective component.

    ``cap`` is read only in ``gd_capped`` mode and must be absent otherwise: a
    cap on an ``sft`` group would silently bound the rewrite, and a cap on an
    ``anchor`` group has no meaning at all.

    ``anchor_weight`` is the alternative way to spend the anchor: instead of
    REPLACING the group's supervised term (mode ``anchor``), it adds the KL
    term alongside it, from the same forward pass.  That exists so a hybrid row
    has exactly the same interleaved stream, and therefore exactly the same
    number of optimizer steps, as the incumbent it is compared against.  Two
    separate group entries pointing at one file would have lengthened every
    epoch by 183 micro-batches and quietly changed the schedule under test.
    """

    name: str
    path: str | Path
    mode: Literal["sft", "gd", "gd_capped", "anchor"]
    weight: float = 1.0
    cap: float | None = None
    anchor_weight: float | None = None

    def describe(self) -> dict[str, Any]:
        return {"name": self.name, "mode": self.mode, "weight": self.weight,
                "cap": self.cap, "anchor_weight": self.anchor_weight}


def validate_stage2_groups(groups: list[PreservationGroupSpec]) -> list[str]:
    """Structural invariants on a Stage-2 objective."""
    errors: list[str] = []
    names = [g.name for g in groups]
    if len(names) != len(set(names)):
        errors.append(f"repeated group {names}")
    anchors = [g for g in groups if g.mode == ANCHOR_MODE]
    capped = [g for g in groups if g.mode == CAPPED_MODE]
    augmented = [g for g in groups if g.anchor_weight is not None]
    for g in groups:
        if g.mode not in STAGE2_MODES:
            errors.append(f"{g.name}: unknown mode {g.mode!r}; expected one "
                          f"of {list(STAGE2_MODES)}")
        if g.weight <= 0:
            errors.append(f"{g.name}: weight must be > 0")
        if g.mode == CAPPED_MODE:
            if g.cap is None:
                errors.append(f"{g.name}: mode {CAPPED_MODE!r} needs a cap in "
                              f"nats; without one it is unbounded ascent, "
                              f"which is the behaviour Stage 2 exists to "
                              f"bound")
            elif g.cap <= 0:
                errors.append(f"{g.name}: cap must be > 0 nats, got {g.cap}")
        elif g.cap is not None:
            errors.append(f"{g.name}: a cap is meaningless in mode "
                          f"{g.mode!r} and would silently do nothing")
    for c in capped:
        if c.name not in TARGET_GROUPS:
            #: The cap bounds the term that GENERATES the displacement.  On the
            #: preserved group there is no ascent to bound, so a cap there is
            #: either a no-op or a mistake about which group does the damage.
            errors.append(
                f"{c.name}: {CAPPED_MODE!r} belongs on a target group "
                f"({list(TARGET_GROUPS)}), which is where the unbounded "
                f"ascent is; capping the preserved group bounds nothing")
    for a in anchors:
        if a.name != PRESERVE_GROUP:
            #: Not a style rule.  Anchoring ``fine_target`` would oppose the
            #: suppression term on the prompts the method exists to change,
            #: and anchoring ``target_level`` would oppose the rewrite, so
            #: either would quietly turn the study into "do not unlearn".
            errors.append(
                f"{a.name}: only the {PRESERVE_GROUP!r} group may carry mode "
                f"{ANCHOR_MODE!r}. Anchoring a target group opposes the "
                f"transformation the method exists to perform.")
        if a.anchor_weight is not None:
            errors.append(
                f"{a.name}: mode {ANCHOR_MODE!r} already applies the KL term; "
                f"setting anchor_weight as well would apply it twice")
    for g in augmented:
        if g.mode != "sft":
            errors.append(f"{g.name}: anchor_weight augments an sft term, so "
                          f"it cannot be set in mode {g.mode!r}")
        if g.name != PRESERVE_GROUP:
            errors.append(
                f"{g.name}: only the {PRESERVE_GROUP!r} group may be anchored")
        if g.anchor_weight <= 0:
            errors.append(f"{g.name}: anchor_weight must be > 0, got "
                          f"{g.anchor_weight}")
    if len(anchors) + len(augmented) > 1:
        errors.append(
            f"at most one anchored group, got "
            f"{[g.name for g in anchors + augmented]}")
    if len(capped) > 1:
        errors.append(
            f"at most one capped group, got {[c.name for c in capped]}")
    return errors


def train_with_preservation(
    method_id: str,
    groups: list[PreservationGroupSpec],
    output_dir: str | Path,
    device: str = "cuda:0",
    recipe: ReferenceRecipe | None = None,
    init_adapter_dir: str | Path | None = None,
    cache_dir: str | Path | None = None,
    mf_adapter_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Train one MF->MU candidate, optionally capped and/or anchored.

    Signature-compatible with ``train_unlearning``: passing only ``sft``/``gd``
    groups and no ``cache_dir`` runs the frozen recipe unchanged, which is what
    the byte-reproduction control relies on.
    """
    import torch
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoModelForImageTextToText, AutoProcessor

    recipe = recipe or ReferenceRecipe()
    output_dir = Path(output_dir)

    errors = validate_stage2_groups(groups)
    if errors:
        raise ValueError(f"{method_id}: invalid Stage-2 objective: {errors}")
    anchored = [g for g in groups
                if g.mode == ANCHOR_MODE or g.anchor_weight is not None]
    cache: ReferenceCache | None = None
    if anchored:
        if cache_dir is None:
            raise ValueError(
                f"{method_id}: group {anchored[0].name!r} carries mode "
                f"{ANCHOR_MODE!r} but no cache_dir was given, so there is no "
                f"reference distribution to anchor against")
        cache = ReferenceCache.load(cache_dir)
        if mf_adapter_dir is not None:
            #: Refuse before training rather than after: an anchor built from a
            #: different MF preserves a different function, and the resulting
            #: adapter would look perfectly ordinary.
            cache.assert_built_from(mf_adapter_dir)

    (output_dir / "adapters").mkdir(parents=True, exist_ok=True)
    set_recipe_seeds(recipe.seed)

    from granunlearn.config import _find_repo_root
    repo_root = _find_repo_root(Path.cwd()) or Path.cwd()
    loaded = {g.name: load_state_examples(g.path, repo_root=repo_root)
              for g in groups}
    for g in groups:
        log.info("[%s] group %s: %d examples (%s, weight %.4f%s)",
                 method_id, g.name, len(loaded[g.name]), g.mode, g.weight,
                 f", cap {g.cap:g} nats" if g.cap is not None else "")

    processor = AutoProcessor.from_pretrained(recipe.model_id)
    if processor.tokenizer.padding_side != "left":
        processor.tokenizer.padding_side = "left"
    model = AutoModelForImageTextToText.from_pretrained(
        recipe.model_id, device_map={"": device},
        torch_dtype=torch.bfloat16 if recipe.bf16 else torch.float32,
    )
    if recipe.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    if init_adapter_dir is not None:
        model = PeftModel.from_pretrained(
            model, str(init_adapter_dir), is_trainable=True)
        log.info("[%s] continuing from MF adapter %s",
                 method_id, init_adapter_dir)
    else:
        lora_config = LoraConfig(
            r=recipe.lora_r, lora_alpha=recipe.lora_alpha,
            lora_dropout=recipe.lora_dropout,
            target_modules=list(recipe.lora_target_modules), bias="none")
        model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=recipe.learning_rate, weight_decay=recipe.weight_decay)

    summary: dict[str, Any] = {
        "method_id": method_id,
        "recipe": recipe.to_dict(),
        "init_adapter_dir": str(init_adapter_dir) if init_adapter_dir
        else None,
        "groups": [dict(g.describe(), num_examples=len(loaded[g.name]))
                   for g in groups],
        "epochs": [],
        "num_optimizer_steps": 0,
        "device": device,
    }
    if cache is not None:
        summary["preservation_anchor"] = {
            "cache_dir": str(cache_dir),
            "kl_direction": cache.sidecar["kl_direction"],
            "mf_adapter_sha256": cache.sidecar["mf_adapter_sha256"],
            "base_model_revision": cache.sidecar["base_model_revision"],
            "num_preserved_examples": cache.sidecar["num_examples"],
            "num_preserved_positions": cache.sidecar["num_positions"],
            "vocab_size": cache.sidecar["vocab_size"],
            "cache_tensor_sha256": cache.sidecar["tensor_sha256"],
            "anchored_groups": [g.name for g in anchored],
            #: The KL coefficient, which is ``weight`` for a group whose mode IS
            #: ``anchor`` and ``anchor_weight`` for one that keeps its
            #: supervised term.  Recording ``weight`` for both would misreport
            #: the hybrid row's anchor strength as 1.0.
            "anchor_weight": {
                g.name: (g.weight if g.mode == ANCHOR_MODE
                         else g.anchor_weight) for g in anchored},
            "anchor_alongside_supervised_term": {
                g.name: g.mode != ANCHOR_MODE for g in anchored},
            "note": (
                "The anchor term is weight * KL(p_MF || p_theta) over the "
                "supervised positions of the fit-half retain group. It is "
                "exactly zero with exactly zero gradient at theta = MF, so its "
                "optimum is MF -- which is no unlearning at all. That is the "
                "behaviour the calibration measured, and it is why this row is "
                "a test of the preregistered mechanism rather than a candidate "
                "expected to win."),
        }

    global_step = 0
    t0 = time.time()
    accum = recipe.gradient_accumulation_steps
    worst_nll_disagreement = 0.0

    def anchor_kl(example_, out_, enc_, epoch_):
        """``KL(p_MF || p_theta)`` for one preserved example, with two guards.

        Shared by both anchor paths so they cannot drift apart: ``anchor`` mode
        applies this term alone and ``anchor_weight`` applies it beside the
        supervised term, but the quantity is computed once, identically.
        """
        nonlocal worst_nll_disagreement
        assert cache is not None
        logp, targets = supervised_logprobs(out_.logits, enc_["labels"])
        #: Guard 1 -- the extraction must reproduce the model's own loss.  A
        #: causal LM predicts ``labels[i]`` from ``logits[i-1]``, and an
        #: off-by-one is invisible in the numbers: it just anchors every
        #: position against its neighbour's reference distribution.
        worst_nll_disagreement = max(
            worst_nll_disagreement,
            assert_nll_matches_model(
                logp, targets, out_.loss.item(),
                f"{method_id}/ep{epoch_ + 1}/{example_.example_id}"))
        ref_logp, _ref_targets = cache.slice_for(example_.example_id)
        #: Guard 2 -- the supervised tokens must be the ones the cache recorded,
        #: so cached and current log-probabilities are the same positions.  The
        #: targets are not otherwise needed: the KL runs over the whole
        #: vocabulary, not just the labelled token.
        cache.assert_aligned(example_.example_id, targets)
        return anchor_loss(ref_logp.to(device), logp)

    for epoch in range(recipe.num_epochs):
        # deterministic per-group shuffles, then round-robin interleave
        streams: list[list[tuple[str, int]]] = []
        for gi, g in enumerate(groups):
            order = list(range(len(loaded[g.name])))
            random.Random(f"{recipe.seed}:{epoch}:{gi}").shuffle(order)
            streams.append([(g.name, idx) for idx in order])
        stream: list[tuple[str, int]] = []
        longest = max(len(s) for s in streams)
        for pos in range(longest):
            for s in streams:
                if pos < len(s):
                    stream.append(s[pos])

        spec_by_name = {g.name: g for g in groups}
        group_loss: dict[str, list[float]] = defaultdict(list)
        group_kl: dict[str, list[float]] = defaultdict(list)
        #: How often the cap was already binding.  A cap that never binds is
        #: the incumbent recipe with extra bookkeeping, and a cap that binds on
        #: every example from the first step is a cap set below where the model
        #: started; both would make the row uninformative, and this is what
        #: says which happened.
        group_at_cap: dict[str, list[bool]] = defaultdict(list)
        epoch_loss, n_batches = 0.0, 0
        optimizer.zero_grad()
        for i, (gname, idx) in enumerate(stream):
            spec = spec_by_name[gname]
            sign = -1.0 if spec.mode in ("gd", CAPPED_MODE) else 1.0
            group_start = (i // accum) * accum
            group_size = min(accum, len(stream) - group_start)
            example = loaded[gname][idx]
            enc = _encode_example(
                example, processor, recipe.max_length,
                recipe.max_image_pixels)
            enc = {k: v.to(device) for k, v in enc.items()
                   if torch.is_tensor(v)}
            out = model(**enc)

            #: The UNCAPPED NLL is recorded for every group under the same key
            #: Stage 1 used, so the two studies are comparable on the trajectory
            #: of the same quantity: Stage 1 watched ``avg_loss_retain`` rise
            #: from ~0.03 to 0.49-0.98 while an SFT term minimised it, and
            #: ``avg_loss_fine_target`` is the coordinate the calibration found
            #: governs retention damage.
            group_loss[gname].append(out.loss.item())

            if spec.mode == CAPPED_MODE:
                capped = torch.clamp(out.loss, max=spec.cap)
                group_at_cap[gname].append(
                    bool(out.loss.item() > spec.cap))
                term = sign * spec.weight * capped
                applied = sign * spec.weight * float(capped.item())
            elif spec.mode == ANCHOR_MODE:
                kl = anchor_kl(example, out, enc, epoch)
                group_kl[gname].append(float(kl.item()))
                term = spec.weight * kl
                applied = spec.weight * float(kl.item())
            else:
                term = sign * spec.weight * out.loss
                #: The frozen trainer's own expression, kept verbatim: this is
                #: the branch the byte-reproduction control runs through, so
                #: even the bookkeeping float is computed the same way it was.
                applied = sign * spec.weight * out.loss.item()
                if spec.anchor_weight is not None:
                    #: One backward, not two.  Both terms are functions of the
                    #: same ``out.logits``, so a second ``backward()`` would
                    #: traverse an already-freed graph and raise.
                    kl = anchor_kl(example, out, enc, epoch)
                    group_kl[gname].append(float(kl.item()))
                    term = term + spec.anchor_weight * kl
                    applied += spec.anchor_weight * float(kl.item())
            (term / group_size).backward()
            epoch_loss += applied
            n_batches += 1
            if (i + 1) % accum == 0 or i + 1 == len(stream):
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

        epoch_entry: dict[str, Any] = {"epoch": epoch + 1}
        for gname, losses in group_loss.items():
            epoch_entry[f"avg_loss_{gname}"] = round(
                sum(losses) / len(losses), 4)
        for gname, kls in group_kl.items():
            epoch_entry[f"avg_kl_{gname}"] = round(
                sum(kls) / len(kls), 6)
        for gname, flags in group_at_cap.items():
            epoch_entry[f"frac_at_cap_{gname}"] = round(
                sum(flags) / len(flags), 4)
        epoch_entry["avg_objective"] = round(
            epoch_loss / max(n_batches, 1), 4)
        summary["epochs"].append(epoch_entry)
        log.info("[%s] epoch %d/%d %s", method_id, epoch + 1,
                 recipe.num_epochs,
                 {k: v for k, v in epoch_entry.items() if k != "epoch"})

    summary["num_optimizer_steps"] = global_step
    summary["train_seconds"] = round(time.time() - t0, 1)
    if cache is not None:
        summary["preservation_anchor"]["nll_agreement_max"] = \
            worst_nll_disagreement
    model.save_pretrained(output_dir / "adapters")
    processor.save_pretrained(output_dir / "processor")
    with open(output_dir / "training_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    log.info("[%s] saved adapters -> %s (%d steps, %.1fs)", method_id,
             output_dir, global_step, summary["train_seconds"])
    return summary


def make_control_checkpoint(
    method_id: str,
    init_adapter_dir: str | Path,
    output_dir: str | Path,
    recipe: ReferenceRecipe | None = None,
) -> dict[str, Any]:
    """B0 for Stage 2: the MF adapter copied, exactly as the frozen no-op does.

    Reimplemented rather than imported because ``make_noop_checkpoint`` is
    inside a hash-bound module whose behaviour Stage 2 must not depend on
    changing; the copy is six lines and the byte comparison that follows it is
    what actually establishes equivalence.
    """
    import shutil
    output_dir = Path(output_dir)
    init_adapter_dir = Path(init_adapter_dir)
    recipe = recipe or ReferenceRecipe()
    if (output_dir / "adapters").exists():
        shutil.rmtree(output_dir / "adapters")
    shutil.copytree(init_adapter_dir, output_dir / "adapters")
    proc_src = init_adapter_dir.parent / "processor"
    if proc_src.exists():
        if (output_dir / "processor").exists():
            shutil.rmtree(output_dir / "processor")
        shutil.copytree(proc_src, output_dir / "processor")
    summary = {
        "method_id": method_id,
        "noop": True,
        "recipe": recipe.to_dict(),
        "init_adapter_dir": str(init_adapter_dir),
        "num_optimizer_steps": 0,
        "groups": [],
        "note": "MF adapter copied unchanged. Stage 2 reports B0's eight "
                "numbers from the parquet the Stage-1c freeze already bound, "
                "so this copy exists to keep the grid shape Stage 1 used.",
    }
    with open(output_dir / "training_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    log.info("[%s] no-op checkpoint written -> %s", method_id, output_dir)
    return summary
