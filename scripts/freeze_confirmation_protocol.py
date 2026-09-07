"""Freeze the Iteration 11C confirmation protocol.

    python scripts/freeze_confirmation_protocol.py --tag pilot100
    python scripts/freeze_confirmation_protocol.py --tag pilot100 --check-only

Stage 2 of the confirmation phase, run BEFORE any confirmation probe is
built and before any confirmation inference.  It writes
``data/reports/mllmu_pilot100_confirmation_freeze.json`` binding everything
the confirmatory claims depend on:

* the selected checkpoints, by full adapter contract (weights AND
  ``adapter_config.json``), cross-checked against what the 11R sidecars
  actually recorded for those checkpoint ids;
* B3's granularity lambda and the rest of its training recipe;
* the generation settings, read from the committed prediction manifest and
  cross-checked against the ``reference_eval`` module defaults;
* the scorer and the analysis code, by content hash;
* delta = 0.05 AND ITS ROLE, which the 11C-2a decisions changed;
* the CI unit, bootstrap size and seed, read from the function signature
  rather than written down beside it;
* the familywise alpha and the ONE-SIDED convention that makes the Holm
  threshold and the cluster requirement the same test instead of one being a
  two-sided reading of the other (Iteration 11C-R1's finding #4);
* the PRIMARY ESTIMAND, named once, with the per-stratum decomposition
  demoted to a pre-specified secondary and the fixed-cohort limit recorded,
  because "pooled" and "the stratum that carries the effect" are different
  quantities and a preregistration that names both claims neither
  (finding #3);
* the SELECTED confirmation size — ONE row of the power grid, the command
  line that fetches it, and the identifiers frozen now versus the ones stage
  3 is obliged to commit before scoring (finding #1);
* the sealed-split and score-exactly-once invariants, as assertions with a
  named enforcement point, because an invariant nobody is obliged to check
  is a comment.

Nothing is scored and no GPU is touched.

Every number here is READ from a committed artifact or a module constant.
A protocol freeze that restates its own parameters is a second copy of them,
and two copies is how the frozen value and the executed value come to
differ — the same failure mode Iteration 11R1 closed for the sidecars.

Refuses to overwrite an existing freeze without ``--allow-refreeze``: a
protocol that can be quietly re-frozen after inference is not a protocol.
"""

from __future__ import annotations

import argparse
import inspect
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from granunlearn.config import _find_repo_root
from granunlearn.evaluation.paired_ci import PAIRED_METRICS, paired_rate_diff_ci
from granunlearn.evaluation.prediction_provenance import (
    CODE_FINGERPRINT_MODULES,
    PROVENANCE_CONTRACT_VERSION,
    adapter_contract,
    code_fingerprint,
    dataset_fingerprint,
    environment_fingerprint,
    resolve_adapter_dir,
    sha256_file,
)
from granunlearn.evaluation.reference_eval import (
    DEFAULT_MAX_IMAGE_PIXELS,
    DEFAULT_MAX_LENGTH,
    DEFAULT_MAX_NEW_TOKENS,
)
from granunlearn.logging_utils import setup_logger

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_pilot100_final import EQUIVALENCE_MARGIN_TGA  # noqa: E402
from power_analysis_confirmation import (  # noqa: E402
    ALPHA_ONE_SIDED,
    BOOTSTRAP_SEED,
    CLAIM_KIND,
    CLAIM_KIND_VS_MG,
    CONFIRM_FETCH_IMAGES_PER_SPECIES,
    CONFIRM_FETCH_OUT,
    CONFIRM_FETCH_SEED,
    CONFIRM_NEW_PHOTOS_PER_SPECIES,
    CONFIRM_NEW_WORDING_PROBES_PER_PERSON,
    FAMILYWISE_ALPHA,
    PRIMARY_ESTIMAND,
    PRIMARY_ESTIMAND_STRATA,
    PRIMARY_FAMILY,
    STRATUM_ESTIMAND_STATUS,
    WHAT_THE_CONFIRMATION_DOES_NOT_ESTABLISH,
    z,
)

log = setup_logger("freeze_confirmation_protocol")

#: The states the confirmation split scores, and why each is there.
#: Deliberately minimal: a state no claim references is GPU time spent
#: producing a number nobody preregistered, and every extra state in the
#: same batch layout is also extra decoding-noise surface.
CONFIRMATION_STATES = ("B3", "B0", "MG")

STATE_ROLE = {
    "B3": "subject of both primary claims and of every descriptive interval",
    "B0": "the no-op comparator: B3-vs-B0 carries TGA superiority and FILR "
          "reduction, and the descriptive retention intervals",
    "MG": "the granularity oracle, DESCRIPTIVE only: its interval is "
          "published with an explicit statement that equivalence is NOT "
          "concluded",
}

#: States deliberately NOT scored.  Recorded because an omission is a
#: decision a reviewer should be able to audit, and because "we did not
#: measure that" must not be discoverable only by noticing an absence.
EXCLUDED_STATES = {
    "BASE": "no confirmation claim references the un-finetuned model",
    "MF": "excluded only if the measured adapter contract proves B0 IS the "
          "no-op copy of it; see b0_is_the_no_op. If that check fails the "
          "freeze refuses rather than dropping a distinct state.",
    "MN": "the negative-control oracle carries no confirmation claim",
    "B1": "not the subject of any preregistered claim",
    "B2": "not the subject of any preregistered claim",
    "B2R": "not the subject of any preregistered claim",
}

#: Analysis scripts whose bytes must not move between the freeze and the
#: scoring run.  These are NOT in CODE_FINGERPRINT_MODULES, which binds the
#: modules that turn queries into scores; these turn scores into claims, and
#: a silent edit to either would change what the frozen protocol means
#: without changing a single prediction.
ANALYSIS_SCRIPTS = (
    "scripts/evaluate_pilot100_final.py",
    "scripts/power_analysis_confirmation.py",
    "scripts/select_unlearning_checkpoints.py",
    "scripts/freeze_confirmation_protocol.py",
)


#: Fields excluded when ``--check-only`` compares a fresh derivation against
#: the committed freeze.  Each is excluded for a stated reason, and this is a
#: module constant so the script and its tests share ONE definition rather
#: than two lists that can drift apart.
#:
#: * ``frozen_at_utc`` / ``git_commit`` / ``git_dirty`` — legitimately
#:   different between two derivations, and informational in the sidecar
#:   contract for the same reason.
#: * ``environment`` — records what is importable IN THE CURRENT PROCESS.
#:   Under the CPU-only CI emulation torch, transformers and peft are
#:   blocked, so their versions come back null and the block cannot match a
#:   freeze written on a GPU machine.  That is precisely why it is diagnostic
#:   and binds nothing: including it here would make the drift check fail on
#:   an environment difference that cannot move a decoded token, and a drift
#:   check that cries wolf gets ignored.
VOLATILE_FREEZE_FIELDS = frozenset({
    "frozen_at_utc", "git_commit", "git_dirty", "environment"})


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _load_report(reports: Path, name: str) -> dict[str, Any]:
    p = reports / name
    if not p.exists():
        raise SystemExit(f"required committed report is missing: {p}")
    return json.loads(p.read_text())


def _bound(p: Path, repo_root: Path) -> dict[str, Any]:
    """Path + sha256 of a committed artifact this freeze rests on."""
    return {
        "path": str(p.relative_to(repo_root)) if p.is_relative_to(repo_root)
        else str(p),
        "sha256": sha256_file(p),
    }


def _ci_parameters() -> dict[str, Any]:
    """The paired-CI settings, read off the function signature.

    Written down beside the function these would be a second copy of the
    defaults, and the copy is what a freeze would bind while the function is
    what actually runs.
    """
    sig = inspect.signature(paired_rate_diff_ci)
    p = sig.parameters
    return {
        "n_bootstrap": p["n_bootstrap"].default,
        "ci_level": p["ci_level"].default,
        "seed": p["seed"].default,
        "read_from": "inspect.signature(paired_rate_diff_ci)",
    }


def build_freeze(repo_root: Path, tag: str) -> dict[str, Any]:
    reports = repo_root / "data" / "reports"
    data_dir = repo_root / "data" / f"mllmu_hier_{tag}"

    selection = _load_report(reports, f"mllmu_{tag}_unlearning_selection.json")
    power = _load_report(reports, f"mllmu_{tag}_confirmation_power.json")
    manifest = _load_report(reports, f"mllmu_{tag}_prediction_manifest.json")
    final_eval = _load_report(reports, f"mllmu_{tag}_final_evaluation.json")

    refusals: list[str] = []

    # ---- the decisions this freeze inherits ----
    prereg = power["preregistration_decisions"]
    if prereg["primary_family"] != list(PRIMARY_FAMILY):
        refusals.append(
            f"the power report's primary family {prereg['primary_family']} "
            f"does not match the module's {list(PRIMARY_FAMILY)}")

    # The declared threshold and the sizing level must be the SAME test.
    # Iteration 11C-R1's finding #4: the report declared a one-sided Holm
    # worst case of familywise/k but the sizing function halved its alpha
    # again, so the cluster requirement was computed at one-sided 0.0125
    # while the preregistration said 0.025.  Conservative, but it meant the
    # published requirement was not the requirement of the published test.
    holm = power.get("holm_primary_family") or {}
    k = len(PRIMARY_FAMILY)
    want_alpha = FAMILYWISE_ALPHA / k
    if holm.get("familywise_alpha") != FAMILYWISE_ALPHA:
        refusals.append(
            f"the power report declares familywise alpha "
            f"{holm.get('familywise_alpha')} but the module declares "
            f"{FAMILYWISE_ALPHA}; the preregistration must state ONE "
            f"familywise error rate")
    if holm.get("sizing_alpha_one_sided") != round(want_alpha, 6):
        refusals.append(
            f"the power report sized the primary claims at a one-sided alpha "
            f"of {holm.get('sizing_alpha_one_sided')} while its own Holm "
            f"worst-case threshold is {round(want_alpha, 6)} = familywise/k; "
            f"the threshold and the sizing are not the same test")
    if holm.get("critical_value_z") != round(z(1 - want_alpha), 6):
        refusals.append(
            f"the power report's sizing critical value "
            f"{holm.get('critical_value_z')} is not z(1 - familywise/k) = "
            f"{round(z(1 - want_alpha), 6)}, so the cluster requirement was "
            f"computed at a different alpha than the one declared")

    # The estimand and the selected size are checked against the module
    # rather than copied from it, because a freeze that restated them would
    # bind its own copy and the sizing code could move underneath.  A
    # disagreement is a protocol contradiction, not a preference.
    estimand = power.get("primary_estimand") or {}
    if estimand.get("declared") != PRIMARY_ESTIMAND:
        refusals.append(
            f"the power report declares the primary estimand as "
            f"{estimand.get('declared')!r} but the module declares "
            f"{PRIMARY_ESTIMAND!r}")
    if estimand.get("strata_whose_probes_enter_it") != \
            list(PRIMARY_ESTIMAND_STRATA):
        refusals.append(
            f"the power report's primary estimand draws on "
            f"{estimand.get('strata_whose_probes_enter_it')} but the module "
            f"declares {list(PRIMARY_ESTIMAND_STRATA)}")
    size = power.get("confirmation_size") or {}
    photos = size.get("held_out_photographs") or {}
    words = size.get("new_wording_probes") or {}
    for field, want in (
            ("new_photographs_per_species", CONFIRM_NEW_PHOTOS_PER_SPECIES),
            ("new_photographs_total",
             CONFIRM_NEW_PHOTOS_PER_SPECIES
             * photos.get("species_covered", 0)),
            ("fetch_images_per_species", CONFIRM_FETCH_IMAGES_PER_SPECIES),
            ("fetch_seed", CONFIRM_FETCH_SEED),
            ("fetch_out", CONFIRM_FETCH_OUT)):
        if photos.get(field) != want:
            refusals.append(
                f"the power report's selected confirmation size sets "
                f"{field} = {photos.get(field)!r} but the module declares "
                f"{want!r}, so the size the analysis selected is not the "
                "size the sizing code would build")
    for field, want in (
            ("new_probes_per_target_person",
             CONFIRM_NEW_WORDING_PROBES_PER_PERSON),
            ("new_wording_probes_total",
             CONFIRM_NEW_WORDING_PROBES_PER_PERSON
             * words.get("target_persons", 0))):
        if words.get(field) != want:
            refusals.append(
                f"the power report's selected confirmation size sets "
                f"{field} = {words.get(field)!r} but the module declares "
                f"{want!r}")

    # ---- checkpoints, bound by full adapter contract ----
    selected_ids = dict(selection["selected"])
    by_ckpt = {r["checkpoint_id"]: r for r in manifest["predictions"]}
    checkpoints: dict[str, Any] = {}
    for state in CONFIRMATION_STATES + ("MF",):
        ckpt_id = (selected_ids[state] if state in selected_ids
                   else state)          # reference states are their own id
        adapter_dir = resolve_adapter_dir(ckpt_id, repo_root, tag)
        contract = adapter_contract(adapter_dir)
        recorded = by_ckpt.get(ckpt_id, {})
        recorded_contract = recorded.get("adapter_contract_sha256")
        entry: dict[str, Any] = {
            "checkpoint_id": ckpt_id,
            "adapter_dir": (str(adapter_dir.relative_to(repo_root))
                            if adapter_dir and
                            adapter_dir.is_relative_to(repo_root)
                            else str(adapter_dir) if adapter_dir else None),
            "adapter_contract": contract,
            "base_model_revision": recorded.get("base_model_revision"),
            "role": STATE_ROLE.get(state, EXCLUDED_STATES.get(state)),
        }
        # The freeze must bind the SAME bytes 11R scored, not merely some
        # adapter that happens to be sitting in the expected directory.
        #
        # The ORDER of these checks is load-bearing.  An ABSENT adapter
        # directory still produces a contract: no files hashed, both listed
        # as missing, and a roll-up sha256 that is an ordinary-looking hash
        # of the empty map.  Comparing that against the sidecars first would
        # report "the checkpoint has moved since the results being confirmed
        # were produced" — a confident diagnosis of the wrong thing, in the
        # one environment (a fresh clone, where the adapters are gitignored)
        # where it is most likely to be read.
        if contract is None:
            refusals.append(f"{state}: no adapter contract could be resolved "
                            f"for checkpoint_id {ckpt_id}")
        elif contract.get("missing_files"):
            refusals.append(
                f"{state}: adapter contract is missing "
                f"{contract['missing_files']} under {entry['adapter_dir']}; "
                f"the adapters are gitignored, so a checkout without them "
                f"cannot be frozen, and this is NOT evidence that the "
                f"checkpoint moved")
        elif recorded_contract and contract["sha256"] != recorded_contract:
            refusals.append(
                f"{state}: adapter contract {contract['sha256'][:16]} does "
                f"not match the {recorded_contract[:16]} the 11R sidecars "
                f"recorded for {ckpt_id}; the checkpoint has moved since the "
                f"results being confirmed were produced")
        cand = selection["candidates"].get(ckpt_id)
        if cand:
            cfg = cand["config"]
            recipe = dict(cfg.get("recipe") or {})
            entry["recipe"] = recipe
            entry["num_optimizer_steps"] = cfg.get("num_optimizer_steps")
            entry["init_adapter_dir"] = cfg.get("init_adapter_dir")
            entry["unlearning_groups"] = cfg.get("groups")
            entry["noop"] = cfg.get("noop", False)
            entry["distance_to_mg_on_the_exploratory_split"] = \
                cand.get("distance_to_mg")
            if state == "B3":
                fine = next((g for g in (cfg.get("groups") or [])
                             if g.get("name") == "fine_target"), None)
                entry["granularity_lambda"] = (fine or {}).get("weight")
                if entry["granularity_lambda"] is None:
                    refusals.append(
                        "B3: no fine_target group weight in the selection "
                        "report, so the lambda being confirmed is unknown")
        checkpoints[state] = entry

    # ---- is B0 really the no-op copy of MF? ----
    # Measured, not asserted.  Excluding MF from the confirmation is only
    # justified if B0's adapter contract is byte-identical to MF's; 11R
    # proved the two produce identical OUTPUTS on the exploratory test split,
    # but that is a consequence of identical weights under an identical batch
    # layout, and the confirmation gets a new split.
    b0c = checkpoints["B0"]["adapter_contract"] or {}
    mfc = checkpoints["MF"]["adapter_contract"] or {}
    b0, mf = b0c.get("sha256"), mfc.get("sha256")
    # Two contracts that hashed NO files roll up to the same sha256, so
    # without this guard a checkout holding no adapters at all would "prove"
    # B0 is a byte-identical copy of M_F and exclude M_F on the strength of
    # nothing.  Measured equality of two empty maps is not a measurement.
    hashed_no_files = sorted(n for n, c in (("B0", b0c), ("MF", mfc))
                             if not c.get("files"))
    both_complete = not hashed_no_files
    b0_is_noop = both_complete and b0 == mf
    if not both_complete:
        refusals.append(
            f"the B0/M_F no-op check could not be performed: "
            f"{hashed_no_files} hashed no adapter files at all, and an empty "
            f"file map compares equal to every other empty file map, so "
            f"absence would otherwise count as proof that B0 is the no-op "
            f"copy of M_F")
    elif not b0_is_noop:
        refusals.append(
            f"B0's adapter contract ({str(b0)[:16]}) is not identical to "
            f"MF's ({str(mf)[:16]}), so MF is a distinct state and cannot be "
            f"excluded from the confirmation on the strength of B0")
    checkpoints.pop("MF", None)

    # ---- generation settings, from the committed manifest ----
    # Every 11R pass recorded its own generation_config; the confirmation
    # must use the same one, and the module defaults must still agree with it
    # or the fingerprint the sidecars bind has drifted from the generator.
    gen_records = {json.dumps(r["generation_config"], sort_keys=True)
                   for r in manifest["predictions"]}
    if len(gen_records) != 1:
        refusals.append(
            f"the 30 committed predictions were not all generated under one "
            f"configuration ({len(gen_records)} distinct); a confirmation "
            f"cannot inherit an ambiguous layout")
    generation = json.loads(sorted(gen_records)[0]) if gen_records else {}
    defaults = {
        "max_new_tokens": DEFAULT_MAX_NEW_TOKENS,
        "max_length": DEFAULT_MAX_LENGTH,
        "max_image_pixels": DEFAULT_MAX_IMAGE_PIXELS,
    }
    mismatches = {k: (generation.get(k), v)
                  for k, v in defaults.items() if generation.get(k) != v}
    if mismatches:
        refusals.append(
            f"reference_eval defaults no longer match the generation config "
            f"the 11R evidence was produced under: {mismatches}")

    # ---- delta = 0.05 and the role the 11C-2a decisions gave it ----
    # The freeze preserves the margin, but not as a tested threshold: both
    # margin claims were demoted, so nothing runs a TOST or a
    # non-inferiority test against it.  Recording it as "the margin" without
    # saying so would imply a test that will never be run.
    ret = power["retention_claim_decision"]
    mgd = power["mg_equivalence_decision"]
    margin = {
        "value": EQUIVALENCE_MARGIN_TGA,
        "read_from": "evaluate_pilot100_final.EQUIVALENCE_MARGIN_TGA",
        "role": "REPORTING YARDSTICK, not a tested margin",
        "what_it_is_used_for": (
            "the published half-widths are read against it and the reports "
            "state whether an interval could have concluded equivalence at "
            "it; no TOST and no non-inferiority test is run against it on the "
            "confirmation split"),
        "why": (
            "both margin claims were demoted by the 11C-2a decisions, so "
            "there is no hypothesis test left for this margin to serve"),
        "retention_decision": ret["decision"],
        "retention_declared_margin": ret["declared_margin"],
        "mg_decision": mgd["decision"],
        "mg_equivalence_test_run": mgd["equivalence_test_run"],
        "margin_rationale": final_eval.get("equivalence_vs_MG", {}).get(
            "margin_rationale") if isinstance(
                final_eval.get("equivalence_vs_MG"), dict) else None,
    }

    code = code_fingerprint(repo_root)
    analysis_scripts = {
        rel: sha256_file(repo_root / rel) for rel in ANALYSIS_SCRIPTS
        if (repo_root / rel).exists()}
    missing_scripts = [rel for rel in ANALYSIS_SCRIPTS
                       if rel not in analysis_scripts]
    if missing_scripts:
        refusals.append(f"analysis scripts absent: {missing_scripts}")

    freeze: dict[str, Any] = {
        "freeze": "iteration_11c_confirmation_protocol",
        "frozen_at_utc": _utcnow(),
        "tag": tag,
        "iteration": "11C",
        "provenance_contract_version": PROVENANCE_CONTRACT_VERSION,
        "git_commit": code.get("git_commit"),
        "git_dirty": code.get("git_dirty"),
        "purpose": (
            "Bind every parameter the confirmatory claims depend on before "
            "any confirmation probe is built or scored, so that a result "
            "produced afterwards cannot be explained by a parameter that "
            "moved beforehand."),
        "inputs_bound": {
            "power_analysis": _bound(
                reports / f"mllmu_{tag}_confirmation_power.json", repo_root),
            "selection": _bound(
                reports / f"mllmu_{tag}_unlearning_selection.json", repo_root),
            "final_evaluation": _bound(
                reports / f"mllmu_{tag}_final_evaluation.json", repo_root),
            "prediction_manifest": _bound(
                reports / f"mllmu_{tag}_prediction_manifest.json", repo_root),
        },
        "checkpoints": checkpoints,
        "b0_is_the_no_op": {
            "measured": True,
            "b0_adapter_contract_sha256": b0,
            "mf_adapter_contract_sha256": mf,
            "both_contracts_hashed_real_files": both_complete,
            "identical": b0_is_noop,
            "how": "adapter_contract() over weights AND adapter_config.json",
            "consequence": (
                "MF is excluded from the confirmation because B0 is a "
                "byte-identical copy of it, measured here rather than "
                "inherited from 11R's output comparison"),
            "why_completeness_is_part_of_the_measurement": (
                "an adapter directory that is absent hashes zero files, and "
                "the roll-up of an empty file map equals the roll-up of every "
                "other empty file map; equality of two such contracts would "
                "look like a measurement and be nothing"),
        },
        "states": {
            "scored": list(CONFIRMATION_STATES),
            "scored_because": dict(STATE_ROLE),
            "excluded": dict(EXCLUDED_STATES),
        },
        "generation": {
            **generation,
            "source": ("the single generation_config shared by all 30 "
                       "committed 11R predictions"),
            "module_defaults_cross_check": defaults,
            "module_defaults_match": not mismatches,
            "base_model_revision": next(
                (r["base_model_revision"] for r in manifest["predictions"]
                 if r.get("base_model_revision")), None),
            "batch_layout": (
                "ONE uniform layout for every scored state: all confirmation "
                "queries in a single pass, same batch_size and "
                "image_batch_size for B3, B0 and MG. Batched greedy decoding "
                "is not bit-stable across batch compositions, so two layouts "
                "would put decoding noise inside the paired difference "
                "itself."),
        },
        "scorer": {
            "modules_sha256": {
                rel: h for rel, h in (code.get("modules_sha256") or {}).items()
                if rel.endswith(("scoring.py", "hierarchy_metrics.py"))},
            "equivalence_margin": margin,
        },
        "analysis": {
            "ci_unit": ("entity - the iNaturalist species or the MLLMU "
                        "person - which is the unit the bootstrap resamples"),
            "ci_method": "paired percentile bootstrap",
            "averaging_units": {
                "entity_macro": (
                    "the statistic the CI is built for and the one the "
                    "interval covers"),
                "row_micro": (
                    "the published hierarchy_metrics rate; the two differ "
                    "because entities contribute unequal numbers of probes, "
                    "so subtracting two published rates does not reproduce "
                    "the CI's centre"),
            },
            "bootstrap": _ci_parameters(),
            "icc_bootstrap_seed": BOOTSTRAP_SEED,
            "paired_metrics": list(PAIRED_METRICS),
            "familywise_alpha": FAMILYWISE_ALPHA,
            "alpha_one_sided": ALPHA_ONE_SIDED,
            "alpha_convention": (
                f"familywise alpha is {FAMILYWISE_ALPHA}, applied to "
                f"ONE-SIDED p-values; a single unadjusted claim sits at "
                f"{ALPHA_ONE_SIDED} = familywise/2 and Holm's worst case for "
                f"one of {len(PRIMARY_FAMILY)} claims is familywise/k = "
                f"{FAMILYWISE_ALPHA / len(PRIMARY_FAMILY)}. Every sizing "
                "function in the power module takes a one-sided alpha and "
                "uses z(1 - alpha), so the declared threshold and the "
                "reported cluster requirement are the same test."),
        },
        "primary_estimand": {
            "declared": PRIMARY_ESTIMAND,
            "definition": estimand.get("definition"),
            "unit_of_inference": estimand.get("unit_of_inference"),
            "entities_in_scope": estimand.get("entities_in_scope"),
            "entities_in_scope_by_source":
                estimand.get("entities_in_scope_by_source"),
            "entity_ids_sha256": estimand.get("entity_ids_sha256"),
            "strata_whose_probes_enter_it": list(PRIMARY_ESTIMAND_STRATA),
            "per_stratum_decomposition_status": STRATUM_ESTIMAND_STATUS,
            "what_it_establishes": estimand.get("what_it_establishes"),
            "what_it_does_not_establish":
                WHAT_THE_CONFIRMATION_DOES_NOT_ESTABLISH,
            "read_from": "primary_estimand in the power report bound above; "
                         "the declaration itself is the module constant, so "
                         "a disagreement refuses rather than picks one",
        },
        "confirmation_size": {
            "selected": size.get("selected"),
            "held_out_photographs": photos,
            "new_wording_probes": words,
            "totals": size.get("totals"),
            "frozen_now": size.get("frozen_now"),
            "frozen_at_stage_3_before_any_scoring":
                size.get("frozen_at_stage_3_before_any_scoring"),
            "read_from": "confirmation_size in the power report bound above, "
                         "cross-checked against the CONFIRM_* module "
                         "constants in build_freeze",
            "why_the_size_is_in_the_freeze": (
                "stage 3 builds probes and stage 5 scores them once. If the "
                "size were only in the power report, a builder could pick a "
                "different row of the grid and nothing would notice until "
                "after scoring; bound here, the split that gets built has to "
                "match a number committed before any inference ran."),
        },
        "claims": {
            "primary_family": list(PRIMARY_FAMILY),
            "primary_estimand": PRIMARY_ESTIMAND,
            "primary_claim_kinds": {
                n: CLAIM_KIND[n.split(":")[1]] for n in PRIMARY_FAMILY},
            "k": len(PRIMARY_FAMILY),
            "multiplicity": "Holm",
            "familywise_alpha": FAMILYWISE_ALPHA,
            "holm_thresholds": [
                round(FAMILYWISE_ALPHA / (len(PRIMARY_FAMILY) - i), 6)
                for i in range(len(PRIMARY_FAMILY))],
            "thresholds_apply_to": "one-sided p-values",
            "worst_case_alpha_for_a_single_claim": round(
                FAMILYWISE_ALPHA / len(PRIMARY_FAMILY), 6),
            "sizing_alpha_one_sided": holm.get("sizing_alpha_one_sided"),
            "sizing_critical_value_z": holm.get("critical_value_z"),
            "thresholds_and_sizing_agree":
                holm.get("thresholds_and_sizing_agree"),
            "clusters_required_at_that_threshold": {
                n: v["n_at_holm_worst_case_alpha_power80"]
                for n, v in (holm.get("per_claim") or {}).items()},
            "cluster_ceiling_of_the_primary_claims": (
                power.get("feasibility", {}).get("ceilings", {})
                .get("entity_clusters_available_for_pooled_target_claims")),
            "descriptive_and_why": {
                "B3_minus_B0:retain_same": ret["decision"],
                "B3_minus_B0:retain_other": ret["decision"],
                "B3_minus_MG:tga": mgd["decision"],
                "B3_minus_MG:filr": mgd["decision"],
            },
            "retention_justification_is_ceiling_independent": (
                ret.get("sensitivity_to_the_ceiling_choice", {})
                .get("delta0.05_infeasible_under_every_candidate")),
            "mg_claim_kinds": dict(CLAIM_KIND_VS_MG),
            "evidence": "preregistration_decisions in the power report bound "
                        "above",
        },
        "code": {
            "fingerprinted_modules": code.get("modules_sha256"),
            "fingerprinted_module_list": list(CODE_FINGERPRINT_MODULES),
            "analysis_scripts_sha256": analysis_scripts,
            "why_analysis_scripts_are_bound_separately": (
                "CODE_FINGERPRINT_MODULES binds the modules that turn queries "
                "into scores. These turn scores into claims, and an edit to "
                "either would change what the frozen protocol means without "
                "changing a single prediction - so the sidecars would still "
                "verify while the conclusion moved."),
        },
        "exploratory_dataset": {
            **dataset_fingerprint(data_dir, repo_root),
            "role": (
                "the dataset the checkpoints were SELECTED on. It is bound "
                "here to prove what the confirmation is not: the "
                "confirmation split is a different dataset whose own hashes "
                "are committed at stage 4, and no query in it may overlap "
                "this one."),
            "selection_scope": selection["selection_scope"],
            "selection_basis": selection["basis"],
        },
        "sealed_split_invariants": [
            {
                "invariant": "no confirmation query_id, association, "
                             "paraphrase template or photograph appears in "
                             "the exploratory dataset bound above",
                "why": "the reference-state gate inspected the exploratory "
                       "test split before candidate selection, which is why "
                       "those results stay exploratory; a confirmation probe "
                       "reused from that split inherits the exposure",
                "enforced_by": "the stage-3 probe builder, by hash and by "
                               "query_id, before the split is written",
                "checked_at": "stage 3, before the stage-4 tag",
            },
            {
                "invariant": "the confirmation split never enters the "
                             "reference-state gate",
                "why": "the gate is a go/no-go on the oracle, and letting it "
                       "see confirmation data would make the confirmation "
                       "exploratory in the same way",
                "enforced_by": "evaluate_reference_states.py is pointed at "
                               "the exploratory data_dir only; the freeze "
                               "records that data_dir here",
                "checked_at": "stage 5, before scoring",
            },
            {
                "invariant": "the confirmation split never enters candidate "
                             "selection or any other go/no-go decision",
                "why": "selection is already complete and frozen; re-running "
                       "it on confirmation data would be selecting on the "
                       "test set",
                "enforced_by": "select_unlearning_checkpoints.py keeps "
                               "selection_scope train+val on the "
                               "exploratory dataset; no confirmation artifact "
                               "is an input to it",
                "checked_at": "stage 5, before scoring",
            },
            {
                "invariant": "partial state results are not inspected before "
                             "every scored state has finished",
                "why": "looking at B3 before MG has run is how a layout or "
                       "scoring change gets made mid-flight and the "
                       "score-exactly-once property is lost",
                "enforced_by": "the stage-5 runner assembles only after all "
                               "states' sidecars verify; crashes resume "
                               "through verified sidecars alone",
                "checked_at": "stage 5",
            },
            {
                "invariant": "a failed primary claim is reported as failed",
                "why": "retuning on confirmation data converts a "
                       "confirmatory result back into an exploratory one",
                "enforced_by": "this freeze, plus the analysis-script hashes "
                               "bound above: a retune changes those hashes",
                "checked_at": "stage 5, at reporting",
            },
            {
                "invariant": f"the split stage 3 builds is the size frozen "
                             f"here: "
                             f"{CONFIRM_NEW_WORDING_PROBES_PER_PERSON} new "
                             f"wording probes per target person and "
                             f"{CONFIRM_NEW_PHOTOS_PER_SPECIES} new "
                             f"photographs per species, fetched by the named "
                             f"command line and no other",
                "why": "the size was selected from a grid before any "
                       "confirmation probe existed. Building a different row "
                       "of that grid would make the reported power and mde "
                       "describe a design that was not run, and choosing the "
                       "row after seeing the data is the same error as "
                       "choosing a margin after seeing that the declared one "
                       "cannot be reached",
                "enforced_by": "confirmation_size above, cross-checked "
                               "against the CONFIRM_* module constants at "
                               "freeze time and against the built split's "
                               "committed query and photograph hashes at "
                               "stage 4",
                "checked_at": "stage 4, before scoring",
            },
        ],
        "environment": environment_fingerprint(),
        "refusals": refusals,
    }
    return freeze


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Freeze the Iteration 11C confirmation protocol")
    ap.add_argument("--tag", default="pilot100")
    ap.add_argument("--output", default=None)
    ap.add_argument("--check-only", action="store_true",
                    help="re-derive and compare against the committed freeze "
                         "instead of writing one")
    ap.add_argument("--allow-refreeze", action="store_true",
                    help="overwrite an existing freeze; refused by default "
                         "because a protocol that can be silently re-frozen "
                         "after inference is not a protocol")
    args = ap.parse_args()

    repo_root = _find_repo_root(Path.cwd()) or Path.cwd()
    out = Path(args.output or repo_root / "data" / "reports"
               / f"mllmu_{args.tag}_confirmation_freeze.json")

    freeze = build_freeze(repo_root, args.tag)

    if freeze["refusals"]:
        print("REFUSED — the protocol cannot be frozen:")
        for r in freeze["refusals"]:
            print(f"  * {r}")
        return 1

    # Volatile fields that legitimately differ between two derivations; see
    # VOLATILE_FREEZE_FIELDS for why each is excluded.
    volatile = VOLATILE_FREEZE_FIELDS

    if args.check_only:
        if not out.exists():
            print(f"REFUSED — no freeze to check at {out}")
            return 1
        committed = json.loads(out.read_text())
        diffs = []
        for k in sorted(set(committed) | set(freeze)):
            if k in volatile:
                continue
            a, b = committed.get(k), freeze.get(k)
            if json.dumps(a, sort_keys=True) != json.dumps(b, sort_keys=True):
                diffs.append(k)
        if diffs:
            print("DRIFT — the freeze no longer matches the repository:")
            for k in diffs:
                print(f"  * {k}")
            return 1
        print(f"OK — the freeze at {out.name} still matches the repository "
              f"(volatile fields excluded: {sorted(volatile)})")
        return 0

    if out.exists() and not args.allow_refreeze:
        print(f"REFUSED — {out.name} already exists. A protocol freeze is "
              f"written once, before inference; pass --allow-refreeze only "
              f"if the confirmation has not been scored, and record why.")
        return 1

    out.write_text(json.dumps(freeze, indent=2, sort_keys=True))
    log.info("wrote %s", out)

    print(f"\nfroze the Iteration 11C confirmation protocol -> {out}")
    print(f"  git commit   {freeze['git_commit']}"
          f"{' (dirty)' if freeze['git_dirty'] else ''}")
    print(f"  states       {', '.join(freeze['states']['scored'])}")
    print(f"  B0 is MF     {freeze['b0_is_the_no_op']['identical']} "
          f"(measured by adapter contract)")
    g = freeze["generation"]
    print(f"  generation   batch={g.get('batch_size')} "
          f"image_batch={g.get('image_batch_size')} "
          f"max_new_tokens={g.get('max_new_tokens')} "
          f"do_sample={g.get('do_sample')}")
    print(f"  base model   {g.get('base_model_revision')}")
    b3 = freeze["checkpoints"]["B3"]
    print(f"  B3           lambda={b3.get('granularity_lambda')} "
          f"steps={b3.get('num_optimizer_steps')} "
          f"contract={str((b3.get('adapter_contract') or {}).get('sha256'))[:16]}")
    m = freeze["scorer"]["equivalence_margin"]
    print(f"  delta={m['value']} role: {m['role']}")
    a = freeze["analysis"]
    print(f"  CI           {a['ci_method']} over {a['ci_unit']}")
    print(f"  bootstrap    n={a['bootstrap']['n_bootstrap']} "
          f"level={a['bootstrap']['ci_level']} seed={a['bootstrap']['seed']}"
          f" | ICC seed={a['icc_bootstrap_seed']}")
    c = freeze["claims"]
    print(f"  estimand     {freeze['primary_estimand']['declared']} over "
          f"{freeze['primary_estimand']['entities_in_scope']} entities "
          f"(ids {str(freeze['primary_estimand']['entity_ids_sha256'])[:16]})")
    print(f"  primary      {', '.join(c['primary_family'])} "
          f"({c['multiplicity']} k={c['k']}, familywise alpha "
          f"{c['familywise_alpha']}, worst-case one-sided "
          f"{c['worst_case_alpha_for_a_single_claim']})")
    print(f"  sized at     one-sided alpha {c['sizing_alpha_one_sided']}, "
          f"z={c['sizing_critical_value_z']} — the same test the threshold "
          f"declares, applied to {c['thresholds_apply_to']}")
    print(f"  needs        {c['clusters_required_at_that_threshold']} "
          f"clusters at that threshold, against a ceiling of "
          f"{c['cluster_ceiling_of_the_primary_claims']}")
    cs = freeze["confirmation_size"]
    print(f"  size         {cs['new_wording_probes']['new_wording_probes_total']}"
          f" new wording probes + "
          f"{cs['held_out_photographs']['new_photographs_total']} new "
          f"photographs = {cs['totals']['new_target_probes']} target probes")
    print(f"  fetch        {cs['held_out_photographs']['fetch_command']}")
    print(f"  invariants   {len(freeze['sealed_split_invariants'])} sealed-split"
          f" / score-once assertions recorded")
    return 0


if __name__ == "__main__":
    sys.exit(main())
