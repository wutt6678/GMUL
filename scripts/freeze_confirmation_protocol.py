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
    PRIMARY_FAMILY,
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
        if contract is None:
            refusals.append(f"{state}: no adapter contract could be resolved "
                            f"for checkpoint_id {ckpt_id}")
        elif recorded_contract and contract["sha256"] != recorded_contract:
            refusals.append(
                f"{state}: adapter contract {contract['sha256'][:16]} does "
                f"not match the {recorded_contract[:16]} the 11R sidecars "
                f"recorded for {ckpt_id}; the checkpoint has moved since the "
                f"results being confirmed were produced")
        elif contract.get("missing_files"):
            refusals.append(
                f"{state}: adapter contract is missing "
                f"{contract['missing_files']}")
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
    b0 = (checkpoints["B0"]["adapter_contract"] or {}).get("sha256")
    mf = (checkpoints["MF"]["adapter_contract"] or {}).get("sha256")
    b0_is_noop = bool(b0) and b0 == mf
    if not b0_is_noop:
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
            "identical": b0_is_noop,
            "how": "adapter_contract() over weights AND adapter_config.json",
            "consequence": (
                "MF is excluded from the confirmation because B0 is a "
                "byte-identical copy of it, measured here rather than "
                "inherited from 11R's output comparison"),
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
            "alpha_one_sided": ALPHA_ONE_SIDED,
        },
        "claims": {
            "primary_family": list(PRIMARY_FAMILY),
            "primary_claim_kinds": {
                n: CLAIM_KIND[n.split(":")[1]] for n in PRIMARY_FAMILY},
            "k": len(PRIMARY_FAMILY),
            "multiplicity": "Holm",
            "holm_thresholds": [
                round(2 * ALPHA_ONE_SIDED / (len(PRIMARY_FAMILY) - i), 6)
                for i in range(len(PRIMARY_FAMILY))],
            "worst_case_alpha_for_a_single_claim": round(
                2 * ALPHA_ONE_SIDED / len(PRIMARY_FAMILY), 6),
            "descriptive_and_why": {
                "B3_minus_B0:retain_same": ret["decision"],
                "B3_minus_B0:retain_other": ret["decision"],
                "B3_minus_MG:tga": mgd["decision"],
                "B3_minus_MG:filr": mgd["decision"],
            },
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
    print(f"  primary      {', '.join(c['primary_family'])} "
          f"({c['multiplicity']} k={c['k']}, worst-case alpha "
          f"{c['worst_case_alpha_for_a_single_claim']})")
    print(f"  invariants   {len(freeze['sealed_split_invariants'])} sealed-split"
          f" / score-once assertions recorded")
    return 0


if __name__ == "__main__":
    sys.exit(main())
