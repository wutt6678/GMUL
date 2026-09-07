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
  two-sided reading of the other (Iteration 11C-R1's finding #4), with the
  three levels that all look like 0.025 kept apart: familywise/2 is one arm
  of a two-sided interval, familywise/k is Holm's first threshold, and a
  STANDALONE directional claim is tested at familywise itself
  (11C-R2's finding #3);
* the PRIMARY TEST — statistic, null, p-value method, draw count, seed, the
  direction of each claim, Holm's ordering, tie handling, stopping rule and
  pass/fail rule, the achieved level it was measured to deliver, and a hash
  of the module that implements it, because a threshold with no statistic
  behind it cannot be executed as frozen (11C-R2's finding #3);
* the PRIMARY ESTIMAND, named once, with the per-stratum decomposition
  demoted to a pre-specified secondary and the fixed-cohort limit recorded,
  because "pooled" and "the stratum that carries the effect" are different
  quantities and a preregistration that names both claims neither
  (finding #3);
* the SELECTED confirmation size — ONE row of the power grid, the command
  line that fetches it, and the identifiers frozen now versus the ones stage
  3 is obliged to commit before scoring (finding #1), with the TARGET budget
  and the OVERALL allocated budget as separate numbers: 936 was published as
  the target total when it was the whole photograph budget over 36 species
  (11C-R2's finding #2);
* the RETENTION probe allocation — route, templates per entity, total, and
  the measured media supply that makes the image route impossible to renew,
  because descriptive intervals with no probes behind them describe the
  exploratory split rather than the confirmation one (11C-R2's finding #2);
* the sealed-split and score-exactly-once invariants, as assertions with a
  named enforcement point, because an invariant nobody is obliged to check
  is a comment.  The target-association set is REQUIRED to be identical and
  the queries, template ids and texts, and photograph hashes are required to
  be new: those are opposite requirements, and stating them as one rule
  forbade the confirmation from existing (11C-R2's finding #1).

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
import hashlib
import inspect
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from granunlearn.config import _find_repo_root
from granunlearn.evaluation.paired_ci import (
    CLAIM_DIRECTION,
    PAIRED_METRICS,
    holm_family,
    one_sided_permutation_pvalue,
    paired_rate_diff_ci,
)
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
    ALPHA_STANDALONE_ONE_SIDED,
    BOOTSTRAP_SEED,
    CLAIM_KIND,
    CLAIM_KIND_VS_MG,
    CONFIRM_FETCH_IMAGES_PER_SPECIES,
    CONFIRM_FETCH_OUT,
    CONFIRM_FETCH_ROLE,
    CONFIRM_FETCH_SEED,
    CONFIRM_FETCH_TAG,
    CONFIRM_NEW_PHOTOS_PER_SPECIES,
    CONFIRM_NEW_WORDING_PROBES_PER_PERSON,
    CONFIRM_RETENTION_NEW_TEMPLATES_PER_ENTITY,
    CONFIRM_RETENTION_ROUTE,
    CONFIRM_WORDING_FAMILIES,
    EXPLORATORY_IMAGE_MANIFEST,
    FAMILYWISE_ALPHA,
    HOLM_WORST_CASE_ALPHA,
    N_PERMUTATIONS,
    PERMUTATION_SEED,
    PHOTO_SELECTION_RULE,
    PRIMARY_ESTIMAND,
    PRIMARY_ESTIMAND_STRATA,
    PRIMARY_FAMILY,
    STRATUM_ESTIMAND_STATUS,
    WHAT_THE_CONFIRMATION_DOES_NOT_ESTABLISH,
    z,
)

#: The module holding the primary test.  It is deliberately NOT in
#: ``CODE_FINGERPRINT_MODULES``: adding it there would change the code
#: fingerprint inside all 30 committed 11R sidecars and refuse their reuse.
#: So it is hashed on its own and bound here, which is the only place the
#: procedure that decides the primary claims is pinned at all.
PAIRED_CI_MODULE = "src/granunlearn/evaluation/paired_ci.py"

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
    # Iteration 11C stage 3c.  The freeze binds PHOTO_SELECTION_RULE, but a
    # rule with no bound executor is a rule anyone can implement differently:
    # the script that turns the 24 drawn photographs per species into the 12
    # kept ones decides which photographs the confirmation actually scores, so
    # its bytes are bound beside the rule it executes.
    "scripts/select_confirmation_photographs.py",
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


def drift_keys(committed: dict[str, Any], fresh: dict[str, Any]) -> list[str]:
    """Top-level blocks that differ once :data:`VOLATILE_FREEZE_FIELDS` goes.

    ONE definition, used by ``--check-only`` and imported by the tests that
    assert what does and does not count as drift.  The same reasoning that
    made the exclusion set a module constant applies to the comparison: two
    copies is two chances for the tool and its tests to disagree about the
    exact thing the tests exist to check.
    """
    return [k for k in sorted(set(committed) | set(fresh))
            if k not in VOLATILE_FREEZE_FIELDS
            and json.dumps(committed.get(k), sort_keys=True)
            != json.dumps(fresh.get(k), sort_keys=True)]


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


def _primary_test_refusals(power: dict[str, Any], repo_root: Path) -> list[str]:
    """Refuse to freeze a primary test that is not fully specified.

    Iteration 11C-R2's finding #3: the protocol declared Holm thresholds and
    cluster requirements while the repository contained no one-sided p-value
    at all - ``paired_ci`` offered a 1000-resample percentile interval and
    nothing that returned a p.  A threshold with no statistic behind it is
    not a frozen test, and the first person to score would have had to invent
    one, after seeing the data.

    Every check here compares the committed report against the module and the
    code, so a specification that drifts from its own implementation refuses
    the freeze instead of being published beside it.
    """
    out: list[str] = []
    pt = power.get("primary_test")
    if not pt:
        return ["the power report has no primary_test block, so the "
                "statistic, null, p-value method, draw count, seed and "
                "directions of the primary claims are unfrozen"]
    agree = pt.get("specification_and_implementation_agree") or {}
    if not agree.get("all_agree"):
        bad = {kk: vv for kk, vv in agree.items()
               if vv is not True
               and kk != "every_value_read_off_the_implementation"}
        out.append(
            f"the power report's specification and its implementation "
            f"disagree: {json.dumps(bad)}")
    if pt.get("n_permutations") != N_PERMUTATIONS:
        out.append(
            f"the power report freezes {pt.get('n_permutations')} "
            f"permutations but the module declares {N_PERMUTATIONS}; the "
            "p-value grid, and so the smallest reportable p, differ")
    if pt.get("permutation_seed") != PERMUTATION_SEED:
        out.append(
            f"the power report freezes permutation seed "
            f"{pt.get('permutation_seed')} but the module declares "
            f"{PERMUTATION_SEED}")
    directions = pt.get("direction_by_metric") or {}
    want_directions = {c.split(":")[1]: CLAIM_DIRECTION[c.split(":")[1]]
                       for c in PRIMARY_FAMILY}
    if directions != want_directions:
        out.append(
            f"the power report freezes claim directions {directions} but the "
            f"module declares {want_directions}")
    #: Checked on BOTH sides.  The module's map is what a scorer will read and
    #: the report's is what the freeze publishes, and only checking the
    #: published one would let a scorer run a different test than the frozen
    #: description of it.
    for label, dirs in (("the module declares", dict(CLAIM_DIRECTION)),
                        ("the power report freezes", directions)):
        if sorted(dirs.values()) != ["greater", "less"]:
            out.append(
                f"{label} directions {dirs}, which are not one 'greater' and "
                "one 'less'. TGA is an accuracy and FILR a leakage rate, so "
                "the two primary claims point in OPPOSITE directions on the "
                "same B3-B0 difference, and one shared sign convention would "
                "test one of them backwards")
    if sorted(CLAIM_DIRECTION) != sorted(
            c.split(":")[1] for c in PRIMARY_FAMILY):
        out.append(
            f"CLAIM_DIRECTION covers {sorted(CLAIM_DIRECTION)} but the "
            f"primary family's metrics are "
            f"{sorted(c.split(':')[1] for c in PRIMARY_FAMILY)}; a claim "
            "with no declared direction would be tested in whichever "
            "direction the default happens to be")
    mult = pt.get("multiplicity") or {}
    k_fam = len(PRIMARY_FAMILY)
    want_thresholds = [round(FAMILYWISE_ALPHA / (k_fam - i), 6)
                       for i in range(k_fam)]
    if mult.get("thresholds") != want_thresholds:
        out.append(
            f"the power report freezes Holm thresholds "
            f"{mult.get('thresholds')} but familywise/{k_fam} step-down for "
            f"k = {k_fam} gives {want_thresholds}")
    if mult.get("worst_case_alpha_for_a_single_claim") != round(
            HOLM_WORST_CASE_ALPHA, 6):
        out.append(
            f"the frozen worst-case single-claim alpha "
            f"{mult.get('worst_case_alpha_for_a_single_claim')} is not "
            f"familywise/k = {round(HOLM_WORST_CASE_ALPHA, 6)}")
    for field in ("ordering", "tie_handling", "pass_fail_rule",
                  "why_the_stopping_rule_is_part_of_the_rule"):
        if not mult.get(field):
            out.append(f"the frozen Holm block does not state {field}, so "
                       "the procedure is not reproducible from the freeze")
    # The level the frozen test ACHIEVES, measured in the power report by
    # re-signing the real differences.  Freezing a threshold the procedure
    # overshoots would publish a familywise rate the test does not deliver.
    calib = pt.get("achieved_level_under_the_real_null") or {}
    if calib.get("estimable") is False:
        out.append(f"the achieved level of the frozen test could not be "
                   f"measured: {calib.get('reason')}")
    else:
        for claim in PRIMARY_FAMILY:
            entry = calib.get(claim) or {}
            if not entry.get("achieved_level_is_nominal_within_two_se"):
                out.append(
                    f"{claim}: the frozen test achieves a level of "
                    f"{entry.get('achieved_level_at_the_threshold')} at the "
                    f"{entry.get('threshold')} threshold, outside two Monte "
                    f"Carlo standard errors of nominal "
                    f"{entry.get('two_se_band_around_the_nominal')}, so the "
                    "declared threshold is not the threshold the procedure "
                    "delivers")
            if not entry.get("permutations_are_the_frozen_count"):
                out.append(
                    f"{claim}: the achieved level was calibrated at "
                    f"{entry.get('permutations_per_replicate')} draws, not "
                    f"the frozen {N_PERMUTATIONS}, which measures a "
                    "different procedure's level and reports it as this "
                    "test's")
        fw = pt.get("achieved_familywise_level_under_the_global_null") or {}
        if not fw.get("achieved_is_nominal_within_two_se"):
            out.append(
                f"the Holm procedure achieves a familywise rejection rate of "
                f"{fw.get('achieved')} against a declared "
                f"{fw.get('nominal')}, outside two Monte Carlo standard "
                f"errors {fw.get('two_se_band_around_the_nominal')}, so "
                "familywise alpha is not the rate the procedure delivers")
    # Hash the implementation here rather than trusting the report's copy: a
    # freeze that recorded a hash the report supplied would not notice the
    # module changing between the report and the freeze.
    impl = pt.get("implementation") or {}
    module = repo_root / PAIRED_CI_MODULE
    live = sha256_file(module) if module.exists() else None
    if not module.exists():
        out.append(f"{PAIRED_CI_MODULE} does not exist, so the primary test "
                   "has no implementation to freeze")
    elif impl.get("sha256") != live:
        out.append(
            f"the power report hashes {PAIRED_CI_MODULE} as "
            f"{impl.get('sha256')} but it is now {live}; the module holding "
            "the primary test changed after the analysis that specified it")
    if impl.get("module") != PAIRED_CI_MODULE:
        out.append(
            f"the power report names {impl.get('module')!r} as the primary "
            f"test's module but the freeze binds {PAIRED_CI_MODULE!r}")
    return out


def _retention_refusals(power: dict[str, Any]) -> list[str]:
    """Refuse to promise retention intervals with no probes behind them.

    Iteration 11C-R2's finding #2: the freeze published descriptive
    retain_same and retain_other intervals while the selected size allocated
    zero retention probes.  The 504 new wordings are TARGET-family probes on
    target persons, so those intervals would have described the exploratory
    split the reference-state gate had already seen.
    """
    out: list[str] = []
    ra = power.get("retention_probe_allocation")
    if not ra:
        return ["the power report has no retention_probe_allocation block, "
                "so the retention intervals the freeze promises have no "
                "allocated probes"]
    if ra.get("route") != CONFIRM_RETENTION_ROUTE:
        out.append(
            f"the power report allocates retention probes on the "
            f"{ra.get('route')!r} route but the module declares "
            f"{CONFIRM_RETENTION_ROUTE!r}")
    if ra.get("new_templates_per_entity") != \
            CONFIRM_RETENTION_NEW_TEMPLATES_PER_ENTITY:
        out.append(
            f"the power report allocates "
            f"{ra.get('new_templates_per_entity')} new retention templates "
            f"per entity but the module declares "
            f"{CONFIRM_RETENTION_NEW_TEMPLATES_PER_ENTITY}")
    per_metric = ra.get("per_metric") or {}
    for metric in ("retain_same", "retain_other"):
        entry = per_metric.get(metric) or {}
        if not entry.get("entities_carried_by"):
            out.append(f"no entity carries {metric}, so the descriptive "
                       f"{metric} interval the freeze promises cannot be "
                       "computed at all")
        elif entry.get("new_probes") != entry["entities_carried_by"] * \
                CONFIRM_RETENTION_NEW_TEMPLATES_PER_ENTITY:
            out.append(
                f"{metric}: the report allocates {entry.get('new_probes')} "
                f"retention probes for {entry['entities_carried_by']} "
                f"entities at "
                f"{CONFIRM_RETENTION_NEW_TEMPLATES_PER_ENTITY} templates "
                "each, which is not the same number")
        if entry.get("route") != CONFIRM_RETENTION_ROUTE:
            out.append(f"{metric}: allocated on the {entry.get('route')!r} "
                       f"route, not the declared "
                       f"{CONFIRM_RETENTION_ROUTE!r}")
    want_total = sum((per_metric.get(m) or {}).get("new_probes", 0)
                     for m in ("retain_same", "retain_other"))
    if ra.get("new_retention_probes_total") != want_total:
        out.append(
            f"the report totals {ra.get('new_retention_probes_total')} "
            f"retention probes but its own per-metric allocation sums to "
            f"{want_total}")
    # The counts the SIZE block publishes must be the same arithmetic, or the
    # totals block is describing a different design than the allocation.
    size = power.get("confirmation_size") or {}
    totals = size.get("totals") or {}
    photos = size.get("held_out_photographs") or {}
    words = size.get("new_wording_probes") or {}
    n_photos = photos.get("new_photographs_total", 0)
    n_words = words.get("new_wording_probes_total", 0)
    n_ret = ra.get("new_retention_probes_total", 0)
    if totals.get("new_target_probes") != n_words + n_photos:
        out.append(
            f"totals.new_target_probes is {totals.get('new_target_probes')} "
            f"but {n_words} new wordings plus {n_photos} new photographs is "
            f"{n_words + n_photos}")
    if totals.get("new_retention_probes") != n_ret:
        out.append(
            f"totals.new_retention_probes is "
            f"{totals.get('new_retention_probes')} but the allocation says "
            f"{n_ret}")
    if totals.get("total_new_probes_allocated") != n_words + n_photos + n_ret:
        out.append(
            f"totals.total_new_probes_allocated is "
            f"{totals.get('total_new_probes_allocated')} but the three "
            f"allocations sum to {n_words + n_photos + n_ret}")
    if totals.get("new_target_probes") == totals.get(
            "total_new_probes_allocated") and n_ret:
        out.append(
            "the target total and the overall allocated total are the same "
            f"number, {totals.get('new_target_probes')}, while "
            f"{n_ret} retention probes are allocated: one of the two labels "
            "is wrong, which is exactly the 936 mislabelling")
    return out


def _fetch_role_refusals(power: dict[str, Any]) -> list[str]:
    """Refuse a fetch command that would not fetch the target species.

    ``--limit-species 30`` reads like the right command and is not: the 6
    retain-only species are interleaved through ``SPECIES_LIST``, so the
    first 30 entries contain only 24 of the 30 target species.  The command
    is what a human runs at stage 3, so it is checked, not assumed.
    """
    out: list[str] = []
    photos = (power.get("confirmation_size") or {}).get(
        "held_out_photographs") or {}
    census = (power.get("entity_role_census") or {}).get("by_role") or {}
    target_inat = ((census.get("target") or {}).get("by_source") or {}).get(
        "inaturalist")
    if photos.get("species_covered") != target_inat:
        out.append(
            f"the selected size covers {photos.get('species_covered')} "
            f"species but the census counts {target_inat} TARGET iNaturalist "
            "species; the photograph budget is not for the entities the "
            "target metrics are defined on")
    if photos.get("fetch_role") != CONFIRM_FETCH_ROLE:
        out.append(
            f"the frozen fetch names role {photos.get('fetch_role')!r} but "
            f"the module declares {CONFIRM_FETCH_ROLE!r}")
    if photos.get("fetch_tag") != CONFIRM_FETCH_TAG:
        out.append(
            f"the frozen fetch names tag {photos.get('fetch_tag')!r} but the "
            f"module declares {CONFIRM_FETCH_TAG!r}")
    cmd = photos.get("fetch_command") or ""
    if f"--role {CONFIRM_FETCH_ROLE}" not in cmd:
        out.append(
            f"the frozen fetch command does not select species by role: "
            f"{cmd!r}. A --limit-species count would fetch the FIRST n "
            "entries of SPECIES_LIST, which contains the retain-only species "
            "and so misses target ones")
    if "--limit-species" in cmd:
        out.append(f"the frozen fetch command limits species by COUNT "
                   f"rather than by role: {cmd!r}")
    return out


def _photo_selection_refusals(power: dict[str, Any],
                              repo_root: Path) -> list[str]:
    """Refuse to freeze a photograph-novelty rule that cannot be executed.

    Iteration 11C stage 3.  The confirmation keeps 12 of the 24 photographs
    each species draws, and "which 12" is a choice -- so it has to be frozen
    before the fetch runs, not made afterwards over the pool that came back.

    An earlier revision specified it positionally: the seeded shuffle was
    claimed to nest, making the last 12 of the longer draw new BY
    CONSTRUCTION.  That is false for this fetch, for two reasons that are
    properties of ``fetch_inat_species.py`` and were measured against it.
    (1) ``rng`` is built once outside the species loop, so the state reaching
    a species depends on the pool lengths before it; ``pilot_v1`` walked 36
    species and ``--role target`` walks 30, so every species after the first
    dropped one draws a different order.  (2) ``chosen.sort`` re-orders the
    accepted set by ``(observation_id, photo_id)`` before files are named, so
    position on disk is not position in the draw either.  What replaces it is
    a hash rule against the exploratory image manifest -- which is why that
    manifest's own contents are hashed and bound here.
    """
    out: list[str] = []
    size = power.get("confirmation_size") or {}
    photos = size.get("held_out_photographs") or {}
    supply = (power.get("new_photograph_supply") or {}).get(
        "seeded_refetch") or {}
    rule = photos.get("selection_rule")
    if rule != PHOTO_SELECTION_RULE:
        out.append(
            f"the frozen photograph selection rule is not the rule the module "
            f"declares. Report: {rule!r}. Module: {PHOTO_SELECTION_RULE!r}")
    if supply.get("selection_rule") != rule:
        out.append(
            "the selection rule is stated twice and the two copies differ "
            "(confirmation_size.held_out_photographs vs "
            "new_photograph_supply.seeded_refetch); a builder would follow "
            "one and the freeze would certify the other")
    if isinstance(rule, str):
        if "sha256" not in rule:
            out.append(
                "the frozen selection rule does not name a content hash, so "
                "it cannot establish that a photograph is new")
        for positional in ("first 12", "last 12", "by construction",
                           "by position"):
            if positional in rule:
                out.append(
                    f"the frozen selection rule selects photographs "
                    f"POSITIONALLY ({positional!r} appears in {rule!r}). "
                    "Position carries no information here: the accepted set "
                    "is re-sorted by (observation_id, photo_id) and the draw "
                    "does not nest with pilot_v1's")
    n_short = photos.get("refuse_a_species_with_fewer_than_n_disjoint")
    if n_short != CONFIRM_NEW_PHOTOS_PER_SPECIES:
        out.append(
            f"the frozen rule would accept a species yielding "
            f"{n_short} disjoint photographs where the design needs "
            f"{CONFIRM_NEW_PHOTOS_PER_SPECIES}; a short species must refuse "
            "rather than be padded from the exploratory pool")
    withdrawn = supply.get("draw_nesting_is_not_relied_on") or {}
    for key in ("claim_an_earlier_revision_made",
                "refutation_1_rng_state_advances_across_species",
                "refutation_2_the_accepted_set_is_re_sorted",
                "what_is_true_and_why_it_is_not_enough"):
        if not withdrawn.get(key):
            out.append(
                f"the withdrawn shuffle-nesting claim is not recorded with its "
                f"refutation ({key!r} is missing), so nothing stops the next "
                "revision re-deriving novelty from the seed")
    frozen_now = size.get("frozen_now") or {}
    if frozen_now.get("exploratory_photograph_sha256_manifest") != \
            EXPLORATORY_IMAGE_MANIFEST:
        out.append(
            f"the novelty rule is disjointness against "
            f"{frozen_now.get('exploratory_photograph_sha256_manifest')!r} "
            f"but the module names {EXPLORATORY_IMAGE_MANIFEST!r}")
    recorded = frozen_now.get("exploratory_photograph_sha256_manifest_sha256")
    is_hex = isinstance(recorded, str) and len(recorded) == 64
    if is_hex:
        try:
            bytes.fromhex(recorded)
        except ValueError:
            is_hex = False
    path = repo_root / EXPLORATORY_IMAGE_MANIFEST
    live = sha256_file(path) if path.exists() else None
    if not is_hex:
        # Fail CLOSED.  A report generated where the manifest is absent
        # records a reason string instead of a hash, and freezing that would
        # certify a novelty rule with nothing to be disjoint from.
        out.append(
            f"the exploratory image manifest's own hash is not a sha256 "
            f"({recorded!r}), so the photograph novelty rule is bound to "
            "nothing")
    elif live is None:
        out.append(
            f"{EXPLORATORY_IMAGE_MANIFEST} is absent, so it cannot be shown "
            "that any confirmation photograph is disjoint from exploratory "
            "media")
    elif recorded != live:
        out.append(
            f"the exploratory image manifest has DRIFTED: the report binds "
            f"{recorded} but {EXPLORATORY_IMAGE_MANIFEST} now hashes to "
            f"{live}. The set of media the confirmation must avoid has "
            "changed since the rule was written")
    return out


def _portrait_exemption_refusals(power: dict[str, Any],
                                 repo_root: Path) -> list[str]:
    """Refuse to freeze a portrait exemption the measurement does not force.

    Iteration 11C stage 3.  The sealed rules required every confirmation
    photograph sha256 to be new and no exploratory media to be reused.  That
    is satisfiable for the 30 target species, whose new photographs were
    fetched, and unsatisfiable for the 42 target persons: each has exactly one
    portrait, the wording stratum is image-route by construction, and a
    text-route probe carries ``image_split = None`` so it sits in neither
    primary stratum.  As written the rules therefore forbade the frozen
    estimand -- the same shape of defect as 11C-R2's impossible
    association-disjointness invariant.

    The exemption is granted as a HASH LIST and not as a category, so what has
    to be refused is any exemption that is not exactly the measured one: a
    list that is not exploratory media, a list whose set hash does not match
    its own contents, or a list granted while the measurement that forces it
    has quietly stopped holding.
    """
    out: list[str] = []
    block = power.get("portrait_reuse_exemption") or {}
    if not block:
        out.append(
            "the power report carries no portrait_reuse_exemption, so the "
            "sealed-split novelty rules forbid the frozen primary estimand: "
            "the 504 wording probes are image-route over persons with one "
            "portrait each")
        return out
    portraits = block.get("portraits") or {}
    hashes = portraits.get("sha256") or []
    n = portraits.get("count", 0)
    words = (power.get("confirmation_size") or {}).get(
        "new_wording_probes") or {}
    persons = words.get("target_persons")
    if n != len(hashes):
        out.append(
            f"the portrait exemption counts {n} portraits but lists "
            f"{len(hashes)} hashes; an exemption whose size is not its own "
            "list has no boundary")
    if n != persons:
        out.append(
            f"the exemption covers {n} portraits but the wording stratum is "
            f"sized over {persons} target persons; the exemption must be "
            "exactly the persons the 504 probes sit on, or it exempts media "
            "no confirmation probe needs")
    if len(set(hashes)) != len(hashes):
        out.append("the portrait exemption lists a duplicate sha256")
    recomputed = hashlib.sha256("\n".join(hashes).encode()).hexdigest()
    if portraits.get("set_sha256") != recomputed:
        out.append(
            f"the portrait set hashes to {portraits.get('set_sha256')} but its "
            f"own list hashes to {recomputed}; the bound set hash does not "
            "describe the bound list")
    frozen_now = (power.get("confirmation_size") or {}).get(
        "frozen_now") or {}
    if frozen_now.get("required_repeat_portrait_set_sha256") != \
            portraits.get("set_sha256"):
        out.append(
            "confirmation_size.frozen_now and portrait_reuse_exemption bind "
            "different portrait set hashes, so the stage-3 builder and the "
            "sealed invariant would exempt different media")
    # An exemption must exempt something that IS exploratory media, checked
    # against the live manifest rather than against the report's own claim.
    manifest_path = repo_root / EXPLORATORY_IMAGE_MANIFEST
    exploratory: set[str] = set()
    if manifest_path.exists():
        exploratory = {
            (v or {}).get("sha256")
            for v in (json.loads(manifest_path.read_text())
                      .get("images") or {}).values()}
    not_exploratory = sorted(set(hashes) - exploratory)
    if exploratory and not_exploratory:
        out.append(
            f"{len(not_exploratory)} of the {len(hashes)} exempted portraits "
            f"are NOT in {EXPLORATORY_IMAGE_MANIFEST} "
            f"({', '.join(h[:12] for h in not_exploratory[:3])}...); an "
            "exemption for media the exploratory run never used exempts "
            "nothing and hides a mis-measured list")
    # The three measurements that force the exemption.  If any stops holding,
    # the exemption is no longer necessary and must not stay granted.
    for key, want in (
            ("wording_stratum_is_entirely_image_route", True),
            ("every_target_person_has_exactly_one_photograph", True),
            ("text_route_probes_are_in_neither_primary_stratum", True)):
        if block.get(key) is not want:
            out.append(
                f"the portrait exemption rests on {key}={want} but the "
                f"measurement says {block.get(key)!r}; if that changed the "
                "exemption is no longer forced and the novelty rule should "
                "apply in full again")
    if block.get("wording_stratum_num_families") != CONFIRM_WORDING_FAMILIES:
        out.append(
            f"the exploratory wording stratum uses "
            f"{block.get('wording_stratum_num_families')} probe families but "
            f"the design mirrors {CONFIRM_WORDING_FAMILIES}; the confirmation "
            "would not be re-measuring the stratum it is sized on")
    if block.get("wording_stratum_clusters") != persons:
        out.append(
            f"the exploratory wording stratum has "
            f"{block.get('wording_stratum_clusters')} clusters but the size "
            f"block budgets {persons} target persons")
    rules = ((power.get("confirmation_size") or {}).get(
        "frozen_at_stage_3_before_any_scoring") or {}).get(
        "collision_rules") or []
    if not any("bounded exception" in r for r in rules):
        out.append(
            "no collision_rule states the portrait exception, so the rule a "
            "stage-3 builder reads and the exemption the freeze grants "
            "contradict each other")
    return out


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

    # ---- the test itself, the retention allocation, and the fetch command --
    # Iteration 11C-R2: thresholds without a p-value, intervals without
    # probes, and a species count instead of a species ROLE are all things a
    # freeze can state and a scorer still be unable to execute.
    refusals.extend(_primary_test_refusals(power, repo_root))
    refusals.extend(_retention_refusals(power))
    refusals.extend(_fetch_role_refusals(power))
    refusals.extend(_photo_selection_refusals(power, repo_root))
    refusals.extend(_portrait_exemption_refusals(power, repo_root))

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
    pt = power.get("primary_test") or {}
    retention_alloc = power.get("retention_probe_allocation") or {}
    #: Recomputed here, not copied from the report: see
    #: ``primary_test.implementation.why_it_is_hashed_separately``.
    paired_ci_path = repo_root / PAIRED_CI_MODULE
    paired_ci_live = sha256_file(paired_ci_path) \
        if paired_ci_path.exists() else None

    # ---- what must be IDENTICAL and what must be NEW ----
    # Iteration 11C-R2's finding #1 needs both, with a hash on each side: an
    # invariant that binds ``None`` reads like a bound hash in the JSON and
    # is the same fail-open shape as an adapter contract hashed over an
    # absent directory, so a missing value refuses rather than freezes.
    frozen_now = size.get("frozen_now") or {}
    census_target = ((power.get("entity_role_census") or {})
                     .get("by_role") or {}).get("target") or {}
    target_assoc_sha = frozen_now.get("target_association_ids_sha256")
    n_target_assoc = len(frozen_now.get("target_association_ids") or [])
    target_entity_sha = frozen_now.get("target_entity_ids_sha256")
    n_target_entity = len(frozen_now.get("target_entity_ids") or [])
    expl_templates_sha = frozen_now.get("exploratory_template_ids_sha256")
    expl_templates_n = len(frozen_now.get("exploratory_template_ids") or [])
    #: The second REQUIRED repeat.  Refused rather than trusted if the
    #: measurement that forces it does not hold: see
    #: ``_portrait_exemption_refusals``.
    portrait_block = power.get("portrait_reuse_exemption") or {}
    portraits = portrait_block.get("portraits") or {}
    portrait_set_sha = portraits.get("set_sha256")
    n_portraits = portraits.get("count", 0)
    n_gate_saw = portrait_block.get(
        "portraits_the_exploratory_gate_exercised", 0)
    for label, value in (
            ("target_association_ids_sha256", target_assoc_sha),
            ("target_entity_ids_sha256", target_entity_sha),
            ("exploratory_template_ids_sha256", expl_templates_sha),
            ("required_repeat_portrait_set_sha256", portrait_set_sha)):
        if not value:
            refusals.append(
                f"the power report does not bind {label}, so the sealed-split "
                "invariants would be recorded against nothing")
    if not n_target_assoc:
        refusals.append(
            "the power report lists no target association_ids, so the "
            "requirement that the confirmation reproduce them cannot be "
            "checked at stage 3")
    if target_assoc_sha != census_target.get("association_ids_sha256"):
        refusals.append(
            f"confirmation_size.frozen_now hashes the target associations as "
            f"{target_assoc_sha} but entity_role_census hashes them as "
            f"{census_target.get('association_ids_sha256')}; the same set "
            "cannot have two hashes")
    if not expl_templates_n:
        refusals.append(
            "the power report lists no exploratory template ids, so the "
            "novelty invariant has nothing to be novel against")
    collision_rules = (size.get("frozen_at_stage_3_before_any_scoring")
                       or {}).get("collision_rules") or []
    if not collision_rules:
        refusals.append(
            "the power report binds no collision_rules, so stage 3 would "
            "decide for itself what may and may not repeat")
    if not frozen_now.get(
            "target_association_ids_are_required_to_be_identical"):
        refusals.append(
            "the power report does not state that the target-association set "
            "must be IDENTICAL, only that other things must be new; without "
            "the identity requirement the rule set forbids the confirmation "
            "from existing")
    if collision_rules and not any("IDENTICAL" in r for r in collision_rules):
        refusals.append(
            "none of the collision rules requires anything to be identical. "
            "A rule set made only of prohibitions is Iteration 11C-R2's "
            "finding #1: it forbids the confirmation from re-testing the "
            "associations the adapters unlearned, which is the only thing it "
            "is for")
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
            "alpha_standalone_one_sided": ALPHA_STANDALONE_ONE_SIDED,
            "holm_worst_case_alpha": HOLM_WORST_CASE_ALPHA,
            "alpha_convention": (
                f"familywise alpha is {FAMILYWISE_ALPHA}, applied to "
                f"ONE-SIDED p-values. One-sidedness does not halve anything: "
                f"a STANDALONE directional claim at this familywise rate is "
                f"tested at one-sided {ALPHA_STANDALONE_ONE_SIDED}. The "
                f"primary claims are a family of {len(PRIMARY_FAMILY)}, so "
                f"Holm charges the first of them familywise/k = "
                f"{HOLM_WORST_CASE_ALPHA}, and that is the level the sizing "
                f"uses. {ALPHA_ONE_SIDED} = familywise/2 is a SEPARATE "
                "quantity - the one arm of a two-sided 95% interval, i.e. the "
                "level a TOST arm or a non-inferiority bound is read off at - "
                "which happens to equal familywise/k here only because "
                f"k = {len(PRIMARY_FAMILY)}. At k = 3 Holm's first threshold "
                f"would be {round(FAMILYWISE_ALPHA / 3, 4)} while "
                f"familywise/2 stayed {ALPHA_ONE_SIDED}. Every sizing "
                "function in the power module takes a one-sided alpha and "
                "uses z(1 - alpha), so the declared threshold and the "
                "reported cluster requirement are the same test."),
        },
        "primary_test": {
            "applies_to": pt.get("applies_to"),
            "statistic": pt.get("statistic"),
            "why_the_statistic_must_be_the_one_the_interval_covers":
                pt.get(
                    "why_the_statistic_must_be_the_one_the_interval_covers"),
            "null_hypothesis": pt.get("null_hypothesis"),
            "alternative_by_claim": pt.get("alternative_by_claim"),
            "direction_by_metric": pt.get("direction_by_metric"),
            "why_the_directions_differ": pt.get("why_the_directions_differ"),
            "p_value_method": pt.get("p_value_method"),
            "why_sign_flips_and_not_the_bootstrap":
                pt.get("why_sign_flips_and_not_the_bootstrap"),
            "n_permutations": N_PERMUTATIONS,
            "permutation_seed": PERMUTATION_SEED,
            "smallest_reportable_p_value":
                pt.get("smallest_reportable_p_value"),
            "observed_vector_included_in_the_null":
                pt.get("observed_vector_included_in_the_null"),
            "multiplicity": {
                kk: vv for kk, vv in (pt.get("multiplicity") or {}).items()
                if kk != "worked_examples"},
            "worked_examples": (pt.get("multiplicity") or {}).get(
                "worked_examples"),
            "achieved_level_under_the_real_null":
                pt.get("achieved_level_under_the_real_null"),
            "achieved_familywise_level_under_the_global_null":
                pt.get("achieved_familywise_level_under_the_global_null"),
            "achieved_level_note": pt.get("achieved_level_note"),
            "interval_still_published": pt.get("interval_still_published"),
            "implementation": {
                "module": PAIRED_CI_MODULE,
                "sha256": paired_ci_live,
                "functions": (pt.get("implementation") or {}).get(
                    "functions"),
                "signatures": {
                    "one_sided_permutation_pvalue": list(
                        inspect.signature(
                            one_sided_permutation_pvalue).parameters),
                    "holm_family": list(
                        inspect.signature(holm_family).parameters),
                    "paired_rate_diff_ci": list(
                        inspect.signature(paired_rate_diff_ci).parameters),
                },
                "in_the_code_fingerprint":
                    PAIRED_CI_MODULE in CODE_FINGERPRINT_MODULES,
                "why_it_is_hashed_separately": (
                    "this module is NOT in CODE_FINGERPRINT_MODULES: adding "
                    "it would change the code fingerprint inside all 30 "
                    "committed 11R sidecars and refuse their reuse. Before "
                    "Iteration 11C-R2 it was bound nowhere at all, so the "
                    "procedure that decides the primary claims was the one "
                    "part of the analysis nothing pinned. The hash is "
                    "recomputed HERE rather than copied from the power "
                    "report, so a module edited between the report and the "
                    "freeze refuses instead of freezing the stale hash."),
            },
            "read_from": "primary_test in the power report bound above, with "
                         "the draw count, seed and directions taken from the "
                         "module constants and cross-checked against it; a "
                         "disagreement refuses rather than picks one",
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
            "probe_budget": {
                "new_wordings_on_target_persons":
                    words.get("new_wording_probes_total"),
                "new_photographs_on_target_species":
                    photos.get("new_photographs_total"),
                "primary_target_total":
                    (words.get("new_wording_probes_total") or 0)
                    + (photos.get("new_photographs_total") or 0),
                "new_retention_probes":
                    retention_alloc.get("new_retention_probes_total"),
                "total_allocated":
                    (size.get("totals") or {}).get(
                        "total_new_probes_allocated"),
                #: Taken from the report rather than restated here: the
                #: sentence names five numbers, and a copy of it in the
                #: freeze is a second place for one of them to go stale.
                "correction_to_the_previous_revision":
                    (size.get("totals") or {}).get(
                        "correction_to_the_previous_revision"),
                "why_the_two_totals_are_reported_separately": (
                    "the primary claims are computed over the target probes "
                    "alone, so a reader who takes the overall total for the "
                    "target total overstates the primary design by whatever "
                    "the descriptive probes cost"),
            },
        },
        "retention_probes": {
            "status": retention_alloc.get("status"),
            "route": CONFIRM_RETENTION_ROUTE,
            "new_templates_per_entity":
                CONFIRM_RETENTION_NEW_TEMPLATES_PER_ENTITY,
            "total": retention_alloc.get("new_retention_probes_total"),
            "entities_covered":
                retention_alloc.get("distinct_entities_covered"),
            "retain_other_entities_are_a_subset_of_retain_same":
                retention_alloc.get(
                    "retain_other_entities_are_a_subset_of_retain_same"),
            "per_metric": retention_alloc.get("per_metric"),
            "media_supply_the_route_decision_rests_on":
                retention_alloc.get(
                    "media_supply_the_route_decision_rests_on"),
            "why_the_image_route_is_omitted":
                retention_alloc.get("why_the_image_route_is_omitted"),
            "consequence_for_the_photograph_fetch":
                retention_alloc.get("consequence_for_the_photograph_fetch"),
            "what_the_confirmation_retention_estimand_is":
                retention_alloc.get(
                    "what_the_confirmation_retention_estimand_is"),
            "how_it_differs_from_the_exploratory_retention_number":
                retention_alloc.get(
                    "how_it_differs_from_the_exploratory_retention_number"),
            "batch_layout_noise_floor_on_retain_metrics":
                retention_alloc.get(
                    "batch_layout_noise_floor_on_retain_metrics"),
            "why_this_count": retention_alloc.get(
                "why_this_count_and_not_one_that_matches_the_exploratory_"
                "precision"),
            "read_from": "retention_probe_allocation in the power report "
                         "bound above, cross-checked against the "
                         "CONFIRM_RETENTION_* module constants",
            "why_this_is_in_the_freeze": (
                "the freeze has always promised descriptive retain_same and "
                "retain_other intervals. Until Iteration 11C-R2 it allocated "
                "no probes to compute them from, so the promise was "
                "unfalsifiable and the intervals would silently have "
                "described the exploratory split the reference-state gate "
                "had already seen. Bound here, the intervals have a named "
                "probe budget behind them and the omission of the image "
                "route is a recorded decision rather than an absence."),
        },
        "portrait_exemption": {
            "exempted": n_portraits,
            "set_sha256": portrait_set_sha,
            "sha256": portraits.get("sha256"),
            "paths": portraits.get("paths"),
            "image_ids": portraits.get("image_ids"),
            "target_persons": portrait_block.get("target_persons"),
            "target_person_ids": portrait_block.get("target_person_ids"),
            "photographs_per_target_person":
                portrait_block.get("photographs_per_target_person"),
            "the_exploratory_gate_exercised": n_gate_saw,
            "applies_to": (
                f"the {(size.get('new_wording_probes') or {}).get('new_wording_probes_total')} "
                "new wording probes on the target persons, and nothing else"),
            "does_not_apply_to": (
                f"the {(photos.get('new_photographs_total'))} new species "
                "photographs, every one of which must be hash-disjoint from "
                f"the exploratory manifest, or the "
                f"{retention_alloc.get('new_retention_probes_total')} "
                "retention probes, which are text-only and carry no image at "
                "all"),
            "what_is_still_new_on_an_exempted_probe":
                portrait_block.get("what_is_still_new_on_those_probes"),
            "measurement_it_rests_on": {
                "wording_stratum_probes":
                    portrait_block.get("wording_stratum_probes"),
                "wording_stratum_routes":
                    portrait_block.get("wording_stratum_routes"),
                "wording_stratum_is_entirely_image_route":
                    portrait_block.get(
                        "wording_stratum_is_entirely_image_route"),
                "wording_stratum_families":
                    portrait_block.get("wording_stratum_families"),
                "every_target_person_has_exactly_one_photograph":
                    portrait_block.get(
                        "every_target_person_has_exactly_one_photograph"),
                "text_route_test_probes":
                    portrait_block.get("text_route_test_probes"),
                "text_route_image_split_values":
                    portrait_block.get("text_route_image_split_values"),
                "text_route_probes_are_in_neither_primary_stratum":
                    portrait_block.get(
                        "text_route_probes_are_in_neither_primary_stratum"),
                "why_that_decides_the_route":
                    portrait_block.get("why_that_decides_the_route"),
            },
            "rejected_alternatives": next(
                (d.get("rejected") for d in
                 (prereg.get("decisions") or [])
                 if d.get("id") == "portrait_reuse"), []),
            "read_from": "portrait_reuse_exemption in the power report bound "
                         "above; refused rather than copied if the "
                         "measurement that forces it does not hold",
            "why_this_is_in_the_freeze": (
                "As sealed at 11C-R2, invariants 2 and 3 required every "
                "confirmation photograph sha256 to be new and no exploratory "
                "media to be reused. Read against the frozen primary estimand "
                "that is unsatisfiable: the estimand pools over 72 entities "
                "across two IMAGE strata, the wording stratum's 42 clusters "
                "are persons with one portrait each, and a text-route probe "
                "has image_split None so it is in neither stratum. The rules "
                "therefore forbade the design they were sealing - the same "
                "defect 11C-R2 found in the impossible "
                "association-disjointness invariant. The exemption is granted "
                "as a hash list of exactly 42 portraits so it cannot widen "
                "without a re-freeze, and the cost is recorded rather than "
                "argued away: the gate scored all 42."),
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
            # Iteration 11C-R2's finding #1.  The single invariant that used
            # to stand here read "no confirmation query_id, ASSOCIATION,
            # paraphrase template or photograph appears in the exploratory
            # dataset", and it was impossible: the confirmation exists to
            # re-test the SAME entity-attribute associations the preserved
            # adapters unlearned, with new photographs and new wording.  An
            # association-disjoint confirmation would have measured adapters
            # against associations they were never trained on, and could not
            # have confirmed anything.  What must be identical and what must
            # be new are therefore separate invariants with separate hashes.
            {
                "invariant": "the confirmation's target-association set is "
                             "REQUIRED to be IDENTICAL to the frozen one",
                "bound_by": {
                    "target_association_ids_sha256": target_assoc_sha,
                    "target_association_ids": n_target_assoc,
                    "target_entity_ids_sha256": target_entity_sha,
                    "target_entity_ids": n_target_entity,
                },
                "why": "the preserved adapters were trained to unlearn "
                       "exactly these entity-attribute pairs. A "
                       "confirmation about different associations would be a "
                       "measurement of something no adapter in this protocol "
                       "was trained on, so it could not confirm the result it "
                       "claims to confirm. Identity here is the point, not a "
                       "leak.",
                "why_it_is_not_a_leak": (
                    "what the reference-state gate and candidate selection "
                    "were exposed to was the PROBES - the queries, the "
                    "template wordings and the photographs - not the abstract "
                    "association they ask about. Repeating the association "
                    "with new probes tests whether the unlearning survives "
                    "the surface form it was measured on, which is the "
                    "question; repeating the probes would test nothing."),
                "enforced_by": "the stage-3 probe builder, which derives its "
                               "target associations from manifest.json and "
                               "must reproduce this sha256",
                "checked_at": "stage 3, before the stage-4 tag",
            },
            {
                "invariant": "every confirmation query_id, template_id and "
                             "template TEXT is NEW, and every photograph "
                             "sha256 is new except the hashed target-person "
                             "portraits",
                "bound_by": {
                    "exploratory_template_ids_sha256": expl_templates_sha,
                    "exploratory_template_ids": expl_templates_n,
                    "exploratory_photograph_sha256_manifest":
                        "data/mllmu_hier_pilot100/image_manifest.json",
                    "exploratory_query_ids": "the exploratory queries "
                                             "parquet bound above",
                    "exempted_portraits": n_portraits,
                    "exempted_portrait_set_sha256": portrait_set_sha,
                    "exemption_is_a_list_not_a_category": (
                        "exactly these 42 hashes and no other photograph. An "
                        "exemption stated as 'person portraits may repeat' "
                        "could grow to any media; stated as a hash list it "
                        "cannot"),
                },
                "why": "the reference-state gate inspected the exploratory "
                       "test split before candidate selection, which is why "
                       "those results stay exploratory. A confirmation probe "
                       "reused from that split inherits the exposure, "
                       "whatever it is a probe OF.",
                "why_template_text_and_not_only_template_id": (
                    "a new id over the same wording is the same probe with a "
                    "new label. Novelty is a property of the bytes shown to "
                    "the model, so the text is checked and not just the "
                    "identifier that names it"),
                "why_the_portraits_are_exempt": (
                    "each of the 42 target persons has exactly one "
                    "photograph, so there is no second portrait to fetch and "
                    "no pool to fetch it from. The wording stratum is "
                    "image-route by construction - all 180 of its exploratory "
                    "probes carry a portrait - and a text-route probe carries "
                    "image_split None, so it sits in NEITHER primary stratum. "
                    "Requiring a new portrait therefore does not make the "
                    "confirmation stricter, it makes the frozen estimand "
                    "unbuildable: the pooled 72-entity estimand would collapse "
                    "to 30 species clusters"),
                "why_the_exemption_does_not_give_up_the_novelty_argument": (
                    f"what the gate was exposed to on the person side was a "
                    f"PAIR of a portrait and a wording, and all "
                    f"{n_gate_saw} of these portraits were in the probes it "
                    f"scored - that cost is disclosed rather than argued "
                    f"away. What stays new is the probe: a new query_id over "
                    f"a new template text is a probe the gate never scored, "
                    f"which is what the stratum is named for - seen photo, "
                    f"UNSEEN wording"),
                "enforced_by": "the stage-3 probe builder, by hash and by "
                               "query_id, before the split is written",
                "checked_at": "stage 3, before the stage-4 tag",
            },
            {
                "invariant": "no exploratory QUERY or MEDIA is reused beyond "
                             "the two required repeats",
                "bound_by": {
                    "what_may_repeat": [
                        "the target-association set, invariant 1 above",
                        f"the {n_portraits} target-person portraits, set "
                        f"sha256 {portrait_set_sha}",
                    ],
                    "what_may_not": "query_ids, template ids, template "
                                    "texts, and every other photograph's "
                                    "sha256 - including all 360 new species "
                                    "photographs without exception",
                    "why_exactly_two_things_may_repeat": (
                        "both are what the adapters were trained on rather "
                        "than what the gate measured: the target "
                        "entity-attribute pairs, and the one portrait each "
                        "target person has. Everything the gate scored that "
                        "could be renewed was renewed"),
                },
                "why": "this is the invariant that used to be stated as "
                       "association-disjointness. Stated over queries and "
                       "media it is both satisfiable and the thing that "
                       "actually matters: it is the bytes the gate saw, not "
                       "the association they were about, that make the "
                       "exploratory results exploratory. It is stated with "
                       "its two exceptions because an invariant that forbids "
                       "the design it is sealing is not a strict invariant, "
                       "it is an unexecutable one - and the exception is "
                       "bounded by a hash list, so it cannot be widened "
                       "without a re-freeze.",
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
        diffs = drift_keys(committed, freeze)
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
    pb = cs["probe_budget"]
    print(f"  size         {cs['new_wording_probes']['new_wording_probes_total']}"
          f" new wording probes + "
          f"{cs['held_out_photographs']['new_photographs_total']} new "
          f"photographs = {pb['primary_target_total']} TARGET probes")
    print(f"  retention    {pb['new_retention_probes']} descriptive "
          f"retention probes ({freeze['retention_probes']['route']}, "
          f"{freeze['retention_probes']['new_templates_per_entity']} new "
          f"templates x "
          f"{freeze['retention_probes']['entities_covered']} entities), so "
          f"{pb['total_allocated']} allocated in all")
    print(f"  fetch        {cs['held_out_photographs']['fetch_command']}")
    print(f"  select       {cs['held_out_photographs']['selection_rule']}")
    t = freeze["primary_test"]
    print(f"  test         {t['p_value_method']}, n={t['n_permutations']} "
          f"seed={t['permutation_seed']}, directions "
          f"{json.dumps(t['direction_by_metric'])}")
    print(f"  holm         thresholds "
          f"{json.dumps(t['multiplicity']['thresholds'])}, "
          f"{t['multiplicity']['ordering']}")
    fw = t["achieved_familywise_level_under_the_global_null"]
    print(f"  achieved     familywise {fw['achieved']} against a nominal "
          f"{fw['nominal']} (inside 2 SE = "
          f"{fw['achieved_is_nominal_within_two_se']})")
    print(f"  impl         {t['implementation']['module']} "
          f"{str(t['implementation']['sha256'])[:16]} "
          f"(in the code fingerprint: "
          f"{t['implementation']['in_the_code_fingerprint']})")
    inv = freeze["sealed_split_invariants"]
    print(f"  invariants   {len(inv)} sealed-split / score-once assertions "
          f"recorded, {sum(1 for i in inv if 'IDENTICAL' in i['invariant'])} "
          f"of them requiring identity rather than novelty")
    pb = freeze["portrait_exemption"]
    print(f"  portraits    {pb['exempted']} target-person portraits may "
          f"repeat (set sha256 {str(pb['set_sha256'])[:16]}); the gate saw "
          f"{pb['the_exploratory_gate_exercised']} of them, disclosed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
