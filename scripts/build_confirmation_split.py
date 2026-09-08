#!/usr/bin/env python
"""Build and seal the 1,209-query confirmation split (Iteration 11C stage 3d).

    python scripts/build_confirmation_split.py --tag pilot100
    python scripts/build_confirmation_split.py --check-only

What it builds
--------------
``data/mllmu_hier_confirm100/`` holding ``associations.parquet``,
``queries.parquet`` and ``manifest.json``, plus a committed report at
``data/reports/mllmu_confirm100_split_report.json``.

    504  new-wording probes    42 target persons x 3 families x 4 templates
    360  held-out photographs  30 target species x 12 selected photographs
    210  retain_same probes    70 entities x 3, hash-sampled facts
    135  retain_other probes   45 donor entities x 3, hash-sampled pairs
    ----
    1209 queries, every one on split "test"

Nothing here is chosen.  Which association each wording probe asks comes from
``probe_allocation.target_associations.rows``; which retained fact or donor
pair each retention probe asks comes from
``probe_allocation.retention.rows_retain_{same,other}``; which photograph each
species probe is served comes from the selection report's canonical order.
All three are bound by sha256 in the freeze, so a builder that deviated would
fail a hash rather than silently redefine the estimand.

Why the exploratory validators are run and then classified
----------------------------------------------------------
``validate_queries`` and ``validate_image_splits`` encode the EXPLORATORY
design: three splits per (association, family), one template per
(association, family, split), a reserved training photograph at ``images[0]``,
and image-route retention coverage wherever a retained association has a
photograph.  The confirmation is deliberately different on all four counts --
it is evaluation-only, so there is no training split and no training
photograph, it asks four wordings per fact instead of one, and its retention
route is text-only by a frozen decision.

Both live in ``CODE_FINGERPRINT_MODULES``, so neither can be parameterised:
editing them would change ``code.modules_sha256`` inside all thirty committed
11R sidecars and refuse their reuse.  They are therefore run UNMODIFIED and
their errors are classified.  Exactly the classes this design forces are
tolerated, each is counted and carries its reason, and any error outside them
refuses the build.  That keeps every check that does apply -- duplicate ids,
answer/level consistency, the FILR fields, "a retain probe must not ask a
target association", "the donor must be a retained association", "no template
may repeat" and the retain-fact dedupe -- while naming the exceptions instead
of switching the validator off.

Why ``assign_split_images`` is not used
---------------------------------------
The 11R post-pass round-robins a query over ``image_pools(assoc, split)``,
which reserves ``images[0]`` for training and halves the remainder into val
and test.  A confirmation species association carries exactly its 12 selected
photographs and no exploratory one -- carrying the exploratory training
photograph would put exploratory media into this dataset's image manifest and
breach sealed invariant 2 -- so there is nothing to reserve and no val split
to fill.  Photographs are assigned explicitly instead: the k-th probe of a
species in ``query_id`` order gets the k-th selected photograph in
``selection_index`` order, a bijection the build asserts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import pandas as pd  # noqa: E402

from granunlearn.evaluation.confirmation_templates import (  # noqa: E402
    CONFIRM_IMAGE_FAMILIES,
    CONFIRM_NEW_TEMPLATES,
    CONFIRM_TEMPLATE_IDS,
    EXPLORATORY_TEMPLATES_PER_FAMILY,
    install_confirmation_templates,
    validate_confirmation_templates,
)
from granunlearn.evaluation.image_splits import (  # noqa: E402
    image_stratum,
    validate_image_splits,
)
from granunlearn.evaluation.query_generation import (  # noqa: E402
    FAMILY_TEMPLATES,
    _make_query,
    validate_queries,
)
from granunlearn.evaluation.reference_eval import (  # noqa: E402
    load_associations_parquet,
    load_queries_parquet,
)
from granunlearn.schema import AssociationRecord, ImageRef  # noqa: E402

logging.basicConfig(level=logging.INFO,
                    format="[%(asctime)s] %(levelname)-8s %(name)s — %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S")
log = logging.getLogger("build_confirmation_split")

REPORTS = "data/reports"
FREEZE_REPORT = f"{REPORTS}/mllmu_pilot100_confirmation_freeze.json"
POWER_REPORT = f"{REPORTS}/mllmu_pilot100_confirmation_power.json"
SELECTION_REPORT = f"{REPORTS}/mllmu_confirm100_photograph_selection.json"
SPLIT_REPORT = f"{REPORTS}/mllmu_confirm100_split_report.json"

#: Every confirmation query is on this split.  The dataset is scored once by
#: already-trained adapters, so there is nothing to train or tune on.
CONFIRM_SPLIT = "test"
#: Template indices the confirmation occupies: 3..6 for the image families
#: and 3..5 for the retention families.  Index 3 is the first that cannot
#: collide with an exploratory ``template_id``.
FIRST_NEW_INDEX = EXPLORATORY_TEMPLATES_PER_FAMILY
#: Prefix for confirmation ``image_id`` values, so they cannot be confused
#: with the exploratory ``inat_*`` ids in any report that pools the two.
IMAGE_ID_PREFIX = "cfm"
CONFIRM_SEED = 42

#: ``sorted(SPLITS)`` is alphabetical, so the expected side of the coverage
#: message is ``['test', 'train', 'val']``, and the observed side repeats
#: ``'test'`` once per query in the group -- four times for a wording family
#: and three for a retention one.  Both patterns are anchored and admit only
#: a list of 'test', so a group that somehow carried a train or val query
#: would land in the unclassified list and refuse the build.
_TEST_ONLY = r"\['test'(?:, 'test')*\]"
_EXPECTED_SPLITS = r"\['test', 'train', 'val'\]"

#: The exploratory checks this design cannot satisfy, with the reason each is
#: tolerated.  Anything NOT matching one of these refuses the build.
INAPPLICABLE_EXPLORATORY_CHECKS: dict[str, tuple[re.Pattern, str]] = {
    "split_coverage_per_association_family": (
        re.compile(rf"^.+/.+: split coverage {_TEST_ONLY} != "
                   rf"{_EXPECTED_SPLITS}$"),
        "the confirmation is evaluation-only: the adapters are already "
        "trained, so there is no train or val split to cover and every query "
        "is on test"),
    "split_coverage_per_retain_other_target": (
        re.compile(rf"^retain_\w+ for .+: split coverage {_TEST_ONLY} != "
                   rf"{_EXPECTED_SPLITS}$"),
        "the same test-only property, reported per (target, family) for the "
        "retain_other probes"),
    "retain_same_text_coverage": (
        re.compile(r"^retain_same_entity missing for .+ / (train|val|test)$"),
        "11R asked EVERY retained fact of an entity on all three splits; the "
        "confirmation asks a hash-sampled three per entity on test, which is "
        "why its estimand is labelled 'hash-sampled retained facts, text "
        "route' and must not be quoted against the 11R rate"),
    "retain_same_image_coverage": (
        re.compile(r"^retain_same_entity_image missing for .+ / "
                   r"(train|val|test)$"),
        "the retention route is text-only by a frozen decision, so there is "
        "no image-route retention probe to be missing"),
    "reserved_training_index": (
        re.compile(r"^.+: image_seen_in_training=False contradicts the "
                   r"reserved training photograph .+$"),
        "images[0] of a confirmation species association is one of the 12 "
        "SELECTED photographs, not a photograph training consumed; carrying "
        "the exploratory training photograph to satisfy this check would put "
        "exploratory media in this dataset's image manifest and breach "
        "sealed invariant 2"),
}


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rows_sha(rows: Any) -> str:
    """The same hash the freeze binds allocation rows under."""
    return hashlib.sha256(json.dumps(
        rows, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _distribution(values) -> dict:
    """Counts of counts, keyed by STRING.

    A ``Counter`` over probe counts has int keys, and ``json.dump`` silently
    rewrites those to strings, so a report carrying them can never compare
    equal to the build that produced it -- the committed evidence and a fresh
    derivation would disagree on key type alone.  Same convention as
    ``power_analysis_confirmation._distribution``.
    """
    out: dict[str, int] = {}
    for n in sorted(set(values)):
        out[str(n)] = sum(1 for v in values if v == n)
    return out


def _read(repo_root: Path, rel: str) -> dict:
    path = repo_root / rel
    if not path.exists():
        raise SystemExit(f"REFUSED - {rel} is not there; the confirmation "
                         "split is built from the frozen design, so nothing "
                         "may be inferred in its absence")
    return json.loads(path.read_text())


def read_frozen_design(repo_root: Path) -> dict[str, Any]:
    """Everything the build is allowed to decide nothing about.

    Read from the three committed reports and refused if the freeze itself
    refuses, so a build cannot proceed on a protocol the freeze would not
    sign.
    """
    import freeze_confirmation_protocol as fz

    freeze = _read(repo_root, FREEZE_REPORT)
    power = _read(repo_root, POWER_REPORT)
    selection = _read(repo_root, SELECTION_REPORT)

    refusals: list[str] = []
    for fn in (fz._primary_test_refusals, fz._retention_refusals,
               fz._fetch_role_refusals, fz._photo_selection_refusals,
               fz._portrait_exemption_refusals, fz._probe_allocation_refusals,
               fz._protocol_amendment_refusals):
        try:
            got = (fn(power, repo_root)
                   if fn in (fz._primary_test_refusals,
                             fz._photo_selection_refusals,
                             fz._portrait_exemption_refusals,
                             fz._protocol_amendment_refusals) else fn(power))
        except TypeError:
            got = fn(power)
        refusals.extend(f"{fn.__name__}: {r}" for r in got)
    if refusals:
        raise SystemExit(
            "REFUSED - the frozen design does not satisfy its own checks, so "
            "there is nothing well-defined to build:\n  "
            + "\n  ".join(refusals))

    alloc = power["probe_allocation"]
    size = power["confirmation_size"]
    fn = freeze["confirmation_size"]["frozen_now"]
    pe = freeze["portrait_exemption"]
    design = {
        "target_rows": alloc["target_associations"]["rows"],
        "retain_same_rows": alloc["retention"]["rows_retain_same"],
        "retain_other_rows": alloc["retention"]["rows_retain_other"],
        "row_hashes": freeze["probe_allocation"]["row_hashes"],
        "selected": selection["selected"],
        "selection_hash": selection["selected_sha256_list_sha256"],
        "target_association_ids": fn["target_association_ids"],
        "target_entity_ids": fn["target_entity_ids"],
        "exploratory_query_ids": set(pe.get("exploratory_query_ids") or []),
        "exploratory_template_ids": set(fn["exploratory_template_ids"]),
        "exempted_portraits": pe["sha256"],
        "exempted_portrait_paths": pe["paths"],
        "exempted_portrait_image_ids": pe["image_ids"],
        "wording_probes_expected":
            size["new_wording_probes"]["new_wording_probes_total"],
        "photograph_probes_expected":
            size["held_out_photographs"]["new_photographs_total"],
        "retention_probes_expected":
            size["totals"]["new_retention_probes"],
        "total_expected": alloc["confirmation_probes_total"],
        "invariants": freeze["sealed_split_invariants"],
        "exploratory_dir": f"data/mllmu_hier_{size['held_out_photographs']['fetch_tag']}",
    }
    #: The allocation rows are the freeze's own bound objects: rebuild their
    #: hashes and refuse on any drift, because a row that moved changes which
    #: fact a probe asks and therefore what the estimand is.
    for key, rows in (("target_association_rows", design["target_rows"]),
                      ("retain_same_rows", design["retain_same_rows"]),
                      ("retain_other_rows", design["retain_other_rows"])):
        bound = design["row_hashes"][key]
        got = _rows_sha(rows)
        if got != bound:
            raise SystemExit(
                f"REFUSED - the {key} in {POWER_REPORT} hash to {got[:16]} "
                f"but the freeze binds {bound[:16]}; the allocation that "
                "decides which fact each probe asks has moved since it was "
                "sealed")
    if len(design["target_rows"]) != design["wording_probes_expected"]:
        raise SystemExit(
            f"REFUSED - {len(design['target_rows'])} allocated wording rows "
            f"against a frozen budget of {design['wording_probes_expected']}")
    if len(design["selected"]) != design["photograph_probes_expected"]:
        raise SystemExit(
            f"REFUSED - {len(design['selected'])} selected photographs "
            f"against a frozen budget of {design['photograph_probes_expected']}")
    if (len(design["retain_same_rows"]) + len(design["retain_other_rows"])) \
            != design["retention_probes_expected"]:
        raise SystemExit(
            f"REFUSED - {len(design['retain_same_rows'])} + "
            f"{len(design['retain_other_rows'])} retention rows against a "
            f"frozen budget of {design['retention_probes_expected']}")
    return design


def confirmation_associations(design: dict, repo_root: Path
                              ) -> tuple[list, dict]:
    """The associations this split's probes ask, with only the media they use.

    A confirmation association carries exactly the photographs its probes are
    served: the exempted portrait for a probed person target association, the
    12 selected photographs for a probed species target association, and
    nothing for a retained association -- retention is text-route, so a
    photograph there would enter the image manifest without ever being asked
    about.  Everything else is copied from the frozen exploratory record
    verbatim, so the levels the scorer reads are the levels 11R scored.
    """
    exp_dir = repo_root / design["exploratory_dir"]
    exploratory = load_associations_parquet(exp_dir / "associations.parquet")
    by_id = {a.association_id: a for a in exploratory}

    target_ids = set(design["target_association_ids"])
    person_ids = set(json.loads(
        (repo_root / FREEZE_REPORT).read_text()
    )["portrait_exemption"]["target_person_ids"])

    #: species -> its 12 selected photographs, in selection_index order
    photos: dict[str, list[dict]] = defaultdict(list)
    for rec in design["selected"]:
        photos[rec["species"]].append(rec)
    for species, recs in photos.items():
        if sorted(r["selection_index"] for r in recs) != list(range(len(recs))):
            raise SystemExit(
                f"REFUSED - {species}'s selection indices are "
                f"{sorted(r['selection_index'] for r in recs)}, not 0.."
                f"{len(recs) - 1}; the canonical order the rule selects in is "
                "not the order the photographs would be served in")
        recs.sort(key=lambda r: r["selection_index"])

    #: The exempted portrait per person, taken from the exploratory record so
    #: the image_id and path are the ones the adapters were trained on.
    portrait_of: dict[str, ImageRef] = {}
    for a in exploratory:
        if a.entity_id in person_ids and a.association_id in target_ids \
                and a.images and a.entity_id not in portrait_of:
            portrait_of[a.entity_id] = a.images[0]
    missing = sorted(person_ids - set(portrait_of))
    if missing:
        raise SystemExit(
            f"REFUSED - {len(missing)} exempted person(s) have no portrait in "
            f"the exploratory associations ({missing[0]}, ...), so their "
            "wording probes could not be served the exempted photograph")

    needed, why = needed_associations(design, by_id)
    absent = sorted(needed - set(by_id))
    if absent:
        raise SystemExit(
            f"REFUSED - {len(absent)} association(s) the allocation asks for "
            f"are not in the exploratory dataset ({absent[0]}, ...)")

    out: list[AssociationRecord] = []
    kinds: Counter = Counter()
    for aid in sorted(needed):
        a = by_id[aid]
        if aid in target_ids and a.entity_id in person_ids:
            images = [portrait_of[a.entity_id]]
            kinds["person_target_keeps_its_exempted_portrait"] += 1
        elif aid in target_ids:
            images = [
                ImageRef(
                    image_id=f"{IMAGE_ID_PREFIX}_{r['species'].replace(' ', '_')}"
                             f"_{r['selection_index']:03d}",
                    path=(repo_root / "data/raw/inaturalist/confirm_v1"
                          / r["pool_file_name"]).relative_to(repo_root).as_posix(),
                    source="confirmation_fetch",
                    split="test",
                ) for r in photos[a.entity_id]]
            kinds["species_target_gets_its_12_selected_photographs"] += 1
        else:
            images = []
            kinds["retained_carries_no_photograph"] += 1
        out.append(a.model_copy(update={"images": images}))
    measurement = {
        "associations": len(out),
        "by_kind": dict(sorted(kinds.items())),
        "needed_because": why,
        "distinct_photographs_referenced":
            len({i.path for a in out for i in a.images}),
        "image_id_scheme": (
            f"{IMAGE_ID_PREFIX}_<Species_with_underscores>_<index:03d>, so a "
            "confirmation photograph cannot be confused with the exploratory "
            "inat_* id it is not"),
        "why_only_probed_associations": (
            "the confirmation is evaluation-only, so an association no probe "
            "asks contributes nothing to any metric; carrying all 477 would "
            "also drag every exploratory photograph into this dataset's image "
            "manifest and breach sealed invariant 2"),
        "why_retained_associations_carry_no_photograph": (
            "retention is text-route by a frozen decision, so a photograph on "
            "a retained association would be hashed into the manifest without "
            "any probe ever being served it"),
        "everything_else_is_copied_verbatim": (
            "levels, entity_name, attribute_name and textual_context come from "
            "the frozen exploratory record, because the scorer reads the "
            "answer hierarchy from them and the confirmation must score the "
            "same facts 11R did"),
    }
    return out, measurement


def needed_associations(design: dict,
                        exploratory_by_id: dict) -> tuple[set, dict]:
    """The association ids the 1,209 probes ask, and why each is needed.

    Three of the four allocations name association ids directly.  The
    photograph allocation does not: the selection report names SPECIES, so its
    30 associations are resolved here through the frozen target set rather
    than assumed -- a species whose association is not in that set would mean
    the fetch selected photographs for an entity the design never targeted.
    """
    wording = {r["association_id"] for r in design["target_rows"]}
    species = {r["species"] for r in design["selected"]}
    target_ids = set(design["target_association_ids"])
    species_assoc = {
        a.entity_id for a in exploratory_by_id.values()
        if a.entity_id in species and a.association_id in target_ids}
    species_ids = {a.association_id for a in exploratory_by_id.values()
                   if a.entity_id in species and a.association_id in target_ids}
    unresolved = sorted(species - species_assoc)
    if unresolved:
        raise SystemExit(
            f"REFUSED - {len(unresolved)} selected species have no "
            f"association in the frozen target set ({unresolved[0]}, ...), so "
            "their photographs were fetched for an entity this design does "
            "not target")
    same = {r["item"] for r in design["retain_same_rows"]}
    other_donor = {r["item"].split("|")[0] for r in design["retain_other_rows"]}
    other_target = {r["item"].split("|")[1] for r in design["retain_other_rows"]}
    return (wording | species_ids | same | other_donor | other_target, {
        "asked_by_a_wording_probe": len(wording),
        "asked_by_a_photograph_probe": len(species_ids),
        "asked_by_a_retain_same_probe": len(same),
        "donors_asked_by_a_retain_other_probe": len(other_donor),
        "targets_paired_by_a_retain_other_probe": len(other_target),
        "distinct_in_all": len(wording | species_ids | same | other_donor
                               | other_target),
    })


def confirmation_queries(design: dict, assocs: list,
                         repo_root: Path) -> tuple[list, dict]:
    """The 1,209 queries, built by ``_make_query`` and then assigned images.

    ``_make_query`` is reused rather than reimplemented because it is what
    computed the ``expected_level``, ``acceptable_answer_ids``,
    ``leakage_forbidden_ids`` and
    ``post_unlearning_acceptable_answer_ids`` the scorer reads.  Reimplementing
    them here would produce a confirmation whose FILR fields were derived by
    different code than the exploratory ones they are compared against.
    """
    refusals = validate_confirmation_templates(
        design["exploratory_template_ids"],
        {t for f in FAMILY_TEMPLATES
         for t in FAMILY_TEMPLATES[f][:EXPLORATORY_TEMPLATES_PER_FAMILY]})
    if refusals:
        raise SystemExit(
            "REFUSED - the confirmation templates are not new:\n  "
            + "\n  ".join(refusals))
    installed = install_confirmation_templates()
    by_id = {a.association_id: a for a in assocs}

    queries = []
    #: 504 wording probes + 360 photograph probes: the same 3 families x 4
    #: templates, differing only in which association they ask and so which
    #: photograph they are served.
    for row in design["target_rows"]:
        fam = CONFIRM_IMAGE_FAMILIES[row["family_index"]]
        idx = FIRST_NEW_INDEX + row["template_index"]
        aid = row["association_id"]
        if aid not in by_id:
            raise SystemExit(f"REFUSED - {aid} is not in the split")
        queries.append(_make_query(
            by_id[aid], fam, CONFIRM_SPLIT, idx, CONFIRM_SEED,
            f"{aid}__{fam}__c{idx}__{CONFIRM_SPLIT}", True))
    for species_rows in _species_probe_rows(design, by_id):
        queries.extend(species_rows)
    #: 210 retain_same probes: one per allocated row, template index cycling
    #: with the row's position inside its entity so an entity that cycles
    #: because it has one fact still asks it in three different wordings.
    for fam, rows in (("retain_same_entity", design["retain_same_rows"]),
                      ("retain_other_entity", design["retain_other_rows"])):
        per_entity: Counter = Counter()
        for row in rows:
            entity = row["entity_id"]
            j = per_entity[entity]
            per_entity[entity] += 1
            idx = FIRST_NEW_INDEX + j
            if fam == "retain_same_entity":
                aid = row["item"]
                qid = f"{aid}__{fam}__c{idx}__{CONFIRM_SPLIT}"
                queries.append(_make_query(
                    by_id[aid], fam, CONFIRM_SPLIT, idx, CONFIRM_SEED, qid,
                    False))
            else:
                donor, target = row["item"].split("|")
                qid = (f"{donor}__{fam}__for_{target}__c{idx}__"
                       f"{CONFIRM_SPLIT}")
                queries.append(_make_query(
                    by_id[donor], fam, CONFIRM_SPLIT, idx, CONFIRM_SEED, qid,
                    False, target_association_id=target))

    queries, image_info = assign_confirmation_images(queries, assocs, design)
    measurement = {
        "queries": len(queries),
        "template_ids": sorted({q.template_id for q in queries}),
        "num_template_ids": len({q.template_id for q in queries}),
        "templates_installed": installed,
        "by_family": dict(sorted(Counter(q.family for q in queries).items())),
        "by_route": dict(sorted(Counter(q.route for q in queries).items())),
        "by_split": dict(sorted(Counter(q.split for q in queries).items())),
        "images": image_info,
    }
    return queries, measurement


def _species_probe_rows(design: dict, by_id: dict) -> list:
    """The 360 photograph probes, 12 per species, each served its own.

    Built as one group per species so the bijection between the 12 probes and
    the 12 selected photographs can be asserted where it is made rather than
    hoped for downstream.
    """
    #: A species' target association, resolved from the frozen target set: the
    #: selection report names species, the associations name ids.
    species_assoc: dict[str, str] = {}
    for aid, a in by_id.items():
        if a.dataset == "inaturalist" and a.entity_id in \
                {r["species"] for r in design["selected"]}:
            if a.entity_id in species_assoc:
                raise SystemExit(
                    f"REFUSED - species {a.entity_id} has more than one "
                    f"association in the split ({species_assoc[a.entity_id]} "
                    f"and {aid}), so 12 photograph probes would not have one "
                    "association to ask")
            species_assoc[a.entity_id] = aid
    missing = sorted({r["species"] for r in design["selected"]}
                     - set(species_assoc))
    if missing:
        raise SystemExit(
            f"REFUSED - {len(missing)} selected species have no association in "
            f"the split ({missing[0]}, ...)")
    groups = []
    for species in sorted(species_assoc):
        aid = species_assoc[species]
        rows = []
        for f, fam in enumerate(CONFIRM_IMAGE_FAMILIES):
            for j in range(len(CONFIRM_NEW_TEMPLATES[fam])):
                idx = FIRST_NEW_INDEX + j
                rows.append((f, j, idx, fam))
        #: 3 families x 4 templates = 12 probes, one per selected photograph.
        if len(rows) != len(design["selected"]) // len(species_assoc):
            raise SystemExit(
                f"REFUSED - {species} would get {len(rows)} photograph probes "
                "where the selection gives it "
                f"{len(design['selected']) // len(species_assoc)} photographs")
        probes = [_make_query(
            by_id[aid], fam, CONFIRM_SPLIT, idx, CONFIRM_SEED,
            f"{aid}__{fam}__c{idx}__{CONFIRM_SPLIT}", True)
            for _f, _j, idx, fam in rows]
        groups.append(sorted(probes, key=lambda q: q.query_id))
    return groups


def assign_confirmation_images(queries: list, assocs: list,
                               design: dict) -> tuple[list, dict]:
    """Serve each image probe its photograph explicitly.

    Person wording probes get that person's exempted portrait -- the one
    photograph they have, which is why the stratum is
    ``seen_photo_unseen_wording`` and why the exemption exists.  Species
    photograph probes get the k-th selected photograph in ``query_id`` order,
    a bijection, which is what makes the stratum ``held_out_photo`` with 12
    distinct photographs per species instead of 3.
    """
    by_id = {a.association_id: a for a in assocs}
    persons = {a.entity_id for a in assocs
               if a.images and len(a.images) == 1}
    groups: dict[str, list] = defaultdict(list)
    for q in queries:
        assoc = by_id[q.association_id]
        if assoc.entity_id in persons:
            continue
        if q.family in CONFIRM_IMAGE_FAMILIES:
            groups[q.association_id].append(q)

    updates: dict[str, dict] = {}
    for aid, group in groups.items():
        assoc = by_id[aid]
        ordered = sorted(group, key=lambda q: q.query_id)
        if len(ordered) != len(assoc.images):
            raise SystemExit(
                f"REFUSED - {aid} has {len(ordered)} photograph probes and "
                f"{len(assoc.images)} photographs, so they cannot be paired "
                "one to one and some photograph would be served twice while "
                "another was never served at all")
        for q, image in zip(ordered, assoc.images):
            updates[q.query_id] = {
                "image_ids": [image.image_id],
                "image_split": "test",
                "image_seen_in_training": False,
            }
    #: Person wording probes keep the portrait ``_make_query`` already served
    #: them, but the flag and the split label are set explicitly rather than
    #: inherited, because the stratum is derived from the flag and the flag is
    #: the whole content of the exemption.
    for q in queries:
        assoc = by_id[q.association_id]
        if assoc.entity_id in persons and q.family in CONFIRM_IMAGE_FAMILIES:
            if not q.image_ids:
                raise SystemExit(
                    f"REFUSED - {q.query_id} is a wording probe on "
                    f"{assoc.entity_id} but was served no photograph, so it "
                    "would fall out of both primary strata")
            updates[q.query_id] = {
                "image_ids": [assoc.images[0].image_id],
                "image_split": "train",
                "image_seen_in_training": True,
            }
    out = [q.model_copy(update=updates[q.query_id])
           if q.query_id in updates else q for q in queries]
    info = {
        "probes_assigned_explicitly": len(updates),
        "species_probes_paired_one_to_one": sum(len(g) for g in groups.values()),
        "person_probes_serving_the_exempted_portrait":
            sum(1 for q in out if q.image_ids and q.image_seen_in_training),
        "assign_split_images_was_not_used": (
            "the 11R post-pass reserves images[0] for training and halves the "
            "rest into val and test; a confirmation species association has no "
            "exploratory photograph to reserve and no val split, so all 12 are "
            "test photographs by construction"),
        "strata": dict(sorted(Counter(
            str(image_stratum(q)) for q in out).items())),
    }
    return out, info


def classify_exploratory_errors(errors: list[str], which: str) -> dict:
    """Sort a validator's errors into tolerated classes and everything else.

    Returns ``{classified: {class: [errors]}, unclassified: [errors]}``.  The
    unclassified list is what refuses the build, so tolerating a class can
    never swallow an error nobody predicted.
    """
    classified: dict[str, list[str]] = defaultdict(list)
    unclassified: list[str] = []
    for e in errors:
        for name, (pattern, _reason) in INAPPLICABLE_EXPLORATORY_CHECKS.items():
            if pattern.match(e):
                classified[name].append(e)
                break
        else:
            unclassified.append(e)
    return {
        "validator": which,
        "errors_total": len(errors),
        "classified": {k: sorted(v) for k, v in sorted(classified.items())},
        "classified_counts": {k: len(v)
                              for k, v in sorted(classified.items())},
        "unclassified": sorted(unclassified),
        "why_each_class_is_tolerated": {
            k: INAPPLICABLE_EXPLORATORY_CHECKS[k][1]
            for k in sorted(classified)},
    }


def sealed_invariants(design: dict, queries: list, assocs: list,
                      repo_root: Path) -> dict:
    """Check the split against the freeze's eight sealed invariants.

    Run BEFORE anything is written, so a split that breaches one is never on
    disk to be scored.  Four of the eight are properties of a split and are
    measured here; four are properties of how the scoring stage behaves and
    cannot be measured from a dataset, so they are named with where they are
    enforced rather than reported as passed.
    """
    exp_dir = repo_root / design["exploratory_dir"]
    exp_queries = load_queries_parquet(exp_dir / "queries.parquet")
    exp_manifest = json.loads(
        (repo_root / "data/mllmu_hier_pilot100/image_manifest.json").read_text())
    exp_hashes = {v["sha256"] for v in exp_manifest["images"].values()}
    exp_texts = {t for f in FAMILY_TEMPLATES
                 for t in FAMILY_TEMPLATES[f][:EXPLORATORY_TEMPLATES_PER_FAMILY]}

    target_ids = set(design["target_association_ids"])
    carried_targets = sorted(a.association_id for a in assocs
                             if a.association_id in target_ids)
    target_entities = sorted({a.entity_id for a in assocs
                              if a.association_id in target_ids})
    #: Every photograph this split references, by the path it will be hashed
    #: from once build_image_manifest seals it.
    referenced = sorted({i.path for a in assocs for i in a.images})
    #: The 42 exempted portraits keep their exploratory path, so their sha256
    #: is read from the exploratory manifest; the 360 selected photographs are
    #: read from the selection report, which re-hashed every file on disk.
    #: Neither is re-hashed here: build_image_manifest does that over the bytes
    #: when it seals, and hashing twice in two places is how the two come to
    #: disagree.
    portraits = set(design["exempted_portraits"])
    manifest_by_path = {k: v["sha256"]
                        for k, v in exp_manifest["images"].items()}
    sha_of = {p: manifest_by_path[p] for p in referenced if p in manifest_by_path}
    for rec in design["selected"]:
        rel = (repo_root / "data/raw/inaturalist/confirm_v1"
               / rec["pool_file_name"]).relative_to(repo_root).as_posix()
        if rel in set(referenced):
            sha_of[rel] = rec["sha256"]
    unresolved = sorted(set(referenced) - set(sha_of))

    per_person = Counter(
        a.entity_id for q in queries if q.family in CONFIRM_IMAGE_FAMILIES
        and q.image_seen_in_training
        for a in [next(x for x in assocs
                       if x.association_id == q.association_id)])
    per_species = Counter(
        q.association_id for q in queries
        if q.family in CONFIRM_IMAGE_FAMILIES and not q.image_seen_in_training)

    new_hashes = sorted(set(sha_of.values()) - portraits)
    checks = {
        "1_target_association_set_is_identical": {
            "satisfied": (carried_targets == sorted(target_ids)
                          and target_entities == sorted(design["target_entity_ids"])
                          and hashlib.sha256("\n".join(carried_targets).encode()
                                             ).hexdigest()
                          == _freeze_sha(repo_root, "target_association_ids_sha256")
                          and hashlib.sha256("\n".join(target_entities).encode()
                                             ).hexdigest()
                          == _freeze_sha(repo_root, "target_entity_ids_sha256")),
            "carried_target_associations": len(carried_targets),
            "frozen_target_associations": len(target_ids),
            "carried_target_entities": len(target_entities),
            "frozen_target_entities": len(design["target_entity_ids"]),
            "how_it_is_checked": (
                "the ids are recomputed from the built associations and their "
                "sha256 compared to the two hashes the freeze binds, so "
                "'identical' is a hash comparison and not a count"),
        },
        "2_every_id_and_text_and_photograph_is_new": {
            "satisfied": (
                not ({q.query_id for q in queries}
                     & {x.query_id for x in exp_queries})
                and not ({q.template_id for q in queries}
                         & design["exploratory_template_ids"])
                and not ({FAMILY_TEMPLATES[q.family][int(q.template_id.split(':')[1])]
                          for q in queries} & exp_texts)
                and not (set(new_hashes) & exp_hashes)
                and not unresolved),
            "query_ids_reused": sorted(
                {q.query_id for q in queries}
                & {x.query_id for x in exp_queries}),
            "template_ids_reused": sorted(
                {q.template_id for q in queries}
                & design["exploratory_template_ids"]),
            "template_texts_reused": sorted(
                {FAMILY_TEMPLATES[q.family][int(q.template_id.split(':')[1])]
                 for q in queries} & exp_texts),
            "photographs_that_are_exploratory_media":
                sorted(set(sha_of.values()) & exp_hashes),
            "photographs_exempted_by_hash": len(
                set(sha_of.values()) & exp_hashes & portraits),
            "photographs_new": len(new_hashes),
            "referenced_paths_with_no_recorded_sha256": unresolved,
            "how_it_is_checked": (
                "query ids and template ids against the exploratory parquet "
                "and the freeze's bound id set, template TEXTS by resolving "
                "each probe's index back through FAMILY_TEMPLATES, and "
                "photographs by sha256 against the exploratory image manifest "
                "minus the 42 hashed portraits"),
        },
        "3_no_exploratory_query_or_media_beyond_the_two_repeats": {
            "satisfied": (
                not ({q.query_id for q in queries}
                     & {x.query_id for x in exp_queries})
                and set(sha_of.values()) & exp_hashes <= portraits),
            "what_repeats": [
                f"the {len(carried_targets)} frozen target-association ids",
                f"the {len(portraits)} hashed target-person portraits",
            ],
            "exploratory_queries_reused": len(
                {q.query_id for q in queries}
                & {x.query_id for x in exp_queries}),
            "exploratory_media_reused_beyond_the_exemption": sorted(
                (set(sha_of.values()) & exp_hashes) - portraits),
        },
        "8_the_split_is_the_size_frozen_here": {
            "satisfied": (
                len(queries) == design["total_expected"]
                and set(per_person.values()) == {12}
                and len(per_person) == 42
                and set(per_species.values()) == {12}
                and len(per_species) == 30),
            "queries": len(queries),
            "frozen_total": design["total_expected"],
            "wording_probes_per_target_person":
                _distribution(per_person.values()),
            "target_persons": len(per_person),
            "photograph_probes_per_species_association":
                _distribution(per_species.values()),
            "species_associations": len(per_species),
        },
    }
    #: Four of the eight are not properties of a dataset.  Reporting them as
    #: passed would be a claim nothing here can support, so they are named
    #: with where they are enforced instead.
    checks["not_a_property_of_the_split"] = {
        "4_never_enters_the_reference_state_gate":
            "enforced at scoring: the reference-state gate reads "
            f"{design['exploratory_dir']}, and this dataset is a different "
            "directory no gate script is pointed at",
        "5_never_enters_candidate_selection":
            "enforced at scoring: select_unlearning_checkpoints.py reads the "
            "exploratory predictions only; the confirmation is scored after "
            "the candidate is already chosen",
        "6_partial_results_not_inspected":
            "enforced by procedure at stage 5: B3, B0 and M_G are scored in "
            "one uniform batch layout and read only when all three exist",
        "7_a_failed_claim_is_reported_as_failed":
            "enforced by the frozen pass/fail rule in "
            "primary_test.multiplicity.pass_fail_rule, which the scorer "
            "applies without discretion",
        "why_they_are_listed_rather_than_checked": (
            "a dataset cannot demonstrate how it will be used; claiming these "
            "four as verified here would report a property no measurement in "
            "this build supports"),
    }
    checks["all_split_properties_hold"] = all(
        v["satisfied"] for k, v in checks.items() if isinstance(v, dict)
        and "satisfied" in v)
    return checks


def _freeze_sha(repo_root: Path, key: str) -> str | None:
    fn = json.loads((repo_root / FREEZE_REPORT).read_text())[
        "confirmation_size"]["frozen_now"]
    return fn.get(key)


def build_split(repo_root: Path) -> dict:
    design = read_frozen_design(repo_root)
    assocs, assoc_info = confirmation_associations(design, repo_root)
    queries, query_info = confirmation_queries(design, assocs, repo_root)

    #: The exploratory validators, run unmodified and then classified.  Both
    #: live in CODE_FINGERPRINT_MODULES, so neither can be told that this
    #: design is test-only; what can be done is to name every error the design
    #: forces and refuse on any other.
    partition = {
        "target_association_ids": design["target_association_ids"],
        "retain_association_ids": json.loads(
            (repo_root / design["exploratory_dir"] / "manifest.json"
             ).read_text())["retain_association_ids"],
    }
    per_entity: dict[str, dict] = {}
    by_id = {a.association_id: a for a in assocs}
    for rid in partition["retain_association_ids"]:
        if rid in by_id:
            per_entity.setdefault(by_id[rid].entity_id, {"retain": []})
            per_entity[by_id[rid].entity_id]["retain"].append(rid)
    partition["per_entity"] = per_entity
    retain_facts: dict[str, set] = {}
    exp_assocs = load_associations_parquet(
        repo_root / design["exploratory_dir"] / "associations.parquet")
    exp_by_id = {a.association_id: a for a in exp_assocs}
    for rid in partition["retain_association_ids"]:
        a = exp_by_id[rid]
        facts = retain_facts.setdefault(a.entity_id, set())
        facts.add(a.levels[0].value)
        facts.update(a.textual_context)

    q_errors, q_stats = validate_queries(
        queries, assocs, partition=partition,
        retain_facts_by_entity=retain_facts)
    i_errors = validate_image_splits(queries, assocs, CONFIRM_SEED)
    q_class = classify_exploratory_errors(q_errors, "validate_queries")
    i_class = classify_exploratory_errors(i_errors, "validate_image_splits")
    if q_class["unclassified"] or i_class["unclassified"]:
        raise SystemExit(
            "REFUSED - the exploratory validators report errors this design "
            "does not account for:\n  "
            + "\n  ".join((q_class["unclassified"] + i_class["unclassified"])
                          [:20]))

    invariants = sealed_invariants(design, queries, assocs, repo_root)
    if not invariants["all_split_properties_hold"]:
        failed = sorted(k for k, v in invariants.items()
                        if isinstance(v, dict) and v.get("satisfied") is False)
        raise SystemExit(
            f"REFUSED - the built split breaches sealed invariant(s) "
            f"{failed}; nothing has been written")

    report = {
        "stage": "11C-3d: the confirmation split, built from the frozen design",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "what_this_is": (
            "the 1,209-probe confirmation split, built entirely from "
            "allocations the freeze binds by hash; every choice about which "
            "fact or photograph a probe asks was sealed before this script "
            "ran"),
        "dataset_dir": "data/mllmu_hier_confirm100",
        "associations": assoc_info,
        "queries": query_info,
        "exploratory_validators": {
            "run_unmodified_because": (
                "validate_queries and validate_image_splits are both in "
                "CODE_FINGERPRINT_MODULES; parameterising either would change "
                "code.modules_sha256 inside all thirty committed 11R sidecars "
                "and refuse their reuse"),
            "validate_queries": q_class,
            "validate_image_splits": i_class,
            "stats": {k: v for k, v in q_stats.items()
                      if k in ("num_queries", "by_split", "by_family",
                               "by_route", "num_errors")},
        },
        "sealed_invariants": invariants,
        "sizes": {
            "queries": len(queries),
            "associations": len(assocs),
            "distinct_query_ids": len({q.query_id for q in queries}),
            "distinct_template_ids": len({q.template_id for q in queries}),
            "distinct_photographs": len(
                {i.path for a in assocs for i in a.images}),
        },
        #: Booleans only.  A count in this dict would be counted as a failed
        #: check by the ``is not True`` test below, and a check that cannot
        #: pass is one nobody reads.
        "checks": {
            "every_query_id_is_distinct":
                len({q.query_id for q in queries}) == len(queries)
                == design["total_expected"],
            "every_template_id_is_a_confirmation_one":
                {q.template_id for q in queries} <= set(CONFIRM_TEMPLATE_IDS),
            "every_confirmation_template_is_used":
                {q.template_id for q in queries} == set(CONFIRM_TEMPLATE_IDS),
            "every_query_is_on_the_test_split":
                {q.split for q in queries} == {CONFIRM_SPLIT},
            "all_split_properties_hold":
                invariants["all_split_properties_hold"],
            "no_unclassified_validator_error":
                not q_class["unclassified"] and not i_class["unclassified"],
            "selection_hash_unchanged":
                design["selection_hash"]
                == "c794162cbe393521e19f4485713ebdfc9917e97dbfefa5f50f97146bd85ed744",
        },
        #: Renamed from ``next_step`` for the reason given at the same field in
        #: select_confirmation_photographs.py, and with more force here: this
        #: report is REBUILT by ``--allow-rebuild``, so a field claiming to
        #: describe the present is wrong again every time the freeze runs over
        #: it.  Naming the time it describes is the version that stays true.
        "stage_at_generation_time": (
            "stage 3 of 5 (build genuinely new probes): queries and "
            "associations written, the photographs not yet sealed into a "
            "manifest and the protocol not yet re-frozen over them"),
        "next_step_at_generation_time": (
            "python scripts/build_image_manifest.py --tag confirm100 to seal "
            "the photographs, then stage 4: commit the query, image and "
            "checkpoint hashes and tag the freeze before any inference"),
    }
    failures = [k for k, v in report["checks"].items() if v is not True]
    if failures:
        raise SystemExit(
            f"REFUSED - the built split fails its own checks: {failures}")
    return {"report": report, "queries": queries, "associations": assocs}


def write_split(repo_root: Path, built: dict) -> None:
    out = repo_root / "data" / "mllmu_hier_confirm100"
    out.mkdir(parents=True, exist_ok=True)
    queries, assocs = built["queries"], built["associations"]
    pd.DataFrame([a.model_dump() for a in assocs]).to_parquet(
        out / "associations.parquet", index=False)
    pd.DataFrame([q.model_dump() for q in queries]).to_parquet(
        out / "queries.parquet", index=False)
    manifest = {
        "dataset": "mllmu_hier_confirm100",
        "version": "confirm100_v1",
        "iteration": "11C",
        "seed": CONFIRM_SEED,
        "split_mode": "entity_level",
        "purpose": (
            "evaluation-only confirmation split: the adapters are already "
            "trained, so this dataset has no training or validation use and "
            "every query is on the test split"),
        "num_entities": len({a.entity_id for a in assocs}),
        "num_associations": len(assocs),
        "num_queries": len(queries),
        "num_image_references": sum(len(a.images) for a in assocs),
        "num_unique_images": len({i.path for a in assocs for i in a.images}),
        "target_association_ids": sorted(
            a.association_id for a in assocs
            if a.association_id in set(
                json.loads((repo_root / FREEZE_REPORT).read_text()
                           )["confirmation_size"]["frozen_now"]
                ["target_association_ids"])),
        "queries_by_family": dict(sorted(
            Counter(q.family for q in queries).items())),
        "queries_by_route": dict(sorted(
            Counter(q.route for q in queries).items())),
        "image_strata": dict(sorted(Counter(
            str(image_stratum(q)) for q in queries if q.image_ids).items())),
        "image_split_policy": {
            "no_training_photograph_is_reserved": (
                "images[0] of a species association is one of its 12 selected "
                "photographs. The exploratory training photograph is "
                "deliberately NOT carried, because it would enter this "
                "dataset's image manifest and breach sealed invariant 2"),
            "every_species_photograph_is_a_test_photograph": (
                "there is no val split: nothing is tuned on this dataset, so "
                "halving the photographs would reserve six of them from the "
                "only probes that exist"),
            "person_portraits_are_train_labelled": (
                "the 42 exempted portraits keep their exploratory image_id, "
                "path and train label, and every probe served one carries "
                "image_seen_in_training=True, which is what places it in "
                "seen_photo_unseen_wording"),
        },
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    manifest["frozen_artifact_sha256"] = {
        "associations.parquet": _sha256_file(out / "associations.parquet"),
        "queries.parquet": _sha256_file(out / "queries.parquet"),
        "manifest_pre_hash_note":
            "hashes computed over the final parquet artifacts",
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (repo_root / SPLIT_REPORT).write_text(
        json.dumps(built["report"], indent=2, sort_keys=True) + "\n")
    log.info("wrote %s", out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--tag", default="pilot100",
                    help="exploratory dataset tag the design is frozen over")
    ap.add_argument("--check-only", action="store_true",
                    help="build and verify, but write nothing")
    ap.add_argument("--allow-rebuild", action="store_true",
                    help="overwrite an existing confirmation split")
    args = ap.parse_args()

    out = REPO_ROOT / "data" / "mllmu_hier_confirm100"
    if out.exists() and not (args.check_only or args.allow_rebuild):
        raise SystemExit(
            f"REFUSED - {out} already exists. A confirmation split is built "
            "once; pass --allow-rebuild to replace it deliberately, or "
            "--check-only to verify the build without writing it")

    built = build_split(REPO_ROOT)
    r = built["report"]
    c = r["checks"]
    print(f"  queries          {r['sizes']['queries']} over "
          f"{r['sizes']['associations']} associations, "
          f"{r['sizes']['distinct_photographs']} distinct photographs")
    print(f"  families         {json.dumps(r['queries']['by_family'])}")
    print(f"  routes           {json.dumps(r['queries']['by_route'])}")
    print(f"  strata           {json.dumps(r['queries']['images']['strata'])}")
    print(f"  template ids     {r['sizes']['distinct_template_ids']} new, all "
          f"confirmation ones: {c['every_template_id_is_a_confirmation_one']}, "
          f"all used: {c['every_confirmation_template_is_used']}")
    inv = r["sealed_invariants"]
    for k in sorted(k for k in inv if k.startswith(("1_", "2_", "3_", "8_"))):
        v = inv[k]
        print(f"  invariant {k.split('_')[0]}      satisfied={v['satisfied']}")
    print(f"  validators       validate_queries "
          f"{r['exploratory_validators']['validate_queries']['errors_total']} "
          f"errors, all in "
          f"{len(r['exploratory_validators']['validate_queries']['classified_counts'])}"
          f" tolerated class(es); validate_image_splits "
          f"{r['exploratory_validators']['validate_image_splits']['errors_total']}"
          f"; unclassified "
          f"{len(r['exploratory_validators']['validate_queries']['unclassified'])}"
          f"+{len(r['exploratory_validators']['validate_image_splits']['unclassified'])}")
    if args.check_only:
        print("  --check-only: nothing written")
        return 0
    write_split(REPO_ROOT, built)
    print(f"\nwrote {out} and {SPLIT_REPORT}")
    print(f"  next: {r['next_step_at_generation_time']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
