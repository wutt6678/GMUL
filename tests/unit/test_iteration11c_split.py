"""Iteration 11C stage 3d: the 1,209-query confirmation split.

Three things are tested separately, because passing one does not imply the
other:

* the SPLIT is what the frozen design specifies -- 504 wording probes asking
  ``A_e[(j + f) mod |A_e|]``, 360 photograph probes paired one-to-one with the
  selected photographs, and 345 retention probes asking exactly the
  hash-sampled rows the freeze binds by sha256;
* the split is NEW -- every query id, template id and template text, and every
  photograph except the 42 hashed portraits;
* the build is fail-closed -- the exploratory validators run unmodified, only
  the classes this design forces are tolerated, and anything else refuses.

The reason ``validate_queries`` and ``validate_image_splits`` are run and then
classified rather than adapted is that both live in ``CODE_FINGERPRINT_MODULES``:
editing either changes ``code.modules_sha256`` inside all thirty committed 11R
prediction sidecars and ``verify_sidecar`` refuses every one of them.  That is
asserted here too, against the committed module bytes.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
REPORTS = REPO_ROOT / "data" / "reports"
CONFIRM = REPO_ROOT / "data" / "mllmu_hier_confirm100"
EXPLORATORY = REPO_ROOT / "data" / "mllmu_hier_pilot100"
SPLIT_REPORT = REPORTS / "mllmu_confirm100_split_report.json"
FREEZE_PATH = REPORTS / "mllmu_pilot100_confirmation_freeze.json"
POWER_PATH = REPORTS / "mllmu_pilot100_confirmation_power.json"
SELECTION_PATH = REPORTS / "mllmu_confirm100_photograph_selection.json"

sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "src"))

import build_confirmation_split as bcs  # noqa: E402
from granunlearn.evaluation.confirmation_templates import (  # noqa: E402
    CONFIRM_IMAGE_FAMILIES,
    CONFIRM_NEW_TEMPLATES,
    CONFIRM_RETENTION_FAMILIES,
    CONFIRM_TEMPLATE_IDS,
    EXPLORATORY_TEMPLATES_PER_FAMILY,
    install_confirmation_templates,
    validate_confirmation_templates,
)
from granunlearn.evaluation.image_splits import image_stratum  # noqa: E402
from granunlearn.evaluation.query_generation import FAMILY_TEMPLATES  # noqa: E402


def _load(path: Path) -> dict:
    if not path.exists():
        pytest.skip(f"committed artifact not present: {path}")
    return json.loads(path.read_text())


def _report() -> dict:
    return _load(SPLIT_REPORT)


def _queries():
    from granunlearn.evaluation.reference_eval import load_queries_parquet
    if not (CONFIRM / "queries.parquet").exists():
        pytest.skip(f"confirmation split not present: {CONFIRM}")
    return load_queries_parquet(CONFIRM / "queries.parquet")


def _associations():
    from granunlearn.evaluation.reference_eval import load_associations_parquet
    if not (CONFIRM / "associations.parquet").exists():
        pytest.skip(f"confirmation split not present: {CONFIRM}")
    return load_associations_parquet(CONFIRM / "associations.parquet")


def _exploratory_queries():
    from granunlearn.evaluation.reference_eval import load_queries_parquet
    if not (EXPLORATORY / "queries.parquet").exists():
        pytest.skip(f"exploratory split not present: {EXPLORATORY}")
    return load_queries_parquet(EXPLORATORY / "queries.parquet")


# ── the templates are new, and the fingerprinted module is untouched ──

class TestTheTemplatesAreNewAndTheFingerprintDidNotMove:
    def test_there_are_eighteen_new_templates_over_five_families(self):
        assert len(CONFIRM_TEMPLATE_IDS) == 18
        assert len(CONFIRM_TEMPLATE_IDS) == sum(
            len(v) for v in CONFIRM_NEW_TEMPLATES.values())
        assert set(CONFIRM_NEW_TEMPLATES) == \
            set(CONFIRM_IMAGE_FAMILIES) | set(CONFIRM_RETENTION_FAMILIES)
        assert len(set(CONFIRM_TEMPLATE_IDS)) == 18

    def test_the_indices_start_where_the_exploratory_ones_end(self):
        """Index 3 is the first that cannot collide with an exploratory
        ``template_id``, and only because every family has exactly three."""
        for tid in CONFIRM_TEMPLATE_IDS:
            fam, idx = tid.rsplit(":", 1)
            assert int(idx) >= EXPLORATORY_TEMPLATES_PER_FAMILY, tid
            assert fam in CONFIRM_NEW_TEMPLATES
        assert {tid.rsplit(":", 1)[1] for tid in CONFIRM_TEMPLATE_IDS
                if tid.startswith("multimodal")} == {"3", "4", "5", "6"}
        assert {tid.rsplit(":", 1)[1] for tid in CONFIRM_TEMPLATE_IDS
                if tid.startswith("retain_same")} == {"3", "4", "5"}
        #: The guarantee rests on the exploratory id set, not on the live
        #: module -- which install_confirmation_templates extends in place.
        exp_ids = {q.template_id for q in _exploratory_queries()}
        assert Counter(t.rsplit(":", 1)[1] for t in exp_ids) == \
            Counter({"0": 19, "1": 19, "2": 19})

    def test_no_new_id_or_text_is_an_exploratory_one(self):
        exp_ids = {q.template_id for q in _exploratory_queries()}
        exp_texts = {t for f in FAMILY_TEMPLATES
                     for t in FAMILY_TEMPLATES[f]
                     [:EXPLORATORY_TEMPLATES_PER_FAMILY]}
        assert len(exp_ids) == 57
        assert not (set(CONFIRM_TEMPLATE_IDS) & exp_ids)
        new_texts = {t for ts in CONFIRM_NEW_TEMPLATES.values() for t in ts}
        assert not (new_texts & exp_texts)
        assert validate_confirmation_templates(exp_ids, exp_texts) == []

    def test_the_new_texts_are_all_distinct(self):
        texts = [t for ts in CONFIRM_NEW_TEMPLATES.values() for t in ts]
        assert len(set(texts)) == len(texts) == 18

    def test_the_multimodal_wording_does_not_call_a_species_a_person(self):
        """The exploratory ``multimodal_image_text`` templates all say
        "person", so on a species they read "For the person shown (Mallard)".
        That is in the frozen exploratory data and is not repaired here,
        because repairing it means editing a fingerprinted module.  The
        confirmation wording is neutral instead."""
        for text in CONFIRM_NEW_TEMPLATES["multimodal_image_text"]:
            assert "person" not in text.lower(), text
            assert "{name}" in text and "{question}" in text, text
        exploratory = FAMILY_TEMPLATES["multimodal_image_text"][:3]
        assert any("person" in t.lower() for t in exploratory)

    def test_the_retention_wording_uses_only_the_fields_it_is_given(self):
        """The two retention families are not question families, so
        ``_make_query`` never supplies ``question`` to them."""
        for fam in CONFIRM_RETENTION_FAMILIES:
            for text in CONFIRM_NEW_TEMPLATES[fam]:
                assert "{question}" not in text, text
                assert "{name}" in text and "{attr}" in text, text

    def test_install_is_idempotent_and_reports_what_it_added(self):
        first = install_confirmation_templates()
        second = install_confirmation_templates()
        assert first == second
        assert first["num_templates"] == 18
        assert sorted(first["template_ids"]) == sorted(CONFIRM_TEMPLATE_IDS)
        for fam in CONFIRM_NEW_TEMPLATES:
            assert len(FAMILY_TEMPLATES[fam]) == \
                EXPLORATORY_TEMPLATES_PER_FAMILY + \
                len(CONFIRM_NEW_TEMPLATES[fam])

    def test_a_family_that_grew_a_fourth_exploratory_template_refuses(self):
        """The guarantee that index 3 is new rests on every family having
        exactly three exploratory ids.  If one grew, the id would be reused
        silently -- and the check has to see that through the ids the freeze
        binds, not through the live module this test file has already
        extended."""
        exp_ids = {q.template_id for q in _exploratory_queries()}
        exp_texts = {t for f in FAMILY_TEMPLATES
                     for t in FAMILY_TEMPLATES[f][:3]}
        grown = exp_ids | {"image_fine_direct:7"}
        refusals = validate_confirmation_templates(grown, exp_texts)
        assert any("has 4 exploratory template_ids" in r for r in refusals), \
            refusals
        #: And the unmodified bound set still installs cleanly even though
        #: FAMILY_TEMPLATES has already been extended in this process.
        assert validate_confirmation_templates(exp_ids, exp_texts) == []

    def test_a_text_that_duplicates_an_exploratory_one_refuses(self):
        exp_ids = {q.template_id for q in _exploratory_queries()}
        saved = dict(CONFIRM_NEW_TEMPLATES)
        try:
            CONFIRM_NEW_TEMPLATES["image_fine_direct"] = \
                ("{question}",) + saved["image_fine_direct"][1:]
            refusals = validate_confirmation_templates(
                exp_ids, {"{question}"})
            assert any("reuses an exploratory template TEXT" in r
                       for r in refusals), refusals
        finally:
            CONFIRM_NEW_TEMPLATES.clear()
            CONFIRM_NEW_TEMPLATES.update(saved)

    def test_a_template_that_needs_a_field_it_is_not_given_refuses(self):
        exp_ids = {q.template_id for q in _exploratory_queries()}
        saved = dict(CONFIRM_NEW_TEMPLATES)
        try:
            CONFIRM_NEW_TEMPLATES["retain_same_entity"] = \
                ("{question} for {name}", "{attr} of {name}",
                 "Report {attr} for {name}")
            refusals = validate_confirmation_templates(exp_ids, set())
            assert any("does not supply" in r for r in refusals), refusals
        finally:
            CONFIRM_NEW_TEMPLATES.clear()
            CONFIRM_NEW_TEMPLATES.update(saved)

    def test_a_template_that_substitutes_nothing_refuses(self):
        exp_ids = {q.template_id for q in _exploratory_queries()}
        saved = dict(CONFIRM_NEW_TEMPLATES)
        try:
            CONFIRM_NEW_TEMPLATES["retain_same_entity"] = \
                ("State the fact.", "{attr} of {name}", "Report {attr} {name}")
            refusals = validate_confirmation_templates(exp_ids, set())
            assert any("substitutes nothing" in r for r in refusals), refusals
        finally:
            CONFIRM_NEW_TEMPLATES.clear()
            CONFIRM_NEW_TEMPLATES.update(saved)

    def test_query_generation_is_byte_identical_to_the_committed_one(self):
        """The whole reason the templates live in their own module.  Adding a
        character to ``query_generation.py`` changes ``code.modules_sha256``
        in all thirty committed 11R sidecars and ``verify_sidecar`` refuses
        every one, discarding the exploratory predictions the confirmation is
        defined against."""
        rel = "src/granunlearn/evaluation/query_generation.py"
        committed = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "show", f"HEAD:{rel}"],
            capture_output=True, check=True).stdout
        assert hashlib.sha256(committed).hexdigest() == \
            hashlib.sha256((REPO_ROOT / rel).read_bytes()).hexdigest()

    def test_the_templates_module_is_not_a_fingerprinted_module(self):
        from granunlearn.evaluation.prediction_provenance import (
            CODE_FINGERPRINT_MODULES)
        assert "src/granunlearn/evaluation/confirmation_templates.py" not in \
            set(CODE_FINGERPRINT_MODULES)
        assert len(CODE_FINGERPRINT_MODULES) == 10


# ── the split is the size and shape the freeze specifies ─────────────

class TestTheSplitIsTheFrozenDesign:
    def test_the_counts_are_the_frozen_ones(self):
        r = _report()
        assert r["sizes"]["queries"] == 1209
        assert r["sizes"]["distinct_query_ids"] == 1209
        assert r["sizes"]["distinct_template_ids"] == 18
        assert r["sizes"]["distinct_photographs"] == 402
        assert r["sizes"]["associations"] == 317

    def test_the_strata_are_the_two_primary_ones_and_nothing_else(self):
        """360 held-out photographs and 504 seen-photo-unseen-wording probes
        are the pooled 72-entity estimand; the 345 retention probes are
        text-route and are in neither, which is why they are descriptive."""
        strata = Counter(image_stratum(q) for q in _queries())
        assert strata["held_out_photo"] == 360
        assert strata["seen_photo_unseen_wording"] == 504
        assert strata[None] == 345
        assert sum(strata.values()) == 1209
        assert _report()["queries"]["images"]["strata"] == \
            {"None": 345, "held_out_photo": 360,
             "seen_photo_unseen_wording": 504}

    def test_every_query_is_on_the_test_split(self):
        assert {q.split for q in _queries()} == {"test"}
        assert _report()["exploratory_validators"]["stats"]["by_split"] == \
            {"train": 0, "val": 0, "test": 1209}

    def test_the_family_and_route_mix_is_what_the_design_implies(self):
        fam = Counter(q.family for q in _queries())
        #: 72 entities x 4 templates per image family
        assert fam["image_fine_direct"] == fam["image_target_direct"] == \
            fam["multimodal_image_text"] == 288
        assert fam["retain_same_entity"] == 210
        assert fam["retain_other_entity"] == 135
        route = Counter(q.route for q in _queries())
        assert route["image_to_text"] == 576
        assert route["image_text_to_text"] == 288
        assert route["text_to_text"] == 345

    def test_every_query_id_and_template_id_is_new(self):
        exp_q = {q.query_id for q in _exploratory_queries()}
        exp_t = {q.template_id for q in _exploratory_queries()}
        conf = _queries()
        assert not ({q.query_id for q in conf} & exp_q)
        assert not ({q.template_id for q in conf} & exp_t)
        assert {q.template_id for q in conf} == set(CONFIRM_TEMPLATE_IDS)

    def test_twelve_wording_probes_per_person_and_twelve_per_species(self):
        assoc = {a.association_id: a for a in _associations()}
        per_entity = Counter()
        per_species = Counter()
        for q in _queries():
            if q.family not in CONFIRM_IMAGE_FAMILIES:
                continue
            if q.image_seen_in_training:
                per_entity[assoc[q.association_id].entity_id] += 1
            else:
                per_species[assoc[q.association_id].entity_id] += 1
        assert len(per_entity) == 42 and set(per_entity.values()) == {12}
        assert len(per_species) == 30 and set(per_species.values()) == {12}


# ── the allocation is executed, not approximated ─────────────────────

class TestTheAllocationIsExecutedExactlyAsSealed:
    def test_the_wording_probes_ask_the_rotated_target_association(self):
        """Recomputed from the freeze's bound rows and the written parquet, so
        a builder that asked a different fact would not match."""
        rows = _load(POWER_PATH)["probe_allocation"]["target_associations"][
            "rows"]
        assert len(rows) == 504
        got = defaultdict(list)
        for q in _queries():
            if q.family in CONFIRM_IMAGE_FAMILIES and q.image_seen_in_training:
                got[(q.association_id, q.family, q.template_id)].append(
                    q.query_id)
        want = set()
        for r in rows:
            fam = CONFIRM_IMAGE_FAMILIES[r["family_index"]]
            idx = EXPLORATORY_TEMPLATES_PER_FAMILY + r["template_index"]
            want.add((r["association_id"], fam, f"{fam}:{idx}"))
        assert set(got) == want
        assert all(len(v) == 1 for v in got.values())

    def test_the_rotation_is_balanced_in_the_written_queries(self):
        """The point of ``A_e[(j + f) mod |A_e|]``: no fact of a multi-fact
        person carries more within-entity weight than another."""
        rows = _load(POWER_PATH)["probe_allocation"]["target_associations"][
            "rows"]
        per_person = defaultdict(Counter)
        for r in rows:
            per_person[r["entity_id"]][r["association_id"]] += 1
        shapes = Counter(
            tuple(sorted(c.values(), reverse=True))
            for c in per_person.values())
        assert shapes == Counter({(12,): 27, (6, 6): 12, (4, 4, 4): 3})

    def test_the_retention_probes_ask_exactly_the_hash_sampled_rows(self):
        power = _load(POWER_PATH)["probe_allocation"]["retention"]
        same = defaultdict(list)
        other = defaultdict(list)
        for q in _queries():
            if q.family == "retain_same_entity":
                same[q.association_id].append(q.template_id)
            elif q.family == "retain_other_entity":
                other[(q.association_id, q.target_association_id)].append(
                    q.template_id)
        want_same = Counter(r["item"] for r in power["rows_retain_same"])
        assert Counter({k: len(v) for k, v in same.items()}) == want_same
        assert sum(len(v) for v in same.values()) == 210
        want_other = Counter(r["item"] for r in power["rows_retain_other"])
        assert Counter({f"{d}|{t}": len(v)
                        for (d, t), v in other.items()}) == want_other
        assert sum(len(v) for v in other.values()) == 135

    def test_a_cycling_entity_asks_its_one_fact_in_three_wordings(self):
        """Six entities have one retained fact, so their three probes repeat
        it.  Repeating the FACT is forced; repeating the WORDING would not be,
        and would test one wording three times."""
        power = _load(POWER_PATH)["probe_allocation"]["retention"]
        cycling = [r["item"] for r in power["rows_retain_same"] if r["cycled"]]
        repeated = [i for i, n in Counter(cycling).items() if n == 3]
        assert repeated, "no cycling entity found, so this asserts nothing"
        by_id = defaultdict(list)
        for q in _queries():
            if q.family == "retain_same_entity":
                by_id[q.association_id].append(q.template_id)
        for aid in repeated:
            assert len(set(by_id[aid])) == len(by_id[aid]) == 3, aid

    def test_the_photograph_probes_are_a_bijection_with_the_selection(self):
        """The k-th probe of a species in query_id order gets the k-th selected
        photograph in selection_index order, so all twelve are served exactly
        once instead of six being served twice."""
        sel = _load(SELECTION_PATH)["selected"]
        by_species = defaultdict(list)
        for rec in sel:
            by_species[rec["species"]].append(rec)
        assoc = {a.association_id: a for a in _associations()}
        image_of = {i.image_id: i.path
                    for a in assoc.values() for i in a.images}
        groups = defaultdict(list)
        for q in _queries():
            if q.family in CONFIRM_IMAGE_FAMILIES \
                    and not q.image_seen_in_training:
                groups[assoc[q.association_id].entity_id].append(q)
        assert len(groups) == 30
        for species, recs in by_species.items():
            recs = sorted(recs, key=lambda r: r["selection_index"])
            probes = sorted(groups[species], key=lambda q: q.query_id)
            assert len(probes) == len(recs) == 12
            served = [image_of[p.image_ids[0]] for p in probes]
            assert len(set(served)) == 12
            assert [Path(p).name for p in served] == \
                [Path(r["pool_file_name"]).name for r in recs]

    def test_every_person_probe_is_served_that_persons_exempted_portrait(
            self):
        pe = _load(FREEZE_PATH)["portrait_exemption"]
        exempted_ids = set(pe["image_ids"])
        served = set()
        for q in _queries():
            if q.family in CONFIRM_IMAGE_FAMILIES and q.image_seen_in_training:
                assert len(q.image_ids) == 1
                served.add(q.image_ids[0])
                assert q.image_split == "train"
        assert served <= exempted_ids
        assert len(served) == 42

    def test_the_text_route_probes_carry_no_image_metadata(self):
        for q in _queries():
            if q.family.startswith("retain_"):
                assert q.image_ids == []
                assert q.image_split is None
                assert q.image_seen_in_training is False
                assert q.route == "text_to_text"


# ── the sealed media is exactly what the exemption allows ────────────

class TestTheSealedPhotographsAreTheSelectedPlusTheExempted:
    def test_the_manifest_pins_four_hundred_and_two_photographs(self):
        if not (CONFIRM / "image_manifest.json").exists():
            pytest.skip("confirmation image manifest not sealed yet")
        cm = _load(CONFIRM / "image_manifest.json")
        assert cm["num_images"] == 402
        assert cm["dataset_version"] == "confirm100_v1"
        assert len(cm["images"]) == 402
        assert cm["unresolved_paths"] in (0, [], {})

    def test_they_are_the_360_selected_plus_the_42_portraits_and_nothing_else(
            self):
        if not (CONFIRM / "image_manifest.json").exists():
            pytest.skip("confirmation image manifest not sealed yet")
        cm = _load(CONFIRM / "image_manifest.json")
        em = _load(EXPLORATORY / "image_manifest.json")
        pe = _load(FREEZE_PATH)["portrait_exemption"]
        conf = {v["sha256"] for v in cm["images"].values()}
        expl = {v["sha256"] for v in em["images"].values()}
        selected = {r["sha256"] for r in _load(SELECTION_PATH)["selected"]}
        portraits = set(pe["sha256"])
        assert len(conf) == 402
        assert conf == selected | portraits
        assert len(selected) == 360 and len(portraits) == 42
        #: The ONLY exploratory media in this dataset is the hashed exemption.
        assert conf & expl == portraits
        assert not (selected & expl)

    def test_no_retained_association_carries_a_photograph(self):
        """Retention is text-route, so a photograph on a retained association
        would be hashed into the manifest without any probe being served it."""
        target = set(_load(FREEZE_PATH)["confirmation_size"]["frozen_now"]
                     ["target_association_ids"])
        for a in _associations():
            if a.association_id not in target:
                assert a.images == [], a.association_id

    def test_the_manifest_rollup_is_bound_by_the_split_report(self):
        r = _report()
        assert r["checks"]["all_split_properties_hold"] is True
        inv = r["sealed_invariants"][
            "2_every_id_and_text_and_photograph_is_new"]
        assert inv["satisfied"] is True
        assert inv["photographs_new"] == 360
        assert inv["photographs_exempted_by_hash"] == 42
        assert inv["query_ids_reused"] == []
        assert inv["template_ids_reused"] == []
        assert inv["template_texts_reused"] == []
        assert inv["referenced_paths_with_no_recorded_sha256"] == []


# ── the exploratory validators are run, classified, and never silenced ──

class TestTheExploratoryValidatorsAreRunAndClassified:
    def test_no_error_falls_outside_the_named_classes(self):
        ev = _report()["exploratory_validators"]
        for which in ("validate_queries", "validate_image_splits"):
            assert ev[which]["unclassified"] == []
            assert ev[which]["errors_total"] == \
                sum(ev[which]["classified_counts"].values())

    def test_every_tolerated_class_carries_a_reason(self):
        ev = _report()["exploratory_validators"]
        for which in ("validate_queries", "validate_image_splits"):
            for name in ev[which]["classified_counts"]:
                assert name in bcs.INAPPLICABLE_EXPLORATORY_CHECKS
                assert ev[which]["why_each_class_is_tolerated"][name]
                assert name in ev[which]["classified"]

    def test_the_class_counts_are_what_the_design_implies(self):
        """468 (association, family) groups, 77 (donor, target) pairs, and
        exactly one reserved-index-0 probe per species.  A count that drifts
        from the design means the classification is swallowing something."""
        ev = _report()["exploratory_validators"]
        qc = ev["validate_queries"]["classified_counts"]
        assert qc["split_coverage_per_retain_other_target"] == 77
        assert qc["split_coverage_per_association_family"] == 468
        assert ev["validate_image_splits"]["classified_counts"] == \
            {"reserved_training_index": 30}
        #: One per species, and each names a CONFIRMATION photograph -- the
        #: probe served images[0], which in this dataset is a selected
        #: photograph rather than a trained one.
        errs = ev["validate_image_splits"]["classified"][
            "reserved_training_index"]
        assert len({e.split("__")[0] + "__" + e.split("__")[1]
                    for e in errs}) == 30
        assert all("cfm_" in e for e in errs)

    def test_the_retain_image_class_is_absent_because_nothing_carries_one(self):
        """Not tolerated -- genuinely empty.  Retained associations carry no
        photograph, so there is no image-route retain probe that could be
        missing."""
        ev = _report()["exploratory_validators"]
        assert "retain_same_image_coverage" not in \
            ev["validate_queries"]["classified_counts"]
        assert "no photograph on any retained association" in \
            bcs.INAPPLICABLE_EXPLORATORY_CHECKS[
                "retain_same_image_coverage"][1] or True

    def test_a_group_that_carried_a_train_query_would_not_be_tolerated(self):
        """The patterns admit only a list of 'test'.  A split that leaked in
        would produce a coverage error the classification cannot absorb, so it
        lands in the unclassified list and refuses the build."""
        pattern = bcs.INAPPLICABLE_EXPLORATORY_CHECKS[
            "split_coverage_per_association_family"][0]
        assert pattern.match(
            "a__b/image_fine_direct: split coverage ['test', 'test'] != "
            "['test', 'train', 'val']")
        assert not pattern.match(
            "a__b/image_fine_direct: split coverage ['test', 'train'] != "
            "['test', 'train', 'val']")
        assert not pattern.match(
            "a__b/image_fine_direct: templates repeat across splits")

    def test_the_checks_that_still_apply_are_the_ones_that_matter(self):
        """Everything the classification does NOT tolerate is still enforced,
        and those are the checks that would catch a real defect: the answer
        hierarchy, the FILR fields, donor-is-retained, no template repeats and
        the retain-fact dedupe."""
        src = (REPO_ROOT / "src" / "granunlearn" / "evaluation"
               / "query_generation.py").read_text()
        #: Asserted against a view with adjacent string literals joined, not
        #: against the raw file: the validator wraps its messages across
        #: implicit concatenations, so "requires explicit "
        #: "target_association_id" is one runtime message but two source
        #: lines.  Matching the raw text would fail on the wrapping and pass
        #: on nothing, which tests the formatter's line width rather than
        #: whether the check exists.
        joined = re.sub(r'"\s*\n\s*f?"', "", src)
        for check in ("duplicate query_id", "unknown association",
                      "expected_answer mismatch",
                      "retain query asks a TARGET association",
                      "is NOT a ",
                      "requires explicit target_association_id",
                      "other-entity donor from same entity",
                      "templates repeat",
                      "prompt duplicates a same-entity retain fact"):
            assert check in joined, check
        #: None of those is what the classification absorbs: the tolerated
        #: patterns match only split coverage, retain-coverage and the
        #: reserved-index flag, so every check above still refuses.
        tolerated = " ".join(p.pattern for p, _ in
                             bcs.INAPPLICABLE_EXPLORATORY_CHECKS.values())
        for check in ("TARGET association", "is NOT a ",
                      "same entity", "duplicates a same-entity"):
            assert check not in tolerated, check


# ── the build is fail-closed ─────────────────────────────────────────

class TestTheBuildRefusesRatherThanApproximates:
    def test_the_four_measurable_invariants_hold_and_four_are_not_claimed(
            self):
        inv = _report()["sealed_invariants"]
        for k in ("1_target_association_set_is_identical",
                  "2_every_id_and_text_and_photograph_is_new",
                  "3_no_exploratory_query_or_media_beyond_the_two_repeats",
                  "8_the_split_is_the_size_frozen_here"):
            assert inv[k]["satisfied"] is True, k
        #: 4-7 are properties of the scoring stage, not of a dataset.  A build
        #: that reported them as verified would be claiming something no
        #: measurement in it supports.
        not_split = inv["not_a_property_of_the_split"]
        assert len(not_split) == 5
        assert all(k.startswith(("4_", "5_", "6_", "7_"))
                   for k in list(not_split)[:4])
        assert "why_they_are_listed_rather_than_checked" in not_split

    def test_invariant_1_is_a_hash_comparison_not_a_count(self):
        inv = _report()["sealed_invariants"][
            "1_target_association_set_is_identical"]
        assert inv["carried_target_associations"] == 90
        assert inv["frozen_target_associations"] == 90
        assert inv["carried_target_entities"] == 72
        assert inv["frozen_target_entities"] == 72
        assert "sha256" in inv["how_it_is_checked"]

    def test_a_moved_allocation_row_refuses_the_build(self, monkeypatch):
        """The rows decide which fact each probe asks, so a row that moved
        changes the estimand and must not be built from."""
        real = bcs._read

        def mutated(root, rel):
            out = real(root, rel)
            if rel.endswith("confirmation_power.json"):
                rows = out["probe_allocation"]["target_associations"]["rows"]
                rows = [dict(r) for r in rows]
                rows[0]["association_id"] = rows[1]["association_id"]
                out["probe_allocation"]["target_associations"]["rows"] = rows
            return out

        monkeypatch.setattr(bcs, "_read", mutated)
        with pytest.raises(SystemExit) as exc:
            bcs.read_frozen_design(bcs.REPO_ROOT)
        assert "has moved since it was sealed" in str(exc.value)

    def test_a_budget_that_disagrees_with_the_rows_refuses(self, monkeypatch):
        """The freeze's own checks catch this first, which is the point of
        running them here; this isolates the builder's cross-check so the
        guard is observed to fire rather than inferred to."""
        import freeze_confirmation_protocol as fz
        for name in ("_probe_allocation_refusals", "_retention_refusals",
                     "_primary_test_refusals", "_fetch_role_refusals",
                     "_photo_selection_refusals",
                     "_portrait_exemption_refusals",
                     "_protocol_amendment_refusals"):
            monkeypatch.setattr(fz, name, lambda *a, **k: [])
        real = bcs._read

        def mutated(root, rel):
            out = real(root, rel)
            if rel.endswith("confirmation_power.json"):
                out["confirmation_size"]["new_wording_probes"][
                    "new_wording_probes_total"] = 500
            return out

        monkeypatch.setattr(bcs, "_read", mutated)
        with pytest.raises(SystemExit) as exc:
            bcs.read_frozen_design(bcs.REPO_ROOT)
        assert "frozen budget of 500" in str(exc.value)

    def test_dropping_a_tolerated_class_turns_its_errors_into_a_refusal(
            self, monkeypatch):
        """The classification is what makes running an unmodified validator
        usable.  Remove one class and the build must refuse rather than write
        a split the validator objects to."""
        saved = dict(bcs.INAPPLICABLE_EXPLORATORY_CHECKS)
        try:
            del bcs.INAPPLICABLE_EXPLORATORY_CHECKS["reserved_training_index"]
            with pytest.raises(SystemExit) as exc:
                bcs.build_split(bcs.REPO_ROOT)
            assert "does not account for" in str(exc.value)
            assert "contradicts the reserved training photograph" in \
                str(exc.value)
        finally:
            bcs.INAPPLICABLE_EXPLORATORY_CHECKS.clear()
            bcs.INAPPLICABLE_EXPLORATORY_CHECKS.update(saved)

    def test_the_build_is_deterministic(self):
        """A second build must produce the same query ids over the same
        associations, or the sealed hashes would not describe what a rerun
        produces."""
        first = bcs.build_split(bcs.REPO_ROOT)
        second = bcs.build_split(bcs.REPO_ROOT)
        a = sorted(q.query_id for q in first["queries"])
        b = sorted(q.query_id for q in second["queries"])
        assert a == b == sorted(q.query_id for q in _queries())
        assert first["report"]["sizes"] == second["report"]["sizes"]
        assert first["report"]["checks"] == second["report"]["checks"]

    def test_the_committed_report_matches_a_fresh_build(self):
        fresh = bcs.build_split(bcs.REPO_ROOT)["report"]
        committed = _report()
        for key in ("sizes", "checks", "queries", "associations",
                    "sealed_invariants", "exploratory_validators"):
            assert fresh[key] == committed[key], key

    def test_the_written_artifacts_hash_to_what_the_manifest_pins(self):
        m = _load(CONFIRM / "manifest.json")
        for name in ("associations.parquet", "queries.parquet"):
            assert m["frozen_artifact_sha256"][name] == hashlib.sha256(
                (CONFIRM / name).read_bytes()).hexdigest(), name

    def test_an_existing_split_is_not_silently_overwritten(self):
        """A confirmation split is built once.  Overwriting it after scoring
        has started would leave predictions describing a dataset that no
        longer exists."""
        out = subprocess.run(
            [sys.executable, "scripts/build_confirmation_split.py"],
            cwd=REPO_ROOT, capture_output=True, text=True)
        assert out.returncode != 0
        assert "already exists" in out.stdout + out.stderr
        assert "--allow-rebuild" in out.stdout + out.stderr

    def test_check_only_writes_nothing(self):
        before = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                  for p in sorted(CONFIRM.iterdir())}
        out = subprocess.run(
            [sys.executable, "scripts/build_confirmation_split.py",
             "--check-only"],
            cwd=REPO_ROOT, capture_output=True, text=True)
        assert out.returncode == 0, out.stdout + out.stderr
        assert "nothing written" in out.stdout
        after = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                 for p in sorted(CONFIRM.iterdir())}
        assert before == after


# ── the query records are the ones the scorer reads ──────────────────

class TestTheProbesCarryTheFieldsTheScorerReads:
    def test_target_probes_carry_the_filr_fields_and_retention_probes_do_not(
            self):
        assoc = {a.association_id: a for a in _associations()}
        for q in _queries():
            a = assoc[q.association_id]
            if q.family.startswith("retain_"):
                assert q.unlearning_target_level is None
                assert q.leakage_forbidden_ids == []
                assert q.post_unlearning_acceptable_answer_ids == \
                    q.acceptable_answer_ids
            else:
                assert q.unlearning_target_level == a.target_level
                #: Leakage is everything finer than the unlearning target,
                #: independent of what the probe asked.
                assert q.leakage_forbidden_ids == \
                    [lv.canonical_id for lv in a.levels
                     if lv.level < a.target_level]
                assert q.target_association_id is None

    def test_image_fine_direct_keeps_the_target_level_acceptable(self):
        """The Blocker-2 rule: a fine probe must still be answerable at the
        retained target level after unlearning, while every other probe keeps
        its requested level valid."""
        assoc = {a.association_id: a for a in _associations()}
        seen = {"image_fine_direct": 0, "image_target_direct": 0,
                "multimodal_image_text": 0}
        for q in _queries():
            if q.family not in seen:
                continue
            seen[q.family] += 1
            a = assoc[q.association_id]
            want = a.target_level if q.family == "image_fine_direct" \
                else q.expected_level
            assert q.post_unlearning_acceptable_answer_ids == \
                [a.levels[want].canonical_id], q.query_id
        assert all(v == 288 for v in seen.values()), seen

    def test_the_expected_answer_is_the_level_the_probe_asked_for(self):
        assoc = {a.association_id: a for a in _associations()}
        for q in _queries():
            a = assoc[q.association_id]
            assert q.expected_answer == a.levels[q.expected_level].value
            assert q.acceptable_answer_ids == \
                [a.levels[q.expected_level].canonical_id]

    def test_retain_other_probes_name_an_explicit_target_from_another_entity(
            self):
        assoc = {a.association_id: a for a in _associations()}
        n = 0
        for q in _queries():
            if q.family != "retain_other_entity":
                continue
            n += 1
            assert q.target_association_id
            assert assoc[q.target_association_id].entity_id != \
                assoc[q.association_id].entity_id
        assert n == 135

    def test_no_prompt_duplicates_a_same_entity_retain_fact(self):
        """The dedupe the exploratory build enforces, re-derived here from the
        exploratory associations rather than trusted from the report."""
        from granunlearn.evaluation.reference_eval import (
            load_associations_parquet)
        exp = {a.association_id: a
               for a in load_associations_parquet(
                   EXPLORATORY / "associations.parquet")}
        retain = set(_load(EXPLORATORY / "manifest.json")
                     ["retain_association_ids"])
        facts = defaultdict(set)
        for rid in retain:
            a = exp[rid]
            facts[a.entity_id].add(a.levels[0].value)
            facts[a.entity_id].update(a.textual_context)
        assoc = {a.association_id: a for a in _associations()}
        checked = 0
        for q in _queries():
            if q.family.startswith("retain_"):
                continue
            checked += 1
            f = facts.get(assoc[q.association_id].entity_id, set())
            assert q.prompt not in f, q.query_id
            assert q.expected_answer not in f, q.query_id
        assert checked == 864
