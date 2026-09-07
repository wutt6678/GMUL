"""Iteration 11C stage 2: the confirmation protocol freeze.

These tests exist because a protocol freeze that is not enforced is a
comment.  The value of freezing before inference is entirely in the
refusals: a freeze that binds a checkpoint which has moved, or that lets a
second copy of itself be written after scoring, is worse than no freeze at
all because it looks like discipline.

Two kinds of assertion live here.  The committed freeze is checked against
the evidence it claims to be derived from — the 11R prediction manifest, the
selection report, the power analysis — because a parameter written into the
freeze by hand instead of read from its source is a second copy that can
diverge.  The refusal paths are exercised by mutation, because a gate that
has never been observed to fire cannot be trusted to.
"""

from __future__ import annotations

import inspect
import json
import shlex
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
REPORTS = REPO_ROOT / "data" / "reports"
FREEZE_PATH = REPORTS / "mllmu_pilot100_confirmation_freeze.json"
MANIFEST_PATH = REPORTS / "mllmu_pilot100_prediction_manifest.json"
POWER_PATH = REPORTS / "mllmu_pilot100_confirmation_power.json"
SELECTION_PATH = REPORTS / "mllmu_pilot100_unlearning_selection.json"

sys.path.insert(0, str(REPO_ROOT / "scripts"))
import freeze_confirmation_protocol as fz  # noqa: E402
from freeze_confirmation_protocol import (  # noqa: E402
    ANALYSIS_SCRIPTS,
    CONFIRMATION_STATES,
    VOLATILE_FREEZE_FIELDS,
    build_freeze,
)
from power_analysis_confirmation import (  # noqa: E402
    ALPHA_ONE_SIDED,
    BOOTSTRAP_SEED,
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


def _load(path: Path = FREEZE_PATH) -> dict:
    if not path.exists():
        pytest.skip(f"committed evidence not present: {path}")
    return json.loads(path.read_text())


def _adapters_present() -> bool:
    """Whether this checkout holds the adapter bytes the freeze binds.

    The adapters are gitignored, so a fresh clone has the committed freeze
    and not the checkpoints.  Tests that MUTATE contract resolution need real
    directories to mutate: without them every adapter hashes to the same
    empty-file-map contract, the "checkpoint has moved" refusal fires first,
    and the mutation under test is never reached.  Tests that only read the
    committed freeze need nothing and must keep running everywhere.
    """
    f = _load()
    dirs = [REPO_ROOT / e["adapter_dir"]
            for e in f["checkpoints"].values() if e.get("adapter_dir")]
    return bool(dirs) and all(d.is_dir() for d in dirs)


def _require_adapters() -> None:
    if not _adapters_present():
        pytest.skip(
            "the adapter directories the freeze binds are gitignored and "
            "absent from this checkout, so the contract mutation under test "
            "cannot be distinguished from a missing checkpoint; see "
            "TestTheFreezeFailsClosedWithoutTheArtifacts for what IS "
            "asserted here")


# ── the checkpoints are bound by contract, and the gate fires ──────

class TestTheCheckpointsAreBoundByContract:
    def test_every_scored_state_matches_what_11r_actually_scored(self):
        """The freeze must bind the SAME bytes that produced the results
        being confirmed, not merely some adapter sitting in the expected
        directory.  A checkpoint that moved between selection and
        confirmation would otherwise be invisible."""
        f = _load()
        m = _load(MANIFEST_PATH)
        recorded = {r["checkpoint_id"]: r.get("adapter_contract_sha256")
                    for r in m["predictions"]}
        assert f["refusals"] == []
        assert set(f["checkpoints"]) == set(CONFIRMATION_STATES)
        for state, entry in f["checkpoints"].items():
            cid = entry["checkpoint_id"]
            contract = entry["adapter_contract"]
            assert contract is not None, state
            assert contract["sha256"] == recorded[cid], (state, cid)
            assert contract["missing_files"] == [], state

    def test_the_contract_covers_the_configuration_not_only_the_weights(self):
        """Iteration 11R1's finding: hashing adapter_model.safetensors alone
        leaves rank, alpha, dropout and target_modules unbound, and those
        live in adapter_config.json.  A freeze that inherited that gap would
        certify an adapter whose behaviour had changed."""
        f = _load()
        for state, entry in f["checkpoints"].items():
            files = entry["adapter_contract"]["files"]
            assert set(files) == {"adapter_model.safetensors",
                                  "adapter_config.json"}, (state, files)
            assert len(set(files.values())) == 2, state

    def test_b3s_lambda_is_read_from_the_selection_report(self):
        f = _load()
        s = _load(SELECTION_PATH)
        b3 = f["checkpoints"]["B3"]
        cid = b3["checkpoint_id"]
        groups = s["candidates"][cid]["config"]["groups"]
        want = next(g["weight"] for g in groups if g["name"] == "fine_target")
        assert b3["granularity_lambda"] == want == 0.5
        assert b3["num_optimizer_steps"] == \
            s["candidates"][cid]["config"]["num_optimizer_steps"]
        assert b3["unlearning_groups"] == groups

    def test_a_moved_checkpoint_refuses_the_freeze(self, monkeypatch):
        """The mutation that matters.  Returning a contract that does not
        match the manifest must produce a refusal, not a freeze that quietly
        binds the wrong adapter."""
        _require_adapters()
        real = fz.adapter_contract

        def shifted(adapter_dir):
            out = real(adapter_dir)
            if out is not None:
                out = dict(out, sha256="0" * 64)
            return out

        monkeypatch.setattr(fz, "adapter_contract", shifted)
        f = build_freeze(REPO_ROOT, "pilot100")
        assert f["refusals"], "a moved checkpoint did not refuse the freeze"
        assert any("does not match" in r and "11R sidecars" in r
                   for r in f["refusals"]), f["refusals"]

    def test_an_unresolvable_checkpoint_refuses_the_freeze(self, monkeypatch):
        monkeypatch.setattr(fz, "adapter_contract", lambda _d: None)
        f = build_freeze(REPO_ROOT, "pilot100")
        assert any("no adapter contract could be resolved" in r
                   for r in f["refusals"]), f["refusals"]

    def test_an_incomplete_adapter_directory_refuses_the_freeze(self,
                                                               monkeypatch):
        _require_adapters()
        real = fz.adapter_contract

        def missing_config(adapter_dir):
            out = real(adapter_dir)
            return None if out is None else dict(
                out, missing_files=["adapter_config.json"])

        monkeypatch.setattr(fz, "adapter_contract", missing_config)
        f = build_freeze(REPO_ROOT, "pilot100")
        assert any("adapter contract is missing" in r
                   for r in f["refusals"]), f["refusals"]


# ── excluding M_F is a measurement, not an inheritance ─────────────

class TestTheNoOpExclusionIsMeasured:
    def test_b0_is_mf_by_adapter_contract(self):
        """11R proved B0 and M_F produce identical OUTPUTS on the
        exploratory test split.  That is a consequence of identical weights
        under an identical batch layout, and the confirmation gets a new
        split — so the exclusion is re-derived from the bytes here."""
        f = _load()
        b = f["b0_is_the_no_op"]
        assert b["measured"] is True
        assert b["identical"] is True
        assert b["b0_adapter_contract_sha256"] == \
            b["mf_adapter_contract_sha256"]
        assert b["b0_adapter_contract_sha256"] == \
            f["checkpoints"]["B0"]["adapter_contract"]["sha256"]

    def test_mf_is_not_scored_and_says_why(self):
        f = _load()
        assert "MF" not in f["states"]["scored"]
        assert "MF" in f["states"]["excluded"]
        assert "b0_is_the_no_op" in f["states"]["excluded"]["MF"]

    def test_a_distinct_mf_would_refuse_rather_than_be_dropped(self,
                                                               monkeypatch):
        """If B0 were NOT a copy of M_F, dropping M_F would silently remove a
        distinct state from the confirmation.  The freeze must refuse
        instead."""
        _require_adapters()
        real = fz.adapter_contract

        def distinguish_mf(adapter_dir):
            out = real(adapter_dir)
            if out is None:
                return None
            # M_F resolves to its own reference directory; shift only that one
            return (dict(out, sha256="f" * 64)
                    if str(adapter_dir).endswith("/MF/adapters") else out)

        monkeypatch.setattr(fz, "adapter_contract", distinguish_mf)
        f = build_freeze(REPO_ROOT, "pilot100")
        assert any("cannot be excluded" in r for r in f["refusals"]), \
            f["refusals"]

    def test_every_excluded_state_has_a_reason(self):
        f = _load()
        for state, reason in f["states"]["excluded"].items():
            assert reason and len(reason) > 20, (state, reason)
            assert state not in f["states"]["scored"], state
        for state in f["states"]["scored"]:
            assert f["states"]["scored_because"][state]


# ── a checkout without the artifacts refuses rather than degrades ───

class TestTheFreezeFailsClosedWithoutTheArtifacts:
    """What IS asserted in the environment the mutation tests skip in.

    The adapters are gitignored, so a fresh clone — which is what CI runs —
    has the committed freeze and reports but none of the bytes they were
    derived from.  Four tests above need real adapter directories to mutate
    and skip there.  This class is why that is not a hole: absence is
    SIMULATED rather than waited for, so every assertion here runs in every
    environment, including the artifact-bearing one.

    It exists because the naive behaviour was measured, not imagined.  An
    absent adapter directory still yields a contract — zero files hashed, a
    roll-up sha256 that looks like any other — and two such contracts are
    EQUAL, so the "is B0 the no-op copy of M_F" measurement came back True
    from a checkout holding no adapters at all.
    """

    @staticmethod
    def _build_without_adapters(monkeypatch, tmp_path):
        """Point contract resolution at a directory that does not exist.

        That is exactly what a fresh clone looks like to ``build_freeze``:
        ``resolve_adapter_dir`` returns a path, the path is not there, and
        ``adapter_contract`` hashes zero files.
        """
        gone = tmp_path / "adapters" / "not-checked-out"
        assert not gone.exists()
        monkeypatch.setattr(fz, "resolve_adapter_dir",
                            lambda *_a, **_k: gone)
        return build_freeze(REPO_ROOT, "pilot100")

    def test_every_bound_state_refuses(self, monkeypatch, tmp_path):
        f = self._build_without_adapters(monkeypatch, tmp_path)
        assert f["refusals"], "absent adapters did not refuse the freeze"
        for state in CONFIRMATION_STATES + ("MF",):
            assert any(r.startswith(f"{state}:") for r in f["refusals"]), \
                (state, f["refusals"])

    def test_the_refusal_diagnoses_absence_not_a_moved_checkpoint(
            self, monkeypatch, tmp_path):
        """The refusal must say the adapters are not there.  "The checkpoint
        has moved since the results being confirmed were produced" is a
        confident diagnosis of the wrong thing, and a fresh clone is the one
        place it is most likely to be read and acted on."""
        f = self._build_without_adapters(monkeypatch, tmp_path)
        assert any("adapter contract is missing" in r and "gitignored" in r
                   for r in f["refusals"]), f["refusals"]
        assert not any("has moved" in r for r in f["refusals"]), \
            f["refusals"]

    def test_absence_does_not_prove_b0_is_the_no_op(self, monkeypatch,
                                                    tmp_path):
        """The trap, stated as an assertion: the two hashes ARE equal, and
        the measurement must still be False, because equality of two empty
        file maps is not evidence about any adapter."""
        from granunlearn.evaluation.prediction_provenance import (
            _canonical_rollup)
        f = self._build_without_adapters(monkeypatch, tmp_path)
        b = f["b0_is_the_no_op"]
        empty = _canonical_rollup({})
        assert b["b0_adapter_contract_sha256"] == empty
        assert b["mf_adapter_contract_sha256"] == empty
        assert b["both_contracts_hashed_real_files"] is False
        assert b["identical"] is False, \
            "two empty file maps were accepted as a measurement"
        assert any("no-op check could not be performed" in r
                   for r in f["refusals"]), f["refusals"]

    def test_the_cli_writes_no_file_at_all(self, monkeypatch, tmp_path,
                                           capsys):
        """The strongest form of failing closed: no artifact is produced, so
        there is nothing for a later stage to mistake for a freeze."""
        gone = tmp_path / "adapters" / "not-checked-out"
        monkeypatch.setattr(fz, "resolve_adapter_dir",
                            lambda *_a, **_k: gone)
        monkeypatch.setattr(fz, "_find_repo_root", lambda _cwd: REPO_ROOT)
        out = tmp_path / "freeze.json"
        monkeypatch.setattr(sys, "argv",
                            ["freeze_confirmation_protocol.py",
                             "--tag", "pilot100", "--output", str(out)])
        assert fz.main() == 1
        assert not out.exists(), "a freeze was written from absent adapters"
        assert "REFUSED" in capsys.readouterr().out

    def test_the_committed_freeze_was_not_built_this_way(self):
        """The complement of the four skips, and it runs in a fresh clone
        because it reads only committed JSON: the freeze on record must show
        that it was derived where the bytes were present."""
        from granunlearn.evaluation.prediction_provenance import (
            _canonical_rollup)
        f = _load()
        b = f["b0_is_the_no_op"]
        assert b["both_contracts_hashed_real_files"] is True
        assert b["identical"] is True
        assert "empty file map" in \
            b["why_completeness_is_part_of_the_measurement"]
        empty = _canonical_rollup({})
        for state in CONFIRMATION_STATES:
            c = f["checkpoints"][state]["adapter_contract"]
            assert c["files"], state
            assert c["sha256"] != empty, state


# ── the generation settings come from the evidence ─────────────────

class TestTheGenerationSettingsComeFromTheEvidence:
    def test_they_are_read_from_the_committed_predictions(self):
        f = _load()
        m = _load(MANIFEST_PATH)
        configs = {json.dumps(r["generation_config"], sort_keys=True)
                   for r in m["predictions"]}
        assert len(configs) == 1, "11R evidence is not single-layout"
        want = json.loads(configs.pop())
        for key, value in want.items():
            assert f["generation"][key] == value, key

    def test_the_module_defaults_still_agree_with_the_evidence(self):
        """reference_eval's defaults are what a future run would use; if they
        drift from the frozen config the sidecar fingerprint stops describing
        the generator."""
        f = _load()
        assert f["generation"]["module_defaults_match"] is True
        cross = f["generation"]["module_defaults_cross_check"]
        assert cross["max_new_tokens"] == f["generation"]["max_new_tokens"]
        assert cross["max_length"] == f["generation"]["max_length"]
        assert cross["max_image_pixels"] == f["generation"]["max_image_pixels"]

    def test_decoding_is_greedy_and_the_layout_is_single(self):
        """Stage 5 requires one uniform batch layout for all states, because
        batched greedy decoding is not bit-stable across compositions."""
        f = _load()
        assert f["generation"]["do_sample"] is False
        assert "ONE uniform layout" in f["generation"]["batch_layout"]
        assert f["generation"]["base_model_revision"]

    def test_an_ambiguous_generation_history_refuses(self, monkeypatch):
        """If the 30 committed predictions had not all been generated under
        one configuration, there would be no single layout for the
        confirmation to inherit, and guessing one would put decoding noise
        inside the paired difference."""
        real = fz._load_report

        def two_layouts(reports, name):
            data = real(reports, name)
            if name.endswith("prediction_manifest.json"):
                data = json.loads(json.dumps(data))
                data["predictions"][0]["generation_config"][
                    "max_new_tokens"] = 64
            return data

        monkeypatch.setattr(fz, "_load_report", two_layouts)
        f = build_freeze(REPO_ROOT, "pilot100")
        assert any("not all generated under one configuration" in r
                   for r in f["refusals"]), f["refusals"]

    def test_drifted_module_defaults_refuse(self, monkeypatch):
        monkeypatch.setattr(fz, "DEFAULT_MAX_NEW_TOKENS", 64)
        f = build_freeze(REPO_ROOT, "pilot100")
        assert any("defaults no longer match" in r for r in f["refusals"]), \
            f["refusals"]


# ── the margin's ROLE is stated, not implied ───────────────────────

class TestTheMarginRoleIsStatedNotImplied:
    def test_delta_is_preserved_but_is_not_a_tested_margin(self):
        """Stage 2 said preserve delta = 0.05; the 11C-2a decisions removed
        every hypothesis test that would have used it.  Recording it as "the
        margin" without saying so would imply a TOST that will never run."""
        f = _load()
        m = f["scorer"]["equivalence_margin"]
        assert m["value"] == 0.05
        assert m["read_from"].endswith("EQUIVALENCE_MARGIN_TGA")
        assert "not a tested margin" in m["role"].lower()
        assert "yardstick" in m["role"].lower()
        assert "no TOST" in m["what_it_is_used_for"]

    def test_the_margin_role_points_at_the_decisions_that_caused_it(self):
        f = _load()
        p = _load(POWER_PATH)
        m = f["scorer"]["equivalence_margin"]
        assert m["retention_decision"] == \
            p["retention_claim_decision"]["decision"]
        assert m["retention_declared_margin"] is None
        assert m["mg_decision"] == p["mg_equivalence_decision"]["decision"]
        assert m["mg_equivalence_test_run"] is False

    def test_the_module_constant_is_the_one_bound(self):
        from evaluate_pilot100_final import EQUIVALENCE_MARGIN_TGA
        assert _load()["scorer"]["equivalence_margin"]["value"] == \
            EQUIVALENCE_MARGIN_TGA


# ── the analysis parameters are read, not restated ─────────────────

class TestTheAnalysisParametersAreReadNotRestated:
    def test_the_bootstrap_settings_come_off_the_signature(self):
        from granunlearn.evaluation.paired_ci import paired_rate_diff_ci
        sig = inspect.signature(paired_rate_diff_ci).parameters
        f = _load()
        b = f["analysis"]["bootstrap"]
        assert b["n_bootstrap"] == sig["n_bootstrap"].default
        assert b["ci_level"] == sig["ci_level"].default
        assert b["seed"] == sig["seed"].default
        assert "inspect.signature" in b["read_from"]

    def test_the_ci_unit_and_both_averaging_units_are_declared(self):
        f = _load()
        a = f["analysis"]
        assert "entity" in a["ci_unit"]
        assert "percentile bootstrap" in a["ci_method"]
        assert set(a["averaging_units"]) == {"entity_macro", "row_micro"}
        # the two units are not interchangeable, and the freeze must say why
        assert "unequal numbers of probes" in a["averaging_units"]["row_micro"]

    def test_the_icc_seed_is_the_preregistration_seed(self):
        assert _load()["analysis"]["icc_bootstrap_seed"] == BOOTSTRAP_SEED

    def test_the_paired_metric_set_is_the_frozen_one(self):
        from granunlearn.evaluation.paired_ci import PAIRED_METRICS
        assert _load()["analysis"]["paired_metrics"] == list(PAIRED_METRICS)

    def test_alpha_matches_the_power_analysis(self):
        assert _load()["analysis"]["alpha_one_sided"] == ALPHA_ONE_SIDED


# ── the claim family is the one the power analysis produced ────────

class TestTheClaimFamilyMatchesThePowerAnalysis:
    def test_the_primary_family_is_two_superiority_claims(self):
        f = _load()
        c = f["claims"]
        assert c["primary_family"] == list(PRIMARY_FAMILY)
        assert c["k"] == 2
        assert c["multiplicity"] == "Holm"
        assert all(v == "superiority"
                   for v in c["primary_claim_kinds"].values())

    def test_the_holm_thresholds_are_the_declared_sequence(self):
        c = _load()["claims"]
        assert c["holm_thresholds"] == [0.025, 0.05]
        assert c["worst_case_alpha_for_a_single_claim"] == 0.025

    def test_one_familywise_alpha_is_declared_and_everything_derives_from_it(
            self):
        """Iteration 11C-R1's finding #4 asked the preregistration to state
        whether familywise alpha is 0.05 or 0.025.  It states 0.05, and each
        level below must be that rate divided by something rather than a
        second independent number: one unadjusted claim sits at familywise/2,
        Holm's worst case for one of k claims at familywise/k."""
        f = _load()
        a, c = f["analysis"], f["claims"]
        assert a["familywise_alpha"] == FAMILYWISE_ALPHA == 0.05
        assert c["familywise_alpha"] == FAMILYWISE_ALPHA
        assert a["alpha_one_sided"] == ALPHA_ONE_SIDED == FAMILYWISE_ALPHA / 2
        k = len(PRIMARY_FAMILY)
        assert c["k"] == k
        assert c["worst_case_alpha_for_a_single_claim"] == \
            round(FAMILYWISE_ALPHA / k, 6)
        assert c["holm_thresholds"] == [
            round(FAMILYWISE_ALPHA / (k - i), 6) for i in range(k)]
        assert c["thresholds_apply_to"] == "one-sided p-values"
        assert "ONE-SIDED" in a["alpha_convention"]

    def test_the_sizing_used_the_declared_alpha_and_not_half_of_it(self):
        """The regression itself.  The threshold was one-sided 0.025 while
        the sizing function halved its alpha again, so the cluster
        requirement was computed at one-sided 0.0125 (z = 2.2414) and
        published beside a preregistration that said 0.025 (z = 1.959964).
        Conservative, but it was not the requirement OF the declared test, so
        the two could not be checked against each other."""
        c = _load()["claims"]
        declared = FAMILYWISE_ALPHA / len(PRIMARY_FAMILY)
        assert c["sizing_alpha_one_sided"] == round(declared, 6) == 0.025
        assert c["sizing_critical_value_z"] == round(z(1 - declared), 6) \
            == 1.959964
        assert c["sizing_critical_value_z"] != round(z(1 - declared / 2), 6), \
            "the sizing critical value is the halved-alpha reading"

    def test_the_cluster_requirement_is_read_from_the_report_at_that_alpha(
            self):
        p = _load(POWER_PATH)
        h = p["holm_primary_family"]
        c = _load()["claims"]
        assert h["familywise_alpha"] == FAMILYWISE_ALPHA
        assert h["sizing_alpha_one_sided"] == c["sizing_alpha_one_sided"]
        assert h["critical_value_z"] == c["sizing_critical_value_z"]
        assert c["clusters_required_at_that_threshold"] == {
            n: v["n_at_holm_worst_case_alpha_power80"]
            for n, v in h["per_claim"].items()}
        # 5 and 4 rather than the 7 and 5 the halved alpha produced; both
        # are far under the ceiling, so correcting the convention clarifies
        # the requirement without changing any conclusion
        assert c["clusters_required_at_that_threshold"] == \
            {"B3_minus_B0:tga": 5, "B3_minus_B0:filr": 4}
        ceiling = c["cluster_ceiling_of_the_primary_claims"]
        assert ceiling == 72
        assert all(v < ceiling
                   for v in c["clusters_required_at_that_threshold"].values())

    def test_a_report_sized_at_the_wrong_alpha_refuses(self, monkeypatch):
        """If the sizing ever drifts back to halving its alpha, the freeze
        must refuse rather than publish a requirement that is not the
        requirement of the test it sits beside."""
        real = fz._load_report

        def halved(reports, name):
            out = real(reports, name)
            if not name.endswith("confirmation_power.json"):
                return out
            out = dict(out)
            h = dict(out["holm_primary_family"])
            h["sizing_alpha_one_sided"] = round(
                FAMILYWISE_ALPHA / len(PRIMARY_FAMILY) / 2, 6)
            h["critical_value_z"] = round(
                z(1 - h["sizing_alpha_one_sided"]), 6)
            out["holm_primary_family"] = h
            return out

        monkeypatch.setattr(fz, "_load_report", halved)
        f = build_freeze(REPO_ROOT, "pilot100")
        assert any("not the same test" in r for r in f["refusals"]), \
            f["refusals"]
        assert any("critical value" in r for r in f["refusals"]), \
            f["refusals"]

    def test_every_demoted_claim_is_listed_with_its_reason(self):
        f = _load()
        p = _load(POWER_PATH)
        d = f["claims"]["descriptive_and_why"]
        assert set(d) == {"B3_minus_B0:retain_same",
                          "B3_minus_B0:retain_other",
                          "B3_minus_MG:tga", "B3_minus_MG:filr"}
        ret = p["retention_claim_decision"]["decision"]
        mg = p["mg_equivalence_decision"]["decision"]
        assert d["B3_minus_B0:retain_same"] == ret
        assert d["B3_minus_MG:tga"] == mg
        # and no demoted claim is also a primary one
        assert not set(d) & set(f["claims"]["primary_family"])

    def test_a_disagreement_with_the_power_report_refuses(self, monkeypatch):
        """The freeze inherits its family from the committed power analysis.
        If the module and the report ever disagree, that is a protocol
        contradiction and must refuse rather than pick one."""
        monkeypatch.setattr(fz, "PRIMARY_FAMILY", ("B3_minus_B0:tga",))
        f = build_freeze(REPO_ROOT, "pilot100")
        assert any("primary family" in r for r in f["refusals"]), \
            f["refusals"]


# ── the primary estimand is named once ────────────────────────────

class TestThePrimaryEstimandIsBoundOnce:
    """Iteration 11C-R1's finding #3: the primary family was pooled
    B3_minus_B0:tga/filr while the prose said the claims were carried solely
    by seen_photo_unseen_wording.  Those are different quantities with
    different cluster counts and different variances, and a preregistration
    that names both names neither."""

    def test_it_is_declared_and_scoped_to_a_counted_entity_set(self):
        e = _load()["primary_estimand"]
        assert e["declared"] == PRIMARY_ESTIMAND == \
            "pooled_over_target_entities"
        assert e["entities_in_scope"] == 72
        assert e["entities_in_scope_by_source"] == {"inaturalist": 30,
                                                    "mllmu_hier": 42}
        assert sum(e["entities_in_scope_by_source"].values()) == \
            e["entities_in_scope"]
        assert len(e["entity_ids_sha256"]) == 64
        assert e["definition"] and len(e["definition"]) > 40
        assert "entity" in e["unit_of_inference"]

    def test_the_entity_set_is_the_target_role_not_the_entity_total(self):
        """Finding #2's correction, at the level of the estimand: 72 is the
        number of entities the target metrics are DEFINED on, not the 100
        entities in the dataset."""
        f = _load()
        p = _load(POWER_PATH)
        e = f["primary_estimand"]
        ceil = p["feasibility"]["ceilings"]
        assert e["entities_in_scope"] == \
            ceil["entity_clusters_available_for_pooled_target_claims"] == 72
        assert ceil["entities_total"] == 100
        assert e["entities_in_scope"] != ceil["entities_total"]
        assert e["entity_ids_sha256"] == \
            p["entity_role_census"]["by_role"]["target"]["entity_ids_sha256"]
        # and the freeze's own ceiling for the primary claims is that 72
        assert f["claims"]["cluster_ceiling_of_the_primary_claims"] == \
            e["entities_in_scope"]

    def test_the_freeze_names_one_estimand_in_every_place_it_appears(self):
        f = _load()
        assert f["claims"]["primary_estimand"] == \
            f["primary_estimand"]["declared"]
        assert _load(POWER_PATH)["primary_estimand"]["declared"] == \
            f["primary_estimand"]["declared"]
        assert f["primary_estimand"]["strata_whose_probes_enter_it"] == \
            list(PRIMARY_ESTIMAND_STRATA)
        assert "a disagreement refuses" in f["primary_estimand"]["read_from"]

    def test_the_stratum_decomposition_is_secondary_with_no_holm_entry(self):
        f = _load()
        e = f["primary_estimand"]
        assert e["per_stratum_decomposition_status"] == \
            STRATUM_ESTIMAND_STATUS
        assert "no_holm_entry" in STRATUM_ESTIMAND_STATUS
        assert "secondary" in STRATUM_ESTIMAND_STATUS
        # the family really is only the two pooled claims: no stratum name
        # appears in it, and k counts nothing else
        fam = f["claims"]["primary_family"]
        assert f["claims"]["k"] == len(fam) == 2
        assert not any("seen_photo" in n or "held_out" in n for n in fam)

    def test_the_fixed_cohort_limit_is_recorded_not_left_to_the_reader(self):
        """Reusing the same entities with new probes cannot enlarge the
        entity set, so what the confirmation does NOT establish belongs in
        the preregistration.  It applies to the pooled estimand and to every
        per-stratum estimand alike, so choosing between them does not escape
        it."""
        e = _load()["primary_estimand"]
        limit = e["what_it_does_not_establish"]
        assert limit == WHAT_THE_CONFIRMATION_DOES_NOT_ESTABLISH
        assert "population-level replication" in limit
        assert "does not create new entity clusters" in limit
        assert e["what_it_establishes"] and len(e["what_it_establishes"]) > 40

    def test_a_contradictory_estimand_refuses(self, monkeypatch):
        monkeypatch.setattr(fz, "PRIMARY_ESTIMAND",
                            "seen_photo_unseen_wording_only")
        f = build_freeze(REPO_ROOT, "pilot100")
        assert any("primary estimand" in r for r in f["refusals"]), \
            f["refusals"]

    def test_a_contradictory_stratum_set_refuses(self, monkeypatch):
        monkeypatch.setattr(fz, "PRIMARY_ESTIMAND_STRATA", ("held_out_photo",))
        f = build_freeze(REPO_ROOT, "pilot100")
        assert any("primary estimand draws on" in r for r in f["refusals"]), \
            f["refusals"]


# ── the size is one selected row, not a grid ──────────────────────

class TestTheSelectedSizeIsBoundNotAGrid:
    """Iteration 11C-R1's finding #1: a table of 3/6/9/12/24 candidates with
    nothing selected is not a design.  One row is chosen here, the command
    line that produces it is named, and every other row stays visible as a
    rejected candidate."""

    def test_one_row_is_selected_and_the_others_stay_rejected(self):
        cs = _load()["confirmation_size"]
        ph, w = cs["held_out_photographs"], cs["new_wording_probes"]
        assert cs["selected"] is True
        assert ph["new_photographs_per_species"] == \
            CONFIRM_NEW_PHOTOS_PER_SPECIES == 12
        assert ph["species_covered"] == 36
        assert ph["new_photographs_total"] == 432 == \
            ph["new_photographs_per_species"] * ph["species_covered"]
        assert w["new_probes_per_target_person"] == \
            CONFIRM_NEW_WORDING_PROBES_PER_PERSON == 12
        assert w["target_persons"] == 42
        assert w["new_wording_probes_total"] == 504 == \
            w["new_probes_per_target_person"] * w["target_persons"]
        # the selected row is not also listed as a rejected candidate
        assert "12" not in ph["rejected_candidates"]
        assert "12" not in w["rejected_candidates"]
        assert {"3", "6", "9", "24"} <= set(ph["rejected_candidates"])
        assert "a different row of the grid" in cs[
            "why_the_size_is_in_the_freeze"]
        assert "rejected candidate" in _load(POWER_PATH)[
            "confirmation_size"]["what_this_block_is"]

    def test_the_fetch_is_one_named_command_line_whose_flags_are_the_size(
            self):
        """Parsing the command is the check.  A command string whose flags do
        not equal the bound fields is a second copy of the size, and the
        second copy is what a builder would follow."""
        ph = _load()["confirmation_size"]["held_out_photographs"]
        assert ph["fetch_images_per_species"] == \
            CONFIRM_FETCH_IMAGES_PER_SPECIES == 24
        assert ph["fetch_seed"] == CONFIRM_FETCH_SEED == 42
        assert ph["fetch_out"] == CONFIRM_FETCH_OUT
        argv = shlex.split(ph["fetch_command"])
        assert argv[:2] == ["python", "scripts/fetch_inat_species.py"]
        flags = dict(zip(argv[2::2], argv[3::2]))
        assert set(flags) == {"--seed", "--images-per-species", "--out"}
        assert flags["--images-per-species"] == \
            str(ph["fetch_images_per_species"])
        assert flags["--seed"] == str(ph["fetch_seed"])
        assert flags["--out"] == ph["fetch_out"]
        assert (REPO_ROOT / "scripts" / "fetch_inat_species.py").exists()

    def test_the_fetch_is_a_superset_of_what_is_already_frozen(self):
        """24 fetched per species of which 12 are new: the seeded superset
        reproduces the 12 exploratory photographs so they can be EXCLUDED by
        hash rather than overwritten, which is what keeps the frozen evidence
        intact (Iteration 11C-3a)."""
        ph = _load()["confirmation_size"]["held_out_photographs"]
        assert ph["already_allocated_per_species"] == 12
        assert ph["fetch_images_per_species"] == \
            ph["new_photographs_per_species"] + \
            ph["already_allocated_per_species"]

    def test_the_totals_are_the_sum_of_the_two_strata(self):
        f = _load()
        cs = f["confirmation_size"]
        t = cs["totals"]
        assert t["of_which_new_photographs"] == \
            cs["held_out_photographs"]["new_photographs_total"]
        assert t["of_which_new_wordings"] == \
            cs["new_wording_probes"]["new_wording_probes_total"]
        assert t["new_target_probes"] == 936 == \
            t["of_which_new_photographs"] + t["of_which_new_wordings"]
        assert t["entity_clusters"] == \
            f["primary_estimand"]["entities_in_scope"]
        assert t["scored_states"] == len(CONFIRMATION_STATES)

    def test_what_is_frozen_now_is_identifiers_not_intentions(self):
        fn = _load()["confirmation_size"]["frozen_now"]
        assert len(fn["target_entity_ids"]) == 72
        assert len(fn["retain_entity_ids"]) == 70
        assert len(set(fn["target_entity_ids"])) == 72
        assert len(set(fn["retain_entity_ids"])) == 70
        assert len({*fn["target_entity_ids"],
                    *fn["retain_entity_ids"]}) == 100
        assert len(fn["exploratory_template_ids"]) == 57
        for key in ("target_entity_ids", "retain_entity_ids"):
            assert len(fn[key + "_sha256"]) == 64, key
        assert len(fn["exploratory_template_ids_sha256"]) == 64
        assert (REPO_ROOT
                / fn["exploratory_photograph_sha256_manifest"]).exists()

    def test_the_identifiers_that_do_not_exist_yet_are_a_named_obligation(
            self):
        """Query ids, template ids and photograph hashes cannot be frozen
        before stage 3 builds them.  Binding the OBLIGATION is what keeps the
        size a preregistration instead of a description of whatever stage 3
        happens to produce."""
        s3 = _load()["confirmation_size"][
            "frozen_at_stage_3_before_any_scoring"]
        for key in ("confirmation_query_id_list_sha256",
                    "confirmation_template_ids_and_file_sha256",
                    "confirmation_photograph_sha256_manifest"):
            assert s3[key].startswith("to be committed"), (key, s3[key])
        assert "do not exist yet" in s3["why_these_cannot_be_frozen_here"]
        rules = " ".join(s3["collision_rules"])
        for needle in ("template_id", "photograph sha256", "query_id",
                       "reference-state gate"):
            assert needle in rules, needle

    def test_the_size_is_also_a_sealed_split_invariant(self):
        inv = _load()["sealed_split_invariants"]
        hit = [i for i in inv if "size frozen here" in i["invariant"]]
        assert len(hit) == 1, [i["invariant"] for i in inv]
        assert hit[0]["checked_at"].startswith("stage 4")
        assert "confirmation_size" in hit[0]["enforced_by"]
        assert str(CONFIRM_NEW_WORDING_PROBES_PER_PERSON) in \
            hit[0]["invariant"]
        assert str(CONFIRM_NEW_PHOTOS_PER_SPECIES) in hit[0]["invariant"]
        assert "fetched by the named command line and no other" in \
            hit[0]["invariant"]

    def test_a_different_row_of_the_photograph_grid_refuses(self,
                                                            monkeypatch):
        monkeypatch.setattr(fz, "CONFIRM_NEW_PHOTOS_PER_SPECIES", 6)
        f = build_freeze(REPO_ROOT, "pilot100")
        assert any("selected confirmation size" in r for r in f["refusals"]), \
            f["refusals"]
        assert any("new_photographs_per_species" in r for r in f["refusals"]), \
            f["refusals"]

    def test_a_different_number_of_wording_probes_refuses(self, monkeypatch):
        monkeypatch.setattr(fz, "CONFIRM_NEW_WORDING_PROBES_PER_PERSON", 3)
        f = build_freeze(REPO_ROOT, "pilot100")
        assert any("new_probes_per_target_person" in r for r in f["refusals"]), \
            f["refusals"]

    def test_a_different_fetch_command_refuses(self, monkeypatch):
        """Same size, different command line: a superset fetch that pulled 12
        instead of 24 could not exclude the exploratory photographs by hash,
        so the command is bound and not only the counts."""
        monkeypatch.setattr(fz, "CONFIRM_FETCH_IMAGES_PER_SPECIES", 12)
        f = build_freeze(REPO_ROOT, "pilot100")
        assert any("fetch_images_per_species" in r for r in f["refusals"]), \
            f["refusals"]


# ── the code that turns scores into claims is bound too ────────────

class TestTheAnalysisCodeIsBound:
    def test_the_fingerprinted_modules_are_the_sidecar_set(self):
        from granunlearn.evaluation.prediction_provenance import (
            CODE_FINGERPRINT_MODULES)
        f = _load()
        assert f["code"]["fingerprinted_module_list"] == \
            list(CODE_FINGERPRINT_MODULES)
        assert set(f["code"]["fingerprinted_modules"]) == \
            set(CODE_FINGERPRINT_MODULES)

    def test_the_analysis_scripts_are_bound_separately_and_say_why(self):
        f = _load()
        bound = f["code"]["analysis_scripts_sha256"]
        assert set(bound) == set(ANALYSIS_SCRIPTS)
        assert len(set(bound.values())) == len(bound)
        assert "without changing a single prediction" in \
            f["code"]["why_analysis_scripts_are_bound_separately"]

    def test_the_scorer_modules_are_identified(self):
        f = _load()
        mods = f["scorer"]["modules_sha256"]
        assert set(mods) == {"src/granunlearn/evaluation/scoring.py",
                             "src/granunlearn/evaluation/hierarchy_metrics.py"}

    def test_binding_an_extra_script_is_detected_as_drift(self, monkeypatch):
        """--check-only compares the committed freeze against a fresh
        derivation.  If the set of bound analysis scripts changes, the
        comparison must notice — otherwise the freeze certifies code it never
        hashed."""
        committed = _load()
        monkeypatch.setattr(
            fz, "ANALYSIS_SCRIPTS",
            tuple(ANALYSIS_SCRIPTS) + ("scripts/write_provenance.py",))
        fresh = build_freeze(REPO_ROOT, "pilot100")
        drift = [k for k in set(committed) | set(fresh)
                 if k not in VOLATILE_FREEZE_FIELDS
                 and json.dumps(committed.get(k), sort_keys=True)
                 != json.dumps(fresh.get(k), sort_keys=True)]
        assert "code" in drift, drift


# ── the sealed-split invariants name an enforcement point ──────────

class TestTheSealedSplitInvariantsAreActionable:
    def test_every_invariant_names_what_enforces_it_and_when(self):
        """An invariant with no enforcement point is a comment.  Each one
        must say which artifact or stage checks it, so stage 5 can be held to
        it."""
        f = _load()
        inv = f["sealed_split_invariants"]
        assert len(inv) == 6
        for item in inv:
            assert set(item) == {"invariant", "why", "enforced_by",
                                 "checked_at"}, item
            for key in ("invariant", "why", "enforced_by"):
                assert item[key] and len(item[key]) > 15, (key, item[key])
            # the enforcement point must be a named stage of the
            # confirmation sequence, not "later" or "somewhere"
            assert item["checked_at"].startswith("stage"), item["checked_at"]

    def test_the_three_ways_the_split_could_leak_are_all_covered(self):
        """The user's constraint was that the confirmation split must never
        enter the reference gate, candidate selection, or another go/no-go
        decision.  All three appear, plus the disjointness they depend on."""
        f = _load()
        text = " ".join(i["invariant"] for i in f["sealed_split_invariants"])
        assert "reference-state gate" in text
        assert "candidate selection" in text
        assert "go/no-go" in text
        assert "query_id" in text and "photograph" in text

    def test_the_score_once_properties_are_recorded(self):
        f = _load()
        text = " ".join(i["invariant"] for i in f["sealed_split_invariants"])
        assert "partial state results" in text
        assert "reported as failed" in text

    def test_the_exploratory_dataset_is_bound_and_labelled_as_such(self):
        """Bound to prove what the confirmation is NOT: selection happened on
        this dataset's train+val, and the gate saw its test split, which is
        exactly why those results stay exploratory."""
        f = _load()
        e = f["exploratory_dataset"]
        assert e["version"] == "pilot100_v2"
        assert e["selection_scope"] == ["train", "val"]
        assert "different dataset" in e["role"]
        assert e["image_manifest_sha256"]
        assert set(e["artifacts_sha256"]) >= {"associations.parquet",
                                              "queries.parquet",
                                              "manifest.json"}

    def test_the_inputs_the_freeze_rests_on_are_hashed(self):
        f = _load()
        bound = f["inputs_bound"]
        assert set(bound) == {"power_analysis", "selection",
                              "final_evaluation", "prediction_manifest"}
        for name, entry in bound.items():
            assert entry["sha256"] and len(entry["sha256"]) == 64, name
            assert (REPO_ROOT / entry["path"]).exists(), name
            # the hash must be the file's, not a copy of something else
            import hashlib
            assert entry["sha256"] == hashlib.sha256(
                (REPO_ROOT / entry["path"]).read_bytes()).hexdigest(), name


# ── the freeze cannot be silently rewritten ────────────────────────

class TestTheFreezeIsWrittenOnce:
    def test_the_committed_freeze_is_self_consistent(self):
        f = _load()
        assert f["refusals"] == []
        assert f["freeze"] == "iteration_11c_confirmation_protocol"
        assert f["iteration"] == "11C"
        assert f["provenance_contract_version"] == 2
        assert f["git_commit"] and len(f["git_commit"]) == 40

    def test_an_unmutated_derivation_refuses_only_over_absent_artifacts(self):
        """Baseline for every refusal test in this file, and the one that
        keeps them honest: without it a ``build_freeze`` that refused
        unconditionally would make each of them pass while asserting
        nothing.

        In an artifact-bearing checkout an unmutated derivation must produce
        no refusals at all.  In a fresh clone the adapters are gitignored and
        it must refuse — but ONLY over that, so the refusals are enumerated
        rather than merely expected to be non-empty.
        """
        f = build_freeze(REPO_ROOT, "pilot100")
        if _adapters_present():
            assert f["refusals"] == []
        else:
            states = set(CONFIRMATION_STATES) | {"MF"}
            assert f["refusals"], "absent adapters did not refuse"
            for r in f["refusals"]:
                assert r.split(":")[0] in states or \
                    "no-op check could not be performed" in r, r

    def test_a_fresh_derivation_matches_the_committed_freeze(self):
        """What --check-only does.  Volatile fields are excluded; everything
        else must be identical or the freeze no longer describes the
        repository.  Needs the adapters, because a derivation that could not
        resolve them differs from the committed freeze in exactly the block
        this compares."""
        _require_adapters()
        committed = _load()
        fresh = build_freeze(REPO_ROOT, "pilot100")
        for k in sorted(set(committed) | set(fresh)):
            if k in VOLATILE_FREEZE_FIELDS:
                continue
            assert json.dumps(committed.get(k), sort_keys=True) == \
                json.dumps(fresh.get(k), sort_keys=True), k

    def test_the_drift_check_excludes_only_what_it_documents(self):
        """Every excluded field is a hole in the drift check, so the set must
        stay exactly as small as its stated reasons require.  Widening it
        silently would let a real parameter move undetected."""
        assert VOLATILE_FREEZE_FIELDS == {
            "frozen_at_utc", "git_commit", "git_dirty", "environment"}
        committed = _load()
        # nothing binding may be in the excluded set
        for bound in ("checkpoints", "generation", "scorer", "analysis",
                      "claims", "code", "exploratory_dataset",
                      "sealed_split_invariants", "inputs_bound", "states",
                      "b0_is_the_no_op", "primary_estimand",
                      "confirmation_size"):
            assert bound in committed, bound
            assert bound not in VOLATILE_FREEZE_FIELDS, bound

    def test_the_environment_is_excluded_because_it_is_process_dependent(
            self):
        """Under the CPU-only CI emulation torch, transformers and peft are
        unimportable, so environment_fingerprint() reports null versions and
        cannot match a freeze written on a GPU machine.  The block is
        diagnostic for exactly that reason — which is also why it must not be
        allowed to fail the drift check."""
        from granunlearn.evaluation.prediction_provenance import (
            environment_fingerprint)
        assert "environment" in VOLATILE_FREEZE_FIELDS
        live = environment_fingerprint()
        assert set(live) == {"python_executable", "python_version",
                             "package_versions"}
        assert _load()["environment"]["python_version"] == \
            live["python_version"]

    def test_a_moved_margin_is_detected_as_drift(self, monkeypatch):
        """The clearest failure the freeze exists to prevent: delta changing
        between the freeze and the scoring run."""
        committed = _load()
        monkeypatch.setattr(fz, "EQUIVALENCE_MARGIN_TGA", 0.10)
        fresh = build_freeze(REPO_ROOT, "pilot100")
        assert fresh["scorer"]["equivalence_margin"]["value"] == 0.10
        assert json.dumps(committed["scorer"], sort_keys=True) != \
            json.dumps(fresh["scorer"], sort_keys=True)

    def test_a_moved_bootstrap_seed_is_detected_as_drift(self, monkeypatch):
        from granunlearn.evaluation import paired_ci
        sig = inspect.signature(paired_ci.paired_rate_diff_ci)
        replaced = sig.replace(
            parameters=[p.replace(default=7) if p.name == "seed" else p
                        for p in sig.parameters.values()])
        monkeypatch.setattr(paired_ci.paired_rate_diff_ci, "__signature__",
                            replaced, raising=False)
        fresh = build_freeze(REPO_ROOT, "pilot100")
        assert fresh["analysis"]["bootstrap"]["seed"] == 7
        assert fresh["analysis"]["bootstrap"]["seed"] != \
            _load()["analysis"]["bootstrap"]["seed"]

    def test_the_environment_is_recorded_but_not_binding(self):
        """Same stance as the sidecars: an interpreter path is diagnostic, and
        making it binding would invalidate the freeze over a patch upgrade
        that cannot move a decoded token.  The binding fingerprint therefore
        covers source modules only."""
        f = _load()
        e = f["environment"]
        assert e["python_version"]
        assert all(rel.startswith("src/granunlearn/")
                   for rel in f["code"]["fingerprinted_module_list"])
        assert not any("environment" in rel
                       for rel in f["code"]["fingerprinted_modules"])
