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

import hashlib
import inspect
import json
import re
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
    drift_keys,
)
from power_analysis_confirmation import (  # noqa: E402
    ALPHA_ONE_SIDED,
    ALPHA_STANDALONE_ONE_SIDED,
    BOOTSTRAP_SEED,
    CONFIRM_FETCH_IMAGES_PER_SPECIES,
    CONFIRM_FETCH_OUT,
    CONFIRM_FETCH_ROLE,
    CONFIRM_FETCH_SEED,
    CONFIRM_FETCH_TAG,
    CONFIRM_NEW_PHOTOS_PER_SPECIES,
    CONFIRM_NEW_WORDING_PROBES_PER_PERSON,
    CONFIRM_RETENTION_NEW_TEMPLATES_PER_ENTITY,
    CONFIRM_RETENTION_ROUTE,
    EXPLORATORY_IMAGE_MANIFEST,
    FAMILYWISE_ALPHA,
    HOLM_WORST_CASE_ALPHA,
    N_PERMUTATIONS,
    PERMUTATION_SEED,
    PHOTO_SELECTION_RULE,
    PRIMARY_ESTIMAND,
    PRIMARY_ESTIMAND_STRATA,
    PRIMARY_FAMILY,
    RETENTION_ESTIMAND_LABEL,
    RETENTION_SAMPLING_RULE,
    RETENTION_SAMPLING_SALTS,
    STRATUM_ESTIMAND_STATUS,
    TARGET_ASSOCIATION_ALLOCATION_RULE,
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


def _freeze_over_mutated_power(monkeypatch, mutate) -> dict:
    """``build_freeze`` against a power report transformed by ``mutate``.

    Every refusal test in this module needs the same wiring, and a copy of it
    per test is a copy per test that can stop matching the real loader.
    """
    real = fz._load_report

    def wrapped(reports, name):
        out = real(reports, name)
        if not name.endswith("confirmation_power.json"):
            return out
        return mutate(out)

    monkeypatch.setattr(fz, "_load_report", wrapped)
    return build_freeze(REPO_ROOT, "pilot100")


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
        second independent number: familywise/2 is one arm of a two-sided
        interval, Holm's worst case for one of k claims is familywise/k, and
        the two coincide here only because k = 2.  Iteration 11C-R2's finding
        #3 is that neither is what ONE-SIDEDNESS costs - a standalone
        directional claim is tested at familywise itself."""
        f = _load()
        a, c = f["analysis"], f["claims"]
        assert a["familywise_alpha"] == FAMILYWISE_ALPHA == 0.05
        assert c["familywise_alpha"] == FAMILYWISE_ALPHA
        assert a["alpha_one_sided"] == ALPHA_ONE_SIDED == FAMILYWISE_ALPHA / 2
        assert a["alpha_standalone_one_sided"] == \
            ALPHA_STANDALONE_ONE_SIDED == FAMILYWISE_ALPHA
        assert a["holm_worst_case_alpha"] == HOLM_WORST_CASE_ALPHA == \
            a["alpha_one_sided"]
        assert a["alpha_standalone_one_sided"] != a["alpha_one_sided"], \
            "the standalone level must stay a separate number, or the k = 2 " \
            "coincidence reads as a rule"
        assert round(FAMILYWISE_ALPHA / 3, 4) != ALPHA_ONE_SIDED, \
            "at k = 3 Holm's first threshold moves while familywise/2 does " \
            "not, which is the only way to see that they are different " \
            "quantities"
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
        # Iteration 11C-R2's finding #2: the photograph budget covers the
        # TARGET species.  36 was the count the seeded re-fetch could cover,
        # which includes 6 retain-only species whose only purpose was the
        # retain image families - and those are not renewed.
        assert ph["species_covered"] == 30
        assert ph["new_photographs_total"] == 360 == \
            ph["new_photographs_per_species"] * ph["species_covered"]
        assert "TARGET species" in ph["species_covered_is"]
        assert "36" in ph["species_covered_is"]
        assert "no confirmation purpose" in ph["why_only_the_target_species"]
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
        assert ph["fetch_role"] == CONFIRM_FETCH_ROLE == "target"
        assert ph["fetch_tag"] == CONFIRM_FETCH_TAG == "pilot100"
        argv = shlex.split(ph["fetch_command"])
        assert argv[:2] == ["python", "scripts/fetch_inat_species.py"]
        flags = dict(zip(argv[2::2], argv[3::2]))
        assert set(flags) == {"--seed", "--images-per-species", "--role",
                              "--tag", "--out"}
        assert flags["--images-per-species"] == \
            str(ph["fetch_images_per_species"])
        assert flags["--seed"] == str(ph["fetch_seed"])
        assert flags["--role"] == ph["fetch_role"]
        assert flags["--tag"] == ph["fetch_tag"]
        assert flags["--out"] == ph["fetch_out"]
        assert (REPO_ROOT / "scripts" / "fetch_inat_species.py").exists()

    def test_the_fetch_names_a_role_because_a_count_would_miss_species(self):
        """``--limit-species 30`` reads like the same instruction and is not:
        the 6 retain-only species are interleaved through SPECIES_LIST, so the
        first 30 entries hold only 24 target species.  Iteration 11C-R2."""
        ph = _load()["confirmation_size"]["held_out_photographs"]
        assert "--limit-species" not in ph["fetch_command"]
        why = ph["why_the_command_names_a_role_and_not_a_count"]
        assert "24 of the 30" in why
        assert "interleaved" in why
        # and the fetcher really does derive the role rather than slice
        sys.path.insert(0, str(REPO_ROOT / "scripts"))
        import fetch_inat_species as fetch
        target = fetch.species_for_role("target", "pilot100")
        assert len(target) == ph["species_covered"] == 30
        assert len(fetch.SPECIES_LIST[:30] ) != 30 or \
            set(fetch.SPECIES_LIST[:30]) != set(target)
        assert set(target) & set(fetch.SPECIES_LIST[:30]) != set(target), \
            "if the first 30 entries were the target species the role flag " \
            "would be decoration"
        retain = fetch.species_for_role("retain", "pilot100")
        assert len(retain) == 6
        assert not set(retain) & set(target)

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

    def test_the_totals_separate_the_target_budget_from_the_whole(self):
        """Iteration 11C-R2's finding #2.  936 was published as
        ``new_target_probes``; it was 504 target-person wordings plus 432
        photographs over 36 species, i.e. the whole allocated photograph
        budget including the 72 destined for retain-only species.  The target
        total is 864, and once retention is allocated the overall total is a
        third number again."""
        f = _load()
        cs = f["confirmation_size"]
        t = cs["totals"]
        pb = cs["probe_budget"]
        n_words = cs["new_wording_probes"]["new_wording_probes_total"]
        n_photos = cs["held_out_photographs"]["new_photographs_total"]
        n_ret = f["retention_probes"]["total"]
        assert t["of_which_new_wordings_on_target_persons"] == n_words == 504
        assert t["of_which_new_photographs_on_target_species"] == \
            n_photos == 360
        assert t["new_target_probes"] == 864 == n_words + n_photos
        assert t["new_retention_probes"] == n_ret == 345
        assert t["total_new_probes_allocated"] == 1209 == \
            n_words + n_photos + n_ret
        # the three numbers are three DIFFERENT numbers
        assert len({t["new_target_probes"], t["new_retention_probes"],
                    t["total_new_probes_allocated"]}) == 3
        assert t["total_generations_at_three_scored_states"] == \
            3 * t["total_new_probes_allocated"]
        assert t["scored_states"] == len(CONFIRMATION_STATES)
        assert t["entity_clusters_for_the_primary_claims"] == \
            f["primary_estimand"]["entities_in_scope"] == 72
        assert t["entity_clusters_for_retention"] == \
            f["retention_probes"]["entities_covered"] == 70
        # the freeze's own budget block agrees with the report's totals
        assert pb["primary_target_total"] == t["new_target_probes"]
        assert pb["new_retention_probes"] == t["new_retention_probes"]
        assert pb["total_allocated"] == t["total_new_probes_allocated"]
        assert pb["new_wordings_on_target_persons"] == n_words
        assert pb["new_photographs_on_target_species"] == n_photos
        # and the mislabelling is corrected in the artifact, not only here
        corr = t["correction_to_the_previous_revision"]
        assert "936" in corr and "864" in corr and "72" in corr
        assert pb["correction_to_the_previous_revision"] == corr
        assert "overstates the primary design" in pb[
            "why_the_two_totals_are_reported_separately"]

    def test_no_total_in_the_freeze_still_calls_936_the_target_count(self):
        """The mislabelled number is the defect, so the artifact is searched
        for it rather than only the fields a reader would think to check."""
        f = _load()
        blob = json.dumps(f)
        assert '"new_target_probes": 936' not in blob
        assert f["confirmation_size"]["totals"]["new_target_probes"] != 936
        hits = []

        def _walk(node, path):
            if isinstance(node, dict):
                for kk, vv in node.items():
                    if vv == 936:
                        hits.append(f"{path}.{kk}")
                    _walk(vv, f"{path}.{kk}")
            elif isinstance(node, list):
                for ii, vv in enumerate(node):
                    _walk(vv, f"{path}[{ii}]")

        _walk(f, "freeze")
        assert hits == [], f"936 still bound as a value at {hits}"

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

    def test_a_command_that_limits_by_count_refuses(self, monkeypatch):
        """``--limit-species`` in place of ``--role`` fetches 24 of the 30
        target species. The command is what a human runs at stage 3, so the
        flag is checked and not only the count it implies."""
        def mutate(out):
            out = dict(out)
            cs = dict(out["confirmation_size"])
            ph = dict(cs["held_out_photographs"])
            ph["fetch_command"] = ph["fetch_command"].replace(
                "--role target --tag pilot100", "--limit-species 30")
            cs["held_out_photographs"] = ph
            out["confirmation_size"] = cs
            return out

        f = _freeze_over_mutated_power(monkeypatch, mutate)
        assert any("does not select species by role" in r
                   for r in f["refusals"]), f["refusals"]
        assert any("limits species by COUNT" in r
                   for r in f["refusals"]), f["refusals"]

    def test_a_species_count_that_is_not_the_target_role_refuses(
            self, monkeypatch):
        """36 is how many species the seeded re-fetch could cover; 30 is how
        many carry a target association. Binding the wrong one budgets
        photographs for entities no target metric is defined on."""
        def mutate(out):
            out = dict(out)
            cs = dict(out["confirmation_size"])
            ph = dict(cs["held_out_photographs"])
            ph["species_covered"] = 36
            ph["new_photographs_total"] = 12 * 36
            cs["held_out_photographs"] = ph
            cs["totals"] = dict(cs["totals"],
                                new_target_probes=504 + 432,
                                of_which_new_photographs_on_target_species=432,
                                total_new_probes_allocated=504 + 432 + 345)
            out["confirmation_size"] = cs
            return out

        f = _freeze_over_mutated_power(monkeypatch, mutate)
        assert any("TARGET iNaturalist" in r for r in f["refusals"]), \
            f["refusals"]


# ── which 12 of the 24 photographs, decided by a rule and not by hand ─
# The rule is an outcome-blind amendment: it was published after the pool was
# fetched and before the subset was selected.  See
# TestTheAmendmentIsDisclosedInTheFreeze below for the timeline.

class TestThePhotographSelectionRuleIsFrozenAndExecutable:
    """Iteration 11C stage 3.

    The fetch draws 24 photographs per species and the design keeps 12.  An
    earlier revision specified WHICH 12 positionally, by claiming the seeded
    shuffle nests so that the exploratory photographs come first and the rest
    are new by construction.  Measured against ``fetch_inat_species.py`` that
    is false twice over: ``rng`` is built once for the whole species walk and
    ``--role target`` walks 30 species where ``pilot_v1`` walked 36, so the
    draw differs for every species after the first dropped one; and
    ``chosen.sort`` re-orders the accepted set before files are named, so
    position on disk is canonical order and not draw order.

    The frozen rule is therefore a hash rule, and "which 12" is sealed here
    rather than left to whoever runs the selection over the pool that came
    back.  Each refusal below was observed to fire.
    """

    def _photos(self, f=None):
        return (f or _load())["confirmation_size"]["held_out_photographs"]

    def test_the_frozen_rule_is_the_module_rule_and_names_a_hash(self):
        ph = self._photos()
        assert ph["selection_rule"] == PHOTO_SELECTION_RULE
        assert "sha256" in ph["selection_rule"]
        assert EXPLORATORY_IMAGE_MANIFEST in ph["selection_rule"]
        assert "(observation_id, photo_id)" in ph["selection_rule"]
        assert "refuse the species" in ph["selection_rule"]
        assert ph["refuse_a_species_with_fewer_than_n_disjoint"] == \
            CONFIRM_NEW_PHOTOS_PER_SPECIES == 12
        # both copies of the rule are the same string, so a builder reading
        # either one follows the frozen rule
        supply = _load(POWER_PATH)["new_photograph_supply"]["seeded_refetch"]
        assert supply["selection_rule"] == ph["selection_rule"]

    def test_the_manifest_the_rule_is_disjoint_from_is_hashed(self):
        """Naming the file is not binding it: the rule is disjointness against
        its CONTENTS, so a manifest that changed would silently redefine what
        counts as a new photograph."""
        fn = _load()["confirmation_size"]["frozen_now"]
        assert fn["exploratory_photograph_sha256_manifest"] == \
            EXPLORATORY_IMAGE_MANIFEST
        recorded = fn["exploratory_photograph_sha256_manifest_sha256"]
        path = REPO_ROOT / EXPLORATORY_IMAGE_MANIFEST
        if not path.exists():
            pytest.skip(f"committed manifest not present: {path}")
        assert recorded == fz.sha256_file(path)
        assert len(recorded) == 64
        bytes.fromhex(recorded)

    def test_a_positional_selection_rule_refuses(self, monkeypatch):
        """The mistake the correction exists to prevent, restated as a rule."""
        def mutate(out):
            out = dict(out)
            cs = dict(out["confirmation_size"])
            ph = dict(cs["held_out_photographs"])
            ph["selection_rule"] = (
                "take the last 12 of the seeded draw, which are new by "
                "construction")
            cs["held_out_photographs"] = ph
            out["confirmation_size"] = cs
            return out

        f = _freeze_over_mutated_power(monkeypatch, mutate)
        assert any("POSITIONALLY" in r for r in f["refusals"]), f["refusals"]
        assert any("not the rule the module declares" in r
                   for r in f["refusals"]), f["refusals"]

    def test_a_rule_that_names_no_content_hash_refuses(self, monkeypatch):
        def mutate(out):
            out = dict(out)
            cs = dict(out["confirmation_size"])
            ph = dict(cs["held_out_photographs"])
            ph["selection_rule"] = (
                "take the 12 highest-resolution photographs per species")
            cs["held_out_photographs"] = ph
            out["confirmation_size"] = cs
            return out

        f = _freeze_over_mutated_power(monkeypatch, mutate)
        assert any("does not name a content hash" in r
                   for r in f["refusals"]), f["refusals"]

    def test_two_differing_copies_of_the_rule_refuse(self, monkeypatch):
        """The rule is published in the size block and in the supply block.  A
        builder follows one and the freeze certifies the other unless a
        disagreement is refused."""
        def mutate(out):
            out = dict(out)
            supply = dict(out["new_photograph_supply"])
            sr = dict(supply["seeded_refetch"])
            sr["selection_rule"] = PHOTO_SELECTION_RULE + " (or any 12)"
            supply["seeded_refetch"] = sr
            out["new_photograph_supply"] = supply
            return out

        f = _freeze_over_mutated_power(monkeypatch, mutate)
        assert any("stated twice and the two copies differ" in r
                   for r in f["refusals"]), f["refusals"]

    def test_accepting_a_short_species_refuses(self, monkeypatch):
        """A species yielding fewer than 12 disjoint photographs must refuse
        rather than be padded from the exploratory pool, which is the one
        thing the novelty invariant forbids."""
        def mutate(out):
            out = dict(out)
            cs = dict(out["confirmation_size"])
            ph = dict(cs["held_out_photographs"])
            ph["refuse_a_species_with_fewer_than_n_disjoint"] = 6
            cs["held_out_photographs"] = ph
            out["confirmation_size"] = cs
            return out

        f = _freeze_over_mutated_power(monkeypatch, mutate)
        assert any("disjoint photographs where the design needs" in r
                   for r in f["refusals"]), f["refusals"]

    def test_the_withdrawn_claim_must_stay_recorded(self, monkeypatch):
        """Deleting the refutation is how the claim comes back: the seed looks
        like it ought to nest, and nothing in a freeze that only states the
        rule says why the obvious argument fails."""
        def mutate(out):
            out = dict(out)
            supply = dict(out["new_photograph_supply"])
            sr = dict(supply["seeded_refetch"])
            sr["draw_nesting_is_not_relied_on"] = {
                "claim_an_earlier_revision_made":
                    sr["draw_nesting_is_not_relied_on"][
                        "claim_an_earlier_revision_made"]}
            supply["seeded_refetch"] = sr
            out["new_photograph_supply"] = supply
            return out

        f = _freeze_over_mutated_power(monkeypatch, mutate)
        missing = [r for r in f["refusals"]
                   if "not recorded with its refutation" in r]
        assert len(missing) == 3, missing

    def test_a_drifted_exploratory_manifest_refuses(self, monkeypatch):
        def mutate(out):
            out = dict(out)
            cs = dict(out["confirmation_size"])
            fn = dict(cs["frozen_now"])
            fn["exploratory_photograph_sha256_manifest_sha256"] = "0" * 64
            cs["frozen_now"] = fn
            out["confirmation_size"] = cs
            return out

        f = _freeze_over_mutated_power(monkeypatch, mutate)
        assert any("has DRIFTED" in r for r in f["refusals"]), f["refusals"]

    def test_an_unhashable_manifest_record_fails_closed(self, monkeypatch):
        """A report generated where the manifest is absent records a reason
        string instead of a hash.  Freezing that would certify a novelty rule
        with nothing to be disjoint from -- the same fail-open shape as the
        empty adapter contract 11C-R1 found."""
        def mutate(out):
            out = dict(out)
            cs = dict(out["confirmation_size"])
            fn = dict(cs["frozen_now"])
            fn["exploratory_photograph_sha256_manifest_sha256"] = \
                "REFUSED-ABSENT: not there"
            cs["frozen_now"] = fn
            out["confirmation_size"] = cs
            return out

        f = _freeze_over_mutated_power(monkeypatch, mutate)
        assert any("is not a sha256" in r for r in f["refusals"]), \
            f["refusals"]

    def test_a_module_rule_that_disagrees_with_the_report_refuses(self,
                                                                 monkeypatch):
        """The check runs against the module, not against a copy of it, so
        moving the rule in code without regenerating is caught."""
        monkeypatch.setattr(fz, "PHOTO_SELECTION_RULE",
                            PHOTO_SELECTION_RULE + " ")
        f = build_freeze(REPO_ROOT, "pilot100")
        assert any("not the rule the module declares" in r
                   for r in f["refusals"]), f["refusals"]


# ── the portrait exemption is a hash list, not a permission ────────

def _mutate_portraits(out, **portraits):
    """Rewrite fields of ``portrait_reuse_exemption.portraits``."""
    out = dict(out)
    block = dict(out["portrait_reuse_exemption"])
    block["portraits"] = dict(block["portraits"], **portraits)
    out["portrait_reuse_exemption"] = block
    return out


def _mutate_portrait_block(out, **fields):
    """Rewrite top-level fields of ``portrait_reuse_exemption``."""
    out = dict(out)
    out["portrait_reuse_exemption"] = dict(
        out["portrait_reuse_exemption"], **fields)
    return out


class TestThePortraitExemptionIsBoundedByAHashList:
    """Iteration 11C stage 3.

    As sealed at 11C-R2, invariants 2 and 3 required every confirmation
    photograph sha256 to be new and no exploratory media to be reused.  The
    frozen primary estimand pools over 72 entities across two IMAGE strata;
    the wording stratum's 42 clusters are MLLMU persons with exactly one
    portrait each; and a text-route probe carries ``image_split = None``, so
    it is in neither stratum.  The rules as written therefore forbade the
    design they were sealing.

    The exemption that replaces them is granted as a list of 42 hashes rather
    than as a category, so it cannot widen without a re-freeze -- and each
    refusal below was observed to fire.
    """

    def test_the_exemption_is_the_measured_42_and_is_bound_by_hash(self):
        f = _load()
        pe = f["portrait_exemption"]
        p = _load(POWER_PATH)["portrait_reuse_exemption"]
        assert pe["exempted"] == p["portraits"]["count"] == 42
        assert pe["target_persons"] == 42
        assert pe["set_sha256"] == p["portraits"]["set_sha256"]
        # the set hash must be recomputable from the list it claims to bind
        assert hashlib.sha256(
            "\n".join(pe["sha256"]).encode()).hexdigest() == pe["set_sha256"]
        assert len(pe["sha256"]) == len(set(pe["sha256"])) == 42
        assert len(pe["paths"]) == len(pe["image_ids"]) == 42
        # and it must be the same set the size block binds
        fn = f["confirmation_size"]["frozen_now"]
        assert fn["required_repeat_portrait_set_sha256"] == pe["set_sha256"]
        assert fn["required_repeat_portrait_count"] == 42

    def test_every_exempted_hash_is_really_exploratory_media(self):
        """An exemption for media the exploratory run never used would exempt
        nothing and hide a mis-measured list."""
        pe = _load()["portrait_exemption"]
        man = REPO_ROOT / EXPLORATORY_IMAGE_MANIFEST
        if not man.exists():
            pytest.skip(f"committed manifest not present: {man}")
        images = json.loads(man.read_text())["images"]
        exploratory = {v["sha256"] for v in images.values()}
        assert set(pe["sha256"]) <= exploratory
        assert all(p in images for p in pe["paths"])
        assert pe["photographs_per_target_person"] == {"1": 42}

    def test_the_cost_is_disclosed_rather_than_argued_away(self):
        """The gate scored all 42 portraits.  The freeze says so beside the
        exemption, because an exemption that hid its own cost would read as a
        reason the person-side result is not exploratory at all."""
        f = _load()
        pe = f["portrait_exemption"]
        assert pe["the_exploratory_gate_exercised"] == 42
        inv = f["sealed_split_invariants"][1]
        assert "that cost is disclosed rather than argued away" in \
            inv["why_the_exemption_does_not_give_up_the_novelty_argument"]
        assert pe["what_is_still_new_on_an_exempted_probe"] == \
            ["query_id", "template_id", "template text"]

    def test_the_invariants_state_the_exception_instead_of_contradicting_it(
            self):
        inv = _load()["sealed_split_invariants"]
        assert len(inv) == 8
        assert "except the hashed target-person portraits" in inv[1][
            "invariant"]
        assert inv[1]["bound_by"]["exempted_portraits"] == 42
        assert inv[1]["bound_by"]["exempted_portrait_set_sha256"]
        assert "beyond the two required repeats" in inv[2]["invariant"]
        # exactly two things may repeat, and they are named
        may = inv[2]["bound_by"]["what_may_repeat"]
        assert isinstance(may, list) and len(may) == 2
        assert "target-association set" in may[0]
        assert "42 target-person portraits" in may[1]
        assert "360 new species photographs without exception" in \
            inv[2]["bound_by"]["what_may_not"]
        assert "unexecutable" in inv[2]["why"]

    def test_the_exemption_does_not_reach_the_species_or_retention_probes(
            self):
        pe = _load()["portrait_exemption"]
        assert "504" in pe["applies_to"]
        assert "nothing else" in pe["applies_to"]
        assert "360" in pe["does_not_apply_to"]
        assert "hash-disjoint" in pe["does_not_apply_to"]
        assert "345" in pe["does_not_apply_to"]
        assert "text-only" in pe["does_not_apply_to"]

    def test_the_measurement_that_forces_it_is_in_the_freeze(self):
        m = _load()["portrait_exemption"]["measurement_it_rests_on"]
        assert m["wording_stratum_probes"] == 180
        assert m["wording_stratum_is_entirely_image_route"] is True
        assert set(m["wording_stratum_routes"]) == \
            {"image_text_to_text", "image_to_text"}
        assert set(m["wording_stratum_families"]) == \
            {"image_fine_direct", "image_target_direct",
             "multimodal_image_text"}
        assert m["every_target_person_has_exactly_one_photograph"] is True
        assert m["text_route_image_split_values"] == ["None"]
        assert m["text_route_probes_are_in_neither_primary_stratum"] is True
        assert m["text_route_test_probes"] > 0

    def test_the_rejected_alternatives_are_recorded(self):
        rej = _load()["portrait_exemption"]["rejected_alternatives"]
        assert len(rej) == 4
        joined = " ".join(rej)
        assert "text-route" in joined
        assert "species only" in joined
        assert "72 clusters to 30" in joined or "from 72" in joined
        assert "no boundary" in joined

    def test_a_missing_exemption_block_refuses(self, monkeypatch):
        """Without it the novelty rules stand unqualified, and they forbid the
        estimand -- which is the defect this block exists to close."""
        def mutate(out):
            out = dict(out)
            out.pop("portrait_reuse_exemption")
            return out

        f = _freeze_over_mutated_power(monkeypatch, mutate)
        assert any("no portrait_reuse_exemption" in r
                   for r in f["refusals"]), f["refusals"]

    def test_an_exemption_wider_than_the_wording_stratum_refuses(self,
                                                                monkeypatch):
        """Exempting more portraits than there are target persons would exempt
        media no confirmation probe needs."""
        def mutate(out):
            return _mutate_portraits(out, count=43)

        f = _freeze_over_mutated_power(monkeypatch, mutate)
        assert any("sized over" in r for r in f["refusals"]), f["refusals"]

    def test_a_set_hash_that_does_not_describe_its_list_refuses(self,
                                                               monkeypatch):
        def mutate(out):
            return _mutate_portraits(out, set_sha256="a" * 64)

        f = _freeze_over_mutated_power(monkeypatch, mutate)
        assert any("does not describe the bound list" in r
                   for r in f["refusals"]), f["refusals"]

    def test_a_duplicated_portrait_hash_refuses(self, monkeypatch):
        def mutate(out):
            p = out["portrait_reuse_exemption"]["portraits"]
            dup = list(p["sha256"])
            dup[1] = dup[0]
            return _mutate_portraits(out, sha256=dup)

        f = _freeze_over_mutated_power(monkeypatch, mutate)
        assert any("duplicate sha256" in r for r in f["refusals"]), \
            f["refusals"]

    def test_exempting_media_that_is_not_exploratory_refuses(self,
                                                            monkeypatch):
        def mutate(out):
            p = out["portrait_reuse_exemption"]["portraits"]
            hashes = list(p["sha256"])
            hashes[0] = "b" * 64
            return _mutate_portraits(
                out, sha256=hashes,
                set_sha256=hashlib.sha256(
                    "\n".join(hashes).encode()).hexdigest())

        f = _freeze_over_mutated_power(monkeypatch, mutate)
        assert any("are NOT in" in r for r in f["refusals"]), f["refusals"]

    def test_the_two_bindings_disagreeing_refuses(self, monkeypatch):
        def mutate(out):
            out = _mutate_portraits(out, set_sha256="c" * 64)
            cs = dict(out["confirmation_size"])
            cs["frozen_now"] = dict(
                cs["frozen_now"], required_repeat_portrait_set_sha256="d" * 64)
            out["confirmation_size"] = cs
            return out

        f = _freeze_over_mutated_power(monkeypatch, mutate)
        assert any("would exempt different media" in r
                   for r in f["refusals"]), f["refusals"]

    def test_a_measurement_that_stopped_forcing_it_refuses(self, monkeypatch):
        """If a target person ever had two photographs, or the wording stratum
        stopped being image-route, the exemption would no longer be necessary
        and the novelty rule should apply in full again."""
        for field in ("every_target_person_has_exactly_one_photograph",
                      "wording_stratum_is_entirely_image_route",
                      "text_route_probes_are_in_neither_primary_stratum"):
            f = _freeze_over_mutated_power(
                monkeypatch,
                lambda out, field=field: _mutate_portrait_block(
                    out, **{field: False}))
            assert any(field in r and "no longer forced" in r
                       for r in f["refusals"]), (field, f["refusals"])

    def test_a_wording_stratum_that_is_not_the_one_sized_refuses(self,
                                                                monkeypatch):
        f = _freeze_over_mutated_power(
            monkeypatch,
            lambda out: _mutate_portrait_block(
                out, wording_stratum_num_families=2))
        assert any("probe families" in r for r in f["refusals"]), \
            f["refusals"]
        f = _freeze_over_mutated_power(
            monkeypatch,
            lambda out: _mutate_portrait_block(
                out, wording_stratum_clusters=30))
        assert any("clusters but the size block budgets" in r
                   for r in f["refusals"]), f["refusals"]

    def test_a_collision_rule_that_hides_the_exception_refuses(self,
                                                              monkeypatch):
        """The rule a stage-3 builder reads and the exemption the freeze grants
        must say the same thing, or the builder follows the stricter one and
        the split cannot be built."""
        def mutate(out):
            out = dict(out)
            cs = dict(out["confirmation_size"])
            stage3 = dict(cs["frozen_at_stage_3_before_any_scoring"])
            stage3["collision_rules"] = [
                r for r in stage3["collision_rules"]
                if "bounded exception" not in r]
            cs["frozen_at_stage_3_before_any_scoring"] = stage3
            out["confirmation_size"] = cs
            return out

        f = _freeze_over_mutated_power(monkeypatch, mutate)
        assert any("no collision_rule states the portrait exception" in r
                   for r in f["refusals"]), f["refusals"]


# ── the retention intervals have probes behind them ────────────────

class TestTheRetentionProbesAreAllocatedNotPromised:
    """Iteration 11C-R2's finding #2, second half.  The freeze has always
    promised descriptive retain_same and retain_other intervals while
    allocating NO probes to compute them from: the 504 new wordings are
    TARGET-family probes on the 42 target persons, and the 70 entities
    retain_same is defined on (45 for retain_other) were given nothing.  A
    promised interval with no probes behind it is not a plan, and the
    intervals it would have produced describe the exploratory split the
    reference-state gate already saw."""

    def test_the_allocation_is_frozen_with_a_route_and_a_count(self):
        r = _load()["retention_probes"]
        assert r["route"] == CONFIRM_RETENTION_ROUTE == "text_only"
        assert r["new_templates_per_entity"] == \
            CONFIRM_RETENTION_NEW_TEMPLATES_PER_ENTITY == 3
        assert r["total"] == 345
        assert r["entities_covered"] == 70
        assert r["status"].startswith("DESCRIPTIVE")
        assert "no Holm entry" in r["status"]
        assert r["read_from"].startswith("retention_probe_allocation")
        assert "unfalsifiable" in r["why_this_is_in_the_freeze"]
        assert "reference-state gate" in r["why_this_is_in_the_freeze"]

    def test_the_count_is_the_sum_of_the_two_metrics_own_allocations(self):
        r = _load()["retention_probes"]
        pm = r["per_metric"]
        assert set(pm) == {"retain_same", "retain_other"}
        m = CONFIRM_RETENTION_NEW_TEMPLATES_PER_ENTITY
        assert pm["retain_same"]["entities_carried_by"] == 70
        assert pm["retain_other"]["entities_carried_by"] == 45
        for name, e in pm.items():
            assert e["route"] == r["route"], name
            assert e["new_probes_per_entity"] == m, name
            assert e["new_probes"] == m * e["entities_carried_by"], name
        assert pm["retain_same"]["new_probes"] == 210
        assert pm["retain_other"]["new_probes"] == 135
        assert r["total"] == sum(e["new_probes"] for e in pm.values())
        assert r["retain_other_entities_are_a_subset_of_retain_same"] is True
        # and the size block's totals carry the same number
        assert _load()["confirmation_size"]["totals"][
            "new_retention_probes"] == r["total"]

    def test_the_count_is_set_by_the_noise_floor_not_by_the_exploratory_number(
            self):
        """The count is not "match pilot100_v2's precision".  Precision finer
        than the batch-layout floor cannot be distinguished from batched
        decoding noise, so buying it spends generations on a decimal nobody
        can act on."""
        r = _load()["retention_probes"]
        floor = r["batch_layout_noise_floor_on_retain_metrics"]
        assert floor == 0.0556
        for name, e in r["per_metric"].items():
            assert e["selected_count_is_at_or_above_the_noise_floor"] is True, \
                name
            assert e["half_width_at_the_selected_count"] >= floor, name
            # the rejected counts are all recorded, and all finer than the
            # floor is where the argument lives
            for m, hw in e["half_width_at_rejected_counts"].items():
                assert float(hw) < e["half_width_at_the_selected_count"] \
                    or int(m) < CONFIRM_RETENTION_NEW_TEMPLATES_PER_ENTITY, \
                    (name, m, hw)
        same = r["per_metric"]["retain_same"]
        assert same["probes_per_entity_that_would_match_the_exploratory_"
                    "precision"] == 8
        assert same["half_width_at_the_exploratory_count"] < floor
        assert "cannot resolve" in r["why_this_count"]
        assert "noise floor" in r["why_this_count"]

    def test_the_image_route_omission_is_a_measured_constraint(self):
        """Not a preference.  Each of the 64 MLLMU persons has exactly ONE
        photograph in pilot100_v2, so an image-route retention probe on a
        person either reuses exploratory media - which the novelty invariant
        forbids - or does not exist."""
        r = _load()["retention_probes"]
        m = r["media_supply_the_route_decision_rests_on"]
        assert m["retention_entities"] == r["entities_covered"] == 70
        assert m["photographs_per_retention_entity"] == {"1": 64, "12": 6}
        assert m["entities_with_a_single_photograph_or_none"] == 64
        assert m["entities_a_new_photograph_could_cover"] == 6
        assert m["entities_no_new_photograph_can_cover"] == 64
        assert m["entities_a_new_photograph_could_cover"] + \
            m["entities_no_new_photograph_can_cover"] == 70
        assert m["by_source"]["inaturalist"] == 6
        assert "reuses exploratory media" in r["why_the_image_route_is_omitted"]
        assert "64" in r["why_the_image_route_is_omitted"]
        assert "no confirmation purpose" in \
            r["consequence_for_the_photograph_fetch"]

    def test_the_confirmation_retention_estimand_is_not_the_exploratory_one(
            self):
        """pilot100_v2's retain metrics pool BOTH routes.  The confirmation's
        are text-only, so the two numbers are not the same estimand and must
        not be subtracted or called a replication of each other."""
        r = _load()["retention_probes"]
        assert "BOTH routes" in \
            r["how_it_differs_from_the_exploratory_retention_number"]
        assert "must not be subtracted" in \
            r["how_it_differs_from_the_exploratory_retention_number"]
        assert "TEXT route only" in \
            r["what_the_confirmation_retention_estimand_is"]
        assert str(CONFIRM_RETENTION_NEW_TEMPLATES_PER_ENTITY) in \
            r["what_the_confirmation_retention_estimand_is"]
        # retention stays out of the claim family
        assert not set(_load()["claims"]["primary_family"]) & {
            "B3_minus_B0:retain_same", "B3_minus_B0:retain_other"}

    def test_a_report_with_no_retention_allocation_refuses(self, monkeypatch):
        def mutate(out):
            out = dict(out)
            out.pop("retention_probe_allocation")
            return out

        f = _freeze_over_mutated_power(monkeypatch, mutate)
        assert any("no retention_probe_allocation" in r
                   for r in f["refusals"]), f["refusals"]

    def test_a_different_route_refuses(self, monkeypatch):
        monkeypatch.setattr(fz, "CONFIRM_RETENTION_ROUTE", "both")
        f = build_freeze(REPO_ROOT, "pilot100")
        assert any("route" in r for r in f["refusals"]), f["refusals"]

    def test_a_different_template_count_refuses(self, monkeypatch):
        monkeypatch.setattr(
            fz, "CONFIRM_RETENTION_NEW_TEMPLATES_PER_ENTITY", 12)
        f = build_freeze(REPO_ROOT, "pilot100")
        assert any("new retention templates" in r
                   for r in f["refusals"]), f["refusals"]

    def test_an_allocation_whose_own_arithmetic_disagrees_refuses(
            self, monkeypatch):
        """The total must be the sum of the per-metric allocations.  A total
        written independently of them is a second copy of the design."""
        def mutate(out):
            out = dict(out)
            ra = dict(out["retention_probe_allocation"])
            ra["new_retention_probes_total"] = 560
            out["retention_probe_allocation"] = ra
            cs = dict(out["confirmation_size"])
            cs["totals"] = dict(cs["totals"], new_retention_probes=560,
                                total_new_probes_allocated=864 + 560)
            out["confirmation_size"] = cs
            return out

        f = _freeze_over_mutated_power(monkeypatch, mutate)
        assert any("its own per-metric allocation sums to" in r
                   for r in f["refusals"]), f["refusals"]

    def test_a_target_total_that_equals_the_overall_total_refuses(
            self, monkeypatch):
        """The 936 mislabelling, reproduced: with retention allocated, the
        target total and the overall total CANNOT be the same number, and a
        report that says they are has one label wrong."""
        def mutate(out):
            out = dict(out)
            cs = dict(out["confirmation_size"])
            cs["totals"] = dict(cs["totals"], new_target_probes=1209)
            out["confirmation_size"] = cs
            return out

        f = _freeze_over_mutated_power(monkeypatch, mutate)
        assert any("one of the two labels" in r for r in f["refusals"]), \
            f["refusals"]
        assert any("totals.new_target_probes is 1209" in r
                   for r in f["refusals"]), f["refusals"]

    def test_a_metric_nobody_carries_refuses(self, monkeypatch):
        def mutate(out):
            out = dict(out)
            ra = dict(out["retention_probe_allocation"])
            pm = {k: dict(v) for k, v in ra["per_metric"].items()}
            pm["retain_other"]["entities_carried_by"] = 0
            pm["retain_other"]["new_probes"] = 0
            ra["per_metric"] = pm
            ra["new_retention_probes_total"] = 210
            out["retention_probe_allocation"] = ra
            cs = dict(out["confirmation_size"])
            cs["totals"] = dict(cs["totals"], new_retention_probes=210,
                                total_new_probes_allocated=864 + 210)
            out["confirmation_size"] = cs
            return out

        f = _freeze_over_mutated_power(monkeypatch, mutate)
        assert any("no entity carries retain_other" in r
                   for r in f["refusals"]), f["refusals"]


# ── the primary hypothesis test is frozen, not described ────────────

class TestThePrimaryTestIsFrozenNotDescribed:
    """Iteration 11C-R2's finding #3.  Holm and the power sizing were made
    consistent in 11C-R1 - one-sided thresholds [0.025, 0.05], requirements
    of 5 TGA and 4 FILR clusters - but the repository contained no one-sided
    p-value at all.  ``paired_ci`` offered a 1000-resample percentile
    interval and nothing that returned a p, so the frozen thresholds could
    not be executed as frozen and the first person to score would have had to
    invent a test, after seeing the data."""

    def test_the_statistic_and_the_null_are_named(self):
        t = _load()["primary_test"]
        assert t["applies_to"] == list(PRIMARY_FAMILY)
        assert "entity-macro paired difference" in t["statistic"]
        # the statistic must be the quantity the published interval covers,
        # or a claim can be rejected while its own interval contains zero
        assert "paired_rate_diff_ci" in t["statistic"]
        assert "contradict each other" in t[
            "why_the_statistic_must_be_the_one_the_interval_covers"]
        assert "symmetric about zero" in t["null_hypothesis"]
        assert "2^k sign-flip vectors" in t["null_hypothesis"]
        assert t["p_value_method"] == \
            "Monte Carlo cluster sign-flip permutation"
        assert "poorly calibrated" in t["why_sign_flips_and_not_the_bootstrap"]
        assert t["interval_still_published"].startswith(
            "the percentile bootstrap CI")

    def test_the_draw_count_and_seed_are_the_module_constants(self):
        t = _load()["primary_test"]
        assert t["n_permutations"] == N_PERMUTATIONS == 10000
        assert t["permutation_seed"] == PERMUTATION_SEED == 20260908
        assert t["smallest_reportable_p_value"] == round(
            1.0 / (N_PERMUTATIONS + 1), 6)
        assert t["observed_vector_included_in_the_null"] is True
        # the smallest reportable p must sit far below the threshold it is
        # compared against, or "p <= 0.025" is an artefact of the draw count
        assert t["smallest_reportable_p_value"] < \
            HOLM_WORST_CASE_ALPHA / 10
        seeds = {t["permutation_seed"],
                 _load()["analysis"]["bootstrap"]["seed"],
                 _load()["analysis"]["icc_bootstrap_seed"]}
        assert len(seeds) == 3, seeds

    def test_each_claim_has_its_own_direction_and_they_differ(self):
        """TGA is an accuracy, so better is HIGHER; FILR is a leakage rate, so
        better is LOWER.  Both claims are about the same B3-B0 difference and
        point in OPPOSITE directions on it, so one shared sign convention
        would test one of them backwards.  The sizing code sizes |theta|,
        which is right for a power table and wrong for a test."""
        from granunlearn.evaluation.paired_ci import CLAIM_DIRECTION
        t = _load()["primary_test"]
        assert t["direction_by_metric"] == dict(CLAIM_DIRECTION) == \
            {"tga": "greater", "filr": "less"}
        assert t["alternative_by_claim"] == {
            "B3_minus_B0:tga": "theta > 0",
            "B3_minus_B0:filr": "theta < 0"}
        assert len(set(t["direction_by_metric"].values())) == 2
        assert "OPPOSITE directions" in t["why_the_directions_differ"]
        assert "|theta|" in t["why_the_directions_differ"]

    def test_holm_is_frozen_with_its_ordering_ties_and_stopping_rule(self):
        t = _load()["primary_test"]["multiplicity"]
        k = len(PRIMARY_FAMILY)
        assert t["procedure"] == "Holm step-down"
        assert t["familywise_alpha"] == FAMILYWISE_ALPHA
        assert t["thresholds_apply_to"] == "one-sided p-values"
        assert t["thresholds"] == [round(FAMILYWISE_ALPHA / (k - i), 6)
                                   for i in range(k)] == [0.025, 0.05]
        assert t["worst_case_alpha_for_a_single_claim"] == \
            round(HOLM_WORST_CASE_ALPHA, 6) == 0.025
        assert t["ordering"] == "ascending p-value, ties broken by claim name"
        assert "cannot change a verdict" in t["tie_handling"]
        assert "FIRST non-rejection ends the procedure" in t["pass_fail_rule"]
        assert "retained whether or not it clears its own threshold" in \
            t["pass_fail_rule"]
        assert "understates the familywise error rate" in \
            t["why_the_stopping_rule_is_part_of_the_rule"]
        # the worked examples are the real function's output, not a table
        ex = _load()["primary_test"]["worked_examples"]
        assert set(ex) == {"both_clear",
                           "first_fails_so_the_second_is_retained"}
        assert ex["both_clear"]["all_rejected"] is True
        assert ex["first_fails_so_the_second_is_retained"]["rejected"] == []
        assert ex["first_fails_so_the_second_is_retained"]["all_rejected"] \
            is False
        steps = ex["first_fails_so_the_second_is_retained"]["steps"]
        assert [s["clears_its_own_threshold"] for s in steps] == \
            [False, True], steps
        assert [s["rejected"] for s in steps] == [False, False], steps

    def test_the_level_the_frozen_test_achieves_is_measured_not_assumed(self):
        """A threshold is only a threshold if the procedure delivers it.  The
        achieved level is measured by re-signing the REAL 11R differences,
        which is the null the test assumes and preserves the real
        discreteness."""
        t = _load()["primary_test"]
        cal = t["achieved_level_under_the_real_null"]
        assert set(cal) == set(PRIMARY_FAMILY)
        for claim, e in cal.items():
            assert e["num_clusters"] == 72, claim
            assert e["threshold"] == HOLM_WORST_CASE_ALPHA, claim
            assert e["replicates"] >= 1000, claim
            assert e["permutations_are_the_frozen_count"] is True, claim
            assert e["permutations_per_replicate"] == N_PERMUTATIONS, claim
            assert e["sign_flips_shared_with_the_other_claims"] is True, claim
            assert e["achieved_level_is_nominal_within_two_se"] is True, claim
            lo, hi = e["two_se_band_around_the_nominal"]
            assert lo <= e["achieved_level_at_the_threshold"] <= hi, claim
            # uniform p-values under H0 have mean 0.5; this is the sharper of
            # the two diagnostics because every replicate contributes
            assert 0.4 < e["mean_p_under_the_null"] < 0.6, claim
            assert 0 < e["zero_difference_fraction"] < 0.5, claim

    def test_the_familywise_rate_is_measured_on_the_procedure_itself(self):
        """Marginal levels at every step do NOT imply the familywise rate:
        the two claims share their 72 clusters and are correlated, so the
        joint re-signing applies ONE sign vector per entity to BOTH and runs
        the real ``holm_family``.  This is the number the declared familywise
        alpha is a promise about."""
        fw = _load()["primary_test"][
            "achieved_familywise_level_under_the_global_null"]
        assert fw["nominal"] == FAMILYWISE_ALPHA
        assert fw["procedure"].startswith("holm_family as implemented")
        assert fw["one_sign_vector_per_entity_shared_by_both_claims"] is True
        assert fw["achieved_is_nominal_within_two_se"] is True
        lo, hi = fw["two_se_band_around_the_nominal"]
        assert lo <= fw["achieved"] <= hi
        assert fw["replicates"] >= 1000
        assert "not the per-claim marginal levels" in fw["what_it_measures"]

    def test_the_implementation_is_hashed_and_the_hash_is_recomputed(self):
        """``paired_ci.py`` is NOT in CODE_FINGERPRINT_MODULES - adding it
        would change the code fingerprint inside all 30 committed sidecars and
        refuse their reuse - and before 11C-R2 it was bound nowhere, so the
        procedure deciding the primary claims was the one part of the
        analysis nothing pinned."""
        from granunlearn.evaluation.prediction_provenance import (
            CODE_FINGERPRINT_MODULES, sha256_file)
        t = _load()["primary_test"]["implementation"]
        assert t["module"] == fz.PAIRED_CI_MODULE == \
            "src/granunlearn/evaluation/paired_ci.py"
        assert t["in_the_code_fingerprint"] is False
        assert t["module"] not in CODE_FINGERPRINT_MODULES
        assert t["sha256"] == sha256_file(REPO_ROOT / t["module"])
        assert len(t["sha256"]) == 64
        assert "refuse their reuse" in t["why_it_is_hashed_separately"]
        assert "recomputed HERE" in t["why_it_is_hashed_separately"]
        assert set(t["functions"]) == {
            "one_sided_permutation_pvalue", "holm_family",
            "paired_rate_diff_ci"}
        # the signatures are recorded so a changed parameter list is drift
        assert t["signatures"]["one_sided_permutation_pvalue"] == [
            "diffs", "direction", "n_permutations", "seed"]
        assert t["signatures"]["holm_family"] == ["p_values",
                                                  "familywise_alpha"]

    def test_the_specification_and_its_implementation_agree(self):
        a = _load(POWER_PATH)["primary_test"][
            "specification_and_implementation_agree"]
        assert a["all_agree"] is True
        assert a["n_permutations_is_the_declared_count"] is True
        assert a["seed_is_the_declared_seed"] is True
        assert a["claim_direction_covers_exactly_the_primary_metrics"] is True
        assert a["holm_family_takes_alpha_as_a_required_argument"] is True
        read = a["every_value_read_off_the_implementation"]
        assert read["n_permutations"] == N_PERMUTATIONS
        assert read["seed"] == PERMUTATION_SEED

    def test_a_moved_permutation_count_refuses(self, monkeypatch):
        monkeypatch.setattr(fz, "N_PERMUTATIONS", 1000)
        f = build_freeze(REPO_ROOT, "pilot100")
        assert any("permutations but the module declares" in r
                   for r in f["refusals"]), f["refusals"]

    def test_a_moved_seed_refuses(self, monkeypatch):
        monkeypatch.setattr(fz, "PERMUTATION_SEED", 42)
        f = build_freeze(REPO_ROOT, "pilot100")
        assert any("permutation seed" in r for r in f["refusals"]), \
            f["refusals"]

    def test_one_direction_for_both_claims_refuses(self, monkeypatch):
        """The failure mode that would test FILR backwards."""
        monkeypatch.setattr(fz, "CLAIM_DIRECTION",
                            {"tga": "greater", "filr": "greater"})
        f = build_freeze(REPO_ROOT, "pilot100")
        assert any("backwards" in r for r in f["refusals"]), f["refusals"]

    def test_an_edited_paired_ci_refuses(self, monkeypatch):
        """The freeze re-hashes the module rather than trusting the report's
        copy, so an edit between the analysis and the freeze refuses."""
        real = fz.sha256_file
        monkeypatch.setattr(
            fz, "sha256_file",
            lambda p: "0" * 64 if str(p).endswith("paired_ci.py")
            else real(p))
        f = build_freeze(REPO_ROOT, "pilot100")
        assert any("changed after the analysis that specified it" in r
                   for r in f["refusals"]), f["refusals"]

    def test_a_test_that_overshoots_its_own_threshold_refuses(
            self, monkeypatch):
        """Non-vacuity of the calibration gate: freezing a threshold the
        procedure does not deliver would publish a familywise rate the test
        does not control."""
        def mutate(out):
            out = dict(out)
            pt = dict(out["primary_test"])
            cal = {k: dict(v) for k, v in
                   pt["achieved_level_under_the_real_null"].items()}
            cal["B3_minus_B0:tga"][
                "achieved_level_is_nominal_within_two_se"] = False
            pt["achieved_level_under_the_real_null"] = cal
            fw = dict(pt["achieved_familywise_level_under_the_global_null"])
            fw["achieved_is_nominal_within_two_se"] = False
            pt["achieved_familywise_level_under_the_global_null"] = fw
            out["primary_test"] = pt
            return out

        f = _freeze_over_mutated_power(monkeypatch, mutate)
        assert any("the procedure delivers" in r for r in f["refusals"]), \
            f["refusals"]
        assert any("familywise rejection rate" in r
                   for r in f["refusals"]), f["refusals"]

    def test_a_report_with_no_primary_test_refuses(self, monkeypatch):
        def mutate(out):
            out = dict(out)
            out.pop("primary_test")
            return out

        f = _freeze_over_mutated_power(monkeypatch, mutate)
        assert any("no primary_test block" in r for r in f["refusals"]), \
            f["refusals"]

    def test_moved_holm_thresholds_refuse(self, monkeypatch):
        def mutate(out):
            out = dict(out)
            pt = dict(out["primary_test"])
            pt["multiplicity"] = dict(pt["multiplicity"],
                                      thresholds=[0.0125, 0.025])
            out["primary_test"] = pt
            return out

        f = _freeze_over_mutated_power(monkeypatch, mutate)
        assert any("Holm thresholds" in r for r in f["refusals"]), \
            f["refusals"]


# ── 0.025 has two sources and only one of them applies here ──────────

class TestTheAlphaConventionSeparatesItsTwoSources:
    """Iteration 11C-R2's finding #3, second half.  The freeze and the power
    report both said a single unadjusted one-sided claim is tested at 0.025.
    A STANDALONE directional claim at familywise 0.05 is tested at one-sided
    0.05: one-sidedness spends nothing.  0.025 is right here because it is
    Holm's first threshold for k = 2 - and separately because it is the one
    arm of a two-sided 95% interval that a TOST or non-inferiority bound is
    read off at.  The two coincide only at k = 2."""

    def test_both_levels_are_named_and_only_one_is_used(self):
        a = _load()["analysis"]
        assert a["familywise_alpha"] == FAMILYWISE_ALPHA == 0.05
        assert a["alpha_one_sided"] == ALPHA_ONE_SIDED == 0.025
        assert a["alpha_standalone_one_sided"] == \
            ALPHA_STANDALONE_ONE_SIDED == 0.05
        assert a["holm_worst_case_alpha"] == HOLM_WORST_CASE_ALPHA == 0.025
        assert a["alpha_standalone_one_sided"] != a["holm_worst_case_alpha"]

    def test_the_convention_no_longer_calls_0025_the_unadjusted_level(self):
        a = _load()["analysis"]["alpha_convention"]
        assert "unadjusted" not in a
        assert "does not halve" in a
        assert "STANDALONE" in a
        assert "property of k=2" in a or "only because" in a
        # and the same correction is in the power report, in both places it
        # appeared
        p = _load(POWER_PATH)
        for text in (p["design"]["alpha_convention"],):
            assert "unadjusted" not in text
            assert "does NOT halve" in text
            assert "0.0167" in text, text   # what Holm's first step is at k=3
        chosen = [d for d in p["preregistration_decisions"]["decisions"]
                  if d["id"] == "familywise_alpha"][0]
        assert "STANDALONE" in chosen["chosen"]
        assert any("unadjusted one-sided claim" in r
                   for r in chosen["rejected"]), chosen["rejected"]

    def test_the_sizing_uses_the_holm_threshold_and_not_the_standalone_one(
            self):
        c = _load()["claims"]
        assert c["sizing_alpha_one_sided"] == HOLM_WORST_CASE_ALPHA == 0.025
        assert c["sizing_alpha_one_sided"] != ALPHA_STANDALONE_ONE_SIDED
        assert c["sizing_critical_value_z"] == \
            round(z(1 - HOLM_WORST_CASE_ALPHA), 6)
        assert "z(1 - familywise/k)" in c["thresholds_and_sizing_agree"]
        assert "same test" in c["thresholds_and_sizing_agree"]
        assert c["clusters_required_at_that_threshold"] == {
            "B3_minus_B0:tga": 5, "B3_minus_B0:filr": 4}

    def test_the_test_thresholds_and_the_sizing_are_the_same_number(self):
        """The point of the correction: 11C-R1 made the thresholds and the
        sizing agree, and 11C-R2 must not break that while fixing the
        prose."""
        f = _load()
        assert f["primary_test"]["multiplicity"]["thresholds"][0] == \
            f["claims"]["sizing_alpha_one_sided"] == \
            f["analysis"]["holm_worst_case_alpha"]
        assert f["primary_test"]["multiplicity"][
            "worst_case_alpha_for_a_single_claim"] == \
            f["claims"]["sizing_alpha_one_sided"]


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
        drift = drift_keys(committed, fresh)
        assert "code" in drift, drift


# ── the sealed-split invariants name an enforcement point ──────────

class TestTheSealedSplitInvariantsAreActionable:
    def test_every_invariant_names_what_enforces_it_and_when(self):
        """An invariant with no enforcement point is a comment.  Each one
        must say which artifact or stage checks it, so stage 5 can be held to
        it."""
        f = _load()
        inv = f["sealed_split_invariants"]
        assert len(inv) == 8
        for item in inv:
            # the four keys every invariant must carry; extras are allowed
            # because Iteration 11C-R2's identity invariant also binds the
            # hashes it is checked against, and a hash with no key to live in
            # would be prose
            assert {"invariant", "why", "enforced_by",
                    "checked_at"} <= set(item), item
            for key in ("invariant", "why", "enforced_by"):
                assert item[key] and len(item[key]) > 15, (key, item[key])
            # the enforcement point must be a named stage of the
            # confirmation sequence, not "later" or "somewhere"
            assert item["checked_at"].startswith("stage"), item["checked_at"]

    def test_what_must_be_identical_and_what_must_be_new_are_separate(self):
        """Iteration 11C-R2's finding #1.  One invariant used to forbid any
        confirmation ASSOCIATION from appearing in the exploratory dataset,
        which is impossible: the confirmation re-tests the same
        entity-attribute associations with new photographs and new wording.
        The requirement is now split, and each side is bound by a hash."""
        f = _load()
        inv = f["sealed_split_invariants"]
        same = [i for i in inv if "IDENTICAL" in i["invariant"]]
        assert len(same) == 1, [i["invariant"] for i in inv]
        b = same[0]["bound_by"]
        fn = f["confirmation_size"]["frozen_now"]
        assert b["target_association_ids_sha256"] == \
            fn["target_association_ids_sha256"]
        assert len(b["target_association_ids_sha256"]) == 64
        assert b["target_association_ids"] == \
            len(fn["target_association_ids"]) > 0
        assert b["target_entity_ids_sha256"] == fn["target_entity_ids_sha256"]
        assert b["target_entity_ids"] == 72 == len(fn["target_entity_ids"])
        assert "could not confirm" in same[0]["why"]
        assert "repeating the probes would test nothing" in \
            same[0]["why_it_is_not_a_leak"]

        new = [i for i in inv if "is NEW" in i["invariant"]]
        assert len(new) == 1, [i["invariant"] for i in inv]
        nb = new[0]["bound_by"]
        assert nb["exploratory_template_ids_sha256"] == \
            fn["exploratory_template_ids_sha256"]
        assert nb["exploratory_template_ids"] == \
            len(fn["exploratory_template_ids"]) == 57
        assert (REPO_ROOT
                / nb["exploratory_photograph_sha256_manifest"]).exists()
        assert "template TEXT" in new[0]["invariant"]
        assert "same probe with a new label" in \
            new[0]["why_template_text_and_not_only_template_id"]

        reuse = [i for i in inv if "no exploratory QUERY or MEDIA" in
                 i["invariant"]]
        assert len(reuse) == 1, [i["invariant"] for i in inv]
        assert "association-disjointness" in reuse[0]["why"]
        assert reuse[0]["bound_by"]["what_may_not"].startswith("query_ids")

    def test_no_invariant_forbids_the_associations_from_repeating(self):
        """The impossible rule is searched for, not just its replacement
        asserted present: a freeze could carry both and the contradiction
        would only surface at stage 3."""
        f = _load()
        text = " ".join(json.dumps(i) for i in f["sealed_split_invariants"])
        assert "no confirmation query_id, association" not in text
        for item in f["sealed_split_invariants"]:
            rule = item["invariant"].lower()
            if "association" in rule:
                assert "identical" in rule, item["invariant"]
        # and the collision rules the stage-3 builder reads agree
        rules = _load()["confirmation_size"][
            "frozen_at_stage_3_before_any_scoring"]["collision_rules"]
        assert any("REQUIRED to be IDENTICAL" in r for r in rules), rules
        assert not any(r.startswith("no confirmation association")
                       for r in rules)

    def test_the_three_ways_the_split_could_leak_are_all_covered(self):
        """The user's constraint was that the confirmation split must never
        enter the reference gate, candidate selection, or another go/no-go
        decision.  All three appear, plus the identity and novelty rules they
        depend on."""
        f = _load()
        text = " ".join(i["invariant"] for i in f["sealed_split_invariants"])
        assert "reference-state gate" in text
        assert "candidate selection" in text
        assert "go/no-go" in text
        assert "query_id" in text and "photograph" in text
        assert "IDENTICAL" in text and "NEW" in text

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
        # drift_keys is the script's own comparison, imported rather than
        # restated: a test that reimplemented it could pass while --check-only
        # disagreed
        assert drift_keys(_load(), build_freeze(REPO_ROOT, "pilot100")) == []

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
        allowed to fail the drift check.

        So the property is NOT that the recorded interpreter matches the one
        running this test, and asserting that was a bug: the freeze was
        written under 3.10.20 with torch 2.8.0+cu128 while the GitHub runner
        is 3.10.21 with no torch installed at all, so equality fails on every
        machine except the one that wrote the freeze.  That is the reason the
        field is excluded, not a defect in the exclusion.  What IS assertable
        anywhere is that a difference in this block does not register as
        drift - so the difference is MANUFACTURED here rather than inherited
        from whichever two interpreters happen to be in play.
        """
        from granunlearn.evaluation.prediction_provenance import (
            environment_fingerprint)
        assert "environment" in VOLATILE_FREEZE_FIELDS
        live = environment_fingerprint()
        committed = _load()
        recorded = committed["environment"]
        assert set(live) == {"python_executable", "python_version",
                             "package_versions"}
        # same SHAPE whatever the values.  The probed package list is a fixed
        # tuple inside environment_fingerprint(), so its KEYS are
        # machine-independent even though its versions are not.
        assert set(recorded) == set(live)
        assert set(recorded["package_versions"]) == \
            set(live["package_versions"])
        assert re.fullmatch(r"\d+\.\d+\.\d+", recorded["python_version"])
        assert recorded["python_executable"]

        # a different interpreter, different versions, no torch at all: the
        # block differs and the drift check must not care
        elsewhere = dict(
            recorded,
            python_executable="/opt/hostedtoolcache/Python/3.10.21/x64/bin/"
                              "python",
            python_version="9.99.99",
            package_versions={k: None for k in recorded["package_versions"]})
        assert elsewhere != recorded
        assert drift_keys(committed, dict(committed,
                                          environment=elsewhere)) == []

        # and the comparison is not simply blind: a BINDING block still
        # flags, so the empty result above is the exclusion doing its job
        retuned = dict(committed, analysis=dict(committed["analysis"],
                                                icc_bootstrap_seed=999))
        assert drift_keys(committed, retuned) == ["analysis"]

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


# ── which fact each probe asks, and when the rule was fixed ────────

def _mutate_target_alloc(out, **fields):
    """Rewrite fields of ``probe_allocation.target_associations``."""
    out = dict(out)
    pa = dict(out["probe_allocation"])
    pa["target_associations"] = dict(pa["target_associations"], **fields)
    out["probe_allocation"] = pa
    return out


def _mutate_retention(out, **fields):
    """Rewrite fields of ``probe_allocation.retention``."""
    out = dict(out)
    pa = dict(out["probe_allocation"])
    pa["retention"] = dict(pa["retention"], **fields)
    out["probe_allocation"] = pa
    return out


def _mutate_amendment(out, **fields):
    """Rewrite the single disclosed amendment."""
    out = dict(out)
    block = dict(out["protocol_amendments"])
    block["amendments"] = [dict(block["amendments"][0], **fields)]
    out["protocol_amendments"] = block
    return out


class TestWhichFactEachProbeAsksIsFrozenAndRefusable:
    """Iteration 11C stage 3, review findings 2 and 3.

    Two rules decide which FACT a probe asks.  Because the primary statistic
    averages over entities, both are choices about the estimand, so both are
    bound here and refused if the measurement behind them stops holding.  Each
    refusal below was observed to fire.
    """

    def test_the_real_report_produces_no_refusal(self):
        assert fz._probe_allocation_refusals(_load(POWER_PATH)) == []

    def test_the_freeze_binds_the_rules_the_module_declares(self):
        alloc = _load()["probe_allocation"]
        assert alloc["target_association_rule"] == \
            TARGET_ASSOCIATION_ALLOCATION_RULE
        assert alloc["retention_sampling_rule"] == RETENTION_SAMPLING_RULE
        assert alloc["retention_sampling_salts"] == \
            dict(sorted(RETENTION_SAMPLING_SALTS.items()))
        assert alloc["retention_estimand_label"] == RETENTION_ESTIMAND_LABEL

    def test_the_freeze_binds_the_measured_balance_and_the_rejected_rule(self):
        alloc = _load()["probe_allocation"]
        assert alloc["target_associations_per_person"] == \
            {"1": 27, "2": 12, "3": 3}
        assert alloc["probes_per_association_given_its_size"] == \
            {"1": [[12]], "2": [[6, 6]], "3": [[4, 4, 4]]}
        assert alloc["balanced_sizes"] == ["1", "2", "3"]
        assert alloc["unbalanced_sizes"] == []
        assert alloc["allocation_is_balanced_for_every_observed_size"] is True
        mix = alloc["attribute_mix_over_retain_same_probes"]
        assert mix["attributes_the_rejected_rule_would_have_excluded"] == \
            ["salary"]
        #: Absent, not zero: the mix counts what was selected, so an attribute
        #: the rejected rule never selects has no key at all.  That absence is
        #: the measurement, and it is what the exclusion list above names.
        assert "salary" not in mix["rejected_sorted_first_three"]
        assert mix["rejected_sorted_first_three"]["residence"] == 1
        assert mix["rejected_sorted_first_three"]["occupation"] == 6
        assert mix["this_rule"]["salary"] > 0
        assert mix["attributes_this_rule_covers"] == \
            mix["attributes_in_the_population"] == 8

    def test_the_allocation_rows_are_bound_by_hash_not_copied(self):
        """The 504 + 210 + 135 rows live in the power report; the freeze binds
        their hashes so a builder that allocated differently fails a hash
        rather than disagreeing with prose nobody re-reads."""
        alloc = _load()["probe_allocation"]
        p = _load(POWER_PATH)["probe_allocation"]
        hashes = alloc["row_hashes"]
        assert set(hashes) == {"target_association_rows", "retain_same_rows",
                               "retain_other_rows"}
        for key, rows in (("target_association_rows",
                           p["target_associations"]["rows"]),
                          ("retain_same_rows",
                           p["retention"]["rows_retain_same"]),
                          ("retain_other_rows",
                           p["retention"]["rows_retain_other"])):
            assert hashes[key] == hashlib.sha256(json.dumps(
                rows, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest(), key
        assert len({v for v in hashes.values()}) == 3
        assert alloc["rows_total"] == 849
        assert alloc["photograph_probes"] == 360
        assert alloc["confirmation_probes_total"] == 1209

    def test_a_rule_that_is_not_the_modules_own_refuses(self):
        out = _mutate_target_alloc(
            _load(POWER_PATH), rule=TARGET_ASSOCIATION_ALLOCATION_RULE + " ")
        assert any("not the module's" in r
                   for r in fz._probe_allocation_refusals(out))

    def test_a_salt_that_moves_refuses(self):
        """The rank is a function of the salt, so a different salt is a
        different sample and the frozen rows would no longer be the rows."""
        out = _mutate_retention(
            _load(POWER_PATH),
            salts={"retain_same_entity": "moved",
                   "retain_other_entity":
                       RETENTION_SAMPLING_SALTS["retain_other_entity"]})
        assert any("not the module's RETENTION_SAMPLING_SALTS" in r
                   for r in fz._probe_allocation_refusals(out))

    def test_salts_that_are_not_domain_separated_refuse(self, monkeypatch):
        """One salt for both families makes the second ranking a deterministic
        function of the first, which is not a second sample."""
        monkeypatch.setattr(fz, "RETENTION_SAMPLING_SALTS",
                            {"retain_same_entity": "one",
                             "retain_other_entity": "one"})
        assert any("not domain-separated" in r
                   for r in fz._probe_allocation_refusals(_load(POWER_PATH)))

    def test_an_unbalanced_allocation_refuses(self):
        out = _mutate_target_alloc(
            _load(POWER_PATH),
            allocation_is_balanced_for_every_observed_size=False,
            unbalanced_sizes=["3"])
        refusals = fz._probe_allocation_refusals(out)
        assert any("uneven for association-list size(s)" in r
                   for r in refusals), refusals

    def test_balance_claimed_over_a_size_the_data_lacks_refuses(self):
        """The vacuous-truth shape: a flag quantified over sizes that divide 12
        reports success when no observed size divides 12, which is exactly when
        the allocation is uneven."""
        out = _mutate_target_alloc(
            _load(POWER_PATH),
            every_observed_size_divides_the_probe_count=False)
        assert any("does not divide" in r
                   for r in fz._probe_allocation_refusals(out))

    def test_a_retention_estimand_that_is_not_labelled_a_sample_refuses(self):
        """The label is what stops this rate being quoted against the 11R one,
        which averaged every retained fact rather than a sampled three."""
        out = _mutate_retention(_load(POWER_PATH),
                                estimates="retained facts, text route")
        assert any("rather than" in r
                   for r in fz._probe_allocation_refusals(out))

    def test_retention_rows_that_are_not_three_per_entity_refuse(self):
        out = _mutate_retention(
            _load(POWER_PATH),
            retain_same=dict(_load(POWER_PATH)["probe_allocation"]
                             ["retention"]["retain_same"], rows=209))
        assert any("retain_same allocates 209 rows" in r
                   for r in fz._probe_allocation_refusals(out))
        out = _mutate_retention(
            _load(POWER_PATH),
            retain_other=dict(_load(POWER_PATH)["probe_allocation"]
                              ["retention"]["retain_other"], rows=134))
        assert any("retain_other allocates 134 rows" in r
                   for r in fz._probe_allocation_refusals(out))

    def test_an_attribute_the_sample_never_asks_refuses(self):
        out = _mutate_retention(
            _load(POWER_PATH),
            attribute_mix_over_retain_same_probes=dict(
                _load(POWER_PATH)["probe_allocation"]["retention"]
                ["attribute_mix_over_retain_same_probes"],
                attributes_this_rule_covers=7))
        assert any("would never be asked about" in r
                   for r in fz._probe_allocation_refusals(out))

    def test_losing_the_evidence_against_sorted_order_refuses(self):
        """The rejection of sorted order rests on a measurement.  A report that
        no longer records it would leave the rejection asserting itself."""
        out = _mutate_retention(
            _load(POWER_PATH),
            attribute_mix_over_retain_same_probes=dict(
                _load(POWER_PATH)["probe_allocation"]["retention"]
                ["attribute_mix_over_retain_same_probes"],
                attributes_the_rejected_rule_would_have_excluded=[]))
        assert any("rejects the rule" in r or "rests on nothing measured" in r
                   for r in fz._probe_allocation_refusals(out))

    def test_a_rejected_rule_that_stops_rejecting_itself_refuses(self):
        """If the sorted allocation is recomputed into one that contains salary
        facts, the evidence no longer supports the decision it was recorded
        for -- which means either the ids stopped sorting by attribute or the
        measurement is wrong, and both have to stop the freeze."""
        out = _mutate_retention(
            _load(POWER_PATH),
            attribute_mix_over_retain_same_probes=dict(
                _load(POWER_PATH)["probe_allocation"]["retention"]
                ["attribute_mix_over_retain_same_probes"],
                rejected_sorted_first_three={"salary": 49}))
        assert any("no longer rejects the rule" in r
                   for r in fz._probe_allocation_refusals(out))

    def test_two_budgets_for_one_probe_count_refuse(self):
        out = dict(_load(POWER_PATH))
        pa = dict(out["probe_allocation"], confirmation_probes_total=1210)
        out["probe_allocation"] = pa
        assert any("confirmation_probes_total is 1210" in r
                   for r in fz._probe_allocation_refusals(out))

    def test_a_phograph_budget_that_disagrees_with_the_frozen_one_refuses(
            self):
        """The 360 photograph probes are not allocated by these rules, but they
        are frozen elsewhere in the same report; two numbers for one budget is
        how a total quietly becomes two totals."""
        out = dict(_load(POWER_PATH))
        pa = dict(out["probe_allocation"])
        pa["photograph_probes_are_not_allocated_here"] = dict(
            pa["photograph_probes_are_not_allocated_here"], count=432)
        out["probe_allocation"] = pa
        refusals = fz._probe_allocation_refusals(out)
        assert any("while the frozen held-out-photograph budget states 360"
                   in r for r in refusals), refusals

    def test_a_missing_allocation_block_refuses_rather_than_omitting(self):
        out = dict(_load(POWER_PATH))
        out.pop("probe_allocation")
        refusals = fz._probe_allocation_refusals(out)
        assert len(refusals) == 1
        assert "no probe_allocation block" in refusals[0]

    def test_the_refusals_are_wired_into_the_freeze(self, monkeypatch):
        """A refusal that is never reached refuses nothing."""
        f = _freeze_over_mutated_power(
            monkeypatch,
            lambda out: _mutate_target_alloc(
                out, allocation_is_balanced_for_every_observed_size=False,
                unbalanced_sizes=["2"]))
        assert any("uneven for association-list size(s)" in r
                   for r in f["refusals"]), f["refusals"]


class TestTheAmendmentIsDisclosedInTheFreeze:
    """Iteration 11C stage 3, review finding 1.

    The photograph-selection rule was described as sealed before the fetch.
    The repository's own timestamps say otherwise, so the freeze discloses the
    amendment with its ordering measured from artifacts -- and refuses to
    freeze if the disclosure stops holding, so it cannot quietly become a
    claim of preregistration.
    """

    def test_the_real_report_produces_no_refusal(self):
        assert fz._protocol_amendment_refusals(_load(POWER_PATH)) == []

    def test_the_freeze_carries_the_amendment_and_the_retraction(self):
        am = _load()["protocol_amendment"]
        assert am["kind"] == "outcome-blind protocol amendment"
        assert "PHOTO_SELECTION_RULE" in am["what"]
        assert am["prior_freeze"]["contained_the_rule"] is False
        assert am["amending_freeze"]["contained_the_rule"] is True
        assert am["confirmation_prediction_files"] == 0
        assert "BEFORE the fetch" in am["retracted_claim"]
        assert am["ordering"]["every_ordering_claim_is_measured"] is True
        assert "independent seed" in \
            am["if_literal_pre_fetch_preregistration_is_required"]

    def test_the_disclosed_timeline_is_ordered(self):
        am = _load()["protocol_amendment"]
        assert am["prior_freeze"]["frozen_at_utc"] \
            < am["pool_acquired_at_utc"] \
            < am["amending_freeze"]["frozen_at_utc"] \
            < am["subset_selected_at_utc"]

    def test_a_second_amendment_refuses(self):
        """An amendment list that can silently grow is not a disclosure."""
        out = dict(_load(POWER_PATH))
        block = dict(out["protocol_amendments"])
        block["amendments"] = list(block["amendments"]) * 2
        out["protocol_amendments"] = block
        refusals = fz._protocol_amendment_refusals(out)
        assert any("2 protocol amendments" in r for r in refusals), refusals

    def test_a_missing_disclosure_refuses(self):
        out = dict(_load(POWER_PATH))
        out.pop("protocol_amendments")
        assert fz._protocol_amendment_refusals(out)

    def test_calling_it_preregistration_refuses(self):
        out = _mutate_amendment(_load(POWER_PATH), kind="preregistration")
        assert any("only label the timestamps support" in r
                   for r in fz._protocol_amendment_refusals(out))

    def test_an_amendment_that_does_not_name_the_rule_refuses(self):
        out = _mutate_amendment(_load(POWER_PATH), what="the retention rule")
        assert any("does not name the photograph selection rule" in r
                   for r in fz._protocol_amendment_refusals(out))

    @pytest.mark.parametrize("key", [
        "rule_was_absent_when_the_pool_was_fetched",
        "rule_was_published_after_the_pool",
        "rule_was_published_before_the_subset_was_selected",
        "rule_was_published_before_any_model_output",
        "every_ordering_claim_is_measured",
    ])
    def test_any_ordering_claim_that_stops_holding_refuses(self, key):
        power = _load(POWER_PATH)
        ordering = dict(power["protocol_amendments"]["amendments"][0]
                        ["ordering"], **{key: False})
        out = _mutate_amendment(power, ordering=ordering)
        refusals = fz._protocol_amendment_refusals(out)
        assert any(key in r for r in refusals), refusals

    def test_claiming_the_prior_freeze_had_the_rule_refuses(self):
        out = _mutate_amendment(
            _load(POWER_PATH),
            prior_freeze=dict(_load(POWER_PATH)["protocol_amendments"]
                              ["amendments"][0]["prior_freeze"],
                              contained_the_rule=True))
        assert any("contradicts its own claim" in r
                   for r in fz._protocol_amendment_refusals(out))

    def test_scoring_before_amending_refuses(self):
        """Once a confirmation prediction exists the amendment is no longer
        outcome-blind, and the freeze may not keep claiming it is."""
        out = _mutate_amendment(_load(POWER_PATH),
                                confirmation_prediction_files=3)
        assert any("no longer outcome-blind" in r
                   for r in fz._protocol_amendment_refusals(out))

    def test_dropping_the_retraction_refuses(self):
        """Leaving a false timeline in the prose beside a true one is worse
        than either alone."""
        out = dict(_load(POWER_PATH))
        out["protocol_amendments"] = dict(out["protocol_amendments"],
                                          retracted_claim="")
        assert any("does not retract" in r
                   for r in fz._protocol_amendment_refusals(out))

    def test_the_refusals_are_wired_into_the_freeze(self, monkeypatch):
        f = _freeze_over_mutated_power(
            monkeypatch,
            lambda out: _mutate_amendment(out, kind="preregistration"))
        assert any("only label the timestamps support" in r
                   for r in f["refusals"]), f["refusals"]

    def test_no_tracked_source_still_carries_the_false_timeline(self):
        """Correcting a claim in the place you remembered is not correcting it.

        The first correction fixed the module comment and the report prose and
        left two copies behind -- a section header in this file and a module
        docstring in the selection tests -- both of which kept telling a reader
        the rule predated the fetch.  A sweep over every tracked source file is
        the only check that catches the copies nobody remembered, so the
        disclosure cannot be contradicted by the code beside it.
        """
        #: Assembled from fragments so that THIS file does not contain the
        #: claims it sweeps for.  A sweep that has to allowlist its own source
        #: is a sweep that can be defeated by editing the allowlist, and the
        #: two files it would have to allowlist are the two that already
        #: proved the point by carrying stale copies.
        lower, upper = "before the fetch", "BEFORE the fetch"
        phrases = (f"Frozen {lower} runs",
                   f"Frozen here, {lower} is run",
                   f"sealed {upper} ran",
                   f"decided {lower}")
        offenders = []
        for root in ("scripts", "src", "tests"):
            for path in sorted((REPO_ROOT / root).rglob("*.py")):
                text = path.read_text(errors="replace")
                for phrase in phrases:
                    if phrase in text:
                        offenders.append(
                            f"{path.relative_to(REPO_ROOT)}: {phrase!r}")
        assert offenders == [], offenders
        #: The fragment all four phrases share is what makes the sweep
        #: non-vacuous: if the retraction stopped naming it, the sweep would be
        #: searching for a claim the repository no longer retracts.
        assert upper in _load()["protocol_amendment"]["retracted_claim"]
