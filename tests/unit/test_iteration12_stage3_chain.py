"""The Stage-3 lane chain's completeness gate, and the chain's lane discipline.

Two things are pinned here.  First, the adapter gate between training and
generation: it must be able to say YES on a structurally complete row, or it is
not a gate but a constant refusal, and it must fire on a truncated container, a
wrong anchor weight, a missing summary and rows that share bytes.  Second, the
chain's lane bookkeeping: one log per lane, one captured exit status per lane,
the claimed device recorded from inside the lane, and generation ordered after
the completeness gate rather than after a file-existence test.

The gate is exercised against a fabricated checkpoint tree under ``tmp_path``
whose adapters are symlinks to the incumbent B6's real bytes, so a positive
verdict is measured on a genuine safetensors container rather than on a stub.
Nothing is written inside the repository: a B7 adapter under the real Stage-3
root would permanently lock the freeze.
"""

from __future__ import annotations

import json
import shutil
import struct
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import check_iter12_stage3_adapters as cis3
import freeze_iter12_stage3 as fis3

from granunlearn.training import stage2_grid as s2g
from granunlearn.training import stage3_grid as s3g

PY = sys.executable
CHAIN = REPO_ROOT / "scripts" / "lanes" / "iter12_stage3.sh"
B6_ID = "B6_beta13.642_lam0.5_lr2e-05_ep5"
B6_CKPT = REPO_ROOT / "data/checkpoints/mllmu_iter12_stage2" / B6_ID
B6_ADAPTER = B6_CKPT / "adapters" / "adapter_model.safetensors"
CACHE = REPO_ROOT / "data/mllmu_hier_pilot100/mf_reference_logprobs"


def grid() -> list[s3g.Stage3CandidateSpec]:
    cal = json.loads((REPO_ROOT / s2g.CALIBRATION_REPORT).read_text())
    return s3g.stage3_grid(cal["grid_rule"]["beta_star"])


def b7_rows() -> list[s3g.Stage3CandidateSpec]:
    return s3g.b7_rows(grid())


def sidecar() -> dict:
    return json.loads((CACHE / "sidecar.json").read_text())


def fake_summary(b6_summary: dict, spec: s3g.Stage3CandidateSpec,
                 mf_dir: Path) -> dict:
    """B6's summary with only what a B7 row legitimately differs on."""
    out = json.loads(json.dumps(b6_summary))
    out["method_id"] = spec.candidate_id
    for group in out["groups"]:
        if group["name"] == "retain":
            group["weight"] = spec.effective_anchor_weight
    out["preservation_anchor"]["anchor_weight"] = {
        "retain": spec.effective_anchor_weight}
    out["init_adapter_dir"] = str(mf_dir)
    return out


@pytest.fixture
def fake_root(tmp_path) -> Path:
    """A repo-shaped tree whose four B7 rows are structurally complete.

    Adapters are symlinks to B6's real container, so the topology comparison is
    against genuine bytes; the summaries are B6's with this row's anchor weight.
    """
    root = tmp_path / "repo"
    b6_summary = json.loads((B6_CKPT / "training_summary.json").read_text())
    mf = root / "data/checkpoints/mllmu_pilot100/MF/adapters"
    mf.mkdir(parents=True)
    (root / "data/reports").mkdir(parents=True)
    shutil.copy(REPO_ROOT / s2g.CALIBRATION_REPORT,
                root / s2g.CALIBRATION_REPORT)
    cache_dir = root / "data/mllmu_hier_pilot100/mf_reference_logprobs"
    cache_dir.mkdir(parents=True)
    shutil.copy(CACHE / "sidecar.json", cache_dir / "sidecar.json")

    ref = root / f"data/checkpoints/mllmu_iter12_stage2/{B6_ID}"
    (ref / "adapters").mkdir(parents=True)
    (ref / "adapters" / "adapter_model.safetensors").symlink_to(B6_ADAPTER)
    shutil.copy(B6_CKPT / "adapters" / "adapter_config.json",
                ref / "adapters" / "adapter_config.json")
    (ref / "training_summary.json").write_text(json.dumps(b6_summary))

    for spec in b7_rows():
        row = root / f"data/checkpoints/mllmu_iter12_stage3/{spec.candidate_id}"
        (row / "adapters").mkdir(parents=True)
        (row / "adapters" / "adapter_model.safetensors").symlink_to(B6_ADAPTER)
        shutil.copy(B6_CKPT / "adapters" / "adapter_config.json",
                    row / "adapters" / "adapter_config.json")
        (row / "training_summary.json").write_text(
            json.dumps(fake_summary(b6_summary, spec, mf)))
    return root


def write_truncated(path: Path) -> None:
    """A container whose header promises more bytes than the file holds."""
    header = json.dumps({"w": {"dtype": "F32", "shape": [4, 4],
                               "data_offsets": [0, 100000]}}).encode()
    path.write_bytes(struct.pack("<Q", len(header)) + header + b"\x00" * 16)


def write_valid_small(path: Path, shape: list[int]) -> None:
    """A whole container with a topology of our choosing."""
    payload = b"\x00" * 64
    header = json.dumps({"w": {"dtype": "F32", "shape": shape,
                               "data_offsets": [0, len(payload)]}}).encode()
    path.write_bytes(struct.pack("<Q", len(header)) + header + payload)


# ──────────────────────────────────────────────────────────────────────
class TestTheContainerArithmetic:
    @pytest.mark.skipif(not B6_ADAPTER.exists(),
                        reason="B6's gitignored adapter is absent here")
    def test_a_real_adapter_is_whole(self):
        out = cis3.safetensors_layout(B6_ADAPTER)
        assert out["whole"] is True
        assert out["why_not"] is None
        assert out["num_tensors"] > 0
        assert out["size_bytes"] == B6_ADAPTER.stat().st_size
        assert out["size_bytes"] == out["expected_bytes"]

    def test_a_truncated_container_is_caught_without_torch(self, tmp_path):
        path = tmp_path / "adapter_model.safetensors"
        write_truncated(path)
        out = cis3.safetensors_layout(path)
        assert out["whole"] is False
        assert "truncated" in out["why_not"]

    def test_garbage_and_stubs_are_not_whole(self, tmp_path):
        garbage = tmp_path / "garbage.safetensors"
        garbage.write_bytes(b"not a safetensors container at all")
        out = cis3.safetensors_layout(garbage)
        assert out["whole"] is False
        assert "not a safetensors container" in out["why_not"], (
            "a foreign file's first eight bytes are an arbitrary u64; the gate "
            "must bound the declared header length by the file size rather "
            "than try to read it")
        tiny = tmp_path / "tiny.safetensors"
        tiny.write_bytes(b"\x01\x02")
        out = cis3.safetensors_layout(tiny)
        assert out["whole"] is False and "8-byte header" in out["why_not"]

    def test_an_absurd_header_length_does_not_take_the_gate_down(self, tmp_path):
        path = tmp_path / "huge.safetensors"
        path.write_bytes(struct.pack("<Q", 2**63) + b"\x00" * 32)
        out = cis3.safetensors_layout(path)
        assert out["whole"] is False
        assert "not a safetensors container" in out["why_not"]

    def test_trailing_bytes_are_also_a_failure(self, tmp_path):
        path = tmp_path / "padded.safetensors"
        write_valid_small(path, [4, 4])
        path.write_bytes(path.read_bytes() + b"\x00" * 4096)
        out = cis3.safetensors_layout(path)
        assert out["whole"] is False
        assert "trailing" in out["why_not"]

    def test_the_gate_imports_neither_torch_nor_safetensors(self):
        src = (REPO_ROOT / "scripts/check_iter12_stage3_adapters.py").read_text()
        for banned in ("import torch", "from torch", "import safetensors",
                       "from safetensors"):
            assert banned not in src, (
                f"the gate must run with no GPU and no framework, but it has "
                f"{banned!r}")


# ──────────────────────────────────────────────────────────────────────
@pytest.mark.skipif(not B6_ADAPTER.exists(),
                    reason="B6's gitignored adapter is absent here")
class TestTheGateCanSayYes:
    def test_a_complete_row_passes_every_check(self, fake_root):
        spec = b7_rows()[0]
        ref = cis3.b6_reference(fake_root, B6_ID)
        out = cis3.check_row(fake_root, spec, ref, sidecar(),
                             s3g.mf_adapters(fake_root))
        failed = [n for n, c in out["checks"].items() if not c["holds"]]
        assert failed == [], json.dumps(out["checks"], indent=1)
        assert out["holds"] is True
        assert out["adapter_sha256"] == ref["adapter_sha256"]

    def test_the_reference_is_the_incumbents_own_bytes(self, fake_root):
        ref = cis3.b6_reference(fake_root, B6_ID)
        assert ref["num_optimizer_steps"] == 230
        assert len(ref["shapes"]) > 0
        assert len(ref["adapter_sha256"]) == 64
        assert [g["name"] for g in ref["groups"]] == \
            ["fine_target", "target_level", "retain"]

    def test_the_recipe_checks_are_about_this_row_not_any_row(self, fake_root):
        spec = b7_rows()[0]
        other = b7_rows()[-1]
        row = fake_root / f"data/checkpoints/mllmu_iter12_stage3/{spec.candidate_id}"
        summary = json.loads((row / "training_summary.json").read_text())
        for group in summary["groups"]:
            if group["name"] == "retain":
                group["weight"] = other.effective_anchor_weight
        summary["preservation_anchor"]["anchor_weight"] = {
            "retain": other.effective_anchor_weight}
        (row / "training_summary.json").write_text(json.dumps(summary))
        out = cis3.check_row(fake_root, spec, cis3.b6_reference(fake_root, B6_ID),
                             sidecar(), s3g.mf_adapters(fake_root))
        assert out["holds"] is False
        assert not out["checks"][
            "retain_group_carries_the_effective_anchor_weight"]["holds"]
        assert not out["checks"]["anchored_against_the_bound_cache"]["holds"]
        assert not out["checks"]["groups_are_b6s_three_at_this_rows_anchor_weight"]["holds"]

    def test_a_truncated_adapter_fails_the_row(self, fake_root):
        spec = b7_rows()[0]
        adapter = (fake_root / "data/checkpoints/mllmu_iter12_stage3" /
                   spec.candidate_id / "adapters/adapter_model.safetensors")
        adapter.unlink()
        write_truncated(adapter)
        out = cis3.check_row(fake_root, spec, cis3.b6_reference(fake_root, B6_ID),
                             sidecar(), s3g.mf_adapters(fake_root))
        assert out["holds"] is False
        assert not out["checks"]["safetensors_container_is_whole"]["holds"]

    def test_a_different_lora_topology_is_a_different_experiment(self, fake_root):
        spec = b7_rows()[0]
        adapter = (fake_root / "data/checkpoints/mllmu_iter12_stage3" /
                   spec.candidate_id / "adapters/adapter_model.safetensors")
        adapter.unlink()
        write_valid_small(adapter, [8, 8])
        out = cis3.check_row(fake_root, spec, cis3.b6_reference(fake_root, B6_ID),
                             sidecar(), s3g.mf_adapters(fake_root))
        assert out["holds"] is False
        assert not out["checks"]["lora_topology_matches_b6"]["holds"]
        assert out["checks"]["safetensors_container_is_whole"]["holds"]

    def test_a_missing_summary_is_not_a_trained_row(self, fake_root):
        spec = b7_rows()[0]
        (fake_root / "data/checkpoints/mllmu_iter12_stage3" /
         spec.candidate_id / "training_summary.json").unlink()
        out = cis3.check_row(fake_root, spec, cis3.b6_reference(fake_root, B6_ID),
                             sidecar(), s3g.mf_adapters(fake_root))
        assert out["holds"] is False
        assert not out["checks"]["summary_file_exists"]["holds"]

    def test_a_row_that_started_somewhere_else_is_rejected(self, fake_root):
        spec = b7_rows()[0]
        row = fake_root / f"data/checkpoints/mllmu_iter12_stage3/{spec.candidate_id}"
        summary = json.loads((row / "training_summary.json").read_text())
        summary["init_adapter_dir"] = "/somewhere/else/MG/adapters"
        (row / "training_summary.json").write_text(json.dumps(summary))
        out = cis3.check_row(fake_root, spec, cis3.b6_reference(fake_root, B6_ID),
                             sidecar(), s3g.mf_adapters(fake_root))
        assert not out["checks"]["started_from_the_canonical_mf_adapter"]["holds"]

    def test_rows_sharing_the_incumbents_bytes_fail_the_cross_row_gate(
            self, fake_root):
        #: Every per-row check passes -- the adapters are B6's bytes, with B6's
        #: topology -- and the gate still refuses, because four rows that share
        #: one hash are four copies of one row and a B7 row equal to B6's adapter
        #: is a row whose extra image-anchor weight did nothing.
        out = cis3.check_all(fake_root)
        assert out["holds"] is False
        assert out["across_rows"]["all_adapter_hashes_distinct"] is False
        assert out["across_rows"]["none_equals_the_incumbent_bytes"] is False
        assert all(r["holds"] for r in out["per_row"])
        assert any("distinct" in p for p in out["problems"])
        assert any("byte-identical to B6" in p for p in out["problems"])


# ──────────────────────────────────────────────────────────────────────
class TestTheGateRefusesTheRealRepositoryRightNow:
    def test_no_adapter_means_no_generation(self):
        out = cis3.check_all(REPO_ROOT)
        assert out["holds"] is False
        assert out["rows_expected"] == out["rows_checked"] == 4
        ids = {r["candidate_id"] for r in out["per_row"]}
        per_row = {p.split(":")[0] for p in out["problems"] if
                   p.split(":")[0] in ids}
        assert per_row == ids, "every row must be named as incomplete"
        assert any("adapter_file_exists" in p for p in out["problems"])
        assert not out["across_rows"]["all_adapter_hashes_distinct"]

    def test_the_entry_point_exits_non_zero(self):
        p = subprocess.run(
            (PY, "scripts/check_iter12_stage3_adapters.py"),
            cwd=REPO_ROOT, capture_output=True, text=True, check=False)
        assert p.returncode == 1
        assert "REFUSING" in p.stdout

    def test_it_is_not_a_protocol_path_and_needed_no_refreeze(self):
        rel = "scripts/check_iter12_stage3_adapters.py"
        assert rel not in fis3.PROTOCOL_PATHS
        assert rel not in json.loads(
            (REPO_ROOT / s3g.FREEZE_REPORT).read_text())["hashes"][
                "protocol_paths"]
        assert "not a protocol path" in cis3.check_all(REPO_ROOT)[
            "what_this_gate_is_not"]

    def test_it_reads_no_confirmation_evidence(self):
        src = (REPO_ROOT / "scripts/check_iter12_stage3_adapters.py").read_text()
        assert "confirm100" not in src
        assert "mllmu_hier_confirm" not in src


# ──────────────────────────────────────────────────────────────────────
class TestTheChainLaneDiscipline:
    def script(self) -> str:
        return CHAIN.read_text()

    def test_the_script_is_valid_bash(self):
        p = subprocess.run(("bash", "-n", str(CHAIN)), capture_output=True,
                           text=True, check=False)
        assert p.returncode == 0, p.stderr

    def test_no_phase_shares_one_log_between_lanes(self):
        src = self.script()
        assert "TRAIN_LOG=" not in src and "GEN_LOG=" not in src, (
            "a single file every lane appends to interleaves four tracebacks "
            "into something attributable to none of them")
        assert 'TRAIN_LOGDIR="$LOGDIR/iter12_stage3_train"' in src
        assert 'GEN_LOGDIR="$LOGDIR/iter12_stage3_gen"' in src
        assert 'log="$TRAIN_LOGDIR/train_lane$n.log"' in src
        assert 'log="$GEN_LOGDIR/gen_${cid}.log"' in src
        assert 'wait_for_gpu.sh "$TRAIN_MIN_FREE" "$log"' in src
        assert 'wait_for_gpu.sh "$GEN_MIN_FREE" "$log"' in src

    def test_each_lane_records_the_device_it_actually_ran_on(self):
        src = self.script()
        assert "LANE_PROLOGUE=" in src
        assert "CUDA_VISIBLE_DEVICES=" in src
        assert 'exec "$0" "$@"' in src, (
            "the wrapper must exec, so the lane's pid stays the trainer's pid "
            "and SIGTERM reaches python rather than a shell that already left")
        assert 'bash -c "$LANE_PROLOGUE"' in src
        assert src.count('bash -c "$LANE_PROLOGUE"') == 2, (
            "both GPU phases wrap their lanes")
        assert "CLAIMED GPU" in src, (
            "the chain journal must also carry the waiter's physical index")

    def test_every_lane_exit_status_is_captured_and_acted_on(self):
        src = self.script()
        assert 'for p in "${pids[@]}"; do wait "$p"; done' not in src, (
            "waiting without reading $? discards every lane's status")
        for phase in ("lane_pids", "gen_pids"):
            assert f'wait "${{{phase}[$i]}}"; rc=$?' in src
            #: The empty-array guard, spelled without an f-string because
            #: ``${!`` is not one: bash < 4.4 rejects "${arr[@]}" under set -u
            #: when the array is empty, and a phase with no lanes must not crash.
            guard = '${' + phase + '[@]+"${!' + phase + '[@]}"}'
            assert guard in src
        assert "train_failed=$((train_failed + 1))" in src
        assert "gen_failed=$((gen_failed + 1))" in src
        assert 'if [ "$train_failed" -ne 0 ]; then' in src
        assert 'if [ "$gen_failed" -ne 0 ]; then' in src
        assert "report_lane" in src

    def test_generation_is_ordered_after_the_completeness_gate(self):
        #: Ordered on the EXECUTABLE text only: the header comment block names
        #: every phase, so an index into the raw file measures the prose.
        code = "\n".join(line for line in self.script().splitlines()
                         if not line.lstrip().startswith("#"))
        gate = code.index("check_iter12_stage3_adapters.py --json-out")
        first_gen = code.index("--generate-only")
        assert gate < first_gen, (
            "a generation lane must not be launchable before the adapter "
            "completeness gate has run")
        src = self.script()
        assert src.index("# ── train ─") < src.index("# ── check ─") < \
            src.index("# ── gen ─") < src.index("# ── verify ─")
        between = code[gate:first_gen]
        assert "exit 4" in between, (
            "the gate must stop the chain rather than warn it")
        assert 'adapter_model.safetensors" ]' not in src, (
            "the old existence-only check must be gone: existence is not "
            "completeness")

    def test_the_chain_never_amends_the_freeze(self):
        src = self.script()
        assert "--refreeze" not in src
        for call in ("freeze_iter12_stage3.py", "freeze_iter12_stage2.py",
                     "freeze_iter12_route_stratification.py",
                     "freeze_iter12_selection_protocol.py"):
            assert f"{call} --check-only" in src
        assert "build_iter12_stage3_basis.py --check-only" in src

    def test_the_chain_stops_when_a_selection_is_already_filed(self):
        src = self.script()
        assert 'if [ -f "$SELECTION" ]; then' in src
        assert src.index('if [ -f "$SELECTION" ]; then') < src.index("# ── pre ─")

    def test_a_dry_run_of_the_preconditions_passes_without_a_gpu(self):
        env = {"PRE_ONLY": "1", "PATH": "/usr/bin:/bin",
               "HOME": str(Path.home())}
        p = subprocess.run(("bash", str(CHAIN)), cwd=REPO_ROOT, env=env,
                           capture_output=True, text=True, check=False,
                           timeout=900)
        assert p.returncode == 0, p.stdout[-4000:] + p.stderr[-2000:]
        assert "PRE_ONLY=1" in p.stdout
        assert "four freezes and the basis match" in p.stdout
        assert "still amendable" in p.stdout
        assert not (REPO_ROOT / "data/checkpoints/mllmu_iter12_stage3").exists()


# ──────────────────────────────────────────────────────────────────────
WRAPPER = REPO_ROOT / "scripts" / "lanes" / "train_attempt.sh"


class TestAPerAttemptScaffoldingCleanup:
    """An OOM during the model load must cost one attempt, not one row.

    The trainer creates ``<row>/adapters/`` before it loads the model and skips
    any row whose directory exists, so an attempt that dies during the load —
    which is what a co-tenant landing ~29 GiB on a freshly claimed card causes
    — leaves an empty directory that silently removes that row from every later
    attempt.  Measured over one night: 8 claims, 6 OOMs, all four rows
    poisoned, both lanes exiting 0 having trained nothing, and the chain
    stopped by the completeness gate with zero weight files.

    Every case here runs the real wrapper with ``/bin/echo`` as the
    interpreter, so the exec is proven reached without needing torch or a GPU,
    and against a ``tmp_path`` checkpoint root, so nothing is ever created
    under the real Stage-3 root (which would permanently lock the freeze).
    """

    def run(self, root: Path,
            ids: str = "rowA,rowB") -> subprocess.CompletedProcess:
        return subprocess.run(
            ("bash", str(WRAPPER), "/bin/echo", str(root), ids, "cuda:0"),
            cwd=REPO_ROOT, capture_output=True, text=True, check=False)

    def scaffolding(self, root: Path, row: str) -> Path:
        d = root / row / "adapters"
        d.mkdir(parents=True)
        return d

    def test_an_empty_adapters_dir_is_removed(self, tmp_path):
        self.scaffolding(tmp_path, "rowA")
        assert self.run(tmp_path).returncode == 0
        assert not (tmp_path / "rowA" / "adapters").exists()

    def test_the_empty_row_dir_goes_with_it(self, tmp_path):
        self.scaffolding(tmp_path, "rowA")
        self.run(tmp_path)
        assert not (tmp_path / "rowA").exists(), (
            "an empty row directory left behind is what arms the freeze's "
            "adapters_exist() glob on a run that produced no adapter")

    def test_a_populated_adapters_dir_is_never_removed(self, tmp_path):
        d = self.scaffolding(tmp_path, "rowA")
        (d / "adapter_model.safetensors").write_bytes(b"\x00" * 8)
        out = self.run(tmp_path)
        assert (d / "adapter_model.safetensors").exists()
        assert (tmp_path / "rowA" / "adapters").exists()
        assert "keeping" in out.stdout, (
            "the decision to preserve a trained row is evidence, so it is "
            "printed rather than left to be inferred from a surviving file")

    def test_a_dir_holding_only_empty_subdirs_still_counts_as_untrained(
            self, tmp_path):
        d = self.scaffolding(tmp_path, "rowA")
        (d / "nested").mkdir()
        self.run(tmp_path)
        assert not d.exists(), (
            "a partially created tree with no file in it is scaffolding too")

    def test_a_mixed_lane_keeps_the_trained_row_and_retries_the_lost_one(
            self, tmp_path):
        done = self.scaffolding(tmp_path, "rowA")
        (done / "adapter_model.safetensors").write_bytes(b"\x00" * 8)
        self.scaffolding(tmp_path, "rowB")
        self.run(tmp_path)
        assert (done / "adapter_model.safetensors").exists(), (
            "resumption depends on a completed row staying skipped")
        assert not (tmp_path / "rowB" / "adapters").exists()

    def test_a_row_with_no_directory_is_left_alone(self, tmp_path):
        out = self.run(tmp_path, ids="rowC")
        assert out.returncode == 0
        assert not (tmp_path / "rowC").exists()
        assert "cleared" not in out.stdout

    def test_it_still_execs_the_trainer_with_the_lanes_argv(self, tmp_path):
        out = self.run(tmp_path, ids="rowA,rowB")
        assert "scripts/train_iter12_stage3.py" in out.stdout
        assert "--candidates rowA,rowB" in out.stdout
        assert "--device cuda:0" in out.stdout

    def test_the_cleanup_is_ordered_before_the_exec(self):
        src = WRAPPER.read_text()
        assert src.index("-type d -empty -delete") < src.index('exec "$PY"'), (
            "cleaning after the trainer starts would race the directory the "
            "trainer is about to create")

    def test_the_wrapper_is_valid_bash(self):
        p = subprocess.run(("bash", "-n", str(WRAPPER)),
                           capture_output=True, text=True, check=False)
        assert p.returncode == 0, p.stderr

    def test_it_refuses_to_run_without_its_arguments(self):
        p = subprocess.run(("bash", str(WRAPPER)), cwd=REPO_ROOT,
                           capture_output=True, text=True, check=False)
        assert p.returncode != 0
        assert "usage" in p.stdout + p.stderr

    def test_the_chain_runs_it_once_per_claim_not_once_per_launch(self):
        src = CHAIN.read_text()
        train = src[src.index("# ── train ─"):src.index("# ── check ─")]
        wait = train.index('wait_for_gpu.sh "$TRAIN_MIN_FREE" "$log"')
        wrap = train.index("bash scripts/lanes/train_attempt.sh")
        end = train.index("< /dev/null > /dev/null 2>&1 &")
        assert wait < wrap < end, (
            "the cleanup must sit inside the command the WAITER runs, so it "
            "happens on every re-queued attempt; done at launch it would run "
            "once, before the OOM that creates the scaffolding")
        assert '"$PY" "$STAGE3_CKPT" "$ids" "$DEVICE"' in train
        assert '--candidates "$ids"' not in train, (
            "the train lane must reach the trainer only through the wrapper; "
            "--plan-lanes is the one direct call that remains, and it writes "
            "no adapter")

    def test_the_wrapper_needed_no_refreeze(self):
        rel = "scripts/lanes/train_attempt.sh"
        assert rel not in fis3.PROTOCOL_PATHS
        assert rel not in json.loads(
            (REPO_ROOT / s3g.FREEZE_REPORT).read_text())["hashes"][
                "protocol_paths"]
        assert str(CHAIN.relative_to(REPO_ROOT)) not in fis3.PROTOCOL_PATHS
