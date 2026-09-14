"""Read the Stage-2 faithfulness control's result into a committed report.

Two things make this script necessary rather than redundant.

The control's ``CONTROL.json`` lives under ``data/checkpoints/``, which is
gitignored, so the artifact that licences the entire Stage-2 comparison -- the
one thing that establishes a difference between Stage 1 and Stage 2 is a
difference between objectives and not between two copies of a training loop --
is not in a clone.  Its numbers are copied here so they can be checked.

And ``CONTROL.json`` contains a sentence that is WRONG.  Its ``detail`` field
says the two loops "produced bitwise-identical adapters under a pinned hash
seed", and the control's log line says they "agree bitwise".  Neither is true:
``gate.gap_frozen_vs_new.bitwise_identical`` is ``false`` in the same document,
and the gate passed on its magnitude branch instead.  The sentence was written
for the zero-floor branch, which is the one the CPU stub tests exercise; on a
GPU the floor is not zero, so a different branch decided and the prose did not
follow it.  ``scripts/train_iter12_stage2.py`` is hash-bound by the Stage-2
freeze, and the freeze locked the moment the control wrote its first adapter,
so the sentence cannot be corrected without breaking ``--check-only`` -- which
the selector requires in order to run at all.  The correction is filed here
instead, beside the numbers, naming the fields that are authoritative.

Reads only artifacts that already exist and writes only its own report.  It
moves no criterion: the gate's outcome, its inputs and the freeze are all read,
never recomputed or reinterpreted.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from granunlearn.training import stage2_grid as sg

OUT_REPORT = "data/reports/mllmu_iter12_stage2_control_reading.json"
#: The superseded run that established the retired criterion was unsatisfiable.
SUPERSEDED = ("outputs/superseded/iter12_stage2_control_v1/frozen_loop_rerun/"
              "RERUN_COMPARISON.json")


def control_root(repo_root: Path) -> Path:
    """The control's directory, DISCOVERED rather than retyped.

    Its id is a constant in ``train_iter12_stage2.py``, which imports torch at
    module level, so this script cannot import it -- and retyping the string
    would let the two drift.  ``control_dir`` is expected to hold exactly one
    control; anything else is refused rather than guessed at, because picking
    one of several would silently report a run that is not the current one.
    """
    root = sg.control_dir(repo_root)
    if not root.exists():
        raise SystemExit(f"REFUSING: {root} does not exist, so the control has "
                         f"not been run")
    found = sorted(p for p in root.iterdir() if p.is_dir())
    if len(found) != 1:
        raise SystemExit(
            f"REFUSING: {root} holds {len(found)} control directories "
            f"({[p.name for p in found]}); this script reports exactly one and "
            f"will not choose between them")
    marker = found[0] / "CONTROL.json"
    if not marker.exists():
        raise SystemExit(f"REFUSING: {marker} does not exist, so the control "
                         f"did not finish")
    return found[0]


def target_modules_order(adapters: Path) -> list:
    """The order PEFT serialised ``target_modules`` in, read not assumed.

    This is the quantity the retired criterion turned on, so the report records
    it for all three runs rather than describing it.
    """
    cfg = adapters / "adapter_config.json"
    if not cfg.exists():
        return []
    return json.loads(cfg.read_text())["target_modules"]


def _steps_by_run(root: Path, control: dict) -> dict:
    """How far each of the three runs trained, read from its own summary."""
    out = {}
    for label in control["runs"]:
        summary = root / label / "training_summary.json"
        out[label] = json.loads(summary.read_text())["num_optimizer_steps"] \
            if summary.exists() else None
    return out


def build(repo_root: Path) -> dict:
    root = control_root(repo_root)
    control = json.loads((root / "CONTROL.json").read_text())
    gate = control["gate"]
    floor = control["noise_floor_frozen_loop_vs_itself"]
    across = gate["gap_frozen_vs_new"]
    filed = control["filed_stage1_adapter_reported_not_gated"]

    orders = {label: target_modules_order(root / label / "adapters")
              for label in control["runs"]}
    distinct = {tuple(v) for v in orders.values() if v}

    #: The superseded cross-process measurement, for the comparison that says
    #: what pinning the seed actually bought.  Absent from a clone is possible
    #: in principle, so it is reported rather than required.
    sup_path = repo_root / SUPERSEDED
    superseded = json.loads(sup_path.read_text()) if sup_path.exists() else None

    def ratio(a: dict, b: dict) -> float | None:
        if not a.get("max_abs_gap") or not b.get("max_abs_gap"):
            return None
        return round(a["max_abs_gap"] / b["max_abs_gap"], 4)

    return {
        "what_this_is": (
            "The Stage-2 faithfulness control's result, copied out of a "
            "gitignored checkpoint directory into a committed report, plus a "
            "correction to one sentence in it that is wrong. Written by "
            "scripts/annotate_iter12_stage2_control.py, which reads the "
            "control's own CONTROL.json and recomputes nothing."),
        "read_from": str((root / "CONTROL.json").relative_to(repo_root)),
        "why_it_is_not_left_there": (
            "data/checkpoints/ is gitignored, so the artifact that licences the "
            "whole Stage-2 comparison would exist on one machine only. A clone "
            "could not check the claim that the two training loops agree."),
        "control": control["control"],
        "gate_passed": control["gate_passed"],
        "python_hash_seed": control["python_hash_seed"],
        "epochs": control["epochs"],
        "runs": control["runs"],
        #: Copied for the same reason as the gaps: the checkpoint directory is
        #: gitignored, so a claim about how far each run trained would otherwise
        #: be checkable on one machine only.  Equal step counts are part of what
        #: makes the three comparable -- two runs of different length would
        #: differ for a reason that has nothing to do with either loop.
        "num_optimizer_steps_by_run": _steps_by_run(root, control),
        "the_gate": {
            "required": gate["required"],
            "noise_floor_frozen_loop_vs_itself": floor,
            "gap_frozen_vs_new": across,
            "across_is_within_the_floor":
                across["max_abs_gap"] <= floor["max_abs_gap"],
            "across_to_floor_ratio": ratio(across, floor),
            #: Built from the comparison rather than asserted beside it.  The
            #: first version of this report said the gap was narrower "on both
            #: the worst tensor and the mean" unconditionally, which is a claim
            #: the script never checked -- true of this run, and silently false
            #: of any re-run where the mean went the other way.
            "read_this_way": (
                "the gap between the two loops is narrower than the gap between "
                "two runs of ONE of them "
                + ("on both the worst tensor and the mean over all "
                   f"{across['num_tensors']}"
                   if across["mean_abs_gap"] <= floor["mean_abs_gap"] else
                   "on the worst tensor, though not on the mean over all "
                   f"{across['num_tensors']}")
                + ", so the two loops are not distinguishable at the resolution "
                  "this stack has"),
        },
        "filed_stage1_adapter_reported_not_gated": filed,
        "a_wording_defect_in_the_control_report": {
            "the_sentence": control["detail"],
            "and_the_log_line": (
                "CONTROL PASSED: the two loops agree bitwise (noise floor max "
                f"{floor['max_abs_gap']})"),
            "what_is_wrong_with_it": (
                "It says the adapters were bitwise-identical. They were not: "
                "gate.gap_frozen_vs_new.bitwise_identical is false in the same "
                "document, and the gate decided on its magnitude branch."),
            "the_fields_that_are_authoritative": [
                "gate.required",
                "gate.gap_frozen_vs_new.bitwise_identical",
                "gate.gap_frozen_vs_new.max_abs_gap",
                "noise_floor_frozen_loop_vs_itself",
            ],
            "how_it_happened": (
                "The sentence is written for the branch that fires when the "
                "noise floor is exactly zero, which is the branch the CPU stub "
                "tests exercise and the branch a pinned hash seed in one "
                "process was expected to buy. On a GPU the floor is not zero "
                f"(max {floor['max_abs_gap']:.3e}), so the magnitude branch "
                "decided and the prose did not follow it. It is a defect in a "
                "summary string, not in the gate: the comparison, the branch "
                "selection and the recorded numbers are all correct."),
            "why_it_is_not_corrected_in_place": (
                "scripts/train_iter12_stage2.py is one of the eight protocol "
                "paths the Stage-2 freeze binds by sha256, and the freeze "
                "refuses amendment once any adapter exists -- which the control "
                "itself caused, by design. Editing the sentence would fail "
                "--check-only, and select_iter12_stage2.py refuses to run when "
                "the freeze does not match, so the study would be unable to "
                "score. The correction is filed here instead, which is what the "
                "freeze's own disclosure section does for the naming defect it "
                "cannot fix either."),
        },
        "the_hash_seed_pinning_did_what_it_claimed": {
            "target_modules_order_by_run": orders,
            "all_runs_share_one_order": len(distinct) == 1,
            #: Expected to be FALSE, and its being false is the finding rather
            #: than a problem: the filed adapter was written under a randomised
            #: seed, so a pinned seed reproduces A ORDER, not that one.
            "the_shared_order_is_also_the_filed_order": (
                bool(distinct) and len(distinct) == 1
                and next(iter(distinct)) == tuple(
                    target_modules_order(
                        sg.stage1_ckpt_root(repo_root)
                        / control["reproduces_the_objective_of"] / "adapters"))),
            "why_this_matters": (
                "PeftConfig.target_modules is a set, so its serialised order -- "
                "and the LoRA injection order behind it, which decides which "
                "module draws which dropout mask -- depends on PYTHONHASHSEED. "
                "One shared order across all three runs is the pinning working, "
                "and it is what makes the three comparable at all. That the "
                "shared order is NOT the filed one is the reason the retired "
                "criterion could not be satisfied by any run: pinning a seed "
                "reproduces an order, and only the seed that wrote the filed "
                "adapter would have reproduced that one."),
        },
        "what_the_noise_is_made_of": {
            "in_process_pinned_seed_floor": floor["max_abs_gap"],
            "across_the_two_loops": across["max_abs_gap"],
            "against_the_filed_adapter": filed["gap"].get("max_abs_gap"),
            "filed_to_floor_ratio": ratio(filed["gap"], floor),
            "cross_process_unpinned_from_the_superseded_run": (
                superseded["weight_gap_rerun_vs_filed"]["max_abs_gap"]
                if superseded else None),
            #: Every magnitude in this sentence is computed from the numbers
            #: beside it, so the prose cannot drift away from them -- and so a
            #: clone without the superseded run says so instead of quoting a
            #: ratio it does not have.
            "read_this_way": (
                (f"Pinning the seed and staying in one process cut the run-to-run "
                 f"gap {round(superseded['weight_gap_rerun_vs_filed']['max_abs_gap'] / floor['max_abs_gap'], 1)}-fold "
                 f"against the superseded cross-process measurement "
                 f"({superseded['weight_gap_rerun_vs_filed']['max_abs_gap']:.3e} "
                 f"down to {floor['max_abs_gap']:.3e}), and "
                 if superseded else
                 "The superseded cross-process measurement is not in this clone, "
                 "so no ratio against it is quoted here; ")
                + f"what is left is GPU floating-point nondeterminism. The filed "
                  f"adapter sits {ratio(filed['gap'], floor)} times further away "
                  f"again ({filed['gap']['max_abs_gap']:.3e}), and that distance "
                  f"is dominated by its different target_modules order rather "
                  f"than by anything the two loops compute differently."),
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo-root", default=None)
    ap.add_argument("--check-only", action="store_true",
                    help="Verify the committed reading instead of writing it")
    args = ap.parse_args()
    repo_root = Path(args.repo_root).resolve() if args.repo_root \
        else Path(__file__).resolve().parents[1]
    out = repo_root / OUT_REPORT

    #: Refused BEFORE the control is read, not after.  The refusal is about the
    #: filed report and should not depend on the checkpoint directory still
    #: being there -- which matters because data/checkpoints/ is gitignored, so
    #: in a clone the control is absent while the report is present, and the
    #: opposite order would report a missing control instead of the overwrite it
    #: was asked to make.
    if out.exists() and not args.check_only:
        raise SystemExit(
            f"REFUSING to overwrite {out}. It already exists; delete it "
            f"deliberately first, so a re-run cannot silently replace a filed "
            f"reading of the control.")

    document = build(repo_root)

    if args.check_only:
        if not out.exists():
            raise SystemExit(f"REFUSING: {out} does not exist")
        filed = json.loads(out.read_text())
        diffs = [k for k in set(filed) | set(document) if filed.get(k) != document.get(k)]
        if diffs:
            for d in sorted(diffs):
                print(f"  MISMATCH {d}")
            raise SystemExit(1)
        print(f"OK — {out.name} still matches the control's own CONTROL.json")
        return

    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(document, f, indent=2, ensure_ascii=False)
    print(f"control reading -> {out}")
    print(f"  gate_passed: {document['gate_passed']} "
          f"({document['the_gate']['required']})")
    print(f"  noise floor max {document['the_gate']['noise_floor_frozen_loop_vs_itself']['max_abs_gap']:.3e}"
          f" | across loops max {document['the_gate']['gap_frozen_vs_new']['max_abs_gap']:.3e}"
          f" | ratio {document['the_gate']['across_to_floor_ratio']}")
    print(f"  all three runs share one target_modules order: "
          f"{document['the_hash_seed_pinning_did_what_it_claimed']['all_runs_share_one_order']}")
    print("  DISCLOSED: the control's own detail string claims bitwise "
          "identity; the structured fields say otherwise and are authoritative")


if __name__ == "__main__":
    main()
