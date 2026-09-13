"""Complete the Stage-1b report after the frozen analyzer crashed.

WHAT HAPPENED
-------------
``scripts/analyze_iter12_seed_replication.py`` line 325 stores
``rs.distance_to_mg(...)`` as-is, but that function returns
``tuple[float | None, list[str]]`` — its own caller inside
``retention_selection.py`` unpacks it into ``dist, used``.  Line 386 collects
those tuples into ``dists`` and line 394 hands them to ``aggregate``, whose
first act is ``float(v)``, so the scoring run died with ``TypeError: float()
argument must be ... not 'tuple'`` — after all eight replicates had been scored
and before the report was assembled.

WHY THIS IS A SEPARATE FILE RATHER THAN A FIX
---------------------------------------------
Both files on that path are ``PROTOCOL_PATHS`` in
``data/reports/mllmu_iter12_seed_replication_freeze.json``, and that freeze
refuses amendment now that six replicates have adapters on disk.  That refusal
is correct and is the reason the freeze exists: a protocol edited after its
replicates exist could be fitted to them.  So the frozen bytes stay frozen, and
this script runs them unchanged and records what it had to work around.

WHAT IT CHANGES
---------------
One call.  ``rs.distance_to_mg`` is patched for the duration of the run to
return the distance alone — what line 325 plainly intended and what lines 386
and 394 require.  Three things are proven here rather than asserted in prose,
because a patch to a sealed module is exactly the kind of change that should
not be taken on trust:

1. the four frozen protocol paths still hash to what the freeze records, so
   what ran is what was preregistered;
2. no function on the decision path reaches ``distance_to_mg``, by an AST walk
   recomputed at run time rather than quoted from a comment;
3. every call the patch actually served came from the analyzer's line 325, by
   caller frame — so nothing else in the process consumed the patched
   behaviour.

``distance_to_mg`` is not decision-bearing in Stage 1b in any case: the frozen
report says so itself ("the floor decides eligibility, exactly as in Stage 1"),
and the freeze records that the parents were not selected on D_G.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "src"))

import analyze_iter12_seed_replication as aisr
import freeze_iter12_seed_replication as fisr

from granunlearn.evaluation import retention_selection as rs

#: Beside the report rather than inside it.  The report is what the frozen
#: analyzer writes, and this script cannot add a field to it without editing a
#: frozen file — so the disclosure lives in its own artifact and the report
#: stays exactly what the preregistered code produces.
REPAIR_RECORD = "data/reports/mllmu_iter12_seed_replication.REPAIR.json"

#: Where the analyzer calls the patched function.  Derived from its source
#: rather than written down, so this cannot go stale when the frozen file is
#: read at a different offset, and so "exactly one call site" is itself a
#: checked claim rather than a comment.
ANALYZER_REL = "scripts/analyze_iter12_seed_replication.py"

#: Functions whose transitive closure must not reach ``distance_to_mg``.  These
#: produce the two decision-bearing fields (``floor_on_the_mean``,
#: ``passes_on_the_mean``) and the numbers fed into them.
DECISION_PATH = {
    "src/granunlearn/evaluation/retention_selection.py":
        ("floor_check", "probe_retention", "trainval_vector"),
    "src/granunlearn/training/seed_replication.py":
        ("mean_candidate", "aggregate", "range_classification"),
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def frozen_paths_match_the_freeze() -> dict[str, Any]:
    """The four protocol paths, hashed now against the freeze's own record.

    Re-derived from the committed freeze rather than from a constant here, so
    this cannot pass by agreeing with itself.
    """
    freeze = json.loads((REPO_ROOT / fisr.OUT_REPORT).read_text())
    recorded = freeze["hashes"]["protocol_paths"]
    per_path = {}
    for rel, want in sorted(recorded.items()):
        got = _sha256(REPO_ROOT / rel)
        per_path[rel] = {"freeze_records": want, "now": got,
                         "matches": got == want}
    drifted = sorted(p for p, v in per_path.items() if not v["matches"])
    if drifted:
        raise SystemExit(
            f"REFUSING to repair: the frozen protocol paths have drifted "
            f"({drifted}). Whatever ran would no longer be what was "
            f"preregistered, and a repair on top of that proves nothing.")
    return {"num_protocol_paths": len(per_path), "per_path": per_path,
            "all_match": True}


def _module_functions(rel: str) -> dict[str, ast.FunctionDef]:
    tree = ast.parse((REPO_ROOT / rel).read_text())
    return {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}


def _calls(node: ast.AST, known: set[str]) -> set[str]:
    out = set()
    for x in ast.walk(node):
        if isinstance(x, ast.Call):
            name = x.func.attr if isinstance(x.func, ast.Attribute) \
                else x.func.id if isinstance(x.func, ast.Name) else None
            if name in known:
                out.add(name)
    return out


def decision_path_cannot_reach_the_patched_function() -> dict[str, Any]:
    """AST walk, recomputed here rather than quoted from the docstring.

    The patch is on ``rs.distance_to_mg``.  If any function that produces a
    decision-bearing field reached it, the patch could move the verdict, and
    this script would be changing the result it claims merely to report.
    """
    per_module = {}
    reaches = []
    for rel, roots in DECISION_PATH.items():
        fns = _module_functions(rel)
        known = set(fns)
        closure: dict[str, list[str]] = {}
        for root in roots:
            if root not in fns:
                raise SystemExit(
                    f"REFUSING: {rel} has no function {root}, so the decision "
                    f"path this script claims to have checked is not the one "
                    f"that exists")
            seen, stack = set(), [root]
            while stack:
                for callee in _calls(fns[stack.pop()], known):
                    if callee not in seen:
                        seen.add(callee)
                        stack.append(callee)
            closure[root] = sorted(seen - {root})
            if "distance_to_mg" in seen:
                reaches.append(f"{rel}:{root}")
        per_module[rel] = closure
    if reaches:
        raise SystemExit(
            f"REFUSING to repair: {reaches} reach distance_to_mg, so patching "
            f"it could change the verdict rather than only a reported number")
    return {"walked": per_module,
            "functions_reaching_distance_to_mg": reaches,
            "conclusion": ("none of the decision-bearing functions can see the "
                           "patched call, so the patch cannot move the verdict")}


def no_other_module_calls_it() -> dict[str, Any]:
    """The patch is module-global, so any other caller would be affected too.

    ``retention_selection`` calls its own ``distance_to_mg`` from the Stage-1
    selection code, which this study never runs; checked rather than assumed,
    because an unnoticed second caller is precisely how a 'surgical' patch
    stops being surgical.
    """
    out = subprocess.run(
        ("git", "grep", "-n", "distance_to_mg", "--", "*.py"),
        cwd=REPO_ROOT, capture_output=True, text=True, check=False)
    callers = {}
    for line in out.stdout.splitlines():
        rel, _, rest = line.partition(":")
        if not rel.endswith(".py"):
            continue
        callers.setdefault(rel, []).append(rest.strip()[:100])
    return {"files_mentioning_it": sorted(callers),
            "per_file": callers}


def analyzer_call_sites() -> set[int]:
    """Lines in the frozen analyzer that call ``rs.distance_to_mg``.

    Read out of its AST, so the set is whatever the frozen file actually
    contains.  More than one would mean the patch is not confined to the buggy
    call, which is the whole basis for calling it surgical.
    """
    tree = ast.parse((REPO_ROOT / ANALYZER_REL).read_text())
    sites = {n.lineno for n in ast.walk(tree)
             if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
             and n.func.attr == "distance_to_mg"
             and isinstance(n.func.value, ast.Name)
             and n.func.value.id == "rs"}
    if len(sites) != 1:
        raise SystemExit(
            f"REFUSING to repair: the analyzer calls rs.distance_to_mg at "
            f"{sorted(sites)}, and patching the module would change all of "
            f"them, not just the one that stores a tuple")
    return sites


def unwrap_distance_to_mg(log: list[dict[str, Any]]) -> Any:
    """Return the distance alone, and record who asked.

    The caller frame is captured because "only the analyzer's line 325 saw
    this" is the claim that makes the patch surgical, and a claim about which
    code ran is worth more when it is measured during the run than when it is
    argued beforehand.
    """
    original = rs.distance_to_mg

    def patched(*args: Any, **kwargs: Any) -> Any:
        frame = sys._getframe(1)
        log.append({"caller_file": Path(frame.f_code.co_filename).name,
                    "caller_line": frame.f_lineno})
        dist, used = original(*args, **kwargs)
        if dist is None:
            raise SystemExit(
                f"REFUSING: distance_to_mg returned None (unused components "
                f"{used}), and aggregate would turn that into a crash rather "
                f"than a report")
        return dist

    return patched


def prove_the_crash_is_real(device: str) -> dict[str, Any]:
    """Run the frozen analyzer unpatched and require the TypeError.

    Cheap — every prediction is reused from disk, so no model loads — and worth
    it: a repair whose bug cannot be reproduced is a repair nobody can check,
    and this also shows the patch below is load-bearing rather than cosmetic.
    """
    out = REPO_ROOT / aisr.OUT_REPORT
    #: Hashed either side rather than checked for existence afterwards.  Once
    #: this script has succeeded once, the report exists — and "does the file
    #: exist" would then blame the crashed run for a report the PATCHED run
    #: wrote, which is the one thing this evidence has to rule out.
    before = _sha256(out) if out.exists() else None
    code = (
        "import sys;sys.path.insert(0,'scripts');sys.path.insert(0,'src');"
        "import analyze_iter12_seed_replication as a;"
        f"sys.argv=['repair','--device',{device!r}];a.main()")
    p = subprocess.run((sys.executable, "-c", code), cwd=REPO_ROOT,
                       capture_output=True, text=True, check=False)
    after = _sha256(out) if out.exists() else None
    tail = p.stderr.strip().splitlines()[-1:] or [""]
    reproduced = p.returncode != 0 and "not 'tuple'" in p.stderr
    if not reproduced:
        raise SystemExit(
            f"REFUSING to repair: the crash did not reproduce (rc="
            f"{p.returncode}, last line {tail[0]!r}). Either the frozen code "
            f"changed or this script is wrong about what it is fixing.")
    return {"command_returncode": p.returncode,
            "exception": tail[0],
            "reproduced": True,
            "crash_line": next(
                (ln.strip() for ln in reversed(p.stderr.splitlines())
                 if "analyze_iter12_seed_replication.py" in ln
                 and "line" in ln), None),
            "report_sha256_before_the_crashed_run": before,
            "report_sha256_after_the_crashed_run": after,
            "the_crashed_run_wrote_or_changed_the_report": before != after}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--device", default="cuda:0",
                    help="must match the frozen invocation; every prediction "
                         "is reused from disk so no model is loaded")
    ap.add_argument("--skip-crash-proof", action="store_true",
                    help="do not first reproduce the crash unpatched")
    args = ap.parse_args()

    print("repair: verifying the frozen protocol paths are untouched")
    frozen = frozen_paths_match_the_freeze()
    print(f"  {frozen['num_protocol_paths']} protocol paths, all matching "
          f"the freeze")

    print("repair: proving the patch cannot reach the decision")
    ast_proof = decision_path_cannot_reach_the_patched_function()
    print(f"  {ast_proof['conclusion']}")
    callers = no_other_module_calls_it()
    print(f"  files mentioning distance_to_mg: "
          f"{callers['files_mentioning_it']}")

    crash = None
    if not args.skip_crash_proof:
        print("repair: reproducing the crash unpatched")
        crash = prove_the_crash_is_real(args.device)
        print(f"  {crash['exception']}")
        print(f"  the crashed run wrote or changed the report: "
              f"{crash['the_crashed_run_wrote_or_changed_the_report']}")

    print("repair: running the frozen analyzer with the one call unwrapped")
    #: Derived before the run, so a second call site stops the repair before it
    #: writes anything rather than after.
    sites = analyzer_call_sites()
    site = next(iter(sites))
    print(f"  the analyzer calls rs.distance_to_mg at exactly one line: {site}")
    call_log: list[dict[str, Any]] = []
    original = rs.distance_to_mg
    rs.distance_to_mg = unwrap_distance_to_mg(call_log)
    try:
        sys.argv = ["repair", "--device", args.device]
        aisr.main()
    finally:
        rs.distance_to_mg = original

    #: Every patched call must have come from that one site.  Eight replicates,
    #: one call each — measured during the run rather than inferred from the
    #: AST, because the AST shows what the file contains and this shows what
    #: actually executed.
    unexpected = [c for c in call_log
                  if (c["caller_file"], c["caller_line"])
                  != (Path(ANALYZER_REL).name, site)]
    if unexpected:
        raise SystemExit(
            f"REFUSING to keep this report: the patch served calls from "
            f"{unexpected}, not only {ANALYZER_REL}:{site}, so it was not "
            f"surgical and the report cannot be attributed to the frozen code")
    print(f"  {len(call_log)} patched call(s), all from "
          f"{ANALYZER_REL}:{site}")

    report = json.loads((REPO_ROOT / aisr.OUT_REPORT).read_text())
    record = {
        "what_this_is": (
            "a record that the frozen Stage-1b analyzer crashed and that this "
            "script completed its report by running it with one call "
            "unwrapped. It is not part of the preregistered protocol and no "
            "frozen file was edited to produce it."),
        "generated_utc": datetime.now(timezone.utc).isoformat(
            timespec="seconds"),
        "git_commit": subprocess.run(
            ("git", "rev-parse", "HEAD"), cwd=REPO_ROOT,
            capture_output=True, text=True, check=True).stdout.strip(),
        "frozen_files_edited": [],
        "the_bug": {
            "site": "scripts/analyze_iter12_seed_replication.py:325",
            "stored": "rs.distance_to_mg(...) — a tuple[float | None, list]",
            "intended": "the distance alone; that function's own caller in "
                        "retention_selection.py unpacks it as dist, used",
            "where_it_surfaces": "line 386 collects those tuples into `dists` "
                                 "and line 394 passes them to aggregate, whose "
                                 "first statement is float(v)",
            "decision_bearing": False,
            "why_not": ("the report says distance_to_mg is 'reported per "
                        "replicate and as a mean, but the floor decides "
                        "eligibility, exactly as in Stage 1'"),
        },
        "why_a_separate_file": (
            "both files on that path are PROTOCOL_PATHS in the Stage-1b "
            "freeze, which refuses amendment once replicates have adapters on "
            "disk; that refusal is the control, so it was honoured rather "
            "than worked around by deleting the freeze"),
        "the_patch": {
            "target": "granunlearn.evaluation.retention_selection."
                      "distance_to_mg",
            "effect": "returns the first element of the tuple it would have "
                      "returned",
            "scope": "the duration of one aisr.main() call, restored in a "
                     "finally block",
            "proof_it_cannot_move_the_verdict": ast_proof,
            "proof_only_one_site_saw_it": {
                "calls_served": len(call_log),
                "expected_call_site": f"{ANALYZER_REL}:{site}",
                "call_sites_in_the_analyzer": sorted(sites),
                "calls_from_anywhere_else": unexpected,
            },
            "other_files_that_mention_the_function": callers,
        },
        "crash_reproduced_first": crash,
        "frozen_protocol_paths_verified_before_running": frozen,
        "what_still_cannot_be_claimed": (
            "that the preregistered analysis ran to completion unaided. It "
            "did not: it crashed, and a script written after the crash "
            "finished it. The decision rule is the frozen one and the frozen "
            "bytes are unchanged, but the ordering 'protocol then result' "
            "holds for the rule and not for this script."),
    }
    out = REPO_ROOT / REPAIR_RECORD
    out.write_text(json.dumps(record, indent=2, sort_keys=False) + "\n")
    print(f"repair: wrote {out.relative_to(REPO_ROOT)}")

    verdicts = report.get("verdicts", {})
    print("repair: the decision, as the frozen rule produced it")
    for parent, v in sorted(verdicts.items()):
        print(f"  {parent}: passes_on_the_mean={v['passes_on_the_mean']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
