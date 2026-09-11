"""Iteration 12: split the retained entities into a replayed FIT half and a
never-replayed PROBE half.

    python scripts/build_iter12_retention_probe.py

WHY THIS SCRIPT EXISTS
----------------------
The Iteration 11C confirmation and every pilot-100 selection number measured
retention on the SAME associations that the ``retain`` replay group is fitted
on.  That group is exactly the 387 retained associations, and it backs 100% of
the retention queries in all three splits:

    retain_same_entity    387 associations per split, 387 in the replay group
    retain_other_entity    65 donor associations,      65 in the replay group

So a candidate carrying ``sft retain`` is scored IN-SAMPLE on retention while
B0 (the no-op MF copy) is scored OUT-OF-SAMPLE.  A floor that says "retention
must not fall below B0" then compares two different quantities, and it cannot
discriminate: applied to the existing pilot-100 grid it passes exactly the two
candidates that carry a retain group and fails all nine that do not.  It
selects FOR replay instead of testing it.

The fix is a partition of the 70 retained ENTITIES into a fit half, whose
associations are replayed, and a probe half, whose associations are never
replayed by any Iteration 12 candidate.  The floor and every retention number
this study reports are then measured on probe queries, where B0 and a replay
candidate are scored on the same footing: neither was trained on them.

Splitting at the ENTITY level rather than the association level is what makes
"never replayed" true of a whole entity's knowledge.  An association-level
split would leave every probe fact belonging to an entity that was still
partly rehearsed, which is a weaker claim and a leakier one.

WHAT IS AND IS NOT WRITTEN
--------------------------
``fine_target.jsonl`` and ``target_level.jsonl`` are COPIED BYTE-FOR-BYTE from
the pilot-100 group directory.  The target-side objective must not move: the
only difference between an Iteration 12 candidate and the incumbent it is
compared against is how much retained knowledge is rehearsed.  Only
``retain.jsonl`` is new, and it is the fit half of the same examples the
pilot-100 group already contained -- re-derived, filtered, never rewritten.

Nothing here touches the frozen dataset.  ``dataset_fingerprint`` hashes only
``associations.parquet``, ``queries.parquet``, ``manifest.json`` and the image
manifest roll-up, so a new sibling group directory leaves the confirmation
freeze byte-identical -- verified before this script was written, and asserted
again by ``tests/unit/test_iteration12_protocol.py``.

This script needs no GPU and reads no prediction.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

from granunlearn.config import _find_repo_root
from granunlearn.evaluation.reference_eval import (
    load_associations_parquet,
    load_queries_parquet,
)
from granunlearn.logging_utils import setup_logger
from granunlearn.training.unlearning_datasets import (
    build_unlearning_group,
    validate_unlearning_groups,
)

log = setup_logger("build_iter12_retention_probe")

#: The exploratory dataset this study reuses.  Iteration 12 does NOT re-freeze
#: a dataset: same queries, same associations, same ``pilot100_v2`` version, so
#: its numbers stay comparable with the pilot-100 selection it succeeds.
TAG = "pilot100"
DATA_DIR = f"data/mllmu_hier_{TAG}"
PILOT_GROUPS = f"{DATA_DIR}/unlearning"
ITER12_GROUPS = f"{DATA_DIR}/unlearning_iter12"
PARTITION_REPORT = f"data/reports/mllmu_{TAG}_target_retain.json"
OUT_REPORT = "data/reports/mllmu_iter12_retention_probe.json"

#: The two group files that must not change by one byte.
COPIED_GROUPS = ("fine_target.jsonl", "target_level.jsonl")

#: Selection scope.  The frozen TEST split is never used for selection, and
#: the sealed confirmation split is never read at all.
SELECTION_SPLITS = ("train", "val")
RETENTION_FAMILIES = ("retain_same_entity", "retain_other_entity")

#: The partition is derived, never drawn by hand: entities are ordered by a
#: seeded hash of their id and cut in half.  Reproducible from the committed
#: association pool alone, so a reviewer can rebuild it without this script.
PROBE_SEED = 42
PROBE_TAG = "iter12probe"


def hash_rule(entity_id: str) -> str:
    """The frozen ordering key for one entity."""
    return hashlib.sha256(
        f"{PROBE_SEED}:{PROBE_TAG}:{entity_id}".encode()).hexdigest()


def split_entities(retained_entity_ids: list[str]) -> tuple[list[str], list[str]]:
    """(probe, fit) -- the first half of the hash order is never replayed."""
    ordered = sorted(set(retained_entity_ids), key=hash_rule)
    cut = len(ordered) // 2
    return ordered[:cut], ordered[cut:]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def resolvable_gaps(probe_entities: set[str], queries, associations) -> dict:
    """The smallest difference each frozen retention number can express.

    This is what makes the floor's numerical tolerance honest.  Every
    retention value here is a mean of 0/1 outcomes -- pooled for row-micro,
    averaged over per-entity means for entity-macro -- so the achievable
    values are spaced, not continuous.  The smallest non-zero spacing is the
    smallest difference the measurement can express at all, and a tolerance
    strictly below it can only ever absorb float REPRESENTATION error.  It can
    never decide a real comparison, which is the difference between a
    tolerance and a margin.
    """
    entity_of = {a.association_id: a.entity_id for a in associations}
    out: dict[str, Any] = {}
    for family in RETENTION_FAMILIES:
        rows = [q for q in queries
                if q.family == family and q.split in SELECTION_SPLITS
                and entity_of.get(q.association_id) in probe_entities]
        per_entity = collections.Counter(entity_of[q.association_id]
                                         for q in rows)
        n, k = len(rows), len(per_entity)
        if not n or not k:
            raise SystemExit(
                f"{family}: the probe half produced no {SELECTION_SPLITS} "
                f"queries -- the partition is unusable")
        widest = max(per_entity.values())
        out[family] = {
            "probe_queries": n,
            "probe_entities": k,
            "max_queries_for_one_entity": widest,
            "row_micro": {
                "min_resolvable_gap": round(1.0 / n, 10),
                "why": f"one flipped outcome out of {n} pooled queries",
            },
            "entity_macro": {
                "min_resolvable_gap": round(1.0 / (k * widest), 10),
                "why": (f"one flipped outcome in the widest of {k} entities "
                        f"({widest} queries), then averaged over entities"),
            },
        }
    return out


def build_report(repo_root: Path, write: bool = False) -> dict[str, Any]:
    """Derive the partition, write the fit-half groups, return the report."""
    data_dir = repo_root / DATA_DIR
    associations = load_associations_parquet(data_dir / "associations.parquet")
    queries = load_queries_parquet(data_dir / "queries.parquet")
    partition = json.loads((repo_root / PARTITION_REPORT).read_text())

    retain_ids = set(partition["retain_association_ids"])
    target_ids = set(partition["target_association_ids"])
    by_id = {a.association_id: a for a in associations}
    retained = [by_id[i] for i in sorted(retain_ids) if i in by_id]
    if len(retained) != len(retain_ids):
        raise SystemExit(
            f"{len(retain_ids) - len(retained)} retained association(s) in "
            f"the partition are absent from associations.parquet")

    probe_ents, fit_ents = split_entities(
        [a.entity_id for a in retained])
    probe_set, fit_set = set(probe_ents), set(fit_ents)
    probe_assoc = sorted(a.association_id for a in retained
                         if a.entity_id in probe_set)
    fit_assoc = sorted(a.association_id for a in retained
                       if a.entity_id in fit_set)

    #: The load-bearing invariant, checked before anything is written.
    assert probe_set & fit_set == set(), "an entity is in both halves"
    assert sorted(probe_assoc + fit_assoc) == sorted(retain_ids), \
        "the two halves do not exactly cover the retained associations"
    assert set(probe_assoc) & set(fit_assoc) == set(), \
        "an association is in both halves"
    assert set(probe_assoc) & target_ids == set(), \
        "a probe association is also a target"

    gaps = resolvable_gaps(probe_set, queries, associations)
    #: The tightest of the four frozen retention numbers (2 families x 2
    #: estimands).  The tolerance is checked against THIS one, so a single
    #: epsilon is provably below every gap it is applied to.
    min_gap = min(block["min_resolvable_gap"]
                  for family in gaps.values()
                  for block in (family["row_micro"], family["entity_macro"]))

    out_dir = repo_root / ITER12_GROUPS
    if write:
        out_dir.mkdir(parents=True, exist_ok=True)
        for name in COPIED_GROUPS:
            shutil.copyfile(repo_root / PILOT_GROUPS / name, out_dir / name)

    #: The replay group is the fit half of the SAME examples, re-derived
    #: through the group builder and filtered -- not retyped, so the prompt
    #: template, level index and image paths cannot drift from the pilot's.
    all_retain = build_unlearning_group(associations, partition, "retain")
    fit_examples = [ex for ex in all_retain if ex.entity_id in fit_set]
    if len(fit_examples) != len(fit_assoc):
        raise SystemExit(
            f"the fit half should yield {len(fit_assoc)} replay examples, "
            f"the group builder yielded {len(fit_examples)}")
    leaked = sorted({ex.association_id for ex in fit_examples} & set(probe_assoc))
    if leaked:
        raise SystemExit(
            f"{len(leaked)} PROBE association(s) reached the replay group "
            f"(e.g. {leaked[:3]}): the partition does not hold")

    #: validate_unlearning_groups demands that ``retain`` cover exactly the
    #: partition's retain set, so it is handed the DERIVED partition whose
    #: retain set is the fit half.  The target-side checks still run against
    #: the untouched target set, which is the part that must not move.
    derived = dict(partition)
    derived["retain_association_ids"] = fit_assoc
    fine = build_unlearning_group(associations, partition, "fine_target")
    tlevel = build_unlearning_group(associations, partition, "target_level")
    errors = validate_unlearning_groups(
        {"fine_target": fine, "target_level": tlevel,
         "retain": fit_examples}, derived)
    if errors:
        raise SystemExit(f"fit-half groups failed validation: {errors[:5]}")

    if write:
        with open(out_dir / "retain.jsonl", "w") as f:
            f.writelines(ex.model_dump_json() + "\n" for ex in fit_examples)
        manifest = {
            "groups": {
                "fine_target": {
                    "path": str(out_dir / "fine_target.jsonl"),
                    "num_examples": len(fine),
                    "provenance": "copied byte-for-byte from "
                                  f"{PILOT_GROUPS}/fine_target.jsonl",
                },
                "target_level": {
                    "path": str(out_dir / "target_level.jsonl"),
                    "num_examples": len(tlevel),
                    "provenance": "copied byte-for-byte from "
                                  f"{PILOT_GROUPS}/target_level.jsonl",
                },
                "retain": {
                    "path": str(out_dir / "retain.jsonl"),
                    "num_examples": len(fit_examples),
                    "provenance": "the FIT half of the pilot-100 retain "
                                  "group; probe associations are excluded by "
                                  "construction and asserted absent",
                },
            },
            "partition_report": OUT_REPORT,
            "note": (
                "Iteration 12 training groups.  Identical to the pilot-100 "
                "groups except that retain.jsonl holds only the fit half of "
                "the retained entities, so the probe half is never rehearsed "
                "by any candidate and retention can be measured "
                "out-of-sample."),
        }
        with open(out_dir / "unlearning_groups_manifest.json", "w") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)

    entity_of = {a.association_id: a.entity_id for a in associations}
    stream_share_pilot = len(retain_ids) / (
        len(target_ids) * 2 + len(retain_ids))
    stream_share_fit = len(fit_assoc) / (len(target_ids) * 2 + len(fit_assoc))

    return {
        "iteration": "12",
        "purpose": (
            "Make the retention floor measurable.  Every retained association "
            "in this dataset backs a retention query, and the pilot-100 "
            "replay group contains all of them, so retention was scored "
            "in-sample for replay candidates and out-of-sample for B0.  The "
            "probe half is never replayed, which puts both on the same "
            "footing."),
        "reuses_dataset": {
            "tag": TAG,
            "data_dir": DATA_DIR,
            "version": json.loads(
                (data_dir / "manifest.json").read_text()).get("version"),
            "why_not_a_new_dataset": (
                "Iteration 12 changes which knowledge is rehearsed, not which "
                "queries are asked.  Keeping the pilot-100 queries, "
                "associations and dataset version means its candidates stay "
                "directly comparable with the pilot-100 selection they "
                "succeed, and the frozen confirmation dataset fingerprint is "
                "untouched."),
        },
        "partition_rule": {
            "unit": "entity",
            "why_entity_not_association": (
                "An association-level split leaves a probe fact belonging to "
                "an entity whose other facts were rehearsed, so the entity "
                "was partly seen.  Entity-level makes 'never replayed' true "
                "of a whole entity's knowledge."),
            "seed": PROBE_SEED,
            "ordering": (
                f"entities sorted ascending by sha256("
                f"'{PROBE_SEED}:{PROBE_TAG}:<entity_id>')"),
            "cut": "first half of the ordering is PROBE, remainder is FIT",
            "derived_from": PARTITION_REPORT,
            "no_randomness_beyond_the_seed": True,
        },
        "halves": {
            "probe": {
                "entities": probe_ents,
                "num_entities": len(probe_ents),
                "associations": probe_assoc,
                "num_associations": len(probe_assoc),
                "by_hierarchy_type": dict(collections.Counter(
                    by_id[i].hierarchy_type for i in probe_assoc)),
                "replayed_by_any_candidate": False,
            },
            "fit": {
                "entities": fit_ents,
                "num_entities": len(fit_ents),
                "associations": fit_assoc,
                "num_associations": len(fit_assoc),
                "by_hierarchy_type": dict(collections.Counter(
                    by_id[i].hierarchy_type for i in fit_assoc)),
                "replayed_by_any_candidate": True,
            },
            "entities_carrying_both_a_target_and_retained_knowledge":
                len({a.entity_id for a in retained}
                    & {entity_of[i] for i in target_ids if i in entity_of}),
        },
        "probe_measurement_basis": {
            "splits": list(SELECTION_SPLITS),
            "families": list(RETENTION_FAMILIES),
            "per_family": gaps,
            "note": (
                "The frozen test split is not part of the selection basis, and "
                "the sealed confirmation split is never read by this study."),
        },
        "numerical_tolerance": {
            "floor_epsilon": 1e-9,
            "smallest_resolvable_gap_over_the_four_frozen_numbers": min_gap,
            "epsilon_is_below_every_resolvable_gap": 1e-9 < min_gap,
            "orders_of_magnitude_below": (
                round(min_gap / 1e-9, 1) if min_gap else None),
            "what_this_permits": (
                "Absorbing float representation error when a candidate is "
                "EXACTLY at B0's level, so an identical score is not reported "
                "as a shortfall by the last bit of a division."),
            "what_this_forbids": (
                "Requiring any improvement over B0.  The smallest difference "
                "these measurements can express is ~2e-3 and the tolerance is "
                "1e-9, so it cannot flip a real comparison.  This is a "
                "tolerance, NOT a margin: a +0.02 retention margin was "
                "considered and rejected because nothing in the evidence "
                "supports a threshold of that size."),
        },
        "replay_strength": {
            "pilot100_retain_examples": len(retain_ids),
            "iter12_fit_examples": len(fit_assoc),
            "how_weight_enters_the_objective": (
                "loss = sign * weight * example_loss / accumulation_window, "
                "and each epoch interleaves the groups round-robin, so a "
                "group's influence is its weight times its share of the "
                "stream.  The share is set by example COUNT, which this "
                "partition halves."),
            "retain_share_of_stream_pilot100": round(stream_share_pilot, 6),
            "retain_share_of_stream_iter12_at_weight_1": round(
                stream_share_fit, 6),
            "weight_reproducing_the_pilot_influence": round(
                stream_share_pilot / stream_share_fit, 4),
            "consequence": (
                "The incumbent recipe retrained on the fit half is NOT the "
                "incumbent: halving the replay group also cuts its share of "
                "every epoch's gradient stream.  That retrained row is "
                "therefore the closest available control, and the weight sweep "
                "spans from below the pilot's effective influence to well "
                "above it rather than assuming weight 1.0 reproduces it."),
        },
        "limitations": [
            ("The probe half carries 1 of the 6 taxonomic retained "
            "associations, so probe retention speaks almost entirely to "
            "semantic and numeric knowledge. Taxonomic retention is not "
            "measurable on this partition and is recorded as unmeasured, not "
            "as preserved."),
            ("retain_other_entity reaches 23 probe donor entities, so its "
            "entity-macro value averages 23 clusters. That is enough for a "
            "point-estimate floor and too few for an interval; this study "
            "selects, it does not do inference."),
            ("Halving the replay group halves the rehearsal available to the "
            "mechanism under test. A candidate that fails the floor may fail "
            "for want of examples rather than for want of the mechanism, "
            "which is why the weight sweep, not the group size, is the knob."),
            ("The partition is derived from the pilot-100 association pool, "
            "which is the whole pool: 90 target plus 387 retained associations "
            "account for all 477. There is no untouched reserve, so a probe "
            "half can only be carved out of the replay group."),
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the Iteration 12 fit/probe retention partition")
    parser.add_argument("--check-only", action="store_true",
                        help="Derive the partition and verify the committed "
                             "report and group files without writing")
    args = parser.parse_args()

    repo_root = _find_repo_root(Path.cwd()) or Path.cwd()
    out_path = repo_root / OUT_REPORT

    if args.check_only:
        if not out_path.exists():
            raise SystemExit(f"{OUT_REPORT} is absent -- nothing to check")
        committed = json.loads(out_path.read_text())
        fresh = build_report(repo_root, write=False)
        if committed != fresh:
            keys = sorted(set(_flat(committed)) ^ set(_flat(fresh))) or \
                sorted(k for k in _flat(committed)
                       if _flat(committed).get(k) != _flat(fresh).get(k))
            raise SystemExit(
                f"{OUT_REPORT} does not match what the committed association "
                f"pool produces; {len(keys)} field(s) differ, e.g. {keys[:3]}")
        groups = repo_root / ITER12_GROUPS
        for name in COPIED_GROUPS:
            a, b = groups / name, repo_root / PILOT_GROUPS / name
            if _sha256(a) != _sha256(b):
                raise SystemExit(
                    f"{name} is not byte-identical to the pilot-100 group it "
                    f"must be copied from")
        replay = {json.loads(line)["association_id"] for line in
                  (groups / "retain.jsonl").read_text().splitlines()
                  if line.strip()}
        probe = set(committed["halves"]["probe"]["associations"])
        if replay & probe:
            raise SystemExit(
                f"{len(replay & probe)} probe association(s) are in the "
                f"committed replay group")
        log.info("OK: the committed partition, group files and probe/fit "
                 "disjointness all match what the association pool produces")
        return

    #: A frozen partition is not something a second run may quietly revise.
    #: Rebuilding must reproduce it exactly, so a mismatch is a refusal
    #: rather than an overwrite.
    if out_path.exists():
        committed = json.loads(out_path.read_text())
        fresh = build_report(repo_root, write=False)
        if committed != fresh:
            raise SystemExit(
                f"{OUT_REPORT} already exists and a fresh derivation differs "
                f"from it. Refusing to overwrite a partition that candidates "
                f"may already have been trained against; inspect the "
                f"difference and decide deliberately.")

    report = build_report(repo_root, write=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    h = report["halves"]
    log.info("probe: %d entities / %d associations (never replayed)",
             h["probe"]["num_entities"], h["probe"]["num_associations"])
    log.info("fit:   %d entities / %d associations (the replay group)",
             h["fit"]["num_entities"], h["fit"]["num_associations"])
    log.info("smallest resolvable retention gap %.8f vs epsilon %g",
             report["numerical_tolerance"][
                 "smallest_resolvable_gap_over_the_four_frozen_numbers"],
             report["numerical_tolerance"]["floor_epsilon"])
    log.info("Wrote %s and %s", out_path, repo_root / ITER12_GROUPS)


def _flat(doc: dict, prefix: tuple = ()) -> dict:
    out: dict = {}
    for key, value in doc.items():
        path = prefix + (key,)
        if isinstance(value, dict):
            out.update(_flat(value, path))
        else:
            out[".".join(path)] = value
    return out


if __name__ == "__main__":
    main()
