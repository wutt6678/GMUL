"""Derive and freeze the route-stratified retention measurement basis.

Stage 1c reads the image-route retention families beside the text-route ones the
frozen floor already used.  This script writes down WHAT that basis is -- every
denominator, the route-independence evidence, the resolution each number has and
the largest resolution pilot-100 could ever give it -- before any candidate is
scored against it.

THE DISCIPLINE THAT MAKES THE POST-HOC CLAIM CHECKABLE
------------------------------------------------------
The image route was noticed after Stage 1b was scored, so this basis is not
blind.  What keeps that honest is that **nothing in this report is derived from
a prediction**.  Every field here is a property of ``queries.parquet``,
``associations.parquet`` and the already-frozen fit/probe partition.  That has a
concrete consequence: ``--check-only`` re-derives the whole document from the
dataset alone and refuses on any difference, so a reviewer can confirm the basis
was selected from structure rather than from outcomes -- and can confirm it
without loading a single prediction parquet.

The route asymmetry that motivated this (the image route losing ~2.5x more
retention across the eight Stage-1 candidates) is an OUTCOME and therefore lives
in the analysis report, not here.  Putting it here would make the basis
re-derivation depend on predictions and would destroy the check above.

THE PARTITION IS BORROWED, NOT RE-DRAWN
---------------------------------------
The fit/probe split is frozen by ``mllmu_iter12_retention_probe.json`` and by the
Stage-1 protocol freeze.  This script re-derives it with that builder's own
``hash_rule`` and ``split_entities`` and REFUSES if it disagrees with the
committed partition, so the route strata sit on exactly the entities the filed
results were measured on.  Drawing a new partition here would make the two
studies incomparable and would be a second post-hoc choice.

Nothing here reads the sealed confirmation split, and nothing edits a frozen
path: ``build_iter12_retention_probe`` and ``retention_selection`` are imported
and called.
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path
from typing import Any

import build_iter12_retention_probe as bip

from granunlearn.config import _find_repo_root
from granunlearn.evaluation import retention_selection as rs
from granunlearn.evaluation import route_stratified_retention as rt
from granunlearn.evaluation.reference_eval import (
    load_associations_parquet,
    load_queries_parquet,
)
from granunlearn.logging_utils import setup_logger

log = setup_logger(__name__)

TAG = "pilot100"
DATA_DIR = f"data/mllmu_hier_{TAG}"

#: The frozen partition this basis is built on top of.
PARTITION_REPORT = "data/reports/mllmu_iter12_retention_probe.json"
#: The partition's own input, needed to re-derive it and prove it matches.
TARGET_RETAIN_REPORT = f"data/reports/mllmu_{TAG}_target_retain.json"
OUT_REPORT = "data/reports/mllmu_iter12_route_probe.json"

#: Same selection scope as the frozen basis: the TEST split stays out of
#: selection and the sealed confirmation split is never read.
SELECTION_SPLITS = rt.SCOPE_SPLITS


def _families() -> dict[str, tuple[str, str]]:
    return dict(rt.ROUTES)


def resolvable_gaps(probe_entities: set[str], queries, associations) -> dict:
    """Smallest expressible difference for all eight floor numbers.

    Same construction as the frozen builder's, extended over both routes: every
    retention value is a mean of 0/1 outcomes, so achievable values are spaced
    and the smallest non-zero spacing is the resolution of the number.
    """
    entity_of = {a.association_id: a.entity_id for a in associations}
    out: dict[str, Any] = {}
    for route, families in rt.ROUTES:
        per_route: dict[str, Any] = {}
        for family in families:
            rows = [q for q in queries
                    if q.family == family and q.split in SELECTION_SPLITS
                    and entity_of.get(q.association_id) in probe_entities]
            per_entity = collections.Counter(entity_of[q.association_id]
                                             for q in rows)
            n, k = len(rows), len(per_entity)
            if not n or not k:
                raise SystemExit(
                    f"{family}: the frozen probe half produced no "
                    f"{SELECTION_SPLITS} queries -- the route stratum is "
                    f"unusable")
            widest = max(per_entity.values())
            per_route[family] = {
                "probe_queries": n,
                "probe_entities": k,
                "max_queries_for_one_entity": widest,
                "row_micro": {
                    "min_resolvable_gap": round(1.0 / n, 10),
                    "why": f"one flipped outcome out of {n} pooled queries",
                },
                "entity_macro": {
                    "min_resolvable_gap": round(1.0 / (k * widest), 10),
                    "why": (f"one flipped outcome in the widest of {k} "
                            f"entities ({widest} queries), then averaged over "
                            f"entities"),
                },
            }
        out[route] = per_route
    return out


def power_ceiling(probe_entities: set[str], queries, associations) -> dict:
    """The largest n pilot-100 could ever give each retention number.

    This is the part of the report that bounds the whole exercise.  The
    association pool is exhausted -- 90 target plus 387 retained is all 477 --
    so a probe can only be carved out of the retained pool, and retain-other
    depends on donor entities of which the probe half holds 23.  Whatever
    mechanism Stage 2 tests, it will be judged on these denominators unless a
    larger dataset is built, and it is better to know the ceiling now than to
    discover it after another negative result.
    """
    entity_of = {a.association_id: a.entity_id for a in associations}
    scopes = {
        "train+val, text route (the frozen basis)": (SELECTION_SPLITS, ("text",)),
        "train+val, both routes (what Stage 1c reads)":
            (SELECTION_SPLITS, ("text", "image")),
        "all splits, text route": (("train", "val", "test"), ("text",)),
        "all splits, both routes (the absolute ceiling)":
            (("train", "val", "test"), ("text", "image")),
    }
    out: dict[str, Any] = {}
    for label, (splits, routes) in scopes.items():
        block: dict[str, int] = {}
        for route in routes:
            for family in _families()[route]:
                rows = [q for q in queries
                        if q.family == family and q.split in splits
                        and entity_of.get(q.association_id) in probe_entities]
                block[f"{route}.{family}"] = len(rows)
        same = sum(v for k, v in block.items() if "same" in k)
        other = sum(v for k, v in block.items() if "other" in k)
        out[label] = {
            "per_number": block,
            "retain_same_total": same,
            "retain_other_total": other,
            "row_micro_resolution_same": round(1.0 / same, 10) if same else None,
            "row_micro_resolution_other": round(1.0 / other, 10) if other else None,
        }
    ceiling = out["all splits, both routes (the absolute ceiling)"]
    out["what_the_ceiling_means"] = {
        "retain_other_is_the_binding_number": True,
        "why": (
            "retain_other reaches only 23 probe donor entities, so its largest "
            f"possible denominator in this dataset is "
            f"{ceiling['retain_other_total']} queries and its best possible "
            f"row-micro resolution is "
            f"{ceiling['row_micro_resolution_other']}. The pool is exhausted: "
            "90 target plus 387 retained associations account for all 477, so "
            "there is no reserve to draw a larger probe from."),
        "consequence": (
            "Stage 1c roughly doubles the frozen basis and makes both parents' "
            "shortfalls exceed their between-seed spread, but it does not make "
            "retain-other well powered. A mechanism whose retention effect is "
            "smaller than about "
            f"{ceiling['row_micro_resolution_other']} cannot be resolved on "
            "this dataset at all, at any number of seeds."),
    }
    return out


def route_independence(queries, associations, probe_entities: set[str]) -> dict:
    """Evidence that the image route is a second measurement, not a duplicate.

    If the two routes shared templates or covered different associations, the
    image stratum would either be the same number twice or a number about
    something else.  Neither would justify adding four conditions to the floor.
    """
    entity_of = {a.association_id: a.entity_id for a in associations}
    scope = [q for q in queries if q.split in SELECTION_SPLITS]
    out: dict[str, Any] = {"scope": list(SELECTION_SPLITS)}
    for short, other in (("retain_same_entity", "retain_same_entity_image"),
                         ("retain_other_entity", "retain_other_entity_image")):
        t = [q for q in scope if q.family == short]
        i = [q for q in scope if q.family == other]
        ta = {q.association_id for q in t}
        ia = {q.association_id for q in i}
        tt = {q.template_id for q in t}
        it = {q.template_id for q in i}
        tp = {entity_of.get(q.association_id) for q in t} & probe_entities
        ip = {entity_of.get(q.association_id) for q in i} & probe_entities
        out[short] = {
            "text_queries": len(t),
            "image_queries": len(i),
            "association_sets_identical": ta == ia,
            "num_associations": len(ta),
            "template_sets_disjoint": not (tt & it),
            "text_templates": sorted(tt),
            "image_templates": sorted(it),
            "shared_templates": sorted(tt & it),
            "routes": {"text": sorted({q.route for q in t}),
                       "image": sorted({q.route for q in i})},
            "probe_entities_text": len(tp),
            "probe_entities_image": len(ip),
            "same_probe_entities": tp == ip,
        }
    image_families = _families()["image"]
    img = [q for q in scope if q.family in image_families
           and entity_of.get(q.association_id) in probe_entities]
    txt = [q for q in scope if q.family in rs.RETENTION_FAMILIES
           and entity_of.get(q.association_id) in probe_entities]
    out["image_provenance"] = {
        "probe_image_queries": len(img),
        #: String keys, not bools.  ``json.dump`` turns a True key into "true",
        #: so a counter keyed on bools makes a fresh derivation differ from the
        #: committed report it is supposed to reproduce -- and ``_flat`` joins
        #: keys with "." and raises on a non-string.  The two bugs cancel into
        #: one confusing failure, which is worse than either alone.
        "image_seen_in_training": dict(collections.Counter(
            "seen_in_training" if q.image_seen_in_training else "held_out"
            for q in img)),
        "image_split": dict(collections.Counter(str(q.image_split)
                                                for q in img)),
        "note": (
            "Almost every probe image query uses a photograph that was in "
            "training, with wording that was not: this is the sealed metric's "
            "seen_photo_unseen_wording stratum. The ASSOCIATION is never "
            "rehearsed -- the probe half is excluded from the replay group -- "
            "and the photograph was in training for B0 and every candidate "
            "alike, so candidate-minus-B0 on this route is a fair comparison. "
            "It is nonetheless a different instrument from the text route, "
            "which is the reason for stratifying rather than pooling."),
    }
    out["text_provenance"] = {
        "probe_text_queries": len(txt),
        "queries_carrying_an_image": sum(1 for q in txt if q.image_ids),
        "note": "The text route carries no image at all.",
    }
    return out


def build_report(repo_root: Path) -> dict[str, Any]:
    """The whole basis document, from the dataset and the frozen partition."""
    data_dir = repo_root / DATA_DIR
    queries = load_queries_parquet(data_dir / "queries.parquet")
    associations = load_associations_parquet(data_dir / "associations.parquet")

    partition = json.loads((repo_root / PARTITION_REPORT).read_text())
    probe_entities = set(partition["halves"]["probe"]["entities"])
    fit_entities = set(partition["halves"]["fit"]["entities"])

    #: Re-derive the partition with the frozen builder's own rule and refuse on
    #: any disagreement, so this basis cannot silently sit on different entities
    #: than the filed results were measured on.  The derivation mirrors
    #: ``build_iter12_retention_probe.build_report`` line for line: the pool is
    #: given as ASSOCIATION ids and entities come from associations.parquet.
    tr = json.loads((repo_root / TARGET_RETAIN_REPORT).read_text())
    retain_ids = set(tr["retain_association_ids"])
    by_id = {a.association_id: a for a in associations}
    retained = [by_id[i] for i in sorted(retain_ids) if i in by_id]
    if len(retained) != len(retain_ids):
        raise SystemExit(
            f"{len(retain_ids) - len(retained)} retained association(s) in "
            f"{TARGET_RETAIN_REPORT} are absent from associations.parquet")
    re_probe, re_fit = bip.split_entities([a.entity_id for a in retained])
    if set(re_probe) != probe_entities or set(re_fit) != fit_entities:
        raise SystemExit(
            f"{PARTITION_REPORT} does not match what the committed association "
            f"pool and seed {bip.PROBE_SEED} produce; refusing to build a "
            f"route basis on a partition that is not the frozen one")

    gaps = resolvable_gaps(probe_entities, queries, associations)
    #: The tightest of the eight numbers (2 routes x 2 families x 2 estimands).
    #: The tolerance is checked against THIS one, so a single epsilon is
    #: provably below every gap it is applied to.
    min_gap = min(block["min_resolvable_gap"]
                  for route in gaps.values()
                  for family in route.values()
                  for block in (family["row_micro"], family["entity_macro"]))

    return {
        "iteration": "12",
        "stage": "1c",
        "purpose": (
            "State the route-stratified retention measurement basis before any "
            "candidate is scored against it. The frozen floor reads four "
            "text-route numbers; the probe half's associations also back "
            "image-route families over identical association sets with "
            "disjoint templates, and their predictions already exist for every "
            "Stage-1 and Stage-1b state, so the denominators double at zero "
            "GPU cost."),
        "reuses_partition": {
            "report": PARTITION_REPORT,
            "probe_entities": len(probe_entities),
            "fit_entities": len(fit_entities),
            "re_derived_and_matched": True,
            "rule": f"sha256('{bip.PROBE_SEED}:{bip.PROBE_TAG}:<entity_id>'), "
                    f"ascending, first half probe",
            "why_not_a_new_partition": (
                "A new split would make Stage 1c incomparable with the filed "
                "Stage-1 and Stage-1b results and would be a second post-hoc "
                "choice. The route strata sit on exactly the entities the filed "
                "results were measured on."),
        },
        "no_prediction_is_read": {
            "holds": True,
            "inputs": [f"{DATA_DIR}/queries.parquet",
                       f"{DATA_DIR}/associations.parquet",
                       PARTITION_REPORT, TARGET_RETAIN_REPORT],
            "why_this_matters": (
                "The image route was identified after Stage 1b was scored, so "
                "this basis is not blind. Every field here is a property of the "
                "dataset and the frozen partition, never of a prediction, so "
                "--check-only re-derives the whole document without loading a "
                "single parquet of model outputs and a reviewer can confirm the "
                "basis was chosen from structure rather than from outcomes. The "
                "route asymmetry that motivated it is an outcome and lives in "
                "the analysis report instead."),
        },
        "floor_numbers": {
            "count": len(rt.FLOOR_NUMBER_KEYS),
            "keys": list(rt.FLOOR_NUMBER_KEYS),
            "routes": [route for route, _ in rt.ROUTES],
            "estimands": list(rs.RETENTION_ESTIMANDS),
            "rule": "candidate >= b0 - epsilon on all eight; missing "
                    "disqualifies",
            "epsilon": 1e-9,
            "epsilon_is_a_tolerance_not_a_margin": True,
            "pooled_reported_but_not_floored": list(rt.POOLED_FAMILIES),
            "why_stratified_not_pooled": (
                "Pooling would average two instruments that do not measure the "
                "same thing and report a number describing neither. Each route "
                "keeps its own four numbers and the floor requires all eight, "
                "which makes it HARDER to pass than the four-number floor it "
                "extends."),
        },
        "measurement_basis": {
            "splits": list(SELECTION_SPLITS),
            "per_route": gaps,
            "note": ("The frozen test split is not part of the selection basis "
                     "and the sealed confirmation split is never read."),
        },
        "route_independence": route_independence(queries, associations,
                                                 probe_entities),
        "power_ceiling": power_ceiling(probe_entities, queries, associations),
        "numerical_tolerance": {
            "floor_epsilon": 1e-9,
            "smallest_resolvable_gap_over_the_eight_numbers": min_gap,
            "epsilon_is_below_every_resolvable_gap": 1e-9 < min_gap,
            "what_this_forbids": (
                "Requiring any improvement over B0. Stratifying does not change "
                "the tolerance, and the tolerance is still not a margin."),
        },
        "what_this_basis_is_frozen_for": (
            "Stage 2. The mechanism tested next is judged on all eight numbers, "
            "frozen before it is trained. Stage 1c also re-scores Stage 1 and "
            "Stage 1b on this basis, but that re-scoring RE-DECIDES NOTHING: "
            "the filed four-number verdicts stand as filed, the text stratum is "
            "required to reproduce them exactly, and all eight Stage-1 "
            "candidates fail on both routes."),
        "limitations": [
            ("retain_other reaches 23 probe donor entities. Even at the "
             "absolute ceiling -- every split and both routes -- it is a small "
             "denominator, and a no-margin point floor on it stays weak. This "
             "is a property of pilot-100, not of the stratification."),
            ("The probe half carries 1 of the 6 taxonomic retained "
             "associations, so both routes speak almost entirely to semantic "
             "and numeric knowledge. Taxonomic retention remains unmeasured, "
             "not preserved."),
            ("Almost all probe image queries are seen_photo_unseen_wording. "
             "The image route therefore measures recall cued by a familiar "
             "photograph under novel wording, which is not the same instrument "
             "as the text route and must not be averaged with it."),
            ("Adding four conditions to the floor makes it harder to pass. "
             "That is the point, but it also means a Stage-2 mechanism could "
             "fail on the image route while preserving text-route retention, "
             "and the report has to say which route failed rather than only "
             "that the floor failed."),
            ("This basis was identified after Stage 1b was scored. It is "
             "disclosed, it is derived only from dataset structure, and its "
             "direction is against interest -- it enlarges the measured "
             "retention loss -- but it is not blind and is not claimed to "
             "be."),
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the Iteration 12 route-stratified retention basis")
    parser.add_argument("--check-only", action="store_true",
                        help="Re-derive the basis from the dataset and verify "
                             "the committed report without writing")
    args = parser.parse_args()

    repo_root = _find_repo_root(Path.cwd()) or Path.cwd()
    out_path = repo_root / OUT_REPORT
    fresh = build_report(repo_root)

    if args.check_only:
        if not out_path.exists():
            raise SystemExit(f"{OUT_REPORT} is absent -- nothing to check")
        committed = json.loads(out_path.read_text())
        if committed != fresh:
            fc, ff = bip._flat(committed), bip._flat(fresh)
            keys = sorted(set(fc) ^ set(ff)) or \
                sorted(k for k in fc if fc.get(k) != ff.get(k))
            raise SystemExit(
                f"{OUT_REPORT} does not match what the dataset and the frozen "
                f"partition produce; {len(keys)} field(s) differ, "
                f"e.g. {keys[:3]}")
        log.info("OK: the committed route basis re-derives exactly from "
                 "queries.parquet, associations.parquet and the frozen "
                 "partition, with no prediction read")
        return

    #: A measurement basis is not something a second run may quietly revise.
    if out_path.exists():
        committed = json.loads(out_path.read_text())
        if committed != fresh:
            raise SystemExit(
                f"{OUT_REPORT} already exists and a fresh derivation differs "
                f"from it. Refusing to overwrite a basis candidates may "
                f"already have been scored against; inspect the difference and "
                f"decide deliberately.")
        log.info("%s already exists and re-derives exactly; nothing written",
                 OUT_REPORT)
        return

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(fresh, f, indent=2, ensure_ascii=False)
        f.write("\n")
    n = fresh["floor_numbers"]["count"]
    ceil = fresh["power_ceiling"]["all splits, both routes "
                                  "(the absolute ceiling)"]
    log.info("Wrote %s: %d floor numbers over %d routes",
             out_path, n, len(fresh["floor_numbers"]["routes"]))
    log.info("absolute ceiling: retain_same %d, retain_other %d queries",
             ceil["retain_same_total"], ceil["retain_other_total"])
    log.info("smallest resolvable gap %.8f vs epsilon %g",
             fresh["numerical_tolerance"][
                 "smallest_resolvable_gap_over_the_eight_numbers"],
             fresh["numerical_tolerance"]["floor_epsilon"])


if __name__ == "__main__":
    main()
