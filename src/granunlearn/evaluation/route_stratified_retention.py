"""Iteration 12 Stage 1c: the retention floor, stratified by answer route.

WHAT THIS ADDS
--------------
The frozen floor reads four numbers -- ``retain_same_entity`` and
``retain_other_entity``, each at row-micro and entity-macro -- and both of those
families are ``text_to_text``.  The probe half's 204 associations back a second
pair of families over exactly the same associations, ``*_image``, whose route is
``image_to_text``, and the sealed :func:`compute_hierarchy_metrics` has always
emitted blocks for them (plus a pooled ``*_all_routes``).  Iteration 12 never
read them.

Stage 1b showed why that matters.  Its binding failure was retain-other
row-micro, where the between-seed sd (0.0225, 0.0515) EXCEEDED the shortfall it
was adjudicating (-0.0165, -0.0263): 76 queries resolve in units of 1/76 while
one seed moved the number by up to 9 queries.  The image families cover the same
associations with disjoint templates, so reading them doubles every denominator
-- and their predictions were already generated for all ten Stage-1 states and
all seven Stage-1b states, because those runs covered all 4,518 train+val
queries.  The power is available at zero GPU cost.

WHY STRATIFIED RATHER THAN POOLED
---------------------------------
Pooling text and image into one rate would mix two instruments.  They are not
interchangeable measurements of the same thing: across the eight Stage-1
candidates the image route loses roughly 2.5x more retention than the text route
(retain-same -0.1373 vs -0.0558, retain-other -0.1382 vs -0.0559), and it is the
worse route for 7 of the 8.  Averaging them would report a single number that
describes neither.  So each route keeps its own four numbers and the floor
requires all eight, which makes the floor HARDER to pass than the one it
extends.  The pooled ``*_all_routes`` blocks are reported beside the decision as
a power comparison and floor nothing.

WHAT IS FROZEN, AND WHAT IS HONEST ABOUT IT
-------------------------------------------
The comparison rule is unchanged from :func:`~granunlearn.evaluation.
retention_selection.floor_check`: ``candidate >= baseline - FLOOR_EPSILON``,
missing disqualifies rather than passing, and the epsilon is a float tolerance
and not a margin.  The text stratum is not reimplemented here at all -- it is
:func:`~granunlearn.evaluation.retention_selection.probe_retention` called
directly, so its four numbers are byte-identical to the ones already filed, and
:func:`text_stratum_reproduces_the_frozen_floor` asserts at run time that the
stratified verdict restricted to text equals the frozen floor's verdict.  An
extension that disagrees with the rule it extends on the stratum they share
would be a different rule wearing its name.

The disclosure: this basis was identified AFTER Stage 1b was scored, so it is
not blind.  Two things keep it from being a rule chosen for its answer.  It was
found in the dataset's structure -- the image families exist, cover identical
association sets, and share no templates with the text ones -- not in the
outcomes; and its direction is against interest, since it enlarges the measured
retention loss and makes the floor harder.  Every candidate fails on both routes,
so it changes no Stage-1 verdict.  What it is legitimately preregistered for is
Stage 2: the mechanism that comes next is judged on all eight numbers, frozen
before it is trained.

Nothing here edits a frozen path.  ``retention_selection`` and
``hierarchy_metrics`` are hash-bound by the Stage-1 and confirmation freezes
respectively and are only imported and called.  Nothing reads the sealed
confirmation split.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from granunlearn.evaluation import retention_selection as rs
from granunlearn.evaluation.hierarchy_metrics import compute_hierarchy_metrics
from granunlearn.schema import AssociationRecord, PredictionRecord, QueryRecord

#: The two answer routes the floor is stratified over, and the retention family
#: each route reads.  Order is part of the contract: the eight floor numbers are
#: enumerated from it, so a reordering changes the report's key order.
ROUTES: tuple[tuple[str, tuple[str, str]], ...] = (
    ("text", ("retain_same_entity", "retain_other_entity")),
    ("image", ("retain_same_entity_image", "retain_other_entity_image")),
)

#: Reported beside the decision, floored by nothing.  These are the sealed
#: metric's own pooled blocks -- the same 408+408 and 76+76 queries, averaged --
#: and they exist here so a reader can see what pooling would have said.
POOLED_FAMILIES: tuple[str, ...] = (
    "retain_same_entity_all_routes",
    "retain_other_entity_all_routes",
)

#: The eight numbers the stratified floor reads, as (route, family, estimand).
FLOOR_NUMBERS_ROUTES: tuple[tuple[str, str, str], ...] = tuple(
    (route, family, estimand)
    for route, families in ROUTES
    for family in families
    for estimand in rs.RETENTION_ESTIMANDS
)

#: ``text.retain_same_entity.row_micro`` -- the flat key a floor vector uses.
def number_key(route: str, family: str, estimand: str) -> str:
    return f"{route}.{family}.{estimand}"


FLOOR_NUMBER_KEYS: tuple[str, ...] = tuple(
    number_key(*t) for t in FLOOR_NUMBERS_ROUTES)

#: Same splits as the frozen basis.  The test split stays out of the selection
#: basis, and the sealed confirmation split is never read.
SCOPE_SPLITS: tuple[str, ...] = ("train", "val")


def image_route_retention(predictions: Sequence[PredictionRecord],
                          queries: Sequence[QueryRecord],
                          associations: Sequence[AssociationRecord],
                          probe_entities: Sequence[str]) -> dict[str, Any]:
    """The image route's four numbers, with the sealed metric cross-checking.

    Mirrors :func:`~granunlearn.evaluation.retention_selection.probe_retention`
    rather than calling it, because that function's family list is the frozen
    ``RETENTION_FAMILIES`` and its cross-check reads ``sealed[family]`` for those
    names only.  The discipline is copied exactly: row-micro is read from the
    sealed implementation applied to probe-filtered rows and compared against a
    direct computation over precisely the rows passed in, so a filter that stops
    reaching the sealed path raises instead of filing a plausible wrong number.
    """
    entity_map = rs.entity_of(associations)
    probe = frozenset(probe_entities)
    split_of = {q.query_id: q.split for q in queries}
    probe_preds = [p for p in predictions
                   if split_of.get(p.query_id) in SCOPE_SPLITS]

    families = dict(ROUTES)["image"]
    rows = {family: rs._rows(probe_preds, queries, entity_map, family, probe)
            for family in families}
    #: The sealed metric is handed ONLY probe image-route retention rows.
    probe_only = [p for family in families for _, p in rows[family]]
    sealed = compute_hierarchy_metrics(probe_only, queries, associations,
                                       split=None)
    out: dict[str, Any] = {"scope": list(SCOPE_SPLITS),
                           "probe_entities": len(probe),
                           "route": "image_to_text",
                           "source": "computed here; row-micro cross-checked "
                                     "against the sealed hierarchy_metrics"}
    for family in families:
        family_rows = rows[family]
        block = sealed.get(family) or {}
        micro = block.get("baseline_accuracy")
        micro = round(float(micro), rs.RATE_DECIMALS) if micro is not None \
            else None
        direct = rs.row_micro_rate(family_rows)
        if micro != direct:
            raise AssertionError(
                f"{family}: the sealed metric returned {micro} over rows that "
                f"compute directly to {direct}, so the probe restriction did "
                f"not reach it -- the image stratum would have been measured "
                f"on in-sample retention")
        if block.get("num_queries") != len(family_rows):
            raise AssertionError(
                f"{family}: the sealed metric counted "
                f"{block.get('num_queries')} queries but {len(family_rows)} "
                f"probe rows were passed to it")
        out[family] = {
            "num_queries": len(family_rows),
            "num_entities": len({e for e, _ in family_rows}),
            "row_micro": micro,
            "entity_macro": rs.entity_macro_rate(family_rows),
        }
    return out


def pooled_all_routes(predictions: Sequence[PredictionRecord],
                      queries: Sequence[QueryRecord],
                      associations: Sequence[AssociationRecord],
                      probe_entities: Sequence[str]) -> dict[str, Any]:
    """The sealed metric's pooled blocks over both routes.  Floors nothing."""
    entity_map = rs.entity_of(associations)
    probe = frozenset(probe_entities)
    split_of = {q.query_id: q.split for q in queries}
    probe_preds = [p for p in predictions
                   if split_of.get(p.query_id) in SCOPE_SPLITS]
    families = tuple(f for _, fams in ROUTES for f in fams)
    rows = [r for family in families
            for r in rs._rows(probe_preds, queries, entity_map, family, probe)]
    sealed = compute_hierarchy_metrics([p for _, p in rows], queries,
                                       associations, split=None)
    out: dict[str, Any] = {
        "floored": False,
        "why_reported": "what pooling the two routes would have said, so the "
                        "stratified choice is checkable against its alternative",
    }
    for family in POOLED_FAMILIES:
        block = sealed.get(family) or {}
        micro = block.get("baseline_accuracy")
        out[family] = {
            "num_queries": block.get("num_queries"),
            "row_micro": round(float(micro), rs.RATE_DECIMALS)
            if micro is not None else None,
        }
    return out


def probe_retention_by_route(predictions: Sequence[PredictionRecord],
                             queries: Sequence[QueryRecord],
                             associations: Sequence[AssociationRecord],
                             probe_entities: Sequence[str]) -> dict[str, Any]:
    """Both routes' retention on the probe half, plus the pooled view.

    The text stratum IS :func:`~...retention_selection.probe_retention`, called
    rather than copied, so its numbers cannot drift from the filed ones.
    """
    text = rs.probe_retention(predictions, queries, associations,
                              probe_entities)
    image = image_route_retention(predictions, queries, associations,
                                  probe_entities)
    return {
        "scope": list(SCOPE_SPLITS),
        "probe_entities": len(frozenset(probe_entities)),
        "routes": {"text": text, "image": image},
        "pooled_all_routes": pooled_all_routes(predictions, queries,
                                               associations, probe_entities),
    }


def floor_vector(by_route: dict[str, Any]) -> dict[str, float | None]:
    """Flatten both strata into the eight keyed numbers the floor compares."""
    out: dict[str, float | None] = {}
    for route, family, estimand in FLOOR_NUMBERS_ROUTES:
        block = (by_route.get("routes", {}).get(route) or {}).get(family) or {}
        out[number_key(route, family, estimand)] = block.get(estimand)
    return out


def text_only_vector(vector: dict[str, float | None]) -> dict[str, Any]:
    """Re-nest a floor vector's text half into ``floor_check``'s input shape."""
    out: dict[str, Any] = {}
    for family in rs.RETENTION_FAMILIES:
        out[family] = {estimand: vector.get(
            number_key("text", family, estimand))
            for estimand in rs.RETENTION_ESTIMANDS}
    return out


def floor_check_stratified(candidate: dict[str, float | None],
                           baseline: dict[str, float | None],
                           epsilon: float = rs.FLOOR_EPSILON) -> dict[str, Any]:
    """All eight numbers, under the frozen rule, unchanged in semantics.

    ``candidate >= baseline - epsilon`` on every number; a number missing on
    either side DISQUALIFIES, because a retention value that could not be
    computed is not evidence that retention was preserved.  The result carries
    the frozen floor's verdict on the text half alongside, and the two are
    required to agree -- see :func:`text_stratum_reproduces_the_frozen_floor`.
    """
    checks: dict[str, Any] = {}
    eligible = True
    for key in FLOOR_NUMBER_KEYS:
        got = candidate.get(key)
        want = baseline.get(key)
        if got is None or want is None:
            checks[key] = {
                "candidate": got, "b0": want, "passes": False,
                "why": "not measurable on the probe half -- treated as a "
                       "failure, because an absent retention number is not "
                       "evidence of preserved retention",
            }
            eligible = False
            continue
        passes = got >= want - epsilon
        checks[key] = {
            "candidate": got, "b0": want,
            "difference": round(got - want, 6),
            "passes": passes,
            "comparison": f"candidate >= b0 - {epsilon:g}",
        }
        eligible = eligible and passes
    frozen = rs.floor_check(text_only_vector(candidate),
                            text_only_vector(baseline), epsilon)
    per_route = {
        route: all(checks[number_key(route, family, estimand)]["passes"]
                   for family in families
                   for estimand in rs.RETENTION_ESTIMANDS)
        for route, families in ROUTES
    }
    return {
        "eligible": eligible,
        "epsilon": epsilon,
        "epsilon_is_a_tolerance_not_a_margin": True,
        "checks": checks,
        "num_numbers": len(FLOOR_NUMBER_KEYS),
        "direction": rs.FLOOR_DIRECTION,
        "per_route": per_route,
        "frozen_floor_on_the_text_stratum": frozen,
        "estimands_required": list(rs.RETENTION_ESTIMANDS),
        "routes_required": [route for route, _ in ROUTES],
    }


def text_stratum_reproduces_the_frozen_floor(result: dict[str, Any]) -> bool:
    """Does the stratified floor's text half equal ``floor_check``'s verdict?

    Run on every scored candidate rather than once at design time.  The
    stratified floor must be an EXTENSION: on the stratum the two rules share it
    has to say exactly what the frozen rule says, or it is a different rule and
    the comparison with the filed Stage-1 and Stage-1b results is meaningless.
    """
    frozen = result["frozen_floor_on_the_text_stratum"]
    mine = all(result["checks"][number_key("text", family, estimand)]["passes"]
               for family in rs.RETENTION_FAMILIES
               for estimand in rs.RETENTION_ESTIMANDS)
    if mine != frozen["eligible"]:
        raise AssertionError(
            f"the stratified floor's text stratum says {mine} but the frozen "
            f"floor_check says {frozen['eligible']} -- the extension disagrees "
            f"with the rule it extends on the numbers they share")
    for family in rs.RETENTION_FAMILIES:
        for estimand in rs.RETENTION_ESTIMANDS:
            key = f"{family}.{estimand}"
            mine_c = result["checks"][number_key("text", family, estimand)]
            theirs = frozen["checks"][key]
            if mine_c["passes"] != theirs["passes"]:
                raise AssertionError(
                    f"{key}: stratified says {mine_c['passes']}, frozen says "
                    f"{theirs['passes']}")
            if mine_c.get("difference") != theirs.get("difference"):
                raise AssertionError(
                    f"{key}: stratified difference "
                    f"{mine_c.get('difference')} != frozen "
                    f"{theirs.get('difference')}")
    return True
