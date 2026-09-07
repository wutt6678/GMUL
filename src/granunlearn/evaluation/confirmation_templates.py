"""Confirmation-split wording templates (Iteration 11C stage 3d).

Why this module exists separately
---------------------------------
The confirmation needs 18 template texts that do not exist in the
exploratory set: 4 new wordings for each of the three image families
(``image_fine_direct``, ``image_target_direct``, ``multimodal_image_text``)
and 3 new wordings for each of the two text-route retention families.

They cannot be added to ``FAMILY_TEMPLATES`` in
``granunlearn.evaluation.query_generation``.  That module is one of the ten
in ``CODE_FINGERPRINT_MODULES``, so a single added character changes
``code.modules_sha256`` inside all thirty committed 11R prediction sidecars
and ``verify_sidecar`` refuses every one of them -- which would discard the
exploratory predictions the confirmation is defined against.  The precedent
for the alternative is ``paired_ci.py``: deliberately outside the
fingerprint and hashed on its own.

So the templates live here and are installed into ``FAMILY_TEMPLATES`` at
RUNTIME by the builder, which is the only process that needs them.  The
exploratory dataset is never rebuilt, so it never sees them.

Why indices 3 and above
-----------------------
``_make_query`` sets ``template_id = f"{family}:{idx}"``, and every one of
the 19 exploratory families has exactly 3 templates, so the exploratory id
set is ``{family}:0/1/2`` -- 57 ids, which the freeze binds by hash.
Appending at index 3 makes the new ids ``{family}:3..6`` and they are new by
construction rather than by inspection.  ``install_confirmation_templates``
measures that precondition instead of assuming it: a family that had grown
a fourth exploratory template would silently reuse an id, so the install
refuses unless every family still has exactly
:data:`EXPLORATORY_TEMPLATES_PER_FAMILY`.

``template_id`` is a free-form string -- nothing parses it as
``family:int`` or indexes ``FAMILY_TEMPLATES`` with it -- so new ids cannot
break a downstream consumer.

Why the multimodal wording is entity-neutral
--------------------------------------------
The exploratory ``multimodal_image_text`` templates all say "person":
``The person in this image is {name}``, ``For the person shown ({name})``.
Applied to an iNaturalist species they produce "For the person shown
(Mallard)", which is in the frozen exploratory data and is not repaired
here because repairing it would mean editing a fingerprinted module and
rebuilding a dataset that thirty sidecars pin.  The confirmation templates
say "subject" and "shows" instead, so the same family is grammatical on
both entity kinds.  That is a change of wording, not of route: the family
still names the entity alongside the image, which is what distinguishes it
from the two ``image_to_text`` families that never name it.

Available fields
----------------
``_make_query`` formats with ``name``, ``attr``, ``answer``,
``distractor`` and -- for ``QUESTION_FAMILIES`` only -- ``question``.  The
two retention families are NOT question families, so their templates may
use only ``{name}`` and ``{attr}``; the three image families may use
``{question}`` and ``{name}``.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

from granunlearn.evaluation.query_generation import (
    FAMILY_TEMPLATES,
    QUESTION_FAMILIES,
)

#: Every exploratory family has exactly this many templates, so this is the
#: first index that cannot collide with an exploratory ``template_id``.
EXPLORATORY_TEMPLATES_PER_FAMILY = 3

#: The three families the 504 new-wording probes and the 360 held-out
#: photograph probes are built from.  Order matters: it is the ``f`` in
#: ``A_e[(j + f) mod |A_e|]``, so reordering it reorders which target
#: association each probe asks.
CONFIRM_IMAGE_FAMILIES = (
    "image_fine_direct",
    "image_target_direct",
    "multimodal_image_text",
)

#: The two text-route retention families, in the order the frozen allocation
#: names them.
CONFIRM_RETENTION_FAMILIES = (
    "retain_same_entity",
    "retain_other_entity",
)

#: New templates per family.  Four for the image families (3 families x 4 =
#: the 12 wording probes per person, and the 12 photograph probes per
#: species), three for the retention families (the frozen 3 probes per
#: entity).
CONFIRM_NEW_TEMPLATES = {
    "image_fine_direct": (
        "Using only what is visible here, answer: {question}",
        "{question} Reply with the most specific taxon the image supports.",
        "Consider this photograph and nothing else. {question}",
        "{question} Name the finest level you can justify from the image.",
    ),
    "image_target_direct": (
        "With reference to this image only: {question}",
        "{question} State the rank you are naming.",
        "Based on the organism pictured, respond: {question}",
        "{question} Answer at the level the question asks for.",
    ),
    # Entity-neutral on purpose: see the module docstring.
    "multimodal_image_text": (
        "This image shows {name}. {question}",
        "The subject pictured is {name}. {question}",
        "Regarding {name}, shown here: {question}",
        "Given the image of {name} attached: {question}",
    ),
    # Not QUESTION_FAMILIES, so only {name} and {attr} are available.
    "retain_same_entity": (
        "Report the {attr} on file for {name}.",
        "For {name}, what {attr} is recorded?",
        "Give the {attr} associated with {name}.",
    ),
    "retain_other_entity": (
        "Look up the {attr} for {name}.",
        "What {attr} does the record for {name} show?",
        "Provide the {attr} that is listed for {name}.",
    ),
}

#: The template indices each family's new texts occupy, as
#: ``family -> (idx, ...)``.  Derived rather than typed so the ids and the
#: texts cannot drift apart.
CONFIRM_TEMPLATE_INDICES = {
    fam: tuple(range(EXPLORATORY_TEMPLATES_PER_FAMILY,
                     EXPLORATORY_TEMPLATES_PER_FAMILY + len(texts)))
    for fam, texts in CONFIRM_NEW_TEMPLATES.items()
}

#: The 18 new template ids, in a stable order.
CONFIRM_TEMPLATE_IDS = tuple(
    f"{fam}:{idx}"
    for fam in (*CONFIRM_IMAGE_FAMILIES, *CONFIRM_RETENTION_FAMILIES)
    for idx in CONFIRM_TEMPLATE_INDICES[fam])


def _check_fields(fam: str, texts: tuple[str, ...]) -> list[str]:
    """Refuse a template that formats with a field its family never gets."""
    allowed = {"name", "attr", "answer", "distractor"}
    if fam in QUESTION_FAMILIES:
        allowed = allowed | {"question"}
    bad = []
    for text in texts:
        used = {f for f in ("name", "attr", "answer", "distractor",
                            "question") if "{" + f + "}" in text}
        unknown = sorted(used - allowed)
        if unknown:
            bad.append(f"{fam}: {text!r} uses {unknown}, which "
                       f"_make_query does not supply for this family")
        if not used:
            bad.append(f"{fam}: {text!r} substitutes nothing, so every "
                       "entity would get an identical prompt")
    return bad


def validate_confirmation_templates(exploratory_ids: set[str],
                                    exploratory_texts: set[str]) -> list[str]:
    """Everything that must hold before these templates are installed.

    Returns a list of refusals (empty == installable).  Checked against the
    exploratory set the freeze binds rather than against this module's own
    assumptions, because "new" is a property of the pair.

    The per-family exploratory count is derived from ``exploratory_ids`` and
    NOT from ``len(FAMILY_TEMPLATES[fam])``.  Reading the live list would make
    this function refuse its own second call: ``install_confirmation_templates``
    extends ``FAMILY_TEMPLATES`` in place, so after one install the live length
    is 7 and the check would report that index 3 is no longer guaranteed new
    -- which is false, and would make the build un-runnable twice in one
    process.  The bound id set does not move when the module does.
    """
    out: list[str] = []
    exploratory_counts = Counter(
        tid.rsplit(":", 1)[0] for tid in exploratory_ids)
    for fam, texts in CONFIRM_NEW_TEMPLATES.items():
        if fam not in FAMILY_TEMPLATES:
            out.append(f"{fam} is not a family the generator knows")
            continue
        have = exploratory_counts.get(fam, 0)
        if have != EXPLORATORY_TEMPLATES_PER_FAMILY:
            out.append(
                f"{fam} has {have} exploratory template_ids, not "
                f"{EXPLORATORY_TEMPLATES_PER_FAMILY}, so index "
                f"{EXPLORATORY_TEMPLATES_PER_FAMILY} is no longer guaranteed "
                "to be a new template_id")
        if len(FAMILY_TEMPLATES[fam]) < have:
            out.append(
                f"{fam} holds {len(FAMILY_TEMPLATES[fam])} templates but the "
                f"exploratory set binds {have} ids for it, so the module no "
                "longer contains the templates those ids refer to")
        for idx, text in zip(CONFIRM_TEMPLATE_INDICES[fam], texts):
            if f"{fam}:{idx}" in exploratory_ids:
                out.append(f"{fam}:{idx} is already an exploratory "
                           "template_id")
            if text in exploratory_texts:
                out.append(f"{fam}:{idx} reuses an exploratory template TEXT")
    #: ``_check_fields`` returns a LIST per family, so extending over the
    #: generator would append the lists themselves and break this function's
    #: ``list[str]`` contract -- every caller that substring-matches a refusal
    #: would then be testing list membership instead.  Extend each family's
    #: refusals in turn.
    for fam, texts in CONFIRM_NEW_TEMPLATES.items():
        out.extend(_check_fields(fam, texts))
    #: Two confirmation templates sharing a text would make one wording do
    #: double duty, and the 4-per-family count that sizes the design would
    #: overstate how much wording variety was actually tested.
    texts = [t for ts in CONFIRM_NEW_TEMPLATES.values() for t in ts]
    if len(set(texts)) != len(texts):
        dupes = sorted({t for t in texts if texts.count(t) > 1})
        out.append(f"confirmation templates are not distinct texts: {dupes}")
    return out


def install_confirmation_templates() -> dict[str, Any]:
    """Extend ``FAMILY_TEMPLATES`` in place and report what was added.

    Idempotent: a second call in the same process finds the texts already
    appended and appends nothing, which matters because a builder and its
    tests both import this module.
    """
    added: dict[str, list[str]] = {}
    for fam, texts in CONFIRM_NEW_TEMPLATES.items():
        existing = FAMILY_TEMPLATES[fam]
        start = EXPLORATORY_TEMPLATES_PER_FAMILY
        if list(existing[start:]) == list(texts):
            added[fam] = [f"{fam}:{i}" for i in
                          range(start, start + len(texts))]
            continue
        if len(existing) != start:
            raise RuntimeError(
                f"{fam} has {len(existing)} templates where "
                f"{start} exploratory ones were expected; refusing to append "
                "because the resulting template_ids would not be the new "
                "ones the freeze binds")
        #: A list, not a tuple: FAMILY_TEMPLATES holds lists and _make_query
        #: indexes them positionally.
        existing.extend(texts)
        added[fam] = [f"{fam}:{i}" for i in range(start, start + len(texts))]
    installed = [i for ids in added.values() for i in ids]
    if sorted(installed) != sorted(CONFIRM_TEMPLATE_IDS):
        raise RuntimeError(
            f"installed {len(installed)} template ids but "
            f"CONFIRM_TEMPLATE_IDS names {len(CONFIRM_TEMPLATE_IDS)}; the "
            "declaration and the install have drifted")
    return {
        "families": dict(sorted(added.items())),
        "template_ids": sorted(installed),
        "num_templates": len(installed),
        "texts": {f"{fam}:{idx}": text
                  for fam, texts in CONFIRM_NEW_TEMPLATES.items()
                  for idx, text in zip(CONFIRM_TEMPLATE_INDICES[fam], texts)},
    }
