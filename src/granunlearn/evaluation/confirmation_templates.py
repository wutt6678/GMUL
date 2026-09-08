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

import re
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
    #: ENTITY-NEUTRAL, and neutral about what the image can show.  The first
    #: revision of these eight wrappers was written for the taxonomic stratum
    #: and then applied to the MLLMU persons as well, because all twelve image
    #: templates are shared by both strata.  Four of them named a taxon, a
    #: rank or an organism outright, and two more instructed the model to use
    #: "only what is visible here" -- which is not answerable for a salary, a
    #: birthplace or a birth date, none of which is a visible property of a
    #: portrait.  On the person stratum that put 168 of its 504 probes, exactly
    #: one third, behind a wrapper that contradicted the question it wrapped,
    #: and the person stratum is the one that carried the exploratory effect.
    #: See protocol_amendments in the power report.
    #:
    #: What a wrapper may say is therefore bounded: it may refer to the image
    #: and to the act of answering, and it may ask for precision or for the
    #: requested specificity.  It may not name a kind of entity, a level of a
    #: taxonomy, or a channel of evidence the question does not use.
    "image_fine_direct": (
        "Review the attached image, then answer precisely: {question}",
        "{question} Respond with the exact answer.",
        "Considering the attached image, answer: {question}",
        "{question} Give the most specific answer requested.",
    ),
    "image_target_direct": (
        "Use the attached image when answering: {question}",
        "{question} Respond at the requested level of specificity.",
        "Regarding the subject shown, answer: {question}",
        "{question} Give only the requested category or range.",
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


#: --------------------------------------------------------------------------
#: The bound on what a SHARED wrapper may say.
#:
#: Every template in ``CONFIRM_IMAGE_FAMILIES`` renders on BOTH strata: the 42
#: MLLMU persons and the 30 iNaturalist species.  Nothing in this module says
#: which, so a wrapper written with one stratum in mind is silently applied to
#: the other, and the only way to see that is to render it and read the result.
#:
#: The first revision of the eight ``image_fine_direct`` /
#: ``image_target_direct`` wrappers was written for the taxonomic stratum.  On
#: the person stratum that put 168 of its 504 probes -- exactly one third --
#: behind a wrapper that contradicted the question it wrapped, and the person
#: stratum is the one that carried the exploratory effect.  Two classes of
#: wording are therefore refused, and they are refused SEPARATELY because they
#: are different defects:
#:
#: 1. ``ENTITY_SPECIFIC_VOCABULARY`` -- naming a kind of entity or a level of a
#:    taxonomy asserts something about the subject.  "Reply with the most
#:    specific taxon" has no answer for a salary; "State the rank you are
#:    naming" has none for a birth date.
#: 2. ``EVIDENCE_RESTRICTING_PHRASES`` -- limiting the model to what the image
#:    shows.  This is NOT domain vocabulary, so a vocabulary-only repair would
#:    have kept it, and it is just as wrong: none of the seven person
#:    attributes in this split (salary, birthplace, date of birth, residence,
#:    occupation, education, height) is a visible property of a portrait.
#:
#: What a wrapper MAY do is refer to the image and to the act of answering, and
#: ask for precision or for the requested specificity.
#: --------------------------------------------------------------------------
ENTITY_SPECIFIC_VOCABULARY = (
    "taxon", "taxa", "taxonomic", "taxonomy", "rank", "organism", "species",
    "person", "people", "human", "animal", "plant", "bird", "insect",
    "portrait", "face",
    #: The taxonomic idiom for "most specific taxon", banned as a bigram.
    #: ``level`` alone is NOT banned and must not be: the sanctioned neutral
    #: wordings say "the requested level of specificity" and "the level the
    #: question asks for", so banning the word would refuse the repair.
    "finest level",
)

EVIDENCE_RESTRICTING_PHRASES = (
    "only what is visible", "nothing else", "from the image",
    "the image supports", "justify from", "visible", "as seen",
    #: Added by running this bound over ``RETIRED_WRAPPER_WORDINGS``: the first
    #: seven phrases caught six of the eight retired wordings and let
    #: "With reference to this image only" through, because it restricts the
    #: evidence channel without using the word "visible" or the phrase
    #: "nothing else".  A bound validated only against the module it ships
    #: with keeps exactly the holes nobody happened to test.
    "image only", "only the image", "image alone", "based on the image",
)

#: The eight wordings this bound retires, quoted verbatim from the sealed
#: module so the record says what was removed and not merely that something
#: was.  Reproduce with
#:   git show 18fd88d:src/granunlearn/evaluation/confirmation_templates.py
#:
#: SEVEN of the eight leak, in two distinct classes:
#:
#:   * FOUR name a taxon, a rank, an organism or a taxonomic level --
#:     ``image_fine_direct:4`` and ``:6``, ``image_target_direct:4`` and
#:     ``:5``.  At 42 persons apiece these are the 4 x 42 = 168 probes, one
#:     third of the 504-probe person stratum, that the review counted.
#:   * THREE restrict the model to the visual channel --
#:     ``image_fine_direct:3`` and ``:5``, ``image_target_direct:3``.  A
#:     further 3 x 42 = 126 person-stratum probes, which no vocabulary list
#:     catches, because none of the seven person attributes in this split is a
#:     visible property of a portrait.
#:
#: The eighth, ``image_target_direct:6`` "Answer at the level the question
#: asks for", is already neutral and was replaced for uniformity rather than
#: because it leaked -- all eight are rewritten together so that no wrapper in
#: a family is distinguishable from its siblings by vintage.
RETIRED_WRAPPER_WORDINGS = {
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
}

#: The fields ``_make_query`` substitutes.  Blanking all of them leaves the
#: wrapper's OWN words, which is the only part this module is responsible for:
#: the question comes from the fingerprinted exploratory generator and
#: legitimately says "the person shown in this image", and ``{name}`` is an
#: entity the wrapper was given.
TEMPLATE_FIELDS = ("question", "name", "attr", "answer", "distractor")


def wrapper_fragments(text: str) -> list[str]:
    """The wrapper's own wording, with every substituted field blanked out."""
    sentinel = "\x00"
    for field in TEMPLATE_FIELDS:
        text = text.replace("{" + field + "}", sentinel)
    return [p.strip() for p in text.split(sentinel) if p.strip()]


def offending_wrapper_vocabulary(fragments: list[str]) -> list[str]:
    """Which banned wordings appear in a wrapper's own fragments.

    Word-bounded on purpose: "Give the most specific answer requested" must
    not be refused because ``specific`` contains the letters of ``species``.
    """
    joined = " ".join(fragments).lower()
    hits = [w for w in ENTITY_SPECIFIC_VOCABULARY
            if re.search(r"\b" + re.escape(w) + r"\b", joined)]
    hits += [p for p in EVIDENCE_RESTRICTING_PHRASES if p in joined]
    return hits


def wrapper_neutrality_refusals() -> list[str]:
    """Refuse a confirmation wrapper that is not neutral about its stratum.

    Checked over every family in ``CONFIRM_NEW_TEMPLATES``, not only the three
    image ones: the retention wrappers are shared across strata too, and a
    bound that covers only the families that have already leaked is a bound
    that waits for the next one.
    """
    out: list[str] = []
    for fam, texts in CONFIRM_NEW_TEMPLATES.items():
        for idx, text in zip(CONFIRM_TEMPLATE_INDICES[fam], texts):
            hits = offending_wrapper_vocabulary(wrapper_fragments(text))
            if hits:
                out.append(
                    f"{fam}:{idx} is shared by the person and the species "
                    f"stratum but its own wording says {hits}: {text!r}")
    return out


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
    #: The stratum-neutrality bound.  Enforced HERE, in the validator the
    #: builder calls, and not only in a test: a test that fails after a bad
    #: wrapper is committed still leaves a built split on disk, while a
    #: refusal here means the leaking split is never written at all.
    out.extend(wrapper_neutrality_refusals())
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
