"""Attribute inventory + core hierarchies for SALMU (Iteration 10).

Inventory division (per the Iteration 10 plan):

* CORE SEMANTIC (hierarchy built only where the released source
  information supports it):
    - city      fine=city  -> target/ancestor=country
      (the released profiles carry city + country_code; no region tier
       exists in the released metadata, so region is UNSUPPORTED here)
    - job       fine=job   -> target=profession class -> ancestor=sector
      (deterministic keyword taxonomy over the released job vocabulary)
    - blood     fine="A+"  -> target="A" (ABO group, Rh factor dropped)
* CORE NUMERIC: none — SALMU personas carry no numeric granularity
  attributes (no dates, salaries, measurements).  Documented, not
  fabricated.
* UNSUPPORTED: name (identity anchor, not a granularity attribute).
* AUXILIARY REDACTION ONLY (never part of the core hierarchical task):
    phone_number, emails, iban, credit_card, passport
  -> referenced from ``salmu_aux_redaction/`` only.

One controlled caption template per level (mirrors the MLLMU
single-template policy); MG trains ONLY on generalized target captions.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from granunlearn.logging_utils import setup_logger

log = setup_logger("salmu_hierarchy")

CORE_SEMANTIC = ("city", "job", "blood_type")
CORE_NUMERIC: tuple[str, ...] = ()
UNSUPPORTED = ("name",)
AUX_REDACTION = ("phone_number", "emails", "iban", "credit_card",
                 "passport")

ISO_COUNTRY = {
    "RU": "Russia", "US": "the United States", "CN": "China",
    "IN": "India", "BR": "Brazil", "DE": "Germany", "FR": "France",
    "GB": "the United Kingdom", "JP": "Japan", "IT": "Italy",
    "ES": "Spain", "MX": "Mexico", "TR": "Turkey", "NG": "Nigeria",
    "EG": "Egypt", "ZA": "South Africa", "KE": "Kenya", "AR": "Argentina",
    "CO": "Colombia", "CL": "Chile", "PE": "Peru", "CA": "Canada",
    "AU": "Australia", "NZ": "New Zealand", "KR": "South Korea",
    "ID": "Indonesia", "TH": "Thailand", "VN": "Vietnam",
    "PH": "the Philippines", "MY": "Malaysia", "SG": "Singapore",
    "PK": "Pakistan", "BD": "Bangladesh", "IR": "Iran", "IQ": "Iraq",
    "SA": "Saudi Arabia", "AE": "the United Arab Emirates",
    "IL": "Israel", "PL": "Poland", "UA": "Ukraine", "RO": "Romania",
    "GR": "Greece", "PT": "Portugal", "NL": "the Netherlands",
    "BE": "Belgium", "SE": "Sweden", "NO": "Norway", "FI": "Finland",
    "DK": "Denmark", "CZ": "Czechia", "HU": "Hungary", "AT": "Austria",
    "CH": "Switzerland", "IE": "Ireland", "CM": "Cameroon",
    "GH": "Ghana", "SN": "Senegal", "MA": "Morocco", "DZ": "Algeria",
    "TN": "Tunisia", "ET": "Ethiopia", "TZ": "Tanzania",
    "UG": "Uganda", "AO": "Angola", "MZ": "Mozambique",
    "BF": "Burkina Faso", "BI": "Burundi", "BW": "Botswana",
    "LY": "Libya",
    # Remaining ISO-3166 alpha-2 codes present in SALMU personas
    "AL": "Albania", "AM": "Armenia", "AZ": "Azerbaijan",
    "BA": "Bosnia and Herzegovina", "BG": "Bulgaria",
    "BN": "Brunei", "BY": "Belarus",
    "CR": "Costa Rica", "CY": "Cyprus",
    "EC": "Ecuador", "EE": "Estonia",
    "GE": "Georgia", "GT": "Guatemala",
    "HN": "Honduras", "HR": "Croatia",
    "JM": "Jamaica", "KG": "Kyrgyzstan",
    "KH": "Cambodia", "KW": "Kuwait",
    "KZ": "Kazakhstan", "LA": "Laos",
    "LK": "Sri Lanka", "LT": "Lithuania",
    "LV": "Latvia", "MD": "Moldova",
    "MK": "North Macedonia", "MM": "Myanmar",
    "MN": "Mongolia", "MT": "Malta",
    "MU": "Mauritius", "MV": "Maldives",
    "NI": "Nicaragua", "NP": "Nepal",
    "OM": "Oman", "PA": "Panama",
    "QA": "Qatar",
    "RS": "Serbia", "SI": "Slovenia",
    "SK": "Slovakia",
    "SV": "El Salvador",
    "SY": "Syria", "TM": "Turkmenistan",
    "TT": "Trinidad and Tobago",
    "TW": "Taiwan",
    "UY": "Uruguay",
    "UZ": "Uzbekistan",
    "VE": "Venezuela",
    "XK": "Kosovo",
}

# Deterministic job taxonomy: first matching keyword wins (rules are
# checked IN ORDER; committed as part of the hierarchy artifact).
# Each rule: (keyword, profession_class, sector)
JOB_RULES: list[tuple[str, str, str]] = [
    # specific-before-generic: jobs containing 'officer'/'adviser'/etc.
    # that are NOT protective-service or generic advisory roles
    ("information officer", "information professional", "media"),
    ("communications officer", "communications professional", "media"),
    ("press officer", "communications professional", "media"),
    ("public relations officer", "communications professional", "media"),
    ("marketing officer", "marketing professional", "business"),
    ("advertising officer", "marketing professional", "media"),
    ("sales officer", "retail seller", "retail"),
    ("retail officer", "retail seller", "retail"),
    ("merchandising officer", "retail seller", "retail"),
    ("customer service officer", "support worker", "business"),
    ("customer support officer", "support worker", "business"),
    ("call centre officer", "support worker", "business"),
    ("admin officer", "administration professional", "business"),
    ("administration officer", "administration professional", "business"),
    ("office officer", "administration professional", "business"),
    ("records officer", "administration professional", "business"),
    ("data officer", "analytics professional", "IT"),
    ("data entry officer", "analytics professional", "IT"),
    ("research officer", "research professional", "science"),
    ("policy officer", "government official", "government"),
    ("development officer", "social-services professional", "government"),
    ("community officer", "social-services professional", "government"),
    ("housing officer", "property professional", "business"),
    ("accommodation officer", "property professional", "business"),
    ("events officer", "administration professional", "business"),
    ("exhibitions officer", "heritage professional", "arts"),
    ("museum officer", "heritage professional", "arts"),
    ("arts officer", "heritage professional", "arts"),
    ("education officer", "education professional", "education"),
    ("training officer", "education professional", "education"),
    ("welfare officer", "social-services professional", "government"),
    ("charity officer", "social-services professional", "government"),
    ("fundraising officer", "social-services professional", "business"),
    ("membership officer", "administration professional", "business"),
    ("recruitment officer", "HR professional", "business"),
    ("human resources officer", "HR professional", "business"),
    ("personnel officer", "HR professional", "business"),
    ("pensions officer", "finance professional", "finance"),
    ("payroll officer", "finance professional", "finance"),
    ("accounts officer", "finance professional", "finance"),
    ("finance officer", "finance professional", "finance"),
    ("insurance officer", "finance professional", "finance"),
    ("claims officer", "finance professional", "finance"),
    ("compliance officer", "legal professional", "legal"),
    ("regulatory officer", "legal professional", "legal"),
    ("licensing officer", "government official", "government"),
    ("electoral officer", "government official", "government"),
    ("intelligence officer", "investigation professional", "public safety"),
    ("scientific officer", "research professional", "science"),
    ("laboratory officer", "laboratory scientist", "science"),
    ("quality officer", "technical specialist", "manufacturing"),
    ("safety officer", "environmental professional", "public safety"),
    ("conservation officer", "environmental professional", "science"),
    ("wildlife officer", "environmental professional", "science"),
    ("fisheries officer", "agricultural worker", "agriculture"),
    ("agricultural officer", "agricultural worker", "agriculture"),
    ("estate officer", "property professional", "business"),
    ("land officer", "property professional", "business"),
    ("logistics officer", "supply-chain professional", "transport"),
    ("transport officer", "transport attendant", "transport"),
    ("fleet officer", "transport attendant", "transport"),
    ("stores officer", "supply-chain worker", "transport"),
    ("procurement officer", "procurement professional", "business"),
    ("purchasing officer", "procurement professional", "business"),
    ("contracts officer", "procurement professional", "business"),
    ("catering officer", "culinary professional", "hospitality"),
    ("hospitality officer", "hospitality attendant", "hospitality"),
    ("recreation officer", "sports professional", "sports"),
    ("sports officer", "sports professional", "sports"),
    ("youth officer", "social-services professional", "government"),
    ("outreach officer", "social-services professional", "government"),
    ("liaison officer", "communications professional", "business"),
    ("public affairs officer", "communications professional", "media"),
    ("media officer", "media professional", "media"),
    ("publishing officer", "media professional", "media"),
    ("editorial officer", "media professional", "media"),
    ("broadcast officer", "media professional", "media"),
    ("operations officer", "management professional", "business"),
    ("projects officer", "management professional", "business"),
    ("programme officer", "management professional", "business"),
    ("planning officer", "planning professional", "government"),
    ("technical officer", "technical specialist", "engineering"),
    ("engineering officer", "engineering professional", "engineering"),
    ("maintenance officer", "technical specialist", "facilities"),
    ("facilities officer", "technical specialist", "facilities"),
    ("security officer", "protective-service officer", "public safety"),
    ("custody officer", "protective-service officer", "public safety"),
    ("border officer", "protective-service officer", "public safety"),
    ("financial adviser", "advisory professional", "finance"),
    ("financial controller", "finance professional", "finance"),
    ("financial trader", "finance professional", "finance"),
    ("chartered loss adjuster", "finance professional", "finance"),
    ("careers adviser", "education professional", "education"),
    ("adult guidance worker", "education professional", "education"),
    ("counsellor", "therapy professional", "healthcare"),
    ("youth worker", "social-services professional", "government"),
    ("aid worker", "social-services professional", "government"),
    ("international aid development worker", "social-services professional", "government"),
    ("international aid/development worker", "social-services professional", "government"),
    ("lobbyist", "communications professional", "government"),
    ("civil service fast streamer", "government official", "government"),
    ("health and safety inspector", "environmental professional", "public safety"),
    ("health promotion specialist", "health education professional", "healthcare"),
    ("occupational hygienist", "environmental professional", "healthcare"),
    ("exercise physiologist", "therapy professional", "healthcare"),
    ("acupuncturist", "complementary therapist", "healthcare"),
    ("homeopath", "complementary therapist", "healthcare"),
    ("osteopath", "complementary therapist", "healthcare"),
    ("optometrist", "vision-care professional", "healthcare"),
    ("orthoptist", "vision-care professional", "healthcare"),
    ("podiatrist", "therapy professional", "healthcare"),
    ("oncologist", "physician", "healthcare"),
    ("ophthalmologist", "physician", "healthcare"),
    ("pathologist", "physician", "healthcare"),
    ("haematologist", "physician", "healthcare"),
    ("immunologist", "research professional", "healthcare"),
    ("toxicologist", "research professional", "healthcare"),
    ("diagnostic radiographer", "imaging professional", "healthcare"),
    ("therapeutic radiographer", "imaging professional", "healthcare"),
    ("clinical cytogeneticist", "laboratory scientist", "healthcare"),
    ("cytogeneticist", "laboratory scientist", "healthcare"),
    ("clinical molecular geneticist", "laboratory scientist", "healthcare"),
    ("molecular geneticist", "laboratory scientist", "science"),
    ("clinical research associate", "research professional", "healthcare"),
    ("animal technologist", "laboratory scientist", "science"),
    ("herpetologist", "research professional", "science"),
    ("ecologist", "environmental professional", "science"),
    ("hydrologist", "environmental professional", "science"),
    ("astronomer", "research professional", "science"),
    ("metallurgist", "research professional", "manufacturing"),
    ("field seismologist", "research professional", "extractive"),
    ("mudlogger", "extractive technician", "extractive"),
    ("ergonomist", "design professional", "science"),
    ("colour technologist", "chemical technician", "manufacturing"),
    ("brewing technologist", "food producer", "manufacturing"),
    ("technical brewer", "food producer", "manufacturing"),
    ("commercial horticulturist", "agricultural worker", "agriculture"),
    ("ranger/warden", "environmental professional", "agriculture"),
    ("warden/ranger", "environmental professional", "agriculture"),
    ("animator", "creative professional", "media"),
    ("illustrator", "creative professional", "media"),
    ("medical illustrator", "creative professional", "media"),
    ("printmaker", "creative professional", "arts"),
    ("multimedia specialist", "digital professional", "IT"),
    ("copy", "creative professional", "media"),
    ("sub", "media professional", "media"),
    ("bookseller", "retail seller", "retail"),
    ("retail buyer", "procurement professional", "retail"),
    ("retail merchandiser", "retail seller", "retail"),
    ("visual merchandiser", "design professional", "retail"),
    ("industrial buyer", "procurement professional", "manufacturing"),
    ("dealer", "retail seller", "retail"),
    ("air broker", "advisory professional", "transport"),
    ("media buyer", "marketing professional", "media"),
    ("cabin crew", "aviation professional", "transport"),
    ("freight forwarder", "supply-chain professional", "transport"),
    ("licensed conveyancer", "legal professional", "legal"),
    ("best boy", "film production worker", "media"),
    ("museum/gallery conservator", "heritage professional", "arts"),
    ("doctor", "physician", "healthcare"),
    ("physician", "physician", "healthcare"),
    ("surgeon", "physician", "healthcare"),
    ("nurse", "nursing professional", "healthcare"),
    ("dentist", "dental professional", "healthcare"),
    ("pharmacist", "pharmacy professional", "healthcare"),
    ("therapist", "therapy professional", "healthcare"),
    ("psychologist", "therapy professional", "healthcare"),
    ("psychiatrist", "physician", "healthcare"),
    ("optician", "vision-care professional", "healthcare"),
    ("embryologist", "laboratory scientist", "healthcare"),
    ("radiologist", "physician", "healthcare"),
    ("veterinarian", "veterinary professional", "healthcare"),
    ("dietitian", "nutrition professional", "healthcare"),
    ("engineer", "engineering professional", "engineering"),
    ("architect", "design professional", "construction"),
    ("surveyor", "construction technician", "construction"),
    ("builder", "construction worker", "construction"),
    ("electrician", "electrical trades worker", "construction"),
    ("plumber", "plumbing trades worker", "construction"),
    ("carpenter", "wood trades worker", "construction"),
    ("welder", "metal trades worker", "manufacturing"),
    ("mechanic", "vehicle technician", "transport"),
    ("pilot", "aviation professional", "transport"),
    ("driver", "driving professional", "transport"),
    ("sailor", "maritime professional", "transport"),
    ("ship", "maritime professional", "transport"),
    ("conductor", "transport attendant", "transport"),
    ("teacher", "education professional", "education"),
    ("professor", "higher-education professional", "education"),
    ("lecturer", "higher-education professional", "education"),
    ("tutor", "education professional", "education"),
    ("instructor", "education professional", "education"),
    ("trainer", "education professional", "education"),
    ("librarian", "information professional", "education"),
    ("researcher", "research professional", "science"),
    ("scientist", "research professional", "science"),
    ("analyst", "analytics professional", "business"),
    ("economist", "analytics professional", "business"),
    ("statistician", "analytics professional", "science"),
    ("accountant", "finance professional", "finance"),
    ("auditor", "finance professional", "finance"),
    ("banker", "finance professional", "finance"),
    ("advisor", "advisory professional", "business"),
    ("consultant", "advisory professional", "business"),
    ("manager", "management professional", "business"),
    ("director", "management professional", "business"),
    ("executive", "management professional", "business"),
    ("administrator", "administration professional", "business"),
    ("coordinator", "administration professional", "business"),
    ("assistant", "support worker", "business"),
    ("secretary", "support worker", "business"),
    ("clerk", "support worker", "business"),
    ("receptionist", "support worker", "hospitality"),
    ("lawyer", "legal professional", "legal"),
    ("attorney", "legal professional", "legal"),
    ("judge", "legal professional", "legal"),
    ("notary", "legal professional", "legal"),
    ("paralegal", "legal support worker", "legal"),
    ("police", "protective-service officer", "public safety"),
    ("officer", "protective-service officer", "public safety"),
    ("firefighter", "protective-service officer", "public safety"),
    ("soldier", "military personnel", "public safety"),
    ("guard", "protective-service officer", "public safety"),
    ("detective", "investigation professional", "public safety"),
    ("politician", "government official", "government"),
    ("minister", "government official", "government"),
    ("mayor", "government official", "government"),
    ("diplomat", "government official", "government"),
    ("civil servant", "government official", "government"),
    ("social worker", "social-services professional", "government"),
    ("journalist", "media professional", "media"),
    ("reporter", "media professional", "media"),
    ("editor", "media professional", "media"),
    ("writer", "creative professional", "media"),
    ("author", "creative professional", "media"),
    ("translator", "language professional", "media"),
    ("interpreter", "language professional", "media"),
    ("photographer", "creative professional", "media"),
    ("designer", "design professional", "media"),
    ("artist", "creative professional", "arts"),
    ("musician", "performing artist", "arts"),
    ("singer", "performing artist", "arts"),
    ("dancer", "performing artist", "arts"),
    ("actor", "performing artist", "arts"),
    ("chef", "culinary professional", "hospitality"),
    ("cook", "culinary professional", "hospitality"),
    ("waiter", "hospitality attendant", "hospitality"),
    ("bartender", "hospitality attendant", "hospitality"),
    ("barista", "hospitality attendant", "hospitality"),
    ("guide", "tourism professional", "hospitality"),
    ("travel agent", "tourism professional", "hospitality"),
    ("farmer", "agricultural worker", "agriculture"),
    ("fisher", "agricultural worker", "agriculture"),
    ("gardener", "grounds worker", "agriculture"),
    ("florist", "retail seller", "retail"),
    ("shop", "retail seller", "retail"),
    ("sales", "retail seller", "retail"),
    ("cashier", "retail seller", "retail"),
    ("baker", "food producer", "manufacturing"),
    ("butcher", "food producer", "manufacturing"),
    ("tailor", "textile worker", "manufacturing"),
    ("seamstress", "textile worker", "manufacturing"),
    ("printer", "production technician", "manufacturing"),
    ("operator", "production technician", "manufacturing"),
    ("technician", "technical specialist", "engineering"),
    ("developer", "software professional", "IT"),
    ("programmer", "software professional", "IT"),
    ("software", "software professional", "IT"),
    ("web", "digital professional", "IT"),
    ("data", "analytics professional", "IT"),
    ("IT", "IT specialist", "IT"),
    ("network", "IT specialist", "IT"),
    ("database", "IT specialist", "IT"),
    ("systems", "IT specialist", "IT"),
    ("computer", "IT specialist", "IT"),
    ("communications", "media professional", "media"),
    ("public relations", "communications professional", "media"),
    ("marketing", "marketing professional", "business"),
    ("advertising", "marketing professional", "media"),
    ("insurance", "finance professional", "finance"),
    ("real estate", "property professional", "business"),
    ("estate agent", "property professional", "business"),
    ("cleaner", "cleaning worker", "facilities"),
    ("janitor", "cleaning worker", "facilities"),
    ("housekeeper", "cleaning worker", "hospitality"),
    ("carer", "care worker", "healthcare"),
    ("care worker", "care worker", "healthcare"),
    ("midwife", "nursing professional", "healthcare"),
    ("paramedic", "emergency medical worker", "healthcare"),
    ("fitness", "sports professional", "sports"),
    ("coach", "sports professional", "sports"),
    ("athlete", "sports professional", "sports"),
    ("sport", "sports professional", "sports"),
    ("priest", "religious worker", "religious"),
    ("chaplain", "religious worker", "religious"),
    ("missionary", "religious worker", "religious"),
    ("air traffic", "aviation professional", "transport"),
    ("customs", "government official", "government"),
    ("immigration", "government official", "government"),
    ("postal", "postal worker", "transport"),
    ("postman", "postal worker", "transport"),
    ("courier", "postal worker", "transport"),
    ("event", "administration professional", "business"),
    ("human resources", "HR professional", "business"),
    ("personnel", "HR professional", "business"),
    ("recruit", "HR professional", "business"),
    ("customer", "support worker", "business"),
    ("call centre", "support worker", "business"),
    ("claims", "finance professional", "finance"),
    ("underwriter", "finance professional", "finance"),
    ("tax", "finance professional", "finance"),
    ("purchasing", "procurement professional", "business"),
    ("logistics", "supply-chain professional", "transport"),
    ("warehouse", "supply-chain worker", "transport"),
    ("miner", "extractive worker", "extractive"),
    ("geologist", "research professional", "extractive"),
    ("environmental", "environmental professional", "science"),
    ("biologist", "research professional", "science"),
    ("chemist", "research professional", "science"),
    ("physicist", "research professional", "science"),
    ("meteorologist", "research professional", "science"),
    ("actuary", "analytics professional", "finance"),
    ("survey", "analytics professional", "business"),
    ("planner", "planning professional", "government"),
    ("urban", "planning professional", "government"),
    ("copywriter", "creative professional", "media"),
    ("proofreader", "media professional", "media"),
    ("broadcaster", "media professional", "media"),
    ("producer", "media professional", "media"),
    ("presenter", "media professional", "media"),
    ("curator", "heritage professional", "arts"),
    ("archaeologist", "heritage professional", "science"),
    ("historian", "heritage professional", "education"),
    ("cartographer", "technical specialist", "science"),
    ("purchaser", "procurement professional", "business"),
    ("auctioneer", "retail seller", "retail"),
    ("jeweller", "craft worker", "retail"),
    ("watchmaker", "craft worker", "retail"),
    ("cobbler", "craft worker", "retail"),
    ("locksmith", "trades worker", "facilities"),
    ("roofer", "construction worker", "construction"),
    ("painter", "trades worker", "construction"),
    ("decorator", "trades worker", "construction"),
    ("plasterer", "construction worker", "construction"),
    ("bricklayer", "construction worker", "construction"),
    ("glazier", "construction worker", "construction"),
    ("scaffolder", "construction worker", "construction"),
    ("crane", "construction technician", "construction"),
    ("forklift", "warehouse worker", "transport"),
    ("bus", "driving professional", "transport"),
    ("taxi", "driving professional", "transport"),
    ("train", "transport attendant", "transport"),
    ("ambulance", "emergency medical worker", "healthcare"),
]

JOB_FALLBACK = ("general professional", "other")

# One controlled template per hierarchy + level (MG trains ONLY on the
# target-level template; MF uses the released paraphrased fine captions).
GENERIC_TEMPLATES = {
    "city": {1: "{name} lives in {value}."},
    "job": {1: "{name} works as {article} {value}.",
            2: "{name} works in the {value} sector."},
    "blood_type": {1: "{name}'s blood type is {value}."},
}

# Nameless variants (no identity name — used for evaluation probes so
# that sibling / target contrasts test branch specificity rather than
# identity discrimination).
NAMELESS_TEMPLATES = {
    "city": {1: "A person who lives in {value}."},
    "job": {1: "A person who works as {article} {value}.",
            2: "A person who works in the {value} sector."},
    "blood_type": {1: "A person whose blood type is {value}."},
}


def classify_job(job: str) -> tuple[str, str]:
    """Deterministic (profession_class, sector) for a released job."""
    lowered = job.lower()
    for keyword, pclass, sector in JOB_RULES:
        if keyword.lower() in lowered:
            return pclass, sector
    return JOB_FALLBACK


def country_name(country_code: str) -> str:
    return ISO_COUNTRY.get(country_code.upper(), country_code.upper())


def abo_group(blood_type: str) -> str:
    """'A+' / 'O-' / 'AB+' -> ABO group letter(s)."""
    return blood_type.strip().rstrip("+-")


def build_persona_hierarchies(
    identities: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Core hierarchical associations per persona.

    Returns ``{identity_id: {attr: {"levels": [fine, ..., coarsest],
    "target_level": idx, "source": ...}}}`` — levels ordered finest
    first, matching the MLLMU AssociationRecord convention.
    """
    out: dict[str, dict[str, Any]] = {}
    for iid in sorted(identities):
        p = identities[iid]
        hier: dict[str, Any] = {}
        if p.get("city") and p.get("country_code"):
            hier["city"] = {
                "levels": [p["city"], country_name(p["country_code"])],
                "target_level": 1,
                "source": "identities_metadata city + country_code",
            }
        if p.get("job"):
            pclass, sector = classify_job(p["job"])
            hier["job"] = {
                "levels": [p["job"], pclass, sector],
                "target_level": 1,
                "source": "identities_metadata job + deterministic "
                          "taxonomy",
            }
        if p.get("blood_type"):
            hier["blood_type"] = {
                "levels": [p["blood_type"], abo_group(p["blood_type"])],
                "target_level": 1,
                "source": "identities_metadata blood_type (Rh dropped)",
            }
        out[iid] = hier
    return out


def generalized_caption(
    name: str, attribute: str, level_index: int, value: str
) -> str:
    """Controlled caption at a given hierarchy level."""
    template = GENERIC_TEMPLATES[attribute].get(level_index)
    if template is None:
        raise ValueError(
            f"No generalized template for {attribute} level {level_index}")
    article = "an" if value[:1].lower() in "aeiou" else "a"
    return template.format(name=name, value=value, article=article)


def nameless_caption(
    attribute: str, level_index: int, value: str
) -> str:
    """Nameless caption for evaluation probes (no identity anchor).

    Uses the same hierarchy level/value as ``generalized_caption`` but
    omits the persona name so that sibling vs target contrasts test
    branch specificity rather than identity discrimination.
    """
    template = NAMELESS_TEMPLATES[attribute].get(level_index)
    if template is None:
        raise ValueError(
            f"No nameless template for {attribute} level {level_index}")
    article = "an" if value[:1].lower() in "aeiou" else "a"
    return template.format(value=value, article=article)


def write_attribute_inventory(out_path: str | Path,
                              identities: dict[str, dict]) -> dict:
    """Commit the inventory division with coverage statistics."""
    from collections import Counter
    jobs = Counter(p["job"] for p in identities.values() if p.get("job"))
    fallback_jobs = sorted(j for j in jobs
                           if classify_job(j) == JOB_FALLBACK)
    inv = {
        "core_semantic": list(CORE_SEMANTIC),
        "core_numeric": list(CORE_NUMERIC),
        "core_numeric_note": (
            "SALMU personas carry no numeric granularity attributes "
            "(no dates/salaries/measurements); the phenomenon is tested "
            "on semantic granularity here."),
        "unsupported": list(UNSUPPORTED),
        "unsupported_note": "name is the identity anchor, not a "
                            "granularity attribute",
        "aux_redaction": list(AUX_REDACTION),
        "hierarchies": {
            "city": {"chain": ["city", "country"],
                     "region_note": "region unsupported: released "
                                    "metadata carries city + country only"},
            "job": {"chain": ["job", "profession_class", "sector"],
                    "taxonomy": "deterministic keyword rules, first match "
                                "wins, fallback 'general professional/other'",
                    "num_unique_jobs": len(jobs),
                    "num_fallback_jobs": len(fallback_jobs),
                    "fallback_jobs": fallback_jobs},
            "blood_type": {"chain": ["full blood type", "ABO group"]},
        },
        "templates": GENERIC_TEMPLATES,
    }
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(inv, f, indent=2, ensure_ascii=False)
    log.info("Attribute inventory -> %s (%d fallback jobs of %d)",
             out_path, len(fallback_jobs), len(jobs))
    return inv
