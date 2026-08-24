"""Qwen-assisted semantic hierarchy pipeline (Iteration 5).

    occupation / education source values
             |
        Qwen proposal  (strict JSON schema, greedy decoding)
             |
      deterministic validation  (structure + guardrails, no LLM)
             |
   independent Qwen verification  (different prompt role)
             |
    ambiguity / confidence gate
             |
       accepted hierarchy  (or explicit rejection reason)
             |
      manual audit sample  (seeded, committed)

KEY PRINCIPLE: rejection over forced hierarchy construction.  If no
defensible chain can be produced and verified, the value is EXCLUDED
with a recorded reason — never padded with artificial abstractions.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any

from granunlearn.hierarchy.canonicalize import normalize

# ---------------------------------------------------------------------------
# Prompt templates (their sha256 is the committed prompt_version)
# ---------------------------------------------------------------------------

PROPOSAL_PROMPT = """You are a careful taxonomy builder for a knowledge-unlearning research dataset.

Task: for the {attribute_phrase} value below, propose a specificity hierarchy from the exact value to increasingly general categories.

Source value: "{value}"

Strict rules:
1. The first element of "chain" MUST be exactly the source value.
2. Each subsequent element MUST be a true, widely accepted generalization of the previous element.
3. Use 2 to 5 levels total. Stop earlier rather than inventing a category.
4. Do NOT add information that is not entailed by the source value (no prestige, rankings, discipline, geography, degree level, employer, or dates).
5. If you cannot produce a defensible chain, return an empty chain.
{attribute_extra_rules}
Output STRICT JSON ONLY (no markdown, no commentary), exactly this schema:
{{"chain": ["level0", "level1", "..."], "confidence": 0.0, "ambiguity_note": ""}}

"confidence" is your honest 0-1 confidence that the WHOLE chain is correct.
"""

EDUCATION_EXTRA_RULES = """Education-specific rules (be conservative):
- Generalization levels may only describe the INSTITUTION TYPE (e.g. "university", "higher-education institution", "educational institution").
- NEVER infer academic prestige, discipline, geographic category, or degree level from the institution name.
- If the institution type cannot be determined from the name alone, return an empty chain.
"""

OCCUPATION_EXTRA_RULES = ""

VERIFICATION_PROMPT = """You are an INDEPENDENT verifier for a knowledge-unlearning research dataset. You did not create this proposal; judge it strictly.

Source value: "{value}"
Proposed hierarchy (finest to coarsest): {chain_json}

Check EVERY step:
1. Is each level a true generalization of the level directly above it?
2. Does any level introduce information not entailed by the source value (e.g. prestige, rankings, discipline, geography, degree level, employer)?
3. Is any level vague, ambiguous, or debatable?

Reject unless the entire chain is defensible. When in doubt, reject.

Output STRICT JSON ONLY (no markdown, no commentary), exactly this schema:
{{"step_valid": [true, false], "unsupported_information": false, "ambiguous": false, "verdict": "accept", "confidence": 0.0, "reason": ""}}

"step_valid" has one boolean per adjacent pair (length = len(chain) - 1).
"verdict" is "accept" only if every step is valid, there is no unsupported information, and nothing is ambiguous.
"""


def prompt_version(*templates: str) -> str:
    """Stable version hash of the exact prompt texts."""
    h = hashlib.sha256()
    for t in templates:
        h.update(t.encode())
        h.update(b"\x00")
    return h.hexdigest()[:16]


ATTRIBUTE_PROMPT_PARTS = {
    "occupation": {
        "attribute_phrase": "occupation",
        "attribute_extra_rules": OCCUPATION_EXTRA_RULES,
    },
    "education": {
        "attribute_phrase": "education (attended institution)",
        "attribute_extra_rules": EDUCATION_EXTRA_RULES,
    },
}


# ---------------------------------------------------------------------------
# Deterministic validation (no LLM)
# ---------------------------------------------------------------------------

#: Terms that must never appear in INFERRED (coarser) levels of an
#: education chain — they indicate unsupported information.
EDUCATION_FORBIDDEN_TERMS = (
    "bachelor", "master", "phd", "doctorate", "doctoral", "degree",
    "diploma", "prestigious", "prestige", "elite", "top-ranked",
    "top ranked", "ranking", "ranked", "ivy league", "selective",
    "best ", "leading ", "world-class",
)

_JSON_FENCE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.S)


_CHAIN_ARRAY = re.compile(r'"chain"\s*:\s*(\[.*?\])', re.S)
_CONFIDENCE = re.compile(r'"confidence"\s*:\s*([0-9]*\.?[0-9]+)')


def _recover_truncated_object(text: str) -> dict | None:
    """Recover a chain object from output truncated before its closing brace.

    Models sometimes exhaust the token budget inside a trailing free-text
    field (e.g. ``ambiguity_note``) after the structured part is complete.
    If a full ``"chain": [...]`` array (and optionally ``confidence``) is
    present, we reconstruct JUST those fields — free-text tails are never
    synthesized.
    """
    m = _CHAIN_ARRAY.search(text)
    if not m:
        return None
    try:
        chain = json.loads(m.group(1))
    except json.JSONDecodeError:
        return None
    if not isinstance(chain, list):
        return None
    obj: dict[str, Any] = {"chain": chain}
    c = _CONFIDENCE.search(text)
    if c:
        obj["confidence"] = float(c.group(1))
    return obj


def extract_json(text: str) -> dict | None:
    """Extract the first JSON object from a model output (fence-tolerant).

    Falls back to recovering a complete ``chain`` array from outputs
    truncated before the closing brace.
    """
    candidates = []
    m = _JSON_FENCE.search(text)
    if m:
        candidates.append(m.group(1))
    start, end = text.find("{"), text.rfind("}")
    if 0 <= start < end:
        candidates.append(text[start:end + 1])
    for cand in candidates:
        try:
            obj = json.loads(cand)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            continue
    return _recover_truncated_object(text)


def validate_proposal(
    attribute: str, source_value: str, obj: dict[str, Any]
) -> tuple[list[str] | None, str | None]:
    """Deterministic structural validation of a proposal.

    Returns ``(chain, None)`` on success or ``(None, reason)`` on failure.
    This stage uses NO LLM judgment — only structure and guardrails.
    """
    chain = obj.get("chain")
    if not isinstance(chain, list) or not all(isinstance(c, str) for c in chain):
        return None, "schema_invalid"
    chain = [c.strip() for c in chain]
    if len(chain) == 0:
        # The model itself declined: an honourable rejection, not a failure.
        return None, "model_declined"
    if len(chain) < 2:
        return None, "deterministic_invalid:chain_too_short"
    if len(chain) > 5:
        return None, "deterministic_invalid:chain_too_long"
    if any(not c for c in chain):
        return None, "deterministic_invalid:empty_level"
    if normalize(chain[0]) != normalize(source_value):
        return None, "deterministic_invalid:first_level_mismatch"
    norms = [normalize(c) for c in chain]
    if len(set(norms)) != len(norms):
        return None, "deterministic_invalid:duplicate_levels"

    if attribute == "education":
        for level in chain[1:]:
            low = level.lower()
            for term in EDUCATION_FORBIDDEN_TERMS:
                if term in low:
                    return None, f"education_guardrail:forbidden_term:{term.strip()}"
    return chain, None


def validate_verification(
    chain: list[str], obj: dict[str, Any]
) -> tuple[bool, str]:
    """Structural check of a verification output; returns (ok, reason).

    Deterministic leniency rule: verifiers occasionally emit one boolean
    per LEVEL instead of per adjacent pair.  If ``step_valid`` has exactly
    ``len(chain)`` entries and they are ALL true, it is collapsed to the
    first ``len(chain) - 1`` entries.  Any false entry, or any other length
    mismatch, stays a schema violation (never loosened for rejections).
    """
    steps = obj.get("step_valid")
    if not isinstance(steps, list) or not all(isinstance(s, bool) for s in steps):
        return False, "verifier_schema_invalid"
    n_steps = len(chain) - 1
    if len(steps) == len(chain) and len(steps) == n_steps + 1 and all(steps):
        steps = steps[:n_steps]
        obj["step_valid"] = steps  # normalize for downstream evidence
    if len(steps) != n_steps:
        return False, "verifier_schema_invalid"
    if obj.get("verdict") not in ("accept", "reject"):
        return False, "verifier_schema_invalid"
    if not isinstance(obj.get("confidence"), (int, float)):
        return False, "verifier_schema_invalid"
    return True, ""


# ---------------------------------------------------------------------------
# JSONL cache (keyed by prompt version + distinct value)
# ---------------------------------------------------------------------------

class JsonlCache:
    """Append-friendly JSONL cache keyed by (attribute, prompt_version, value)."""

    def __init__(self, cache_dir: Path | str):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.hits = 0
        self.writes = 0

    def _path(self, kind: str, prompt_ver: str) -> Path:
        return self.cache_dir / f"{kind}__{prompt_ver}.jsonl"

    def load(self, kind: str, prompt_ver: str) -> dict[str, dict]:
        path = self._path(kind, prompt_ver)
        entries: dict[str, dict] = {}
        if path.exists():
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        row = json.loads(line)
                        # Re-parse with the CURRENT extractor: improvements
                        # apply to cached raw outputs without regeneration.
                        row["parsed"] = extract_json(row.get("raw_output", ""))
                        entries[row["key"]] = row
        return entries

    def append(self, kind: str, prompt_ver: str, rows: list[dict]) -> None:
        path = self._path(kind, prompt_ver)
        with open(path, "a") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        self.writes += len(rows)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class SemanticPipelineResult:
    def __init__(self) -> None:
        self.accepted_chains: dict[str, dict[str, list[str]]] = {}  # attr -> value -> chain
        self.outcomes: dict[str, Counter] = {}        # attr -> reason -> count (distinct values)
        self.profile_outcomes: dict[str, Counter] = {}  # attr -> reason -> count (profiles)
        self.details: dict[str, list[dict]] = {}      # attr -> per-value records
        self.report: dict[str, Any] = {}


def run_semantic_pipeline(
    generator,
    attribute_values: dict[str, list[tuple[str, str]]],
    config: dict[str, Any],
    cache_dir: Path | str,
    seed: int = 42,
) -> SemanticPipelineResult:
    """Run proposal -> validation -> verification -> gate for each attribute.

    Parameters
    ----------
    generator : QwenGenerator (or any object with .generate(prompts) and
        .provenance()); a fake client is used in unit tests.
    attribute_values : dict
        ``attribute -> [(entity_id, raw_value), ...]`` (per-profile pairs).
    config : dict
        Keys: ``min_proposal_confidence`` (0.6), ``min_verification_confidence``
        (0.7), ``audit_sample_accepted`` (10), ``audit_sample_rejected`` (5).
    """
    min_prop_conf = float(config.get("min_proposal_confidence", 0.6))
    min_ver_conf = float(config.get("min_verification_confidence", 0.7))
    generate_kwargs = dict(config.get("generate_kwargs", {}))
    cache_identity = dict(config.get("cache_identity", {}))
    # Generation settings AND model-load configuration are part of the
    # cache identity: a rerun with a different token budget or load mode
    # (4-bit vs BF16) must never reuse prior outputs.
    settings_ver = hashlib.sha256(
        json.dumps({**generate_kwargs, **cache_identity}, sort_keys=True)
        .encode()).hexdigest()[:8]
    prop_ver = prompt_version(PROPOSAL_PROMPT, EDUCATION_EXTRA_RULES,
                              OCCUPATION_EXTRA_RULES) + "-" + settings_ver
    ver_ver = prompt_version(VERIFICATION_PROMPT) + "-" + settings_ver

    cache = JsonlCache(cache_dir)
    result = SemanticPipelineResult()

    for attr, pairs in attribute_values.items():
        parts = ATTRIBUTE_PROMPT_PARTS[attr]
        distinct_values = sorted({v for _e, v in pairs})

        prop_cache = cache.load("proposals", prop_ver)
        todo = [v for v in distinct_values if v not in prop_cache]
        if todo:
            prompts = [PROPOSAL_PROMPT.format(
                attribute_phrase=parts["attribute_phrase"],
                value=v,
                attribute_extra_rules=parts["attribute_extra_rules"],
            ) for v in todo]
            outputs = generator.generate(prompts, **generate_kwargs)
            new_rows = []
            for v, out in zip(todo, outputs):
                row = {"key": v, "raw_output": out,
                       "parsed": extract_json(out)}
                prop_cache[v] = row
                new_rows.append(row)
            cache.append("proposals", prop_ver, new_rows)
        cache.hits += len(distinct_values) - len(todo)

        # ---- Deterministic validation ------------------------------------
        valid_chains: dict[str, list[str]] = {}
        outcomes: Counter[str] = Counter()
        details: list[dict] = []
        for v in distinct_values:
            parsed = prop_cache[v].get("parsed")
            if not isinstance(parsed, dict):
                outcomes["proposal_schema_invalid"] += 1
                details.append({"value": v, "reason": "proposal_schema_invalid"})
                continue
            chain, reason = validate_proposal(attr, v, parsed)
            if chain is None:
                outcomes[reason] += 1
                details.append({"value": v, "reason": reason})
                continue
            conf = parsed.get("confidence")
            if not isinstance(conf, (int, float)) or conf < min_prop_conf:
                outcomes["low_proposal_confidence"] += 1
                details.append({"value": v, "reason": "low_proposal_confidence",
                                "proposal_confidence": conf, "chain": chain})
                continue
            valid_chains[v] = chain
            details.append({"value": v, "reason": None, "chain": chain,
                            "proposal_confidence": conf})

        # ---- Independent verification --------------------------------------
        ver_cache = cache.load("verifications", ver_ver)
        ver_todo = [v for v in valid_chains if (attr, v) not in
                    {((k.split("\x1f", 1)[0]), k.split("\x1f", 1)[1])
                     for k in ver_cache}]
        if ver_todo:
            prompts = [VERIFICATION_PROMPT.format(
                value=v, chain_json=json.dumps(valid_chains[v], ensure_ascii=False),
            ) for v in ver_todo]
            outputs = generator.generate(prompts, **generate_kwargs)
            new_rows = []
            for v, out in zip(ver_todo, outputs):
                row = {"key": f"{attr}\x1f{v}", "raw_output": out,
                       "parsed": extract_json(out)}
                ver_cache[f"{attr}\x1f{v}"] = row
                new_rows.append(row)
            cache.append("verifications", ver_ver, new_rows)
        cache.hits += len(valid_chains) - len(ver_todo)

        # ---- Ambiguity / confidence gate -----------------------------------
        accepted: dict[str, list[str]] = {}
        for v, chain in valid_chains.items():
            ver_row = ver_cache.get(f"{attr}\x1f{v}", {})
            parsed = ver_row.get("parsed")
            detail = next(d for d in details if d["value"] == v)
            if not isinstance(parsed, dict):
                outcomes["verifier_schema_invalid"] += 1
                detail["reason"] = "verifier_schema_invalid"
                continue
            ok, reason = validate_verification(chain, parsed)
            if not ok:
                outcomes[reason] += 1
                detail["reason"] = reason
                continue
            detail["verification"] = parsed
            rejected_reason = None
            if parsed["verdict"] != "accept":
                rejected_reason = "verification_rejected"
            elif parsed.get("unsupported_information"):
                rejected_reason = "unsupported_information"
            elif parsed.get("ambiguous"):
                rejected_reason = "ambiguous"
            elif float(parsed["confidence"]) < min_ver_conf:
                rejected_reason = "low_verification_confidence"
            elif not all(parsed["step_valid"]):
                rejected_reason = "verification_step_invalid"
            if rejected_reason:
                outcomes[rejected_reason] += 1
                detail["reason"] = rejected_reason
                continue
            outcomes["accepted"] += 1
            detail["reason"] = "accepted"
            accepted[v] = chain

        # ---- Per-profile outcomes ------------------------------------------
        value_reason = {d["value"]: d["reason"] for d in details}
        profile_outcomes: Counter[str] = Counter()
        for _e, v in pairs:
            profile_outcomes[value_reason[v]] += 1

        result.accepted_chains[attr] = accepted
        result.outcomes[attr] = outcomes
        result.profile_outcomes[attr] = profile_outcomes
        result.details[attr] = details

    # ---- Report + manual audit sample --------------------------------------
    rng = random.Random(seed)
    n_acc = int(config.get("audit_sample_accepted", 10))
    n_rej = int(config.get("audit_sample_rejected", 5))
    audit: list[dict] = []
    for attr, details in result.details.items():
        accepted_rows = [d for d in details if d["reason"] == "accepted"]
        rejected_rows = [d for d in details if d["reason"] not in
                         ("accepted", "model_declined")]
        picked = rng.sample(accepted_rows, min(n_acc, len(accepted_rows)))
        picked += rng.sample(rejected_rows, min(n_rej, len(rejected_rows)))
        for d in picked:
            audit.append({
                "attribute": attr,
                "value": d["value"],
                "chain": d.get("chain"),
                "pipeline_reason": d["reason"],
                "verification": d.get("verification"),
                # To be filled by the human auditor BEFORE the smoke build:
                "auditor_verdict": None,
                "auditor_notes": "",
            })

    provenance = (generator.provenance()
                  if hasattr(generator, "provenance") else {})
    total_distinct = {a: len(result.outcomes[a]) and
                      sum(result.outcomes[a].values())
                      for a in result.outcomes}
    result.report = {
        "stage": "qwen_assisted_semantic_hierarchies",
        "principle": "rejection over forced hierarchy construction",
        "generation_provenance": provenance,
        "prompt_version_proposal": prop_ver,
        "prompt_version_verification": ver_ver,
        "cache_dir": str(Path(cache_dir)),
        "cache_hits": cache.hits,
        "cache_writes": cache.writes,
        "gates": {
            "min_proposal_confidence": min_prop_conf,
            "min_verification_confidence": min_ver_conf,
        },
        "attributes": {},
        "manual_audit_sample_size": len(audit),
        "audit_status": "PENDING — fill auditor_verdict before smoke build",
    }
    for attr in attribute_values:
        n_total = sum(result.outcomes[attr].values())
        n_acc = result.outcomes[attr]["accepted"]
        result.report["attributes"][attr] = {
            "distinct_values": n_total,
            "accepted": n_acc,
            "acceptance_rate": round(n_acc / n_total, 4) if n_total else 0.0,
            "rejection_reasons": dict(sorted(
                (k, v) for k, v in result.outcomes[attr].items()
                if k != "accepted")),
            "profiles_total": sum(result.profile_outcomes[attr].values()),
            "profiles_with_accepted_hierarchy":
                result.profile_outcomes[attr]["accepted"],
        }
    result.audit_sample = audit
    return result
