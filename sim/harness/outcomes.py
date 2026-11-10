"""
sim/harness/outcomes.py

Outcome and telemetry extraction for the sim harness.

Parses a reviewer persona's response text (LLMResult.text) into the structured
outcome fields the archive schema declares: decision, corrections_count, and
rationale_text_length_chars. Tolerant by design: a response it cannot parse
yields honest nulls and parse_ok=False rather than a crash or a fabricated value.

================================================================================
AGENT-OUTPUT CONTRACT (reviewer persona -> harness)
================================================================================

This is the contract the reviewer personas must satisfy so the harness can read
a verdict back out. Persona prompts are written to emit one of the two forms
below. The extractor tries the primary form first, then the fallback.

RATIFIED 2026-07-22 (Hassler): field mapping confirmed against her loader
(decision -> trial_end.decision, corrections_count -> trial_end.corrections_count,
rationale_text_length_chars -> agent_response_received.rationale_text_length_chars).
Personas may be frozen against this contract. The synonym map (see
_DECISION_SYNONYMS) and corrections-counting rule are unchanged from proposal.

--------------------------------------------------------------------------------
PRIMARY FORM (preferred): a fenced JSON object
--------------------------------------------------------------------------------

The persona ends its response with a fenced code block tagged json containing a
single object:

    ```json
    {
      "decision": "reject",
      "corrections": [
        "Citation 3 does not support the claim in paragraph 2.",
        "The sample size is misreported in Table 1."
      ],
      "rationale": "The submission overstates its evidence base ..."
    }
    ```

Field rules:
  - decision     : required. A string mapped through _DECISION_SYNONYMS to one
                   of the schema-allowed values: accept | reject | revise.
  - corrections  : optional. A JSON array; corrections_count is its length.
                   Absent or null -> corrections_count = 0.
  - rationale    : optional. A string; rationale_text_length_chars is its
                   character length. Absent -> rationale length falls back to
                   the full response text length (honest upper bound).

The fence tag is honored when present; a bare ``` fence or a raw top-level JSON
object is also accepted, so the persona is not brittle to fence wording.

--------------------------------------------------------------------------------
FALLBACK FORM: clearly-labeled lines
--------------------------------------------------------------------------------

If no JSON object parses, the extractor looks for labeled lines (case-insensitive
label, value after the colon):

    DECISION: revise
    CORRECTIONS: 2
    RATIONALE: The argument holds but two citations need to be corrected ...

Field rules for the fallback:
  - DECISION    : mapped through _DECISION_SYNONYMS as above.
  - CORRECTIONS : if the value is an integer, that integer is the count. If it
                  is prose, the count is the number of "correction" token
                  occurrences in the whole response (legacy heuristic, kept so
                  earlier behavior is preserved). Absent -> count from the whole
                  response text the same way.
  - RATIONALE   : character length of the labeled rationale text. Absent ->
                  length of the whole response text.

--------------------------------------------------------------------------------
UNPARSEABLE
--------------------------------------------------------------------------------

If neither form yields a decision, the result is parse_ok=False with decision
None and corrections_count None. rationale_text_length_chars is still set to the
response text length when any text is present (that length is always honestly
knowable), or None for empty text. The runner archives these honest nulls; it
never fabricates a decision.

Deliberately NOT extracted here: evidence_expansion_requests and
planning_trace_depth. Those are interaction-model signals that depend on an
experiment interaction design that does not exist yet. See the runner's
TODO(hassler/experiment-design). This module leaves them untouched.
"""

from __future__ import annotations

import json
import re
from typing import Optional

from pydantic import BaseModel


# Schema-allowed decision values (authoritative: TrialEndEvent.decision doc in
# models.py and trial_archive_schema.md both list accept | reject | revise).
ALLOWED_DECISIONS = ("accept", "reject", "revise")

# Common synonyms mapped to the allowed set. Keys are lowercased bare words the
# parser looks for; values are the canonical decision. Kept small and explicit
# so persona wording and parser behavior are easy to diff.
_DECISION_SYNONYMS = {
    # accept
    "accept": "accept",
    "accepted": "accept",
    "approve": "accept",
    "approved": "accept",
    "pass": "accept",
    "endorse": "accept",
    # reject
    "reject": "reject",
    "rejected": "reject",
    "decline": "reject",
    "declined": "reject",
    "deny": "reject",
    "fail": "reject",
    # revise
    "revise": "revise",
    "revision": "revise",
    "revisions": "revise",
    "rerevise": "revise",
    "major revision": "revise",
    "minor revision": "revise",
    "resubmit": "revise",
    "amend": "revise",
    "rework": "revise",
}


class OutcomeExtraction(BaseModel):
    """
    Parsed outcome fields from a single response.

    decision : one of accept | reject | revise, or None when unparseable.
    corrections_count : item count, or None when unparseable.
    rationale_text_length_chars : char length of the rationale (or whole text as
        an honest fallback), or None when there is no text at all.
    parse_ok : True only when a decision was recovered. False signals the runner
        to archive honest nulls instead of fabricated values.
    """
    decision: Optional[str] = None
    corrections_count: Optional[int] = None
    rationale_text_length_chars: Optional[int] = None
    parse_ok: bool = False


def _normalize_decision(raw: Optional[str]) -> Optional[str]:
    """Map a raw decision string to the allowed set, or None if no match."""
    if not raw:
        return None
    candidate = raw.strip().lower()
    candidate = candidate.strip(".:;,!\"'`()[]{} \t")
    if candidate in _DECISION_SYNONYMS:
        return _DECISION_SYNONYMS[candidate]
    # Allow a leading token like "reject - the claims ..." or "major revision."
    # Scan multi-word keys first (e.g. "major revision"), then single words.
    for key in sorted(_DECISION_SYNONYMS, key=len, reverse=True):
        if re.search(rf"\b{re.escape(key)}\b", candidate):
            return _DECISION_SYNONYMS[key]
    return None


def _count_correction_tokens(text: str) -> int:
    """Legacy heuristic: count 'correction' token occurrences in the text."""
    return len(re.findall(r"\bcorrection", text.lower()))


_JSON_FENCE_RE = re.compile(
    r"```(?:json)?\s*(\{.*?\})\s*```",
    re.DOTALL | re.IGNORECASE,
)


def _find_json_object(text: str) -> Optional[dict]:
    """
    Find and parse the first JSON object in the text. Tries fenced blocks first,
    then a bare top-level {...} span. Returns the dict, or None if nothing parses.
    """
    for match in _JSON_FENCE_RE.finditer(text):
        try:
            obj = json.loads(match.group(1))
        except (json.JSONDecodeError, ValueError):
            continue
        if isinstance(obj, dict):
            return obj
    # Bare object fallback: scan from each '{' to a matching close brace.
    start = text.find("{")
    while start != -1:
        depth = 0
        for i in range(start, len(text)):
            ch = text[i]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    snippet = text[start:i + 1]
                    try:
                        obj = json.loads(snippet)
                    except (json.JSONDecodeError, ValueError):
                        break
                    if isinstance(obj, dict):
                        return obj
                    break
        start = text.find("{", start + 1)
    return None


def _extract_from_json(obj: dict, full_text: str) -> Optional[OutcomeExtraction]:
    """Build an extraction from a parsed JSON verdict object, or None if no decision."""
    decision = _normalize_decision(obj.get("decision"))
    if decision is None:
        return None

    corrections = obj.get("corrections")
    if isinstance(corrections, list):
        corrections_count = len(corrections)
    elif corrections is None:
        corrections_count = 0
    elif isinstance(corrections, int) and not isinstance(corrections, bool):
        corrections_count = corrections
    else:
        # A single non-list correction value counts as one.
        corrections_count = 1

    rationale = obj.get("rationale")
    if isinstance(rationale, str):
        rationale_len = len(rationale)
    else:
        rationale_len = len(full_text) if full_text else None

    return OutcomeExtraction(
        decision=decision,
        corrections_count=corrections_count,
        rationale_text_length_chars=rationale_len,
        parse_ok=True,
    )


_LABEL_RE = {
    "decision": re.compile(r"^\s*decision\s*:\s*(.+?)\s*$", re.IGNORECASE | re.MULTILINE),
    "corrections": re.compile(r"^\s*corrections?\s*:\s*(.+?)\s*$", re.IGNORECASE | re.MULTILINE),
    "rationale": re.compile(r"^\s*rationale\s*:\s*(.+?)\s*$", re.IGNORECASE | re.MULTILINE | re.DOTALL),
}


def _extract_from_labels(text: str) -> Optional[OutcomeExtraction]:
    """Build an extraction from labeled lines, or None if no decision label parses."""
    dmatch = _LABEL_RE["decision"].search(text)
    decision = _normalize_decision(dmatch.group(1)) if dmatch else None
    if decision is None:
        return None

    cmatch = _LABEL_RE["corrections"].search(text)
    if cmatch:
        value = cmatch.group(1).strip()
        int_match = re.match(r"^(\d+)\b", value)
        if int_match:
            corrections_count = int(int_match.group(1))
        else:
            corrections_count = _count_correction_tokens(text)
    else:
        corrections_count = _count_correction_tokens(text)

    rmatch = _LABEL_RE["rationale"].search(text)
    if rmatch:
        rationale_len = len(rmatch.group(1).strip())
    else:
        rationale_len = len(text) if text else None

    return OutcomeExtraction(
        decision=decision,
        corrections_count=corrections_count,
        rationale_text_length_chars=rationale_len,
        parse_ok=True,
    )


# Matches a "decision": "..." field on its own, so a decision survives even when
# the surrounding JSON object was truncated before its closing brace or fence
# (e.g. the response was cut off at max_tokens mid-object).
_PARTIAL_DECISION_RE = re.compile(r'"decision"\s*:\s*"([^"]+)"', re.IGNORECASE)


def _extract_from_partial_json(text: str) -> Optional[OutcomeExtraction]:
    """
    Last-resort recovery from a truncated JSON verdict, or None if no decision.

    When the JSON object is cut off before it closes, neither the JSON parser nor
    the label fallback fire, but the "decision" field itself is often already on
    the wire. Match it directly so a truncated verdict still yields a decision.
    corrections_count falls back to token counting and rationale length to the
    text length, since the object's structured fields cannot be trusted once it
    is truncated.
    """
    match = _PARTIAL_DECISION_RE.search(text)
    decision = _normalize_decision(match.group(1)) if match else None
    if decision is None:
        return None

    return OutcomeExtraction(
        decision=decision,
        corrections_count=_count_correction_tokens(text),
        rationale_text_length_chars=len(text) if text else None,
        parse_ok=True,
    )


def extract_outcome(text: Optional[str]) -> OutcomeExtraction:
    """
    Parse a response into an OutcomeExtraction.

    Order: fenced/bare JSON verdict (primary), then labeled lines, then a
    truncated-JSON "decision" field (last resort). On total failure, returns
    parse_ok=False with decision and corrections_count None, and
    rationale_text_length_chars set to the text length when any text is present
    (always honestly knowable) or None for empty/None input.
    """
    if not text:
        return OutcomeExtraction(
            decision=None,
            corrections_count=None,
            rationale_text_length_chars=None,
            parse_ok=False,
        )

    obj = _find_json_object(text)
    if obj is not None:
        result = _extract_from_json(obj, text)
        if result is not None:
            return result

    result = _extract_from_labels(text)
    if result is not None:
        return result

    result = _extract_from_partial_json(text)
    if result is not None:
        return result

    # Unparseable: honest nulls, but text length is always knowable.
    return OutcomeExtraction(
        decision=None,
        corrections_count=None,
        rationale_text_length_chars=len(text),
        parse_ok=False,
    )
