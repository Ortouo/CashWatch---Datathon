"""Validate structured local VLM output and bind it to actual decoded frames."""
from __future__ import annotations

import json
from collections.abc import Sequence

from .contracts import EvidenceFrame, InvalidObservationError, Observation, RuleSpec


EVENT_KEYS = {
    "rule_id", "start_frame_id", "end_frame_id", "description", "evidence_frame_ids",
    "assessment", "condition_evidence", "uncertainty",
}


def _fail(message: str) -> None:
    raise InvalidObservationError(message)


def parse_observations(
    raw_text: str,
    frames: Sequence[EvidenceFrame],
    rules: Sequence[RuleSpec],
    *,
    window_id: str,
) -> tuple[Observation, ...]:
    if len(raw_text) > 100_000:
        _fail("Model JSON exceeds the response budget.")
    try:
        # Reject duplicate keys instead of accepting a model's contradictory fields.
        def strict_object(pairs):
            result = {}
            for key, value in pairs:
                if key in result:
                    _fail("Model JSON contains a duplicate key.")
                result[key] = value
            return result
        value = json.loads(
            raw_text, object_pairs_hook=strict_object,
            parse_constant=lambda value: _fail("Model JSON contains a non-finite number."),
        )
    except (json.JSONDecodeError, TypeError) as exc:
        raise InvalidObservationError("Model did not return valid JSON.") from exc
    if not isinstance(value, dict) or set(value) != {"observations"}:
        _fail("Expected a JSON object with only the observations field.")
    events = value["observations"]
    if not isinstance(events, list) or len(events) > 8:
        _fail("observations must be a list of at most eight events per window.")
    available = {frame.frame_id: frame for frame in frames}
    rule_map = {rule.rule_id: rule for rule in rules}
    observations = []
    for index, event in enumerate(events):
        if not isinstance(event, dict) or set(event) != EVENT_KEYS:
            _fail("Observation fields do not match the required schema.")
        rule_id = event["rule_id"]
        if rule_id is not None and (not isinstance(rule_id, str) or rule_id not in rule_map):
            _fail("Observation references a rule that was not provided.")
        assessment = event["assessment"]
        if assessment not in ("supported", "uncertain"):
            _fail("Invalid observation assessment.")
        if assessment == "supported" and rule_id is None:
            _fail("An untyped observation cannot support a rule violation.")
        start_id, end_id = event["start_frame_id"], event["end_frame_id"]
        if not all(isinstance(key, str) and key in available for key in (start_id, end_id)):
            _fail("Observation invents an unavailable boundary frame.")
        start, end = available[start_id].timestamp_sec, available[end_id].timestamp_sec
        if end < start:
            _fail("Observation has reversed temporal boundaries.")
        evidence = event["evidence_frame_ids"]
        if not isinstance(evidence, list) or not 1 <= len(evidence) <= 3:
            _fail("Each observation must have one to three evidence frames.")
        if any(not isinstance(key, str) or key not in available for key in evidence):
            _fail("Observation invents an unavailable evidence frame.")
        if len(set(evidence)) != len(evidence):
            _fail("Evidence frame IDs must be unique.")
        if start_id not in evidence or end_id not in evidence:
            _fail("Boundary frames must be included in the submitted evidence.")
        if any(not start <= available[key].timestamp_sec <= end for key in evidence):
            _fail("Evidence falls outside the event boundaries.")
        description = event["description"]
        if not isinstance(description, str) or not 1 <= len(description.strip()) <= 1200:
            _fail("Observation needs a nonempty concise factual description.")
        uncertainty = event["uncertainty"]
        if uncertainty is not None and (not isinstance(uncertainty, str) or len(uncertainty) > 1000):
            _fail("uncertainty must be null or a concise string.")
        if assessment == "uncertain" and not (uncertainty and uncertainty.strip()):
            _fail("An uncertain observation must explain the missing evidence.")
        if assessment == "supported" and uncertainty is not None:
            _fail("A supported observation cannot also declare unresolved uncertainty.")
        conditions = event["condition_evidence"]
        if not isinstance(conditions, dict):
            _fail("condition_evidence must be a JSON object.")
        required = set(rule_map[rule_id].required_conditions) if rule_id else set()
        if set(conditions) - required:
            _fail("Condition evidence references an unknown rule condition.")
        normalized_conditions: dict[str, tuple[str, ...]] = {}
        for condition, frame_ids in conditions.items():
            if not isinstance(frame_ids, list) or not frame_ids:
                _fail("Each condition needs at least one evidence frame.")
            if any(not isinstance(key, str) or key not in evidence for key in frame_ids):
                _fail("Condition evidence must be included in the event evidence frames.")
            if len(set(frame_ids)) != len(frame_ids):
                _fail("Condition evidence must not duplicate frame IDs.")
            normalized_conditions[condition] = tuple(frame_ids)
        if assessment == "supported" and set(conditions) != required:
            _fail("A required visual condition has not been supported by evidence.")
        observations.append(Observation(
            observation_id=f"{window_id}_e{index:02d}", rule_id=rule_id,
            start_sec=start, end_sec=end, description=description.strip(),
            evidence_frame_ids=tuple(evidence), assessment=assessment,
            condition_evidence=normalized_conditions, uncertainty=uncertainty,
        ))
    return tuple(observations)
