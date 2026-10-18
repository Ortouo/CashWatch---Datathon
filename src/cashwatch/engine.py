"""Ground model observations in a reviewed registry and actual local JPEGs."""
from __future__ import annotations

import base64
import math
from pathlib import Path

from PIL import Image

from .rules import Rule, RuleRegistry
from .schemas import DoneResponse, EvidenceFrameResponse, ViolationResponse
from .vision.contracts import AnalysisResult, EvidenceFrame, Observation, VisionError


class IncompleteAnalysisError(VisionError):
    pass


def _read_jpeg(frame: EvidenceFrame, evidence_root: Path) -> str:
    path = frame.image_path.resolve()
    if not path.is_relative_to(evidence_root.resolve()):
        raise VisionError("Evidence path is outside this job")
    data = path.read_bytes()
    if len(data) > 4 * 1024 * 1024:
        raise VisionError("Evidence JPEG is too large")
    with Image.open(path) as image:
        if image.format != "JPEG":
            raise VisionError("Evidence must be JPEG")
        image.verify()
    return base64.b64encode(data).decode("ascii")


def _supported(observation: Observation, rule: Rule, frames: dict[str, EvidenceFrame]) -> tuple[bool, str]:
    if observation.assessment != "supported" or observation.uncertainty:
        return False, "Не все визуальные условия подтверждены"
    evidence_ids = observation.evidence_frame_ids
    if len(set(evidence_ids)) < rule.minimum_evidence_frames or len(evidence_ids) > 3 or len(set(evidence_ids)) != len(evidence_ids):
        raise VisionError("A supported event has an invalid evidence count")
    if any(frame_id not in frames for frame_id in evidence_ids):
        raise VisionError("Unknown evidence frame ID")
    for index, condition in enumerate(rule.required_conditions):
        refs = observation.condition_evidence.get(f"c{index + 1}", ())
        if not refs or any(frame_id not in evidence_ids for frame_id in refs):
            raise VisionError("A supported event is missing condition evidence")
    if any(not math.isfinite(t) for t in (observation.start_sec, observation.end_sec)):
        raise VisionError("Invalid observation timestamp")
    if observation.start_sec < 0 or observation.end_sec < observation.start_sec:
        raise VisionError("Invalid observation interval")
    if not observation.description.strip():
        raise VisionError("Empty observation description")
    timestamps = [frames[frame_id].timestamp_sec for frame_id in evidence_ids]
    if abs(min(timestamps) - observation.start_sec) > 0.0001 or abs(max(timestamps) - observation.end_sec) > 0.0001:
        raise VisionError("Observation boundaries must be actual evidence timestamps")
    return True, ""


def _validate_coverage(result: AnalysisResult) -> None:
    if not result.evidence_frames or not result.analyzed_intervals:
        raise IncompleteAnalysisError("No sampled evidence or processed intervals")
    if not math.isfinite(result.sample_fps) or result.sample_fps <= 0:
        raise VisionError("Invalid sample rate")
    until = 0.0
    gaps = False
    for start, end in sorted(result.analyzed_intervals):
        if not all(math.isfinite(t) for t in (start, end)) or start < 0 or end <= start or end > result.duration_sec + 0.05:
            raise VisionError("Invalid analyzed interval")
        if start > until + 0.05:
            gaps = True
        until = max(until, end)
    if result.complete and (gaps or until < result.duration_sec - 0.05):
        raise IncompleteAnalysisError("The completeness flag contradicts the analyzed intervals")


def build_report(
    job_id: str,
    result: AnalysisResult,
    registry: RuleRegistry,
    evidence_root: Path,
) -> tuple[dict, dict]:
    rules = {rule.rule_id: rule for rule in registry.active()}
    if not rules:
        raise IncompleteAnalysisError("No reviewed rules have been enabled")
    if not math.isfinite(result.duration_sec) or result.duration_sec <= 0:
        raise VisionError("Invalid media duration")
    _validate_coverage(result)
    frames = {frame.frame_id: frame for frame in result.evidence_frames}
    if len(frames) != len(result.evidence_frames):
        raise VisionError("Duplicate evidence IDs")
    for frame in frames.values():
        if not math.isfinite(frame.timestamp_sec) or not (0 <= frame.timestamp_sec <= result.duration_sec + 0.05):
            raise VisionError("Evidence time is outside the video")

    candidates: list[tuple[Observation, Rule]] = []
    review_items: list[dict] = []
    for observation in result.observations:
        if observation.end_sec > result.duration_sec + 0.05:
            raise VisionError("Observation time is outside the video")
        rule = rules.get(observation.rule_id or "")
        if rule is None:
            if observation.assessment == "supported":
                raise VisionError("Supported observation refers to an unapproved rule")
            review_items.append({"observation_id": observation.observation_id, "reason": "Нет включённого подтверждённого правила", "description": observation.description})
            continue
        valid, reason = _supported(observation, rule, frames)
        if not valid:
            review_items.append({"observation_id": observation.observation_id, "reason": reason, "description": observation.description})
            continue
        # Merge only identical event boundaries and evidence. Two different events
        # may share a context frame and must still be returned separately.
        duplicate = next((i for i, (old, old_rule) in enumerate(candidates)
                          if old_rule.rule_id == rule.rule_id
                          and set(old.evidence_frame_ids) == set(observation.evidence_frame_ids)
                          and old.start_sec == observation.start_sec
                          and old.end_sec == observation.end_sec), None)
        if duplicate is not None:
            old = candidates[duplicate][0]
            if len(observation.evidence_frame_ids) > len(old.evidence_frame_ids):
                candidates[duplicate] = (observation, rule)
        else:
            candidates.append((observation, rule))

    candidates.sort(key=lambda item: item[0].start_sec)
    violations: list[ViolationResponse] = []
    events: list[dict] = []
    for observation, rule in candidates:
        evidence = sorted((frames[frame_id] for frame_id in observation.evidence_frame_ids), key=lambda f: f.timestamp_sec)
        anchor = min(evidence, key=lambda frame: abs(frame.timestamp_sec - observation.start_sec))
        # Do not invent a frame at an arbitrary generated timestamp.
        source = rule.source
        description = (
            f"{observation.description.strip()} Возможное несоответствие требует проверки человеком. "
            f"Источник: {source.title}, {source.clause}. {source.url}"
        )
        violations.append(ViolationResponse(
            type=rule.title,
            timecode_sec=round(anchor.timestamp_sec, 6),
            description=description,
            evidence_frames=[EvidenceFrameResponse(timestamp_sec=round(frame.timestamp_sec, 6), image_base64=_read_jpeg(frame, evidence_root)) for frame in evidence],
        ))
        events.append({
            "rule_id": rule.rule_id, "title": rule.title, "source": source.model_dump(),
            "observation_id": observation.observation_id,
            "start_sec": observation.start_sec, "end_sec": observation.end_sec,
            "condition_evidence": {key: list(value) for key, value in observation.condition_evidence.items()},
        })

    if not result.complete and not violations:
        raise IncompleteAnalysisError("Video analysis is incomplete; a clean verdict would be unsupported")
    if review_items and not violations:
        raise IncompleteAnalysisError("Unresolved observations prevent a clean verdict")
    warnings = list(result.warnings)
    if not result.complete:
        warnings.append("Часть видео не обработана. Показаны только подтверждённые кандидаты; проверка всего видео не завершена.")
    if review_items:
        warnings.append("Есть наблюдения с неподтверждённой применимостью правила; они не включены в список нарушений.")
    done = DoneResponse(job_id=job_id, verdict="violation" if violations else "no_violation", violations=violations)
    details = {
        "duration_sec": result.duration_sec, "model_id": result.model_id,
        "backend": result.backend, "complete": result.complete,
        "sample_fps": result.sample_fps, "analyzed_intervals": result.analyzed_intervals,
        "warnings": warnings, "events": events, "review_items": review_items,
        "rules_checked": list(rules), "quality_validated_on_dataset": False,
    }
    return done.model_dump(), details

