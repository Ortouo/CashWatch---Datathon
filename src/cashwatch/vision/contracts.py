"""Local ML contracts. Evidence paths never leave the host through this module."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


@dataclass(frozen=True)
class RuleSpec:
    rule_id: str
    title: str
    requirement: str
    required_conditions: tuple[str, ...] = ()
    disqualifiers: tuple[str, ...] = ()


@dataclass(frozen=True)
class EvidenceFrame:
    frame_id: str
    timestamp_sec: float
    image_path: Path


@dataclass(frozen=True)
class Observation:
    observation_id: str
    rule_id: str | None
    start_sec: float
    end_sec: float
    description: str
    evidence_frame_ids: tuple[str, ...]
    assessment: Literal["supported", "uncertain"]
    condition_evidence: dict[str, tuple[str, ...]] = field(default_factory=dict)
    uncertainty: str | None = None


@dataclass(frozen=True)
class AnalysisResult:
    observations: tuple[Observation, ...]
    evidence_frames: tuple[EvidenceFrame, ...]
    duration_sec: float
    analyzed_intervals: tuple[tuple[float, float], ...]
    complete: bool
    warnings: tuple[str, ...]
    model_id: str
    backend: str
    sample_fps: float = 1.0


class VisionError(RuntimeError):
    """An analysis error that must not be converted to a clean verdict."""


class MediaError(VisionError):
    pass


class ModelUnavailableError(VisionError):
    pass


class InvalidObservationError(VisionError):
    pass


class GenerationTimeoutError(VisionError):
    pass
