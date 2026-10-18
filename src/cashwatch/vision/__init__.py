"""Offline visual observation backend for CashWatch."""
from .analyzer import LocalVideoAnalyzer
from .contracts import (
    AnalysisResult, EvidenceFrame, GenerationTimeoutError, InvalidObservationError,
    MediaError, ModelUnavailableError, Observation, RuleSpec, VisionError,
)

__all__ = [
    "AnalysisResult", "EvidenceFrame", "GenerationTimeoutError", "InvalidObservationError",
    "LocalVideoAnalyzer", "MediaError", "ModelUnavailableError", "Observation", "RuleSpec", "VisionError",
]
