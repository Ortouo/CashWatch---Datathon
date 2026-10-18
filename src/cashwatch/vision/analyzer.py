"""Bounded full-timeline local analysis, with explicit incomplete results."""
from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from pathlib import Path

from .contracts import (
    AnalysisResult, GenerationTimeoutError, InvalidObservationError,
    Observation, RuleSpec,
)
from .model import LocalQwenVisionModel
from .parsing import parse_observations
from .video import plan_windows, sample_video


class LocalVideoAnalyzer:
    def __init__(
        self,
        model_dir: Path,
        *,
        device: str = "auto",
        sample_fps: float = 1.0,
        window_seconds: float = 12.0,
        overlap_seconds: float = 2.0,
        max_frames_per_window: int = 16,
        max_new_tokens: int = 768,
        max_image_edge: int = 768,
        max_duration_sec: float = 180.0,
    ) -> None:
        self.model = LocalQwenVisionModel(model_dir, device=device, max_new_tokens=max_new_tokens)
        self.sample_fps = sample_fps
        self.window_seconds = window_seconds
        self.overlap_seconds = overlap_seconds
        self.max_frames_per_window = max_frames_per_window
        self.max_image_edge = max_image_edge
        self.max_duration_sec = max_duration_sec

    def warmup(self) -> None:
        """Load local weights before accepting a timed job, when the host permits it."""
        self.model.load()

    def analyze(
        self,
        video_path: Path,
        output_dir: Path,
        rules: Sequence[RuleSpec],
        *,
        deadline_monotonic: float,
        progress: Callable[[float, str], None] | None = None,
    ) -> AnalysisResult:
        # Verify/load weights before decoding. Missing weights are a failed job, not a
        # simulated detector, motion heuristic, or negative prediction.
        self.model.load()
        if time.monotonic() >= deadline_monotonic:
            raise GenerationTimeoutError("The deadline expired during model initialization.")
        if len({rule.rule_id for rule in rules}) != len(rules):
            raise ValueError("Rule IDs must be unique.")
        if progress:
            progress(0.03, "decoding")
        sampled = sample_video(
            video_path, Path(output_dir) / "frames", sample_fps=self.sample_fps,
            max_image_edge=self.max_image_edge, deadline_monotonic=deadline_monotonic,
            max_duration_sec=self.max_duration_sec,
        )
        windows = plan_windows(
            sampled, window_seconds=self.window_seconds, overlap_seconds=self.overlap_seconds,
            max_frames_per_window=self.max_frames_per_window,
        )
        warnings = [
            f"Sparse visual sampling at up to {self.sample_fps:g} fps: actions between samples "
            "may be missed. Complete means every scheduled window was processed, not that "
            "every source frame was inspected.",
            "This model has not been calibrated on the ForteBank competition videos in this release.",
        ]
        if not rules:
            warnings.append("No rule specifications were supplied; observations cannot establish a rule violation.")
        observations: list[Observation] = []
        completed_intervals: list[tuple[float, float]] = []
        complete = True
        for index, (start, end, frames) in enumerate(windows):
            if time.monotonic() >= deadline_monotonic:
                complete = False
                warnings.append(f"Deadline reached before window {index}; some windows were not analyzed.")
                break
            if progress:
                progress(0.10 + 0.85 * index / len(windows), f"analyzing_window_{index + 1}_of_{len(windows)}")
            try:
                raw = self.model.observe(frames, rules, deadline_monotonic=deadline_monotonic)
                window_observations = parse_observations(raw, frames, rules, window_id=f"w{index:04d}")
            except GenerationTimeoutError:
                complete = False
                warnings.append(f"Deadline reached during window {index}; its output was discarded.")
                break
            except InvalidObservationError as exc:
                complete = False
                warnings.append(f"Window {index} failed structured validation: {exc}")
                # An invalid response is never replaced with an empty successful response.
                continue
            observations.extend(window_observations)
            completed_intervals.append((start, end))
        # Overlap can produce exact copies of an event. Merge only identical evidence
        # and rule/assessment/condition sets; distinct events are never merged by prose.
        unique: dict[tuple, Observation] = {}
        for observation in observations:
            key = (
                observation.rule_id, observation.assessment,
                tuple(sorted(observation.evidence_frame_ids)),
                tuple(sorted((k, tuple(sorted(v))) for k, v in observation.condition_evidence.items())),
            )
            if key not in unique:
                unique[key] = observation
        if progress:
            progress(0.98, "analysis_complete" if complete else "analysis_incomplete")
        return AnalysisResult(
            observations=tuple(unique.values()), evidence_frames=sampled.frames,
            duration_sec=sampled.duration_sec, analyzed_intervals=tuple(completed_intervals),
            complete=complete and len(completed_intervals) == len(windows),
            warnings=tuple(warnings), model_id=self.model.model_id, backend=self.model.backend,
            sample_fps=self.sample_fps,
        )
