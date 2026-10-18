"""FFmpeg sampling with presentation timestamps, including MOV rotation metadata.

No frame is uploaded. `select` retains original frame timestamps (relative to the
first decoded frame); `fps` is deliberately not used because it fabricates a
regular output time grid and can duplicate images.
"""
from __future__ import annotations

import json
import math
import re
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

from .contracts import EvidenceFrame, GenerationTimeoutError, MediaError


@dataclass(frozen=True)
class SampledVideo:
    duration_sec: float
    frames: tuple[EvidenceFrame, ...]
    source_width: int
    source_height: int
    sample_fps: float


def _remaining(deadline: float, maximum: float | None = None) -> float:
    seconds = deadline - time.monotonic()
    if seconds <= 0:
        raise GenerationTimeoutError("The analysis deadline was reached.")
    return min(seconds, maximum) if maximum is not None else seconds


def _run(args: list[str], deadline: float, maximum: float | None = None) -> subprocess.CompletedProcess:
    try:
        result = subprocess.run(
            args, capture_output=True, text=True, check=False,
            timeout=_remaining(deadline, maximum),
        )
    except subprocess.TimeoutExpired as exc:
        raise GenerationTimeoutError("Media decoding exceeded the deadline.") from exc
    except OSError as exc:
        raise MediaError(f"Unable to run the local media decoder: {exc.strerror}") from exc
    if result.returncode:
        # Filenames and FFmpeg metadata can contain sensitive text; do not forward stderr.
        raise MediaError(f"Local media command failed with exit code {result.returncode}.")
    return result


def sample_video(
    video_path: Path,
    output_dir: Path,
    *,
    sample_fps: float = 1.0,
    max_image_edge: int = 768,
    deadline_monotonic: float,
    max_duration_sec: float = 180.0,
    max_sampled_frames: int = 1000,
) -> SampledVideo:
    if not math.isfinite(sample_fps) or sample_fps <= 0 or sample_fps > 30:
        raise ValueError("sample_fps must be finite and in (0, 30].")
    if not isinstance(max_image_edge, int) or max_image_edge < 64 or max_image_edge > 2048:
        raise ValueError("max_image_edge must be an integer in [64, 2048].")
    video_path = Path(video_path).resolve(strict=True)
    if not video_path.is_file():
        raise MediaError("Input must be a local regular video file.")
    for binary in ("ffmpeg", "ffprobe"):
        if not shutil.which(binary):
            raise MediaError(f"{binary} is not installed. Install FFmpeg before starting the server.")
    metadata = _run([
        "ffprobe", "-v", "error", "-protocol_whitelist", "file,pipe", "-select_streams", "v:0",
        "-show_entries", "stream=width,height,duration:format=duration", "-of", "json",
        str(video_path),
    ], deadline_monotonic, 20.0)
    try:
        document = json.loads(metadata.stdout)
        stream = document["streams"][0]
        raw_duration = stream.get("duration") or document["format"]["duration"]
        duration = float(raw_duration)
        width, height = int(stream["width"]), int(stream["height"])
    except (ValueError, KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
        raise MediaError("Cannot read a finite duration and video dimensions.") from exc
    if not math.isfinite(duration) or duration <= 0 or duration > max_duration_sec:
        raise MediaError(f"Video duration must be in (0, {max_duration_sec:g}] seconds.")
    if width <= 0 or height <= 0 or width * height > 40_000_000:
        raise MediaError("Unsupported video dimensions.")
    if math.ceil(duration * sample_fps) + 2 > max_sampled_frames:
        raise MediaError("Requested sampling would exceed the frame budget.")
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if any(output_dir.glob("frame_*.jpg")):
        raise MediaError("Use an empty evidence directory for each analysis.")
    interval = 1.0 / sample_fps
    # The comma inside select is escaped for the FFmpeg filter parser, not a shell.
    filters = (
        "setpts=PTS-STARTPTS,"
        f"select=isnan(prev_selected_t)+gte(t-prev_selected_t\\,{interval:.9f}),"
        f"scale=w='min(iw,{max_image_edge})':h='min(ih,{max_image_edge})':"
        "force_original_aspect_ratio=decrease:force_divisible_by=2,showinfo"
    )
    decoded = _run([
        "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "info", "-y",
        "-protocol_whitelist", "file,pipe", "-i", str(video_path),
        "-map", "0:v:0", "-an", "-sn", "-dn",
        "-vf", filters, "-fps_mode", "vfr", "-frames:v", str(max_sampled_frames),
        "-q:v", "3", str(output_dir / "frame_%06d.jpg"),
    ], deadline_monotonic)
    pattern = re.compile(r"\bn:\s*(\d+)\s+pts:\s*[-\d]+\s+pts_time:([-+\d.eE]+)")
    timestamps: dict[int, float] = {}
    for line in decoded.stderr.splitlines():
        if "showinfo" not in line:
            continue
        match = pattern.search(line)
        if match:
            timestamps[int(match.group(1))] = float(match.group(2))
    images = sorted(output_dir.glob("frame_*.jpg"))
    if not images or len(images) != len(timestamps):
        raise MediaError("Decoded frame files do not match their presentation timestamps.")
    frames: list[EvidenceFrame] = []
    for index, path in enumerate(images):
        timestamp = timestamps.get(index)
        if timestamp is None or not math.isfinite(timestamp) or timestamp < -0.001:
            raise MediaError("Invalid presentation timestamp in decoded video.")
        if timestamp > duration + 0.25:
            raise MediaError("Frame timestamp is inconsistent with video duration.")
        timestamp = max(0.0, timestamp)
        if frames and timestamp <= frames[-1].timestamp_sec:
            raise MediaError("Video timestamps are not strictly increasing.")
        frames.append(EvidenceFrame(f"f{index:06d}", timestamp, path.resolve()))
    return SampledVideo(duration, tuple(frames), width, height, sample_fps)


def plan_windows(
    video: SampledVideo,
    *,
    window_seconds: float = 12.0,
    overlap_seconds: float = 2.0,
    max_frames_per_window: int = 16,
) -> tuple[tuple[float, float, tuple[EvidenceFrame, ...]], ...]:
    if not 0 <= overlap_seconds < window_seconds or not math.isfinite(window_seconds):
        raise ValueError("Require 0 <= overlap_seconds < window_seconds.")
    if max_frames_per_window < 2:
        raise ValueError("At least two frames per window must be allowed.")
    windows = []
    start = 0.0
    while start < video.duration_sec:
        end = min(start + window_seconds, video.duration_sec)
        selected = tuple(f for f in video.frames if start <= f.timestamp_sec < end)
        if not selected:
            raise MediaError("A scheduled video window contains no sampled frame.")
        if len(selected) > max_frames_per_window:
            # Do not silently reduce the configured sample rate: that hides lost coverage.
            raise MediaError("Window frame budget is too small for the configured sample rate.")
        windows.append((start, end, selected))
        if end >= video.duration_sec:
            break
        start += window_seconds - overlap_seconds
    return tuple(windows)
