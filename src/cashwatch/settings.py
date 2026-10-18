from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    api_key: str
    data_dir: Path = Path("runtime")
    rules_path: Path = Path("config/rules.json")
    model_dir: Path = Path("models/Qwen3-VL-2B-Instruct")
    model_id: str = "Qwen/Qwen3-VL-2B-Instruct"
    device: str = "auto"
    analysis_timeout_sec: float = 275.0
    max_inflight: int = 1
    max_upload_mb: int = 100
    max_duration_sec: float = 180.0
    retention_hours: float = 1.0
    sample_fps: float = 1.0
    window_seconds: float = 12.0
    overlap_seconds: float = 2.0
    max_frames_per_window: int = 16
    max_new_tokens: int = 768
    max_image_edge: int = 768
    accepted_extensions: tuple[str, ...] = (".mov", ".mp4")

    def __post_init__(self) -> None:
        if self.api_key and len(self.api_key) < 24:
            raise ValueError("CASHWATCH_API_KEY must have at least 24 characters")
        if not (0 < self.analysis_timeout_sec <= 285):
            raise ValueError("CASHWATCH_TIMEOUT_SEC must be in (0, 285]")
        if not (1 <= self.max_inflight <= 8):
            raise ValueError("CASHWATCH_MAX_INFLIGHT must be in [1, 8]")
        if not (1 <= self.max_upload_mb <= 1024):
            raise ValueError("CASHWATCH_MAX_UPLOAD_MB must be in [1, 1024]")
        if self.max_duration_sec <= 0 or self.retention_hours <= 0:
            raise ValueError("Duration and retention must be positive")
        if self.sample_fps <= 0 or self.window_seconds <= 0:
            raise ValueError("Sampling settings must be positive")
        if not (0 <= self.overlap_seconds < self.window_seconds):
            raise ValueError("Overlap must be smaller than the window")
        if self.max_frames_per_window < 2 or self.max_new_tokens < 64:
            raise ValueError("Insufficient model context budget")

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024

    @classmethod
    def from_env(cls) -> "Settings":
        from dotenv import load_dotenv

        load_dotenv(override=False)
        return cls(
            api_key=os.getenv("CASHWATCH_API_KEY", ""),
            data_dir=Path(os.getenv("CASHWATCH_DATA_DIR", "runtime")).resolve(),
            rules_path=Path(os.getenv("CASHWATCH_RULES_PATH", "config/rules.json")).resolve(),
            model_dir=Path(os.getenv("CASHWATCH_MODEL_DIR", "models/Qwen3-VL-2B-Instruct")).resolve(),
            model_id=os.getenv("CASHWATCH_MODEL_ID", "Qwen/Qwen3-VL-2B-Instruct"),
            device=os.getenv("CASHWATCH_DEVICE", "auto"),
            analysis_timeout_sec=float(os.getenv("CASHWATCH_TIMEOUT_SEC", "275")),
            max_inflight=int(os.getenv("CASHWATCH_MAX_INFLIGHT", "1")),
            max_upload_mb=int(os.getenv("CASHWATCH_MAX_UPLOAD_MB", "100")),
            max_duration_sec=float(os.getenv("CASHWATCH_MAX_DURATION_SEC", "180")),
            retention_hours=float(os.getenv("CASHWATCH_RETENTION_HOURS", "1")),
            sample_fps=float(os.getenv("CASHWATCH_SAMPLE_FPS", "1")),
            window_seconds=float(os.getenv("CASHWATCH_WINDOW_SEC", "12")),
            overlap_seconds=float(os.getenv("CASHWATCH_OVERLAP_SEC", "2")),
            max_frames_per_window=int(os.getenv("CASHWATCH_MAX_FRAMES_PER_WINDOW", "16")),
            max_new_tokens=int(os.getenv("CASHWATCH_MAX_NEW_TOKENS", "768")),
            max_image_edge=int(os.getenv("CASHWATCH_IMAGE_EDGE", "768")),
        )

