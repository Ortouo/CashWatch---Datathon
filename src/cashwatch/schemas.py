"""The jury endpoint deliberately keeps the case's exact completed JSON shape."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class EvidenceFrameResponse(StrictModel):
    timestamp_sec: float = Field(ge=0)
    image_base64: str = Field(min_length=4)


class ViolationResponse(StrictModel):
    type: str = Field(min_length=1, max_length=240)
    timecode_sec: float = Field(ge=0)
    description: str = Field(min_length=1, max_length=6000)
    evidence_frames: list[EvidenceFrameResponse] = Field(min_length=1, max_length=3)


class DoneResponse(StrictModel):
    job_id: str
    status: Literal["done"] = "done"
    verdict: Literal["violation", "no_violation"]
    violations: list[ViolationResponse]

    @model_validator(mode="after")
    def verdict_matches_evidence(self) -> "DoneResponse":
        if (self.verdict == "violation") != bool(self.violations):
            raise ValueError("The verdict must agree with the violations list")
        return self


class ProcessingResponse(StrictModel):
    job_id: str
    status: Literal["processing"] = "processing"


class ErrorInfo(StrictModel):
    code: str
    message: str


class FailedResponse(StrictModel):
    job_id: str
    status: Literal["failed"] = "failed"
    error: ErrorInfo

