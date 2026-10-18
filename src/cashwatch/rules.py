from __future__ import annotations

import json
from pathlib import Path

from pydantic import Field

from .schemas import StrictModel


class RuleSource(StrictModel):
    title: str
    url: str
    clause: str
    act_number: str = ""
    status_note: str = ""


class Rule(StrictModel):
    rule_id: str = Field(pattern=r"^[a-z0-9_\-]+$")
    title: str
    requirement: str
    required_conditions: list[str] = Field(min_length=1)
    disqualifiers: list[str] = Field(default_factory=list)
    source: RuleSource
    verification_status: str
    enabled: bool = False
    minimum_evidence_frames: int = Field(default=1, ge=1, le=3)
    notes: str = ""


class RuleRegistry(StrictModel):
    schema_version: int = 1
    rules: list[Rule]

    @classmethod
    def load(cls, path: Path) -> "RuleRegistry":
        registry = cls.model_validate(json.loads(path.read_text(encoding="utf-8")))
        ids = [rule.rule_id for rule in registry.rules]
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate rule IDs")
        if registry.schema_version != 1:
            raise ValueError("Unsupported rules schema")
        return registry

    def active(self) -> list[Rule]:
        return [r for r in self.rules if r.enabled and r.verification_status == "verified_public_text"]

