"""Comparable, bounded runtime identity; never a claim of weight-file integrity.

Only declared checkpoint hashes and package source files are read. A complete
contract means its identity fields are available, not that weights were loaded,
rules enabled, a GPU fitted the model, or inference quality was validated.
Source files are observed on disk; restart the service after changing code.
Neither this snapshot nor a before/after comparison is an atomic file lock.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

from .settings import Settings

_PACKAGE_ROOT = Path(__file__).resolve().parent
_JSON_LIMIT = 2 * 1024 * 1024
_MAX_FILES = 1024
_SOURCE_FILE_LIMIT = 2 * 1024 * 1024
_SOURCE_TOTAL_LIMIT = 32 * 1024 * 1024
_FLOAT_FIELDS = (
    "sample_fps", "window_seconds", "overlap_seconds", "max_duration_sec",
    "analysis_timeout_sec",
)
_INT_FIELDS = (
    "max_frames_per_window", "max_new_tokens", "max_image_edge", "max_inflight", "max_upload_mb",
)
_SETTINGS_FIELDS = ("device", *_FLOAT_FIELDS, *_INT_FIELDS)
_MODEL_FIELDS = ("model_id", "revision", "declared_files_sha256")
_IDENTITY_FIELDS = ("schema_version", "settings", "model", "rules_sha256", "pipeline_sha256")
_CONTRACT_FIELDS = (*_IDENTITY_FIELDS, "fingerprint", "complete", "issues", "weights_hashes_verified")
_ISSUES = {
    "settings": "runtime_settings_invalid",
    "model": "model_provenance_unavailable",
    "rules_sha256": "rules_identity_unavailable",
    "pipeline_sha256": "pipeline_identity_unavailable",
}


def _invalid() -> ValueError:
    return ValueError("runtime_contract_invalid")


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("utf-8")


def _fingerprint(identity: dict) -> str:
    return hashlib.sha256(_canonical_bytes(identity)).hexdigest()


def _unique_pairs(pairs: list[tuple[str, Any]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise _invalid()
        result[key] = value
    return result


def _read_bytes(path: Path, limit: int) -> bytes:
    if not path.is_file() or path.stat().st_size > limit:
        raise _invalid()
    with path.open("rb") as stream:
        data = stream.read(limit + 1)
    if len(data) > limit:
        raise _invalid()
    return data


def _decode_json(data: bytes) -> Any:
    def reject_constant(_value: str) -> None:
        raise _invalid()

    return json.loads(data, object_pairs_hook=_unique_pairs, parse_constant=reject_constant)


def _sha256(value: Any, *, nullable: bool = False) -> str | None:
    if value is None and nullable:
        return None
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", value):
        raise _invalid()
    return value.lower()


def _relative_model_name(value: Any) -> str:
    if not isinstance(value, str) or not 1 <= len(value) <= 512:
        raise _invalid()
    # The downloader can produce Windows separators; the contract uses POSIX.
    name = value.replace("\\", "/")
    if not re.fullmatch(r"[A-Za-z0-9_.\-/]+", name) or any(part in {"", ".", ".."} for part in name.split("/")):
        raise _invalid()
    return name


def _normalize_file_hashes(payload: Any) -> dict[str, str]:
    if not isinstance(payload, dict) or not 1 <= len(payload) <= _MAX_FILES:
        raise _invalid()
    result = {}
    for key, value in payload.items():
        name = _relative_model_name(key)
        if name in result:
            raise _invalid()
        result[name] = _sha256(value)
    if not {"config.json", "tokenizer_config.json"}.issubset(result) or not any(name.endswith(".safetensors") for name in result):
        raise _invalid()
    return dict(sorted(result.items()))


def _empty_model() -> dict:
    return {"model_id": None, "revision": None, "declared_files_sha256": {}}


def _normalize_model(payload: Any) -> dict:
    if not isinstance(payload, dict) or set(payload) != set(_MODEL_FIELDS):
        raise _invalid()
    if payload == _empty_model():
        return _empty_model()
    model_id, revision = payload["model_id"], payload["revision"]
    if not isinstance(model_id, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,95}/[A-Za-z0-9][A-Za-z0-9._-]{0,95}", model_id):
        raise _invalid()
    if not isinstance(revision, str) or not re.fullmatch(r"(?:[0-9a-fA-F]{40}|[0-9a-fA-F]{64})", revision):
        raise _invalid()
    return {
        "model_id": model_id, "revision": revision.lower(),
        "declared_files_sha256": _normalize_file_hashes(payload["declared_files_sha256"]),
    }


def _read_model_identity(model_dir: Path) -> dict:
    manifest = _decode_json(_read_bytes(model_dir / "cashwatch_model_manifest.json", _JSON_LIMIT))
    if not isinstance(manifest, dict) or not isinstance(manifest.get("files"), dict):
        raise _invalid()
    if not 1 <= len(manifest["files"]) <= _MAX_FILES:
        raise _invalid()
    hashes = {}
    for name, metadata in manifest["files"].items():
        if not isinstance(metadata, dict) or type(metadata.get("bytes")) is not int or metadata["bytes"] < 0:
            raise _invalid()
        hashes[name] = metadata.get("sha256")
    return _normalize_model({
        "model_id": manifest.get("model_id"), "revision": manifest.get("resolved_revision"),
        "declared_files_sha256": hashes,
    })


def _normalize_settings(payload: Any) -> dict:
    if not isinstance(payload, dict) or set(payload) != set(_SETTINGS_FIELDS):
        raise _invalid()
    if all(value is None for value in payload.values()):
        return {name: None for name in _SETTINGS_FIELDS}
    if payload["device"] not in ("auto", "cpu", "cuda", "mps"):
        raise _invalid()
    result = {"device": payload["device"]}
    for name in _FLOAT_FIELDS:
        value = payload[name]
        if type(value) not in (int, float) or not math.isfinite(value):
            raise _invalid()
        result[name] = float(value)
    for name in _INT_FIELDS:
        if type(payload[name]) is not int:
            raise _invalid()
        result[name] = payload[name]
    if (not 0 < result["sample_fps"] <= 30 or result["window_seconds"] <= 0
            or not 0 <= result["overlap_seconds"] < result["window_seconds"]
            or result["max_duration_sec"] <= 0 or not 0 < result["analysis_timeout_sec"] <= 285
            or result["max_frames_per_window"] < 2 or result["max_new_tokens"] < 64
            or not 64 <= result["max_image_edge"] <= 2048
            or not 1 <= result["max_inflight"] <= 8 or not 1 <= result["max_upload_mb"] <= 1024):
        raise _invalid()
    return result


def _rules_identity(path: Path) -> str:
    from .rules import RuleRegistry

    data = _read_bytes(path, _JSON_LIMIT)
    payload = _decode_json(data)
    if not isinstance(payload, dict) or type(payload.get("schema_version")) is not int or payload["schema_version"] != 1:
        raise _invalid()
    registry = RuleRegistry.model_validate(payload)
    identifiers = [rule.rule_id for rule in registry.rules]
    if len(identifiers) != len(set(identifiers)):
        raise _invalid()
    return hashlib.sha256(data).hexdigest()


def _pipeline_identity() -> str:
    root = _PACKAGE_ROOT
    if not root.is_dir():
        raise _invalid()
    paths = sorted(root.rglob("*.py"))
    if not 1 <= len(paths) <= _MAX_FILES or root / "__init__.py" not in paths:
        raise _invalid()
    entries = []
    total = 0
    for path in paths:
        if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
            raise _invalid()
        data = _read_bytes(path, _SOURCE_FILE_LIMIT)
        total += len(data)
        if total > _SOURCE_TOTAL_LIMIT:
            raise _invalid()
        entries.append({"path": path.relative_to(root).as_posix(), "sha256": hashlib.sha256(data).hexdigest()})
    return _fingerprint(entries)


def _identity_issues(identity: dict) -> list[str]:
    missing = []
    if identity["settings"]["device"] is None:
        missing.append(_ISSUES["settings"])
    if identity["model"]["model_id"] is None:
        missing.append(_ISSUES["model"])
    for name in ("rules_sha256", "pipeline_sha256"):
        if identity[name] is None:
            missing.append(_ISSUES[name])
    return sorted(missing)


def build_runtime_contract(settings: Settings) -> dict:
    """Return a snapshot without reading media, credentials, or weight contents.

    Unavailable/invalid metadata produces an incomplete contract, not an API
    exception. The file map states downloader-declared hashes only.
    """
    try:
        parameters = _normalize_settings({name: getattr(settings, name) for name in _SETTINGS_FIELDS})
    except (AttributeError, TypeError, ValueError, OverflowError):
        parameters = {name: None for name in _SETTINGS_FIELDS}
    identity = {
        "schema_version": 1, "settings": parameters, "model": _empty_model(),
        "rules_sha256": None, "pipeline_sha256": None,
    }
    for name, operation in (
        ("model", lambda: _read_model_identity(Path(settings.model_dir))),
        ("rules_sha256", lambda: _rules_identity(Path(settings.rules_path))),
        ("pipeline_sha256", _pipeline_identity),
    ):
        try:
            identity[name] = operation()
        except (OSError, ValueError, TypeError, AttributeError, RecursionError, OverflowError):
            # No exception text (which can contain private paths) is exposed.
            pass
    issues = _identity_issues(identity)
    return {
        **identity, "fingerprint": _fingerprint(identity), "complete": not issues,
        "issues": issues, "weights_hashes_verified": False,
    }


def validate_runtime_contract(payload: Any, *, require_complete: bool = False) -> dict:
    """Validate/rebuild a canonical snapshot, raising only fixed safe codes."""
    try:
        if not isinstance(payload, dict) or set(payload) != set(_CONTRACT_FIELDS):
            raise _invalid()
        if type(payload["schema_version"]) is not int or payload["schema_version"] != 1:
            raise _invalid()
        if type(payload["complete"]) is not bool or payload["weights_hashes_verified"] is not False:
            raise _invalid()
        identity = {
            "schema_version": 1,
            "settings": _normalize_settings(payload["settings"]),
            "model": _normalize_model(payload["model"]),
            "rules_sha256": _sha256(payload["rules_sha256"], nullable=True),
            "pipeline_sha256": _sha256(payload["pipeline_sha256"], nullable=True),
        }
        issues = _identity_issues(identity)
        if payload["issues"] != issues or payload["complete"] != (not issues):
            raise _invalid()
        fingerprint = _fingerprint(identity)
        if _sha256(payload["fingerprint"]) != fingerprint:
            raise ValueError("runtime_contract_fingerprint_mismatch")
    except (TypeError, KeyError, RecursionError, OverflowError):
        raise _invalid() from None
    if require_complete and issues:
        raise ValueError("runtime_contract_incomplete")
    return {
        **identity, "fingerprint": fingerprint, "complete": not issues,
        "issues": issues, "weights_hashes_verified": False,
    }


def compare_runtime_contracts(expected: Any, observed: Any) -> list[str]:
    """Return fixed schema paths only, without dynamic model filenames/values."""
    left = validate_runtime_contract(expected)
    right = validate_runtime_contract(observed)
    different = []
    for section, fields in (("settings", _SETTINGS_FIELDS), ("model", _MODEL_FIELDS)):
        different.extend(f"{section}.{name}" for name in fields if left[section][name] != right[section][name])
    different.extend(name for name in ("rules_sha256", "pipeline_sha256") if left[name] != right[name])
    return different
