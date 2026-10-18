#!/usr/bin/env python3
"""Exercise the actual HTTP jury contract against the team's own deployment."""
from __future__ import annotations

import argparse
import base64
import binascii
import hashlib
import io
import json
import math
import os
import re
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from dotenv import load_dotenv
from PIL import Image
from cashwatch.schemas import DoneResponse, FailedResponse, ProcessingResponse
from cashwatch.runtime_contract import compare_runtime_contracts, validate_runtime_contract


@dataclass(frozen=True)
class SelectedVideo:
    video_id: str
    path: Path
    duration_sec: float | None = None


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON key")
        result[key] = value
    return result


def _invalid_constant(value):
    raise ValueError("Non-finite JSON number")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _check_video_file(path: Path) -> None:
    if not path.is_file() or path.suffix.lower() not in {".mov", ".mp4"}:
        raise ValueError(f"Not a local MOV/MP4: {path}")
    if not 0 < path.stat().st_size <= 100 * 1024 * 1024:
        raise ValueError("The smoke checker accepts nonempty videos up to 100 MiB")


def select_manifest_videos(manifest_path: Path, split: str) -> list[SelectedVideo]:
    """Resolve only the requested partition; retain the manifest's exact IDs.

    No filename, label, split or camera identity is inferred. Source files are
    checked against the recorded hash before the first HTTP request is sent.
    """
    if split not in {"validation", "holdout"}:
        raise ValueError("Choose validation or holdout explicitly")
    payload = json.loads(manifest_path.read_text(encoding="utf-8-sig"),
                         object_pairs_hook=_unique_object, parse_constant=_invalid_constant)
    if not isinstance(payload, dict) or type(payload.get("schema_version")) is not int or payload["schema_version"] != 1:
        raise ValueError("Expected a version 1 dataset manifest")
    metadata = payload.get("metadata")
    if not isinstance(metadata, dict) or not isinstance(metadata.get("input_dir"), str):
        raise ValueError("Manifest must provide metadata.input_dir")
    root = Path(metadata["input_dir"])
    if not root.is_absolute():
        raise ValueError("metadata.input_dir must be an absolute directory")
    root = root.resolve(strict=True)
    if not root.is_dir():
        raise ValueError("metadata.input_dir is not a local directory")
    videos = payload.get("videos")
    if not isinstance(videos, list):
        raise ValueError("Manifest videos must be a list")
    selected, known_ids = [], set()
    for video in videos:
        if not isinstance(video, dict):
            raise ValueError("Each manifest video must be an object")
        identifier = video.get("video_id")
        if not isinstance(identifier, str) or not identifier or identifier in known_ids:
            raise ValueError("Manifest video_id values must be nonempty and unique")
        known_ids.add(identifier)
        if video.get("split") not in {"train", "validation", "holdout", "unassigned"}:
            raise ValueError("Unknown manifest split")
        if video["split"] != split:
            continue
        relative = video.get("path")
        if not isinstance(relative, str) or not relative or "\\" in relative:
            raise ValueError("Selected video.path must be a relative POSIX path")
        posix = PurePosixPath(relative)
        if (posix.is_absolute() or ".." in posix.parts or not posix.parts
                or posix.as_posix() != relative or re.match(r"^[A-Za-z]:", relative)):
            raise ValueError("Selected video.path must stay inside metadata.input_dir")
        path = (root / Path(*posix.parts)).resolve(strict=True)
        if not path.is_relative_to(root):
            raise ValueError("Selected video resolves outside metadata.input_dir")
        _check_video_file(path)
        expected_hash = video.get("sha256")
        if not isinstance(expected_hash, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", expected_hash):
            raise ValueError("Selected video needs a valid manifest SHA-256")
        if _sha256_file(path) != expected_hash.lower():
            raise ValueError(f"Selected video changed since manifest creation: {identifier}")
        duration = video.get("duration_sec")
        if type(duration) not in {int, float} or not math.isfinite(duration) or duration <= 0:
            raise ValueError("Selected video needs a positive finite duration")
        selected.append(SelectedVideo(identifier, path, float(duration)))
    if not selected:
        raise ValueError(f"No videos assigned to {split}")
    return selected


def select_manual_videos(paths: list[Path]) -> list[SelectedVideo]:
    selected, known_ids = [], set()
    for raw_path in paths:
        path = raw_path.resolve(strict=True)
        _check_video_file(path)
        identifier = raw_path.name
        if identifier in known_ids:
            raise ValueError("Duplicate basenames: use --manifest and --split to preserve relative video IDs")
        known_ids.add(identifier)
        selected.append(SelectedVideo(identifier, path))
    return selected


def validate_job_response(payload, *, expected_job_id: str | None = None, required_status: str | None = None,
                          duration_sec: float | None = None):
    if not isinstance(payload, dict):
        raise ValueError("Jury response must be a JSON object")
    status = payload.get("status")
    schemas = {"done": DoneResponse, "failed": FailedResponse, "processing": ProcessingResponse}
    if not isinstance(status, str) or status not in schemas or (required_status is not None and status != required_status):
        raise ValueError("Unexpected jury response status")
    response = schemas[status].model_validate(payload)
    if not re.fullmatch(r"[0-9a-f]{32}", response.job_id):
        raise ValueError("The local service returned an invalid job ID")
    if expected_job_id is not None and response.job_id != expected_job_id:
        raise ValueError("Polling returned another video's job ID")
    if isinstance(response, DoneResponse):
        for raw_violation, violation in zip(payload["violations"], response.violations):
            if type(raw_violation.get("timecode_sec")) not in {int, float}:
                raise ValueError("Event timecode must be a JSON number")
            if duration_sec is not None and violation.timecode_sec > duration_sec + 0.05:
                raise ValueError("Event timecode is outside the selected video")
            for raw_frame, frame in zip(raw_violation["evidence_frames"], violation.evidence_frames):
                if type(raw_frame.get("timestamp_sec")) not in {int, float}:
                    raise ValueError("Frame timestamp must be a JSON number")
                if duration_sec is not None and frame.timestamp_sec > duration_sec + 0.05:
                    raise ValueError("Evidence timestamp is outside the selected video")
                try:
                    data = base64.b64decode(frame.image_base64, validate=True)
                    if not data or len(data) > 4 * 1024 * 1024:
                        raise ValueError("Evidence JPEG size is invalid")
                    with Image.open(io.BytesIO(data)) as image:
                        if image.format != "JPEG" or image.width * image.height > 2048 * 2048:
                            raise ValueError("Evidence is not a bounded JPEG image")
                        image.load()
                except (OSError, SyntaxError, ValueError, binascii.Error, Image.DecompressionBombError) as exc:
                    raise ValueError("Jury response contains invalid JPEG evidence") from exc
    return response


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        # Do not forward API credentials to a redirect destination.
        return None


def request_json(url: str, key: str, *, body: bytes | None = None, content_type: str | None = None,
                 timeout: float = 30, expected_status: int = 200):
    headers = {"X-API-Key": key, "Accept": "application/json"}
    if content_type:
        headers["Content-Type"] = content_type
    request = urllib.request.Request(url, data=body, headers=headers, method="POST" if body is not None else "GET")
    opener = urllib.request.build_opener(_NoRedirect())
    with opener.open(request, timeout=timeout) as response:
        if response.status != expected_status:
            raise ValueError("Unexpected HTTP status for the jury endpoint")
        return json.load(response, object_pairs_hook=_unique_object, parse_constant=_invalid_constant)


class RuntimePreflightError(ValueError):
    """Only constant codes and fixed schema labels may reach diagnostics."""

    def __init__(self, code: str, fields: list[str] | None = None):
        self.code = code
        self.fields = fields or []
        super().__init__(code + (": " + ", ".join(self.fields) if self.fields else ""))


def load_expected_benchmark(path: Path) -> dict:
    """Read a successful jury benchmark without retaining its private inputs.

    This checks a local report's internal consistency, not its authenticity or
    the actual GPU/weight bytes. Analyzer-only runs cannot authorize a jury run.
    """
    try:
        if not path.is_file() or path.stat().st_size > 2 * 1024 * 1024:
            raise RuntimePreflightError("expected_benchmark_unreadable")
        report = json.loads(path.read_text(encoding="utf-8-sig"),
                            object_pairs_hook=_unique_object, parse_constant=_invalid_constant)
    except (OSError, UnicodeError, ValueError) as exc:
        raise RuntimePreflightError("expected_benchmark_unreadable_or_invalid_json") from exc
    if (not isinstance(report, dict) or type(report.get("schema_version")) is not int
            or report["schema_version"] != 1 or report.get("kind") != "cashwatch_local_benchmark"
            or report.get("mode") != "jury" or report.get("contract_stable") is not True):
        raise RuntimePreflightError("expected_benchmark_not_stable_jury_run")
    try:
        contract = validate_runtime_contract(report.get("runtime_contract"), require_complete=True)
    except (TypeError, ValueError) as exc:
        raise RuntimePreflightError("expected_benchmark_runtime_contract_invalid") from exc
    device = report.get("requested_device")
    if device not in ("cuda", "mps", "cpu") or device != contract["settings"]["device"]:
        raise RuntimePreflightError("expected_benchmark_device_not_explicit_or_inconsistent")
    requested = report.get("requested_runs")
    runs, summary = report.get("runs"), report.get("summary")
    if (type(requested) is not int or not 1 <= requested <= 10 or not isinstance(runs, list)
            or len(runs) != requested or not isinstance(summary, dict)
            or type(summary.get("completed_runs")) is not int or summary["completed_runs"] != requested
            or summary.get("all_requested_runs_completed_within_budget") is not True):
        raise RuntimePreflightError("expected_benchmark_not_all_runs_completed")
    for run in runs:
        if not isinstance(run, dict):
            raise RuntimePreflightError("expected_benchmark_run_incomplete")
        wall, budget = run.get("wall_seconds"), run.get("latency_budget_seconds")
        selected_device = run.get("selected_device")
        if (run.get("status") != "complete" or run.get("jury_response_status") != "done"
                or any(run.get(flag) is not True for flag in (
                    "model_loaded", "generation_attempted", "analyzer_complete",
                    "cold_model_load_included", "final_metrics_received", "latency_budget_passed"))
                or run.get("supervisor_hard_stop") is not False
                or type(run.get("child_exit_code")) is not int or run["child_exit_code"] != 0
                or not isinstance(selected_device, str)
                or not re.fullmatch(re.escape(device) + r"(?::[0-9]+)?", selected_device)
                or type(wall) not in (int, float) or not math.isfinite(wall) or wall <= 0
                or type(budget) not in (int, float) or not math.isfinite(budget)
                or budget != contract["settings"]["analysis_timeout_sec"] or wall > budget):
            raise RuntimePreflightError("expected_benchmark_run_incomplete_or_inconsistent")
    return contract


def check_server_runtime(base: str, key: str, *, expected: dict | None = None) -> dict | None:
    """Inspect capabilities before constructing or sending a video request."""
    try:
        capabilities = request_json(base + "/api/v1/capabilities", key)
    except urllib.error.HTTPError as exc:
        raise RuntimePreflightError(f"capabilities_http_{int(exc.code)}") from exc
    except (OSError, ValueError, TimeoutError, urllib.error.URLError) as exc:
        raise RuntimePreflightError("capabilities_unavailable_or_invalid_json") from exc
    if not isinstance(capabilities, dict):
        raise RuntimePreflightError("capabilities_invalid")
    if "runtime_contract" not in capabilities:
        if expected is not None:
            raise RuntimePreflightError("server_runtime_contract_missing")
        return None  # Legacy smoke runs remain possible, explicitly unverified.
    try:
        observed = validate_runtime_contract(capabilities["runtime_contract"], require_complete=expected is not None)
    except (TypeError, ValueError) as exc:
        raise RuntimePreflightError("server_runtime_contract_invalid_or_incomplete") from exc
    if expected is not None:
        differences = compare_runtime_contracts(expected, observed)
        if differences:
            raise RuntimePreflightError("runtime_contract_mismatch", differences)
    return observed


def save_checkpoint(path: Path, records: list[dict], metadata: dict, *, expected_count: int,
                    run_finished: bool) -> None:
    """Atomically preserve terminal responses; completeness is not model success.

    Only evaluation IDs and the existing result records are persisted. The key,
    service origin, source paths and manifest path are never added to metadata.
    A cancelled run keeps its last fully written checkpoint.
    """
    complete = len(records) == expected_count
    if run_finished != complete:
        raise ValueError("Checkpoint completion metadata is inconsistent")
    payload = {
        "predictions": records,
        "metadata": {
            **metadata,
            "expected_video_count": expected_count,
            "recorded_video_count": len(records),
            "failed_video_count": sum(record.get("status") == "failed" for record in records),
            "complete": complete,
            "run_finished": run_finished,
        },
        "note": "Local smoke results; not the organizers' official scores. "
                "Complete means every selected video has a terminal record, including failed results.",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as output:
            os.fchmod(output.fileno(), 0o600)
            json.dump(payload, output, ensure_ascii=False, indent=2, allow_nan=False)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True, help="Only your team's own service URL")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--video", type=Path, action="append", help="Manual smoke input; use --manifest for evaluation IDs")
    source.add_argument("--manifest", type=Path, help="Manifest from prepare_dataset.py")
    parser.add_argument("--split", choices=["validation", "holdout"], help="Required with --manifest; never defaults to holdout")
    parser.add_argument("--expect-benchmark", type=Path,
                        help="Require the server to match a successful, stable --mode jury benchmark before upload")
    parser.add_argument("--out", type=Path, default=Path("local_reports/jury_predictions.json"))
    args = parser.parse_args()
    if args.manifest is not None and args.split is None:
        parser.error("--manifest requires an explicit --split validation or --split holdout")
    if args.video is not None and args.split is not None:
        parser.error("--split requires --manifest")
    try:
        videos = select_manifest_videos(args.manifest, args.split) if args.manifest else select_manual_videos(args.video)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    inputs = {video.path for video in videos}
    if args.manifest:
        inputs.add(args.manifest.resolve())
    if args.expect_benchmark:
        inputs.add(args.expect_benchmark.resolve())
    if args.out.resolve() in inputs:
        parser.error("Output must not overwrite a source video, dataset manifest or benchmark")
    load_dotenv(override=False)
    key = os.getenv("CASHWATCH_API_KEY", "")
    if not key:
        parser.error("Set CASHWATCH_API_KEY in your local environment or .env")
    base = args.base_url.rstrip("/")
    parsed = urllib.parse.urlparse(base)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc or parsed.username or parsed.password or parsed.query or parsed.fragment:
        parser.error("Expected an http(s) service origin without credentials/query/fragment")
    try:
        expected_contract = load_expected_benchmark(args.expect_benchmark) if args.expect_benchmark else None
        initial_contract = check_server_runtime(base, key, expected=expected_contract)
    except RuntimePreflightError as exc:
        # Preserve any previous report and send no video if preflight fails.
        parser.error(str(exc))
    records = []
    failures = 0
    metadata = {
        "input_mode": "manifest" if args.manifest else "manual", "split": args.split,
        "runtime_contract": initial_contract,
        "expected_runtime_fingerprint": expected_contract["fingerprint"] if expected_contract else None,
        "runtime_check": {
            "status": "benchmark_matched" if expected_contract else
                      "contract_captured" if initial_contract else "legacy_unverified",
            "capabilities_checks_completed": 1,
            "weights_hashes_verified": False,
        },
    }
    if args.manifest:
        metadata["manifest_sha256"] = _sha256_file(args.manifest)
    save_checkpoint(args.out, records, metadata, expected_count=len(videos), run_finished=False)
    for index, selected in enumerate(videos):
        if index:
            try:
                observed = check_server_runtime(base, key, expected=expected_contract)
                if (initial_contract is None) != (observed is None):
                    raise RuntimePreflightError("runtime_contract_availability_changed")
                if initial_contract is not None:
                    differences = compare_runtime_contracts(initial_contract, observed)
                    if differences:
                        raise RuntimePreflightError("runtime_contract_changed_during_run", differences)
            except RuntimePreflightError as exc:
                metadata["runtime_check"].update(status="stopped_before_upload", error_code=exc.code,
                                                  mismatched_fields=exc.fields)
                save_checkpoint(args.out, records, metadata, expected_count=len(videos), run_finished=False)
                print(json.dumps({"error": exc.code, "mismatched_fields": exc.fields}), file=sys.stderr)
                return 1
            metadata["runtime_check"]["capabilities_checks_completed"] += 1
        video = selected.path
        boundary = f"cashwatch-{uuid.uuid4().hex}"
        # An opaque filename avoids copying paths or arbitrary MIME header content.
        prefix = f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="input{video.suffix.lower()}"\r\nContent-Type: video/quicktime\r\n\r\n'.encode()
        body = prefix + video.read_bytes() + f"\r\n--{boundary}--\r\n".encode()
        started = time.monotonic()
        record = {"video_id": selected.video_id}
        try:
            accepted = validate_job_response(
                request_json(base + "/api/v1/analyze", key, body=body,
                             content_type=f"multipart/form-data; boundary={boundary}", timeout=60, expected_status=202),
                required_status="processing",
            )
            while True:
                remaining = 300 - (time.monotonic() - started)
                if remaining <= 0:
                    raise TimeoutError("The 300-second jury deadline expired")
                result = validate_job_response(
                    request_json(base + f"/api/v1/jobs/{accepted.job_id}", key, timeout=min(30, remaining)),
                    expected_job_id=accepted.job_id, duration_sec=selected.duration_sec,
                )
                if time.monotonic() - started > 300:
                    raise TimeoutError("Result arrived after the 300-second jury deadline")
                if isinstance(result, DoneResponse):
                    record.update(result.model_dump())
                    break
                if isinstance(result, FailedResponse):
                    record.update(result.model_dump())
                    failures += 1
                    break
                time.sleep(min(1, max(0, 300 - (time.monotonic() - started))))
        except (OSError, ValueError, TimeoutError, urllib.error.URLError) as error:
            failures += 1
            record.update({"status": "failed", "error": {"code": type(error).__name__, "message": "Jury smoke check failed; inspect your local service."}})
        record["elapsed_sec"] = round(time.monotonic() - started, 3)
        records.append(record)
        save_checkpoint(args.out, records, metadata, expected_count=len(videos),
                        run_finished=len(records) == len(videos))
        print(json.dumps({"video_id": selected.video_id, "status": record["status"],
                          "elapsed_sec": record["elapsed_sec"]}, ensure_ascii=False))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

