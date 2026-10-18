#!/usr/bin/env python3
"""Build a local video manifest without extracting or describing any frames.

Only the standard library and the local ffprobe executable are used.  ``path``
and ``video_id`` are POSIX paths relative to ``metadata.input_dir``.  No split,
group, label, or event is inferred from a filename or from the video contents.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import tempfile
from collections import defaultdict
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path, PurePosixPath
from typing import Any


SPLITS = {"train", "validation", "holdout", "unassigned"}
ASSIGNED_SPLITS = SPLITS - {"unassigned"}
UNKNOWN_GROUPS = {"", "unassigned", "unknown", "none", "null", "n/a"}
CSV_COLUMNS = {"video", "label", "timecode_sec", "rule_id", "group_id", "camera_id", "split"}


class DatasetError(ValueError):
    """An input problem that must not produce a misleading manifest/report."""


def finite_number(value: Any, name: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DatasetError(f"{name} must be a finite number")
    try:
        number = float(value)
    except (OverflowError, ValueError) as exc:
        raise DatasetError(f"{name} must be a finite number") from exc
    if not math.isfinite(number) or number < 0 or (positive and number == 0):
        raise DatasetError(f"{name} must be {'positive' if positive else 'nonnegative'} and finite")
    return number


def _object_without_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise DatasetError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _invalid_constant(value: str) -> None:
    raise DatasetError(f"Non-finite JSON constant is not allowed: {value}")


def load_json(path: Path) -> Any:
    try:
        return json.loads(
            path.read_text(encoding="utf-8-sig"),
            object_pairs_hook=_object_without_duplicate_keys,
            parse_constant=_invalid_constant,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise DatasetError(f"Cannot read valid JSON from {path}: {exc}") from exc


def write_json(path: Path, payload: Any) -> None:
    """Replace the output atomically, never leaving a partial JSON document."""
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=f".{path.name}.", suffix=".tmp", delete=False) as out:
            temporary = out.name
            json.dump(payload, out, ensure_ascii=False, indent=2, allow_nan=False)
            out.write("\n")
        os.replace(temporary, path)
    finally:
        if temporary is not None and os.path.exists(temporary):
            os.unlink(temporary)


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _probe_positive(value: Any) -> float | None:
    try:
        if isinstance(value, bool):
            return None
        number = float(Fraction(str(value)))
        return number if math.isfinite(number) and number > 0 else None
    except (TypeError, ValueError, ZeroDivisionError, OverflowError):
        return None


def probe_video(path: Path, *, ffprobe: str = "ffprobe", timeout: float = 30.0) -> dict[str, Any]:
    command = [
        ffprobe, "-v", "error", "-select_streams", "v:0", "-show_entries",
        "stream=width,height,avg_frame_rate,r_frame_rate,duration,duration_ts,time_base:format=duration",
        "-of", "json", "-protocol_whitelist", "file,pipe", str(path),
    ]
    try:
        process = subprocess.run(command, stdin=subprocess.DEVNULL, capture_output=True,
                                 text=True, check=False, timeout=timeout)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise DatasetError(f"ffprobe failed for {path.name}: {type(exc).__name__}") from exc
    if process.returncode:
        # Do not reproduce arbitrary media metadata or ffprobe's stderr.
        raise DatasetError(f"ffprobe rejected {path.name} (exit {process.returncode})")
    try:
        payload = json.loads(process.stdout, object_pairs_hook=_object_without_duplicate_keys,
                             parse_constant=_invalid_constant)
        streams = payload.get("streams")
        if not isinstance(streams, list) or not streams or not isinstance(streams[0], dict):
            raise DatasetError(f"No video stream found in {path.name}")
        stream = streams[0]
        duration = _probe_positive(stream.get("duration"))
        if duration is None:
            ticks = _probe_positive(stream.get("duration_ts"))
            base = _probe_positive(stream.get("time_base"))
            if ticks is not None and base is not None:
                duration = _probe_positive(ticks * base)
        if duration is None:
            duration = _probe_positive(payload.get("format", {}).get("duration"))
        fps = _probe_positive(stream.get("avg_frame_rate")) or _probe_positive(stream.get("r_frame_rate"))
        width, height = stream.get("width"), stream.get("height")
        if (duration is None or fps is None or isinstance(width, bool) or isinstance(height, bool)
                or not isinstance(width, int) or not isinstance(height, int) or min(width, height) <= 0):
            raise DatasetError(f"Missing or invalid duration/fps/dimensions in {path.name}")
        return {"duration_sec": duration, "fps": fps, "width": width, "height": height}
    except (json.JSONDecodeError, AttributeError, TypeError) as exc:
        raise DatasetError(f"Invalid ffprobe metadata for {path.name}") from exc


def known_group(value: Any) -> bool:
    return isinstance(value, str) and value.strip().casefold() not in UNKNOWN_GROUPS


def check_split_integrity(videos: list[dict[str, Any]], *, require_assigned_groups: bool) -> dict[str, Any]:
    """Check the entire manifest, including training entries, without opening media."""
    ids: set[str] = set()
    by_hash: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_path: dict[str, list[dict[str, Any]]] = defaultdict(list)
    unknown_assigned: list[str] = []
    for video in videos:
        video_id = video["video_id"]
        if video_id in ids:
            raise DatasetError(f"Duplicate video_id in manifest: {video_id}")
        ids.add(video_id)
        by_hash[video["sha256"].lower()].append(video)
        by_path[video["path"]].append(video)
        if known_group(video.get("group_id")):
            by_group[video["group_id"]].append(video)
        elif video["split"] in ASSIGNED_SPLITS:
            unknown_assigned.append(video_id)
    for kind, collection in (("sha256", by_hash), ("group_id", by_group), ("path", by_path)):
        for entries in collection.values():
            partitions = {entry["split"] for entry in entries} & ASSIGNED_SPLITS
            if len(partitions) > 1:
                names = ", ".join(entry["video_id"] for entry in entries)
                raise DatasetError(f"Split leakage by {kind} across {sorted(partitions)}: {names}")
    if require_assigned_groups and unknown_assigned:
        raise DatasetError("Assigned videos have unknown group_id; independence cannot be checked: "
                           + ", ".join(unknown_assigned))
    duplicates = [
        {"sha256": digest, "video_ids": [entry["video_id"] for entry in entries],
         "splits": sorted({entry["split"] for entry in entries})}
        for digest, entries in sorted(by_hash.items()) if len(entries) > 1
    ]
    return {
        "checked_keys": ["video_id", "sha256", "group_id", "path"],
        "no_detected_cross_split_leakage": True,
        "unknown_group_videos_in_assigned_splits": unknown_assigned,
        "duplicate_hashes": duplicates,
        "note": "No detected overlap is conditional on correct group annotations; it is not proof of independence.",
    }


def _csv_video_key(value: str) -> str:
    if not value or "\\" in value:
        raise DatasetError("CSV video must be a relative POSIX path using /, not an absolute path or backslashes")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or re.match(r"^[A-Za-z]:", value):
        raise DatasetError(f"CSV video path is outside input-dir: {value}")
    return path.as_posix()


def apply_labels(videos: list[dict[str, Any]], labels_path: Path) -> None:
    index = {video["video_id"]: video for video in videos}
    records: dict[str, dict[str, Any]] = {}
    try:
        with labels_path.open("r", encoding="utf-8-sig", newline="") as source:
            reader = csv.DictReader(source)
            if reader.fieldnames is None:
                raise DatasetError("Labels CSV has no header")
            headers = [name.strip() for name in reader.fieldnames]
            if len(set(headers)) != len(headers) or any(not name for name in headers):
                raise DatasetError("Duplicate or empty labels CSV column")
            if not {"video", "label"} <= set(headers):
                raise DatasetError("Labels CSV requires video,label columns")
            if set(headers) - CSV_COLUMNS:
                raise DatasetError(f"Unknown labels CSV columns: {sorted(set(headers) - CSV_COLUMNS)}")
            reader.fieldnames = headers
            for row_number, raw in enumerate(reader, start=2):
                if None in raw or any(value is None for value in raw.values()):
                    raise DatasetError(f"CSV row {row_number} has a wrong number of columns")
                row = {key: value.strip() for key, value in raw.items()}
                key = _csv_video_key(row["video"])
                if key not in index:
                    raise DatasetError(f"CSV row {row_number} refers to an unknown video: {key}")
                if row["label"] not in {"0", "1"}:
                    raise DatasetError(f"CSV row {row_number}: label must be exactly 0 or 1")
                label = int(row["label"])
                record = records.setdefault(key, {"label": label, "events": [], "seen_events": set()})
                if record["label"] != label:
                    raise DatasetError(f"Conflicting labels for {key}")
                for field in ("group_id", "camera_id", "split"):
                    value = row.get(field, "")
                    if field in {"group_id", "camera_id"} and not known_group(value):
                        value = ""  # Canonical unknown value in the output remains unassigned.
                    if value:
                        if field == "split" and value not in SPLITS:
                            raise DatasetError(f"CSV row {row_number}: unknown split {value}")
                        if field in record and record[field] != value:
                            raise DatasetError(f"Conflicting {field} for {key}")
                        record[field] = value
                raw_time, rule_id = row.get("timecode_sec", ""), row.get("rule_id", "")
                if raw_time or rule_id:
                    if label == 0:
                        raise DatasetError(f"CSV row {row_number}: label 0 cannot have an event")
                    if not raw_time:
                        raise DatasetError(f"CSV row {row_number}: rule_id requires timecode_sec")
                    try:
                        timestamp = finite_number(float(raw_time), f"CSV row {row_number} timecode_sec")
                    except (ValueError, OverflowError) as exc:
                        raise DatasetError(f"CSV row {row_number}: invalid timecode_sec") from exc
                    if timestamp > index[key]["duration_sec"]:
                        raise DatasetError(f"CSV row {row_number}: timecode_sec exceeds video duration")
                    event_key = (timestamp, rule_id or None)
                    if event_key in record["seen_events"]:
                        raise DatasetError(f"Duplicate event annotation for {key} at {timestamp}")
                    record["seen_events"].add(event_key)
                    record["events"].append({"timecode_sec": timestamp, "rule_id": rule_id or None})
    except (OSError, UnicodeError, csv.Error) as exc:
        raise DatasetError(f"Cannot read labels CSV {labels_path}: {exc}") from exc
    for key, record in records.items():
        video = index[key]
        for field in ("label", "group_id", "camera_id", "split"):
            if field in record:
                video[field] = record[field]
        video["events"] = sorted(record["events"], key=lambda event: (event["timecode_sec"], event["rule_id"] or ""))


def build_manifest(input_dir: Path, labels: Path | None = None, *, ffprobe: str = "ffprobe",
                   probe_timeout: float = 30.0) -> dict[str, Any]:
    root = input_dir.resolve()
    if not root.is_dir():
        raise DatasetError(f"input-dir is not a directory: {root}")
    finite_number(probe_timeout, "probe_timeout", positive=True)
    paths = sorted((path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in {".mov", ".mp4"}),
                   key=lambda path: path.relative_to(root).as_posix())
    if not paths:
        raise DatasetError("No .mov/.mp4 videos found in input-dir")
    videos: list[dict[str, Any]] = []
    for path in paths:
        if not path.resolve().is_relative_to(root):
            raise DatasetError(f"Video symlink leaves input-dir: {path.relative_to(root).as_posix()}")
        relative = path.relative_to(root).as_posix()
        before = path.stat()
        metadata = probe_video(path, ffprobe=ffprobe, timeout=probe_timeout)
        digest = hash_file(path)
        after = path.stat()
        if (before.st_size, before.st_mtime_ns, before.st_ino) != (after.st_size, after.st_mtime_ns, after.st_ino):
            raise DatasetError(f"Video changed during metadata/hash extraction: {relative}")
        videos.append({
            "video_id": relative, "path": relative, "sha256": digest, **metadata,
            "label": None, "events": [], "group_id": "unassigned", "camera_id": "unassigned", "split": "unassigned",
        })
    if labels is not None:
        apply_labels(videos, labels)
    checks = check_split_integrity(videos, require_assigned_groups=False)
    return {
        "schema_version": 1,
        "metadata": {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "input_dir": str(root),
            "path_mode": "path and video_id are POSIX paths relative to input_dir",
            "labels_source": str(labels.resolve()) if labels is not None else None,
            "labels_sha256": hash_file(labels) if labels is not None else None,
            "split_policy": "Only explicit CSV assignments; no random or automatic split",
            "group_policy": "No group inference; unassigned does not establish independence",
            "duration_policy": "first video stream duration, then duration_ts*time_base, then container duration",
            "fps_policy": "first video stream avg_frame_rate, falling back to r_frame_rate; VFR is not a frame timestamp grid",
            "dimensions_policy": "encoded video stream dimensions, before display rotation",
            "frames_extracted": False,
            "split_integrity": checks,
        },
        "videos": videos,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--labels", type=Path, help="CSV: video,label; see docs/DATA_AND_EVALUATION.md")
    parser.add_argument("--out", type=Path, default=Path("data/manifest.local.json"))
    parser.add_argument("--ffprobe", default="ffprobe")
    parser.add_argument("--probe-timeout", type=float, default=30.0)
    args = parser.parse_args(argv)
    try:
        manifest = build_manifest(args.input_dir, args.labels, ffprobe=args.ffprobe, probe_timeout=args.probe_timeout)
        protected = {(args.input_dir.resolve() / video["path"]).resolve() for video in manifest["videos"]}
        if args.labels is not None:
            protected.add(args.labels.resolve())
        if args.out.resolve() in protected:
            raise DatasetError("Output must not overwrite a video or labels CSV")
        write_json(args.out, manifest)
    except (DatasetError, OSError) as exc:
        print(f"prepare_dataset: {exc}", file=sys.stderr)
        return 2
    print(f"Manifest saved: {args.out} ({len(manifest['videos'])} videos; no frames extracted)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
