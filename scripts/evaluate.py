#!/usr/bin/env python3
"""Offline diagnostic evaluation on validation or holdout; never trains or tunes.

No videos are opened and no images are displayed.  Evidence verification checks
the base64/JPEG envelope and reported timestamps, not image contents or provenance.
"""
from __future__ import annotations

import argparse
import base64
import binascii
import math
import re
import sys
from collections import Counter, deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from .prepare_dataset import (DatasetError, SPLITS, check_split_integrity, finite_number,
                                  hash_file, load_json, write_json)
except ImportError:  # Direct invocation: python scripts/evaluate.py
    from prepare_dataset import (DatasetError, SPLITS, check_split_integrity, finite_number,
                                 hash_file, load_json, write_json)


TOLERANCES = (1.0, 2.0, 5.0)
VIDEO_FIELDS = {"video_id", "path", "sha256", "duration_sec", "fps", "width", "height",
                "label", "events", "group_id", "camera_id", "split"}


def ratio(numerator: int | float, denominator: int | float) -> float | None:
    return numerator / denominator if denominator else None


def prf(tp: int, fp: int, fn: int) -> dict[str, Any]:
    return {"tp": tp, "fp": fp, "fn": fn, "precision": ratio(tp, tp + fp),
            "recall": ratio(tp, tp + fn), "f1": ratio(2 * tp, 2 * tp + fp + fn)}


def percentile(values: list[float], probability: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    low, high = math.floor(position), math.ceil(position)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def _nonempty_string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise DatasetError(f"{name} must be a nonempty string")
    return value


def validate_manifest(payload: Any) -> list[dict[str, Any]]:
    if not isinstance(payload, dict) or type(payload.get("schema_version")) is not int or payload["schema_version"] != 1:
        raise DatasetError("Manifest must have schema_version: 1")
    if not isinstance(payload.get("videos"), list):
        raise DatasetError("Manifest videos must be a list")
    videos: list[dict[str, Any]] = []
    for index, raw in enumerate(payload["videos"]):
        name = f"manifest.videos[{index}]"
        if not isinstance(raw, dict) or VIDEO_FIELDS - raw.keys():
            raise DatasetError(f"{name} is missing required video fields")
        video = dict(raw)
        for key in ("video_id", "path", "sha256", "group_id", "camera_id", "split"):
            _nonempty_string(video[key], f"{name}.{key}")
        if not re.fullmatch(r"[0-9a-fA-F]{64}", video["sha256"]):
            raise DatasetError(f"{name}.sha256 must contain 64 hexadecimal characters")
        if video["split"] not in SPLITS:
            raise DatasetError(f"{name}.split is invalid")
        for key in ("duration_sec", "fps"):
            video[key] = finite_number(video[key], f"{name}.{key}", positive=True)
        for key in ("width", "height"):
            if type(video[key]) is not int or video[key] <= 0:
                raise DatasetError(f"{name}.{key} must be a positive integer")
        if video["label"] is not None and (type(video["label"]) is not int or video["label"] not in {0, 1}):
            raise DatasetError(f"{name}.label must be 0, 1, or null")
        if not isinstance(video["events"], list):
            raise DatasetError(f"{name}.events must be a list")
        if video["events"] and video["label"] != 1:
            raise DatasetError(f"{name}: only label 1 can have event annotations")
        events: list[dict[str, Any]] = []
        seen: set[tuple[float, str | None]] = set()
        for event_index, event in enumerate(video["events"]):
            if not isinstance(event, dict) or "timecode_sec" not in event or "rule_id" not in event:
                raise DatasetError(f"{name}.events[{event_index}] requires timecode_sec and rule_id")
            timestamp = finite_number(event["timecode_sec"], f"{name}.events[{event_index}].timecode_sec")
            if timestamp > video["duration_sec"]:
                raise DatasetError(f"{name}: truth event is outside video duration")
            rule_id = event["rule_id"]
            if rule_id is not None:
                _nonempty_string(rule_id, f"{name}.events[{event_index}].rule_id")
            key = (timestamp, rule_id)
            if key in seen:
                raise DatasetError(f"{name}: duplicate truth event")
            seen.add(key)
            events.append({"timecode_sec": timestamp, "rule_id": rule_id})
        video["events"] = events
        videos.append(video)
    return videos


def read_registry(payload: Any) -> dict[str, str]:
    if not isinstance(payload, dict) or not isinstance(payload.get("rules"), list):
        raise DatasetError("Rule registry requires a rules list")
    by_id: dict[str, str] = {}
    titles: set[str] = set()
    for rule in payload["rules"]:
        if not isinstance(rule, dict):
            raise DatasetError("Invalid registry entry")
        rule_id = _nonempty_string(rule.get("rule_id"), "registry.rule_id")
        title = _nonempty_string(rule.get("title"), "registry.title")
        if rule_id in by_id or title in titles:
            raise DatasetError("Registry has duplicate rule_id or ambiguous duplicate title")
        by_id[rule_id] = title
        titles.add(title)
    return by_id


def index_predictions(payload: Any, all_videos: list[dict[str, Any]], split: str) -> dict[str, dict[str, Any]]:
    records = payload.get("predictions") if isinstance(payload, dict) else payload
    if not isinstance(records, list):
        raise DatasetError("Predictions must be a list or an object containing predictions: []")
    all_ids = {video["video_id"] for video in all_videos}
    selected_ids = {video["video_id"] for video in all_videos if video["split"] == split}
    result: dict[str, dict[str, Any]] = {}
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise DatasetError(f"predictions[{index}] must be an object")
        video_id = _nonempty_string(record.get("video_id"), f"predictions[{index}].video_id")
        if video_id in result:
            raise DatasetError(f"Duplicate prediction video_id: {video_id}")
        if video_id not in all_ids:
            raise DatasetError(f"Prediction refers to an unknown video_id: {video_id}")
        if video_id not in selected_ids:
            raise DatasetError(f"Prediction is outside requested {split} split: {video_id}")
        result[video_id] = record
    return result


def _normalize_prediction(record: dict[str, Any] | None, video: dict[str, Any], *,
                          registry: dict[str, str] | None,
                          evidence_tolerance_sec: float | None) -> dict[str, Any]:
    result: dict[str, Any] = {"valid": False, "failure_reason": None, "detail": None,
                              "events": [], "prediction_label": None, "elapsed_sec": None,
                              "invalid_latency": False, "evidence_frame_count": 0}
    if record is None:
        result["failure_reason"] = "missing_prediction"
        return result
    elapsed = record.get("elapsed_sec")
    if elapsed is not None:
        try:
            result["elapsed_sec"] = finite_number(elapsed, "elapsed_sec")
        except DatasetError as exc:
            result.update(failure_reason="invalid_elapsed_sec", detail=str(exc), invalid_latency=True)
            return result
    status = record.get("status")
    if status != "done":
        result["failure_reason"] = ("reported_failed" if status == "failed" else
                                    "not_completed" if status == "processing" else "invalid_status")
        return result
    try:
        verdict = record.get("verdict")
        if verdict not in {"violation", "no_violation"}:
            raise DatasetError("verdict must be violation or no_violation")
        violations = record.get("violations")
        if not isinstance(violations, list):
            raise DatasetError("violations must be a list")
        if (verdict == "violation") != bool(violations):
            raise DatasetError("verdict conflicts with violations list")
        events: list[dict[str, Any]] = []
        frames_checked = 0
        title_to_id = {title: rule_id for rule_id, title in registry.items()} if registry is not None else {}
        tolerance = evidence_tolerance_sec if evidence_tolerance_sec is not None else 1.0 / video["fps"]
        for event_index, violation in enumerate(violations):
            prefix = f"violations[{event_index}]"
            if not isinstance(violation, dict):
                raise DatasetError(f"{prefix} must be an object")
            title = _nonempty_string(violation.get("type"), f"{prefix}.type")
            timestamp = finite_number(violation.get("timecode_sec"), f"{prefix}.timecode_sec")
            if timestamp > video["duration_sec"]:
                raise DatasetError(f"{prefix}.timecode_sec is outside video duration")
            rule_id = violation.get("rule_id")
            if rule_id is not None:
                _nonempty_string(rule_id, f"{prefix}.rule_id")
            if registry is not None:
                if rule_id is None:
                    rule_id = title_to_id.get(title)
                    if rule_id is None:
                        raise DatasetError(f"{prefix}.type is not an exact registry title")
                elif rule_id not in registry or registry[rule_id] != title:
                    raise DatasetError(f"{prefix}.rule_id and type do not match the supplied registry")
            evidence = violation.get("evidence_frames")
            if not isinstance(evidence, list) or not 1 <= len(evidence) <= 3:
                result["failure_reason"] = "invalid_evidence"
                raise DatasetError(f"{prefix}.evidence_frames must contain 1 to 3 frames")
            times: list[float] = []
            for frame_index, frame in enumerate(evidence):
                result["failure_reason"] = "invalid_evidence"
                frame_prefix = f"{prefix}.evidence_frames[{frame_index}]"
                if not isinstance(frame, dict):
                    raise DatasetError(f"{frame_prefix} must be an object")
                frame_time = finite_number(frame.get("timestamp_sec"), f"{frame_prefix}.timestamp_sec")
                if frame_time > video["duration_sec"]:
                    raise DatasetError(f"{frame_prefix}.timestamp_sec is outside video duration")
                encoded = frame.get("image_base64")
                if not isinstance(encoded, str) or not encoded:
                    raise DatasetError(f"{frame_prefix}.image_base64 must be a nonempty base64 string")
                try:
                    raw = base64.b64decode(encoded, validate=True)
                except (binascii.Error, ValueError) as exc:
                    raise DatasetError(f"{frame_prefix} has invalid base64") from exc
                if len(raw) < 4 or not raw.startswith(b"\xff\xd8\xff") or not raw.endswith(b"\xff\xd9"):
                    raise DatasetError(f"{frame_prefix} has no valid JPEG SOI/marker/EOI envelope")
                times.append(frame_time)
            if any(later <= earlier for earlier, later in zip(times, times[1:])):
                raise DatasetError(f"{prefix}.evidence_frames timestamps must be strictly increasing")
            anchor_delta = min(abs(frame_time - timestamp) for frame_time in times)
            consistent = anchor_delta == 0 if tolerance == 0 else anchor_delta <= tolerance + 1e-9
            if not consistent:
                raise DatasetError(f"{prefix} has no evidence frame within the event timestamp tolerance")
            result["failure_reason"] = None
            frames_checked += len(times)
            events.append({"timecode_sec": timestamp, "rule_id": rule_id,
                           "evidence_anchor_delta_sec": anchor_delta})
        result.update(valid=True, prediction_label=int(verdict == "violation"), events=events,
                      evidence_frame_count=frames_checked, failure_reason=None)
    except (DatasetError, TypeError) as exc:
        result["failure_reason"] = result["failure_reason"] or "invalid_response"
        # The exception never includes image bytes or the model's description.
        result["detail"] = str(exc)
    return result


def match_events(truth: list[dict[str, Any]], predicted: list[dict[str, Any]], tolerance: float, *,
                 require_same_rule: bool = False) -> list[tuple[int, int]]:
    """Deterministic maximum-cardinality bipartite matching, never many-to-one.

    Candidates prefer smaller time differences, but global minimum time error is
    not claimed. Iterative augmenting paths avoid Python recursion depth limits.
    """
    adjacency: dict[int, list[int]] = {}
    for truth_index, target in enumerate(truth):
        candidates = []
        for pred_index, event in enumerate(predicted):
            if require_same_rule and (target["rule_id"] is None or target["rule_id"] != event["rule_id"]):
                continue
            distance = abs(target["timecode_sec"] - event["timecode_sec"])
            if distance <= tolerance + 1e-9:
                candidates.append((distance, event["timecode_sec"], pred_index))
        adjacency[truth_index] = [candidate[2] for candidate in sorted(candidates)]
    by_truth: dict[int, int] = {}
    by_pred: dict[int, int] = {}
    for start in sorted(range(len(truth)), key=lambda index: (truth[index]["timecode_sec"], index)):
        queue = deque([start])
        seen_truth = {start}
        parent_pred: dict[int, int] = {}
        free_pred: int | None = None
        while queue and free_pred is None:
            target = queue.popleft()
            for candidate in adjacency[target]:
                if candidate in parent_pred:
                    continue
                parent_pred[candidate] = target
                if candidate not in by_pred:
                    free_pred = candidate
                    break
                displaced = by_pred[candidate]
                if displaced not in seen_truth:
                    seen_truth.add(displaced)
                    queue.append(displaced)
        if free_pred is None:
            continue
        current = free_pred
        while True:
            target = parent_pred[current]
            old = by_truth.get(target)
            by_truth[target] = current
            by_pred[current] = target
            if old is None:
                break
            current = old
    return sorted(by_truth.items())


def _event_report(videos: list[dict[str, Any]], normalized: dict[str, dict[str, Any]]) -> dict[str, Any]:
    eligible = [video for video in videos if video["label"] == 0 or video["events"]]
    video_only = [video["video_id"] for video in videos if video["label"] == 1 and not video["events"]]
    typed = [video for video in eligible if all(event["rule_id"] is not None for event in video["events"])]
    typed_truth_count = sum(len(video["events"]) for video in typed)
    result: dict[str, Any] = {
        "official_metric": False,
        "matching": "maximum-cardinality one-to-one matching within each video; inclusive time tolerance",
        "annotation_assumption": "Event annotations are exhaustive for eligible videos; positive videos without events are excluded",
        "failure_policy": "Failed/missing predictions provide zero events; annotated positives therefore contribute false negatives",
        "eligible_videos": len(eligible),
        "eligible_failed_videos": sum(not normalized[video["video_id"]]["valid"] for video in eligible),
        "excluded_video_only_positive_ids": video_only,
        "ground_truth_events": sum(len(video["events"]) for video in eligible),
        "predicted_events": sum(len(normalized[video["video_id"]]["events"]) for video in eligible),
        "tolerances_sec": {},
        "type_aware": {
            "available": typed_truth_count > 0,
            "policy": "Exact rule_id only; optional registry resolves exact titles. Mixed typed/untyped truth videos are excluded",
            "eligible_videos": len(typed),
            "excluded_mixed_or_untyped_truth_ids": [video["video_id"] for video in eligible if video not in typed],
            "unmapped_predicted_events": sum(event["rule_id"] is None for video in typed
                                             for event in normalized[video["video_id"]]["events"]),
            "tolerances_sec": {},
        },
    }
    for tolerance in TOLERANCES:
        tp = fp = fn = 0
        differences: list[float] = []
        for video in eligible:
            truth, predicted = video["events"], normalized[video["video_id"]]["events"]
            matches = match_events(truth, predicted, tolerance)
            tp += len(matches)
            fp += len(predicted) - len(matches)
            fn += len(truth) - len(matches)
            differences.extend(abs(truth[i]["timecode_sec"] - predicted[j]["timecode_sec"]) for i, j in matches)
        result["tolerances_sec"][str(int(tolerance))] = {
            **prf(tp, fp, fn), "mean_matched_time_error_sec": ratio(sum(differences), len(differences)),
        }
        if typed_truth_count == 0:
            continue
        tp = fp = fn = 0
        per_rule: dict[str, Counter[str]] = {}
        for video in typed:
            truth, predicted = video["events"], normalized[video["video_id"]]["events"]
            matches = match_events(truth, predicted, tolerance, require_same_rule=True)
            tp += len(matches)
            fp += len(predicted) - len(matches)  # Unknown predicted types stay unmatched; no invented mapping.
            fn += len(truth) - len(matches)
            for target in truth:
                per_rule.setdefault(target["rule_id"], Counter())["truth"] += 1
            for event in predicted:
                if event["rule_id"] is not None:
                    per_rule.setdefault(event["rule_id"], Counter())["predicted"] += 1
            for i, _ in matches:
                per_rule[truth[i]["rule_id"]]["matched"] += 1
        result["type_aware"]["tolerances_sec"][str(int(tolerance))] = {
            **prf(tp, fp, fn),
            "per_rule": {rule_id: prf(counts["matched"], counts["predicted"] - counts["matched"],
                                      counts["truth"] - counts["matched"])
                         for rule_id, counts in sorted(per_rule.items())},
        }
    return result


def evaluate(manifest: Any, predictions: Any, *, split: str, registry: dict[str, str] | None = None,
             evidence_tolerance_sec: float | None = None) -> dict[str, Any]:
    if split not in {"validation", "holdout"}:
        raise DatasetError("Evaluation split must be validation or holdout; train is prohibited")
    if evidence_tolerance_sec is not None:
        finite_number(evidence_tolerance_sec, "evidence_tolerance_sec")
    videos = validate_manifest(manifest)
    checks = check_split_integrity(videos, require_assigned_groups=True)
    selected = [video for video in videos if video["split"] == split]
    if not selected:
        raise DatasetError(f"No videos assigned to {split}")
    if any(video["label"] is None for video in selected):
        raise DatasetError(f"Every {split} video must have an explicit 0 or 1 label")
    if registry is not None:
        for video in selected:
            for event in video["events"]:
                if event["rule_id"] is not None and event["rule_id"] not in registry:
                    raise DatasetError(f"Truth rule_id is absent from supplied registry: {event['rule_id']}")
    indexed = index_predictions(predictions, videos, split)
    normalized = {video["video_id"]: _normalize_prediction(indexed.get(video["video_id"]), video,
                  registry=registry, evidence_tolerance_sec=evidence_tolerance_sec) for video in selected}
    confusion = {"tp": 0, "tn": 0, "fp": 0, "fn": 0}
    failure_reasons: Counter[str] = Counter()
    per_video: list[dict[str, Any]] = []
    latencies: list[float] = []
    within_limit = successful_within_limit = 0
    invalid_latency = 0
    for video in selected:
        result = normalized[video["video_id"]]
        if result["valid"]:
            cell = ("tp" if result["prediction_label"] == 1 else "fn") if video["label"] == 1 else (
                    "fp" if result["prediction_label"] == 1 else "tn")
            confusion[cell] += 1
        else:
            failure_reasons[result["failure_reason"]] += 1
        elapsed = result["elapsed_sec"]
        if elapsed is not None:
            latencies.append(elapsed)
            if elapsed <= 300.0:  # Inclusive boundary, with no rounding before comparison.
                within_limit += 1
                successful_within_limit += int(result["valid"])
        invalid_latency += int(result["invalid_latency"])
        per_video.append({
            "video_id": video["video_id"], "truth_label": video["label"],
            "prediction_label": result["prediction_label"],
            "outcome": "completed" if result["valid"] else "failed",
            "failure_reason": result["failure_reason"], "detail": result["detail"],
            "elapsed_sec": elapsed, "within_300_sec": elapsed <= 300.0 if elapsed is not None else None,
            "verified_evidence_frames": result["evidence_frame_count"],
        })
    tp, tn, fp, fn = (confusion[key] for key in ("tp", "tn", "fp", "fn"))
    completed, total = sum(confusion.values()), len(selected)
    positives, negatives = sum(video["label"] == 1 for video in selected), sum(video["label"] == 0 for video in selected)
    return {
        "schema_version": 1,
        "evaluation_kind": "local_diagnostic_not_official",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "split": split,
        "metadata": {"manifest_video_count": len(videos), "no_training_or_tuning": True,
                     "no_media_opened_or_images_displayed": True,
                     "registry_mapping": "exact supplied titles and IDs" if registry is not None else "explicit prediction rule_id only"},
        "split_integrity": checks,
        "video_level": {
            "metrics_population": "Confusion matrix and PRF use valid completed predictions only; failures remain explicit",
            "total_videos": total, "completed_videos": completed, "failed_videos": total - completed,
            "coverage": ratio(completed, total), "confusion_matrix": confusion,
            "precision": ratio(tp, tp + fp), "recall": ratio(tp, tp + fn), "f1": ratio(2 * tp, 2 * tp + fp + fn),
            "completed_video_accuracy": ratio(tp + tn, completed),
            "effective_video_accuracy": ratio(tp + tn, total),
            "end_to_end_positive_detection_recall": ratio(tp, positives),
            "end_to_end_negative_correct_rate": ratio(tn, negatives),
        },
        "failures": {"count": total - completed, "by_reason": dict(sorted(failure_reasons.items()))},
        "event_level": _event_report(selected, normalized),
        "evidence_validation": {
            "checks": ["strict base64", "JPEG SOI/marker/EOI envelope", "timestamps within video duration",
                       "1 to 3 chronological frames", "at least one frame anchored to event timecode"],
            "event_anchor_tolerance_sec": evidence_tolerance_sec,
            "default_tolerance": "1 / manifest fps per video" if evidence_tolerance_sec is None else "explicit override; 0 means exact equality",
            "validated_frame_count": sum(result["evidence_frame_count"] for result in normalized.values()),
            "invalid_evidence_responses": failure_reasons["invalid_evidence"],
            "limitations": "JPEG pixels are not decoded; image contents, authenticity, and provenance from the source video are not verified. Average fps does not prove VFR frame-grid alignment.",
        },
        "latency": {
            "source": "reported elapsed_sec only; evaluator does not measure model runtime",
            "reported_count": len(latencies), "missing_or_invalid_count": total - len(latencies),
            "invalid_count": invalid_latency,
            "p50_sec": percentile(latencies, 0.50), "p95_sec": percentile(latencies, 0.95),
            "percentile_method": "linear interpolation at (n-1)*p",
            "limit_sec": 300.0, "within_300_sec_count": within_limit,
            "over_300_sec_count": len(latencies) - within_limit,
            "within_limit_rate_among_reported": ratio(within_limit, len(latencies)),
            "within_limit_rate_all_videos": ratio(within_limit, total),
            "successful_within_300_sec_count": successful_within_limit,
            "successful_within_limit_rate_all_videos": ratio(successful_within_limit, total),
        },
        "per_video": per_video,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--split", required=True, choices=("validation", "holdout"))
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--rules", type=Path, help="Optional explicit rule registry; titles must match exactly")
    parser.add_argument("--evidence-time-tolerance-sec", type=float,
                        help="Default: 1/fps for each video. Set 0 for exact anchor timestamp equality")
    args = parser.parse_args(argv)
    try:
        inputs = {args.manifest.resolve(), args.predictions.resolve()}
        if args.rules is not None:
            inputs.add(args.rules.resolve())
        if args.out.resolve() in inputs:
            raise DatasetError("Output must not overwrite an input manifest, predictions, or registry")
        registry = read_registry(load_json(args.rules)) if args.rules is not None else None
        report = evaluate(load_json(args.manifest), load_json(args.predictions), split=args.split,
                          registry=registry, evidence_tolerance_sec=args.evidence_time_tolerance_sec)
        report["metadata"].update(manifest_sha256=hash_file(args.manifest), predictions_sha256=hash_file(args.predictions),
                                   registry_sha256=hash_file(args.rules) if args.rules is not None else None)
        write_json(args.out, report)
    except (DatasetError, OSError) as exc:
        print(f"evaluate: {exc}", file=sys.stderr)
        return 2
    score = report["video_level"]
    print(f"Evaluation saved: {args.out}; {score['completed_videos']}/{score['total_videos']} completed, "
          f"{score['failed_videos']} failed; effective accuracy={score['effective_video_accuracy']:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
