#!/usr/bin/env python3
"""Fit an experimental temporal linear head without video/group/holdout leakage.

This script consumes local numeric features, not an online AI service. It trains
only on train rows, selects an operating threshold only on validation events and
confirmed-normal validation videos, and never evaluates holdout performance.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def assert_disjoint_splits(videos: list[dict]) -> None:
    if not videos:
        raise ValueError("Dataset manifest is empty.")
    seen_ids, seen_sha, group_splits = set(), {}, {}
    for video in videos:
        identifier = video.get("video_id")
        group = video.get("group_id")
        digest = video.get("sha256")
        split = video.get("split")
        if not isinstance(identifier, str) or not identifier or identifier in seen_ids:
            raise ValueError("Video IDs must be nonempty and unique.")
        seen_ids.add(identifier)
        if not isinstance(group, str) or not group:
            raise ValueError("Every video must have a nonempty recording/scenario group_id.")
        if split not in {"train", "validation", "holdout", "unassigned"}:
            raise ValueError("Unknown dataset split.")
        if split != "unassigned" and group.strip().casefold() in {"unassigned", "unknown", "none", "null", "n/a"}:
            raise ValueError("Assigned videos need reviewed recording/scenario group IDs; an unknown group is insufficient.")
        if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError("Every video needs a valid lowercase SHA-256 digest.")
        if digest in seen_sha:
            raise ValueError("Duplicate source SHA-256: deduplicate videos before splitting or training.")
        seen_sha[digest] = identifier
        if group in group_splits and group_splits[group] != split:
            raise ValueError("A recording/scenario group appears in different dataset splits.")
        group_splits[group] = split
        label = video.get("label")
        if label is not None and (type(label) is not int or label not in (0, 1)):
            raise ValueError("Video labels must be integer 0, integer 1, or null.")
        duration = video.get("duration_sec")
        if not isinstance(duration, (int, float)) or not math.isfinite(duration) or duration <= 0:
            raise ValueError("Every source needs a positive finite duration.")
        events = video.get("events", [])
        if not isinstance(events, list):
            raise ValueError("events must be a list.")
        if label == 0 and events:
            raise ValueError("A confirmed-normal video cannot contain anomaly annotations.")
        for event in events:
            value = event.get("timecode_sec")
            if not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= duration:
                raise ValueError("An event timecode is outside its source video.")


def build_training_rows(video_ids, timestamps, videos: list[dict], radius_seconds: float):
    """-1 means excluded, including unannotated sections of a positive video."""
    by_id = {video["video_id"]: video for video in videos}
    labels = np.full(len(video_ids), -1, dtype=np.int64)
    splits = np.empty(len(video_ids), dtype="U16")
    for index, (identifier, timestamp) in enumerate(zip(video_ids, timestamps)):
        if identifier not in by_id:
            raise ValueError("Features reference an unknown source video.")
        video = by_id[identifier]
        if not np.isfinite(timestamp) or not 0 <= timestamp <= video["duration_sec"] + 0.001:
            raise ValueError("Feature timestamp lies outside the labeled video.")
        splits[index] = video["split"]
        if video["label"] == 0:
            labels[index] = 0
        elif video["label"] == 1:
            if any(abs(float(timestamp) - event["timecode_sec"]) <= radius_seconds for event in video.get("events", [])):
                labels[index] = 1
    return labels, splits


def per_video_class_weights(video_ids, labels):
    counts = Counter(video_ids.tolist())
    videos_by_class = {label: set(video_ids[labels == label].tolist()) for label in (0, 1)}
    if not all(videos_by_class.values()):
        raise ValueError("Both classes are required for fitting.")
    weights = np.array([
        1.0 / (counts[identifier] * len(videos_by_class[int(label)]))
        for identifier, label in zip(video_ids, labels)
    ])
    return weights / weights.mean()


def _require_groups(video_ids, labels, videos, split_name, minimum):
    by_id = {video["video_id"]: video for video in videos}
    result = {}
    for label in (0, 1):
        identifiers = set(video_ids[labels == label].tolist())
        groups = {by_id[identifier]["group_id"] for identifier in identifiers}
        result[str(label)] = {"videos": len(identifiers), "groups": len(groups)}
        if len(groups) < minimum:
            raise ValueError(
                f"Insufficient {split_name} data: label={label} has {len(groups)} independent groups, "
                f"requires at least {minimum}. Do not split frames from one video to manufacture samples."
            )
    return result


def validation_event_scores(probabilities, video_ids, timestamps, videos, radius_seconds):
    event_scores, normal_scores = [], []
    event_source_ids, normal_source_ids = [], []
    for video in videos:
        if video["split"] != "validation" or video["label"] is None:
            continue
        rows = video_ids == video["video_id"]
        if not np.any(rows):
            raise ValueError("A labeled validation video has no extracted feature rows.")
        if video["label"] == 0:
            normal_scores.append(float(probabilities[rows].max()))
            normal_source_ids.append(video["video_id"])
        else:
            for event in video.get("events", []):
                neighborhood = rows & (np.abs(timestamps - event["timecode_sec"]) <= radius_seconds)
                if not np.any(neighborhood):
                    raise ValueError("An annotated validation event has no sampled feature window within the event radius.")
                event_scores.append(float(probabilities[neighborhood].max()))
                event_source_ids.append(video["video_id"])
    return np.array(event_scores), np.array(normal_scores), event_source_ids, normal_source_ids


def select_threshold(event_scores, normal_scores, target_recall=0.95, max_normal_fpr=None):
    if len(event_scores) == 0 or len(normal_scores) == 0:
        raise ValueError("Threshold selection requires annotated validation events and confirmed-normal videos.")
    if not np.all(np.isfinite(event_scores)) or not np.all(np.isfinite(normal_scores)):
        raise ValueError("Validation scores must be finite.")
    candidates = np.unique(np.concatenate(([0.0, 1.0], event_scores, normal_scores)))
    feasible = []
    for threshold in candidates:
        recall = float(np.mean(event_scores >= threshold))
        false_positive_rate = float(np.mean(normal_scores >= threshold))
        if recall + 1e-12 < target_recall:
            continue
        if max_normal_fpr is not None and false_positive_rate > max_normal_fpr + 1e-12:
            continue
        feasible.append((false_positive_rate, -float(threshold), recall))
    if not feasible:
        return {"feasible": False, "threshold": None, "reason": "No validation operating point satisfies the requested recall/FPR constraints."}
    false_positive_rate, neg_threshold, recall = min(feasible)
    return {"feasible": True, "threshold": -neg_threshold,
            "validation_event_recall": recall, "validation_normal_video_fpr": false_positive_rate}


def fit_temporal_head(
    embeddings, video_ids, timestamps, metadata, videos,
    *, radius_seconds=2.0, target_recall=0.95, max_normal_fpr=None, minimum_groups=2, regularization_c=0.1, seed=42,
):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    import sklearn
    if not math.isfinite(radius_seconds) or radius_seconds <= 0 or not 0 < target_recall <= 1:
        raise ValueError("Use a positive finite event radius and target recall in (0,1].")
    if minimum_groups < 2 or not math.isfinite(regularization_c) or regularization_c <= 0:
        raise ValueError("Require at least two independent groups per class/split and positive finite C.")
    if max_normal_fpr is not None and not 0 <= max_normal_fpr <= 1:
        raise ValueError("FPR constraint must be in [0,1].")
    assert_disjoint_splits(videos)
    if embeddings.ndim != 2 or len(embeddings) != len(video_ids) or len(video_ids) != len(timestamps):
        raise ValueError("Feature arrays have incompatible dimensions.")
    if embeddings.shape[1] != metadata.get("feature_dimension") or not metadata.get("model_revision"):
        raise ValueError("Feature metadata is missing the correct dimension or model revision.")
    by_id = {video["video_id"]: video for video in videos}
    for identifier in set(video_ids.tolist()):
        if identifier not in by_id or metadata.get("source_sha256", {}).get(identifier) != by_id[identifier]["sha256"]:
            raise ValueError("Feature/source SHA-256 provenance differs from the dataset manifest.")
    labels, splits = build_training_rows(video_ids, timestamps, videos, radius_seconds)
    train_mask = (splits == "train") & (labels >= 0)
    validation_mask = (splits == "validation") & (labels >= 0)
    train_counts = _require_groups(video_ids[train_mask], labels[train_mask], videos, "train", minimum_groups)
    validation_counts = _require_groups(video_ids[validation_mask], labels[validation_mask], videos, "validation", minimum_groups)
    train_x, train_y = embeddings[train_mask], labels[train_mask]
    if not np.all(np.isfinite(train_x)):
        raise ValueError("Train features contain a non-finite value.")
    weights = per_video_class_weights(video_ids[train_mask], train_y)
    scaler = StandardScaler()
    # No validation or holdout row is passed to fit or fit_transform.
    scaler.fit(train_x, sample_weight=weights)
    classifier = LogisticRegression(C=regularization_c, solver="lbfgs", max_iter=2000, random_state=seed)
    classifier.fit(scaler.transform(train_x), train_y, sample_weight=weights)
    if int(classifier.n_iter_.max()) >= 2000:
        raise ValueError("Logistic regression did not converge. Inspect features before trusting this artifact.")
    all_validation_mask = splits == "validation"
    validation_x = embeddings[all_validation_mask]
    if not np.all(np.isfinite(validation_x)):
        raise ValueError("Validation features contain a non-finite value.")
    validation_probabilities = classifier.predict_proba(scaler.transform(validation_x))[:, 1]
    event_scores, normal_scores, event_ids, normal_ids = validation_event_scores(
        validation_probabilities, video_ids[all_validation_mask], timestamps[all_validation_mask], videos, radius_seconds,
    )
    operating_point = select_threshold(event_scores, normal_scores, target_recall, max_normal_fpr)
    report = {
        "experimental": True, "calibration": operating_point,
        "target_validation_event_recall": target_recall, "max_normal_video_fpr": max_normal_fpr,
        "train_independent_units_by_class": train_counts,
        "validation_independent_units_by_class": validation_counts,
        "train_labeled_windows": int(train_mask.sum()), "validation_labeled_windows": int(validation_mask.sum()),
        "validation_annotated_events": len(event_scores), "validation_normal_videos": len(normal_scores),
        "validation_event_video_ids": event_ids, "validation_normal_video_ids": normal_ids,
        "positive_windows_outside_event_radius_excluded": int(np.sum(
            (labels < 0) & np.isin(splits, ["train", "validation"]) &
            np.array([by_id[identifier]["label"] == 1 for identifier in video_ids])
        )),
        "holdout_used_for_training": False, "holdout_used_for_scaling": False,
        "holdout_used_for_threshold": False, "holdout_evaluated": False,
        "warning": "A threshold fitted to a small validation set does not establish out-of-domain or operational recall. Event-time neighborhoods are weak labels, not exact action intervals.",
    }
    artifact = {
        "schema_version": 1, "model_kind": "experimental_frozen_dinov2_temporal_logistic_v1",
        "enabled_in_default_analyzer": False, "feature_metadata": metadata,
        "scaler": {"mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist()},
        "classifier": {"coef": classifier.coef_[0].tolist(), "intercept": float(classifier.intercept_[0]), "classes": [0, 1]},
        "threshold": operating_point.get("threshold"), "calibration_feasible": operating_point["feasible"],
        "settings": {"event_radius_seconds": radius_seconds, "target_recall": target_recall,
                     "minimum_groups_per_class_per_split": minimum_groups, "regularization_c": regularization_c,
                     "seed": seed, "sample_weighting": "equal_total_per_video_then_equal_total_per_class"},
        "fit_video_ids": sorted(set(video_ids[train_mask].tolist())),
        "threshold_video_ids": sorted(set(video_ids[all_validation_mask].tolist())),
        "sklearn_version": sklearn.__version__,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    return artifact, report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--event-radius-seconds", type=float, default=2.0)
    parser.add_argument("--target-recall", type=float, default=0.95)
    parser.add_argument("--max-normal-video-fpr", type=float)
    parser.add_argument("--minimum-groups", type=int, default=2)
    parser.add_argument("--regularization-c", type=float, default=0.1)
    args = parser.parse_args()
    if args.event_radius_seconds <= 0 or not 0 < args.target_recall <= 1 or args.minimum_groups < 2 or args.regularization_c <= 0:
        parser.error("Use positive radius/C, recall in (0,1], and at least two independent groups per class/split.")
    if args.max_normal_video_fpr is not None and not 0 <= args.max_normal_video_fpr <= 1:
        parser.error("FPR constraint must be in [0,1].")
    report_path = args.report or args.output.with_suffix(".report.json")
    if args.output.exists() or report_path.exists():
        parser.error("Output/report already exists. Use new artifact paths.")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1:
        raise ValueError("Expected manifest schema_version=1.")
    with np.load(args.features, allow_pickle=False) as features:
        artifact, report = fit_temporal_head(
            features["embeddings"], features["video_id"], features["timestamp_sec"],
            json.loads(str(features["metadata_json"].item())), manifest["videos"],
            radius_seconds=args.event_radius_seconds, target_recall=args.target_recall,
            max_normal_fpr=args.max_normal_video_fpr, minimum_groups=args.minimum_groups,
            regularization_c=args.regularization_c,
        )
    feature_hash = hashlib.sha256()
    with args.features.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            feature_hash.update(chunk)
    artifact["source_feature_file_sha256"] = feature_hash.hexdigest()
    for path, value in ((args.output, artifact), (report_path, report)):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"artifact": str(args.output), "report": str(report_path),
                      "calibration": report["calibration"], "holdout_evaluated": False}, indent=2))
    if not artifact["calibration_feasible"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
