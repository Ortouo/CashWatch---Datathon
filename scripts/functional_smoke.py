#!/usr/bin/env python3
"""Run one local, four-video functional fit; this is not a quality evaluation.

The immutable exposure ledger is written before extraction. Only two annotated
positive and two confirmed-normal, unique-SHA sources are passed to DINOv2.
Unknown scenario groups remain unknown. This script never selects a threshold
and cannot produce a deployment-ready artifact.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import json
import math
import os
import signal
import subprocess
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from preflight import OFFLINE_ENV, host_info, package_versions, write_private_json

UNKNOWN_GROUPS = {"", "unassigned", "unknown", "none", "null", "n/a"}


def sha256_file(path: Path) -> str:
    checksum = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            checksum.update(chunk)
    return checksum.hexdigest()


def known_group(value) -> str | None:
    return value if isinstance(value, str) and value.strip().casefold() not in UNKNOWN_GROUPS else None


def annotation_signature(video: dict) -> tuple:
    return (video["label"], tuple(sorted(
        (float(event["timecode_sec"]), event.get("rule_id") or "") for event in video["events"]
    )))


def select_sources(manifest: dict) -> tuple[list[dict], dict]:
    """Use manifest metadata only: do not open, hash, decode, or inspect media.

    Differing annotations for identical bytes can represent multiple events and
    are not declared an annotation error. Such SHA groups are simply excluded
    pending human review; no arbitrary alias's timecode wins.
    """
    videos = manifest.get("videos")
    if manifest.get("schema_version") != 1 or not isinstance(videos, list) or not videos:
        raise ValueError("Expected a nonempty version 1 manifest.")
    seen_ids, by_sha = set(), defaultdict(list)
    protected_groups = set()
    for video in videos:
        if not isinstance(video, dict):
            raise ValueError("Manifest video entries must be objects.")
        identifier, digest = video.get("video_id"), video.get("sha256")
        if not isinstance(identifier, str) or not identifier or identifier in seen_ids:
            raise ValueError("Source IDs must be nonempty and unique.")
        seen_ids.add(identifier)
        if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError("Each source must have a valid lowercase SHA-256.")
        if not isinstance(video.get("path"), str) or not video["path"]:
            raise ValueError("Each source needs a local path.")
        split = video.get("split")
        if split not in {"train", "validation", "holdout", "unassigned"}:
            raise ValueError("Unknown dataset split.")
        label, duration = video.get("label"), video.get("duration_sec")
        if label is not None and (type(label) is not int or label not in {0, 1}):
            raise ValueError("Labels must be integer zero, integer one, or null.")
        if type(duration) not in {int, float} or not math.isfinite(duration) or duration <= 0:
            raise ValueError("Video duration must be finite and positive.")
        events = video.get("events")
        if not isinstance(events, list) or (label == 0 and events):
            raise ValueError("Normal sources need empty event annotations.")
        for event in events:
            timestamp = event.get("timecode_sec") if isinstance(event, dict) else None
            if type(timestamp) not in {int, float} or not math.isfinite(timestamp) or not 0 <= timestamp <= duration:
                raise ValueError("An event timecode is invalid or outside its video.")
            if event.get("rule_id") is not None and not isinstance(event["rule_id"], str):
                raise ValueError("rule_id must be a string or null.")
        if split in {"validation", "holdout"}:
            group = known_group(video.get("group_id"))
            if group is not None:
                protected_groups.add(group)
        by_sha[digest].append(video)

    # A held-out SHA can acquire a reviewed group through another alias. Follow
    # the SHA <-> group graph to its fixed point; otherwise a third video in that
    # group could leak into this functional fit despite sharing protected data.
    protected_sha = {digest for digest, aliases in by_sha.items()
                     if any(video["split"] in {"validation", "holdout"} for video in aliases)}
    changed = True
    while changed:
        before = (len(protected_sha), len(protected_groups))
        for digest, aliases in by_sha.items():
            groups = {group for video in aliases if (group := known_group(video.get("group_id"))) is not None}
            if digest in protected_sha or groups.intersection(protected_groups):
                protected_sha.add(digest)
                protected_groups.update(groups)
        changed = before != (len(protected_sha), len(protected_groups))

    eligible = {0: [], 1: []}
    excluded = []
    for digest, aliases in sorted(by_sha.items()):
        reason = None
        if len({annotation_signature(video) for video in aliases}) != 1:
            reason = "identical_bytes_differing_annotations_require_human_review"
        elif any(video["split"] in {"validation", "holdout"} for video in aliases):
            reason = "sha_reserved_for_validation_or_holdout"
        elif digest in protected_sha:
            reason = "known_group_reserved_for_validation_or_holdout"
        else:
            representative = min(aliases, key=lambda video: video["video_id"])
            if representative["label"] is None or (representative["label"] == 1 and not representative["events"]):
                reason = "not_explicitly_normal_or_time_annotated_positive"
        if reason:
            excluded.append({"sha256": digest, "reason": reason,
                             "video_ids_local_only": sorted(video["video_id"] for video in aliases)})
        else:
            eligible[representative["label"]].append(representative)
    if any(len(eligible[label]) < 2 for label in (0, 1)):
        raise ValueError("Functional fit needs two unambiguous unique-SHA sources per class outside protected splits/groups.")
    selected = sorted(eligible[0][:2] + eligible[1][:2], key=lambda video: video["sha256"])
    selected_sha = {video["sha256"] for video in selected}
    reservations = [
        {"sha256": digest, "aliases": [
            {"video_id": video["video_id"], "original_split": video["split"],
             "group_id": video.get("group_id"), "camera_id": video.get("camera_id")}
            for video in sorted(by_sha[digest], key=lambda entry: entry["video_id"])
        ]}
        for digest in sorted(selected_sha)
    ]
    details = {
        "selection_policy": "ascending_sha256_per_class_2_positive_2_normal_unique_bytes_v1",
        "ambiguous_annotation_policy": "exclude_entire_sha_group_pending_human_review",
        "eligible_unique_sha_by_class": {str(label): len(eligible[label]) for label in (0, 1)},
        "manifest_source_entries": len(videos), "manifest_unique_sha": len(by_sha),
        "selected_unique_sha": len(selected_sha), "selected_source_entries": len(selected),
        "selected_sha_alias_entries": sum(len(by_sha[digest]) for digest in selected_sha),
        "excluded_sha_groups_local_only": excluded,
        "sha_exposure_reservations_local_only": reservations,
        "reviewed_scenario_groups_available_for_all_selected": all(known_group(video.get("group_id")) for video in selected),
        "group_ids_inferred": False,
        "future_holdout_instruction": "Exclude selected SHA aliases and, after human grouping, all their scenario groups from any internal holdout.",
    }
    return copy.deepcopy(selected), details


def reserve_exposure(manifest_path: Path, output_dir: Path) -> tuple[dict, dict]:
    """Create an immutable reservation before any source media is opened."""
    source_bytes = manifest_path.read_bytes()
    manifest = json.loads(source_bytes)
    selected, selection = select_sources(manifest)
    input_root = manifest.get("metadata", {}).get("input_dir")
    if not isinstance(input_root, str) or not Path(input_root).is_absolute():
        raise ValueError("Functional extraction requires absolute metadata.input_dir.")
    # A new directory prevents a second invocation from silently changing exposure.
    output_dir.mkdir(mode=0o700, parents=True, exist_ok=False)
    selected_manifest = {
        "schema_version": 1,
        "metadata": {"input_dir": input_root, "purpose": "functional_fit_only",
                     "path_mode": "path and video_id retain their original manifest values",
                     "source_manifest_sha256": hashlib.sha256(source_bytes).hexdigest()},
        "videos": selected,
    }
    # Preserve unknown group/camera values and original splits. The strict
    # production trainer will still reject unreviewed groups and absent validation.
    ledger = {
        "schema_version": 1, "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "reserved_before_media_read", "purpose": "functional_fit_only",
        "source_manifest_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "selected_source_video_ids_local_only": [video["video_id"] for video in selected],
        **selection,
        "irreversible_training_exposure_if_run": True, "threshold_will_be_selected": False,
        "independent_validation_or_holdout_claim_permitted": False,
    }
    write_private_json(output_dir / "training_exposure_ledger.local.json", ledger)
    write_private_json(output_dir / "manifest.selected.local.json", selected_manifest)
    return selected_manifest, ledger


def fit_functional_head(embeddings, video_ids, timestamps, metadata, selected_videos, *, radius_seconds=2.0):
    """Fit only reserved sources, with no threshold, validation, or quality metric."""
    import numpy as np
    import sklearn
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from train_temporal_head import build_training_rows, per_video_class_weights

    if not math.isfinite(radius_seconds) or radius_seconds <= 0:
        raise ValueError("Event radius must be finite and positive.")
    by_id = {video["video_id"]: video for video in selected_videos}
    if len(by_id) != 4 or Counter(video["label"] for video in selected_videos) != {0: 2, 1: 2}:
        raise ValueError("Only the reserved two-positive/two-normal subset can be fitted.")
    if len({video["sha256"] for video in selected_videos}) != 4:
        raise ValueError("Functional sources must have distinct SHA-256 values.")
    if any(video["split"] not in {"train", "unassigned"} for video in selected_videos):
        raise ValueError("Protected validation or holdout sources cannot be fitted.")
    if embeddings.ndim != 2 or video_ids.ndim != 1 or timestamps.ndim != 1:
        raise ValueError("Invalid feature array ranks.")
    if len(embeddings) != len(video_ids) or len(video_ids) != len(timestamps) or not len(embeddings):
        raise ValueError("Feature arrays have incompatible lengths.")
    if not np.all(np.isfinite(embeddings)) or not np.all(np.isfinite(timestamps)):
        raise ValueError("Features and timestamps must be finite.")
    if embeddings.shape[1] != metadata.get("feature_dimension") or not metadata.get("model_revision"):
        raise ValueError("Feature dimension or model revision is absent or inconsistent.")
    if set(video_ids.tolist()) != set(by_id) or set(metadata.get("source_sha256", {})) != set(by_id):
        raise ValueError("Features must contain exactly the four reserved source IDs.")
    for identifier, video in by_id.items():
        if metadata["source_sha256"][identifier] != video["sha256"]:
            raise ValueError("Reserved source and feature SHA-256 provenance differ.")
    labels, _ = build_training_rows(video_ids, timestamps, selected_videos, radius_seconds)
    included = labels >= 0
    if set(video_ids[included].tolist()) != set(by_id):
        raise ValueError("Each reserved video needs a usable labeled feature window.")
    x, y = embeddings[included], labels[included]
    weights = per_video_class_weights(video_ids[included], y)
    scaler = StandardScaler().fit(x, sample_weight=weights)
    classifier = LogisticRegression(C=0.1, solver="lbfgs", max_iter=2000, random_state=42)
    classifier.fit(scaler.transform(x), y, sample_weight=weights)
    iterations = int(classifier.n_iter_.max())
    if iterations >= 2000:
        raise ValueError("Functional logistic fit did not converge.")
    artifact = {
        "schema_version": 1, "model_kind": "functional_only_frozen_dinov2_temporal_logistic_v1",
        "purpose": "functional_fit_only", "deployment_ready": False, "enabled_in_default_analyzer": False,
        "threshold": None, "calibration_performed": False,
        "feature_metadata": metadata,
        "scaler": {"mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist()},
        "classifier": {"coef": classifier.coef_[0].tolist(), "intercept": float(classifier.intercept_[0]), "classes": [0, 1]},
        "settings": {"event_radius_seconds": radius_seconds, "regularization_c": 0.1, "seed": 42,
                     "sample_weighting": "equal_total_per_video_then_equal_total_per_class"},
        "fit_video_ids_local_only": sorted(by_id), "sklearn_version": sklearn.__version__,
        "scenario_group_independence_established": False, "camera_generalization_established": False,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    report = {
        "purpose": "functional_fit_only", "fit_completed": True, "fit_converged": True,
        "optimizer_iterations": iterations, "selected_videos": 4,
        "selected_positive_videos": 2, "selected_normal_videos": 2,
        "feature_windows": int(len(embeddings)), "feature_dimension": int(embeddings.shape[1]),
        "fit_labeled_windows": int(included.sum()),
        "fit_positive_windows": int((labels == 1).sum()), "fit_normal_windows": int((labels == 0).sum()),
        "positive_windows_outside_event_radius_excluded": int((labels < 0).sum()),
        "model_revision": metadata["model_revision"],
        "threshold": None, "deployment_ready": False, "calibration_performed": False,
        "validation_evaluated": False, "holdout_evaluated": False, "quality_metrics": None,
        "scenario_group_independence_established": False, "camera_generalization_established": False,
        "warning": "Four-source functional fit only. Event neighborhoods are weak labels. Unknown scenario/camera groups prevent an independent quality claim.",
    }
    return artifact, report


def extract_reserved_subset(command: list[str], output_dir: Path, timeout_sec: float, environment: dict) -> float:
    """Keep diagnostics private, bound the extractor and its FFmpeg descendants."""
    started = time.monotonic()
    log_path = output_dir / "extraction.log.local.txt"
    descriptor = os.open(log_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as log:
        process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                   env=environment, close_fds=True, start_new_session=True)
        try:
            return_code = process.wait(timeout=timeout_sec)
        except BaseException:
            if process.poll() is None:
                if os.name == "posix":
                    os.killpg(process.pid, signal.SIGKILL)
                else:
                    process.kill()
                process.wait()
            raise
    if return_code != 0:
        raise RuntimeError("Local feature extraction failed; private diagnostics retained.")
    return time.monotonic() - started


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=["cpu", "mps", "cuda"], default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--extract-timeout-seconds", type=float, default=900.0)
    args = parser.parse_args()
    if not 1 <= args.threads <= 32 or not math.isfinite(args.extract_timeout_seconds) or not 10 <= args.extract_timeout_seconds <= 3600:
        parser.error("Use 1..32 threads and an extraction timeout of 10..3600 seconds.")
    output_dir = args.output_dir.resolve()
    stage = "reserve_exposure"
    owned_output = False
    started = time.monotonic()
    try:
        selected_manifest, ledger = reserve_exposure(args.manifest.resolve(strict=True), output_dir)
        owned_output = True
        stage = "network_guard"
        os.environ.update(OFFLINE_ENV)
        os.environ.update({"OMP_NUM_THREADS": str(args.threads), "MKL_NUM_THREADS": str(args.threads),
                           "TOKENIZERS_PARALLELISM": "false"})
        from cashwatch.network_guard import enforce_network_guard
        guard = enforce_network_guard()  # Before Torch, NumPy, sklearn, or their children.
        write_private_json(output_dir / "run_state.local.json", {"stage": stage, "network_isolation": guard})
        print(json.dumps({"stage": "extraction", "reserved_unique_videos": 4,
                          "bank_media_or_descriptions_emitted": False}), flush=True)
        stage = "extract_features"
        feature_path = output_dir / "features.local.npz"
        extraction_elapsed = extract_reserved_subset([
            sys.executable, str(PROJECT_ROOT / "scripts" / "extract_features.py"),
            "--manifest", str(output_dir / "manifest.selected.local.json"),
            "--model-dir", str(args.model_dir.resolve(strict=True)), "--output", str(feature_path),
            "--device", args.device, "--sample-fps", "2", "--window-seconds", "6",
            "--stride-seconds", "2", "--batch-size", "8", "--threads", str(args.threads),
        ], output_dir, args.extract_timeout_seconds, dict(os.environ))
        stage = "functional_fit"
        fit_started = time.monotonic()
        import numpy as np
        with np.load(feature_path, allow_pickle=False) as features:
            metadata = json.loads(str(features["metadata_json"].item()))
            artifact, report = fit_functional_head(
                features["embeddings"], features["video_id"], features["timestamp_sec"], metadata,
                selected_manifest["videos"],
            )
        artifact["source_feature_file_sha256"] = sha256_file(feature_path)
        artifact["training_exposure_ledger_sha256"] = sha256_file(output_dir / "training_exposure_ledger.local.json")
        fit_elapsed = time.monotonic() - fit_started
        write_private_json(output_dir / "head.functional.local.json", artifact)
        report.update({
            "status": "complete", "hardware": host_info(), "packages": package_versions(),
            "training_packages": {name: importlib.metadata.version(name) for name in ("numpy", "scikit-learn")},
            "device": args.device, "configured_cpu_threads": args.threads,
            "feature_extraction_elapsed_sec": extraction_elapsed, "functional_fit_elapsed_sec": fit_elapsed,
            "elapsed_sec": time.monotonic() - started,
            "sampled_frames": metadata.get("total_sampled_frames"),
            "feature_process_peak_rss_bytes": metadata.get("process_peak_rss_bytes"),
            "rss_scope": "Feature extraction Python process only; excludes separate FFmpeg process memory.",
            "network_isolation": guard,
            "ambiguous_sha_groups_excluded": sum(
                entry["reason"] == "identical_bytes_differing_annotations_require_human_review"
                for entry in ledger["excluded_sha_groups_local_only"]),
            "source_files_opened_for_feature_extraction": 4,
            "selected_total_duration_sec": sum(video["duration_sec"] for video in selected_manifest["videos"]),
            "unselected_source_entries_not_opened_by_this_extractor": ledger["manifest_source_entries"] - 4,
            "unique_sha_exposed": 4, "selected_sha_alias_entries_reserved": ledger["selected_sha_alias_entries"],
            "bank_media_sent_to_external_inference": False,
        })
        write_private_json(output_dir / "report.aggregate.json", report)
        write_private_json(output_dir / "run_state.local.json", {"stage": "complete", "status": "complete"})
        print(json.dumps(report, ensure_ascii=False, allow_nan=False), flush=True)
        return 0
    except Exception as exc:
        # Never echo a source name, annotation, FFmpeg stderr, or raw exception text.
        failure = {"status": "failed", "stage": stage, "error_class": type(exc).__name__,
                   "deployment_ready": False, "quality_metrics": None,
                   "elapsed_sec": time.monotonic() - started}
        if owned_output:
            write_private_json(output_dir / "run_state.local.json", failure)
            write_private_json(output_dir / "report.aggregate.json", failure)
        print(json.dumps(failure), flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
