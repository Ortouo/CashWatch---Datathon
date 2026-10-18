"""Synthetic integrity checks for the deliberately non-deployable four-source fit."""
from __future__ import annotations

import copy
import hashlib
import json
import stat
import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
import functional_smoke as smoke


def source(identifier, label, *, sha=None, split="unassigned", group="unassigned", timecode=5.0):
    return {"video_id": identifier, "path": f"not-present/{identifier}.mov",
            "sha256": sha or hashlib.sha256(identifier.encode()).hexdigest(),
            "duration_sec": 10.0, "label": label,
            "events": [{"timecode_sec": timecode, "rule_id": None}] if label == 1 else [],
            "group_id": group, "camera_id": "unassigned", "split": split}


def base_manifest():
    return {"schema_version": 1, "metadata": {"input_dir": "/media-not-present"}, "videos": [
        source("positive-a", 1), source("positive-b", 1),
        source("normal-a", 0), source("normal-b", 0),
    ]}


def test_selection_is_order_independent_and_reserves_all_aliases_without_choosing_ambiguous_timecode():
    manifest = base_manifest()
    manifest["videos"] += [
        source("normal-a-alias", 0, sha=manifest["videos"][2]["sha256"]),
        source("ambiguous-a", 1, sha="0" * 64, timecode=2),
        source("ambiguous-b", 1, sha="0" * 64, timecode=8),
    ]
    original = copy.deepcopy(manifest)
    selected, ledger = smoke.select_sources(manifest)
    assert manifest == original
    reversed_selection, _ = smoke.select_sources({**manifest, "videos": manifest["videos"][::-1]})
    assert selected == reversed_selection
    assert len({video["sha256"] for video in selected}) == 4
    assert "0" * 64 not in {video["sha256"] for video in selected}
    assert ledger["selected_sha_alias_entries"] == 5
    assert ledger["excluded_sha_groups_local_only"][0]["reason"] == "identical_bytes_differing_annotations_require_human_review"
    assert ledger["reviewed_scenario_groups_available_for_all_selected"] is False


def test_existing_holdout_sha_alias_and_known_group_are_excluded_before_source_access():
    manifest = base_manifest()
    manifest["videos"] += [
        source("protected-train-alias", 1, sha="0" * 64),
        source("protected-holdout-alias", 1, sha="0" * 64, split="holdout"),
        source("protected-group-train", 0, sha="1" * 64, group="reviewed-scene"),
        source("protected-group-validation", 0, group="reviewed-scene", split="validation"),
    ]
    selected, ledger = smoke.select_sources(manifest)
    assert all(not video["video_id"].startswith("protected") for video in selected)
    reasons = {entry["reason"] for entry in ledger["excluded_sha_groups_local_only"]}
    assert "sha_reserved_for_validation_or_holdout" in reasons
    assert "known_group_reserved_for_validation_or_holdout" in reasons


def test_protection_follows_transitive_sha_alias_to_group_connections():
    manifest = base_manifest()
    manifest["videos"] += [
        source("protected-heldout-unknown-group", 0, sha="e" * 64, split="holdout"),
        source("protected-alias-known-group-g", 0, sha="e" * 64, group="group-g"),
        source("protected-other-sha-group-g", 0, sha="0" * 64, group="group-g"),
        source("protected-same-sha-group-h", 0, sha="0" * 64, group="group-h"),
        source("protected-transitive-group-h", 0, sha="1" * 64, group="group-h"),
    ]
    selected, ledger = smoke.select_sources(manifest)
    assert all(not video["video_id"].startswith("protected") for video in selected)
    excluded = {entry["sha256"] for entry in ledger["excluded_sha_groups_local_only"]}
    assert {"0" * 64, "1" * 64, "e" * 64}.issubset(excluded)


def test_reservation_precedes_media_read_preserves_unknowns_and_refuses_existing_run(tmp_path):
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(base_manifest()))
    output_dir = tmp_path / "experiment"
    selected, _ = smoke.reserve_exposure(manifest_path, output_dir)
    ledger_path = output_dir / "training_exposure_ledger.local.json"
    assert json.loads(ledger_path.read_text())["status"] == "reserved_before_media_read"
    assert stat.S_IMODE(ledger_path.stat().st_mode) == 0o600
    assert stat.S_IMODE(output_dir.stat().st_mode) == 0o700
    assert all(video["split"] == "unassigned" and video["group_id"] == "unassigned" for video in selected["videos"])
    frozen_ledger = ledger_path.read_bytes()
    with pytest.raises(FileExistsError):
        smoke.reserve_exposure(manifest_path, output_dir)
    assert ledger_path.read_bytes() == frozen_ledger


def numeric_fixture():
    videos = base_manifest()["videos"]
    video_ids = np.array([video["video_id"] for video in videos for _ in range(3)])
    timestamps = np.tile(np.array([0.0, 5.0, 10.0]), 4)
    embeddings = np.random.default_rng(19).normal(size=(12, 4)).astype(np.float32)
    metadata = {"feature_dimension": 4, "model_revision": "synthetic-only",
                "source_sha256": {video["video_id"]: video["sha256"] for video in videos}}
    return embeddings, video_ids, timestamps, metadata, videos


def test_unannotated_positive_windows_cannot_influence_scaler_or_fit_and_no_quality_is_claimed():
    arrays = numeric_fixture()
    artifact, report = smoke.fit_functional_head(*arrays)
    changed = arrays[0].copy()
    changed[[0, 2, 3, 5]] = 100000
    changed_artifact, _ = smoke.fit_functional_head(changed, *arrays[1:])
    assert artifact["scaler"] == changed_artifact["scaler"]
    assert artifact["classifier"] == changed_artifact["classifier"]
    assert artifact["threshold"] is None
    assert artifact["deployment_ready"] is False
    assert artifact["enabled_in_default_analyzer"] is False
    assert report["quality_metrics"] is None
    assert report["holdout_evaluated"] is False
    assert report["fit_positive_windows"] == 2
    assert report["fit_normal_windows"] == 6
    assert report["positive_windows_outside_event_radius_excluded"] == 4


def test_features_with_extra_unknown_source_or_incorrect_hash_are_rejected():
    arrays = numeric_fixture()
    extra_metadata = copy.deepcopy(arrays[3])
    extra_metadata["source_sha256"]["unreserved-holdout"] = "a" * 64
    with pytest.raises(ValueError, match="exactly the four reserved"):
        smoke.fit_functional_head(*arrays[:3], extra_metadata, arrays[4])
    changed_ids = arrays[1].copy()
    changed_ids[0] = "unreserved"
    with pytest.raises(ValueError, match="exactly the four reserved"):
        smoke.fit_functional_head(arrays[0], changed_ids, *arrays[2:])
    wrong_hash = copy.deepcopy(arrays[3])
    wrong_hash["source_sha256"][arrays[4][0]["video_id"]] = "f" * 64
    with pytest.raises(ValueError, match="provenance differ"):
        smoke.fit_functional_head(*arrays[:3], wrong_hash, arrays[4])


def test_insufficient_positive_sampling_fails_instead_of_silent_partial_fit():
    arrays = numeric_fixture()
    changed_times = arrays[2].copy()
    changed_times[1] = 0.0
    with pytest.raises(ValueError, match="Each reserved video"):
        smoke.fit_functional_head(*arrays[:2], changed_times, *arrays[3:])
