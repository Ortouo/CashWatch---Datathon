import base64
from dataclasses import replace

import pytest
from PIL import Image

from cashwatch.engine import IncompleteAnalysisError, build_report
from cashwatch.rules import RuleRegistry
from cashwatch.vision.contracts import AnalysisResult, EvidenceFrame, Observation, VisionError


@pytest.fixture
def fixture(tmp_path):
    frames = []
    for i, timestamp in enumerate((0.5, 1.0, 1.5)):
        path = tmp_path / f"evidence{i}.jpg"
        Image.new("RGB", (32, 32), (i * 50, 20, 10)).save(path)
        frames.append(EvidenceFrame(f"f{i}", timestamp, path))
    registry = RuleRegistry.model_validate({"schema_version": 1, "rules": [{
        "rule_id": "fixture_rule", "title": "Test-only synthetic rule", "requirement": "A fixture condition",
        "required_conditions": ["Visible change", "Visible context"], "disqualifiers": [],
        "source": {"title": "Test source", "url": "https://adilet.zan.kz/rus/docs/TEST", "clause": "test"},
        "verification_status": "verified_public_text", "enabled": True, "minimum_evidence_frames": 2,
    }]})
    observation = Observation("o1", "fixture_rule", 0.5, 1.5, "Synthetic observation", ("f0", "f1", "f2"), "supported", {"c1": ("f0", "f1"), "c2": ("f2",)})
    result = AnalysisResult((observation,), tuple(frames), 2.0, ((0.0, 2.0),), True, (), "fixture", "test")
    return tmp_path, registry, result


def test_real_evidence_time_not_generated_time(fixture):
    root, registry, result = fixture
    response, details = build_report("a", result, registry, root)
    assert response["verdict"] == "violation"
    event = response["violations"][0]
    assert event["timecode_sec"] == 0.5
    assert "Test source" in event["description"]
    assert len(event["evidence_frames"]) == 3
    assert base64.b64decode(event["evidence_frames"][0]["image_base64"], validate=True).startswith(b"\xff\xd8")
    assert details["events"][0]["rule_id"] == "fixture_rule"


def test_missing_condition_never_becomes_supported_violation(fixture):
    root, registry, result = fixture
    changed = replace(result.observations[0], condition_evidence={"c1": ("f0",)})
    with pytest.raises(VisionError):
        build_report("a", replace(result, observations=(changed,)), registry, root)


def test_incomplete_scan_never_yields_clean_verdict(fixture):
    root, registry, result = fixture
    with pytest.raises(IncompleteAnalysisError):
        build_report("a", replace(result, observations=(), complete=False), registry, root)


def test_supported_event_survives_an_unprocessed_region(fixture):
    root, registry, result = fixture
    response, details = build_report("a", replace(result, complete=False), registry, root)
    assert response["verdict"] == "violation"
    assert details["complete"] is False
    assert details["warnings"]


def test_no_enabled_rules_cannot_yield_clean_verdict(fixture):
    root, registry, result = fixture
    registry.rules[0].enabled = False
    with pytest.raises(IncompleteAnalysisError):
        build_report("a", replace(result, observations=()), registry, root)


@pytest.mark.parametrize("rule_id", [None, "fixture_rule"])
def test_unresolved_observation_never_yields_clean_verdict(fixture, rule_id):
    root, registry, result = fixture
    changed = replace(result.observations[0], rule_id=rule_id, assessment="uncertain", uncertainty="Unresolved event")
    with pytest.raises(IncompleteAnalysisError):
        build_report("a", replace(result, observations=(changed,)), registry, root)


def test_supported_event_preserved_with_unresolved_extra_observation(fixture):
    root, registry, result = fixture
    changed = replace(result.observations[0], observation_id="unknown", rule_id=None, assessment="uncertain", uncertainty="Unknown event")
    response, details = build_report("a", replace(result, observations=result.observations + (changed,)), registry, root)
    assert response["verdict"] == "violation"
    assert len(response["violations"]) == 1
    assert details["review_items"]


def test_complete_empty_scan_returns_required_binary_negative(fixture):
    root, registry, result = fixture
    response, _ = build_report("a", replace(result, observations=()), registry, root)
    assert response["verdict"] == "no_violation"
    assert response["violations"] == []


def test_external_path_cannot_be_used_as_evidence(fixture, tmp_path):
    root, registry, result = fixture
    other = tmp_path.parent / "outside-evidence.jpg"
    Image.new("RGB", (32, 32), "red").save(other)
    changed = replace(result.evidence_frames[0], image_path=other)
    try:
        with pytest.raises(VisionError):
            build_report("a", replace(result, evidence_frames=(changed,) + result.evidence_frames[1:]), registry, root)
    finally:
        other.unlink()


def test_impossible_timestamp_is_not_returned(fixture):
    root, registry, result = fixture
    changed = replace(result.evidence_frames[0], timestamp_sec=50)
    with pytest.raises(VisionError):
        build_report("a", replace(result, evidence_frames=(changed,) + result.evidence_frames[1:]), registry, root)


def test_overlapping_windows_do_not_duplicate_same_event(fixture):
    root, registry, result = fixture
    response, _ = build_report("a", replace(result, observations=result.observations * 2), registry, root)
    assert len(response["violations"]) == 1


def test_completeness_without_intervals_or_frames_is_rejected(fixture):
    root, registry, result = fixture
    with pytest.raises(IncompleteAnalysisError):
        build_report("a", replace(result, observations=(), evidence_frames=(), analyzed_intervals=()), registry, root)


def test_unprocessed_gap_cannot_be_marked_complete(fixture):
    root, registry, result = fixture
    with pytest.raises(IncompleteAnalysisError):
        build_report("a", replace(result, observations=(), analyzed_intervals=((0.0, 0.5), (1.0, 2.0))), registry, root)


def test_generated_boundary_not_matching_evidence_is_rejected(fixture):
    root, registry, result = fixture
    observation = replace(result.observations[0], start_sec=0.7)
    with pytest.raises(VisionError):
        build_report("a", replace(result, observations=(observation,)), registry, root)


def test_invented_supported_rule_is_an_error(fixture):
    root, registry, result = fixture
    observation = replace(result.observations[0], rule_id="invented")
    with pytest.raises(VisionError):
        build_report("a", replace(result, observations=(observation,)), registry, root)

