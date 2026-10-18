"""Regressions for case-to-result mapping and the actual returned jury evidence."""
from __future__ import annotations

import base64
import copy
import hashlib
import importlib.util
import io
import json
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from contextlib import redirect_stderr
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from PIL import Image
from cashwatch.runtime_contract import build_runtime_contract
from cashwatch.settings import Settings


def load_checker():
    path = Path(__file__).resolve().parents[1] / "scripts" / "jury_check.py"
    spec = importlib.util.spec_from_file_location("cashwatch_jury_check_under_test", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


CHECKER = load_checker()


class ManifestSelectionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="cashwatch-jury-selection-")
        self.root = Path(self.temporary.name)
        self.inputs = self.root / "input_data"
        self.inputs.mkdir()
        self.report_dir = self.root / "elsewhere"
        self.report_dir.mkdir()
        self.manifest_path = self.report_dir / "manifest.local.json"

    def tearDown(self):
        self.temporary.cleanup()

    def video(self, relative, split="validation", data=b"synthetic decoder-free fixture"):
        path = self.inputs / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return {"video_id": relative, "path": relative, "split": split,
                "duration_sec": 2.0, "sha256": hashlib.sha256(data).hexdigest()}

    def write_manifest(self, videos):
        self.manifest_path.write_text(json.dumps({
            "schema_version": 1, "metadata": {"input_dir": str(self.inputs)}, "videos": videos,
        }), encoding="utf-8")

    def test_relative_ids_survive_duplicate_basenames_and_separate_manifest_directory(self):
        one = self.video("camera_a/clip.mov", data=b"synthetic-one")
        two = self.video("camera_b/clip.mov", data=b"synthetic-two")
        self.write_manifest([one, two])
        selected = CHECKER.select_manifest_videos(self.manifest_path, "validation")
        self.assertEqual([video.video_id for video in selected], ["camera_a/clip.mov", "camera_b/clip.mov"])
        self.assertEqual([video.path for video in selected], [self.inputs / one["path"], self.inputs / two["path"]])

    def test_validation_selection_never_opens_a_holdout_video(self):
        validation = self.video("visible.mov")
        unavailable_holdout = {"video_id": "unavailable_holdout.mov", "path": "unavailable_holdout.mov", "split": "holdout"}
        self.write_manifest([validation, unavailable_holdout])
        selected = CHECKER.select_manifest_videos(self.manifest_path, "validation")
        self.assertEqual([video.video_id for video in selected], ["visible.mov"])

    def test_changed_file_is_rejected_before_upload(self):
        video = self.video("clip.mov")
        self.write_manifest([video])
        (self.inputs / "clip.mov").write_bytes(b"different bytes after labeling")
        with self.assertRaisesRegex(ValueError, "changed since manifest"):
            CHECKER.select_manifest_videos(self.manifest_path, "validation")

    def test_relative_escape_is_rejected(self):
        video = self.video("clip.mov")
        video["path"] = "../outside.mov"
        self.write_manifest([video])
        with self.assertRaisesRegex(ValueError, "inside metadata"):
            CHECKER.select_manifest_videos(self.manifest_path, "validation")

    def test_symlink_escape_is_rejected(self):
        outside = self.root / "outside.mov"
        outside.write_bytes(b"outside fixture")
        (self.inputs / "linked.mov").symlink_to(outside)
        self.write_manifest([{"video_id": "linked.mov", "path": "linked.mov", "split": "validation",
                              "duration_sec": 2.0, "sha256": hashlib.sha256(outside.read_bytes()).hexdigest()}])
        with self.assertRaisesRegex(ValueError, "outside metadata"):
            CHECKER.select_manifest_videos(self.manifest_path, "validation")

    def test_manual_duplicate_basenames_require_manifest(self):
        self.video("a/clip.mov")
        self.video("b/clip.mov")
        with self.assertRaisesRegex(ValueError, "Duplicate basenames"):
            CHECKER.select_manual_videos([self.inputs / "a/clip.mov", self.inputs / "b/clip.mov"])


class ResponseIntegrityTests(unittest.TestCase):
    def valid_response(self):
        buffer = io.BytesIO()
        Image.new("RGB", (16, 16), (30, 30, 30)).save(buffer, format="JPEG")
        return {"job_id": "a" * 32, "status": "done", "verdict": "violation", "violations": [{
            "type": "Synthetic test", "timecode_sec": 0.5, "description": "Synthetic evidence validation only",
            "evidence_frames": [{"timestamp_sec": 0.5, "image_base64": base64.b64encode(buffer.getvalue()).decode()}],
        }]}

    def test_valid_same_job_jpeg_response_is_accepted(self):
        response = CHECKER.validate_job_response(self.valid_response(), expected_job_id="a" * 32, duration_sec=2.0)
        self.assertEqual(response.status, "done")

    def test_another_job_response_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "another video's job"):
            CHECKER.validate_job_response(self.valid_response(), expected_job_id="b" * 32)

    def test_arbitrary_base64_is_not_successful_evidence(self):
        payload = self.valid_response()
        payload["violations"][0]["evidence_frames"][0]["image_base64"] = "xxxx"
        with self.assertRaisesRegex(ValueError, "invalid JPEG"):
            CHECKER.validate_job_response(payload, expected_job_id="a" * 32)

    def test_timestamp_outside_manifest_duration_is_rejected(self):
        payload = self.valid_response()
        payload["violations"][0]["evidence_frames"][0]["timestamp_sec"] = 200.0
        with self.assertRaisesRegex(ValueError, "outside the selected video"):
            CHECKER.validate_job_response(payload, expected_job_id="a" * 32, duration_sec=2.0)

    def test_malformed_status_becomes_a_controlled_error(self):
        with self.assertRaises(ValueError):
            CHECKER.validate_job_response({"job_id": "a" * 32, "status": []})


class CheckpointReportTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="cashwatch-jury-checkpoint-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.inputs = self.root / "synthetic-inputs"
        self.inputs.mkdir()
        videos = []
        for index, name in enumerate(("synthetic-a.mov", "synthetic-b.mov")):
            data = f"synthetic decoder-free fixture {index}".encode()
            (self.inputs / name).write_bytes(data)
            videos.append({"video_id": name, "path": name, "sha256": hashlib.sha256(data).hexdigest(),
                           "split": "validation", "duration_sec": 2.0, "fps": 25.0,
                           "width": 32, "height": 24, "label": index,
                           "events": [] if index == 0 else [{"timecode_sec": 0.5, "rule_id": None}],
                           "group_id": f"synthetic-group-{index}", "camera_id": "synthetic-camera"})
        self.manifest = {"schema_version": 1, "metadata": {"input_dir": str(self.inputs)}, "videos": videos}
        self.manifest_path = self.root / "manifest.local.json"
        self.manifest_path.write_text(json.dumps(self.manifest), encoding="utf-8")
        self.output = self.root / "reports" / "predictions.json"
        self.argv = ["jury_check.py", "--base-url", "http://127.0.0.1:8000", "--manifest",
                     str(self.manifest_path), "--split", "validation", "--out", str(self.output)]
        self.key = "synthetic-checkpoint-test-key"
        self.accepted_a = {"job_id": "a" * 32, "status": "processing"}
        self.accepted_b = {"job_id": "b" * 32, "status": "processing"}
        self.done_a = {"job_id": "a" * 32, "status": "done", "verdict": "no_violation", "violations": []}
        self.legacy_capabilities = {"model_configured": True, "active_rules": 1}

    def evaluate_report(self, payload):
        scripts = Path(__file__).resolve().parents[1] / "scripts"
        sys.path.insert(0, str(scripts))
        try:
            import evaluate
            return evaluate.evaluate(self.manifest, payload, split="validation")
        finally:
            sys.path.pop(0)

    def test_checkpoint_retained_when_next_video_is_interrupted_and_missing_stays_failed(self):
        responses = [self.legacy_capabilities, self.accepted_a, self.done_a,
                     self.legacy_capabilities, KeyboardInterrupt()]
        with patch.object(sys, "argv", self.argv), patch.dict(os.environ, {"CASHWATCH_API_KEY": self.key}), \
                patch.object(CHECKER, "load_dotenv"), patch.object(CHECKER, "request_json", side_effect=responses), \
                redirect_stdout(io.StringIO()), self.assertRaises(KeyboardInterrupt):
            CHECKER.main()
        text = self.output.read_text(encoding="utf-8")
        payload = json.loads(text)
        self.assertEqual([record["video_id"] for record in payload["predictions"]], ["synthetic-a.mov"])
        self.assertEqual(payload["metadata"]["recorded_video_count"], 1)
        self.assertEqual(payload["metadata"]["expected_video_count"], 2)
        self.assertFalse(payload["metadata"]["complete"])
        self.assertFalse(payload["metadata"]["run_finished"])
        self.assertEqual(payload["metadata"]["runtime_check"]["status"], "legacy_unverified")
        self.assertEqual(self.output.stat().st_mode & 0o777, 0o600)
        self.assertNotIn(self.key, text)
        self.assertNotIn(str(self.root), text)
        self.assertNotIn("http://127.0.0.1:8000", text)
        self.assertEqual(list(self.output.parent.glob("*.tmp")), [])
        report = self.evaluate_report(payload)
        self.assertEqual(report["video_level"]["total_videos"], 2)
        self.assertEqual(report["video_level"]["coverage"], 0.5)
        self.assertEqual(report["failures"]["by_reason"], {"missing_prediction": 1})
        self.assertEqual(report["video_level"]["end_to_end_positive_detection_recall"], 0.0)

    def test_final_checkpoint_is_atomic_private_and_complete_including_failed_results(self):
        failed_b = {"job_id": "b" * 32, "status": "failed",
                    "error": {"code": "synthetic_failure", "message": "Synthetic test failure"}}
        responses = [self.legacy_capabilities, self.accepted_a, self.done_a,
                     self.legacy_capabilities, self.accepted_b, failed_b]
        original_replace = os.replace
        snapshots = []

        def checked_replace(source, destination):
            source, destination = Path(source), Path(destination)
            self.assertEqual(source.parent, destination.parent)
            self.assertEqual(source.stat().st_mode & 0o777, 0o600)
            snapshots.append(json.loads(source.read_text(encoding="utf-8")))
            if destination.exists():
                json.loads(destination.read_text(encoding="utf-8"))
            original_replace(source, destination)

        with patch.object(sys, "argv", self.argv), patch.dict(os.environ, {"CASHWATCH_API_KEY": self.key}), \
                patch.object(CHECKER, "load_dotenv"), patch.object(CHECKER, "request_json", side_effect=responses), \
                patch.object(CHECKER.os, "replace", side_effect=checked_replace), redirect_stdout(io.StringIO()):
            self.assertEqual(CHECKER.main(), 1)
        payload = json.loads(self.output.read_text(encoding="utf-8"))
        self.assertEqual([snapshot["metadata"]["recorded_video_count"] for snapshot in snapshots], [0, 1, 2])
        self.assertEqual([snapshot["metadata"]["run_finished"] for snapshot in snapshots], [False, False, True])
        self.assertEqual([record["status"] for record in payload["predictions"]], ["done", "failed"])
        self.assertTrue(payload["metadata"]["complete"])
        self.assertTrue(payload["metadata"]["run_finished"])
        self.assertEqual(payload["metadata"]["failed_video_count"], 1)
        self.assertEqual(payload["metadata"]["manifest_sha256"], hashlib.sha256(self.manifest_path.read_bytes()).hexdigest())
        self.assertEqual(self.output.stat().st_mode & 0o777, 0o600)
        self.assertEqual(list(self.output.parent.glob("*.tmp")), [])
        report = self.evaluate_report(payload)
        self.assertEqual(report["video_level"]["failed_videos"], 1)
        self.assertEqual(report["failures"]["by_reason"], {"reported_failed": 1})


class RuntimeBenchmarkGateTests(unittest.TestCase):
    """Only synthetic reports/responses: these are not GPU or model results."""

    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="cashwatch-runtime-gate-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.model = self.root / "private-model-directory"
        self.model.mkdir()
        self.model_manifest = self.model / "cashwatch_model_manifest.json"
        self.provenance = {
            "model_id": "Synthetic/ContractFixture", "resolved_revision": "1" * 40,
            "downloaded_at_utc": "private provenance fixture only",
            "files": {name: {"sha256": str(index) * 64, "bytes": 1} for index, name in enumerate(
                ("config.json", "tokenizer_config.json", "model.safetensors"), start=1)},
        }
        self.model_manifest.write_text(json.dumps(self.provenance), encoding="utf-8")
        self.rules = self.root / "private-rules.json"
        self.rules.write_text('{"schema_version":1,"rules":[]}', encoding="utf-8")
        self.settings = Settings(api_key="", model_dir=self.model, rules_path=self.rules, device="cuda",
                                 window_seconds=6, overlap_seconds=2, max_frames_per_window=8, max_image_edge=384)
        self.contract = build_runtime_contract(self.settings)
        self.assertTrue(self.contract["complete"])
        self.video_a = self.root / "synthetic-a.mov"
        self.video_b = self.root / "synthetic-b.mov"
        self.video_a.write_bytes(b"synthetic decoder-free a")
        self.video_b.write_bytes(b"synthetic decoder-free b")
        self.output = self.root / "predictions.json"
        self.benchmark_path = self.root / "benchmark.private.json"
        self.benchmark = {
            "schema_version": 1, "kind": "cashwatch_local_benchmark", "mode": "jury",
            "contract_stable": True, "runtime_contract": self.contract,
            "requested_device": "cuda", "requested_runs": 1,
            "source_sha256": "d" * 64, "source_path": str(self.video_a),
            "runs": [{
                "status": "complete", "jury_response_status": "done", "model_loaded": True,
                "generation_attempted": True, "analyzer_complete": True,
                "cold_model_load_included": True, "final_metrics_received": True,
                "latency_budget_passed": True, "supervisor_hard_stop": False,
                "child_exit_code": 0, "selected_device": "cuda:0", "wall_seconds": 12.0,
                "latency_budget_seconds": 275.0,
            }],
            "summary": {"completed_runs": 1, "all_requested_runs_completed_within_budget": True},
        }
        self.write_benchmark(self.benchmark)
        self.key = "synthetic-runtime-gate-key-only"
        self.argv = ["jury_check.py", "--base-url", "http://127.0.0.1:8000", "--video", str(self.video_a),
                     "--video", str(self.video_b), "--expect-benchmark", str(self.benchmark_path),
                     "--out", str(self.output)]
        self.capabilities = {"runtime_contract": self.contract}
        self.accepted_a = {"job_id": "a" * 32, "status": "processing"}
        self.accepted_b = {"job_id": "b" * 32, "status": "processing"}
        self.done_a = {"job_id": "a" * 32, "status": "done", "verdict": "no_violation", "violations": []}
        self.done_b = {**self.done_a, "job_id": "b" * 32}

    def write_benchmark(self, payload):
        self.benchmark_path.write_text(json.dumps(payload), encoding="utf-8")

    def invoke(self, responses):
        with patch.object(sys, "argv", self.argv), patch.dict(os.environ, {"CASHWATCH_API_KEY": self.key}), \
                patch.object(CHECKER, "load_dotenv"), patch.object(CHECKER, "request_json", side_effect=responses) as request, \
                redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            result = CHECKER.main()
        return result, request

    def assert_preflight_rejects(self, responses, *, expected_requests):
        previous = b'{"previous_report":"must survive"}'
        self.output.write_bytes(previous)
        diagnostics = io.StringIO()
        with patch.object(sys, "argv", self.argv), patch.dict(os.environ, {"CASHWATCH_API_KEY": self.key}), \
                patch.object(CHECKER, "load_dotenv"), patch.object(CHECKER, "request_json", side_effect=responses) as request, \
                patch.object(Path, "read_bytes", side_effect=AssertionError("Video body read before preflight")), \
                redirect_stdout(io.StringIO()), redirect_stderr(diagnostics), self.assertRaises(SystemExit) as exit_info:
            CHECKER.main()
        self.assertEqual(exit_info.exception.code, 2)
        self.assertEqual(request.call_count, expected_requests)
        self.assertTrue(all(call.args[0].endswith("/api/v1/capabilities") for call in request.call_args_list))
        self.assertEqual(self.output.read_bytes(), previous)
        self.assertNotIn(self.key, diagnostics.getvalue())
        self.assertNotIn(str(self.root), diagnostics.getvalue())
        return diagnostics.getvalue()

    def test_device_sampling_revision_and_rules_mismatches_stop_before_body_or_post(self):
        variants = [
            ("settings.device", build_runtime_contract(replace(self.settings, device="cpu"))),
            ("settings.max_image_edge", build_runtime_contract(replace(self.settings, max_image_edge=512))),
        ]
        self.model_manifest.write_text(json.dumps({**self.provenance, "resolved_revision": "2" * 40}), encoding="utf-8")
        variants.append(("model.revision", build_runtime_contract(self.settings)))
        self.model_manifest.write_text(json.dumps(self.provenance), encoding="utf-8")
        self.rules.write_text('{"schema_version":1, "rules":[]}', encoding="utf-8")
        variants.append(("rules_sha256", build_runtime_contract(self.settings)))
        for field, observed in variants:
            with self.subTest(field=field):
                text = self.assert_preflight_rejects([{"runtime_contract": observed}], expected_requests=1)
                self.assertIn("runtime_contract_mismatch", text)
                self.assertIn(field, text)

    def test_forged_or_incomplete_server_contract_and_legacy_server_are_rejected(self):
        forged = copy.deepcopy(self.contract)
        forged["settings"]["max_image_edge"] = 512  # Intentionally keep the stale fingerprint.
        incomplete = build_runtime_contract(replace(self.settings, model_dir=self.root / "missing"))
        for capabilities in ({"runtime_contract": forged}, {"runtime_contract": incomplete}, {}):
            with self.subTest(keys=tuple(capabilities)):
                self.assert_preflight_rejects([capabilities], expected_requests=1)

    def test_analyzer_timeout_partial_and_changed_benchmarks_cannot_authorize_upload(self):
        variants = []
        for changes in ({"mode": "analyzer"}, {"contract_stable": False}, {"requested_runs": 2},
                        {"requested_device": "auto"}):
            variants.append({**copy.deepcopy(self.benchmark), **changes})
        for changes in ({"latency_budget_passed": False}, {"status": "timeout"}, {"analyzer_complete": False},
                        {"wall_seconds": 276}, {"selected_device": "cpu"}, {"latency_budget_seconds": 270}):
            candidate = copy.deepcopy(self.benchmark)
            candidate["runs"][0].update(changes)
            variants.append(candidate)
        for index, candidate in enumerate(variants):
            with self.subTest(case=index):
                self.write_benchmark(candidate)
                self.assert_preflight_rejects([], expected_requests=0)

    def test_matching_jury_benchmark_records_contract_without_private_benchmark_inputs(self):
        responses = [self.capabilities, self.accepted_a, self.done_a,
                     self.capabilities, self.accepted_b, self.done_b]
        result, request = self.invoke(responses)
        self.assertEqual(result, 0)
        self.assertEqual(request.call_count, 6)
        self.assertTrue(request.call_args_list[0].args[0].endswith("/capabilities"))
        self.assertTrue(request.call_args_list[3].args[0].endswith("/capabilities"))
        text = self.output.read_text(encoding="utf-8")
        payload = json.loads(text)
        self.assertEqual(payload["metadata"]["runtime_contract"], self.contract)
        self.assertEqual(payload["metadata"]["expected_runtime_fingerprint"], self.contract["fingerprint"])
        self.assertEqual(payload["metadata"]["runtime_check"]["status"], "benchmark_matched")
        self.assertEqual(payload["metadata"]["runtime_check"]["capabilities_checks_completed"], 2)
        self.assertFalse(payload["metadata"]["runtime_check"]["weights_hashes_verified"])
        self.assertTrue(payload["metadata"]["complete"])
        for private in (self.key, str(self.root), "d" * 64, "http://127.0.0.1:8000", "private provenance fixture only"):
            self.assertNotIn(private, text)

    def test_configuration_change_mid_batch_preserves_partial_report_and_skips_next_upload(self):
        changed = build_runtime_contract(replace(self.settings, max_image_edge=512))
        result, request = self.invoke([self.capabilities, self.accepted_a, self.done_a, {"runtime_contract": changed}])
        self.assertEqual(result, 1)
        self.assertEqual(request.call_count, 4)
        posts = [call for call in request.call_args_list if "body" in call.kwargs]
        self.assertEqual(len(posts), 1)
        payload = json.loads(self.output.read_text(encoding="utf-8"))
        self.assertEqual(len(payload["predictions"]), 1)
        self.assertFalse(payload["metadata"]["complete"])
        self.assertFalse(payload["metadata"]["run_finished"])
        self.assertEqual(payload["metadata"]["runtime_check"]["status"], "stopped_before_upload")
        self.assertEqual(payload["metadata"]["runtime_check"]["mismatched_fields"], ["settings.max_image_edge"])

    def test_output_must_not_overwrite_reference_benchmark(self):
        self.argv[-1] = str(self.benchmark_path)
        before = self.benchmark_path.read_bytes()
        with patch.object(sys, "argv", self.argv), patch.object(CHECKER, "request_json") as request, \
                redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            CHECKER.main()
        self.assertEqual(request.call_count, 0)
        self.assertEqual(self.benchmark_path.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
