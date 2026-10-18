"""Synthetic tests for annotation integrity and evaluation; no bank data is used."""
from __future__ import annotations

import base64
import copy
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
import prepare_dataset as prepare  # noqa: E402
import evaluate as evaluator  # noqa: E402
sys.path.pop(0)

# Deliberately only a synthetic marker envelope, not decoded image pixels.
# The evaluator promises envelope validation and must document this limitation.
JPEG_ENVELOPE = base64.b64encode(b"\xff\xd8\xff\xe0\x00\x10SYNTHETIC-ENVELOPE\xff\xd9").decode("ascii")
PROBE = {"duration_sec": 100.0, "fps": 25.0, "width": 640, "height": 480}
REGISTRY = {"r1": "Rule one", "r2": "Rule two"}


def video(video_id: str, label: int | None = 1, *, events=None, split="holdout", group_id=None):
    return {"video_id": video_id, "path": f"{video_id}.mp4", "sha256": hashlib.sha256(video_id.encode()).hexdigest(),
            **PROBE, "label": label, "events": [] if events is None else events,
            "group_id": group_id or f"group-{video_id}", "camera_id": "synthetic-camera", "split": split}


def event(timestamp: float, rule_id: str | None = "r1"):
    return {"timecode_sec": timestamp, "rule_id": rule_id}


def prediction(video_id: str, timestamps=(), *, rule_id="r1", elapsed=None):
    response = {"video_id": video_id, "status": "done", "verdict": "violation" if timestamps else "no_violation",
                "violations": [{"type": REGISTRY.get(rule_id, "Unmapped title"), "timecode_sec": timestamp,
                                "rule_id": rule_id,
                                "evidence_frames": [{"timestamp_sec": timestamp, "image_base64": JPEG_ENVELOPE}]}
                               for timestamp in timestamps]}
    if elapsed is not None:
        response["elapsed_sec"] = elapsed
    return response


def report(videos, predictions, **kwargs):
    return evaluator.evaluate({"schema_version": 1, "videos": videos}, predictions, split="holdout", **kwargs)


class PreparationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="cashwatch-dataset-synthetic-")
        self.root = Path(self.temporary.name)
        (self.root / "a.MOV").write_bytes(b"synthetic-video-a")
        (self.root / "sub").mkdir()
        (self.root / "sub" / "b.mp4").write_bytes(b"synthetic-video-b")
        (self.root / "ignore.avi").write_bytes(b"ignored")

    def tearDown(self):
        self.temporary.cleanup()

    def build(self, csv_text=None):
        labels = None
        if csv_text is not None:
            labels = self.root / "labels.csv"
            labels.write_text(csv_text, encoding="utf-8")
        with patch.object(prepare, "probe_video", return_value=PROBE):
            return prepare.build_manifest(self.root, labels)

    def test_unlabeled_manifest_does_not_infer_labels_splits_or_groups(self):
        manifest = self.build()
        self.assertEqual([entry["video_id"] for entry in manifest["videos"]], ["a.MOV", "sub/b.mp4"])
        self.assertEqual(manifest["metadata"]["input_dir"], str(self.root.resolve()))
        self.assertFalse(manifest["metadata"]["frames_extracted"])
        for entry in manifest["videos"]:
            self.assertIsNone(entry["label"])
            self.assertEqual(entry["events"], [])
            self.assertEqual(entry["split"], "unassigned")
            self.assertEqual(entry["group_id"], "unassigned")
            self.assertEqual(entry["path"], entry["video_id"])
        self.assertEqual(manifest["videos"][0]["sha256"], hashlib.sha256(b"synthetic-video-a").hexdigest())

    def test_multiple_events_merge_and_blank_fields_do_not_overwrite_metadata(self):
        manifest = self.build("video,label,timecode_sec,rule_id,group_id,camera_id,split\n"
                              "a.MOV,1,20,r2,scene-a,camera-a,holdout\n"
                              "a.MOV,1,10,r1,,,\n"
                              "sub/b.mp4,0,,,scene-b,camera-b,validation\n")
        first, second = manifest["videos"]
        self.assertEqual(first["events"], [event(10.0), event(20.0, "r2")])
        self.assertEqual(first["split"], "holdout")
        self.assertEqual(second["label"], 0)
        self.assertEqual(second["events"], [])

    def test_missing_csv_row_stays_unlabeled(self):
        entries = self.build("video,label\na.MOV,1\n")["videos"]
        self.assertEqual(entries[0]["label"], 1)
        self.assertEqual(entries[0]["events"], [])
        self.assertIsNone(entries[1]["label"])
        self.assertEqual(entries[0]["split"], "unassigned")

    def test_unknown_group_and_camera_placeholders_stay_explicitly_unassigned(self):
        entry = self.build("video,label,group_id,camera_id,split\na.MOV,1,unknown,n/a,holdout\n")["videos"][0]
        self.assertEqual(entry["group_id"], "unassigned")
        self.assertEqual(entry["camera_id"], "unassigned")
        with self.assertRaisesRegex(prepare.DatasetError, "unknown group_id"):
            report([entry], [])

    def test_invalid_annotations_are_rejected(self):
        rows = [
            ("video,label,timecode_sec\na.MOV,0,0\n", "label 0"),
            ("video,label,rule_id\na.MOV,1,r1\n", "requires timecode_sec"),
            ("video,label\nmissing.mp4,1\n", "unknown video"),
            ("video,label\nb.mp4,1\n", "unknown video"),
            ("video,label\n../a.MOV,1\n", "outside input-dir"),
            ("video,label\na.MOV,1.0\n", "exactly 0 or 1"),
            ("video,label,split\na.MOV,1,test\n", "unknown split"),
            ("video,label,label\na.MOV,1,1\n", "Duplicate"),
            ("video,label,timestamp\na.MOV,1,3\n", "Unknown labels CSV"),
        ]
        for csv_text, reason in rows:
            with self.subTest(reason=reason), self.assertRaisesRegex(prepare.DatasetError, reason):
                self.build(csv_text)

    def test_event_timestamps_must_be_finite_nonnegative_and_in_bounds(self):
        for value in ("-1", "101", "nan", "inf", "not-a-number"):
            with self.subTest(value=value), self.assertRaises(prepare.DatasetError):
                self.build(f"video,label,timecode_sec\na.MOV,1,{value}\n")
        result = self.build("video,label,timecode_sec\na.MOV,1,0\na.MOV,1,100\n")
        self.assertEqual([entry["timecode_sec"] for entry in result["videos"][0]["events"]], [0, 100])

    def test_conflicting_repeated_rows_and_duplicate_events_are_rejected(self):
        cases = [
            "video,label\na.MOV,0\na.MOV,1\n",
            "video,label,group_id\na.MOV,1,a\na.MOV,1,b\n",
            "video,label,camera_id\na.MOV,1,a\na.MOV,1,b\n",
            "video,label,split\na.MOV,1,train\na.MOV,1,holdout\n",
            "video,label,timecode_sec,rule_id\na.MOV,1,10,r1\na.MOV,1,10,r1\n",
        ]
        for csv_text in cases:
            with self.subTest(csv_text=csv_text), self.assertRaises(prepare.DatasetError):
                self.build(csv_text)

    def test_duplicate_hash_across_splits_is_rejected(self):
        (self.root / "sub" / "b.mp4").write_bytes((self.root / "a.MOV").read_bytes())
        with self.assertRaisesRegex(prepare.DatasetError, "leakage by sha256"):
            self.build("video,label,group_id,split\na.MOV,1,a,train\nsub/b.mp4,1,b,holdout\n")

    def test_duplicate_hash_within_split_is_reported_not_hidden(self):
        (self.root / "sub" / "b.mp4").write_bytes((self.root / "a.MOV").read_bytes())
        result = self.build("video,label,group_id,split\na.MOV,1,a,holdout\nsub/b.mp4,1,a,holdout\n")
        self.assertEqual(len(result["metadata"]["split_integrity"]["duplicate_hashes"]), 1)

    def test_same_scene_across_splits_is_rejected_even_when_bytes_differ(self):
        with self.assertRaisesRegex(prepare.DatasetError, "leakage by group_id"):
            self.build("video,label,group_id,split\na.MOV,1,same-scene,train\nsub/b.mp4,1,same-scene,holdout\n")

    def test_ffprobe_rational_rate_and_timestamp_duration_fallback(self):
        payload = {"streams": [{"width": 640, "height": 480, "avg_frame_rate": "30000/1001",
                                 "r_frame_rate": "30/1", "duration_ts": 90000, "time_base": "1/90000"}]}
        process = subprocess.CompletedProcess([], 0, json.dumps(payload), "")
        with patch.object(prepare.subprocess, "run", return_value=process) as run:
            result = prepare.probe_video(self.root / "a.MOV")
        arguments = run.call_args.args[0]
        self.assertEqual(arguments[arguments.index("-protocol_whitelist") + 1], "file,pipe")
        self.assertAlmostEqual(result["fps"], 30000 / 1001)
        self.assertEqual(result["duration_sec"], 1)

    @unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "FFmpeg/ffprobe unavailable")
    def test_real_ffprobe_on_generated_synthetic_video(self):
        source = self.root / "tiny-synthetic.mp4"
        subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-f", "lavfi", "-i",
                        "color=c=black:s=64x48:r=10:d=1", "-an", "-c:v", "mpeg4", str(source)],
                       check=True, capture_output=True, timeout=20)
        result = prepare.probe_video(source)
        self.assertEqual((result["width"], result["height"], result["fps"]), (64, 48, 10))
        self.assertAlmostEqual(result["duration_sec"], 1, places=3)


class EvaluationTests(unittest.TestCase):
    def test_confusion_matrix_failures_and_effective_accuracy(self):
        videos = [video("tp", events=[event(10)]), video("tn", 0), video("fp", 0),
                  video("fn", events=[event(20)]), video("failed", events=[event(30)]), video("missing", 0)]
        predictions = [prediction("tp", [10]), prediction("tn"), prediction("fp", [5]), prediction("fn"),
                       {"video_id": "failed", "status": "failed"}]
        result = report(videos, predictions)
        metrics = result["video_level"]
        self.assertEqual(metrics["confusion_matrix"], {"tp": 1, "tn": 1, "fp": 1, "fn": 1})
        self.assertEqual(metrics["failed_videos"], 2)
        self.assertEqual(metrics["precision"], 0.5)
        self.assertEqual(metrics["recall"], 0.5)
        self.assertEqual(metrics["f1"], 0.5)
        self.assertEqual(metrics["completed_video_accuracy"], 0.5)
        self.assertAlmostEqual(metrics["effective_video_accuracy"], 2 / 6)
        self.assertAlmostEqual(metrics["end_to_end_positive_detection_recall"], 1 / 3)
        self.assertEqual(result["failures"]["by_reason"], {"missing_prediction": 1, "reported_failed": 1})
        self.assertEqual(result["event_level"]["tolerances_sec"]["1"]["fn"], 2)

    def test_missing_predictions_do_not_become_true_negatives(self):
        result = report([video("a", 0)], [])
        self.assertEqual(result["video_level"]["effective_video_accuracy"], 0)
        self.assertIsNone(result["video_level"]["completed_video_accuracy"])
        for metric in ("precision", "recall", "f1"):
            self.assertIsNone(result["video_level"][metric])
        self.assertIsNone(result["latency"]["p95_sec"])

    def test_no_positive_ground_truth_does_not_invent_prf_denominators(self):
        result = report([video("a", 0)], [prediction("a")])
        self.assertEqual(result["video_level"]["effective_video_accuracy"], 1)
        self.assertIsNone(result["video_level"]["precision"])
        self.assertIsNone(result["video_level"]["recall"])
        self.assertIsNone(result["video_level"]["f1"])
        self.assertFalse(result["event_level"]["type_aware"]["available"])

    def test_invalid_or_inconsistent_completed_response_is_failed(self):
        wrong = prediction("a", [10])
        wrong["verdict"] = "no_violation"
        result = report([video("a", events=[event(10)])], [wrong])
        self.assertEqual(result["failures"]["by_reason"], {"invalid_response": 1})
        self.assertEqual(result["video_level"]["effective_video_accuracy"], 0)

    def test_unfinished_and_unknown_status_are_failures(self):
        result = report([video("a", 0), video("b", 0)],
                        [{"video_id": "a", "status": "processing"}, {"video_id": "b", "status": "completed"}])
        self.assertEqual(result["failures"]["by_reason"], {"invalid_status": 1, "not_completed": 1})

    def test_extra_duplicate_unknown_or_other_split_predictions_are_errors(self):
        videos = [video("a", 0), video("b", 0, split="validation")]
        for predictions in ([prediction("a"), prediction("a")], [prediction("unknown")], [prediction("b")]):
            with self.subTest(predictions=predictions), self.assertRaises(prepare.DatasetError):
                report(videos, predictions)

    def test_manifest_leakage_by_hash_group_path_and_id(self):
        for field in ("sha256", "group_id", "path", "video_id"):
            videos = [video("a", 0, split="train"), video("b", 0)]
            videos[1][field] = videos[0][field]
            with self.subTest(field=field), self.assertRaises(prepare.DatasetError):
                report(videos, [])

    def test_unknown_training_group_blocks_holdout_evaluation(self):
        videos = [video("a", 0), video("b", 0, split="train", group_id="unassigned")]
        with self.assertRaisesRegex(prepare.DatasetError, "unknown group_id"):
            report(videos, [prediction("a")])

    def test_unassigned_unlabeled_video_does_not_get_evaluated(self):
        videos = [video("a", 0), video("b", None, split="unassigned", group_id="unassigned")]
        result = report(videos, [prediction("a")])
        self.assertEqual(result["video_level"]["total_videos"], 1)

    def test_train_evaluation_and_unlabeled_holdout_are_rejected(self):
        with self.assertRaises(prepare.DatasetError):
            evaluator.evaluate({"schema_version": 1, "videos": [video("a", 0, split="train")]}, [], split="train")
        with self.assertRaises(prepare.DatasetError):
            report([video("a", None)], [])

    def test_one_prediction_cannot_match_two_truth_events(self):
        result = report([video("a", events=[event(10), event(10.4)])], [prediction("a", [10.2])])
        metric = result["event_level"]["tolerances_sec"]["1"]
        self.assertEqual((metric["tp"], metric["fp"], metric["fn"]), (1, 0, 1))

    def test_augmenting_path_avoids_greedy_matching_loss(self):
        matched = evaluator.match_events([event(10), event(11)], [event(10.5), event(9.1)], 1)
        self.assertEqual(set(matched), {(0, 1), (1, 0)})

    def test_tolerances_are_diagnostic_and_distinct(self):
        result = report([video("a", events=[event(10)])], [prediction("a", [13])])
        self.assertFalse(result["event_level"]["official_metric"])
        metrics = result["event_level"]["tolerances_sec"]
        self.assertEqual([metrics[key]["tp"] for key in ("1", "2", "5")], [0, 0, 1])

    def test_video_only_positives_are_excluded_from_event_metrics(self):
        result = report([video("a", events=[])], [prediction("a", [10])])
        self.assertEqual(result["video_level"]["confusion_matrix"]["tp"], 1)
        self.assertEqual(result["event_level"]["excluded_video_only_positive_ids"], ["a"])
        self.assertEqual(result["event_level"]["predicted_events"], 0)

    def test_type_titles_are_not_guessed_and_exact_registry_mapping_is_supported(self):
        answer = prediction("a", [10])
        del answer["violations"][0]["rule_id"]
        videos = [video("a", events=[event(10)])]
        no_mapping = report(videos, [answer])
        typed = no_mapping["event_level"]["type_aware"]
        self.assertEqual(typed["unmapped_predicted_events"], 1)
        self.assertEqual(typed["tolerances_sec"]["1"]["tp"], 0)
        self.assertEqual(typed["tolerances_sec"]["1"]["fp"], 1)
        exact = report(videos, [answer], registry=REGISTRY)
        self.assertEqual(exact["event_level"]["type_aware"]["tolerances_sec"]["1"]["tp"], 1)
        answer["violations"][0]["type"] = "Rule 1"
        different = report(videos, [answer], registry=REGISTRY)
        self.assertEqual(different["failures"]["count"], 1)

    def test_mixed_typed_and_unknown_truth_video_is_excluded_from_typed_precision(self):
        videos = [video("a", events=[event(10), event(20, None)]), video("b", events=[event(10)])]
        result = report(videos, [prediction("a", [10, 20]), prediction("b", [10])])
        typed = result["event_level"]["type_aware"]
        self.assertEqual(typed["excluded_mixed_or_untyped_truth_ids"], ["a"])
        self.assertEqual(typed["tolerances_sec"]["1"]["tp"], 1)

    def test_explicit_wrong_type_id_is_not_matched(self):
        result = report([video("a", events=[event(10)])], [prediction("a", [10], rule_id="r2")], registry=REGISTRY)
        self.assertEqual(result["event_level"]["tolerances_sec"]["1"]["tp"], 1)
        typed = result["event_level"]["type_aware"]["tolerances_sec"]["1"]
        self.assertEqual((typed["tp"], typed["fp"], typed["fn"]), (0, 1, 1))

    def test_latency_boundary_missing_values_and_percentiles(self):
        videos = [video(name, 0) for name in ("a", "b", "c", "d")]
        result = report(videos, [prediction("a", elapsed=0), prediction("b", elapsed=300),
                                 prediction("c", elapsed=300.00001), prediction("d")])
        latency = result["latency"]
        self.assertEqual(latency["reported_count"], 3)
        self.assertEqual(latency["missing_or_invalid_count"], 1)
        self.assertEqual(latency["within_300_sec_count"], 2)
        self.assertEqual(latency["over_300_sec_count"], 1)
        self.assertEqual(latency["p50_sec"], 300)
        self.assertGreater(latency["p95_sec"], 300)
        self.assertEqual(latency["within_limit_rate_all_videos"], 0.5)


class EvidenceIntegrityTests(unittest.TestCase):
    def run_answer(self, answer, **kwargs):
        return report([video("a", events=[event(10)])], [answer], **kwargs)

    def test_invalid_base64_and_non_jpeg_are_failures(self):
        for invalid in ("not base64!", base64.b64encode(b"PNG-but-not-JPEG").decode(), JPEG_ENVELOPE[:-2], "😀"):
            answer = prediction("a", [10])
            answer["violations"][0]["evidence_frames"][0]["image_base64"] = invalid
            with self.subTest(invalid=invalid):
                result = self.run_answer(answer)
                self.assertEqual(result["failures"]["by_reason"], {"invalid_evidence": 1})
                self.assertEqual(result["video_level"]["effective_video_accuracy"], 0)

    def test_anchor_tolerance_and_exact_override(self):
        answer = prediction("a", [10])
        answer["violations"][0]["evidence_frames"][0]["timestamp_sec"] = 10.03
        self.assertEqual(self.run_answer(answer)["failures"]["count"], 0)
        self.assertEqual(self.run_answer(answer, evidence_tolerance_sec=0)["failures"]["count"], 1)
        answer["violations"][0]["evidence_frames"][0]["timestamp_sec"] = 10.05
        self.assertEqual(self.run_answer(answer)["failures"]["count"], 1)
        self.assertEqual(self.run_answer(answer, evidence_tolerance_sec=0.1)["failures"]["count"], 0)

    def test_middle_evidence_frame_can_anchor_before_and_after_sequence(self):
        answer = prediction("a", [10])
        answer["violations"][0]["evidence_frames"] = [
            {"timestamp_sec": timestamp, "image_base64": JPEG_ENVELOPE} for timestamp in (9, 10, 11)]
        result = self.run_answer(answer)
        self.assertEqual(result["evidence_validation"]["validated_frame_count"], 3)
        self.assertNotIn("image_base64", json.dumps(result))
        self.assertIn("not decoded", result["evidence_validation"]["limitations"])

    def test_bad_evidence_times_and_counts_are_failures(self):
        for times in ([101], [-1], [10, 10], [11, 10], [], [9, 10, 11, 12]):
            answer = prediction("a", [10])
            answer["violations"][0]["evidence_frames"] = [
                {"timestamp_sec": timestamp, "image_base64": JPEG_ENVELOPE} for timestamp in times]
            with self.subTest(times=times):
                self.assertEqual(self.run_answer(answer)["failures"]["count"], 1)


class CLITests(unittest.TestCase):
    def test_duplicate_json_keys_and_nonfinite_constants_are_rejected(self):
        with tempfile.TemporaryDirectory(prefix="cashwatch-json-synthetic-") as temporary:
            path = Path(temporary) / "input.json"
            for text in ('{"videos":[],"videos":[]}', '{"x":NaN}'):
                path.write_text(text)
                with self.subTest(text=text), self.assertRaises(prepare.DatasetError):
                    prepare.load_json(path)

    def test_evaluate_cli_writes_no_images_and_does_not_overwrite_inputs(self):
        with tempfile.TemporaryDirectory(prefix="cashwatch-eval-synthetic-") as temporary:
            root = Path(temporary)
            manifest, predictions, output = root / "manifest.json", root / "predictions.json", root / "report.json"
            manifest.write_text(json.dumps({"schema_version": 1, "videos": [video("a", events=[event(10)])]}))
            predictions.write_text(json.dumps({"predictions": [prediction("a", [10], elapsed=2.5)]}))
            command = [sys.executable, str(SCRIPTS / "evaluate.py"), "--manifest", str(manifest),
                       "--predictions", str(predictions), "--split", "holdout", "--out", str(output)]
            process = subprocess.run(command, capture_output=True, text=True, timeout=20)
            self.assertEqual(process.returncode, 0, process.stderr)
            content = json.loads(output.read_text())
            self.assertEqual(content["video_level"]["effective_video_accuracy"], 1)
            self.assertNotIn("image_base64", output.read_text())
            self.assertNotIn(JPEG_ENVELOPE, process.stdout + process.stderr + output.read_text())
            original = manifest.read_bytes()
            blocked = subprocess.run(command[:-1] + [str(manifest)], capture_output=True, text=True, timeout=20)
            self.assertEqual(blocked.returncode, 2)
            self.assertEqual(manifest.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
