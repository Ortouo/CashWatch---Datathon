"""Meaningful evidence integrity checks; no pretrained inference is simulated as real."""
from __future__ import annotations

import json
import hashlib
import importlib.util
import shutil
import subprocess
import tempfile
import time
import unittest
from pathlib import Path

from cashwatch.vision.contracts import EvidenceFrame, InvalidObservationError, ModelUnavailableError, RuleSpec
from cashwatch.vision.model import inspect_model_directory
from cashwatch.vision.parsing import parse_observations
from cashwatch.vision.video import plan_windows, sample_video


class ObservationIntegrityTests(unittest.TestCase):
    def setUp(self):
        self.frames = (
            EvidenceFrame("f000001", 2.04, Path("/synthetic/frame1.jpg")),
            EvidenceFrame("f000002", 2.58, Path("/synthetic/frame2.jpg")),
            EvidenceFrame("f000003", 3.10, Path("/synthetic/frame3.jpg")),
        )
        self.rules = (RuleSpec("approved_example", "Synthetic test rule", "Synthetic test only",
                               required_conditions=("object visible", "destination visible")),)
        self.event = {
            "rule_id": "approved_example", "start_frame_id": "f000001", "end_frame_id": "f000003",
            "description": "Synthetic test observation; no bank footage.",
            "evidence_frame_ids": ["f000001", "f000003"], "assessment": "supported",
            "condition_evidence": {"object visible": ["f000001"], "destination visible": ["f000003"]},
            "uncertainty": None,
        }

    def parse(self, event):
        return parse_observations(json.dumps({"observations": [event]}), self.frames, self.rules, window_id="w0")

    def test_timecodes_come_from_decoded_frames(self):
        result = self.parse(self.event)[0]
        self.assertEqual(result.start_sec, 2.04)
        self.assertEqual(result.end_sec, 3.10)

    def test_invented_frame_is_rejected(self):
        self.event["evidence_frame_ids"] = ["f000001", "invented"]
        with self.assertRaises(InvalidObservationError):
            self.parse(self.event)

    def test_missing_required_condition_is_rejected(self):
        del self.event["condition_evidence"]["destination visible"]
        with self.assertRaises(InvalidObservationError):
            self.parse(self.event)

    def test_condition_must_be_visible_in_delivered_evidence(self):
        self.event["condition_evidence"]["object visible"] = ["f000002"]
        with self.assertRaises(InvalidObservationError):
            self.parse(self.event)

    def test_model_cannot_invent_a_rule(self):
        self.event["rule_id"] = "rule_not_provided"
        with self.assertRaises(InvalidObservationError):
            self.parse(self.event)

    def test_malformed_output_never_becomes_empty_success(self):
        with self.assertRaises(InvalidObservationError):
            parse_observations("not JSON", self.frames, self.rules, window_id="w0")

    def test_duplicate_json_keys_are_rejected(self):
        with self.assertRaises(InvalidObservationError):
            parse_observations('{"observations": [], "observations": []}', self.frames, self.rules, window_id="w0")

    def test_inverted_time_interval_is_rejected(self):
        self.event["start_frame_id"], self.event["end_frame_id"] = "f000003", "f000001"
        with self.assertRaises(InvalidObservationError):
            self.parse(self.event)

    def test_no_model_is_an_error(self):
        with tempfile.TemporaryDirectory() as path:
            with self.assertRaises(ModelUnavailableError):
                inspect_model_directory(Path(path))


@unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "FFmpeg is required")
class LocalDecoderTests(unittest.TestCase):
    def test_synthetic_movie_keeps_actual_pts_and_covers_all_windows(self):
        with tempfile.TemporaryDirectory(prefix="cashwatch-synthetic-") as temporary:
            folder = Path(temporary)
            source = folder / "synthetic.mov"
            subprocess.run([
                "ffmpeg", "-nostdin", "-v", "error", "-y", "-f", "lavfi", "-i",
                "testsrc2=size=160x120:rate=12", "-frames:v", "26", "-c:v", "mpeg4", str(source),
            ], check=True, capture_output=True, timeout=20)
            sampled = sample_video(source, folder / "frames", sample_fps=2.0,
                                   deadline_monotonic=time.monotonic() + 20)
            times = [frame.timestamp_sec for frame in sampled.frames]
            self.assertEqual(times, [0.0, 0.5, 1.0, 1.5, 2.0])
            self.assertTrue(all(frame.image_path.is_file() for frame in sampled.frames))
            windows = plan_windows(sampled, window_seconds=1.0, overlap_seconds=0.25, max_frames_per_window=4)
            self.assertEqual(windows[0][0], 0.0)
            self.assertAlmostEqual(windows[-1][1], sampled.duration_sec)
            represented = {frame.frame_id for _, _, frames in windows for frame in frames}
            self.assertEqual(represented, {frame.frame_id for frame in sampled.frames})


class TemporalTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        script = Path(__file__).resolve().parents[1] / "scripts" / "train_temporal_head.py"
        specification = importlib.util.spec_from_file_location("cashwatch_training_under_test", script)
        cls.training = importlib.util.module_from_spec(specification)
        specification.loader.exec_module(cls.training)

    @staticmethod
    def video(identifier, label, split, group=None):
        return {"video_id": identifier, "path": "/synthetic/unused.mov",
                "sha256": hashlib.sha256(identifier.encode()).hexdigest(),
                "duration_sec": 10.0, "label": label,
                "events": [{"timecode_sec": 5.0, "rule_id": None}] if label == 1 else [],
                "group_id": group or identifier, "camera_id": "synthetic", "split": split}

    def test_unannotated_positive_windows_are_never_negative(self):
        import numpy as np
        videos = [self.video("positive", 1, "train"), self.video("normal", 0, "train")]
        labels, _ = self.training.build_training_rows(
            np.array(["positive", "positive", "normal"]), np.array([0., 5., 5.]), videos, 2.0,
        )
        np.testing.assert_array_equal(labels, [-1, 1, 0])

    def test_group_cannot_cross_train_validation(self):
        videos = [self.video("one", 0, "train", group="same_recording"),
                  self.video("two", 1, "validation", group="same_recording")]
        with self.assertRaises(ValueError):
            self.training.assert_disjoint_splits(videos)

    def test_duplicate_content_cannot_cross_split(self):
        videos = [self.video("one", 0, "train"), self.video("two", 1, "validation")]
        videos[1]["sha256"] = videos[0]["sha256"]
        with self.assertRaises(ValueError):
            self.training.assert_disjoint_splits(videos)

    def test_infeasible_threshold_is_reported(self):
        import numpy as np
        result = self.training.select_threshold(np.array([0.6, 0.8]), np.array([0.7, 0.2]),
                                                target_recall=1.0, max_normal_fpr=0.0)
        self.assertFalse(result["feasible"])
        self.assertIsNone(result["threshold"])

    def test_holdout_features_do_not_affect_scaler_weights_or_threshold(self):
        import numpy as np
        videos = []
        values = []
        for split in ("train", "validation"):
            for label in (0, 1):
                for number in range(2):
                    identifier = f"{split}_{label}_{number}"
                    videos.append(self.video(identifier, label, split))
                    values.append([(-1 if label == 0 else 1) * (1.5 + number * 0.5)])
        videos.append(self.video("unseen_holdout", 0, "holdout"))
        values.append([1_000_000.0])
        ids = np.array([video["video_id"] for video in videos])
        x = np.array(values)
        timestamps = np.full(len(ids), 5.0)
        metadata = {"feature_dimension": 1, "model_revision": "synthetic_unit_test",
                    "source_sha256": {video["video_id"]: video["sha256"] for video in videos}}
        first, report = self.training.fit_temporal_head(x, ids, timestamps, metadata, videos)
        x[-1, 0] = float("nan")
        second, _ = self.training.fit_temporal_head(x, ids, timestamps, metadata, videos)
        self.assertEqual(first["scaler"], second["scaler"])
        self.assertEqual(first["classifier"], second["classifier"])
        self.assertEqual(first["threshold"], second["threshold"])
        self.assertNotIn("unseen_holdout", first["fit_video_ids"])
        self.assertNotIn("unseen_holdout", first["threshold_video_ids"])
        self.assertFalse(report["holdout_evaluated"])


if __name__ == "__main__":
    unittest.main()
