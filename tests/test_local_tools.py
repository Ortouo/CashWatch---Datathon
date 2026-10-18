"""Local benchmark integrity tests; no ML weights, real bank media, or network."""
from __future__ import annotations

import importlib.util
import contextlib
import io
import json
import os
import socket
import stat
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


def load_script(name):
    specification = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


preflight = load_script("preflight")
benchmark = load_script("benchmark_local")


def local_unix_sockets_available():
    try:
        first, second = socket.socketpair()
        first.close()
        second.close()
        return True
    except OSError:
        return False


class LocalToolTests(unittest.TestCase):
    def test_private_report_replaces_old_permissions(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "report.json"
            output.write_text("old")
            output.chmod(0o644)
            preflight.write_private_json(output, {"measured": False})
            self.assertEqual(json.loads(output.read_text()), {"measured": False})
            if os.name == "posix":
                self.assertEqual(stat.S_IMODE(output.stat().st_mode), 0o600)

    def test_report_rejects_nonfinite_values_without_replacing_previous_file(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "report.json"
            preflight.write_private_json(output, {"old": True})
            with self.assertRaises(ValueError):
                preflight.write_private_json(output, {"elapsed": float("nan")})
            self.assertEqual(json.loads(output.read_text()), {"old": True})

    def test_m4_profile_covers_its_sampling_budget(self):
        benchmark.validate_settings(benchmark.PROFILES["mac-m4"])
        settings = {**benchmark.PROFILES["mac-m4"], "sample_fps": 2.0}
        with self.assertRaisesRegex(ValueError, "Frame budget"):
            benchmark.validate_settings(settings)

    def test_rtx_profile_keeps_all_sampled_pts_in_bounded_contiguous_windows(self):
        from cashwatch.vision.contracts import EvidenceFrame
        from cashwatch.vision.video import SampledVideo, plan_windows

        profile = benchmark.PROFILES["rtx-4050"]
        benchmark.validate_settings(profile)
        self.assertEqual(profile["sample_fps"], benchmark.PROFILES["current"]["sample_fps"])
        self.assertEqual(profile["max_new_tokens"], benchmark.PROFILES["current"]["max_new_tokens"])
        # Synthetic noninteger PTS; no frame files are opened and no ML runs.
        frames = tuple(EvidenceFrame(f"f{i:06d}", i * 1.001, Path(f"synthetic-{i}.jpg")) for i in range(146))
        sampled = SampledVideo(145.5, frames, 1920, 1080, profile["sample_fps"])
        windows = plan_windows(sampled, **{key: profile[key] for key in
                                          ("window_seconds", "overlap_seconds", "max_frames_per_window")})
        self.assertEqual(windows[0][0], 0.0)
        self.assertEqual(windows[-1][1], sampled.duration_sec)
        self.assertEqual({frame.frame_id for _, _, batch in windows for frame in batch},
                         {frame.frame_id for frame in frames})
        self.assertTrue(all(0 < len(batch) <= profile["max_frames_per_window"] for _, _, batch in windows))
        self.assertTrue(all(start <= previous_end for (_, previous_end, _), (start, _, _) in zip(windows, windows[1:])))

    def test_rtx_profile_requires_explicit_cuda_before_media_access(self):
        with patch.object(sys, "argv", ["benchmark_local.py", "--video", "synthetic.mov",
                                       "--out", "synthetic-report.json", "--profile", "rtx-4050"]), \
             patch.object(benchmark, "os", SimpleNamespace(name="posix")), \
             patch.object(benchmark.Path, "resolve", side_effect=AssertionError("Media access attempted")), \
             patch.object(benchmark, "supervise_run") as supervise, contextlib.redirect_stderr(io.StringIO()) as errors:
            with self.assertRaises(SystemExit) as raised:
                benchmark.main()
        self.assertEqual(raised.exception.code, 2)
        self.assertIn("requires --device cuda", errors.getvalue())
        supervise.assert_not_called()

    def test_native_windows_benchmark_refuses_before_media_access_or_process_creation(self):
        with patch.object(sys, "argv", ["benchmark_local.py", "--video", "synthetic.mov",
                                       "--out", "synthetic-report.json", "--profile", "rtx-4050", "--device", "cuda"]), \
             patch.object(benchmark, "os", SimpleNamespace(name="nt")), \
             patch.object(benchmark.Path, "resolve", side_effect=AssertionError("Media access attempted")), \
             patch.object(benchmark, "supervise_run") as supervise, contextlib.redirect_stderr(io.StringIO()) as errors:
            with self.assertRaises(SystemExit) as raised:
                benchmark.main()
        self.assertEqual(raised.exception.code, 2)
        self.assertIn("Native Windows is unsupported", errors.getvalue())
        supervise.assert_not_called()

    def test_native_windows_preflight_keeps_diagnostics_but_never_claims_readiness(self):
        with tempfile.TemporaryDirectory() as temporary:
            with patch.object(preflight, "sys", SimpleNamespace(platform="win32", version_info=(3, 12))), \
                 patch.object(preflight, "host_info", return_value={"os": "Windows", "python_is_rosetta_translated": False}), \
                 patch.object(preflight, "package_versions", return_value={name: "synthetic" for name in preflight.REQUIRED_PACKAGES}), \
                 patch.object(preflight.shutil, "which", return_value="synthetic-ffmpeg"), \
                 patch.object(preflight, "command_line", return_value="synthetic decoder version"), \
                 patch("cashwatch.vision.model.inspect_model_directory", return_value={"architecture": "qwen3_vl"}), \
                 patch.object(preflight, "probe_runtime_subprocess", return_value={"ok": True, "selected_device": "cuda"}), \
                 patch("cashwatch.network_guard.check_in_subprocess", return_value={"ok": False, "os_enforced": False, "reason": "unsupported_os"}):
                report = preflight.run_preflight(Path(temporary), ROOT / "config/rules.json", "cuda")
        self.assertIn("unsupported_native_windows_supervisor", report["failures"])
        self.assertFalse(report["environment_ready_for_benchmark"])
        self.assertFalse(report["configuration_ready_for_jury"])
        self.assertTrue(report["runtime_probe"]["ok"])
        self.assertEqual(report["checkpoint"]["structural_check"], "passed")
        self.assertTrue(all(binary["available"] for binary in report["binaries"].values()))

    def test_invalid_sampling_cannot_be_timed_as_valid(self):
        for key, value in (("sample_fps", float("nan")), ("overlap_seconds", 12.0), ("max_new_tokens", 32)):
            with self.subTest(key=key):
                with self.assertRaises(ValueError):
                    benchmark.validate_settings({**benchmark.PROFILES["mac-m4"], key: value})

    def test_benchmark_records_exact_profile_contract_and_rejects_midrun_revision_drift(self):
        from cashwatch import runtime_contract

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            video = root / "synthetic.mov"
            video.write_bytes(b"synthetic metadata integration test; never decoded")
            rules = root / "rules.json"
            rules.write_text(json.dumps({"schema_version": 1, "rules": []}))
            model = root / "model"
            model.mkdir()
            manifest = {
                "model_id": "Qwen/Qwen3-VL-2B-Instruct", "resolved_revision": "a" * 40,
                "files": {name: {"bytes": 1, "sha256": "1" * 64} for name in
                          ("config.json", "tokenizer_config.json", "model.safetensors")},
            }
            provenance = model / "cashwatch_model_manifest.json"
            source = root / "synthetic-package"
            source.mkdir()
            (source / "__init__.py").write_text("# Synthetic runtime source\n")
            output = root / "report.json"

            for drift in (False, True):
                with self.subTest(drift=drift):
                    manifest["resolved_revision"] = "a" * 40
                    provenance.write_text(json.dumps(manifest))

                    def simulated_run(_task, seconds):
                        # This tests report plumbing, not real GPU/inference success.
                        if drift:
                            manifest["resolved_revision"] = "b" * 40
                            provenance.write_text(json.dumps(manifest))
                        return {"status": "complete", "latency_budget_passed": True, "wall_seconds": 1.0}

                    arguments = ["benchmark_local.py", "--video", str(video), "--out", str(output),
                                 "--model-dir", str(model), "--rules", str(rules), "--profile", "rtx-4050",
                                 "--device", "cuda", "--mode", "jury", "--seconds", "270"]
                    with patch.object(sys, "argv", arguments), \
                         patch.object(runtime_contract, "_PACKAGE_ROOT", source), \
                         patch.object(benchmark, "host_info", return_value={}), \
                         patch.object(benchmark, "package_versions", return_value={}), \
                         patch.object(benchmark, "supervise_run", side_effect=simulated_run), \
                         contextlib.redirect_stdout(io.StringIO()):
                        code = benchmark.main()
                    report = json.loads(output.read_text())
                    contract = runtime_contract.validate_runtime_contract(report["runtime_contract"], require_complete=True)
                    self.assertEqual(contract["settings"], {
                        **benchmark.PROFILES["rtx-4050"], "device": "cuda", "analysis_timeout_sec": 270.0,
                        "max_inflight": 1, "max_upload_mb": 100,
                    })
                    self.assertEqual(report["contract_stable"], not drift)
                    self.assertEqual(report["runtime_contract_differences"], ["model.revision"] if drift else [])
                    self.assertEqual(code, 1 if drift else 0)

    def test_failure_category_does_not_copy_exception_message(self):
        self.assertEqual(
            benchmark.failure_category(RuntimeError("MPS backend out of memory; private/path")),
            "memory_exhausted",
        )
        self.assertEqual(
            benchmark.failure_category(NotImplementedError("Operator private-data is not implemented for MPS")),
            "mps_operation_unsupported",
        )

    @unittest.skipUnless(sys.platform != "linux" or local_unix_sockets_available(), "The outer sandbox blocks AF_UNIX before the worker guard")
    def test_missing_model_produces_measured_failure_not_latency_pass(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            # A nonempty placeholder is sufficient: missing weights must fail before decode.
            video = root / "private-video.mov"
            video.write_bytes(b"local-test-only")
            output = root / "benchmark.json"
            result = subprocess.run([
                sys.executable, str(ROOT / "scripts/benchmark_local.py"),
                "--video", str(video), "--model-dir", str(root / "absent-model"),
                "--rules", str(ROOT / "config/rules.json"), "--seconds", "5", "--out", str(output),
            ], capture_output=True, text=True, timeout=10, check=False)
            self.assertEqual(result.returncode, 1)
            report = json.loads(output.read_text())
            run = report["runs"][0]
            self.assertEqual(run["error_code"], "model_unavailable")
            self.assertFalse(run["model_loaded"])
            self.assertFalse(run["latency_budget_passed"])
            self.assertFalse(report["accuracy_validated"])
            self.assertNotIn(str(video), output.read_text())
            self.assertNotIn("image_base64", output.read_text())
            self.assertNotIn("private-video", result.stdout)

    def test_jury_mode_does_not_enable_unreviewed_rules(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rules = root / "rules.json"
            rules.write_text(json.dumps({"schema_version": 1, "rules": []}))
            task = {
                "work_dir": str(root), "mode": "jury", "rules_path": str(rules),
                "started_monotonic": __import__("time").monotonic(),
            }
            # This unit test targets the rule gate. Never install an irreversible
            # process filter inside the test runner; real guard tests use children.
            with patch("cashwatch.network_guard.enforce_network_guard", return_value={"os_enforced": False, "reason": "unit_test"}):
                result = benchmark.child_run(task)
            self.assertEqual(result["error_code"], "no_active_reviewed_rules")
            self.assertFalse(result["model_loaded"])
            self.assertNotIn("jury_verdict", result)

    @unittest.skipUnless(os.name == "posix", "Process-group supervisor targets macOS/Linux")
    def test_supervisor_enforces_deadline_without_trusting_checkpoint(self):
        real_popen = subprocess.Popen
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            preflight.write_private_json(root / "checkpoint.json", {"status": "complete"})

            def sleeping_process(_arguments, **kwargs):
                return real_popen([sys.executable, "-c", "import time; time.sleep(60)"], **kwargs)

            with patch.object(benchmark.subprocess, "Popen", side_effect=sleeping_process):
                result = benchmark.supervise_run({"work_dir": str(root)}, 0.15)
            self.assertEqual(result["status"], "timeout")
            self.assertTrue(result["supervisor_hard_stop"])
            self.assertFalse(result["latency_budget_passed"])
            self.assertLess(result["wall_seconds"], 3.0)

    @unittest.skipUnless(os.name == "posix", "Process-group supervisor targets macOS/Linux")
    def test_crashed_child_cannot_pass_by_writing_complete_first(self):
        real_popen = subprocess.Popen
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            preflight.write_private_json(root / "result.json", {"status": "complete"})

            def failing_process(_arguments, **kwargs):
                return real_popen([sys.executable, "-c", "raise SystemExit(2)"], **kwargs)

            with patch.object(benchmark.subprocess, "Popen", side_effect=failing_process):
                result = benchmark.supervise_run({"work_dir": str(root)}, 5.0)
            self.assertEqual(result["error_code"], "child_nonzero_exit")
            self.assertFalse(result["latency_budget_passed"])


if __name__ == "__main__":
    unittest.main()
