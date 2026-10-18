#!/usr/bin/env python3
"""Time the actual offline analyzer on a local video; save numerical metrics only."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import signal
import statistics
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from preflight import OFFLINE_ENV, host_info, package_versions, write_private_json
from cashwatch.runtime_contract import build_runtime_contract, compare_runtime_contracts
from cashwatch.settings import Settings

PROFILES = {
    "current": {
        "sample_fps": 1.0, "window_seconds": 12.0, "overlap_seconds": 2.0,
        "max_frames_per_window": 16, "max_new_tokens": 768,
        "max_image_edge": 768, "max_duration_sec": 180.0,
    },
    "mac-m4": {
        "sample_fps": 1.0, "window_seconds": 12.0, "overlap_seconds": 2.0,
        "max_frames_per_window": 16, "max_new_tokens": 768,
        "max_image_edge": 512, "max_duration_sec": 180.0,
    },
    # Unmeasured initial profile for a small-VRAM GPU. All sampled frames are
    # still scheduled; shorter overlapping windows trade context/call overhead
    # for lower per-call image activation and KV-cache demand.
    "rtx-4050": {
        "sample_fps": 1.0, "window_seconds": 6.0, "overlap_seconds": 2.0,
        "max_frames_per_window": 8, "max_new_tokens": 768,
        "max_image_edge": 384, "max_duration_sec": 180.0,
    },
}


def validate_settings(settings: dict) -> None:
    for name in ("sample_fps", "window_seconds", "overlap_seconds", "max_duration_sec"):
        if not math.isfinite(settings[name]):
            raise ValueError(f"{name} must be finite")
    if not 0 < settings["sample_fps"] <= 30 or settings["window_seconds"] <= 0:
        raise ValueError("Require sample_fps in (0, 30] and positive window_seconds")
    if not 0 <= settings["overlap_seconds"] < settings["window_seconds"]:
        raise ValueError("Require 0 <= overlap_seconds < window_seconds")
    if not 64 <= settings["max_image_edge"] <= 2048:
        raise ValueError("max_image_edge must be in [64, 2048]")
    if not 64 <= settings["max_new_tokens"] <= 4096:
        raise ValueError("max_new_tokens must be in [64, 4096]")
    if not 0 < settings["max_duration_sec"] <= 180:
        raise ValueError("max_duration_sec must be in (0, 180]")
    if settings["max_frames_per_window"] < max(2, math.ceil(settings["sample_fps"] * settings["window_seconds"])):
        raise ValueError("Frame budget cannot cover the chosen sample rate and window length")


def peak_rss_bytes() -> int | None:
    try:
        import resource
        amount = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    except (ImportError, OSError, ValueError):
        return None
    # getrusage reports bytes on macOS, KiB on Linux. Other platforms are unknown.
    if sys.platform == "darwin":
        return int(amount)
    if sys.platform.startswith("linux"):
        return int(amount * 1024)
    return None


def memory_snapshot() -> dict:
    values = {"process_peak_rss_bytes": peak_rss_bytes()}
    torch = sys.modules.get("torch")
    if torch is None:
        return values
    try:
        if torch.backends.mps.is_available():
            for name in ("current_allocated_memory", "driver_allocated_memory", "recommended_max_memory"):
                function = getattr(torch.mps, name, None)
                if function is not None:
                    values[f"mps_{name}_bytes"] = int(function())
        if torch.cuda.is_available():
            values.update({
                "cuda_allocated_bytes": int(torch.cuda.memory_allocated()),
                "cuda_reserved_bytes": int(torch.cuda.memory_reserved()),
                "cuda_peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
                "cuda_peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
            })
    except Exception as exc:
        values["device_metric_error_class"] = type(exc).__name__
    return values


class MetricRecorder:
    """Periodic MPS snapshots are lower bounds on peaks, not exact allocator peaks."""

    def __init__(self, checkpoint: Path, started: float) -> None:
        self.checkpoint = checkpoint
        self.started = started
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.state: dict = {
            "status": "running", "model_loaded": False, "generation_attempted": False,
            "stage": "starting", "stage_events": [], "memory_sample_count": 0,
            "observed_memory_max_bytes": {},
        }
        self.thread = threading.Thread(target=self._loop, name="local-memory-sampler", daemon=True)

    def _sample_unlocked(self) -> None:
        memory = memory_snapshot()
        self.state["memory_sample_count"] += 1
        self.state["last_memory"] = memory
        for name, amount in memory.items():
            if isinstance(amount, int):
                self.state["observed_memory_max_bytes"][name] = max(
                    amount, self.state["observed_memory_max_bytes"].get(name, 0),
                )
        self.state["elapsed_seconds"] = round(time.monotonic() - self.started, 4)
        write_private_json(self.checkpoint, self.state)

    def update(self, **changes) -> None:
        with self.lock:
            self.state.update(changes)
            self._sample_unlocked()

    def stage(self, stage: str, **changes) -> None:
        # The callback emits only fixed stage labels, never generated model text.
        if not re.fullmatch(r"[a-z_0-9]+", stage):
            raise ValueError("Unexpected progress stage")
        with self.lock:
            self.state["stage_events"].append({
                "stage": stage, "elapsed_seconds": round(time.monotonic() - self.started, 4),
            })
            self.state.update(changes)
            self.state["stage"] = stage
            self._sample_unlocked()

    def _loop(self) -> None:
        while not self.stop_event.wait(0.5):
            with self.lock:
                self._sample_unlocked()

    def stop(self) -> dict:
        self.stop_event.set()
        self.thread.join(timeout=2.0)
        with self.lock:
            self._sample_unlocked()
            return dict(self.state)


def synchronize_device(analyzer) -> None:
    if analyzer.model.device == "mps":
        analyzer.model.torch.mps.synchronize()
    elif analyzer.model.device == "cuda":
        analyzer.model.torch.cuda.synchronize()


def failure_category(exception: Exception) -> str:
    """Return a fixed diagnostic code, never the library's potentially sensitive text."""
    from cashwatch.vision.contracts import GenerationTimeoutError, ModelUnavailableError
    from cashwatch.network_guard import NetworkIsolationError

    if isinstance(exception, NetworkIsolationError):
        return "network_isolation_unavailable"

    current: BaseException | None = exception
    for _ in range(5):
        if current is None:
            break
        message = str(current).casefold()
        if "out of memory" in message or type(current).__name__ == "OutOfMemoryError":
            return "memory_exhausted"
        if "mps" in message and ("not implemented" in message or "not supported" in message):
            return "mps_operation_unsupported"
        current = current.__cause__
    if isinstance(exception, GenerationTimeoutError):
        return "timeout"
    if isinstance(exception, ModelUnavailableError):
        return "model_unavailable"
    return "local_pipeline_error"


def child_run(task: dict) -> dict:
    """Same analyzer/parser/registry/report code as the service, without an HTTP layer."""
    os.environ.update(OFFLINE_ENV)
    work_dir = Path(task["work_dir"])
    recorder = MetricRecorder(work_dir / "checkpoint.json", task["started_monotonic"])
    try:
        from cashwatch.network_guard import enforce_network_guard
        # Before the monitoring thread and before every optional ML import.
        network_guard = enforce_network_guard()
        recorder.state["network_isolation"] = network_guard
        recorder.thread.start()
        recorder.stage("loading_dependencies")
        from cashwatch.engine import IncompleteAnalysisError, build_report
        from cashwatch.rules import RuleRegistry
        from cashwatch.schemas import DoneResponse
        from cashwatch.vision.analyzer import LocalVideoAnalyzer
        from cashwatch.worker import rule_specs

        registry = RuleRegistry.load(Path(task["rules_path"]))
        active = registry.active()
        recorder.update(active_reviewed_rules=len(active), benchmark_mode=task["mode"])
        if task["mode"] == "jury" and not active:
            recorder.update(status="failed", error_code="no_active_reviewed_rules")
            return recorder.stop()
        analyzer = LocalVideoAnalyzer(
            Path(task["model_dir"]), device=task["device"], **task["settings"],
        )
        recorder.stage("loading_model")
        load_started = time.monotonic()
        analyzer.warmup()
        synchronize_device(analyzer)
        recorder.update(
            model_loaded=True, model_load_seconds=round(time.monotonic() - load_started, 4),
            selected_device=analyzer.model.device, backend=analyzer.model.backend,
            model_id=analyzer.model.model_id,
            model_dtype=str(next(analyzer.model.model.parameters()).dtype),
        )

        def progress(_fraction: float, stage: str) -> None:
            if stage.startswith("analyzing_window_"):
                recorder.stage(stage, generation_attempted=True)
            else:
                recorder.stage(stage)

        analysis_started = time.monotonic()
        result = analyzer.analyze(
            Path(task["video_path"]), work_dir, rule_specs(registry),
            deadline_monotonic=task["deadline_monotonic"], progress=progress,
        )
        synchronize_device(analyzer)
        recorder.update(
            analyzer_seconds=round(time.monotonic() - analysis_started, 4),
            analyzer_complete=result.complete, duration_sec=result.duration_sec,
            sampled_frames=len(result.evidence_frames), completed_windows=len(result.analyzed_intervals),
            observation_count=len(result.observations),
            supported_observation_count=sum(o.assessment == "supported" for o in result.observations),
            uncertain_observation_count=sum(o.assessment == "uncertain" for o in result.observations),
            warning_count=len(result.warnings),
            structured_validation_failed_windows=sum(
                warning.startswith("Window ") and "failed structured validation:" in warning
                for warning in result.warnings
            ),
        )
        if task["mode"] == "jury":
            recorder.stage("building_jury_report")
            try:
                response, _details = build_report(task["job_id"], result, registry, work_dir)
                validated = DoneResponse.model_validate(response)
                recorder.update(
                    jury_response_status=validated.status, jury_verdict=validated.verdict,
                    jury_violation_count=len(validated.violations),
                )
                # Images and generated descriptions remain in memory and are discarded.
                del response, _details, validated
            except IncompleteAnalysisError:
                recorder.update(jury_response_status="failed", jury_error_code="incomplete_analysis")
        status = "complete" if result.complete else "incomplete"
        if task["mode"] == "jury" and recorder.state.get("jury_response_status") != "done":
            status = "incomplete"
        if time.monotonic() >= task["deadline_monotonic"]:
            status = "timeout"
        recorder.stage("finished", status=status)
    except Exception as exc:
        code = failure_category(exc)
        if recorder.thread.ident is None:
            recorder.thread.start()
        recorder.update(
            status="timeout" if code == "timeout" else "failed",
            error_code=code, error_class=type(exc).__name__,
        )
    return recorder.stop()


def stop_process_group(process: subprocess.Popen) -> None:
    # Each run has a fresh process group; ffmpeg descendants are stopped with it.
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait(timeout=5.0)


def supervise_run(task: dict, seconds: float) -> dict:
    work_dir = Path(task["work_dir"])
    started = time.monotonic()
    # All child phases, including imports and cold model loading, consume the same budget.
    task.update(started_monotonic=started, deadline_monotonic=started + seconds - 0.5)
    write_private_json(work_dir / "task.json", task)
    process = subprocess.Popen(
        [sys.executable, str(Path(__file__).resolve()), "--_child", str(work_dir / "task.json")],
        stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        start_new_session=True, close_fds=True, env={**os.environ, **OFFLINE_ENV},
    )
    timed_out = False
    try:
        process.wait(timeout=max(0.001, started + seconds - time.monotonic()))
    except subprocess.TimeoutExpired:
        timed_out = True
        stop_process_group(process)
    except BaseException:
        stop_process_group(process)
        raise
    elapsed = time.monotonic() - started
    result_path = work_dir / "result.json"
    checkpoint_path = work_dir / "checkpoint.json"
    result = {}
    report_complete = result_path.is_file()
    path = result_path if report_complete else checkpoint_path
    if path.is_file() and path.stat().st_size <= 2 * 1024 * 1024:
        try:
            result = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(result, dict):
                result = {}
        except (OSError, ValueError):
            result = {}
    if timed_out or elapsed > seconds:
        result.update(status="timeout", error_code="supervisor_deadline")
    elif not report_complete or result.get("status") not in {"complete", "incomplete", "failed", "timeout"}:
        result.update(status="failed", error_code="child_result_missing_or_invalid")
    elif result["status"] == "complete" and process.returncode != 0:
        result.update(status="failed", error_code="child_nonzero_exit")
    result.update(
        wall_seconds=round(elapsed, 4), child_exit_code=process.returncode,
        supervisor_hard_stop=timed_out, final_metrics_received=report_complete,
        cold_model_load_included=True, latency_budget_seconds=seconds,
    )
    result["latency_budget_passed"] = result["status"] == "complete" and elapsed <= seconds
    return result


def hash_file(path: Path) -> str:
    checksum = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            checksum.update(block)
    return checksum.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path)
    parser.add_argument("--model-dir", type=Path, default=PROJECT_ROOT / "models/Qwen3-VL-2B-Instruct")
    parser.add_argument("--rules", type=Path, default=PROJECT_ROOT / "config/rules.json")
    parser.add_argument("--mode", choices=("analyzer", "jury"), default="analyzer")
    parser.add_argument("--profile", choices=tuple(PROFILES), default="mac-m4")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--seconds", type=float, default=275.0)
    parser.add_argument("--repeat", type=int, default=1, help="Independent cold processes; no warm run is passed off as cold.")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--sample-fps", type=float)
    parser.add_argument("--window-seconds", type=float)
    parser.add_argument("--overlap-seconds", type=float)
    parser.add_argument("--max-frames-per-window", type=int)
    parser.add_argument("--max-new-tokens", type=int)
    parser.add_argument("--max-image-edge", type=int)
    parser.add_argument("--max-duration-sec", type=float)
    parser.add_argument("--_child", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args._child:
        task = json.loads(args._child.read_text(encoding="utf-8"))
        result = child_run(task)
        write_private_json(Path(task["work_dir"]) / "result.json", result)
        return 0 if result["status"] == "complete" else 1
    if args.video is None or args.out is None:
        parser.error("--video and --out are required")
    if os.name != "posix":
        parser.error("Native Windows is unsupported by the process-group supervisor; use WSL2/Linux or macOS")
    if args.profile == "rtx-4050" and args.device != "cuda":
        parser.error("--profile rtx-4050 requires --device cuda; automatic CPU fallback is not a GPU benchmark")
    if not 5 <= args.seconds <= 285 or not 1 <= args.repeat <= 10:
        parser.error("--seconds must be in [5, 285], --repeat in [1, 10]")
    settings = dict(PROFILES[args.profile])
    for name in settings:
        supplied = getattr(args, name)
        if supplied is not None:
            settings[name] = supplied
    try:
        validate_settings(settings)
        video_path = args.video.resolve(strict=True)
        if not video_path.is_file() or video_path.suffix.lower() not in {".mov", ".mp4"}:
            raise ValueError("--video must be a local MOV or MP4 file")
        if not 0 < video_path.stat().st_size <= 100 * 1024 * 1024:
            raise ValueError("--video must be nonempty and at most 100 MiB")
        output = args.out.resolve()
        if output in {video_path, args.rules.resolve()} or output.is_relative_to(args.model_dir.resolve()):
            raise ValueError("Report output must not overwrite the video, rules, or model files")
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    contract_settings = Settings(
        api_key="", model_dir=args.model_dir, rules_path=args.rules, device=args.device,
        analysis_timeout_sec=args.seconds, max_inflight=1, max_upload_mb=100, **settings,
    )
    runtime_contract = build_runtime_contract(contract_settings)
    report = {
        "schema_version": 1, "kind": "cashwatch_local_benchmark",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "mode": args.mode, "profile": args.profile, "settings": settings, "requested_device": args.device,
        "hardware": host_info(), "packages": package_versions(),
        "source_sha256": hash_file(video_path), "source_bytes": video_path.stat().st_size,
        "rules_sha256": runtime_contract["rules_sha256"],
        "runtime_contract": runtime_contract, "contract_stable": False,
        "requested_runs": args.repeat, "runs": [], "accuracy_validated": False,
        "benchmark_includes_http_upload": False, "benchmark_includes_model_cold_load": True,
        "network_namespace_isolation_verified": False, "offline_flags_used": OFFLINE_ENV,
        "contains_video_frames_or_generated_descriptions": False,
        "memory_notes": [
            "Process peak RSS excludes separate FFmpeg processes. GPU driver and RSS figures overlap on unified memory; do not add them.",
            "MPS figures are sampled every 0.5 seconds and at phase boundaries; observed maxima may miss short peaks.",
            "A killed process supplies only the last checkpoint. CUDA peak counters are reported when available.",
        ],
        "limitations": [
            "Sparse sampling may miss brief actions; processing every scheduled window is not continuous-frame inspection.",
            "Analyzer mode measures inference without creating a jury verdict or bypassing rule review.",
            "One successful run does not validate accuracy or the full HTTP five-minute requirement.",
            "Runtime identity uses declared checkpoint hashes; weight contents are not rehashed, and before/after snapshots are not atomic file locking.",
        ],
    }
    if args.profile == "rtx-4050":
        report["limitations"].extend([
            "The RTX 4050 profile is an unmeasured initial configuration, not a guarantee that the full model fits in 6 GB VRAM.",
            "Shorter overlapping windows increase model-call overhead and reduce temporal context; smaller images may lose small-object detail.",
            "Sampling frequency and strict JSON validation are preserved. No quantization or CPU offload is enabled.",
        ])
    for index in range(args.repeat):
        with tempfile.TemporaryDirectory(prefix="cashwatch-local-benchmark-") as temporary:
            task = {
                "job_id": uuid.uuid4().hex, "work_dir": temporary, "video_path": str(video_path),
                "model_dir": str(args.model_dir.resolve()), "rules_path": str(args.rules.resolve()),
                "mode": args.mode, "device": args.device, "settings": settings,
            }
            try:
                run = supervise_run(task, args.seconds)
            except OSError as exc:
                run = {"status": "failed", "error_code": "child_start_failed", "error_class": type(exc).__name__, "latency_budget_passed": False}
            run["run_number"] = index + 1
            report["runs"].append(run)
        # Temp directories and every extracted frame have now been removed.
        write_private_json(output, report)
        if run["status"] in {"failed", "timeout"}:
            break
    times = [run["wall_seconds"] for run in report["runs"] if "wall_seconds" in run]
    final_contract = build_runtime_contract(contract_settings)
    report["runtime_contract_differences"] = compare_runtime_contracts(runtime_contract, final_contract)
    report["contract_stable"] = not report["runtime_contract_differences"]
    report["summary"] = {
        "completed_runs": len(report["runs"]),
        "all_requested_runs_completed_within_budget": len(report["runs"]) == args.repeat
        and all(run["latency_budget_passed"] for run in report["runs"]),
        "median_wall_seconds": round(statistics.median(times), 4) if times else None,
        "maximum_wall_seconds": max(times) if times else None,
        "accuracy_or_recall_claim": None,
    }
    write_private_json(output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
    return 0 if report["summary"]["all_requested_runs_completed_within_budget"] and report["contract_stable"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
