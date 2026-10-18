#!/usr/bin/env python3
"""Inspect this local host, dependencies and checkpoint without downloading anything."""
from __future__ import annotations

import argparse
import contextlib
import importlib.metadata
import io
import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

OFFLINE_ENV = {
    "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
    "HF_DATASETS_OFFLINE": "1", "HF_HUB_DISABLE_TELEMETRY": "1", "DO_NOT_TRACK": "1",
}
REQUIRED_PACKAGES = (
    "torch", "torchvision", "transformers", "accelerate", "huggingface-hub",
    "pillow", "pydantic", "fastapi", "uvicorn", "python-multipart", "python-dotenv",
)


def write_private_json(path: Path, value: dict) -> None:
    """Atomic local report with owner-only permissions, including when replacing a file."""
    import tempfile

    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def command_line(arguments: list[str], timeout: float = 5.0) -> str | None:
    try:
        result = subprocess.run(arguments, capture_output=True, text=True, timeout=timeout, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def host_info() -> dict:
    """Collect non-identifying hardware fields; no hostname, serial number or user name."""
    total_memory = None
    processor = platform.processor() or None
    translated = False
    if sys.platform == "darwin":
        raw = command_line(["sysctl", "-n", "hw.memsize"])
        if raw and raw.isdigit():
            total_memory = int(raw)
        processor = command_line(["sysctl", "-n", "machdep.cpu.brand_string"]) or processor
        translated = command_line(["sysctl", "-n", "sysctl.proc_translated"]) == "1"
    else:
        try:
            total_memory = os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
        except (AttributeError, OSError, ValueError):
            pass
    return {
        "os": platform.system(), "os_release": platform.release(), "macos_version": platform.mac_ver()[0] or None,
        "architecture": platform.machine(), "processor": processor, "logical_cpus": os.cpu_count(),
        "physical_memory_bytes": total_memory, "python_version": platform.python_version(),
        "python_is_rosetta_translated": translated,
    }


def package_versions() -> dict[str, str | None]:
    versions = {}
    for package in REQUIRED_PACKAGES:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def runtime_probe(requested_device: str) -> dict:
    """Executed in a bounded subprocess. No model or tokenizer is loaded here."""
    os.environ.update(OFFLINE_ENV)
    result: dict = {"ok": False, "model_loaded": False, "generation_tested": False, "checks": {}}
    # Some optional imports print diagnostic text. Reports contain only controlled metadata.
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        try:
            import torch
            result["checks"]["torch_import"] = True
            import torchvision  # noqa: F401 -- force version/binary compatibility check
            result["checks"]["torchvision_import"] = True
            from transformers import AutoProcessor, Qwen3VLForConditionalGeneration  # noqa: F401
            result["checks"]["qwen3_vl_class_import"] = True
            mps_available = bool(torch.backends.mps.is_available())
            cuda_available = bool(torch.cuda.is_available())
            result["devices"] = {
                "mps_built": bool(torch.backends.mps.is_built()), "mps_available": mps_available,
                "cuda_available": cuda_available,
            }
            selected = requested_device
            if selected == "auto":
                selected = "cuda" if cuda_available else ("mps" if mps_available else "cpu")
            result["selected_device"] = selected
            if selected == "mps" and not mps_available:
                result["error_code"] = "mps_unavailable"
                return result
            if selected == "cuda" and not cuda_available:
                result["error_code"] = "cuda_unavailable"
                return result
            dtype = torch.float32 if selected == "cpu" else torch.float16
            if selected == "cuda" and torch.cuda.is_bf16_supported():
                dtype = torch.bfloat16
            result["selected_dtype"] = str(dtype)
            tensor = torch.ones((2, 2), dtype=dtype, device=selected)
            product = tensor @ tensor
            if selected == "mps":
                torch.mps.synchronize()
                for name in ("recommended_max_memory", "current_allocated_memory", "driver_allocated_memory"):
                    function = getattr(torch.mps, name, None)
                    if function is not None:
                        result[f"mps_{name}_bytes"] = int(function())
            elif selected == "cuda":
                torch.cuda.synchronize()
                properties = torch.cuda.get_device_properties(0)
                result["cuda_name"] = properties.name
                result["cuda_total_memory_bytes"] = int(properties.total_memory)
            if product.float().cpu().tolist() != [[2.0, 2.0], [2.0, 2.0]]:
                result["error_code"] = "tiny_device_operation_incorrect"
                return result
            result["checks"]["tiny_device_matmul"] = True
            result["ok"] = True
        except Exception as exc:
            # Exception strings from libraries can contain local paths; keep only the class.
            result["error_code"] = "runtime_import_or_device_error"
            result["error_class"] = type(exc).__name__
    return result


def probe_runtime_subprocess(device: str, timeout_sec: float) -> dict:
    environment = {**os.environ, **OFFLINE_ENV}
    try:
        completed = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--_runtime", "--device", device],
            capture_output=True, text=True, timeout=timeout_sec, env=environment, check=False,
        )
    except subprocess.TimeoutExpired:
        return {"ok": False, "error_code": "runtime_probe_timeout", "model_loaded": False}
    except OSError as exc:
        return {"ok": False, "error_code": "runtime_probe_start_failed", "error_class": type(exc).__name__}
    try:
        result = json.loads(completed.stdout)
        if not isinstance(result, dict) or type(result.get("ok")) is not bool:
            raise ValueError("invalid probe result")
        return result
    except (ValueError, TypeError):
        return {"ok": False, "error_code": "runtime_probe_failed", "exit_code": completed.returncode}


def run_preflight(model_dir: Path, rules_path: Path, device: str, timeout_sec: float = 30.0) -> dict:
    from cashwatch.network_guard import check_in_subprocess
    from cashwatch.vision.contracts import ModelUnavailableError
    from cashwatch.vision.model import inspect_model_directory

    host = host_info()
    versions = package_versions()
    binaries = {}
    for binary in ("ffmpeg", "ffprobe"):
        found = shutil.which(binary)
        raw = command_line([found, "-version"]) if found else None
        binaries[binary] = {"available": bool(raw), "version": raw.splitlines()[0] if raw else None}
    warnings = []
    failures = []
    if sys.platform == "win32":
        failures.append("unsupported_native_windows_supervisor")
        warnings.append("Native Windows lacks this project's process-group hard stop and Linux network guard; use WSL2/Linux for CUDA.")
    if sys.version_info < (3, 11):
        failures.append("python_3_11_or_newer_required")
    if host["python_is_rosetta_translated"]:
        failures.append("use_native_arm64_python_on_apple_silicon")
    for package, version in versions.items():
        if version is None:
            failures.append(f"missing_package:{package}")
    for binary, details in binaries.items():
        if not details["available"]:
            failures.append(f"missing_or_broken_binary:{binary}")
    try:
        info = inspect_model_directory(model_dir)
        info.pop("path", None)
        model = {"structural_check": "passed", "weights_loaded": False, "hashes_verified": False, **info}
        warnings.append("A structural checkpoint check is not a weight-load, hash-integrity or generation test.")
    except (ModelUnavailableError, OSError, ValueError) as exc:
        model = {"structural_check": "failed", "weights_loaded": False, "error_class": type(exc).__name__}
        failures.append("local_model_missing_or_incomplete")
    rules = {"loaded": False, "active_reviewed_count": 0}
    try:
        from cashwatch.rules import RuleRegistry
        registry = RuleRegistry.load(rules_path)
        rules = {"loaded": True, "total_count": len(registry.rules), "active_reviewed_count": len(registry.active())}
    except Exception as exc:
        rules["error_class"] = type(exc).__name__
    if not rules["active_reviewed_count"]:
        warnings.append("No active reviewed rules: analyzer timing is possible, but jury verdicts are blocked.")
    runtime = probe_runtime_subprocess(device, timeout_sec)
    if not runtime.get("ok"):
        failures.append(runtime.get("error_code", "runtime_probe_failed"))
    if runtime.get("selected_device") == "cpu":
        warnings.append("CPU inference uses float32 in this adapter; the five-minute budget is unmeasured.")
    if os.getenv("PYTORCH_ENABLE_MPS_FALLBACK") == "1":
        warnings.append("MPS CPU fallback is enabled; actual work may run on CPU and affect the deadline.")
    if os.getenv("PYTORCH_MPS_HIGH_WATERMARK_RATIO") == "0":
        warnings.append("The MPS high watermark limit is disabled. Restore a bounded limit before benchmarking.")
    network_guard = check_in_subprocess(timeout_sec=min(timeout_sec, 10.0))
    if sys.platform == "linux" and not network_guard.get("ok"):
        failures.append("linux_inference_network_guard_unavailable")
    elif sys.platform == "darwin":
        warnings.append("The Linux socket guard is not available on macOS; independently disable or isolate external networking.")
    disk_target = model_dir.resolve()
    while not disk_target.exists():
        disk_target = disk_target.parent
    free_bytes = shutil.disk_usage(disk_target).free
    if free_bytes < 8 * 1024**3:
        warnings.append("Less than 8 GiB free on the model filesystem; allow space for weights and temporary frames.")
    return {
        "schema_version": 1, "kind": "cashwatch_local_preflight",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "environment_ready_for_benchmark": not failures,
        "configuration_ready_for_jury": not failures and bool(rules["active_reviewed_count"]),
        "accuracy_validated": False, "latency_validated": False,
        "hardware": host, "packages": versions, "binaries": binaries, "runtime_probe": runtime,
        "checkpoint": model, "rules": rules, "model_filesystem_free_bytes": free_bytes,
        "offline_flags_used": OFFLINE_ENV,
        "network_guard_check": network_guard,
        "network_namespace_isolation_verified": False,
        "failures": failures, "warnings": warnings,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=PROJECT_ROOT / "models/Qwen3-VL-2B-Instruct")
    parser.add_argument("--rules", type=Path, default=PROJECT_ROOT / "config/rules.json")
    parser.add_argument("--device", choices=("auto", "mps", "cuda", "cpu"), default="auto")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--runtime-timeout", type=float, default=30.0)
    parser.add_argument("--_runtime", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not 1 <= args.runtime_timeout <= 60:
        parser.error("--runtime-timeout must be in [1, 60] seconds")
    if args._runtime:
        print(json.dumps(runtime_probe(args.device), allow_nan=False))
        return 0
    if args.out:
        output = args.out.resolve()
        if output == args.rules.resolve() or output.is_relative_to(args.model_dir.resolve()):
            parser.error("The output report must not overwrite the rules or model directory")
    report = run_preflight(args.model_dir, args.rules, args.device, args.runtime_timeout)
    if args.out:
        write_private_json(args.out, report)
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
    return 0 if report["environment_ready_for_benchmark"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
