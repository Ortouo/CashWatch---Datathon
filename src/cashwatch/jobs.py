from __future__ import annotations

import importlib.util
import json
import logging
import os
import queue
import shutil
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .rules import RuleRegistry
from .schemas import DoneResponse, FailedResponse
from .settings import Settings
from .store import JobStore
from .vision.contracts import ModelUnavailableError
from .vision.model import inspect_model_directory

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Task:
    job_id: str
    work_dir: Path
    video_path: Path
    deadline_monotonic: float
    accepted_monotonic: float


def runtime_status(settings: Settings) -> dict:
    model_path = settings.model_dir.resolve()
    try:
        inspect_model_directory(model_path)
        weights_present = True
    except (ModelUnavailableError, OSError, ValueError):
        weights_present = False
    packages_present = all(importlib.util.find_spec(name) is not None for name in ("torch", "torchvision", "transformers"))
    decoders_present = all(shutil.which(name) is not None for name in ("ffmpeg", "ffprobe"))
    try:
        active_rules = len(RuleRegistry.load(settings.rules_path).active())
    except (OSError, ValueError):
        active_rules = 0
    return {
        "model_configured": bool(weights_present and packages_present),
        "model_id": settings.model_id,
        "weights_present": weights_present,
        "ml_packages_present": packages_present,
        "decoders_present": decoders_present,
        "active_rules": active_rules,
        # Configuration checks are not a successful model warmup or benchmark.
        "model_warmup_verified": False,
        "analysis_configured": bool(settings.api_key and weights_present and packages_present and decoders_present and active_rules),
        # A configuration check cannot certify accuracy, latency, or deployment.
        "ready_for_submission": False,
        "quality_validated_on_dataset": False,
    }


def kill_process_tree(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    try:
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGKILL)
        else:
            process.kill()
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        log.error("Worker did not terminate after SIGKILL")


def run_local_task(task: Task, settings: Settings, stop_event: threading.Event) -> tuple[dict, dict]:
    remaining = task.deadline_monotonic - time.monotonic()
    if remaining <= 0:
        return JobStore.failure(task.job_id, "timeout", "Истёк лимит обработки видео."), {}
    manifest = {
        "job_id": task.job_id, "video_path": str(task.video_path.resolve()),
        "work_dir": str(task.work_dir.resolve()), "model_dir": str(settings.model_dir.resolve()),
        "model_id": settings.model_id, "rules_path": str(settings.rules_path.resolve()),
        "device": settings.device, "remaining_sec": remaining,
        "deadline_monotonic": task.deadline_monotonic - 1.0,
        "max_duration_sec": settings.max_duration_sec,
        "sample_fps": settings.sample_fps, "window_seconds": settings.window_seconds,
        "overlap_seconds": settings.overlap_seconds, "max_frames_per_window": settings.max_frames_per_window,
        "max_new_tokens": settings.max_new_tokens, "max_image_edge": settings.max_image_edge,
    }
    input_path = task.work_dir / "task.json"
    input_path.write_text(json.dumps(manifest), encoding="utf-8")
    result_path = task.work_dir / "worker_result.json"
    env = os.environ.copy()
    env.pop("CASHWATCH_API_KEY", None)
    env.update({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_HUB_DISABLE_TELEMETRY": "1", "DO_NOT_TRACK": "1", "TOKENIZERS_PARALLELISM": "false"})
    # No logs, frames, prompts, keys, or generated descriptions are sent off-host.
    # Logs also stay out of the API response. The worker emits controlled errors.
    with (task.work_dir / "worker.log").open("wb") as worker_log:
        process = subprocess.Popen(
            [sys.executable, "-m", "cashwatch.worker", str(input_path)],
            stdin=subprocess.DEVNULL, stdout=worker_log, stderr=worker_log,
            env=env, start_new_session=(os.name == "posix"),
        )
        try:
            while process.poll() is None:
                if stop_event.wait(timeout=min(0.1, max(0.001, task.deadline_monotonic - time.monotonic()))):
                    kill_process_tree(process)
                    return JobStore.failure(task.job_id, "interrupted", "Сервис остановлен во время обработки."), {}
                if time.monotonic() >= task.deadline_monotonic:
                    kill_process_tree(process)
                    return JobStore.failure(task.job_id, "timeout", "Истёк лимит обработки. Отрицательный вердикт не сформирован."), {}
            if not result_path.is_file():
                return JobStore.failure(task.job_id, "worker_failed", "Локальный анализатор завершился без результата. Проверьте локальный журнал."), {}
            payload = json.loads(result_path.read_text(encoding="utf-8"))
            response = payload["response"]
            if response.get("status") == "done":
                response = DoneResponse.model_validate(response).model_dump()
            else:
                response = FailedResponse.model_validate(response).model_dump()
            if response["job_id"] != task.job_id:
                raise ValueError("Worker returned another job ID")
            details = payload.get("details", {})
            if not isinstance(details, dict):
                raise ValueError("Invalid worker details")
            return response, details
        finally:
            kill_process_tree(process)


Runner = Callable[[Task, Settings, threading.Event], tuple[dict, dict]]


class JobManager:
    def __init__(self, settings: Settings, store: JobStore, runner: Runner = run_local_task):
        self.settings, self.store, self.runner = settings, store, runner
        self.stop_event = threading.Event()
        self.pending: queue.Queue[Task] = queue.Queue()
        self.lock = threading.Lock()
        self.inflight = 0
        self.thread: threading.Thread | None = None

    def start(self) -> None:
        self.store.recover()
        # A previous interrupted process may have left a raw video behind.
        for path in self.settings.data_dir.glob("jobs/*/input.*"):
            path.unlink(missing_ok=True)
        self.thread = threading.Thread(target=self._loop, name="cashwatch-supervisor", daemon=True)
        self.thread.start()

    def reserve(self) -> bool:
        with self.lock:
            if self.stop_event.is_set() or self.inflight >= self.settings.max_inflight:
                return False
            self.inflight += 1
            return True

    def release(self) -> None:
        with self.lock:
            self.inflight = max(0, self.inflight - 1)

    def submit(self, task: Task, accepted_epoch: float) -> None:
        self.store.create(task.job_id, accepted_epoch, accepted_epoch + self.settings.analysis_timeout_sec)
        self.pending.put(task)

    def _cleanup(self) -> None:
        expired = self.store.delete_expired(time.time() - self.settings.retention_hours * 3600)
        for job_id in expired:
            shutil.rmtree(self.settings.data_dir / "jobs" / job_id, ignore_errors=True)

    def _loop(self) -> None:
        while not self.stop_event.is_set():
            try:
                task = self.pending.get(timeout=1)
            except queue.Empty:
                self._cleanup()
                continue
            try:
                response, details = self.runner(task, self.settings, self.stop_event)
                elapsed = time.monotonic() - task.accepted_monotonic
                if elapsed > self.settings.analysis_timeout_sec:
                    response, details = JobStore.failure(task.job_id, "timeout", "Результат не получен в допустимое время."), {}
                details["elapsed_sec"] = round(elapsed, 3)
                self.store.finish(task.job_id, response, details)
            except Exception:
                # Never echo file paths, prompt content, or arbitrary exception text.
                log.exception("Worker supervisor failed for job %s", task.job_id)
                self.store.finish(task.job_id, JobStore.failure(task.job_id, "internal_error", "Ошибка локальной обработки. Проверьте журнал сервиса."))
            finally:
                task.video_path.unlink(missing_ok=True)
                # Needed JPEGs are already embedded in the protected result.
                shutil.rmtree(task.work_dir / "frames", ignore_errors=True)
                (task.work_dir / "worker_result.json").unlink(missing_ok=True)
                (task.work_dir / "task.json").unlink(missing_ok=True)
                self.release()
                self.pending.task_done()
                self._cleanup()

    def close(self) -> None:
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=8)
        while True:
            try:
                task = self.pending.get_nowait()
            except queue.Empty:
                break
            self.store.finish(task.job_id, JobStore.failure(task.job_id, "interrupted", "Сервис остановлен до начала обработки."))
            task.video_path.unlink(missing_ok=True)
            self.release()
            self.pending.task_done()

