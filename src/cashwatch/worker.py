"""Isolated local inference process; never starts a remote AI client."""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

# Must be set before any optional ML package import.
os.environ.update({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_HUB_DISABLE_TELEMETRY": "1", "DO_NOT_TRACK": "1"})

from .engine import IncompleteAnalysisError, build_report
from .network_guard import NetworkIsolationError, enforce_network_guard
from .rules import RuleRegistry
from .store import JobStore
from .vision.analyzer import LocalVideoAnalyzer
from .vision.contracts import GenerationTimeoutError, MediaError, ModelUnavailableError, RuleSpec, VisionError


def rule_specs(registry: RuleRegistry) -> tuple[RuleSpec, ...]:
    specs = []
    for rule in registry.active():
        # Short condition IDs reduce JSON generation cost. Their meanings stay in
        # the fixed registry; the model cannot introduce another requirement.
        conditions = tuple(f"c{index + 1}" for index in range(len(rule.required_conditions)))
        mapped = "\n".join(f"{key}: {meaning}" for key, meaning in zip(conditions, rule.required_conditions))
        specs.append(RuleSpec(
            rule_id=rule.rule_id, title=rule.title,
            requirement=f"{rule.requirement}\nRequired visible conditions:\n{mapped}",
            required_conditions=conditions, disqualifiers=tuple(rule.disqualifiers),
        ))
    return tuple(specs)


def run(manifest: dict) -> tuple[dict, dict]:
    network_guard = enforce_network_guard()
    registry = RuleRegistry.load(Path(manifest["rules_path"]))
    if not registry.active():
        raise IncompleteAnalysisError("No enabled reviewed rules")
    analyzer = LocalVideoAnalyzer(
        Path(manifest["model_dir"]), device=manifest["device"],
        sample_fps=manifest["sample_fps"], window_seconds=manifest["window_seconds"],
        overlap_seconds=manifest["overlap_seconds"], max_frames_per_window=manifest["max_frames_per_window"],
        max_new_tokens=manifest["max_new_tokens"], max_image_edge=manifest["max_image_edge"],
        max_duration_sec=manifest["max_duration_sec"],
    )
    result = analyzer.analyze(
        Path(manifest["video_path"]), Path(manifest["work_dir"]), rule_specs(registry),
        deadline_monotonic=float(manifest["deadline_monotonic"]),
    )
    response, details = build_report(manifest["job_id"], result, registry, Path(manifest["work_dir"]))
    details["network_isolation"] = network_guard
    return response, details


def main() -> int:
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python -m cashwatch.worker LOCAL_TASK_JSON")
    manifest_path = Path(sys.argv[1]).resolve(strict=True)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    work_dir = Path(manifest["work_dir"]).resolve(strict=True)
    if manifest_path.parent != work_dir or not Path(manifest["video_path"]).resolve().is_relative_to(work_dir):
        raise ValueError("Task input must stay inside its private job directory")
    try:
        response, details = run(manifest)
    except NetworkIsolationError:
        response, details = JobStore.failure(manifest["job_id"], "network_isolation_unavailable", "Защита сетевого доступа Linux worker не установлена. Обработка заблокирована; проверьте libseccomp и параметры контейнера."), {}
    except ModelUnavailableError:
        response, details = JobStore.failure(manifest["job_id"], "model_unavailable", "Локальная модель не загрузилась. Проверьте веса, зависимости и память устройства."), {}
    except GenerationTimeoutError:
        response, details = JobStore.failure(manifest["job_id"], "timeout", "Лимит локальной обработки исчерпан."), {}
    except MediaError:
        response, details = JobStore.failure(manifest["job_id"], "invalid_video", "Видео не удалось декодировать или оно превышает допустимые параметры."), {}
    except IncompleteAnalysisError:
        response, details = JobStore.failure(manifest["job_id"], "incomplete_analysis", "Анализ не завершён или нет проверенных активных правил. Отрицательный вердикт не сформирован."), {}
    except VisionError:
        response, details = JobStore.failure(manifest["job_id"], "analysis_failed", "Невозможно сформировать проверяемый результат локального анализа."), {}
    except Exception as exc:
        # Only the exception class is logged, not inputs, generated text, or secrets.
        print(f"Local worker error class: {type(exc).__name__}", file=sys.stderr)
        response, details = JobStore.failure(manifest["job_id"], "worker_error", "Ошибка локального анализатора. Проверьте совместимость модели и среды."), {}
    output = work_dir / "worker_result.json"
    temporary = output.with_suffix(".tmp")
    temporary.write_text(json.dumps({"response": response, "details": details}, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    temporary.replace(output)
    return 0 if response["status"] == "done" else 1


if __name__ == "__main__":
    raise SystemExit(main())

