from __future__ import annotations

import asyncio
import secrets
import shutil
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Callable

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.datastructures import UploadFile
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.types import ASGIApp, Receive, Scope, Send

from .jobs import JobManager, Runner, Task, run_local_task, runtime_status
from .rules import RuleRegistry
from .runtime_contract import build_runtime_contract
from .schemas import DoneResponse, FailedResponse, ProcessingResponse
from .settings import Settings
from .store import JobStore


class _BodyTooLarge(Exception):
    pass


class RequestGate:
    """Authentication and byte limit happen before multipart parsing/spooling."""

    def __init__(self, app: ASGIApp, settings: Settings):
        self.app, self.settings = app, settings

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        state = scope.setdefault("state", {})
        state["received_monotonic"] = time.monotonic()
        state["received_epoch"] = time.time()
        headers = {key.lower(): value for key, value in scope["headers"]}
        api = scope["path"].startswith("/api/v1/")
        if api:
            if not self.settings.api_key:
                await JSONResponse({"detail": "Создайте локальный API-ключ: python scripts/init_env.py"}, status_code=503)(scope, receive, send)
                return
            supplied = headers.get(b"x-api-key", b"")
            if not secrets.compare_digest(supplied, self.settings.api_key.encode("utf-8")):
                await JSONResponse({"detail": "Неверный или отсутствующий X-API-Key"}, status_code=401)(scope, receive, send)
                return
        limit = self.settings.max_upload_bytes + 2 * 1024 * 1024
        if scope["method"] == "POST":
            try:
                size = int(headers.get(b"content-length", b"0"))
            except ValueError:
                await JSONResponse({"detail": "Invalid Content-Length"}, status_code=400)(scope, receive, send)
                return
            if size < 0 or size > limit:
                await JSONResponse({"detail": "Файл превышает допустимый размер"}, status_code=413)(scope, receive, send)
                return
        consumed = 0
        started = False

        async def checked_receive():
            nonlocal consumed
            message = await receive()
            if message["type"] == "http.request":
                consumed += len(message.get("body", b""))
                if consumed > limit:
                    raise _BodyTooLarge()
            return message

        async def safe_send(message):
            nonlocal started
            if message["type"] == "http.response.start":
                started = True
                message["headers"] = list(message.get("headers", [])) + [
                    (b"x-content-type-options", b"nosniff"),
                    (b"referrer-policy", b"no-referrer"),
                    (b"content-security-policy", b"default-src 'self'; img-src 'self' data: blob:; media-src 'self' blob:; style-src 'self'; script-src 'self'; connect-src 'self'; frame-ancestors 'none'; object-src 'none'; base-uri 'self'"),
                ]
                if api:
                    message["headers"].append((b"cache-control", b"no-store"))
            await send(message)

        try:
            await self.app(scope, checked_receive, safe_send)
        except _BodyTooLarge:
            if not started:
                await JSONResponse({"detail": "Файл превышает допустимый размер"}, status_code=413)(scope, receive, safe_send)


def create_app(
    settings: Settings | None = None,
    *,
    runner: Runner = run_local_task,
    readiness_check: Callable[[Settings], dict] = runtime_status,
) -> FastAPI:
    settings = settings or Settings.from_env()
    settings.data_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    jobs_root = settings.data_dir / "jobs"
    jobs_root.mkdir(exist_ok=True, mode=0o700)
    store = JobStore(settings.data_dir / "jobs.sqlite3")
    manager = JobManager(settings, store, runner)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        manager.start()
        yield
        await asyncio.to_thread(manager.close)

    app = FastAPI(
        title="CashWatch Datathon API", version="0.1.0", lifespan=lifespan,
        docs_url=None, redoc_url=None, openapi_url="/api/v1/openapi.json",
    )
    app.state.manager, app.state.store, app.state.settings = manager, store, settings
    app.add_middleware(RequestGate, settings=settings)
    web_dir = Path(__file__).parent / "web"
    app.mount("/static", StaticFiles(directory=web_dir), name="static")

    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(web_dir / "index.html", headers={"Cache-Control": "no-cache"})

    @app.get("/healthz", include_in_schema=False)
    def health():
        return {"status": "alive"}

    @app.get("/api/v1/capabilities")
    def capabilities():
        return {**readiness_check(settings), "analysis_timeout_sec": settings.analysis_timeout_sec,
                "max_upload_mb": settings.max_upload_mb, "accepted_extensions": list(settings.accepted_extensions),
                "runtime_contract": build_runtime_contract(settings)}

    @app.get("/api/v1/rules")
    def rules():
        try:
            return RuleRegistry.load(settings.rules_path).model_dump()
        except (ValueError, OSError):
            raise HTTPException(503, "Реестр правил недоступен или некорректен")

    @app.post(
        "/api/v1/analyze", status_code=202, response_model=ProcessingResponse,
        openapi_extra={"requestBody": {"required": True, "content": {"multipart/form-data": {"schema": {
            "type": "object", "required": ["file"], "properties": {"file": {"type": "string", "format": "binary"}}
        }}}}},
    )
    async def analyze(request: Request):
        state = readiness_check(settings)
        if not state.get("model_configured"):
            raise HTTPException(503, "Локальная модель не настроена. Установите ML-зависимости и загрузите веса локально.")
        if not state.get("decoders_present", True):
            raise HTTPException(503, "FFmpeg и FFprobe не установлены")
        if state.get("active_rules", 0) < 1:
            raise HTTPException(503, "Ни одно проверенное правило не включено. Сначала подтвердите применимость правил к локальным данным.")
        if not manager.reserve():
            raise HTTPException(429, "Анализатор занят. Повторите запрос после завершения текущего видео.", headers={"Retry-After": "5"})
        job_id = uuid.uuid4().hex
        work_dir = jobs_root / job_id
        submitted = False
        try:
            work_dir.mkdir(mode=0o700)
            if "multipart/form-data" not in request.headers.get("content-type", "").lower():
                raise HTTPException(415, "Ожидается multipart/form-data с полем file")
            remaining = settings.analysis_timeout_sec - (time.monotonic() - request.state.received_monotonic)
            async with asyncio.timeout(max(0.001, remaining)):
                async with request.form(max_files=1, max_fields=0, max_part_size=settings.max_upload_bytes) as form:
                    uploaded = form.get("file")
                    if not isinstance(uploaded, UploadFile):
                        raise HTTPException(422, "Обязательное поле file не найдено")
                    extension = Path(uploaded.filename or "").suffix.lower()
                    if extension not in settings.accepted_extensions:
                        raise HTTPException(415, "Поддерживаются .mov и .mp4")
                    path = work_dir / f"input{extension}"
                    total = 0
                    with path.open("xb") as output:
                        while chunk := await uploaded.read(1024 * 1024):
                            total += len(chunk)
                            if total > settings.max_upload_bytes:
                                raise HTTPException(413, "Файл превышает допустимый размер")
                            output.write(chunk)
                    if not total:
                        raise HTTPException(400, "Загружен пустой файл")
            accepted_monotonic = request.state.received_monotonic
            if time.monotonic() - accepted_monotonic >= settings.analysis_timeout_sec:
                raise HTTPException(408, "Загрузка заняла больше доступного времени обработки")
            task = Task(job_id, work_dir, path, accepted_monotonic + settings.analysis_timeout_sec, accepted_monotonic)
            manager.submit(task, request.state.received_epoch)
            submitted = True
            return ProcessingResponse(job_id=job_id)
        except TimeoutError:
            raise HTTPException(408, "Истёк лимит времени загрузки")
        finally:
            if not submitted:
                manager.release()
                shutil.rmtree(work_dir, ignore_errors=True)

    def lookup(job_id: str) -> dict:
        # Never resolve a filesystem path from this parameter.
        if len(job_id) != 32 or any(char not in "0123456789abcdef" for char in job_id):
            raise HTTPException(404, "Проверка не найдена")
        job = store.get(job_id)
        if not job:
            raise HTTPException(404, "Проверка не найдена или срок хранения истёк")
        if job["status"] == "processing" and time.time() > job["deadline"]:
            store.finish(job_id, JobStore.failure(job_id, "timeout", "Истёк лимит обработки видео."))
            job = store.get(job_id)
        return job

    @app.get("/api/v1/jobs/{job_id}", response_model=ProcessingResponse | DoneResponse | FailedResponse)
    def job_status(job_id: str):
        return lookup(job_id)["response"]

    @app.get("/api/v1/jobs/{job_id}/details")
    def job_details(job_id: str):
        job = lookup(job_id)
        return {"job_id": job_id, "status": job["status"], **job["details"]}

    @app.exception_handler(StarletteHTTPException)
    async def http_error(request: Request, exc: StarletteHTTPException):
        return JSONResponse({"detail": exc.detail}, status_code=exc.status_code, headers=exc.headers)

    original_openapi = app.openapi

    def openapi_with_key():
        schema = original_openapi()
        schema.setdefault("components", {}).setdefault("securitySchemes", {})["ApiKey"] = {"type": "apiKey", "in": "header", "name": "X-API-Key"}
        schema["security"] = [{"ApiKey": []}]
        return schema

    app.openapi = openapi_with_key
    return app


def app_factory() -> FastAPI:
    return create_app()

