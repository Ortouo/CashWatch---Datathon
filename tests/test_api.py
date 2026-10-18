from __future__ import annotations

import base64
import asyncio
import io
import threading
import time
from dataclasses import replace
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from cashwatch.main import create_app
from cashwatch.runtime_contract import build_runtime_contract, validate_runtime_contract
from cashwatch.settings import Settings
from cashwatch.store import JobStore


KEY = "contract-test-key-is-only-used-in-tests"


def configured(_settings):
    # Dependency injection exists only in tests, never as a production env mode.
    return {"model_configured": True, "active_rules": 1, "decoders_present": True, "model_id": "test-fixture", "ready_for_submission": True}


def jpeg():
    buffer = io.BytesIO()
    Image.new("RGB", (16, 16), (50, 80, 120)).save(buffer, format="JPEG")
    return base64.b64encode(buffer.getvalue()).decode()


def negative_runner(task, settings, stop_event):
    return {"job_id": task.job_id, "status": "done", "verdict": "no_violation", "violations": []}, {"complete": True}


def positive_runner(task, settings, stop_event):
    event = {"type": "Synthetic contract fixture", "timecode_sec": 1.5, "description": "Only a contract test, not a detector result", "evidence_frames": [{"timestamp_sec": 1.5, "image_base64": jpeg()}]}
    return {"job_id": task.job_id, "status": "done", "verdict": "violation", "violations": [event, {**event, "timecode_sec": 3.0, "evidence_frames": [{"timestamp_sec": 3.0, "image_base64": jpeg()}]}]}, {"complete": True}


def poll(client, job_id, limit=3):
    deadline = time.monotonic() + limit
    while time.monotonic() < deadline:
        response = client.get(f"/api/v1/jobs/{job_id}", headers={"X-API-Key": KEY})
        assert response.status_code == 200
        payload = response.json()
        if payload["status"] != "processing":
            return payload
        time.sleep(0.01)
    pytest.fail("Test worker did not return")


@pytest.fixture
def settings(tmp_path):
    return Settings(api_key=KEY, data_dir=tmp_path / "runtime", rules_path=tmp_path / "rules.json", model_dir=tmp_path / "missing-model", max_upload_mb=1)


def test_missing_or_incorrect_key_rejected_before_upload(settings):
    with TestClient(create_app(settings, runner=negative_runner, readiness_check=configured)) as client:
        for headers in ({}, {"X-API-Key": "wrong"}):
            response = client.post("/api/v1/analyze", headers=headers, content=b"not multipart")
            assert response.status_code == 401
        assert list((settings.data_dir / "jobs").iterdir()) == []


def test_done_exact_case_schema_and_protected_details(settings):
    with TestClient(create_app(settings, runner=negative_runner, readiness_check=configured)) as client:
        response = client.post("/api/v1/analyze", headers={"X-API-Key": KEY}, files={"file": ("client.mov", b"synthetic", "video/quicktime")})
        assert response.status_code == 202
        initial = response.json()
        assert set(initial) == {"job_id", "status"}
        result = poll(client, initial["job_id"])
        assert result == {"job_id": initial["job_id"], "status": "done", "verdict": "no_violation", "violations": []}
        details_path = f"/api/v1/jobs/{initial['job_id']}/details"
        assert client.get(details_path).status_code == 401
        assert client.get(details_path, headers={"X-API-Key": KEY}).json()["complete"] is True
        assert client.get(f"/api/v1/jobs/{initial['job_id']}", headers={"X-API-Key": KEY}).headers["cache-control"] == "no-store"
    assert not list((settings.data_dir / "jobs").glob("*/input.*"))


def test_multiple_events_have_actual_jpeg_payloads(settings):
    with TestClient(create_app(settings, runner=positive_runner, readiness_check=configured)) as client:
        accepted = client.post("/api/v1/analyze", headers={"X-API-Key": KEY}, files={"file": ("fixture.MOV", b"fixture")}).json()
        result = poll(client, accepted["job_id"])
        assert len(result["violations"]) == 2
        assert result["verdict"] == "violation"
        assert set(result) == {"job_id", "status", "verdict", "violations"}
        for event in result["violations"]:
            data = base64.b64decode(event["evidence_frames"][0]["image_base64"], validate=True)
            with Image.open(io.BytesIO(data)) as image:
                assert image.format == "JPEG"


@pytest.mark.parametrize("filename,content,status", [("bad.txt", b"x", 415), ("empty.mov", b"", 400), ("large.mov", b"x" * (1024 * 1024 + 1), 413)])
def test_bad_uploads_fail_and_release_capacity(settings, filename, content, status):
    app = create_app(settings, runner=negative_runner, readiness_check=configured)
    with TestClient(app) as client:
        response = client.post("/api/v1/analyze", headers={"X-API-Key": KEY}, files={"file": (filename, content)})
        assert response.status_code == status
        assert app.state.manager.inflight == 0
        assert list((settings.data_dir / "jobs").iterdir()) == []


def test_body_limit_before_multipart_spool(settings):
    with TestClient(create_app(settings, runner=negative_runner, readiness_check=configured)) as client:
        response = client.post("/api/v1/analyze", headers={"X-API-Key": KEY, "Content-Length": str(4 * 1024 * 1024)}, content=b"x")
        assert response.status_code == 413


def test_slow_upload_expires_and_releases_capacity(settings):
    settings = replace(settings, analysis_timeout_sec=0.05)
    app = create_app(settings, runner=negative_runner, readiness_check=configured)
    messages = []

    async def receive():
        await asyncio.sleep(0.5)
        return {"type": "http.request", "body": b"--late--\r\n", "more_body": False}

    async def send(message):
        messages.append(message)

    scope = {"type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1",
             "method": "POST", "scheme": "http", "path": "/api/v1/analyze", "raw_path": b"/api/v1/analyze",
             "query_string": b"", "root_path": "", "server": ("test", 80), "client": ("test", 1),
             "headers": [(b"x-api-key", KEY.encode()), (b"content-type", b"multipart/form-data; boundary=late")]}
    started = time.monotonic()
    asyncio.run(app(scope, receive, send))
    assert next(message["status"] for message in messages if message["type"] == "http.response.start") == 408
    assert time.monotonic() - started < 0.5
    assert app.state.manager.inflight == 0
    assert list((settings.data_dir / "jobs").iterdir()) == []


def test_chunked_oversize_body_rejected_without_content_length(settings):
    app = create_app(settings, runner=negative_runner, readiness_check=configured)
    messages = []
    chunks = iter([b'--chunks\r\nContent-Disposition: form-data; name="file"; filename="x.mov"\r\n\r\n', b"x" * (4 * 1024 * 1024)])

    async def receive():
        return {"type": "http.request", "body": next(chunks, b""), "more_body": True}

    async def send(message):
        messages.append(message)

    scope = {"type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1",
             "method": "POST", "scheme": "http", "path": "/api/v1/analyze", "raw_path": b"/api/v1/analyze",
             "query_string": b"", "root_path": "", "server": ("test", 80), "client": ("test", 1),
             "headers": [(b"x-api-key", KEY.encode()), (b"content-type", b"multipart/form-data; boundary=chunks")]}
    asyncio.run(app(scope, receive, send))
    assert next(message["status"] for message in messages if message["type"] == "http.response.start") == 413
    assert app.state.manager.inflight == 0
    assert list((settings.data_dir / "jobs").iterdir()) == []


def test_unconfigured_model_returns_error_not_negative(settings):
    with TestClient(create_app(settings)) as client:
        response = client.post("/api/v1/analyze", headers={"X-API-Key": KEY}, files={"file": ("x.mov", b"x")})
        assert response.status_code == 503
        assert "verdict" not in response.json()
        caps = client.get("/api/v1/capabilities", headers={"X-API-Key": KEY}).json()
        assert caps["model_configured"] is False


def test_capabilities_exposes_valid_contract_without_ml_paths_or_credentials(settings):
    with TestClient(create_app(settings)) as client:
        assert client.get("/api/v1/capabilities").status_code == 401
        response = client.get("/api/v1/capabilities", headers={"X-API-Key": KEY})
        assert response.status_code == 200
        contract = validate_runtime_contract(response.json()["runtime_contract"])
        assert contract == build_runtime_contract(settings)
        assert contract["complete"] is False
        assert "model_provenance_unavailable" in contract["issues"]
        assert contract["weights_hashes_verified"] is False
        assert KEY not in response.text
        assert str(settings.data_dir.parent) not in response.text


def test_no_active_rules_cannot_produce_clean_verdict(settings):
    with TestClient(create_app(settings, readiness_check=lambda s: {"model_configured": True, "active_rules": 0})) as client:
        response = client.post("/api/v1/analyze", headers={"X-API-Key": KEY}, files={"file": ("x.mov", b"x")})
        assert response.status_code == 503
        assert "verdict" not in response.json()


def test_bounded_capacity(settings):
    release = threading.Event()
    def held_runner(task, settings, stop_event):
        while not release.wait(0.01) and not stop_event.is_set():
            pass
        return negative_runner(task, settings, stop_event)
    with TestClient(create_app(settings, runner=held_runner, readiness_check=configured)) as client:
        first = client.post("/api/v1/analyze", headers={"X-API-Key": KEY}, files={"file": ("a.mov", b"fixture")})
        second = client.post("/api/v1/analyze", headers={"X-API-Key": KEY}, files={"file": ("b.mov", b"fixture")})
        assert first.status_code == 202
        assert second.status_code == 429
        assert second.headers["Retry-After"] == "5"
        release.set()
        assert poll(client, first.json()["job_id"])["status"] == "done"


def test_failure_is_explicit(settings):
    def failed(task, settings, stop):
        return JobStore.failure(task.job_id, "invalid_video", "Cannot decode"), {}
    with TestClient(create_app(settings, runner=failed, readiness_check=configured)) as client:
        accepted = client.post("/api/v1/analyze", headers={"X-API-Key": KEY}, files={"file": ("broken.mov", b"not a movie")}).json()
        result = poll(client, accepted["job_id"])
        assert result["status"] == "failed"
        assert "verdict" not in result


def test_unknown_ids_and_paths_do_not_expose_storage(settings):
    with TestClient(create_app(settings, runner=negative_runner, readiness_check=configured)) as client:
        assert client.get("/api/v1/jobs/" + "0" * 32, headers={"X-API-Key": KEY}).status_code == 404
        assert client.get("/api/v1/jobs/not-a-job", headers={"X-API-Key": KEY}).status_code == 404
        assert client.get("/runtime/jobs.sqlite3").status_code == 404
        accepted = client.post("/api/v1/analyze", headers={"X-API-Key": KEY}, files={"file": ("../../escape.mov", b"x")}).json()
        assert poll(client, accepted["job_id"])["status"] == "done"
        assert not (settings.data_dir.parent / "escape.mov").exists()


def test_restart_recovers_processing_jobs(tmp_path):
    store = JobStore(tmp_path / "db.sqlite3")
    store.create("job", time.time(), time.time() + 200)
    store.recover()
    row = store.get("job")
    assert row["response"]["status"] == "failed"
    assert row["response"]["error"]["code"] == "interrupted"


def test_expiration_removes_old_terminal_jobs(tmp_path):
    store = JobStore(tmp_path / "db.sqlite3")
    store.create("job", time.time(), time.time() + 200)
    store.finish("job", {"job_id": "job", "status": "done", "verdict": "no_violation", "violations": []})
    assert store.delete_expired(time.time() + 1) == ["job"]
    assert store.get("job") is None

