from __future__ import annotations

import json
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any


class JobStore:
    """Small durable job index. Media never becomes a static/public directory."""

    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        with self.connect() as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("""CREATE TABLE IF NOT EXISTS jobs (
                job_id TEXT PRIMARY KEY, status TEXT NOT NULL,
                created_at REAL NOT NULL, deadline REAL NOT NULL,
                updated_at REAL NOT NULL, response_json TEXT,
                details_json TEXT NOT NULL DEFAULT '{}')""")

    @contextmanager
    def connect(self):
        conn = sqlite3.connect(self.path, timeout=5)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA secure_delete=ON")
        try:
            with conn:
                yield conn
        finally:
            conn.close()

    def recover(self) -> None:
        """A restart must never leave an old job processing forever."""
        with self.connect() as conn:
            rows = conn.execute("SELECT job_id FROM jobs WHERE status='processing'").fetchall()
            for row in rows:
                payload = self.failure(row["job_id"], "interrupted", "Обработка прервана перезапуском. Загрузите видео повторно.")
                conn.execute(
                    "UPDATE jobs SET status='failed',response_json=?,updated_at=? WHERE job_id=?",
                    (json.dumps(payload, ensure_ascii=False), time.time(), row["job_id"]),
                )

    def create(self, job_id: str, created_at: float, deadline: float) -> None:
        with self.connect() as conn:
            conn.execute(
                "INSERT INTO jobs(job_id,status,created_at,deadline,updated_at) VALUES(?,?,?,?,?)",
                (job_id, "processing", created_at, deadline, created_at),
            )

    @staticmethod
    def failure(job_id: str, code: str, message: str) -> dict[str, Any]:
        return {"job_id": job_id, "status": "failed", "error": {"code": code, "message": message}}

    def finish(self, job_id: str, response: dict[str, Any], details: dict[str, Any] | None = None) -> bool:
        if response.get("status") not in {"done", "failed"}:
            raise ValueError("Invalid terminal status")
        with self.connect() as conn:
            cursor = conn.execute(
                "UPDATE jobs SET status=?,response_json=?,details_json=?,updated_at=? WHERE job_id=? AND status='processing'",
                (response["status"], json.dumps(response, ensure_ascii=False, allow_nan=False),
                 json.dumps(details or {}, ensure_ascii=False, allow_nan=False), time.time(), job_id),
            )
            return cursor.rowcount == 1

    def get(self, job_id: str) -> dict[str, Any] | None:
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM jobs WHERE job_id=?", (job_id,)).fetchone()
        if not row:
            return None
        result = dict(row)
        result["response"] = json.loads(result.pop("response_json")) if row["response_json"] else {"job_id": job_id, "status": "processing"}
        result["details"] = json.loads(result.pop("details_json"))
        return result

    def delete_expired(self, before: float) -> list[str]:
        with self.connect() as conn:
            rows = conn.execute("SELECT job_id FROM jobs WHERE status!='processing' AND updated_at<?", (before,)).fetchall()
            ids = [row["job_id"] for row in rows]
            conn.executemany("DELETE FROM jobs WHERE job_id=?", [(job_id,) for job_id in ids])
        return ids

