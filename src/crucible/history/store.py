"""Local chat-history persistence (SQLite).

Stores chat sessions and their messages so the web app can list past conversations and resume
them. Plain `sqlite3` (stdlib, no new dependency); one shared connection guarded by a lock,
since history writes are small and infrequent. The store is optional — the server runs fine
without it (the UI just won't show past chats).
"""

from __future__ import annotations

import sqlite3
import threading
import time
import uuid
from pathlib import Path

_SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id         TEXT PRIMARY KEY,
    title      TEXT NOT NULL,
    model      TEXT,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS messages (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    role       TEXT NOT NULL,
    content    TEXT NOT NULL,
    created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_messages_session ON messages (session_id);
"""


class HistoryStore:
    def __init__(self, path: str | Path = ":memory:"):
        self._path = str(path)
        if self._path != ":memory:":
            Path(self._path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._db = sqlite3.connect(self._path, check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        with self._db:
            self._db.executescript(_SCHEMA)

    def create_session(self, title: str = "New chat", model: str | None = None) -> dict:
        sid = uuid.uuid4().hex
        now = time.time()
        with self._lock, self._db:
            self._db.execute(
                "INSERT INTO sessions (id, title, model, created_at, updated_at) "
                "VALUES (?,?,?,?,?)",
                (sid, title or "New chat", model, now, now),
            )
        return {
            "id": sid,
            "title": title or "New chat",
            "model": model,
            "created_at": now,
            "updated_at": now,
            "messages_count": 0,
        }

    def list_sessions(self) -> list[dict]:
        with self._lock:
            rows = self._db.execute(
                "SELECT s.*, (SELECT COUNT(*) FROM messages m WHERE m.session_id = s.id) "
                "AS messages_count FROM sessions s ORDER BY updated_at DESC"
            ).fetchall()
        return [
            {
                "id": r["id"],
                "title": r["title"],
                "model": r["model"],
                "created_at": r["created_at"],
                "updated_at": r["updated_at"],
                "messages_count": r["messages_count"],
            }
            for r in rows
        ]

    def get_session(self, sid: str) -> dict | None:
        with self._lock:
            s = self._db.execute("SELECT * FROM sessions WHERE id = ?", (sid,)).fetchone()
            if s is None:
                return None
            msgs = self._db.execute(
                "SELECT role, content, created_at FROM messages WHERE session_id = ? ORDER BY id",
                (sid,),
            ).fetchall()
        return {
            "id": s["id"],
            "title": s["title"],
            "model": s["model"],
            "created_at": s["created_at"],
            "updated_at": s["updated_at"],
            "messages": [
                {"role": m["role"], "content": m["content"], "created_at": m["created_at"]}
                for m in msgs
            ],
        }

    def append_message(self, sid: str, role: str, content: str) -> bool:
        now = time.time()
        with self._lock, self._db:
            if self._db.execute("SELECT 1 FROM sessions WHERE id = ?", (sid,)).fetchone() is None:
                return False
            self._db.execute(
                "INSERT INTO messages (session_id, role, content, created_at) VALUES (?,?,?,?)",
                (sid, role, content, now),
            )
            self._db.execute("UPDATE sessions SET updated_at = ? WHERE id = ?", (now, sid))
        return True

    def rename(self, sid: str, title: str) -> bool:
        with self._lock, self._db:
            n = self._db.execute(
                "UPDATE sessions SET title = ? WHERE id = ?", (title, sid)
            ).rowcount
        return n > 0

    def delete(self, sid: str) -> bool:
        with self._lock, self._db:
            self._db.execute("DELETE FROM messages WHERE session_id = ?", (sid,))
            n = self._db.execute("DELETE FROM sessions WHERE id = ?", (sid,)).rowcount
        return n > 0

    def close(self) -> None:
        self._db.close()
