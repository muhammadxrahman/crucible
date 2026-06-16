"""Chat-history store: sessions and messages persist and round-trip (in-memory SQLite)."""

from __future__ import annotations

from crucible.history import HistoryStore


def test_create_list_and_get_session() -> None:
    h = HistoryStore(":memory:")
    s = h.create_session(title="First chat", model="primary")
    assert s["title"] == "First chat" and s["messages_count"] == 0

    h.append_message(s["id"], "user", "hello")
    h.append_message(s["id"], "assistant", "hi there")

    listed = h.list_sessions()
    assert len(listed) == 1 and listed[0]["messages_count"] == 2

    full = h.get_session(s["id"])
    assert [m["content"] for m in full["messages"]] == ["hello", "hi there"]
    assert [m["role"] for m in full["messages"]] == ["user", "assistant"]


def test_list_orders_by_recent_activity() -> None:
    h = HistoryStore(":memory:")
    a = h.create_session(title="A")
    h.create_session(title="B")
    h.append_message(a["id"], "user", "touch A so it is most recent")
    assert [s["title"] for s in h.list_sessions()] == ["A", "B"]


def test_rename_and_delete() -> None:
    h = HistoryStore(":memory:")
    s = h.create_session()
    assert h.rename(s["id"], "Renamed") is True
    assert h.list_sessions()[0]["title"] == "Renamed"
    assert h.delete(s["id"]) is True
    assert h.list_sessions() == []


def test_persists_across_reopen(tmp_path) -> None:
    db = tmp_path / "history.db"
    h = HistoryStore(db)
    s = h.create_session(title="kept")
    h.append_message(s["id"], "user", "remember me")
    h.close()

    reopened = HistoryStore(db)
    full = reopened.get_session(s["id"])
    assert full["title"] == "kept"
    assert full["messages"][0]["content"] == "remember me"


def test_unknown_session_returns_none_and_false() -> None:
    h = HistoryStore(":memory:")
    assert h.get_session("nope") is None
    assert h.append_message("nope", "user", "x") is False
    assert h.rename("nope", "y") is False
    assert h.delete("nope") is False
