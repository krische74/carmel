"""Tier 32: Ask Hub chat persistence in SQLite."""

from __future__ import annotations

from src.data.storage.sqlite_store import SQLiteStore


def test_log_chat_and_retrieve(tmp_path) -> None:
    store = SQLiteStore(tmp_path / "chat.sqlite")
    store.log_chat("q1", "a1", account_id="default")
    store.log_chat("q2", "a2", account_id="default")
    rows = store.get_chat_history(limit=50, account_id=None)
    assert len(rows) == 2
    assert rows[0]["question"] == "q1"
    assert rows[1]["answer"] == "a2"


def test_chat_history_limit(tmp_path) -> None:
    store = SQLiteStore(tmp_path / "chat2.sqlite")
    for i in range(10):
        store.log_chat(f"q{i}", f"a{i}")
    rows = store.get_chat_history(limit=3, account_id=None)
    assert len(rows) == 3
    assert rows[-1]["question"] == "q9"


def test_chat_history_per_account(tmp_path) -> None:
    store = SQLiteStore(tmp_path / "chat3.sqlite")
    store.log_chat("x", "ax", account_id="A")
    store.log_chat("y", "ay", account_id="B")
    ra = store.get_chat_history(limit=50, account_id="A")
    rb = store.get_chat_history(limit=50, account_id="B")
    assert len(ra) == 1 and ra[0]["question"] == "x"
    assert len(rb) == 1 and rb[0]["question"] == "y"
