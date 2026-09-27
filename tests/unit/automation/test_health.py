"""Tests for process heartbeat file."""

from datetime import UTC, datetime
from pathlib import Path

from src.automation.health import clear_heartbeat, read_heartbeat, write_heartbeat


def test_write_then_read_heartbeat_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "hb.txt"
    before = datetime.now(UTC)
    write_heartbeat(path)
    after = datetime.now(UTC)
    got = read_heartbeat(path)
    assert got is not None
    assert before <= got <= after


def test_read_heartbeat_missing_file_returns_none(tmp_path: Path) -> None:
    assert read_heartbeat(tmp_path / "nope.txt") is None


def test_read_heartbeat_corrupt_content_returns_none(tmp_path: Path) -> None:
    path = tmp_path / "bad.txt"
    path.write_text("not-a-valid-timestamp\n", encoding="utf-8")
    assert read_heartbeat(path) is None


def test_clear_heartbeat_removes_existing_file(tmp_path: Path) -> None:
    path = tmp_path / "hb.txt"
    write_heartbeat(path)
    assert path.exists()
    clear_heartbeat(path)
    assert not path.exists()


def test_clear_heartbeat_is_noop_when_missing(tmp_path: Path) -> None:
    # Must not raise on absent file -- finally-block cleanup runs even when
    # the daemon never managed to write a heartbeat (e.g. immediate abort).
    clear_heartbeat(tmp_path / "never-existed.txt")
