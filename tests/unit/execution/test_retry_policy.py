"""Broker retry policy helpers."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from src.execution.retry_policy import backoff_seconds, should_retry_exception


def test_should_retry_connection_error() -> None:
    assert should_retry_exception(ConnectionError("x")) is True


def test_should_retry_os_and_timeout() -> None:
    assert should_retry_exception(OSError(5, "io")) is True
    assert should_retry_exception(TimeoutError()) is True


def test_should_not_retry_value_or_runtime() -> None:
    assert should_retry_exception(ValueError("bad")) is False
    assert should_retry_exception(RuntimeError("bad")) is False


def test_backoff_exponential_without_jitter() -> None:
    assert backoff_seconds(0, 0.5, use_jitter=False) == pytest.approx(0.5)
    assert backoff_seconds(1, 0.5, use_jitter=False) == pytest.approx(1.0)


def test_backoff_with_jitter_in_range() -> None:
    with patch("src.execution.retry_policy.random.random", return_value=0.25):
        # 0.5 * 2^1 * (0.5 + 0.25) = 0.75
        assert backoff_seconds(1, 0.5, use_jitter=True) == pytest.approx(0.75)
