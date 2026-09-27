"""Retry helpers for transient broker submission failures."""

from __future__ import annotations

import random


def should_retry_exception(exc: Exception) -> bool:
    """Return True when the error is plausibly transient (network / timeout)."""
    return isinstance(exc, (ConnectionError, OSError, TimeoutError))


def backoff_seconds(attempt_index: int, base_seconds: float, *, use_jitter: bool) -> float:
    """Exponential backoff ``base * 2^attempt`` with optional multiplicative jitter in (0.5, 1.5)."""
    delay = float(base_seconds) * (2**int(attempt_index))
    if use_jitter:
        delay *= 0.5 + random.random()
    return delay
