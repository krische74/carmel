"""Simple heartbeat file for external watchdogs (process liveness)."""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)


def write_heartbeat(path: Path) -> None:
    """Write ISO-8601 UTC timestamp (single line) to ``path``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).isoformat()
    path.write_text(stamp + "\n", encoding="utf-8")
    logger.debug("heartbeat written to %s", path)


def read_heartbeat(path: Path) -> datetime | None:
    """Return the parsed heartbeat time, or ``None`` if missing or invalid."""
    if not path.exists():
        return None
    raw = path.read_text(encoding="utf-8").strip()
    try:
        return datetime.fromisoformat(raw)
    except ValueError:
        logger.warning("invalid heartbeat content in %s", path)
        return None


def clear_heartbeat(path: Path) -> None:
    """Remove the heartbeat file if present. No-op when already absent.

    Called from the daemon's graceful-shutdown path so the next startup
    sees no prior heartbeat and skips the duplicate-daemon observation.
    """
    try:
        path.unlink(missing_ok=True)
    except OSError as exc:
        logger.warning("failed to clear heartbeat at %s: %s", path, exc)
