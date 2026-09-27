"""One-shot CLI to validate webhook/email notification wiring."""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from src.automation.alerts import Alert, AlertLevel
from src.automation.notifier import CompositeNotifier, EmailNotifier, Notifier, WebhookNotifier

if TYPE_CHECKING:
    from src.config import Settings

logger = logging.getLogger(__name__)

_TEST_MESSAGE = (
    "Carmel notification test — if you see this, your webhook/email is wired correctly."
)


def _channel_label(notifier: Notifier) -> str:
    if isinstance(notifier, WebhookNotifier):
        return f"webhook ({notifier.provider})"
    if isinstance(notifier, EmailNotifier):
        return "email"
    return type(notifier).__name__.lower()


def run_notify_test(settings: Settings) -> int:
    """Build configured notifiers and send a test alert. Returns exit code (0 = all ok)."""
    from src.automation.runner import _notifier_from_settings

    notifier = _notifier_from_settings(settings)
    if notifier is None:
        logger.error(
            "No notification channels configured. Enable webhook (notification.enabled + "
            "webhook_url) or email (notification.email_enabled + SMTP) in settings / .env.",
        )
        return 1

    alert = Alert(
        timestamp=datetime.now(UTC),
        level=AlertLevel.INFO,
        category="notify_test",
        message=_TEST_MESSAGE,
    )

    print("Sending test alert through configured notifier(s)...", flush=True)

    if isinstance(notifier, CompositeNotifier):
        all_ok = True
        for child in notifier.children:
            label = _channel_label(child)
            try:
                ok = bool(child.send(alert))
            except (OSError, RuntimeError, ValueError, TypeError) as exc:
                logger.warning("%s notifier raised: %s", label, exc)
                ok = False
            status = "SENT" if ok else "FAILED"
            print(f"{label}: {status}", flush=True)
            if not ok:
                all_ok = False
        if all_ok:
            print("\nTest notification delivered successfully.", flush=True)
            return 0
        logger.error("One or more notification channels failed.")
        return 1

    label = _channel_label(notifier)
    try:
        ok = bool(notifier.send(alert))
    except (OSError, RuntimeError, ValueError, TypeError) as exc:
        logger.warning("%s notifier raised: %s", label, exc)
        ok = False
    status = "SENT" if ok else "FAILED"
    print(f"{label}: {status}", flush=True)
    if ok:
        print("\nTest notification delivered successfully.", flush=True)
        return 0
    logger.error("Test notification failed.")
    return 1
