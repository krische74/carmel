"""Pluggable notification channels for operational alerts."""

from __future__ import annotations

import json
import logging
import smtplib
import time
from abc import ABC, abstractmethod
from collections.abc import Callable  # noqa: TC003
from email.mime.text import MIMEText
from typing import Any
from urllib.error import URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from src.automation.alerts import Alert  # noqa: TC001

logger = logging.getLogger(__name__)

# Identifies Carmel to webhook providers. Cloudflare and similar bot-protection
# layers reject the urllib default ("Python-urllib/X.Y") with HTTP 403 + error 1010,
# so we send an explicit, descriptive UA on every webhook POST.
_USER_AGENT = "Carmel-Webhook/1.0"


def _detect_webhook_provider(url: str) -> str:
    """Return one of: 'discord', 'slack', 'telegram', 'generic'.

    Detection is based on the URL hostname; never raises. Unknown hosts → 'generic'.
    """
    if not (url or "").strip():
        return "generic"
    host = (urlparse(url).hostname or "").lower()
    if "discord.com" in host or "discordapp.com" in host:
        return "discord"
    if "hooks.slack.com" in host:
        return "slack"
    if "api.telegram.org" in host:
        return "telegram"
    return "generic"


def _build_discord_payload(alert: Alert) -> dict[str, Any]:
    """Discord webhook: requires 'content' or 'embeds'. Use embed for rich formatting."""
    color = {
        "info": 0x5865F2,
        "warning": 0xFEE75C,
        "critical": 0xED4245,
    }.get(alert.level.value, 0x99AAB5)
    return {
        "embeds": [
            {
                "title": f"[{alert.level.value.upper()}] {alert.category}",
                "description": alert.message,
                "color": color,
                "timestamp": alert.timestamp.isoformat(),
                "footer": {"text": "Carmel"},
            },
        ],
    }


def _build_slack_payload(alert: Alert) -> dict[str, Any]:
    """Slack incoming webhook: requires 'text', supports 'blocks' for rich formatting."""
    emoji = {
        "info": ":information_source:",
        "warning": ":warning:",
        "critical": ":rotating_light:",
    }.get(alert.level.value, ":bell:")
    return {
        "text": f"{emoji} *[{alert.level.value.upper()}] {alert.category}*",
        "blocks": [
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": (
                        f"{emoji} *[{alert.level.value.upper()}] {alert.category}*\n"
                        f"{alert.message}\n"
                        f"_{alert.timestamp.isoformat()}_"
                    ),
                },
            },
        ],
    }


def _build_telegram_payload(alert: Alert, chat_id: str) -> dict[str, Any]:
    """Telegram bot sendMessage: requires 'text' and 'chat_id' for delivery."""
    payload: dict[str, Any] = {
        "text": (
            f"*\\[{alert.level.value.upper()}\\] {alert.category}*\n"
            f"{alert.message}\n"
            f"_{alert.timestamp.isoformat()}_"
        ),
        "parse_mode": "MarkdownV2",
    }
    if chat_id.strip():
        payload["chat_id"] = chat_id.strip()
    return payload


def _build_generic_payload(alert: Alert) -> dict[str, Any]:
    """Generic JSON shape — backwards compatibility with custom endpoints."""
    return {
        "timestamp": alert.timestamp.isoformat(),
        "level": alert.level.value,
        "category": alert.category,
        "message": alert.message,
    }


class Notifier(ABC):
    """Push notification channel for operational alerts."""

    @abstractmethod
    def send(self, alert: Alert) -> bool:
        """Attempt to deliver one alert. Return True on success, False on failure."""


class LogOnlyNotifier(Notifier):
    """Logs the alert at INFO level (no external delivery)."""

    def send(self, alert: Alert) -> bool:
        logger.info("Alert [%s] %s: %s", alert.level.value, alert.category, alert.message)
        return True


class WebhookNotifier(Notifier):
    """POST alert as JSON to a configurable URL (Telegram, Discord, Slack, generic)."""

    def __init__(
        self,
        url: str,
        *,
        telegram_chat_id: str = "",
        timeout_seconds: int = 10,
        max_retries: int = 2,
        backoff_seconds: float = 1.0,
    ) -> None:
        self._url = url.strip()
        self._provider = _detect_webhook_provider(self._url)
        self._telegram_chat_id = (telegram_chat_id or "").strip()
        self._timeout = max(1, int(timeout_seconds))
        self._max_retries = max(0, int(max_retries))
        self._backoff_seconds = float(backoff_seconds)
        if self._provider == "telegram" and not self._telegram_chat_id:
            logger.warning(
                "Telegram webhook URL configured but notification.telegram_chat_id is empty; "
                "messages will fail. Set NOTIFICATION__TELEGRAM_CHAT_ID in .env.",
            )

    @property
    def provider(self) -> str:
        """Detected provider name (discord, slack, telegram, generic)."""
        return self._provider

    def send(self, alert: Alert) -> bool:
        builders: dict[str, Callable[[], dict[str, Any]]] = {
            "discord": lambda: _build_discord_payload(alert),
            "slack": lambda: _build_slack_payload(alert),
            "telegram": lambda: _build_telegram_payload(alert, self._telegram_chat_id),
            "generic": lambda: _build_generic_payload(alert),
        }
        payload = builders[self._provider]()
        data = json.dumps(payload).encode("utf-8")
        req = Request(
            self._url,
            data=data,
            method="POST",
            headers={
                "Content-Type": "application/json",
                "User-Agent": _USER_AGENT,
            },
        )
        for attempt in range(self._max_retries + 1):
            try:
                with urlopen(req, timeout=self._timeout) as resp:
                    _ = resp.read()
            except (URLError, OSError) as exc:
                if attempt == self._max_retries:
                    logger.warning(
                        "Webhook notification failed after %d attempts: %s",
                        self._max_retries + 1,
                        exc,
                    )
                    return False
                time.sleep(self._backoff_seconds * (2**attempt))
            else:
                return True
        return False


class EmailNotifier(Notifier):
    """Send alerts via SMTP (``smtplib``)."""

    def __init__(
        self,
        smtp_host: str,
        smtp_port: int,
        *,
        use_tls: bool,
        username: str,
        password: str,
        from_addr: str,
        to_addrs: list[str],
        timeout_seconds: int = 10,
    ) -> None:
        self._host = smtp_host.strip()
        self._port = max(1, int(smtp_port))
        self._use_tls = bool(use_tls)
        self._username = username
        self._password = password
        self._from = from_addr.strip()
        self._to = [a.strip() for a in to_addrs if str(a).strip()]
        self._timeout = max(1, int(timeout_seconds))

    def send(self, alert: Alert) -> bool:
        if not self._to or not self._host:
            logger.warning("EmailNotifier missing recipients or host; skipping send.")
            return False
        subject = f"[Carmel] [{alert.level.value.upper()}] {alert.category}"
        body = f"{alert.message}\n\nTimestamp (UTC): {alert.timestamp.isoformat()}"
        msg = MIMEText(body, "plain", "utf-8")
        msg["Subject"] = subject
        msg["From"] = self._from
        msg["To"] = ", ".join(self._to)
        try:
            with smtplib.SMTP(self._host, self._port, timeout=self._timeout) as smtp:
                if self._use_tls:
                    smtp.starttls()
                if self._username and self._password:
                    smtp.login(self._username, self._password)
                smtp.sendmail(self._from, self._to, msg.as_string())
        except (smtplib.SMTPException, OSError) as exc:
            logger.warning("Email notification failed: %s", exc)
            return False
        return True


class CompositeNotifier(Notifier):
    """Fan-out to multiple notifiers; succeeds if any child succeeds."""

    def __init__(self, notifiers: list[Notifier]) -> None:
        self._children = list(notifiers)

    @property
    def children(self) -> list[Notifier]:
        """Return child notifiers (copy; read-only access for diagnostics)."""
        return list(self._children)

    def send(self, alert: Alert) -> bool:
        ok = False
        for n in self._children:
            if n.send(alert):
                ok = True
        return ok
