"""Tests for WebhookNotifier, EmailNotifier, and CompositeNotifier."""

from __future__ import annotations

import json
import smtplib
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock
from urllib.error import URLError

from src.automation.alerts import Alert, AlertLevel
from src.automation.notifier import (
    _USER_AGENT,
    CompositeNotifier,
    EmailNotifier,
    LogOnlyNotifier,
    WebhookNotifier,
    _build_discord_payload,
    _build_generic_payload,
    _build_slack_payload,
    _build_telegram_payload,
    _detect_webhook_provider,
)

if TYPE_CHECKING:
    import pytest


def _urlopen_ok_cm() -> MagicMock:
    cm = MagicMock()
    cm.__enter__.return_value.read.return_value = b""
    cm.__exit__.return_value = None
    return cm


def test_email_notifier_sends_via_smtp(monkeypatch: pytest.MonkeyPatch) -> None:
    sent: list[tuple[str, list[str], str]] = []

    class FakeSMTP:
        def __init__(self, *_a: object, **_k: object) -> None:
            pass

        def __enter__(self) -> FakeSMTP:
            return self

        def __exit__(self, *_a: object) -> None:
            return None

        def starttls(self) -> None:
            pass

        def login(self, *_a: object) -> None:
            pass

        def sendmail(self, from_addr: str, to_addrs: list[str], msg: str) -> None:
            sent.append((from_addr, to_addrs, msg))

    monkeypatch.setattr(smtplib, "SMTP", FakeSMTP)
    n = EmailNotifier(
        "localhost",
        25,
        use_tls=True,
        username="u",
        password="p",
        from_addr="a@example.com",
        to_addrs=["b@example.com"],
    )
    alert = Alert(
        timestamp=datetime(2026, 1, 1, 12, 0, tzinfo=UTC),
        level=AlertLevel.INFO,
        category="test",
        message="hello",
    )
    assert n.send(alert) is True
    assert len(sent) == 1
    assert sent[0][0] == "a@example.com"


def test_email_notifier_smtp_exception_returns_false(monkeypatch: pytest.MonkeyPatch) -> None:
    class BoomSMTP:
        def __init__(self, *_a: object, **_k: object) -> None:
            pass

        def __enter__(self) -> BoomSMTP:
            return self

        def __exit__(self, *_a: object) -> None:
            return None

        def starttls(self) -> None:
            raise smtplib.SMTPException("boom")

    monkeypatch.setattr(smtplib, "SMTP", BoomSMTP)
    n = EmailNotifier(
        "localhost",
        25,
        use_tls=True,
        username="",
        password="",
        from_addr="a@example.com",
        to_addrs=["b@example.com"],
    )
    alert = Alert(
        timestamp=datetime(2026, 1, 1, 12, 0, tzinfo=UTC),
        level=AlertLevel.INFO,
        category="test",
        message="hello",
    )
    assert n.send(alert) is False


def test_email_notifier_no_recipients_returns_false() -> None:
    n = EmailNotifier(
        "localhost",
        25,
        use_tls=False,
        username="",
        password="",
        from_addr="a@example.com",
        to_addrs=[],
    )
    alert = Alert(
        timestamp=datetime(2026, 1, 1, 12, 0, tzinfo=UTC),
        level=AlertLevel.INFO,
        category="test",
        message="hello",
    )
    assert n.send(alert) is False


def test_composite_notifier_succeeds_if_any_child_succeeds() -> None:
    failing = MagicMock()
    failing.send.return_value = False
    ok = MagicMock()
    ok.send.return_value = True
    c = CompositeNotifier([failing, ok])
    alert = Alert(
        timestamp=datetime(2026, 1, 1, 12, 0, tzinfo=UTC),
        level=AlertLevel.INFO,
        category="test",
        message="hello",
    )
    assert c.send(alert) is True
    failing.send.assert_called_once_with(alert)
    ok.send.assert_called_once_with(alert)


def test_composite_children_property_returns_list() -> None:
    w = WebhookNotifier("http://example.com/hook")
    e = EmailNotifier(
        "localhost",
        25,
        use_tls=False,
        username="",
        password="",
        from_addr="a@b.com",
        to_addrs=["c@d.com"],
    )
    c = CompositeNotifier([w, e])
    kids = c.children
    assert len(kids) == 2
    assert kids[0] is w
    assert kids[1] is e


def test_composite_children_is_a_copy() -> None:
    w = WebhookNotifier("http://example.com/hook")
    c = CompositeNotifier([w])
    kids = c.children
    kids.append(WebhookNotifier("http://other.example/hook"))
    assert len(c.children) == 1


def test_composite_notifier_fails_if_all_fail() -> None:
    n1 = MagicMock()
    n1.send.return_value = False
    n2 = MagicMock()
    n2.send.return_value = False
    c = CompositeNotifier([n1, n2])
    alert = Alert(
        timestamp=datetime(2026, 1, 1, 12, 0, tzinfo=UTC),
        level=AlertLevel.INFO,
        category="test",
        message="hello",
    )
    assert c.send(alert) is False


def test_log_only_notifier_always_true() -> None:
    n = LogOnlyNotifier()
    alert = Alert(
        timestamp=datetime(2026, 1, 1, 12, 0, tzinfo=UTC),
        level=AlertLevel.WARNING,
        category="x",
        message="y",
    )
    assert n.send(alert) is True


def test_webhook_succeeds_on_first_attempt(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[int] = []

    def mock_urlopen(*_a: object, **_k: object) -> MagicMock:
        calls.append(1)
        return _urlopen_ok_cm()

    monkeypatch.setattr("src.automation.notifier.urlopen", mock_urlopen)
    n = WebhookNotifier("http://example.com/hook")
    alert = Alert(
        timestamp=datetime(2026, 1, 1, 12, 0, tzinfo=UTC),
        level=AlertLevel.INFO,
        category="c",
        message="m",
    )
    assert n.send(alert) is True
    assert len(calls) == 1


def test_webhook_retries_then_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    attempt = [0]

    def mock_urlopen(*_a: object, **_k: object) -> MagicMock:
        attempt[0] += 1
        if attempt[0] == 1:
            raise URLError("transient")
        return _urlopen_ok_cm()

    monkeypatch.setattr("src.automation.notifier.urlopen", mock_urlopen)
    monkeypatch.setattr("src.automation.notifier.time.sleep", lambda *_a, **_k: None)
    n = WebhookNotifier("http://example.com/hook", max_retries=2, backoff_seconds=0.01)
    alert = Alert(
        timestamp=datetime(2026, 1, 1, 12, 0, tzinfo=UTC),
        level=AlertLevel.INFO,
        category="c",
        message="m",
    )
    assert n.send(alert) is True
    assert attempt[0] == 2


def test_detect_provider_discord() -> None:
    u = "https://discord.com/api/webhooks/123/abc"
    assert _detect_webhook_provider(u) == "discord"


def test_detect_provider_discord_legacy_host() -> None:
    u = "https://discordapp.com/api/webhooks/123/abc"
    assert _detect_webhook_provider(u) == "discord"


def test_detect_provider_slack() -> None:
    u = "https://hooks.slack.com/services/T1/B2/abc"
    assert _detect_webhook_provider(u) == "slack"


def test_detect_provider_telegram() -> None:
    u = "https://api.telegram.org/bot12345:abc/sendMessage"
    assert _detect_webhook_provider(u) == "telegram"


def test_detect_provider_generic() -> None:
    u = "https://my-webhook.example.com/notify"
    assert _detect_webhook_provider(u) == "generic"


def test_detect_provider_malformed_url() -> None:
    assert _detect_webhook_provider("not-a-url") == "generic"


def test_detect_provider_empty_url() -> None:
    assert _detect_webhook_provider("") == "generic"


def test_discord_payload_uses_embeds_with_color() -> None:
    alert = Alert(
        timestamp=datetime(2026, 1, 1, 12, 0, tzinfo=UTC),
        level=AlertLevel.INFO,
        category="ingest",
        message="hello",
    )
    p = _build_discord_payload(alert)
    assert "embeds" in p
    assert p["embeds"][0]["color"] == 0x5865F2
    assert "ingest" in p["embeds"][0]["title"]


def test_discord_payload_critical_uses_red() -> None:
    alert = Alert(
        timestamp=datetime(2026, 1, 1, 12, 0, tzinfo=UTC),
        level=AlertLevel.CRITICAL,
        category="x",
        message="y",
    )
    p = _build_discord_payload(alert)
    assert p["embeds"][0]["color"] == 0xED4245


def test_slack_payload_has_text_and_blocks() -> None:
    alert = Alert(
        timestamp=datetime(2026, 1, 1, 12, 0, tzinfo=UTC),
        level=AlertLevel.INFO,
        category="c",
        message="m",
    )
    p = _build_slack_payload(alert)
    assert "text" in p and "blocks" in p


def test_slack_payload_warning_uses_warning_emoji() -> None:
    alert = Alert(
        timestamp=datetime(2026, 1, 1, 12, 0, tzinfo=UTC),
        level=AlertLevel.WARNING,
        category="w",
        message="msg",
    )
    p = _build_slack_payload(alert)
    assert ":warning:" in p["text"]
    assert ":warning:" in p["blocks"][0]["text"]["text"]


def test_telegram_payload_has_text_and_parse_mode() -> None:
    alert = Alert(
        timestamp=datetime(2026, 1, 1, 12, 0, tzinfo=UTC),
        level=AlertLevel.INFO,
        category="c",
        message="m",
    )
    p = _build_telegram_payload(alert, "")
    assert "text" in p and p["parse_mode"] == "MarkdownV2"
    assert "chat_id" not in p


def test_telegram_payload_includes_chat_id() -> None:
    alert = Alert(
        timestamp=datetime(2026, 1, 1, 12, 0, tzinfo=UTC),
        level=AlertLevel.INFO,
        category="c",
        message="m",
    )
    p = _build_telegram_payload(alert, "987654")
    assert p["chat_id"] == "987654"


def test_telegram_payload_omits_chat_id_when_empty() -> None:
    alert = Alert(
        timestamp=datetime(2026, 1, 1, 12, 0, tzinfo=UTC),
        level=AlertLevel.INFO,
        category="c",
        message="m",
    )
    p = _build_telegram_payload(alert, "")
    assert "chat_id" not in p


def test_generic_payload_preserves_all_fields() -> None:
    alert = Alert(
        timestamp=datetime(2026, 1, 1, 12, 0, tzinfo=UTC),
        level=AlertLevel.INFO,
        category="c",
        message="m",
    )
    p = _build_generic_payload(alert)
    assert p["timestamp"] == alert.timestamp.isoformat()
    assert p["level"] == "info"
    assert p["category"] == "c"
    assert p["message"] == "m"


def test_webhook_notifier_detects_discord_provider() -> None:
    n = WebhookNotifier("https://discord.com/api/webhooks/1/token")
    assert n.provider == "discord"


def test_webhook_notifier_sends_discord_format(monkeypatch: pytest.MonkeyPatch) -> None:
    posted: list[dict[str, object]] = []

    def capture_urlopen(req: object, *_a: object, **_k: object) -> MagicMock:
        data = getattr(req, "data", None)
        assert data is not None
        posted.append(json.loads(data.decode("utf-8")))
        return _urlopen_ok_cm()

    monkeypatch.setattr("src.automation.notifier.urlopen", capture_urlopen)
    n = WebhookNotifier("https://discord.com/api/webhooks/1/token")
    alert = Alert(
        timestamp=datetime(2026, 1, 1, 12, 0, tzinfo=UTC),
        level=AlertLevel.INFO,
        category="c",
        message="m",
    )
    assert n.send(alert) is True
    assert posted
    assert "embeds" in posted[0]
    assert "level" not in posted[0]


def test_webhook_notifier_sends_slack_format(monkeypatch: pytest.MonkeyPatch) -> None:
    posted: list[dict[str, object]] = []

    def capture_urlopen(req: object, *_a: object, **_k: object) -> MagicMock:
        data = getattr(req, "data", None)
        assert data is not None
        posted.append(json.loads(data.decode("utf-8")))
        return _urlopen_ok_cm()

    monkeypatch.setattr("src.automation.notifier.urlopen", capture_urlopen)
    n = WebhookNotifier("https://hooks.slack.com/services/T/B/x")
    alert = Alert(
        timestamp=datetime(2026, 1, 1, 12, 0, tzinfo=UTC),
        level=AlertLevel.INFO,
        category="c",
        message="m",
    )
    assert n.send(alert) is True
    assert "blocks" in posted[0] and "text" in posted[0]


def test_webhook_notifier_sends_generic_format_for_unknown_url(monkeypatch: pytest.MonkeyPatch) -> None:
    posted: list[dict[str, object]] = []

    def capture_urlopen(req: object, *_a: object, **_k: object) -> MagicMock:
        data = getattr(req, "data", None)
        assert data is not None
        posted.append(json.loads(data.decode("utf-8")))
        return _urlopen_ok_cm()

    monkeypatch.setattr("src.automation.notifier.urlopen", capture_urlopen)
    n = WebhookNotifier("https://example.com/webhook")
    alert = Alert(
        timestamp=datetime(2026, 1, 1, 12, 0, tzinfo=UTC),
        level=AlertLevel.INFO,
        category="c",
        message="m",
    )
    assert n.send(alert) is True
    assert posted[0].get("level") == "info"


def test_webhook_notifier_sets_user_agent_header(monkeypatch: pytest.MonkeyPatch) -> None:
    """WebhookNotifier must send an explicit User-Agent so Cloudflare-protected
    endpoints (Discord/Slack/Telegram) don't reject the request as a bot."""
    from src.automation import notifier as notifier_mod

    captured_request: dict[str, Any] = {}

    class _FakeResp:
        def __enter__(self) -> _FakeResp:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def read(self) -> bytes:
            return b""

    def _fake_urlopen(req: Any, *_args: object, **_kwargs: object) -> _FakeResp:
        captured_request["url"] = req.full_url
        captured_request["headers"] = dict(req.headers)
        return _FakeResp()

    monkeypatch.setattr(notifier_mod, "urlopen", _fake_urlopen)

    n = WebhookNotifier("https://example.com/webhook")
    alert = Alert(
        timestamp=datetime.now(UTC),
        level=AlertLevel.INFO,
        category="test",
        message="test",
    )
    assert n.send(alert) is True

    headers = captured_request["headers"]
    ua = headers.get("User-agent") or headers.get("User-Agent")
    assert ua is not None, "WebhookNotifier must set User-Agent header"
    assert "Carmel" in ua, f"User-Agent should identify Carmel: {ua!r}"
    assert "Python-urllib" not in ua, "Default urllib UA must not be used"


def test_webhook_user_agent_constant_is_descriptive() -> None:
    """The User-Agent constant should follow the standard format and identify the project."""
    assert "Carmel" in _USER_AGENT
    assert "/" in _USER_AGENT  # version separator
    assert _USER_AGENT.strip() == _USER_AGENT


def test_webhook_notifier_warns_on_telegram_without_chat_id(
    caplog: pytest.LogCaptureFixture,
) -> None:
    import logging

    with caplog.at_level(logging.WARNING):
        WebhookNotifier("https://api.telegram.org/bot123/sendMessage")
    assert "telegram_chat_id" in caplog.text.lower() or "TELEGRAM_CHAT_ID" in caplog.text


def test_webhook_fails_after_exhausting_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    def mock_urlopen(*_a: object, **_k: object) -> MagicMock:
        raise URLError("always")

    monkeypatch.setattr("src.automation.notifier.urlopen", mock_urlopen)
    monkeypatch.setattr("src.automation.notifier.time.sleep", lambda *_a, **_k: None)
    n = WebhookNotifier("http://example.com/hook", max_retries=1, backoff_seconds=0.01)
    alert = Alert(
        timestamp=datetime(2026, 1, 1, 12, 0, tzinfo=UTC),
        level=AlertLevel.INFO,
        category="c",
        message="m",
    )
    assert n.send(alert) is False
