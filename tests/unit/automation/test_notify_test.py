"""Tier 38+: carmel notify-test command."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest  # noqa: TC002

from src.automation.alerts import Alert  # noqa: TC001
from src.automation.notifier import CompositeNotifier, EmailNotifier, WebhookNotifier
from src.automation.notify_test import run_notify_test
from src.config import NotificationConfig, Settings


def _urlopen_ok_cm() -> MagicMock:
    cm = MagicMock()
    cm.__enter__.return_value.read.return_value = b""
    cm.__exit__.return_value = None
    return cm


def test_notify_test_no_notifier_returns_1() -> None:
    s = Settings(
        _yaml_path=None,
        _env_file=None,
        notification=NotificationConfig(enabled=False, webhook_url="", email_enabled=False),
    )
    assert run_notify_test(s) == 1


def test_notify_test_webhook_only_success(capsys: pytest.CaptureFixture[str]) -> None:
    n = WebhookNotifier("https://example.com/hook")
    s = Settings(
        _yaml_path=None,
        _env_file=None,
        notification=NotificationConfig(enabled=True, webhook_url="https://example.com/h"),
    )
    with (
        patch("src.automation.runner._notifier_from_settings", return_value=n),
        patch("src.automation.notifier.urlopen", lambda *_a, **_k: _urlopen_ok_cm()),
    ):
        code = run_notify_test(s)
    assert code == 0
    out = capsys.readouterr().out
    assert "SENT" in out
    assert "webhook (generic):" in out


def test_notify_test_webhook_only_failure(capsys: pytest.CaptureFixture[str]) -> None:
    n = WebhookNotifier("https://example.com/hook")

    def boom(*_a: object, **_k: object) -> MagicMock:
        raise OSError("network")

    s = Settings(
        _yaml_path=None,
        _env_file=None,
        notification=NotificationConfig(enabled=True, webhook_url="https://example.com/h"),
    )
    with (
        patch("src.automation.runner._notifier_from_settings", return_value=n),
        patch("src.automation.notifier.urlopen", boom),
        patch("src.automation.notifier.time.sleep", lambda *_a, **_k: None),
    ):
        code = run_notify_test(s)
    assert code == 1
    out = capsys.readouterr().out
    assert "FAILED" in out


def test_notify_test_composite_reports_each_channel(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    w = WebhookNotifier("http://w.example/hook")
    e = EmailNotifier(
        "localhost",
        25,
        use_tls=False,
        username="",
        password="",
        from_addr="a@b.com",
        to_addrs=["c@d.com"],
    )
    monkeypatch.setattr(w, "send", lambda _a: True)
    monkeypatch.setattr(e, "send", lambda _a: True)
    comp = CompositeNotifier([w, e])
    s = Settings(
        _yaml_path=None,
        _env_file=None,
        notification=NotificationConfig(
            enabled=True,
            webhook_url="https://w.example",
            email_enabled=True,
            smtp_host="localhost",
            email_to=["a@b.com"],
        ),
        smtp_username="u",
        smtp_password="p",
    )
    with patch("src.automation.runner._notifier_from_settings", return_value=comp):
        code = run_notify_test(s)
    assert code == 0
    out = capsys.readouterr().out
    assert "webhook (generic):" in out
    assert "email:" in out
    assert "SENT" in out


def test_notify_test_composite_partial_failure(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    w = WebhookNotifier("http://w.example/hook")
    e = EmailNotifier(
        "localhost",
        25,
        use_tls=False,
        username="",
        password="",
        from_addr="a@b.com",
        to_addrs=["c@d.com"],
    )
    monkeypatch.setattr(w, "send", lambda _a: True)
    monkeypatch.setattr(e, "send", lambda _a: False)
    comp = CompositeNotifier([w, e])
    s = Settings(
        _yaml_path=None,
        _env_file=None,
        notification=NotificationConfig(
            enabled=True,
            webhook_url="https://w.example",
            email_enabled=True,
            smtp_host="localhost",
            email_to=["a@b.com"],
        ),
        smtp_username="u",
        smtp_password="p",
    )
    with patch("src.automation.runner._notifier_from_settings", return_value=comp):
        code = run_notify_test(s)
    assert code == 1
    out = capsys.readouterr().out
    assert "webhook (generic):" in out and "SENT" in out
    assert "email:" in out and "FAILED" in out


def test_notify_test_alert_message_is_clearly_test() -> None:
    captured: list[Alert] = []
    n = WebhookNotifier("https://example.com/hook")

    def capture_send(alert: Alert) -> bool:
        captured.append(alert)
        return True

    n.send = capture_send  # type: ignore[method-assign]
    s = Settings(
        _yaml_path=None,
        _env_file=None,
        notification=NotificationConfig(enabled=True, webhook_url="https://example.com/h"),
    )
    with patch("src.automation.runner._notifier_from_settings", return_value=n):
        run_notify_test(s)
    assert len(captured) == 1
    alert = captured[0]
    assert alert.category == "notify_test"
    assert "notification test" in alert.message.lower()
    assert "Carmel" in alert.message


def test_notify_test_label_includes_webhook_provider(capsys: pytest.CaptureFixture[str]) -> None:
    n = WebhookNotifier("https://discord.com/api/webhooks/123/abc")
    s = Settings(
        _yaml_path=None,
        _env_file=None,
        notification=NotificationConfig(
            enabled=True,
            webhook_url="https://discord.com/api/webhooks/123/abc",
        ),
    )
    with (
        patch("src.automation.runner._notifier_from_settings", return_value=n),
        patch("src.automation.notifier.urlopen", lambda *_a, **_k: _urlopen_ok_cm()),
    ):
        code = run_notify_test(s)
    assert code == 0
    out = capsys.readouterr().out
    assert "webhook (discord):" in out
