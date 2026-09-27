"""Dashboard ``require_auth`` behavior (Streamlit mocked)."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest  # noqa: TC002

from src.config import DashboardConfig, Settings
from src.dashboard.auth import require_auth


def _settings(**dash: object) -> Settings:
    return Settings(
        _yaml_path=None,
        _env_file=None,
        dashboard=DashboardConfig(**dash),
    )


def test_require_auth_disabled_returns_true(monkeypatch: pytest.MonkeyPatch) -> None:
    mock_st = MagicMock()
    mock_st.session_state = {}
    monkeypatch.setattr("src.dashboard.auth.st", mock_st)
    s = _settings(auth_enabled=False, auth_password="secret")
    assert require_auth(s) is True
    mock_st.text_input.assert_not_called()


def test_require_auth_empty_password_returns_true(monkeypatch: pytest.MonkeyPatch) -> None:
    mock_st = MagicMock()
    mock_st.session_state = {}
    monkeypatch.setattr("src.dashboard.auth.st", mock_st)
    s = _settings(auth_enabled=True, auth_password="")
    assert require_auth(s) is True


def test_require_auth_uses_dashboard_password_from_settings_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``Settings.dashboard_password`` (``DASHBOARD_PASSWORD``) overrides YAML."""
    mock_st = MagicMock()
    mock_st.session_state = {}
    mock_st.text_input.return_value = "env_secret"
    mock_st.button.return_value = True
    monkeypatch.setattr("src.dashboard.auth.st", mock_st)
    s = Settings(
        _yaml_path=None,
        _env_file=None,
        dashboard=DashboardConfig(auth_enabled=True, auth_password="yaml_wrong"),
        dashboard_password="env_secret",
    )
    assert require_auth(s) is True
    assert mock_st.session_state.get("dashboard_authenticated") is True


def test_require_auth_authenticated_session_returns_true(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mock_st = MagicMock()
    mock_st.session_state = {"dashboard_authenticated": True}
    monkeypatch.setattr("src.dashboard.auth.st", mock_st)
    s = _settings(auth_enabled=True, auth_password="x")
    assert require_auth(s) is True


def test_require_auth_not_authenticated_returns_false(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mock_st = MagicMock()
    mock_st.session_state = {}
    mock_st.text_input.return_value = ""
    mock_st.button.return_value = False
    monkeypatch.setattr("src.dashboard.auth.st", mock_st)
    s = _settings(auth_enabled=True, auth_password="secret")
    assert require_auth(s) is False
    mock_st.text_input.assert_called()
    mock_st.subheader.assert_called()


def test_require_auth_wrong_password_shows_error(monkeypatch: pytest.MonkeyPatch) -> None:
    mock_st = MagicMock()
    mock_st.session_state = {}
    mock_st.text_input.return_value = "nope"
    mock_st.button.return_value = True
    monkeypatch.setattr("src.dashboard.auth.st", mock_st)
    s = _settings(auth_enabled=True, auth_password="secret")
    assert require_auth(s) is False
    mock_st.error.assert_called_once()
    assert mock_st.session_state.get("dashboard_authenticated") is not True
