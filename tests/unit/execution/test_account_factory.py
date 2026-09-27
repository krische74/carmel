"""Tests for ``build_brokers`` and ``resolve_sqlite_account_id``."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest

from src.config import AccountConfig, BrokerConfig, Settings
from src.execution.account_factory import build_brokers, resolve_sqlite_account_id

if TYPE_CHECKING:
    from pathlib import Path


def _minimal_yaml(path: Path) -> None:
    path.write_text(
        """
app: {name: "X", version: "0.1.0", environment: "development"}
data: {universe: ["SPY"]}
risk: {max_position_pct: 0.25}
""",
        encoding="utf-8",
    )


def test_resolve_sqlite_account_id_non_string_mock_returns_default() -> None:
    assert resolve_sqlite_account_id(MagicMock()) == "default"


def test_resolve_sqlite_account_id_strips_string() -> None:
    b = MagicMock()
    b.get_account_id.return_value = "  IRA  "
    assert resolve_sqlite_account_id(b) == "IRA"


def test_resolve_sqlite_account_id_empty_string_returns_default() -> None:
    b = MagicMock()
    b.get_account_id.return_value = ""
    assert resolve_sqlite_account_id(b) == "default"


def test_resolve_sqlite_account_id_whitespace_only_returns_default() -> None:
    b = MagicMock()
    b.get_account_id.return_value = "   \t  "
    assert resolve_sqlite_account_id(b) == "default"


def test_resolve_sqlite_account_id_raises_returns_default() -> None:
    b = MagicMock()
    b.get_account_id.side_effect = RuntimeError("broker unavailable")
    assert resolve_sqlite_account_id(b) == "default"


@patch("src.execution.account_factory.AlpacaBrokerAdapter.create")
def test_build_brokers_legacy_fallback_uses_default_key(
    mock_create: MagicMock,
    tmp_path: Path,
) -> None:
    """Empty ``broker.accounts`` → one adapter; dict key ``default`` when no default_account."""
    _minimal_yaml(tmp_path / "s.yaml")
    mock_create.return_value = MagicMock()
    settings = Settings(
        _yaml_path=tmp_path / "s.yaml",
        _env_file=None,
        alpaca_api_key="k",
        alpaca_secret_key="s",
        broker=BrokerConfig(accounts=[]),
    )
    out = build_brokers(settings)
    assert list(out.keys()) == ["default"]
    mock_create.assert_called_once()
    assert mock_create.call_args.kwargs["logical_account_id"] == "default"


@patch("src.execution.account_factory.AlpacaBrokerAdapter.create")
def test_build_brokers_legacy_fallback_respects_default_account(
    mock_create: MagicMock,
    tmp_path: Path,
) -> None:
    _minimal_yaml(tmp_path / "s.yaml")
    mock_create.return_value = MagicMock()
    settings = Settings(
        _yaml_path=tmp_path / "s.yaml",
        _env_file=None,
        alpaca_api_key="k",
        alpaca_secret_key="s",
        broker=BrokerConfig(default_account="Main", accounts=[]),
    )
    out = build_brokers(settings)
    assert list(out.keys()) == ["Main"]
    assert mock_create.call_args.kwargs["logical_account_id"] == "Main"


@patch("src.execution.account_factory.AlpacaBrokerAdapter.create")
def test_build_brokers_from_two_account_configs(
    mock_create: MagicMock,
    tmp_path: Path,
) -> None:
    yaml_path = tmp_path / "two.yaml"
    yaml_path.write_text(
        """
app: {name: "X", version: "0.1.0", environment: "development"}
data: {universe: ["SPY"]}
broker:
  accounts:
    - name: "Main"
      account_type: brokerage
      api_key_env: ALPACA_ACCOUNT_MAIN_KEY
      api_secret_env: ALPACA_ACCOUNT_MAIN_SECRET
    - name: "IRA"
      account_type: ira_traditional
      api_key_env: ALPACA_ACCOUNT_IRA_KEY
      api_secret_env: ALPACA_ACCOUNT_IRA_SECRET
risk: {max_position_pct: 0.25}
""",
        encoding="utf-8",
    )
    mock_create.return_value = MagicMock()
    settings = Settings(_yaml_path=yaml_path, _env_file=None)
    with patch.dict(
        "os.environ",
        {
            "ALPACA_ACCOUNT_MAIN_KEY": "k1",
            "ALPACA_ACCOUNT_MAIN_SECRET": "s1",
            "ALPACA_ACCOUNT_IRA_KEY": "k2",
            "ALPACA_ACCOUNT_IRA_SECRET": "s2",
        },
        clear=False,
    ):
        out = build_brokers(settings)
    assert set(out.keys()) == {"Main", "IRA"}
    assert mock_create.call_count == 2
    ids = {
        mock_create.call_args_list[i].kwargs["logical_account_id"]
        for i in range(2)
    }
    assert ids == {"Main", "IRA"}


@patch("src.execution.account_factory.AlpacaBrokerAdapter.create")
def test_build_brokers_blank_account_name_uses_default_account(
    mock_create: MagicMock,
    tmp_path: Path,
) -> None:
    """``AccountConfig.name`` empty/whitespace → label from ``broker.default_account``."""
    _minimal_yaml(tmp_path / "s.yaml")
    mock_create.return_value = MagicMock()
    settings = Settings(
        _yaml_path=tmp_path / "s.yaml",
        _env_file=None,
        broker=BrokerConfig(
            default_account="Primary",
            accounts=[
                AccountConfig(
                    name="   ",
                    api_key_env="E1",
                    api_secret_env="E2",
                ),
            ],
        ),
    )
    with patch.dict(
        "os.environ",
        {"E1": "k1", "E2": "s1"},
        clear=False,
    ):
        out = build_brokers(settings)
    assert list(out.keys()) == ["Primary"]
    mock_create.assert_called_once()
    assert mock_create.call_args[0] == ("k1", "s1")
    assert mock_create.call_args.kwargs["logical_account_id"] == "Primary"


@patch("src.execution.account_factory.AlpacaBrokerAdapter.create")
def test_build_brokers_blank_name_and_blank_default_uses_default_key(
    mock_create: MagicMock,
    tmp_path: Path,
) -> None:
    _minimal_yaml(tmp_path / "s.yaml")
    mock_create.return_value = MagicMock()
    settings = Settings(
        _yaml_path=tmp_path / "s.yaml",
        _env_file=None,
        broker=BrokerConfig(
            default_account="",
            accounts=[
                AccountConfig(
                    name="",
                    api_key_env="E1",
                    api_secret_env="E2",
                ),
            ],
        ),
    )
    with patch.dict(
        "os.environ",
        {"E1": "k1", "E2": "s1"},
        clear=False,
    ):
        out = build_brokers(settings)
    assert list(out.keys()) == ["default"]
    assert mock_create.call_args.kwargs["logical_account_id"] == "default"


def test_build_brokers_raises_when_multi_account_env_missing(tmp_path: Path) -> None:
    yaml_path = tmp_path / "one_acct.yaml"
    yaml_path.write_text(
        """
app: {name: "X", version: "0.1.0", environment: "development"}
data: {universe: ["SPY"]}
broker:
  accounts:
    - name: "Main"
      account_type: brokerage
      api_key_env: MISSING_KEY
      api_secret_env: MISSING_SECRET
risk: {max_position_pct: 0.25}
""",
        encoding="utf-8",
    )
    settings = Settings(_yaml_path=yaml_path, _env_file=None)
    with pytest.raises(ValueError, match="Missing credentials"):
        build_brokers(settings)
