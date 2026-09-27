"""Tier 30: multi-account broker configuration."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from src.config import BrokerConfig, Settings


def test_account_config_parses_correctly(tmp_path: Path) -> None:
    yaml_path = tmp_path / "multi.yaml"
    yaml_path.write_text(
        """
app:
  name: "Carmel"
  version: "0.1.0"
  environment: "development"
data:
  universe: ["SPY"]
broker:
  provider: alpaca
  paper_trading: true
  default_account: "Main"
  accounts:
    - name: "Main"
      account_type: brokerage
      api_key_env: ALPACA_ACCOUNT_MAIN_KEY
      api_secret_env: ALPACA_ACCOUNT_MAIN_SECRET
    - name: "IRA"
      account_type: ira_traditional
      api_key_env: ALPACA_ACCOUNT_IRA_KEY
      api_secret_env: ALPACA_ACCOUNT_IRA_SECRET
risk:
  max_position_pct: 0.25
""",
        encoding="utf-8",
    )
    s = Settings(_yaml_path=yaml_path, _env_file=None)
    assert len(s.broker.accounts) == 2
    assert s.broker.accounts[0].name == "Main"
    assert s.broker.accounts[0].api_key_env == "ALPACA_ACCOUNT_MAIN_KEY"
    assert s.broker.accounts[1].account_type == "ira_traditional"
    assert s.broker.default_account == "Main"


def test_empty_accounts_falls_back_to_legacy() -> None:
    s = Settings(_yaml_path=Path("config/settings.yaml"), _env_file=None)
    assert s.broker.accounts == []


def test_account_keys_loaded_from_env(tmp_path: Path) -> None:
    yaml_path = tmp_path / "one.yaml"
    yaml_path.write_text(
        """
app: {name: "X", version: "0.1.0", environment: "development"}
data: {universe: ["SPY"]}
broker:
  accounts:
    - name: "Main"
      account_type: brokerage
      api_key_env: MY_KEY_VAR
      api_secret_env: MY_SECRET_VAR
risk: {max_position_pct: 0.25}
""",
        encoding="utf-8",
    )
    with patch.dict(
        "os.environ",
        {"MY_KEY_VAR": "key123", "MY_SECRET_VAR": "sec456"},
        clear=False,
    ):
        s = Settings(_yaml_path=yaml_path, _env_file=None)
        ac = s.broker.accounts[0]
        assert __import__("os").environ.get(ac.api_key_env) == "key123"
        assert __import__("os").environ.get(ac.api_secret_env) == "sec456"


def test_broker_config_default_account_optional() -> None:
    b = BrokerConfig()
    assert b.default_account == ""
    assert b.accounts == []
