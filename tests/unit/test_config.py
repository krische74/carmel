"""Tests for the configuration module."""

import logging
import os
from datetime import UTC
from pathlib import Path
from unittest.mock import patch

import pytest

from src.config import (
    PROJECT_ROOT,
    DataConfig,
    MLConfig,
    Settings,
    hub_sqlite_path,
    ml_model_path,
    parquet_dir,
    parse_as_of_iso,
)


class TestSettings:
    """Test suite for the Settings configuration class."""

    def test_loads_default_settings_from_yaml(self, tmp_path: Path):
        """Settings loads values from config/settings.yaml by default."""
        from src.config import Settings

        settings = Settings(
            _yaml_path=Path("config/settings.yaml"),
            _env_file=None,
        )
        assert settings.app.name == "Carmel"
        assert settings.app.version == "0.1.0"

    def test_app_environment_defaults_to_development(self):
        """Without override, environment should be 'development'."""
        from src.config import Settings

        settings = Settings(
            _yaml_path=Path("config/settings.yaml"),
            _env_file=None,
        )
        assert settings.app.environment == "development"

    def test_broker_defaults_to_paper_trading(self):
        """Paper trading must be the default to prevent accidental live trades."""
        from src.config import Settings

        settings = Settings(
            _yaml_path=Path("config/settings.yaml"),
            _env_file=None,
        )
        assert settings.broker.paper_trading is True

    def test_risk_defaults_are_conservative(self):
        """Risk parameters should default to conservative values."""
        from src.config import Settings

        settings = Settings(
            _yaml_path=Path("config/settings.yaml"),
            _env_file=None,
        )
        assert settings.risk.max_position_pct <= 0.25
        assert settings.risk.daily_loss_limit_pct <= 0.05
        assert settings.risk.pdt_protection is True
        assert settings.risk.pdt_threshold == 4
        assert settings.risk.pdt_equity_floor == pytest.approx(25_000.0)

    def test_risk_config_min_order_notional_usd_default(self):
        """min_order_notional_usd defaults to 5.0 (model default and settings.yaml)."""
        from src.config import RiskConfig, Settings

        assert RiskConfig().min_order_notional_usd == pytest.approx(5.0)
        settings = Settings(
            _yaml_path=Path("config/settings.yaml"),
            _env_file=None,
        )
        assert settings.risk.min_order_notional_usd == pytest.approx(5.0)

    def test_cash_sweep_config_defaults(self):
        """CashSweepConfig defaults: enabled, BIL, 2% buffer, $50 min trade (Tier 44B)."""
        from src.config import CashSweepConfig, Settings

        c = CashSweepConfig()
        assert c.enabled is True
        assert c.symbol == "BIL"
        assert c.buffer_pct == pytest.approx(0.02)
        assert c.min_trade_usd == pytest.approx(50.0)

        settings = Settings(_yaml_path=Path("config/settings.yaml"), _env_file=None)
        assert settings.cash_sweep.enabled is True
        assert settings.cash_sweep.symbol == "BIL"
        assert settings.cash_sweep.buffer_pct == pytest.approx(0.02)
        assert settings.cash_sweep.min_trade_usd == pytest.approx(50.0)

    def test_cash_sweep_symbol_in_momentum_universe_rejected(self):
        """A sweep symbol inside the momentum universe must fail config validation."""
        from pydantic import ValidationError

        from src.config import (
            CashSweepConfig,
            DataConfig,
            MomentumConfig,
            Settings,
            StrategyConfig,
        )

        # SHV is momentum's cash_symbol → rotation would bulldoze a swept position.
        with pytest.raises(ValidationError, match="momentum universe"):
            Settings(
                _yaml_path=None,
                _env_file=None,
                data=DataConfig(universe=["SPY", "QQQ"]),
                strategy=StrategyConfig(momentum=MomentumConfig(cash_symbol="SHV")),
                cash_sweep=CashSweepConfig(enabled=True, symbol="SHV"),
            )

        # A symbol in data.universe is equally rejected (case/space-insensitive).
        with pytest.raises(ValidationError, match="momentum universe"):
            Settings(
                _yaml_path=None,
                _env_file=None,
                data=DataConfig(universe=["SPY", "QQQ"]),
                cash_sweep=CashSweepConfig(enabled=True, symbol=" spy "),
            )

    def test_cash_sweep_disabled_allows_universe_symbol(self):
        """When disabled, the momentum-universe guard does not apply."""
        from src.config import CashSweepConfig, DataConfig, Settings

        s = Settings(
            _yaml_path=None,
            _env_file=None,
            data=DataConfig(universe=["SPY", "QQQ"]),
            cash_sweep=CashSweepConfig(enabled=False, symbol="SPY"),
        )
        assert s.cash_sweep.enabled is False

    def test_execution_config_defaults_match_tier28_contract(self):
        """Execution defaults: market orders, no stop-loss, stale cancel window."""
        from src.config import Settings

        settings = Settings(
            _yaml_path=Path("config/settings.yaml"),
            _env_file=None,
        )
        assert settings.execution.default_order_type == "market"
        assert settings.execution.stop_loss_enabled is False
        assert settings.execution.unfilled_timeout_minutes == 30
        assert 0.0 <= settings.execution.limit_offset_bps <= 100.0

    def test_dashboard_config_defaults_and_auth_off(self):
        """Dashboard port/refresh; auth off by default (Tier 29)."""
        from src.config import Settings

        settings = Settings(
            _yaml_path=Path("config/settings.yaml"),
            _env_file=None,
        )
        assert settings.dashboard.port == 8501
        assert settings.dashboard.refresh_interval_seconds == 30
        assert settings.dashboard.auth_enabled is False
        assert settings.dashboard.auth_password == ""

    def test_env_vars_override_yaml(self):
        """Environment variables should take precedence over YAML values."""
        from src.config import Settings

        with patch.dict(os.environ, {"ALPACA_PAPER": "false"}):
            settings = Settings(
                _yaml_path=Path("config/settings.yaml"),
                _env_file=None,
            )
            assert settings.broker.paper_trading is False

    def test_momentum_universe_loaded(self):
        """Momentum strategy universe should be loaded from config."""
        from src.config import Settings

        settings = Settings(
            _yaml_path=Path("config/settings.yaml"),
            _env_file=None,
        )
        assert "SPY" in settings.data.universe
        assert "QQQ" in settings.data.universe
        assert len(settings.data.universe) >= 4

    def test_scheduler_reconcile_cron_defaults_empty(self) -> None:
        """Optional reconciliation cron is off unless configured."""
        from src.config import SchedulerConfig

        assert SchedulerConfig().reconcile_cron == ""

    def test_scheduler_digest_cron_defaults_empty(self) -> None:
        """Weekly digest cron is off unless configured."""
        from src.config import SchedulerConfig

        assert SchedulerConfig().digest_cron == ""

    def test_backtest_warmup_bars_defaults_zero(self) -> None:
        """CLI backtest warm-up is off unless ``backtest.warmup_bars`` is set."""
        from src.config import BacktestConfigSection

        assert BacktestConfigSection().warmup_bars == 0

    def test_yaml_sets_initial_backfill_years_to_twenty(self) -> None:
        """Tier 51: first-time ingest covers the 2007+ SHV-bound momentum window."""
        settings = Settings(
            _yaml_path=Path("config/settings.yaml"),
            _env_file=None,
        )
        assert settings.data.initial_backfill_years == 20

    def test_parquet_dir_resolves_relative(self) -> None:
        """Relative parquet_dir is anchored to project root."""
        s = Settings(
            _yaml_path=None,
            _env_file=None,
            data=DataConfig(parquet_dir="data/parquet", cache_dir="data/cache"),
        )
        assert parquet_dir(s) == PROJECT_ROOT / "data" / "parquet"

    def test_parquet_dir_absolute_unchanged(self, tmp_path: Path) -> None:
        abs_pq = tmp_path / "parquet"
        s = Settings(
            _yaml_path=None,
            _env_file=None,
            data=DataConfig(parquet_dir=str(abs_pq), cache_dir="data/cache"),
        )
        assert parquet_dir(s) == abs_pq

    def test_hub_sqlite_path_resolves_relative(self) -> None:
        """Hub SQLite lives under cache_dir relative to project root."""
        s = Settings(
            _yaml_path=None,
            _env_file=None,
            data=DataConfig(parquet_dir="data/parquet", cache_dir="data/cache"),
        )
        assert hub_sqlite_path(s) == PROJECT_ROOT / "data" / "cache" / "hub_metadata.sqlite"

    def test_strategy_enabled_lists_live_strategies(self):
        """Live workflow uses strategy.enabled from YAML."""
        from src.config import Settings

        settings = Settings(
            _yaml_path=Path("config/settings.yaml"),
            _env_file=None,
        )
        assert "momentum" in settings.strategy.enabled
        assert "dca" in settings.strategy.enabled

    def test_dca_defaults_from_yaml_are_weekly_at_2_5_pct(self) -> None:
        """Tier 45B: default YAML is weekly Mondays at 2.5% of equity (same weekly rate as prior 0.5%/day)."""
        from src.config import Settings

        settings = Settings(
            _yaml_path=Path("config/settings.yaml"),
            _env_file=None,
        )
        assert settings.strategy.dca.frequency == "weekly"
        assert settings.strategy.dca.percent_of_equity == pytest.approx(0.025)

    def test_strategy_ensemble_defaults_from_yaml(self) -> None:
        """Ensemble is off by default; strategy weights match settings.yaml."""
        from src.config import Settings

        settings = Settings(
            _yaml_path=Path("config/settings.yaml"),
            _env_file=None,
        )
        assert settings.strategy.ensemble.enabled is False
        assert settings.strategy.ensemble.strategy_weights[
            "MomentumRotationStrategy"
        ] == pytest.approx(
            0.5,
        )
        assert settings.strategy.ensemble.strategy_weights[
            "MeanReversionStrategy"
        ] == pytest.approx(0.5)
        assert settings.strategy.ensemble.cash_signal_penalty == pytest.approx(0.5)
        assert settings.strategy.ensemble.min_blended_weight == pytest.approx(0.05)

    def test_get_settings_returns_singleton(self):
        """get_settings() should return the same instance on repeated calls."""
        from src.config import get_settings

        s1 = get_settings()
        s2 = get_settings()
        assert s1 is s2

    def test_settings_loads_nested_env_var(self, tmp_path: Path) -> None:
        y = (tmp_path / "s.yaml").resolve()
        # Omit webhook_url (and avoid explicit enabled:) so nested env vars can apply.
        y.write_text("notification:\n  timeout_seconds: 10\n", encoding="utf-8")
        with patch.dict(
            os.environ, {"NOTIFICATION__WEBHOOK_URL": "https://nested.example/hook"}, clear=False
        ):
            s = Settings(_yaml_path=y, _env_file=None)
        assert s.notification.webhook_url == "https://nested.example/hook"

    def test_settings_loads_nested_bool_env_var(self, tmp_path: Path) -> None:
        y = (tmp_path / "s.yaml").resolve()
        y.write_text("notification:\n  timeout_seconds: 10\n", encoding="utf-8")
        with patch.dict(os.environ, {"NOTIFICATION__ENABLED": "true"}, clear=False):
            s = Settings(_yaml_path=y, _env_file=None)
        assert s.notification.enabled is True

    def test_settings_loads_nested_int_env_var(self, tmp_path: Path) -> None:
        y = (tmp_path / "s.yaml").resolve()
        # Omit smtp_port from YAML so env can override the default (587).
        y.write_text("notification:\n  timeout_seconds: 10\n", encoding="utf-8")
        with patch.dict(os.environ, {"NOTIFICATION__SMTP_PORT": "465"}, clear=False):
            s = Settings(_yaml_path=y, _env_file=None)
        assert s.notification.smtp_port == 465

    def test_settings_nested_env_overrides_yaml(self, tmp_path: Path) -> None:
        y = (tmp_path / "s.yaml").resolve()
        y.write_text("notification:\n  timeout_seconds: 10\n", encoding="utf-8")
        with patch.dict(os.environ, {"NOTIFICATION__ENABLED": "true"}, clear=False):
            s = Settings(_yaml_path=y, _env_file=None)
        assert s.notification.enabled is True

    def test_settings_flat_env_var_still_works(self, tmp_path: Path) -> None:
        y = (tmp_path / "s.yaml").resolve()
        y.write_text("app:\n  name: X\n", encoding="utf-8")
        with patch.dict(os.environ, {"ALPACA_API_KEY": "flat-key-123"}, clear=False):
            s = Settings(_yaml_path=y, _env_file=None)
        assert s.alpaca_api_key == "flat-key-123"

    def test_settings_nested_and_flat_coexist(self, tmp_path: Path) -> None:
        y = (tmp_path / "s.yaml").resolve()
        y.write_text("notification:\n  timeout_seconds: 10\n", encoding="utf-8")
        with patch.dict(
            os.environ,
            {
                "FRED_API_KEY": "fred-xyz",
                "NOTIFICATION__WEBHOOK_URL": "https://hook.example",
            },
            clear=False,
        ):
            s = Settings(_yaml_path=y, _env_file=None)
        assert s.fred_api_key == "fred-xyz"
        assert s.notification.webhook_url == "https://hook.example"

    def test_settings_warns_on_yaml_webhook_url(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        import src.config as cfg_mod

        monkeypatch.setattr(cfg_mod, "PROJECT_ROOT", tmp_path)
        y = (tmp_path / "s.yaml").resolve()
        y.write_text(
            "notification:\n  webhook_url: https://yaml-secret.example\n  enabled: true\n",
            encoding="utf-8",
        )
        popped = os.environ.pop("NOTIFICATION__WEBHOOK_URL", None)
        try:
            with caplog.at_level(logging.WARNING):
                cfg_mod.Settings(_yaml_path=y, _env_file=None)
        finally:
            if popped is not None:
                os.environ["NOTIFICATION__WEBHOOK_URL"] = popped
        assert "notification.webhook_url is set in YAML" in caplog.text

    def test_settings_does_not_warn_when_env_var_set(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        import src.config as cfg_mod

        monkeypatch.setattr(cfg_mod, "PROJECT_ROOT", tmp_path)
        y = (tmp_path / "s.yaml").resolve()
        y.write_text(
            "notification:\n  webhook_url: https://yaml.example\n  enabled: true\n",
            encoding="utf-8",
        )
        with (
            caplog.at_level(logging.WARNING),
            patch.dict(
                os.environ,
                {"NOTIFICATION__WEBHOOK_URL": "https://from-env.example"},
                clear=False,
            ),
        ):
            cfg_mod.Settings(_yaml_path=y, _env_file=None)
        assert "notification.webhook_url is set in YAML" not in caplog.text

    def test_settings_does_not_warn_when_dotenv_file_sets_webhook(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        import src.config as cfg_mod

        monkeypatch.setattr(cfg_mod, "PROJECT_ROOT", tmp_path)
        (tmp_path / ".env").write_text(
            "NOTIFICATION__WEBHOOK_URL=https://from-dotenv-file.example\n",
            encoding="utf-8",
        )
        y = (tmp_path / "s.yaml").resolve()
        y.write_text(
            "notification:\n  webhook_url: https://yaml.example\n  enabled: true\n",
            encoding="utf-8",
        )
        monkeypatch.chdir(tmp_path)
        popped = os.environ.pop("NOTIFICATION__WEBHOOK_URL", None)
        try:
            with caplog.at_level(logging.WARNING):
                cfg_mod.Settings(_yaml_path=y, _env_file=None)
        finally:
            if popped is not None:
                os.environ["NOTIFICATION__WEBHOOK_URL"] = popped
        assert "notification.webhook_url is set in YAML" not in caplog.text

    def test_settings_does_not_warn_on_empty_webhook_url(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        import src.config as cfg_mod

        monkeypatch.setattr(cfg_mod, "PROJECT_ROOT", tmp_path)
        y = (tmp_path / "s.yaml").resolve()
        y.write_text("notification:\n  timeout_seconds: 10\n", encoding="utf-8")
        with caplog.at_level(logging.WARNING):
            cfg_mod.Settings(_yaml_path=y, _env_file=None)
        assert "notification.webhook_url is set in YAML" not in caplog.text


def test_parse_as_of_iso_none_and_zulu() -> None:
    """Optional as-of: empty input is None; Z suffix parses as UTC."""
    assert parse_as_of_iso(None) is None
    assert parse_as_of_iso("   ") is None
    got = parse_as_of_iso("2026-01-15T14:30:00Z")
    assert got is not None
    assert got.tzinfo == UTC


def test_setup_logging_silences_apscheduler_executor_at_info() -> None:
    """The interval heartbeat job otherwise floods logs (~99% of volume)."""
    from src.config import setup_logging

    # Capture and restore existing logger state so this test doesn't bleed
    # configuration into others.
    sched_logger = logging.getLogger("apscheduler.executors.default")
    saved_level = sched_logger.level
    saved_root_level = logging.getLogger().level
    saved_handlers = logging.getLogger().handlers[:]
    try:
        setup_logging()
        assert sched_logger.getEffectiveLevel() >= logging.WARNING, (
            "apscheduler.executors.default must be WARNING+ to suppress "
            "per-minute heartbeat job execution logs; otherwise it's ~99% "
            "of file volume."
        )
        # Sanity check the opposite: src.automation must still log INFO.
        app_logger = logging.getLogger("src.automation")
        assert app_logger.getEffectiveLevel() <= logging.INFO
    finally:
        sched_logger.setLevel(saved_level)
        root = logging.getLogger()
        root.setLevel(saved_root_level)
        # Restore handlers to avoid leaking the file handler this test installed.
        for h in root.handlers[:]:
            root.removeHandler(h)
        for h in saved_handlers:
            root.addHandler(h)


class TestMLConfig:
    """Experimental ML settings."""

    @pytest.mark.parametrize("thr", [0.0, 0.25, 0.5, 0.99, 1.0])
    def test_ml_score_threshold_valid_values(self, thr: float) -> None:
        from src.config import MLConfig

        m = MLConfig(score_threshold=thr)
        assert m.score_threshold == pytest.approx(thr)

    @pytest.mark.parametrize("action", ["pass", "block"])
    def test_ml_missing_score_action_valid(self, action: str) -> None:
        from src.config import MLConfig

        m = MLConfig(missing_score_action=action)
        assert m.missing_score_action == action

    def test_ml_score_threshold_rejects_above_one(self) -> None:
        from pydantic import ValidationError

        from src.config import MLConfig

        with pytest.raises(ValidationError):
            MLConfig(score_threshold=1.01)

    def test_ml_score_threshold_rejects_negative(self) -> None:
        from pydantic import ValidationError

        from src.config import MLConfig

        with pytest.raises(ValidationError):
            MLConfig(score_threshold=-0.01)

    def test_ml_missing_score_action_rejects_invalid_literal(self) -> None:
        from pydantic import ValidationError

        from src.config import MLConfig

        with pytest.raises(ValidationError):
            MLConfig(missing_score_action="allow")  # type: ignore[arg-type]

    def test_ml_defaults_from_yaml(self) -> None:
        from src.config import Settings

        settings = Settings(
            _yaml_path=Path("config/settings.yaml"),
            _env_file=None,
        )
        assert settings.ml.enabled is False
        assert settings.ml.affect_orders is False
        assert settings.ml.score_threshold == pytest.approx(0.5)
        assert settings.ml.missing_score_action == "pass"
        assert settings.ml.label_horizon_days == 5
        assert "momentum_model" in settings.ml.model_path

    def test_ml_model_path_resolves_relative_to_project_root(self) -> None:
        s = Settings(
            _yaml_path=None,
            _env_file=None,
            data=DataConfig(),
            ml=MLConfig(model_path="data/ml/x.joblib"),
        )
        p = ml_model_path(s)
        assert p.name == "x.joblib"
        assert "data" in p.parts and "ml" in p.parts


class TestEnsembleConfig:
    """Strategy ensemble blending settings."""

    def test_cash_signal_penalty_rejects_above_one(self) -> None:
        from pydantic import ValidationError

        from src.config import EnsembleConfig

        with pytest.raises(ValidationError):
            EnsembleConfig(cash_signal_penalty=1.01)

    def test_min_blended_weight_rejects_above_one(self) -> None:
        from pydantic import ValidationError

        from src.config import EnsembleConfig

        with pytest.raises(ValidationError):
            EnsembleConfig(min_blended_weight=1.01)

    def test_cash_signal_penalty_can_be_zero(self) -> None:
        from src.config import EnsembleConfig

        c = EnsembleConfig(cash_signal_penalty=0.0)
        assert c.cash_signal_penalty == pytest.approx(0.0)


class TestRegimeAndFundamentalConfig:
    """Regime and fundamental YAML defaults."""

    def test_regime_defaults_from_yaml(self) -> None:
        from src.config import Settings

        settings = Settings(
            _yaml_path=Path("config/settings.yaml"),
            _env_file=None,
        )
        assert settings.regime.enabled is True
        assert settings.regime.vix_symbol == "^VIX"
        assert "DGS10" in settings.regime.fred_series
        assert settings.regime.sizing_adjustment["risk_on"] == pytest.approx(1.0)

    def test_fundamental_defaults(self) -> None:
        from src.config import FundamentalConfig

        f = FundamentalConfig()
        assert f.enabled is False
        assert f.refresh_interval_days == 30


class TestRegimeConfigValidation:
    """Validate RegimeConfig field_validator and model_validator (M1 / M15)."""

    def test_sizing_adjustment_rejects_invalid_keys(self) -> None:
        from src.config import RegimeConfig

        with pytest.raises(ValueError, match="Invalid sizing_adjustment"):
            RegimeConfig(sizing_adjustment={"riskon": 1.0})

    def test_sizing_adjustment_accepts_valid_keys(self) -> None:
        from src.config import RegimeConfig

        cfg = RegimeConfig(
            sizing_adjustment={
                "risk_on": 1.0,
                "cautious": 0.75,
                "defensive": 0.5,
                "crisis": 0.25,
            },
        )
        assert cfg.sizing_adjustment["crisis"] == 0.25

    def test_vix_threshold_ordering_enforced(self) -> None:
        from src.config import RegimeConfig

        with pytest.raises(ValueError, match="vix_low < vix_normal < vix_elevated"):
            RegimeConfig(vix_low=25.0, vix_normal=15.0, vix_elevated=30.0)

    def test_vix_thresholds_equal_rejected(self) -> None:
        from src.config import RegimeConfig

        with pytest.raises(ValueError, match="vix_low < vix_normal < vix_elevated"):
            RegimeConfig(vix_low=15.0, vix_normal=15.0, vix_elevated=30.0)

    def test_fundamental_refresh_interval_lower_bound(self) -> None:
        from src.config import FundamentalConfig

        with pytest.raises(ValueError):
            FundamentalConfig(refresh_interval_days=0)


def test_data_config_validation_exempt_symbols_default() -> None:
    """Default OHLCV validation exempt list includes ^VIX for volatility index spikes."""
    d = DataConfig()
    assert "^VIX" in d.validation_exempt_symbols


def test_data_config_validation_exempt_symbols_empty() -> None:
    """Operators may opt out of exemptions with an empty list."""
    d = DataConfig(validation_exempt_symbols=[])
    assert d.validation_exempt_symbols == []


class TestLoggingSetup:
    """Test suite for the logging configuration."""

    def test_setup_logging_configures_root_logger(self):
        """setup_logging() should configure the root logger without errors."""
        import logging
        import os

        from src.config import setup_logging

        logging.shutdown()
        setup_logging()
        root = logging.getLogger()
        assert root.level <= logging.INFO
        log_dir = os.environ.get("CARMEL_LOG_DIR", "")
        if log_dir:
            rot = [h for h in root.handlers if type(h).__name__ == "RotatingFileHandler"]
            assert rot
            assert all(str(h.baseFilename).startswith(log_dir) for h in rot)

    def test_setup_logging_respects_carmel_log_dir_env(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Explicit CARMEL_LOG_DIR must place carmel.log under that directory."""
        import logging

        from src.config import setup_logging

        dest = tmp_path / "alt_logs"
        dest.mkdir()
        monkeypatch.setenv("CARMEL_LOG_DIR", str(dest))
        logging.shutdown()
        setup_logging()
        root = logging.getLogger()
        file_handlers = [h for h in root.handlers if type(h).__name__ == "RotatingFileHandler"]
        assert file_handlers
        assert Path(file_handlers[0].baseFilename) == dest / "carmel.log"

    def test_setup_logging_applies_rotation_config(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        import logging

        from src.config import LogRotationConfig, setup_logging

        yaml_text = """
version: 1
disable_existing_loggers: false
handlers:
  file:
    class: logging.handlers.RotatingFileHandler
    level: INFO
    filename: logs/x.log
    maxBytes: 10485760
    backupCount: 5
    encoding: utf-8
root:
  level: INFO
  handlers: [file]
"""
        cfg_path = tmp_path / "logging.yaml"
        cfg_path.write_text(yaml_text.strip(), encoding="utf-8")
        monkeypatch.setattr("src.config.PROJECT_ROOT", tmp_path)

        logging.shutdown()
        setup_logging(
            config_path=cfg_path,
            log_rotation=LogRotationConfig(max_bytes=2_097_152, backup_count=7),
        )
        root = logging.getLogger()
        file_handlers = [h for h in root.handlers if type(h).__name__ == "RotatingFileHandler"]
        assert file_handlers
        assert file_handlers[0].maxBytes == 2_097_152
        assert file_handlers[0].backupCount == 7

    def test_default_rotation_values_match_logging_yaml_defaults(self) -> None:
        from src.config import LogRotationConfig

        lr = LogRotationConfig()
        assert lr.max_bytes == 10_485_760
        assert lr.backup_count == 5
