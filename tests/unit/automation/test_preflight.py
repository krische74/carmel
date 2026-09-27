"""Tier 32: preflight checks before live trading."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import pandas as pd

from src.automation.preflight import PreflightStatus, run_preflight_checks
from src.config import (
    BrokerConfig,
    DataConfig,
    MeanReversionConfig,
    MomentumConfig,
    NotificationConfig,
    RiskConfig,
    Settings,
    StrategyConfig,
    TaxConfig,
)
from src.data.symbol_resolver import resolve_required_symbols

if TYPE_CHECKING:
    from pathlib import Path


def _settings(**kwargs) -> Settings:
    base = dict(
        _yaml_path=None,
        _env_file=None,
        alpaca_api_key="k",
        alpaca_secret_key="s",
    )
    base.update(kwargs)
    return Settings(**base)


def _recent_run_at() -> str:
    """A backtest ``run_at`` inside preflight's 30-day recency window.

    Kept relative to now (not a hardcoded date) so tests asserting no preflight
    warnings don't rot as the clock advances.
    """
    return (datetime.now(UTC) - timedelta(days=5)).isoformat()


def test_preflight_passes_valid_config() -> None:
    broker = MagicMock()
    broker.refresh_account = MagicMock()
    broker.get_account_equity.return_value = 50_000.0
    s = _settings()
    s.risk = RiskConfig(daily_loss_limit_pct=0.05, pdt_protection=True)
    s.strategy.enabled = ["dca"]
    s.notification = NotificationConfig(enabled=True, webhook_url="https://example.com/hook")
    s.alpaca_paper = True
    s.broker.paper_trading = True
    store = MagicMock()
    store.get_backtest_runs.return_value = [
        {"run_at": _recent_run_at()},
    ]
    pq = MagicMock()
    pq.has_ohlcv_file.return_value = True
    long_idx = pd.bdate_range("2022-01-03", periods=400, freq="B")
    pq.read_ohlcv.return_value = pd.DataFrame(
        {
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.0,
            "volume": 1e6,
        },
        index=long_idx,
    )
    rep = run_preflight_checks(s, broker, sqlite_store=store, parquet_store=pq)
    assert not rep.failed
    assert not rep.warned
    hist = next(c for c in rep.checks if c.name == "ohlcv_backtest_history")
    assert hist.status == PreflightStatus.PASS


def test_preflight_fails_no_strategies() -> None:
    broker = MagicMock()
    broker.refresh_account = MagicMock()
    broker.get_account_equity.return_value = 10_000.0
    s = MagicMock()
    s.risk = RiskConfig(daily_loss_limit_pct=0.03, pdt_protection=True)
    s.strategy.enabled = []
    s.notification = NotificationConfig(enabled=True, webhook_url="https://example.com")
    s.alpaca_paper = True
    s.broker = BrokerConfig(paper_trading=True)
    rep = run_preflight_checks(s, broker, sqlite_store=MagicMock(get_backtest_runs=lambda **k: []))
    assert rep.failed
    assert any(
        c.name == "strategies_enabled" and c.status == PreflightStatus.FAIL for c in rep.checks
    )


def test_preflight_warns_no_notifications() -> None:
    broker = MagicMock()
    broker.refresh_account = MagicMock()
    broker.get_account_equity.return_value = 30_000.0
    s = _settings()
    s.strategy.enabled = ["momentum"]
    s.notification = NotificationConfig(enabled=False, webhook_url="", email_enabled=False)
    rep = run_preflight_checks(s, broker, sqlite_store=MagicMock(get_backtest_runs=lambda **k: []))
    assert any(
        c.name == "notification_channel" and c.status == PreflightStatus.WARN for c in rep.checks
    )


def test_preflight_warns_no_backtest() -> None:
    broker = MagicMock()
    broker.refresh_account = MagicMock()
    broker.get_account_equity.return_value = 30_000.0
    s = _settings()
    s.strategy.enabled = ["dca"]
    s.notification = NotificationConfig(enabled=True, webhook_url="https://x.test")
    store = MagicMock()
    store.get_backtest_runs.return_value = []
    rep = run_preflight_checks(s, broker, sqlite_store=store)
    assert any(
        c.name == "recent_backtest" and c.status == PreflightStatus.WARN for c in rep.checks
    )


def test_preflight_fails_kill_switch_too_high() -> None:
    broker = MagicMock()
    broker.refresh_account = MagicMock()
    broker.get_account_equity.return_value = 30_000.0
    s = _settings()
    s.risk = RiskConfig(daily_loss_limit_pct=1.0)
    s.strategy.enabled = ["dca"]
    s.notification = NotificationConfig(enabled=True, webhook_url="https://x.test")
    rep = run_preflight_checks(
        s, broker, sqlite_store=MagicMock(get_backtest_runs=lambda **k: [{}])
    )
    assert rep.failed
    assert any(
        c.name == "kill_switch_limit" and c.status == PreflightStatus.FAIL for c in rep.checks
    )


def test_preflight_warns_on_missing_required_symbol() -> None:
    broker = MagicMock()
    broker.refresh_account = MagicMock()
    broker.get_account_equity.return_value = 50_000.0
    s = _settings()
    s.risk = RiskConfig(daily_loss_limit_pct=0.05, pdt_protection=True)
    s.strategy.enabled = ["dca"]
    s.notification = NotificationConfig(enabled=True, webhook_url="https://example.com/hook")
    s.alpaca_paper = True
    s.broker.paper_trading = True
    store = MagicMock()
    store.get_backtest_runs.return_value = [
        {"run_at": "2026-04-01T12:00:00+00:00"},
    ]
    pq = MagicMock()

    def _has(sym: str) -> bool:
        return sym != "SHV"

    pq.has_ohlcv_file.side_effect = _has

    rep = run_preflight_checks(s, broker, sqlite_store=store, parquet_store=pq)
    assert not rep.failed
    assert rep.warned
    cov = next(c for c in rep.checks if c.name == "ohlcv_symbol_coverage")
    assert cov.status == PreflightStatus.WARN
    assert "SHV" in cov.detail


def test_preflight_passes_with_full_symbol_coverage(tmp_path: Path) -> None:
    broker = MagicMock()
    broker.refresh_account = MagicMock()
    broker.get_account_equity.return_value = 50_000.0
    missing = tmp_path / "n.yaml"
    assert not missing.exists()
    s = Settings(
        _yaml_path=missing,
        _env_file=None,
        data=DataConfig(
            parquet_dir="d",
            cache_dir="c",
            universe=["SPY"],
            dca_targets=[],
        ),
        strategy=StrategyConfig(
            momentum=MomentumConfig(cash_symbol="SHV"),
            mean_reversion=MeanReversionConfig(universe=[]),
        ),
        tax=TaxConfig(replacement_map={}),
    )
    s.risk = RiskConfig(daily_loss_limit_pct=0.05, pdt_protection=True)
    s.strategy.enabled = ["dca"]
    s.notification = NotificationConfig(enabled=True, webhook_url="https://example.com/hook")
    s.alpaca_paper = True
    s.broker.paper_trading = True
    store = MagicMock()
    store.get_backtest_runs.return_value = [
        {"run_at": _recent_run_at()},
    ]
    req = resolve_required_symbols(s)
    pq = MagicMock()
    pq.has_ohlcv_file.side_effect = lambda sym: sym in req

    rep = run_preflight_checks(s, broker, sqlite_store=store, parquet_store=pq)
    assert not rep.failed
    assert not rep.warned
    cov = next(c for c in rep.checks if c.name == "ohlcv_symbol_coverage")
    assert cov.status == PreflightStatus.PASS


def test_preflight_warns_on_short_history() -> None:
    broker = MagicMock()
    broker.refresh_account = MagicMock()
    broker.get_account_equity.return_value = 50_000.0
    s = _settings()
    s.risk = RiskConfig(daily_loss_limit_pct=0.05, pdt_protection=True)
    s.strategy.enabled = ["dca"]
    s.notification = NotificationConfig(enabled=True, webhook_url="https://example.com/hook")
    s.alpaca_paper = True
    s.broker.paper_trading = True
    store = MagicMock()
    store.get_backtest_runs.return_value = [
        {"run_at": "2026-04-01T12:00:00+00:00"},
    ]
    pq = MagicMock()
    pq.has_ohlcv_file.return_value = True
    short_idx = pd.bdate_range("2024-01-02", periods=50, freq="B")
    pq.read_ohlcv.return_value = pd.DataFrame(
        {
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.0,
            "volume": 1e6,
        },
        index=short_idx,
    )
    rep = run_preflight_checks(s, broker, sqlite_store=store, parquet_store=pq)
    assert not rep.failed
    assert rep.warned
    hist = next(c for c in rep.checks if c.name == "ohlcv_backtest_history")
    assert hist.status == PreflightStatus.WARN
    assert "365" in hist.detail


def test_preflight_passes_on_sufficient_history(tmp_path: Path) -> None:
    broker = MagicMock()
    broker.refresh_account = MagicMock()
    broker.get_account_equity.return_value = 50_000.0
    missing = tmp_path / "n.yaml"
    assert not missing.exists()
    s = Settings(
        _yaml_path=missing,
        _env_file=None,
        data=DataConfig(
            parquet_dir="d",
            cache_dir="c",
            universe=["SPY"],
            dca_targets=[],
        ),
        strategy=StrategyConfig(
            momentum=MomentumConfig(cash_symbol="SHV"),
            mean_reversion=MeanReversionConfig(universe=[]),
        ),
        tax=TaxConfig(replacement_map={}),
    )
    s.risk = RiskConfig(daily_loss_limit_pct=0.05, pdt_protection=True)
    s.strategy.enabled = ["dca"]
    s.notification = NotificationConfig(enabled=True, webhook_url="https://example.com/hook")
    s.alpaca_paper = True
    s.broker.paper_trading = True
    store = MagicMock()
    store.get_backtest_runs.return_value = [
        {"run_at": _recent_run_at()},
    ]
    req = resolve_required_symbols(s)
    pq = MagicMock()
    pq.has_ohlcv_file.side_effect = lambda sym: sym in req
    long_idx = pd.bdate_range("2022-01-03", periods=400, freq="B")
    long_df = pd.DataFrame(
        {
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.0,
            "volume": 1e6,
        },
        index=long_idx,
    )
    pq.read_ohlcv.return_value = long_df

    rep = run_preflight_checks(s, broker, sqlite_store=store, parquet_store=pq)
    assert not rep.failed
    assert not rep.warned
    hist = next(c for c in rep.checks if c.name == "ohlcv_backtest_history")
    assert hist.status == PreflightStatus.PASS


def test_preflight_fails_when_sweep_symbol_in_momentum_universe() -> None:
    """Tier 44B: a swept symbol inside the momentum universe is a rotation-bulldozer hazard."""
    broker = MagicMock()
    broker.refresh_account = MagicMock()
    broker.get_account_equity.return_value = 50_000.0
    s = _settings()
    s.strategy.enabled = ["dca"]
    # Mutate past the model validator (which blocks this at construction) to exercise
    # the preflight guard on the runtime-mutation path.
    s.cash_sweep.symbol = s.strategy.momentum.cash_symbol  # SHV — inside momentum universe
    rep = run_preflight_checks(
        s, broker, sqlite_store=MagicMock(get_backtest_runs=lambda **k: [{}])
    )
    assert rep.failed
    ch = next(c for c in rep.checks if c.name == "cash_sweep_symbol")
    assert ch.status == PreflightStatus.FAIL
    assert "momentum universe" in ch.detail


def test_preflight_fails_when_enabled_but_no_url() -> None:
    broker = MagicMock()
    broker.refresh_account = MagicMock()
    broker.get_account_equity.return_value = 50_000.0
    s = _settings()
    s.strategy.enabled = ["dca"]
    s.notification = NotificationConfig(enabled=True, webhook_url="", email_enabled=False)
    rep = run_preflight_checks(
        s, broker, sqlite_store=MagicMock(get_backtest_runs=lambda **k: [{}])
    )
    assert rep.failed
    ch = next(c for c in rep.checks if c.name == "notification_channel")
    assert ch.status == PreflightStatus.FAIL
    assert "webhook_url is empty" in ch.detail


def test_preflight_passes_when_webhook_configured() -> None:
    broker = MagicMock()
    broker.refresh_account = MagicMock()
    broker.get_account_equity.return_value = 50_000.0
    s = _settings()
    s.strategy.enabled = ["dca"]
    s.notification = NotificationConfig(
        enabled=True,
        webhook_url="https://hooks.example/test",
        email_enabled=False,
    )
    rep = run_preflight_checks(
        s, broker, sqlite_store=MagicMock(get_backtest_runs=lambda **k: [{}])
    )
    assert not rep.failed
    ch = next(c for c in rep.checks if c.name == "notification_channel")
    assert ch.status == PreflightStatus.PASS
    assert "webhook=True" in ch.detail
