"""Tests for portfolio valuation from tax lots and Parquet prices."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

import pandas as pd
import pytest

from src.reporting.portfolio_analytics import (
    compute_portfolio_valuation,
    load_current_prices,
)

if TYPE_CHECKING:
    from pathlib import Path

from src.portfolio.tax_lots import LotLedger


def test_portfolio_valuation_empty_lots(tmp_path: Path) -> None:
    led = LotLedger(tmp_path / "x.db")
    v = compute_portfolio_valuation(led, {})
    assert v.positions == []
    assert v.total_market_value == 0.0
    assert v.total_cost_basis == 0.0
    assert v.total_unrealized_pnl == 0.0
    assert v.total_realized_pnl == 0.0
    assert v.total_pnl == 0.0


def test_portfolio_valuation_single_symbol_one_lot(tmp_path: Path) -> None:
    led = LotLedger(tmp_path / "x.db")
    led.record_buy("SPY", 10.0, 100.0, datetime(2026, 1, 1, tzinfo=UTC))
    v = compute_portfolio_valuation(led, {"SPY": 120.0})
    assert len(v.positions) == 1
    p = v.positions[0]
    assert p.symbol == "SPY"
    assert p.open_lots == 1
    assert p.total_qty == pytest.approx(10.0)
    assert p.avg_cost_per_share == pytest.approx(100.0)
    assert p.total_cost_basis == pytest.approx(1000.0)
    assert p.current_price == pytest.approx(120.0)
    assert p.market_value == pytest.approx(1200.0)
    assert p.unrealized_pnl == pytest.approx(200.0)
    assert p.unrealized_pnl_pct == pytest.approx(0.20)
    assert v.total_market_value == pytest.approx(1200.0)
    assert v.total_cost_basis == pytest.approx(1000.0)
    assert v.total_unrealized_pnl == pytest.approx(200.0)
    assert v.total_realized_pnl == 0.0
    assert v.total_pnl == pytest.approx(200.0)


def test_portfolio_valuation_single_symbol_multiple_lots(tmp_path: Path) -> None:
    led = LotLedger(tmp_path / "x.db")
    led.record_buy("SPY", 5.0, 100.0, datetime(2026, 1, 1, tzinfo=UTC))
    led.record_buy("SPY", 10.0, 110.0, datetime(2026, 2, 1, tzinfo=UTC))
    # basis 500 + 1100 = 1600, qty 15, avg = 106.666...
    v = compute_portfolio_valuation(led, {"SPY": 120.0})
    p = v.positions[0]
    assert p.total_qty == pytest.approx(15.0)
    assert p.total_cost_basis == pytest.approx(1600.0)
    assert p.avg_cost_per_share == pytest.approx(1600.0 / 15.0)
    assert p.open_lots == 2
    assert p.market_value == pytest.approx(15.0 * 120.0)
    assert p.unrealized_pnl == pytest.approx(p.market_value - 1600.0)


def test_portfolio_valuation_multiple_symbols(tmp_path: Path) -> None:
    led = LotLedger(tmp_path / "x.db")
    led.record_buy("SPY", 10.0, 100.0, datetime(2026, 1, 1, tzinfo=UTC))
    led.record_buy("QQQ", 5.0, 200.0, datetime(2026, 1, 2, tzinfo=UTC))
    prices = {"SPY": 110.0, "QQQ": 210.0}
    v = compute_portfolio_valuation(led, prices)
    assert len(v.positions) == 2
    by = {x.symbol: x for x in v.positions}
    assert by["QQQ"].market_value == pytest.approx(5.0 * 210.0)
    assert by["SPY"].total_cost_basis == pytest.approx(1000.0)
    assert v.total_market_value == pytest.approx(1100.0 + 1050.0)
    assert v.total_cost_basis == pytest.approx(1000.0 + 1000.0)


def test_portfolio_valuation_includes_realized_pnl(tmp_path: Path) -> None:
    led = LotLedger(tmp_path / "x.db")
    led.record_buy("SPY", 10.0, 100.0, datetime(2026, 1, 1, tzinfo=UTC))
    led.record_sell("SPY", 10.0, 120.0, datetime(2026, 2, 1, tzinfo=UTC))
    v = compute_portfolio_valuation(led, {"SPY": 100.0})
    assert v.positions == []
    assert v.total_unrealized_pnl == 0.0
    assert v.total_realized_pnl == pytest.approx(200.0)
    assert v.total_pnl == pytest.approx(200.0)


def test_portfolio_valuation_missing_price_zero_market_value(tmp_path: Path) -> None:
    led = LotLedger(tmp_path / "x.db")
    led.record_buy("DEAD", 10.0, 50.0, datetime(2026, 1, 1, tzinfo=UTC))
    v = compute_portfolio_valuation(led, {})
    assert len(v.positions) == 1
    p = v.positions[0]
    assert p.current_price == 0.0
    assert p.market_value == 0.0
    assert p.total_cost_basis == pytest.approx(500.0)
    assert p.unrealized_pnl == pytest.approx(-500.0)


def test_load_current_prices_from_parquet(tmp_path: Path) -> None:
    from src.data.storage.parquet_store import ParquetStore

    pq = ParquetStore(tmp_path / "pq")
    idx = pd.DatetimeIndex([pd.Timestamp("2026-01-01"), pd.Timestamp("2026-01-02")])
    df = pd.DataFrame(
        {
            "open": [100.0, 101.0],
            "high": [101.0, 102.0],
            "low": [99.0, 100.0],
            "close": [100.5, 105.0],
            "volume": [1e6, 1e6],
        },
        index=idx,
    )
    pq.write_ohlcv("SPY", df)
    prices = load_current_prices(pq, ["SPY"])
    assert prices["SPY"] == pytest.approx(105.0)


def test_load_current_prices_skips_missing_symbol(tmp_path: Path) -> None:
    from src.data.storage.parquet_store import ParquetStore

    pq = ParquetStore(tmp_path / "pq")
    assert load_current_prices(pq, ["NOPE"]) == {}
