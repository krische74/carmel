"""Tests for tax-loss harvest candidate scanning."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

from src.config import Settings, TaxConfig

if TYPE_CHECKING:
    from pathlib import Path
from src.strategy.tax_loss_harvest import TaxLossHarvester


def _settings(
    *,
    harvest_enabled: bool = True,
    threshold_pct: float = 0.05,
    min_loss_usd: float = 50.0,
    replacement_map: dict[str, str] | None = None,
) -> Settings:
    return Settings(
        _yaml_path=None,
        _env_file=None,
        tax=TaxConfig(
            harvest_enabled=harvest_enabled,
            harvest_threshold_pct=threshold_pct,
            harvest_min_loss_dollars=min_loss_usd,
            replacement_map=replacement_map or {},
        ),
    )


def test_harvester_finds_lot_with_loss_above_threshold(tmp_path: Path) -> None:
    from src.portfolio.tax_lots import LotLedger

    db = tmp_path / "hub.sqlite"
    ledger = LotLedger(db)
    opened = datetime(2025, 1, 1, 12, 0, tzinfo=UTC)
    ledger.record_buy("SPY", 10.0, 100.0, opened)
    lot = ledger.get_open_lots("SPY")[0]

    settings = _settings(threshold_pct=0.05, min_loss_usd=50.0)
    h = TaxLossHarvester(lot_ledger=ledger, settings=settings)
    prices = {"SPY": 80.0}
    cands = h.find_candidates(prices, set(), as_of=datetime(2026, 6, 1, 16, 0, tzinfo=UTC))
    assert len(cands) == 1
    assert cands[0].lot_id == lot.id
    assert cands[0].symbol == "SPY"
    assert cands[0].unrealized_loss < 0
    assert abs(cands[0].unrealized_loss) >= 50.0


def test_harvester_skips_lot_with_gain(tmp_path: Path) -> None:
    from src.portfolio.tax_lots import LotLedger

    ledger = LotLedger(tmp_path / "h.sqlite")
    ledger.record_buy("SPY", 10.0, 100.0, datetime(2025, 1, 1, tzinfo=UTC))
    settings = _settings()
    h = TaxLossHarvester(lot_ledger=ledger, settings=settings)
    cands = h.find_candidates({"SPY": 110.0}, set(), as_of=datetime(2026, 6, 1, tzinfo=UTC))
    assert cands == []


def test_harvester_skips_lot_below_pct_threshold(tmp_path: Path) -> None:
    from src.portfolio.tax_lots import LotLedger

    ledger = LotLedger(tmp_path / "h.sqlite")
    ledger.record_buy("SPY", 10.0, 100.0, datetime(2025, 1, 1, tzinfo=UTC))
    settings = _settings(threshold_pct=0.05, min_loss_usd=1.0)
    h = TaxLossHarvester(lot_ledger=ledger, settings=settings)
    cands = h.find_candidates({"SPY": 97.0}, set(), as_of=datetime(2026, 6, 1, tzinfo=UTC))
    assert cands == []


def test_harvester_skips_lot_below_dollar_threshold(tmp_path: Path) -> None:
    from src.portfolio.tax_lots import LotLedger

    ledger = LotLedger(tmp_path / "h.sqlite")
    ledger.record_buy("SPY", 1.0, 100.0, datetime(2025, 1, 1, tzinfo=UTC))
    settings = _settings(threshold_pct=0.01, min_loss_usd=50.0)
    h = TaxLossHarvester(lot_ledger=ledger, settings=settings)
    cands = h.find_candidates({"SPY": 60.0}, set(), as_of=datetime(2026, 6, 1, tzinfo=UTC))
    assert cands == []


def test_harvester_skips_protected_symbol(tmp_path: Path) -> None:
    from src.portfolio.tax_lots import LotLedger

    ledger = LotLedger(tmp_path / "h.sqlite")
    ledger.record_buy("SPY", 10.0, 100.0, datetime(2025, 1, 1, tzinfo=UTC))
    settings = _settings()
    h = TaxLossHarvester(lot_ledger=ledger, settings=settings)
    cands = h.find_candidates({"SPY": 80.0}, {"SPY"}, as_of=datetime(2026, 6, 1, tzinfo=UTC))
    assert cands == []


def test_harvester_skips_symbol_with_recent_sale(tmp_path: Path) -> None:
    from src.portfolio.tax_lots import LotLedger

    ledger = LotLedger(tmp_path / "h.sqlite")
    old_open = datetime(2024, 1, 1, tzinfo=UTC)
    ledger.record_buy("SPY", 10.0, 100.0, old_open)
    sale_day = datetime(2026, 5, 20, 16, 0, tzinfo=UTC)
    ledger.record_sell("SPY", 5.0, 90.0, sale_day)
    ledger.record_buy("SPY", 10.0, 95.0, datetime(2026, 5, 21, tzinfo=UTC))

    settings = _settings()
    h = TaxLossHarvester(lot_ledger=ledger, settings=settings)
    as_of = datetime(2026, 6, 1, 16, 0, tzinfo=UTC)
    px = 70.0
    cands = h.find_candidates({"SPY": px}, set(), as_of=as_of)
    assert cands == []


def test_harvester_skips_lot_purchased_within_30_days(tmp_path: Path) -> None:
    from src.portfolio.tax_lots import LotLedger

    ledger = LotLedger(tmp_path / "h.sqlite")
    recent = datetime(2026, 5, 20, 12, 0, tzinfo=UTC)
    ledger.record_buy("SPY", 10.0, 100.0, recent)
    settings = _settings()
    h = TaxLossHarvester(lot_ledger=ledger, settings=settings)
    as_of = datetime(2026, 6, 1, 16, 0, tzinfo=UTC)
    cands = h.find_candidates({"SPY": 80.0}, set(), as_of=as_of)
    assert cands == []


def test_harvester_includes_replacement_symbol_from_config(tmp_path: Path) -> None:
    from src.portfolio.tax_lots import LotLedger

    ledger = LotLedger(tmp_path / "h.sqlite")
    ledger.record_buy("SPY", 10.0, 100.0, datetime(2025, 1, 1, tzinfo=UTC))
    settings = _settings(replacement_map={"SPY": "IVV"})
    h = TaxLossHarvester(lot_ledger=ledger, settings=settings)
    cands = h.find_candidates({"SPY": 80.0}, set(), as_of=datetime(2026, 6, 1, tzinfo=UTC))
    assert len(cands) == 1
    assert cands[0].replacement_symbol == "IVV"


def test_harvester_works_without_replacement_symbol(tmp_path: Path) -> None:
    from src.portfolio.tax_lots import LotLedger

    ledger = LotLedger(tmp_path / "h.sqlite")
    ledger.record_buy("IWM", 10.0, 100.0, datetime(2025, 1, 1, tzinfo=UTC))
    settings = _settings(replacement_map={})
    h = TaxLossHarvester(lot_ledger=ledger, settings=settings)
    cands = h.find_candidates({"IWM": 80.0}, set(), as_of=datetime(2026, 6, 1, tzinfo=UTC))
    assert len(cands) == 1
    assert cands[0].replacement_symbol is None


def test_harvester_sorts_by_largest_loss_first(tmp_path: Path) -> None:
    from src.portfolio.tax_lots import LotLedger

    ledger = LotLedger(tmp_path / "h.sqlite")
    ledger.record_buy("AAA", 10.0, 50.0, datetime(2025, 1, 1, tzinfo=UTC))
    ledger.record_buy("BBB", 10.0, 100.0, datetime(2025, 1, 1, tzinfo=UTC))
    ledger.record_buy("CCC", 10.0, 200.0, datetime(2025, 1, 1, tzinfo=UTC))
    settings = _settings(min_loss_usd=1.0)
    h = TaxLossHarvester(lot_ledger=ledger, settings=settings)
    prices = {"AAA": 25.0, "BBB": 80.0, "CCC": 150.0}
    cands = h.find_candidates(prices, set(), as_of=datetime(2026, 6, 1, tzinfo=UTC))
    assert len(cands) == 3
    losses = [c.unrealized_loss for c in cands]
    assert losses == sorted(losses)


def test_harvester_returns_empty_when_disabled(tmp_path: Path) -> None:
    from src.portfolio.tax_lots import LotLedger

    ledger = LotLedger(tmp_path / "h.sqlite")
    ledger.record_buy("SPY", 10.0, 100.0, datetime(2025, 1, 1, tzinfo=UTC))
    settings = _settings(harvest_enabled=False)
    h = TaxLossHarvester(lot_ledger=ledger, settings=settings)
    cands = h.find_candidates({"SPY": 80.0}, set(), as_of=datetime(2026, 6, 1, tzinfo=UTC))
    assert cands == []
