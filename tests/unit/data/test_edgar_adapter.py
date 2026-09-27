"""EdgarAdapter interface tests — exercise real code paths."""

from __future__ import annotations

import pytest

from src.data.adapters.edgar_adapter import EdgarAdapter, extract_fundamentals_from_company


def test_edgar_ohlcv_and_macro_return_empty() -> None:
    a = EdgarAdapter()
    assert a.fetch_ohlcv("SPY").empty
    assert a.fetch_macro("DGS10").empty


def test_edgar_fetch_fundamentals_returns_empty_when_edgartools_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When ``edgartools`` is not installed, ``fetch_fundamentals`` returns ``{}``."""
    import builtins

    real_import = builtins.__import__

    def _block_edgartools(name: str, *args: object, **kwargs: object) -> object:
        if name == "edgartools":
            raise ImportError("simulated missing edgartools")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _block_edgartools)
    adapter = EdgarAdapter()
    result = adapter.fetch_fundamentals("AAPL")
    assert result == {}


def test_extract_fundamentals_from_company_maps_metrics() -> None:
    """``extract_fundamentals_from_company`` fills dict when financials expose getters."""

    class FakeFin:
        def get_fiscal_year(self) -> int:
            return 2024

        def get_total_assets(self) -> float:
            return 350e9

        def get_net_income(self) -> float:
            return 90e9

        def get_operating_cash_flow(self) -> float:
            return 100e9

    class FakeCo:
        def get_financials(self) -> FakeFin:
            return FakeFin()

    out = extract_fundamentals_from_company(FakeCo(), "AAPL")
    assert out.get("symbol") == "AAPL"
    assert out.get("period") == "2024"
    assert float(out["total_assets"]) == pytest.approx(350e9)
    assert float(out["net_income"]) == pytest.approx(90e9)


def test_edgar_fetch_skips_known_etf() -> None:
    adapter = EdgarAdapter()
    assert adapter.fetch_fundamentals("SPY") == {}
