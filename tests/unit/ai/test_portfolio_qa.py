"""PortfolioQA unit tests (stores and LLM mocked)."""

from __future__ import annotations

from unittest.mock import MagicMock

from src.ai.portfolio_qa import PortfolioQA


def _make_qa(
    *,
    generate_return: str | None = "LLM answer",
    snaps: list | None = None,
    open_lots: list | None = None,
    execs: list | None = None,
    alerts: list | None = None,
    regime: dict | None = None,
) -> PortfolioQA:
    client = MagicMock()
    client.generate.return_value = generate_return

    sqlite = MagicMock()
    sqlite.get_equity_snapshots.return_value = snaps or []
    sqlite.get_executions.return_value = execs or []
    sqlite.get_alerts.return_value = alerts or []
    sqlite.get_latest_regime.return_value = regime

    parquet = MagicMock()

    ledger = MagicMock()
    ledger.get_open_lots.return_value = open_lots or []

    return PortfolioQA(
        client=client, sqlite_store=sqlite, parquet_store=parquet, lot_ledger=ledger
    )


def _lot(symbol: str = "SPY", qty: float = 10.0, cost: float = 450.0) -> MagicMock:
    lot = MagicMock()
    lot.symbol = symbol
    lot.quantity = qty
    lot.cost_basis = cost
    return lot


def test_build_portfolio_context_includes_equity() -> None:
    qa = _make_qa(snaps=[{"total_market_value": 50000.0, "total_pnl": 500.0}])
    ctx = qa._build_portfolio_context()
    assert "50000" in ctx
    assert "500" in ctx


def test_build_portfolio_context_includes_open_lots() -> None:
    qa = _make_qa(open_lots=[_lot("SPY", 10, 450)])
    ctx = qa._build_portfolio_context()
    assert "SPY" in ctx
    assert "450" in ctx


def test_build_portfolio_context_includes_recent_trades() -> None:
    trade = {"timestamp": "2026-04-01T16:00:00Z", "side": "buy", "symbol": "QQQ", "quantity": 5}
    qa = _make_qa(execs=[trade])
    ctx = qa._build_portfolio_context()
    assert "QQQ" in ctx
    assert "buy" in ctx


def test_build_portfolio_context_includes_regime() -> None:
    qa = _make_qa(regime={"overall": "risk_on", "vix_close": 18.0})
    ctx = qa._build_portfolio_context()
    assert "risk_on" in ctx
    assert "18" in ctx


def test_build_portfolio_context_handles_empty_stores() -> None:
    qa = _make_qa()
    ctx = qa._build_portfolio_context()
    assert "No portfolio data available" in ctx


def test_ask_returns_llm_response() -> None:
    qa = _make_qa(
        generate_return="Your portfolio is worth $50k.",
        snaps=[{"total_market_value": 50000.0, "total_pnl": 0.0}],
    )
    ans = qa.ask("How much is my portfolio worth?")
    assert ans == "Your portfolio is worth $50k."


def test_ask_returns_fallback_when_llm_unavailable() -> None:
    qa = _make_qa(generate_return=None)
    ans = qa.ask("What is my P&L?")
    assert "not available" in ans.lower()


def test_ask_question_included_in_prompt() -> None:
    qa = _make_qa()
    qa.ask("What is my best performing stock?")
    prompt = qa._client.generate.call_args[0][0]
    assert "What is my best performing stock?" in prompt
