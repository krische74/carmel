"""Natural-language Q&A over portfolio data using Ollama."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.ai.llm_client import OllamaClient
    from src.config import Settings
    from src.data.storage.parquet_store import ParquetStore
    from src.data.storage.sqlite_store import SQLiteStore
    from src.portfolio.tax_lots import LotLedger

logger = logging.getLogger(__name__)

QA_SYSTEM_PROMPT = (
    "You are a financial assistant for a married couple's personal portfolio. "
    "Answer questions using ONLY the data provided below. Never fabricate numbers. "
    "Cite specific positions, dates, and dollar amounts. If the provided data is "
    'insufficient, say "I don\'t have data for that."'
)


class PortfolioQA:
    """Answers portfolio questions by injecting structured data into an LLM prompt."""

    def __init__(
        self,
        client: OllamaClient,
        sqlite_store: SQLiteStore,
        parquet_store: ParquetStore,
        lot_ledger: LotLedger | None = None,
        settings: Settings | None = None,
    ) -> None:
        self._client = client
        self._sqlite = sqlite_store
        self._parquet = parquet_store
        self._ledger = lot_ledger
        self._settings = settings

    def _build_portfolio_context(self) -> str:
        """Assemble a structured text block from stores."""
        sections: list[str] = []

        snaps = self._sqlite.get_equity_snapshots()
        if snaps:
            s = snaps[-1]
            mv = s.get("total_market_value", "N/A")
            cash = s.get("cash")
            pnl = s.get("total_pnl", "N/A")
            if cash is not None:
                try:
                    equity = float(mv) + float(cash)
                    sections.append(
                        f"=== Portfolio Snapshot ===\n"
                        f"Equity (MV + cash): ${equity:,.2f}\n"
                        f"Positions MV: ${float(mv):,.2f}\n"
                        f"Cash: ${float(cash):,.2f}\n"
                        f"Total P&L: ${pnl}",
                    )
                except (TypeError, ValueError):
                    sections.append(
                        f"=== Portfolio Snapshot ===\nMarket value: ${mv}\nTotal P&L: ${pnl}",
                    )
            else:
                sections.append(
                    f"=== Portfolio Snapshot ===\n"
                    f"Positions MV: ${mv} (cash unknown)\n"
                    f"Total P&L: ${pnl}",
                )

        if self._ledger is not None:
            open_lots = self._ledger.get_open_lots()
            if open_lots:
                lines = ["=== Open Positions (lots) ==="]
                for lot in open_lots[:30]:
                    lines.append(
                        f"  {lot.symbol}: {lot.quantity:.2f} shares @ ${lot.cost_basis:.2f}"
                    )
                sections.append("\n".join(lines))

        execs = self._sqlite.get_executions(limit=20)
        if execs:
            lines = ["=== Recent Trades (last 20) ==="]
            for e in execs:
                lines.append(
                    f"  {e.get('timestamp', '?')}: {e.get('side', '?')} {e.get('symbol', '?')} "
                    f"qty={e.get('qty', '?')}"
                )
            sections.append("\n".join(lines))

        alerts = self._sqlite.get_alerts(limit=10)
        if alerts:
            lines = ["=== Recent Alerts ==="]
            for a in alerts:
                lines.append(f"  [{a.get('level', '?')}] {a.get('message', '?')[:120]}")
            sections.append("\n".join(lines))

        regime_row = self._sqlite.get_latest_regime()
        if regime_row:
            overall = regime_row.get("overall", "N/A")
            vix = regime_row.get("vix_close", "N/A")
            sections.append(f"=== Market Regime ===\nOverall: {overall}, VIX: {vix}")

        if not sections:
            return "(No portfolio data available.)"
        return "\n\n".join(sections)

    def ask(self, question: str, *, include_docs: bool = False) -> str:
        """Answer a portfolio question using the LLM, or return a fallback message."""
        ctx = self._build_portfolio_context()
        doc_prefix = ""
        if include_docs and self._settings is not None:
            from src.ai.rag_index import retrieve_context

            chunks = retrieve_context(question, self._settings, self._client, self._sqlite)
            if chunks:
                doc_prefix = "Reference documentation (repo):\n" + "\n---\n".join(chunks) + "\n\n"
        prompt = (
            f"{doc_prefix}"
            f"Portfolio data:\n\n{ctx}\n\n"
            f"User question: {question}\n\n"
            "Answer concisely, referencing specific data from above."
        )
        result = self._client.generate(prompt, system=QA_SYSTEM_PROMPT)
        if result is not None:
            return result
        return "LLM is not available. Please check that Ollama is running."
