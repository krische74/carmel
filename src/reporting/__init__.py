"""Reporting: performance metrics from stored data."""

from src.reporting.attribution import (
    BrinsonAttribution,
    BrinsonSummary,
    PositionContribution,
    compute_brinson_attribution,
    compute_brinson_from_snapshots,
    compute_position_contributions,
)
from src.reporting.equity_curve import (
    DailyEquityPoint,
    backfill_equity_curve,
    record_equity_snapshot,
)
from src.reporting.fundamental_score import FScoreResult, compute_piotroski_f_score
from src.reporting.performance import (
    PerformanceSummary,
    compute_equity_curve,
    compute_performance_summary,
)
from src.reporting.portfolio_analytics import (
    PortfolioValuation,
    PositionSummary,
    compute_portfolio_valuation,
    load_current_prices,
)
from src.reporting.returns import (
    ReturnMetrics,
    compute_daily_returns,
    compute_drawdown_series,
    compute_return_metrics,
    compute_rolling_sharpe,
)
from src.reporting.tax_report import (
    TaxLotSummary,
    TaxSummary,
    compute_tax_summary,
    export_form_8949_csv,
)

__all__ = [
    "BrinsonAttribution",
    "BrinsonSummary",
    "DailyEquityPoint",
    "FScoreResult",
    "PerformanceSummary",
    "PortfolioValuation",
    "PositionContribution",
    "PositionSummary",
    "ReturnMetrics",
    "TaxLotSummary",
    "TaxSummary",
    "backfill_equity_curve",
    "compute_brinson_attribution",
    "compute_brinson_from_snapshots",
    "compute_daily_returns",
    "compute_drawdown_series",
    "compute_equity_curve",
    "compute_performance_summary",
    "compute_piotroski_f_score",
    "compute_portfolio_valuation",
    "compute_position_contributions",
    "compute_return_metrics",
    "compute_rolling_sharpe",
    "compute_tax_summary",
    "export_form_8949_csv",
    "load_current_prices",
    "record_equity_snapshot",
]
