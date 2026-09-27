"""SEC EDGAR fundamentals via edgartools (optional dependency)."""

from __future__ import annotations

import contextlib
import logging
import math
import os
import re
from typing import Any

import pandas as pd

from src.data.adapters.base import MarketDataAdapter

logger = logging.getLogger(__name__)

# Symbols that are almost never US-equity 10-K filers (funds / macro).
_DEFAULT_SKIP = frozenset(
    {
        "SPY",
        "QQQ",
        "IWM",
        "DIA",
        "VTI",
        "VOO",
        "VXUS",
        "BND",
        "AGG",
        "TLT",
        "GLD",
        "SLV",
        "SHV",
        "IVV",
        "QQQM",
        "VGLT",
    },
)


def _call_numeric(obj: Any, names: tuple[str, ...]) -> float | None:
    """Try ``get_*`` / ``*`` methods or properties returning a number."""
    for name in names:
        fn = getattr(obj, name, None)
        if callable(fn):
            try:
                v = fn()
                if v is not None and not (isinstance(v, float) and pd.isna(v)):
                    return float(v)
            except (TypeError, ValueError, AttributeError):
                continue
        elif fn is not None and not callable(fn):
            try:
                return float(fn)
            except (TypeError, ValueError):
                continue
    return None


def _statement_to_frame(stmt: Any) -> pd.DataFrame | None:
    if stmt is None:
        return None
    to_df = getattr(stmt, "to_dataframe", None)
    if callable(to_df):
        try:
            df = to_df()
            if isinstance(df, pd.DataFrame) and not df.empty:
                return df
        except (TypeError, ValueError, AttributeError):
            return None
    if isinstance(stmt, pd.DataFrame):
        return stmt
    return None


def _find_numeric_in_frame(df: pd.DataFrame, patterns: tuple[str, ...]) -> float | None:
    """Match row labels (first column or index) against regex; take last numeric column."""
    if df.empty:
        return None
    if df.shape[1] >= 2:
        labels = [str(x).lower() for x in df.iloc[:, 0]]
        col_vals = df.iloc[:, -1]
    else:
        labels = [str(x).lower() for x in df.index]
        col_vals = df.iloc[:, -1]
    for i, lab in enumerate(labels):
        for pat in patterns:
            if re.search(pat, lab, re.I):
                try:
                    v = float(col_vals.iloc[i])
                    if not math.isnan(v):
                        return v
                except (TypeError, ValueError, IndexError):
                    break
    return None


def extract_fundamentals_from_company(company: Any, symbol: str) -> dict[str, Any]:
    """Map an edgartools ``Company`` (or compatible) to ``compute_piotroski_f_score`` inputs."""
    sym_u = symbol.strip().upper()
    out: dict[str, Any] = {"symbol": sym_u, "period": ""}

    fin = None
    gf = getattr(company, "get_financials", None)
    if callable(gf):
        try:
            fin = gf()
        except (TypeError, ValueError, AttributeError, OSError):
            fin = None
    if fin is None:
        fin = getattr(company, "financials", None)

    if fin is None:
        return {}

    fy = _call_numeric(fin, ("get_fiscal_year", "fiscal_year"))
    if fy is not None:
        out["period"] = str(int(fy)) if float(fy).is_integer() else str(fy)
    else:
        out["period"] = str(pd.Timestamp.now(tz="UTC").year)

    out["total_assets"] = _call_numeric(
        fin,
        ("get_total_assets", "total_assets"),
    )
    out["net_income"] = _call_numeric(fin, ("get_net_income", "net_income"))
    out["operating_cash_flow"] = _call_numeric(
        fin,
        ("get_operating_cash_flow", "operating_cash_flow", "cash_from_operations"),
    )

    bs_fn = getattr(fin, "balance_sheet", None)
    inc_fn = getattr(fin, "income_statement", None)
    cf_fn = getattr(fin, "cashflow_statement", None)

    bs = _statement_to_frame(bs_fn() if callable(bs_fn) else None)
    inc = _statement_to_frame(inc_fn() if callable(inc_fn) else None)
    cf = _statement_to_frame(cf_fn() if callable(cf_fn) else None)

    if out["total_assets"] is None and bs is not None:
        out["total_assets"] = _find_numeric_in_frame(
            bs,
            (r"total\s+assets", r"assets\s+total"),
        )
    if out["net_income"] is None and inc is not None:
        out["net_income"] = _find_numeric_in_frame(
            inc,
            (r"net\s+income", r"profit.*net"),
        )
    if out["operating_cash_flow"] is None and cf is not None:
        out["operating_cash_flow"] = _find_numeric_in_frame(
            cf,
            (r"operating.*cash", r"cash.*operat"),
        )

    if out.get("total_assets") is None or float(out["total_assets"]) <= 0:
        return {}

    return out


class EdgarAdapter(MarketDataAdapter):
    """SEC EDGAR financial statements via edgartools when installed."""

    def fetch_ohlcv(
        self,
        symbol: str,
        *,
        start: str | pd.Timestamp | None = None,
        end: str | pd.Timestamp | None = None,
        interval: str = "1d",
        **kwargs: Any,
    ) -> pd.DataFrame:
        """SEC does not serve OHLCV."""
        return pd.DataFrame()

    def fetch_macro(self, series_id: str, **kwargs: Any) -> pd.DataFrame:
        """SEC does not serve macro series."""
        return pd.DataFrame()

    def fetch_fundamentals(self, symbol: str, **kwargs: Any) -> dict[str, Any]:
        """Fetch latest annual-style metrics for a US equity; ``{}`` for ETFs or missing deps."""
        sym = symbol.strip().upper()
        skip_extra = kwargs.get("skip_symbols")
        skip = set(_DEFAULT_SKIP)
        if skip_extra:
            skip |= {str(x).strip().upper() for x in skip_extra}
        if sym in skip:
            return {}

        try:
            from edgar import Company, set_identity
        except ImportError:
            logger.debug("edgar package not installed; pip install '.[fundamental]' for SEC data")
            return {}

        ident = (os.environ.get("EDGAR_IDENTITY") or "").strip()
        if ident:
            with contextlib.suppress(TypeError, ValueError, AttributeError):
                set_identity(ident)

        try:
            company = Company(sym)
            return extract_fundamentals_from_company(company, sym)
        except (TypeError, ValueError, AttributeError, OSError) as exc:
            logger.warning("Edgar fundamentals failed for %s: %s", sym, exc)
            return {}
