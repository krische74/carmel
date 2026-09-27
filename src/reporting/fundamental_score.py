"""Piotroski F-Score from normalized fundamental fields (informational)."""

from __future__ import annotations

import logging
import math
from typing import Any

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class FScoreResult(BaseModel):
    """Piotroski F-Score breakdown (0-9)."""

    symbol: str
    period: str
    score: int = Field(ge=0, le=9)
    profitability: int = Field(ge=0, le=4)
    leverage: int = Field(ge=0, le=3)
    efficiency: int = Field(ge=0, le=2)
    criteria: dict[str, bool]


def compute_piotroski_f_score(fundamentals: dict[str, Any]) -> FScoreResult | None:
    """Compute the 9-point Piotroski F-Score from financial data dict."""
    if not fundamentals:
        return None
    sym = str(fundamentals.get("symbol", "")).strip()
    per = str(fundamentals.get("period", "")).strip()
    if not sym or not per:
        return None

    ta = fundamentals.get("total_assets")
    ni = fundamentals.get("net_income")
    ocf = fundamentals.get("operating_cash_flow")
    td = fundamentals.get("total_debt")
    td_p = fundamentals.get("total_debt_prior")
    cr = fundamentals.get("current_ratio")
    cr_p = fundamentals.get("current_ratio_prior")
    sh = fundamentals.get("shares_outstanding")
    sh_p = fundamentals.get("shares_outstanding_prior")
    gm = fundamentals.get("gross_margin")
    gm_p = fundamentals.get("gross_margin_prior")
    ato = fundamentals.get("asset_turnover")
    ato_p = fundamentals.get("asset_turnover_prior")
    roa = fundamentals.get("roa")
    roa_p = fundamentals.get("roa_prior")

    if ta is None or float(ta) <= 0.0:
        return None

    for label, val in (("total_assets", ta), ("net_income", ni), ("operating_cash_flow", ocf)):
        if val is not None:
            fv = float(val)
            if math.isnan(fv) or math.isinf(fv):
                logger.warning(
                    "F-Score: %s has non-finite %s (%s); returning None", sym, label, fv
                )
                return None

    crit: dict[str, bool] = {}
    prof = lev = eff = 0

    # Profitability (4)
    r1 = float(ni) / float(ta) > 0.0 if ni is not None else False
    crit["positive_roa"] = r1
    prof += int(r1)

    r2 = float(ocf) > 0.0 if ocf is not None else False
    crit["positive_ocf"] = r2
    prof += int(r2)

    r3 = False
    if roa is not None and roa_p is not None:
        r3 = float(roa) > float(roa_p)
    crit["improving_roa"] = r3
    prof += int(r3)

    r4 = False
    if ocf is not None and ni is not None:
        r4 = float(ocf) > float(ni)
    crit["ocf_exceeds_ni"] = r4
    prof += int(r4)

    # Leverage / liquidity (3)
    r5 = False
    if td is not None and td_p is not None:
        r5 = float(td) < float(td_p)
    crit["lower_long_term_debt"] = r5
    lev += int(r5)

    r6 = False
    if cr is not None and cr_p is not None:
        r6 = float(cr) > float(cr_p)
    crit["improving_current_ratio"] = r6
    lev += int(r6)

    r7 = False
    if sh is not None and sh_p is not None:
        r7 = float(sh) <= float(sh_p)
    crit["no_dilution"] = r7
    lev += int(r7)

    # Efficiency (2)
    r8 = False
    if gm is not None and gm_p is not None:
        r8 = float(gm) > float(gm_p)
    crit["improving_gross_margin"] = r8
    eff += int(r8)

    r9 = False
    if ato is not None and ato_p is not None:
        r9 = float(ato) > float(ato_p)
    crit["improving_asset_turnover"] = r9
    eff += int(r9)

    total = prof + lev + eff
    return FScoreResult(
        symbol=sym,
        period=per,
        score=total,
        profitability=prof,
        leverage=lev,
        efficiency=eff,
        criteria=crit,
    )
