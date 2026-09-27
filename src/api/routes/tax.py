"""Tax summary and Form 8949 CSV export (informational, not tax advice)."""

from __future__ import annotations

import io
import logging
from datetime import UTC, datetime

from fastapi import APIRouter, Depends, File, Query, UploadFile
from fastapi.responses import StreamingResponse

from src.api.dependencies import get_lot_ledger, get_settings, get_sqlite_store
from src.config import Settings
from src.data.storage.sqlite_store import SQLiteStore
from src.portfolio.tax_lots import ClosedLot, LotLedger
from src.portfolio.wash_sales import (
    CrossAccountWashSale,
    detect_cross_account_wash_sales,
    partition_lots_by_account,
)
from src.reporting.reconciliation_1099b import (
    ReconciliationReport1099B,
    parse_1099b_csv_fileobj,
    reconcile_1099b,
)
from src.reporting.tax_report import (
    LossCarryforward,
    ScheduleDSummary,
    TaxSummary,
    compute_loss_carryforward,
    compute_tax_summary,
    form_8949_csv_string,
    generate_schedule_d_summary,
    schedule_d_csv_string,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["tax"])


def _norm_account_id(raw: str | None) -> str | None:
    if raw is None:
        return None
    s = str(raw).strip()
    return s if s else None


def _hub_account_types(settings: Settings) -> dict[str, str] | None:
    """Map hub account label → ``AccountConfig.account_type`` (brokerage / IRA / joint)."""
    acc = settings.broker.accounts
    if not acc:
        return None
    return {str(a.name).strip(): str(a.account_type) for a in acc if str(a.name).strip()}


def _closed_lot_year(cl: ClosedLot, year: int) -> bool:
    ca = cl.closed_at
    if isinstance(ca, datetime):
        if ca.tzinfo is None:
            ca = ca.replace(tzinfo=UTC)
        return int(ca.year) == int(year)
    return int(getattr(ca, "year", 0)) == int(year)


@router.get("/tax/summary", response_model=TaxSummary)
def get_tax_summary(
    year: int = Query(..., ge=2000, le=2100, description="Calendar tax year"),
    account_id: str | None = Query(
        None,
        description="Hub account id; omit for all accounts consolidated.",
    ),
    ledger: LotLedger = Depends(get_lot_ledger),
    settings: Settings = Depends(get_settings),
) -> TaxSummary:
    """Year-to-date style totals from closed lots (personal informational use)."""
    aid = _norm_account_id(account_id)
    win = int(settings.tax.wash_sale_window_days)
    wash = ledger.detect_wash_sales(account_id=aid, window_days=win)
    hub = _hub_account_types(settings)
    return compute_tax_summary(
        ledger,
        year,
        wash_sales=wash,
        account_id=aid,
        hub_account_types=hub,
    )


@router.get("/tax/export")
def get_tax_export(
    year: int = Query(..., ge=2000, le=2100),
    account_id: str | None = Query(
        None,
        description="Hub account id; omit for consolidated export.",
    ),
    ledger: LotLedger = Depends(get_lot_ledger),
    settings: Settings = Depends(get_settings),
) -> StreamingResponse:
    """Download Form 8949-style CSV for the selected year."""
    aid = _norm_account_id(account_id)
    win = int(settings.tax.wash_sale_window_days)
    wash = ledger.detect_wash_sales(account_id=aid, window_days=win)
    hub = _hub_account_types(settings)
    summary = compute_tax_summary(
        ledger,
        year,
        wash_sales=wash,
        account_id=aid,
        hub_account_types=hub,
    )
    body = form_8949_csv_string(summary, wash_sales=wash, account_label=aid)
    fname = f"form8949_{year}"
    if aid:
        safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in aid)[:80]
        fname += f"_{safe}"
    fname += ".csv"
    return StreamingResponse(
        iter([body.encode("utf-8")]),
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{fname}"'},
    )


@router.get("/tax/schedule-d", response_model=ScheduleDSummary)
def get_schedule_d(
    year: int = Query(..., ge=2000, le=2100),
    account_id: str | None = Query(None),
    ledger: LotLedger = Depends(get_lot_ledger),
    settings: Settings = Depends(get_settings),
) -> ScheduleDSummary:
    """Schedule D companion totals for the tax year (informational)."""
    aid = _norm_account_id(account_id)
    win = int(settings.tax.wash_sale_window_days)
    wash = ledger.detect_wash_sales(account_id=aid, window_days=win)
    hub = _hub_account_types(settings)
    summary = compute_tax_summary(
        ledger,
        year,
        wash_sales=wash,
        account_id=aid,
        hub_account_types=hub,
    )
    return generate_schedule_d_summary(summary, wash_sales=wash, account_id=aid)


@router.get("/tax/schedule-d/export")
def get_schedule_d_export(
    year: int = Query(..., ge=2000, le=2100),
    account_id: str | None = Query(None),
    ledger: LotLedger = Depends(get_lot_ledger),
    settings: Settings = Depends(get_settings),
) -> StreamingResponse:
    """Download Schedule D summary CSV."""
    aid = _norm_account_id(account_id)
    win = int(settings.tax.wash_sale_window_days)
    wash = ledger.detect_wash_sales(account_id=aid, window_days=win)
    hub = _hub_account_types(settings)
    summary = compute_tax_summary(
        ledger,
        year,
        wash_sales=wash,
        account_id=aid,
        hub_account_types=hub,
    )
    sd = generate_schedule_d_summary(summary, wash_sales=wash, account_id=aid)
    body = schedule_d_csv_string(sd)
    fname = f"schedule_d_{year}"
    if aid:
        safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in aid)[:80]
        fname += f"_{safe}"
    fname += ".csv"
    return StreamingResponse(
        iter([body.encode("utf-8")]),
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{fname}"'},
    )


@router.get("/tax/cross-account-washes", response_model=list[CrossAccountWashSale])
def get_cross_account_washes(
    year: int = Query(..., ge=2000, le=2100, description="Filter by loss sale year"),
    ledger: LotLedger = Depends(get_lot_ledger),
    settings: Settings = Depends(get_settings),
) -> list[CrossAccountWashSale]:
    """Cross-account wash flags (same symbol, replacement in another account)."""
    closed = ledger.get_closed_lots(account_id=None)
    open_lots = ledger.get_open_lots(account_id=None)
    by_c, by_o = partition_lots_by_account(closed, open_lots)
    win = int(settings.tax.wash_sale_window_days)
    raw = detect_cross_account_wash_sales(by_c, by_o, window_days=win)
    return [w for w in raw if int(w.sell_date.year) == int(year)]


@router.get("/tax/carryforward", response_model=LossCarryforward)
def get_tax_carryforward(
    year: int = Query(..., ge=2000, le=2100),
    account_id: str | None = Query(
        None,
        description="Hub account id; omit for consolidated (prior row uses 'default').",
    ),
    ledger: LotLedger = Depends(get_lot_ledger),
    store: SQLiteStore = Depends(get_sqlite_store),
    settings: Settings = Depends(get_settings),
) -> LossCarryforward:
    """Estimated capital loss carryforward after gains, netting, and the 3k deduction."""
    aid_scope = _norm_account_id(account_id)
    aid_store = aid_scope or "default"
    win = int(settings.tax.wash_sale_window_days)
    wash = ledger.detect_wash_sales(account_id=aid_scope, window_days=win)
    hub = _hub_account_types(settings)
    summary = compute_tax_summary(
        ledger,
        year,
        wash_sales=wash,
        account_id=aid_scope,
        hub_account_types=hub,
    )
    prev_row = store.get_loss_carryforward_row(int(year) - 1, account_id=aid_store)
    prior: LossCarryforward | None = None
    if prev_row is not None:
        prior = LossCarryforward(
            year=int(prev_row["year"]),
            account_id=str(prev_row["account_id"]),
            short_term_carryforward=float(prev_row["short_term_carryforward"]),
            long_term_carryforward=float(prev_row["long_term_carryforward"]),
            computed_at=str(prev_row.get("computed_at") or ""),
        )
    return compute_loss_carryforward(summary, prior, account_id=aid_store)


@router.post("/tax/reconcile-1099b", response_model=ReconciliationReport1099B)
async def post_reconcile_1099b(
    year: int = Query(..., ge=2000, le=2100),
    account_id: str | None = Query(None),
    file: UploadFile = File(..., description="Broker 1099-B CSV export"),
    ledger: LotLedger = Depends(get_lot_ledger),
) -> ReconciliationReport1099B:
    """Upload broker 1099-B CSV and compare to internal closed lots for the year."""
    aid = _norm_account_id(account_id)
    raw = await file.read()
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        msg = "1099-B file must be UTF-8 text CSV"
        logger.warning("1099-B upload decode failed")
        raise ValueError(msg) from None
    try:
        broker_records = parse_1099b_csv_fileobj(io.StringIO(text))
    except ValueError:
        raise
    closed = ledger.get_closed_lots(account_id=aid)
    closed_year = [cl for cl in closed if _closed_lot_year(cl, year)]
    return reconcile_1099b(broker_records, closed_year)
