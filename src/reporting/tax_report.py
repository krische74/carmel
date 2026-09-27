"""Year-end tax summaries and Form 8949 CSV export (informational, not tax advice)."""

from __future__ import annotations

import csv
import io
from datetime import UTC, date, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Literal, TextIO

from pydantic import BaseModel, Field, computed_field

if TYPE_CHECKING:
    from src.portfolio.tax_lots import LotLedger
    from src.portfolio.wash_sales import WashSale


class TaxLotSummary(BaseModel):
    """One closed lot with holding-period classification."""

    lot_id: str
    account_id: str = "default"
    symbol: str
    qty: float
    cost_basis: float
    proceeds: float
    realized_pnl: float
    opened_at: datetime
    closed_at: datetime
    holding_days: int
    term: Literal["short", "long"]
    is_tlh: bool = False
    tax_treatment: Literal["taxable", "tax_deferred", "tax_free"] = "taxable"


class Form8949Row(BaseModel):
    """One row for Form 8949-style CSV export (informational)."""

    description: str
    date_acquired: str
    date_sold: str
    proceeds: float
    cost_basis: float
    adjustment_code: str
    adjustment_amount: float
    gain_or_loss: float
    term: Literal["short", "long"]
    account_id: str


class ScheduleDSummary(BaseModel):
    """Schedule D style totals derived from Form 8949 lot rows (informational)."""

    year: int
    account_id: str | None = None
    short_term_proceeds: float
    short_term_cost_basis: float
    short_term_wash_adjustments: float
    short_term_net: float
    long_term_proceeds: float
    long_term_cost_basis: float
    long_term_wash_adjustments: float
    long_term_net: float
    total_net_gain_or_loss: float


class TaxSummary(BaseModel):
    """Aggregated tax-year figures from closed lots (informational)."""

    year: int
    short_term_gains: float
    short_term_losses: float
    short_term_net: float
    long_term_gains: float
    long_term_losses: float
    long_term_net: float
    total_net: float
    harvested_losses: float = Field(
        default=0.0,
        description="Absolute TLH-tagged losses (informational).",
    )
    wash_sale_disallowed: float = 0.0
    lots: list[TaxLotSummary] = Field(default_factory=list)
    account_type: str | None = Field(
        default=None,
        description="Hub portfolio type when a single account scope is selected.",
    )


class LossCarryforward(BaseModel):
    """Remaining capital loss carryforward after gains, cross-term netting, and 3k deduction."""

    year: int
    account_id: str = "default"
    short_term_carryforward: float = 0.0
    long_term_carryforward: float = 0.0
    computed_at: str = ""

    @computed_field
    def total_carryforward(self) -> float:
        """Total unused loss carried to the next tax year (ST + LT buckets)."""
        return float(self.short_term_carryforward) + float(self.long_term_carryforward)


def _is_long_term(opened: date, closed: date) -> bool:
    """IRS long-term: held for more than one year (calendar-date comparison)."""
    try:
        one_year_later = opened.replace(year=opened.year + 1)
    except ValueError:
        one_year_later = opened.replace(year=opened.year + 1, day=28)
    return closed > one_year_later


def _lot_tax_treatment(
    account_id: str,
    *,
    hub_account_types: dict[str, str] | None,
    portfolio_account_type: str | None,
) -> Literal["taxable", "tax_deferred", "tax_free"]:
    """Map hub account / scope to tax treatment for reporting."""
    if portfolio_account_type:
        t = str(portfolio_account_type).strip()
    elif hub_account_types is not None:
        aid = account_id.strip() or "default"
        t = str(hub_account_types.get(aid, "brokerage")).strip()
    else:
        t = "brokerage"
    if t == "ira_traditional":
        return "tax_deferred"
    if t == "ira_roth":
        return "tax_free"
    return "taxable"


def _subset_taxable_summary(summary: TaxSummary) -> TaxSummary:
    """Return a summary with only taxable lots (Schedule D / taxable Form 8949 totals)."""
    tl = [x for x in summary.lots if x.tax_treatment == "taxable"]
    st_g = st_l = lt_g = lt_l = 0.0
    harvested = 0.0
    for lot in tl:
        rp = float(lot.realized_pnl)
        if lot.term == "short":
            if rp >= 0:
                st_g += rp
            else:
                st_l += rp
        else:
            if rp >= 0:
                lt_g += rp
            else:
                lt_l += rp
        if lot.is_tlh and rp < 0:
            harvested += abs(rp)
    return TaxSummary(
        year=int(summary.year),
        short_term_gains=st_g,
        short_term_losses=st_l,
        short_term_net=st_g + st_l,
        long_term_gains=lt_g,
        long_term_losses=lt_l,
        long_term_net=lt_g + lt_l,
        total_net=st_g + st_l + lt_g + lt_l,
        harvested_losses=harvested,
        wash_sale_disallowed=0.0,
        lots=tl,
        account_type=summary.account_type,
    )


def compute_tax_summary(
    lot_ledger: LotLedger,
    year: int,
    *,
    wash_sales: list[WashSale] | None = None,
    account_id: str | None = None,
    hub_account_types: dict[str, str] | None = None,
    portfolio_account_type: str | None = None,
) -> TaxSummary:
    """Aggregate closed-lot P&L by tax year and holding period.

    When ``account_id`` is ``None``, all hub accounts are **consolidated** into one
    summary. When set, only closed lots for that hub account id are included.

    ``hub_account_types`` maps hub ``account_id`` strings to ``AccountConfig.account_type``
    values (``brokerage``, ``ira_traditional``, ``ira_roth``, ``joint``). Lots in
    tax-advantaged accounts remain in ``lots`` for informational export, but **taxable**
    totals (short/long net) include **brokerage** and **joint** only.

    When ``portfolio_account_type`` is set (single-account scope), it overrides per-id
    lookup for every lot in the result.
    """
    closed = lot_ledger.get_closed_lots(account_id=account_id)
    in_year = []
    for cl in closed:
        ca = cl.closed_at
        if ca.tzinfo is None:
            ca = ca.replace(tzinfo=UTC)
        if ca.year != int(year):
            continue
        in_year.append(cl)

    st_g = st_l = lt_g = lt_l = 0.0
    harvested = 0.0
    summaries: list[TaxLotSummary] = []

    for cl in in_year:
        oa = cl.opened_at
        ca = cl.closed_at
        if oa.tzinfo is None:
            oa = oa.replace(tzinfo=UTC)
        if ca.tzinfo is None:
            ca = ca.replace(tzinfo=UTC)
        if isinstance(oa, datetime) and isinstance(ca, datetime):
            oa_date = oa.date()
            ca_date = ca.date()
        elif isinstance(oa, date) and isinstance(ca, date):
            oa_date = oa
            ca_date = ca
        else:
            oa_date = ca_date = date.today()
        holding_days = abs((ca_date - oa_date).days)
        term: Literal["short", "long"] = "long" if _is_long_term(oa_date, ca_date) else "short"
        cb = float(cl.qty) * float(cl.cost_per_share)
        proceeds = float(cl.qty) * float(cl.sell_price)
        rp = float(cl.realized_pnl)
        is_tlh = bool(getattr(cl, "is_tlh", False))
        acct = str(getattr(cl, "account_id", None) or "default").strip() or "default"
        tt = _lot_tax_treatment(
            acct,
            hub_account_types=hub_account_types,
            portfolio_account_type=portfolio_account_type,
        )
        summaries.append(
            TaxLotSummary(
                lot_id=cl.lot_id,
                account_id=acct,
                symbol=cl.symbol,
                qty=float(cl.qty),
                cost_basis=cb,
                proceeds=proceeds,
                realized_pnl=rp,
                opened_at=oa,
                closed_at=ca,
                holding_days=holding_days,
                term=term,
                is_tlh=is_tlh,
                tax_treatment=tt,
            ),
        )
        if tt != "taxable":
            continue
        if term == "short":
            if rp >= 0:
                st_g += rp
            else:
                st_l += rp
        else:
            if rp >= 0:
                lt_g += rp
            else:
                lt_l += rp
        if is_tlh and rp < 0:
            harvested += abs(rp)

    taxable_lot_ids = {s.lot_id for s in summaries if s.tax_treatment == "taxable"}
    wash_disallowed = 0.0
    if wash_sales:
        for ws in wash_sales:
            if int(ws.sell_date.year) != int(year):
                continue
            key = _wash_sale_lot_key(ws)
            if summaries and key not in taxable_lot_ids:
                continue
            wash_disallowed += float(ws.disallowed_loss)

    scope_type: str | None = None
    if portfolio_account_type:
        scope_type = str(portfolio_account_type)
    elif account_id is not None and hub_account_types is not None:
        scope_type = str(hub_account_types.get(account_id.strip() or "default", "brokerage"))

    return TaxSummary(
        year=int(year),
        short_term_gains=st_g,
        short_term_losses=st_l,
        short_term_net=st_g + st_l,
        long_term_gains=lt_g,
        long_term_losses=lt_l,
        long_term_net=lt_g + lt_l,
        total_net=st_g + st_l + lt_g + lt_l,
        harvested_losses=harvested,
        wash_sale_disallowed=wash_disallowed,
        lots=summaries,
        account_type=scope_type,
    )


def compute_loss_carryforward(
    tax_summary: TaxSummary,
    prior_carryforward: LossCarryforward | None = None,
    *,
    annual_deduction_limit: float = 3_000.0,
    account_id: str = "default",
) -> LossCarryforward:
    """Compute remaining loss carryforward after applying gains and the 3k deduction.

    Ordering: apply prior-year ST carryforward against current ST gains, then LT gains;
    then prior LT carryforward against LT gains, then ST gains; remainder joins current
    loss pools. Net ST and LT, combine for overall capital gain/loss, then apply up to
    ``annual_deduction_limit`` against a net capital loss; any excess splits into ST/LT
    carryforward (remainder assigned to the ST bucket when indivisible).
    """
    pst = float(prior_carryforward.short_term_carryforward) if prior_carryforward else 0.0
    plt = float(prior_carryforward.long_term_carryforward) if prior_carryforward else 0.0

    st_g = max(0.0, float(tax_summary.short_term_net))
    st_l = max(0.0, -float(tax_summary.short_term_net))
    lt_g = max(0.0, float(tax_summary.long_term_net))
    lt_l = max(0.0, -float(tax_summary.long_term_net))

    while pst > 0.0 and st_g > 0.0:
        u = min(pst, st_g)
        pst -= u
        st_g -= u
    while pst > 0.0 and lt_g > 0.0:
        u = min(pst, lt_g)
        pst -= u
        lt_g -= u
    st_l += pst
    pst = 0.0

    while plt > 0.0 and lt_g > 0.0:
        u = min(plt, lt_g)
        plt -= u
        lt_g -= u
    while plt > 0.0 and st_g > 0.0:
        u = min(plt, st_g)
        plt -= u
        st_g -= u
    lt_l += plt
    plt = 0.0

    st_net = st_g - st_l
    lt_net = lt_g - lt_l
    total = st_net + lt_net
    if total >= 0.0:
        return LossCarryforward(
            year=int(tax_summary.year),
            account_id=account_id.strip() or "default",
            short_term_carryforward=0.0,
            long_term_carryforward=0.0,
            computed_at=datetime.now(UTC).isoformat(),
        )

    loss = -total
    deduct = min(float(annual_deduction_limit), loss)
    remainder = loss - deduct
    return LossCarryforward(
        year=int(tax_summary.year),
        account_id=account_id.strip() or "default",
        short_term_carryforward=remainder,
        long_term_carryforward=0.0,
        computed_at=datetime.now(UTC).isoformat(),
    )


def consolidated_taxable_and_advantaged(
    summary: TaxSummary,
) -> tuple[TaxSummary, TaxSummary]:
    """Split a consolidated summary into taxable vs tax-advantaged lot subsets."""
    tax_lots = [x for x in summary.lots if x.tax_treatment == "taxable"]
    adv_lots = [x for x in summary.lots if x.tax_treatment != "taxable"]
    taxable_part = _subset_taxable_summary(summary.model_copy(update={"lots": tax_lots}))
    st_g = st_l = lt_g = lt_l = 0.0
    harvested = 0.0
    for lot in adv_lots:
        rp = float(lot.realized_pnl)
        if lot.term == "short":
            if rp >= 0:
                st_g += rp
            else:
                st_l += rp
        else:
            if rp >= 0:
                lt_g += rp
            else:
                lt_l += rp
        if lot.is_tlh and rp < 0:
            harvested += abs(rp)
    adv_summary = TaxSummary(
        year=int(summary.year),
        short_term_gains=st_g,
        short_term_losses=st_l,
        short_term_net=st_g + st_l,
        long_term_gains=lt_g,
        long_term_losses=lt_l,
        long_term_net=lt_g + lt_l,
        total_net=st_g + st_l + lt_g + lt_l,
        harvested_losses=harvested,
        wash_sale_disallowed=0.0,
        lots=adv_lots,
        account_type=None,
    )
    return taxable_part, adv_summary


def _wash_sale_lot_key(ws: object) -> str:
    """Map wash row to tax lot id (``closed_lot_id`` may be ``lot_id|closed_at_iso``)."""
    raw = str(getattr(ws, "closed_lot_id", ""))
    if "|" in raw:
        return raw.split("|", 1)[0].strip()
    return raw.strip()


FORM_8949_COLUMN_HEADERS = [
    "(a) Description",
    "(b) Date Acquired",
    "(c) Date Sold",
    "(d) Proceeds",
    "(e) Cost or Other Basis",
    "(f) Code",
    "(g) Adjustment",
    "(h) Gain or Loss",
]


def _wash_adjustments_by_lot(wash_sales: list[WashSale] | None) -> dict[str, float]:
    ws_by_lot: dict[str, float] = {}
    if wash_sales:
        for ws in wash_sales:
            key = _wash_sale_lot_key(ws)
            ws_by_lot[key] = ws_by_lot.get(key, 0.0) + float(ws.disallowed_loss)
    return ws_by_lot


def build_form_8949_rows(
    summary: TaxSummary,
    wash_sales: list[WashSale] | None = None,
) -> list[Form8949Row]:
    """Build IRS-oriented rows (Part I/II split by ``term`` on each row)."""
    ws_by_lot = _wash_adjustments_by_lot(wash_sales)
    rows: list[Form8949Row] = []
    for lot in summary.lots:
        adj = float(ws_by_lot.get(lot.lot_id, 0.0))
        code = "W" if adj > 0.0 else ""
        adjusted_basis = float(lot.cost_basis) + adj
        gain_or_loss = float(lot.proceeds) - adjusted_basis
        rows.append(
            Form8949Row(
                description=f"{lot.qty:.4f} sh {lot.symbol}",
                date_acquired=_fmt_mmddyyyy(lot.opened_at),
                date_sold=_fmt_mmddyyyy(lot.closed_at),
                proceeds=float(lot.proceeds),
                cost_basis=float(lot.cost_basis),
                adjustment_code=code,
                adjustment_amount=adj,
                gain_or_loss=gain_or_loss,
                term=lot.term,
                account_id=lot.account_id,
            ),
        )
    return rows


def generate_schedule_d_summary(
    tax_summary: TaxSummary,
    *,
    wash_sales: list[WashSale] | None = None,
    account_id: str | None = None,
) -> ScheduleDSummary:
    """Aggregate Schedule D line items from the same lots as Form 8949 (informational)."""
    taxable = _subset_taxable_summary(tax_summary)
    ws_use: list[WashSale] | None = None
    if wash_sales:
        tids = {lot.lot_id for lot in taxable.lots}
        ws_use = [w for w in wash_sales if _wash_sale_lot_key(w) in tids]
    rows = build_form_8949_rows(taxable, ws_use)
    st_p = st_c = st_w = st_n = 0.0
    lt_p = lt_c = lt_w = lt_n = 0.0
    for r in rows:
        if r.term == "short":
            st_p += r.proceeds
            st_c += r.cost_basis
            st_w += r.adjustment_amount
            st_n += r.gain_or_loss
        else:
            lt_p += r.proceeds
            lt_c += r.cost_basis
            lt_w += r.adjustment_amount
            lt_n += r.gain_or_loss
    return ScheduleDSummary(
        year=int(tax_summary.year),
        account_id=account_id,
        short_term_proceeds=st_p,
        short_term_cost_basis=st_c,
        short_term_wash_adjustments=st_w,
        short_term_net=st_n,
        long_term_proceeds=lt_p,
        long_term_cost_basis=lt_c,
        long_term_wash_adjustments=lt_w,
        long_term_net=lt_n,
        total_net_gain_or_loss=st_n + lt_n,
    )


def _write_schedule_d_csv(summary: ScheduleDSummary, f: TextIO) -> None:
    w = csv.writer(f)
    w.writerow(["Line item", "Amount"])
    w.writerow(["Tax year", str(summary.year)])
    if summary.account_id is not None:
        w.writerow(["Account scope", summary.account_id])
    w.writerow(["Part I - Short-term proceeds", f"{summary.short_term_proceeds:.2f}"])
    w.writerow(["Part I - Short-term cost or other basis", f"{summary.short_term_cost_basis:.2f}"])
    w.writerow(
        [
            "Part I - Short-term wash adjustments (code W)",
            f"{summary.short_term_wash_adjustments:.2f}",
        ]
    )
    w.writerow(["Part I - Short-term net gain or loss", f"{summary.short_term_net:.2f}"])
    w.writerow(["Part II - Long-term proceeds", f"{summary.long_term_proceeds:.2f}"])
    w.writerow(["Part II - Long-term cost or other basis", f"{summary.long_term_cost_basis:.2f}"])
    w.writerow(
        [
            "Part II - Long-term wash adjustments (code W)",
            f"{summary.long_term_wash_adjustments:.2f}",
        ]
    )
    w.writerow(["Part II - Long-term net gain or loss", f"{summary.long_term_net:.2f}"])
    w.writerow(["Total net gain or loss", f"{summary.total_net_gain_or_loss:.2f}"])


def export_schedule_d_csv(summary: ScheduleDSummary, output_path: Path) -> Path:
    """Write Schedule D companion CSV (informational)."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        _write_schedule_d_csv(summary, f)
    return output_path


def schedule_d_csv_string(summary: ScheduleDSummary) -> str:
    """In-memory Schedule D CSV text."""
    buf = io.StringIO()
    _write_schedule_d_csv(summary, buf)
    return buf.getvalue()


def _write_form_8949_csv(
    summary: TaxSummary,
    f: TextIO,
    wash_sales: list[WashSale] | None = None,
    *,
    account_label: str | None = None,
) -> None:
    rows = build_form_8949_rows(summary, wash_sales)
    short_rows = [r for r in rows if r.term == "short"]
    long_rows = [r for r in rows if r.term == "long"]
    pad_title = [""] * 7
    pad_scope = [""] * 6
    w = csv.writer(f)
    if any(lot.tax_treatment != "taxable" for lot in summary.lots):
        w.writerow(
            [
                "IRA — Not Reportable on Form 8949 (informational only)",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
            ],
        )

    def write_data_row(r: Form8949Row) -> None:
        w.writerow(
            [
                r.description,
                r.date_acquired,
                r.date_sold,
                f"{r.proceeds:.2f}",
                f"{r.cost_basis:.2f}",
                r.adjustment_code,
                f"{r.adjustment_amount:.2f}",
                f"{r.gain_or_loss:.2f}",
            ],
        )

    if account_label:
        scope = str(account_label).strip()
        w.writerow(["Report scope", f"Account: {scope}", *pad_scope])
    w.writerow(["Form 8949 Part I - Short-Term", *pad_title])
    w.writerow(FORM_8949_COLUMN_HEADERS)
    for r in short_rows:
        write_data_row(r)
    w.writerow(["Form 8949 Part II - Long-Term", *pad_title])
    w.writerow(FORM_8949_COLUMN_HEADERS)
    for r in long_rows:
        write_data_row(r)


def export_form_8949_csv(
    summary: TaxSummary,
    output_path: Path,
    wash_sales: list[WashSale] | None = None,
    *,
    account_label: str | None = None,
) -> Path:
    """Write a CSV suitable for Form 8949-style reporting (informational)."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        _write_form_8949_csv(summary, f, wash_sales, account_label=account_label)
    return output_path


def form_8949_csv_string(
    summary: TaxSummary,
    wash_sales: list[WashSale] | None = None,
    *,
    account_label: str | None = None,
) -> str:
    """In-memory CSV text (e.g. for HTTP download)."""
    buf = io.StringIO()
    _write_form_8949_csv(summary, buf, wash_sales, account_label=account_label)
    return buf.getvalue()


def _fmt_mmddyyyy(dt: datetime | date) -> str:
    """Format a datetime or date as MM/DD/YYYY for Form 8949 CSV."""
    if isinstance(dt, datetime):
        d = dt.date()
    elif isinstance(dt, date):
        d = dt
    else:
        msg = f"Expected datetime or date, got {type(dt).__name__}"
        raise TypeError(msg)
    return f"{d.month:02d}/{d.day:02d}/{d.year:04d}"
