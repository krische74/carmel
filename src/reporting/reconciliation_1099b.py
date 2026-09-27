"""1099-B broker CSV import and reconciliation against internal closed lots (informational)."""

from __future__ import annotations

import csv
import re
from datetime import date, datetime
from pathlib import Path
from typing import Literal, TextIO

from pydantic import BaseModel, Field

from src.portfolio.tax_lots import ClosedLot  # noqa: TC001


class BrokerLotRecord(BaseModel):
    """One row from a broker 1099-B style CSV export."""

    symbol: str
    qty: float
    date_acquired: date
    date_sold: date
    proceeds: float
    cost_basis: float


class LotMatch(BaseModel):
    """Outcome of matching one broker row to an internal closed lot."""

    broker_record: BrokerLotRecord | None = None
    internal_lot: ClosedLot | None = None
    status: Literal[
        "matched",
        "proceeds_mismatch",
        "basis_mismatch",
        "qty_mismatch",
        "missing_internal",
        "missing_broker",
    ]
    variance: float = Field(
        default=0.0,
        description="Combined dollar distance (proceeds + basis) for mismatches.",
    )


class ReconciliationReport1099B(BaseModel):
    """Summary of broker vs internal lot reconciliation."""

    total_broker_records: int
    total_internal_lots: int
    matched: int
    discrepancies: int
    matches: list[LotMatch]


def _norm_header(h: str) -> str:
    s = str(h).strip().lower()
    s = re.sub(r"[\s\-]+", "_", s)
    return s


def _parse_date(raw: str) -> date:
    s = str(raw).strip()
    for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%m/%d/%y"):
        try:
            return datetime.strptime(s, fmt).date()
        except ValueError:
            continue
    return date.fromisoformat(s)


_REQUIRED_CANON = ("symbol", "qty", "date_acquired", "date_sold", "proceeds", "cost_basis")

_HEADER_ALIASES: dict[str, tuple[str, ...]] = {
    "symbol": ("symbol", "ticker"),
    "qty": ("qty", "quantity", "shares", "share_quantity"),
    "date_acquired": ("date_acquired", "acquired", "open_date", "purchase_date"),
    "date_sold": ("date_sold", "sold", "close_date", "sale_date", "date_of_sale"),
    "proceeds": ("proceeds", "sales_proceeds", "amount"),
    "cost_basis": ("cost_basis", "cost", "basis", "cost_or_basis"),
}


def _map_headers(fieldnames: list[str] | None) -> dict[str, str]:
    """Map normalized CSV header -> canonical field name."""
    if not fieldnames:
        msg = "1099-B CSV has no header row"
        raise ValueError(msg)
    norm_to_original: dict[str, str] = {}
    for fn in fieldnames:
        norm_to_original[_norm_header(fn)] = fn
    canon_to_col: dict[str, str] = {}
    for canon, aliases in _HEADER_ALIASES.items():
        for a in aliases:
            if a in norm_to_original:
                canon_to_col[canon] = norm_to_original[a]
                break
    missing = [c for c in _REQUIRED_CANON if c not in canon_to_col]
    if missing:
        msg = f"1099-B CSV missing required columns: {', '.join(missing)}"
        raise ValueError(msg)
    return canon_to_col


def parse_1099b_csv(file_path: Path | str) -> list[BrokerLotRecord]:
    """Parse Alpaca-style 1099-B CSV export into standardized records."""
    path = Path(file_path)
    with open(path, newline="", encoding="utf-8-sig") as f:
        return parse_1099b_csv_fileobj(f)


def parse_1099b_csv_fileobj(f: TextIO) -> list[BrokerLotRecord]:
    """Parse 1099-B CSV from a text stream (e.g. upload)."""
    reader = csv.DictReader(f)
    colmap = _map_headers(reader.fieldnames)
    out: list[BrokerLotRecord] = []
    for row in reader:
        if not row or not any(str(v).strip() for v in row.values() if v is not None):
            continue
        try:
            sym = str(row[colmap["symbol"]]).strip().upper()
            qty = float(str(row[colmap["qty"]]).replace(",", ""))
            da = _parse_date(str(row[colmap["date_acquired"]]))
            ds = _parse_date(str(row[colmap["date_sold"]]))
            proc = float(str(row[colmap["proceeds"]]).replace(",", "").replace("$", ""))
            basis = float(str(row[colmap["cost_basis"]]).replace(",", "").replace("$", ""))
        except (KeyError, TypeError, ValueError) as exc:
            msg = f"Invalid 1099-B CSV row: {exc}"
            raise ValueError(msg) from exc
        out.append(
            BrokerLotRecord(
                symbol=sym,
                qty=qty,
                date_acquired=da,
                date_sold=ds,
                proceeds=proc,
                cost_basis=basis,
            ),
        )
    return out


def _closed_lot_sale_date(cl: ClosedLot) -> date:
    ca = cl.closed_at
    if isinstance(ca, datetime):
        if ca.tzinfo is not None:
            return ca.date()
        return ca.date()
    return ca if isinstance(ca, date) else ca.date()


def _internal_proceeds(cl: ClosedLot) -> float:
    return float(cl.qty) * float(cl.sell_price)


def _internal_cost(cl: ClosedLot) -> float:
    return float(cl.qty) * float(cl.cost_per_share)


def reconcile_1099b(
    broker_records: list[BrokerLotRecord],
    closed_lots: list[ClosedLot],
    *,
    qty_tolerance: float = 1e-4,
    price_tolerance: float = 0.01,
) -> ReconciliationReport1099B:
    """Match broker 1099-B records to internal closed lots; flag discrepancies."""
    from collections import defaultdict

    pools: dict[tuple[str, date], list[ClosedLot]] = defaultdict(list)
    for cl in closed_lots:
        key = (cl.symbol.strip().upper(), _closed_lot_sale_date(cl))
        pools[key].append(cl)

    matched_internal_ids: set[str] = set()
    matches: list[LotMatch] = []

    for br in broker_records:
        key = (br.symbol.strip().upper(), br.date_sold)
        candidates = [cl for cl in pools.get(key, []) if cl.lot_id not in matched_internal_ids]
        qty_ok = [cl for cl in candidates if abs(float(cl.qty) - float(br.qty)) <= qty_tolerance]

        if not candidates:
            matches.append(
                LotMatch(
                    broker_record=br,
                    internal_lot=None,
                    status="missing_internal",
                    variance=float(br.proceeds) + float(br.cost_basis),
                ),
            )
            continue

        if not qty_ok:
            closest = min(candidates, key=lambda c: abs(float(c.qty) - float(br.qty)))
            var = abs(_internal_proceeds(closest) - br.proceeds) + abs(
                _internal_cost(closest) - br.cost_basis,
            )
            matches.append(
                LotMatch(
                    broker_record=br,
                    internal_lot=closest,
                    status="qty_mismatch",
                    variance=var,
                ),
            )
            matched_internal_ids.add(closest.lot_id)
            continue

        chosen = min(
            qty_ok,
            key=lambda c: (abs(float(c.qty) - float(br.qty)), c.lot_id),
        )
        ip = _internal_proceeds(chosen)
        ic = _internal_cost(chosen)
        var_p = abs(ip - br.proceeds)
        var_b = abs(ic - br.cost_basis)
        variance = var_p + var_b

        if var_p <= price_tolerance and var_b <= price_tolerance:
            status: Literal[
                "matched",
                "proceeds_mismatch",
                "basis_mismatch",
                "qty_mismatch",
                "missing_internal",
                "missing_broker",
            ] = "matched"
        elif var_p > price_tolerance and var_b > price_tolerance:
            status = "proceeds_mismatch" if var_p >= var_b else "basis_mismatch"
        elif var_p > price_tolerance:
            status = "proceeds_mismatch"
        else:
            status = "basis_mismatch"

        matches.append(
            LotMatch(
                broker_record=br,
                internal_lot=chosen,
                status=status,
                variance=variance,
            ),
        )
        matched_internal_ids.add(chosen.lot_id)

    for cl in closed_lots:
        if cl.lot_id in matched_internal_ids:
            continue
        matches.append(
            LotMatch(
                broker_record=None,
                internal_lot=cl,
                status="missing_broker",
                variance=_internal_proceeds(cl) + _internal_cost(cl),
            ),
        )

    matched_n = sum(1 for m in matches if m.status == "matched")
    disc_n = sum(1 for m in matches if m.status != "matched")

    return ReconciliationReport1099B(
        total_broker_records=len(broker_records),
        total_internal_lots=len(closed_lots),
        matched=matched_n,
        discrepancies=disc_n,
        matches=matches,
    )
