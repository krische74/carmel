"""Tier 31: 1099-B CSV parse and reconciliation."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path  # noqa: TC003

import pytest

from src.portfolio.tax_lots import ClosedLot
from src.reporting.reconciliation_1099b import (
    BrokerLotRecord,
    parse_1099b_csv,
    reconcile_1099b,
)


def _sample_csv(tmp_path: Path) -> Path:
    p = tmp_path / "b1099.csv"
    p.write_text(
        "Symbol,Qty,Date Acquired,Date Sold,Proceeds,Cost Basis\n"
        "SPY,1.0,2026-01-01,2026-06-01,110.00,100.00\n",
        encoding="utf-8",
    )
    return p


def test_parse_1099b_csv(tmp_path: Path) -> None:
    p = _sample_csv(tmp_path)
    rows = parse_1099b_csv(p)
    assert len(rows) == 1
    assert rows[0].symbol == "SPY"
    assert rows[0].qty == pytest.approx(1.0)
    assert rows[0].proceeds == pytest.approx(110.0)
    assert rows[0].cost_basis == pytest.approx(100.0)


def test_parse_1099b_csv_missing_columns(tmp_path: Path) -> None:
    p = tmp_path / "bad.csv"
    p.write_text("Symbol,Qty\nSPY,1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing required"):
        parse_1099b_csv(p)


def test_parse_1099b_csv_skips_empty_rows(tmp_path: Path) -> None:
    p = tmp_path / "blank_rows.csv"
    p.write_text(
        "Symbol,Qty,Date Acquired,Date Sold,Proceeds,Cost Basis\n"
        "SPY,1.0,2026-01-01,2026-06-01,110.00,100.00\n"
        "\n"
        "  \n"
        "QQQ,2.0,2026-02-01,2026-07-01,220.00,200.00\n"
        "\n",
        encoding="utf-8",
    )
    rows = parse_1099b_csv(p)
    assert len(rows) == 2
    assert {r.symbol for r in rows} == {"SPY", "QQQ"}


def test_parse_1099b_csv_bom_prefix(tmp_path: Path) -> None:
    p = tmp_path / "bom.csv"
    body = (
        "Symbol,Qty,Date Acquired,Date Sold,Proceeds,Cost Basis\n"
        "SPY,1.0,2026-01-01,2026-06-01,110.00,100.00\n"
    )
    p.write_bytes("\ufeff".encode("utf-8") + body.encode("utf-8"))
    rows = parse_1099b_csv(p)
    assert len(rows) == 1
    assert rows[0].symbol == "SPY"


def _cl(
    lid: str,
    sym: str,
    qty: float,
    cps: float,
    sp: float,
    o: datetime,
    c: datetime,
) -> ClosedLot:
    return ClosedLot(
        lot_id=lid,
        symbol=sym,
        qty=qty,
        cost_per_share=cps,
        sell_price=sp,
        opened_at=o,
        closed_at=c,
        realized_pnl=qty * (sp - cps),
    )


def test_reconcile_all_matched() -> None:
    o = datetime(2026, 1, 1, tzinfo=UTC)
    c = datetime(2026, 6, 1, tzinfo=UTC)
    br = BrokerLotRecord(
        symbol="SPY",
        qty=1.0,
        date_acquired=o.date(),
        date_sold=c.date(),
        proceeds=110.0,
        cost_basis=100.0,
    )
    cl = _cl("L1", "SPY", 1.0, 100.0, 110.0, o, c)
    rep = reconcile_1099b([br], [cl])
    assert rep.matched == 1
    assert rep.discrepancies == 0
    assert rep.matches[0].status == "matched"


def test_reconcile_basis_mismatch_proceeds_match() -> None:
    """Broker cost basis differs while proceeds and qty align (H2)."""
    o = datetime(2026, 1, 1, tzinfo=UTC)
    c = datetime(2026, 6, 1, tzinfo=UTC)
    br = BrokerLotRecord(
        symbol="SPY",
        qty=1.0,
        date_acquired=o.date(),
        date_sold=c.date(),
        proceeds=110.0,
        cost_basis=99.0,
    )
    cl = _cl("L1", "SPY", 1.0, 100.0, 110.0, o, c)
    rep = reconcile_1099b([br], [cl], price_tolerance=0.01)
    assert rep.matched == 0
    m = next(x for x in rep.matches if x.broker_record is not None)
    assert m.status == "basis_mismatch"
    assert m.internal_lot is not None
    assert m.internal_lot.lot_id == "L1"


def test_reconcile_proceeds_mismatch() -> None:
    o = datetime(2026, 1, 1, tzinfo=UTC)
    c = datetime(2026, 6, 1, tzinfo=UTC)
    br = BrokerLotRecord(
        symbol="SPY",
        qty=1.0,
        date_acquired=o.date(),
        date_sold=c.date(),
        proceeds=999.0,
        cost_basis=100.0,
    )
    cl = _cl("L1", "SPY", 1.0, 100.0, 110.0, o, c)
    rep = reconcile_1099b([br], [cl], price_tolerance=0.01)
    assert rep.matched == 0
    m = next(x for x in rep.matches if x.broker_record is not None)
    assert m.status == "proceeds_mismatch"


def test_reconcile_missing_internal() -> None:
    br = BrokerLotRecord(
        symbol="QQQ",
        qty=1.0,
        date_acquired=datetime(2026, 1, 1, tzinfo=UTC).date(),
        date_sold=datetime(2026, 6, 1, tzinfo=UTC).date(),
        proceeds=50.0,
        cost_basis=40.0,
    )
    rep = reconcile_1099b([br], [])
    assert rep.matches[0].status == "missing_internal"


def test_reconcile_missing_broker() -> None:
    o = datetime(2026, 1, 1, tzinfo=UTC)
    c = datetime(2026, 6, 1, tzinfo=UTC)
    cl = _cl("L9", "IWM", 2.0, 50.0, 55.0, o, c)
    rep = reconcile_1099b([], [cl])
    assert any(m.status == "missing_broker" for m in rep.matches)


def test_reconcile_qty_mismatch() -> None:
    o = datetime(2026, 2, 1, tzinfo=UTC)
    c = datetime(2026, 7, 1, tzinfo=UTC)
    br = BrokerLotRecord(
        symbol="GLD",
        qty=1.0,
        date_acquired=o.date(),
        date_sold=c.date(),
        proceeds=100.0,
        cost_basis=90.0,
    )
    cl = _cl("L2", "GLD", 5.0, 18.0, 20.0, o, c)
    rep = reconcile_1099b([br], [cl], qty_tolerance=1e-4)
    m = next(x for x in rep.matches if x.broker_record is not None)
    assert m.status == "qty_mismatch"


def test_reconcile_picks_closest_qty_when_multiple_internal_match() -> None:
    """Among lots with same (symbol, date_sold) and qty within tolerance, pick closest qty (M3)."""
    o = datetime(2026, 3, 1, tzinfo=UTC)
    c = datetime(2026, 8, 1, tzinfo=UTC)
    br = BrokerLotRecord(
        symbol="DIA",
        qty=1.0,
        date_acquired=o.date(),
        date_sold=c.date(),
        proceeds=100.0,
        cost_basis=90.0,
    )
    farther = _cl("L_far", "DIA", 1.00008, 90.0, 100.0, o, c)
    exact = _cl("L_exact", "DIA", 1.0, 90.0, 100.0, o, c)
    rep = reconcile_1099b([br], [farther, exact], qty_tolerance=1e-4, price_tolerance=0.01)
    m = next(x for x in rep.matches if x.status == "matched")
    assert m.internal_lot is not None
    assert m.internal_lot.lot_id == "L_exact"
    far_rep = next(x for x in rep.matches if x.status == "missing_broker")
    assert far_rep.internal_lot is not None
    assert far_rep.internal_lot.lot_id == "L_far"
