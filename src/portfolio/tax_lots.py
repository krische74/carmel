"""FIFO tax-lot ledger (cost basis) backed by SQLite."""

from __future__ import annotations

import sqlite3
import uuid
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel

if TYPE_CHECKING:
    from src.portfolio.wash_sales import WashSale


class TaxLot(BaseModel):
    """An open (or partially open) cost-basis lot."""

    id: str
    symbol: str
    qty: float
    cost_per_share: float
    opened_at: datetime
    account_id: str = "default"


class ClosedLot(BaseModel):
    """A lot (or partial lot) that has been sold."""

    lot_id: str
    symbol: str
    qty: float
    cost_per_share: float
    sell_price: float
    opened_at: datetime
    closed_at: datetime
    realized_pnl: float
    is_tlh: bool = False
    account_id: str = "default"


def _dt_to_iso(dt: datetime) -> str:
    return dt.isoformat()


def _dt_from_iso(s: str) -> datetime:
    """Parse SQLite-stored ISO timestamps (``Z`` suffix normalized)."""
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


class LotLedger:
    """FIFO lot tracking: open lots, closed lots, and P&L aggregates."""

    def __init__(self, db_path: str | Path) -> None:
        self._path = Path(db_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()
        self._migrate_tax_lots_closed_columns()
        self._migrate_tax_lots_account_id()

    def _migrate_tax_lots_account_id(self) -> None:
        """Add ``account_id`` to tax lot tables (existing DBs)."""
        with self._connect() as conn:
            for table in ("tax_lots_open", "tax_lots_closed"):
                rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
                names = {str(r[1]) for r in rows}
                if names and "account_id" not in names:
                    conn.execute(
                        f"ALTER TABLE {table} ADD COLUMN account_id TEXT NOT NULL DEFAULT 'default'",
                    )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_tax_open_account ON tax_lots_open(account_id)",
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_tax_closed_account ON tax_lots_closed(account_id)",
            )
            conn.commit()

    def _migrate_tax_lots_closed_columns(self) -> None:
        """Add ``is_tlh`` for tax-loss harvest tagging (existing DBs)."""
        with self._connect() as conn:
            rows = conn.execute("PRAGMA table_info(tax_lots_closed)").fetchall()
            names = {str(r[1]) for r in rows}
            if "is_tlh" not in names:
                conn.execute(
                    "ALTER TABLE tax_lots_closed ADD COLUMN is_tlh INTEGER NOT NULL DEFAULT 0",
                )
            conn.commit()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS tax_lots_open (
                    id TEXT PRIMARY KEY,
                    symbol TEXT NOT NULL,
                    qty REAL NOT NULL,
                    cost_per_share REAL NOT NULL,
                    opened_at TEXT NOT NULL,
                    account_id TEXT NOT NULL DEFAULT 'default'
                );
                CREATE TABLE IF NOT EXISTS tax_lots_closed (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    lot_id TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    qty REAL NOT NULL,
                    cost_per_share REAL NOT NULL,
                    sell_price REAL NOT NULL,
                    opened_at TEXT NOT NULL,
                    closed_at TEXT NOT NULL,
                    realized_pnl REAL NOT NULL,
                    account_id TEXT NOT NULL DEFAULT 'default'
                );
                CREATE INDEX IF NOT EXISTS idx_tax_open_symbol ON tax_lots_open(symbol);
                CREATE INDEX IF NOT EXISTS idx_tax_closed_symbol ON tax_lots_closed(symbol);
                """
            )
            conn.commit()

    def record_buy(
        self,
        symbol: str,
        qty: float,
        cost_per_share: float,
        opened_at: datetime,
        *,
        account_id: str = "default",
    ) -> TaxLot:
        """Insert a new open lot and return it."""
        sym = symbol.strip().upper()
        aid = account_id.strip() or "default"
        lot_id = uuid.uuid4().hex
        lot = TaxLot(
            id=lot_id,
            symbol=sym,
            qty=float(qty),
            cost_per_share=float(cost_per_share),
            opened_at=opened_at,
            account_id=aid,
        )
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO tax_lots_open(id, symbol, qty, cost_per_share, opened_at, account_id)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (lot_id, sym, lot.qty, lot.cost_per_share, _dt_to_iso(opened_at), aid),
            )
            conn.commit()
        return lot

    def record_sell(
        self,
        symbol: str,
        qty: float,
        sell_price: float,
        closed_at: datetime,
        *,
        method: Literal["fifo", "hifo"] = "fifo",
        account_id: str = "default",
    ) -> list[ClosedLot]:
        """Allocate ``qty`` across open lots (FIFO or HIFO); return one ``ClosedLot`` per tranche."""
        sym = symbol.strip().upper()
        aid = account_id.strip() or "default"
        need = float(qty)
        if need <= 0.0:
            return []

        order_sql = (
            "ORDER BY opened_at ASC"
            if method == "fifo"
            else "ORDER BY cost_per_share DESC, opened_at ASC"
        )
        closed: list[ClosedLot] = []
        with self._connect() as conn:
            rows = conn.execute(
                f"""
                SELECT id, qty, cost_per_share, opened_at FROM tax_lots_open
                WHERE symbol = ? AND account_id = ? {order_sql}
                """,
                (sym, aid),
            ).fetchall()

            total_open = sum(float(r["qty"]) for r in rows)
            if need > total_open + 1e-12:
                if total_open <= 1e-12:
                    msg = f"No open lots for {sym}; cannot sell {need} shares (0 available)."
                else:
                    msg = f"Sell qty {need} exceeds open qty {total_open} for {sym}"
                raise ValueError(msg)

            remaining = need
            for r in rows:
                if remaining <= 1e-12:
                    break
                lot_id = str(r["id"])
                open_qty = float(r["qty"])
                cps = float(r["cost_per_share"])
                opened_iso = str(r["opened_at"])
                opened = _dt_from_iso(opened_iso)
                take = min(open_qty, remaining)
                realized = take * (float(sell_price) - cps)
                cl = ClosedLot(
                    lot_id=lot_id,
                    symbol=sym,
                    qty=take,
                    cost_per_share=cps,
                    sell_price=float(sell_price),
                    opened_at=opened,
                    closed_at=closed_at,
                    realized_pnl=realized,
                    is_tlh=False,
                    account_id=aid,
                )
                closed.append(cl)
                conn.execute(
                    """
                    INSERT INTO tax_lots_closed(
                        lot_id, symbol, qty, cost_per_share, sell_price,
                        opened_at, closed_at, realized_pnl, is_tlh, account_id
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        lot_id,
                        sym,
                        take,
                        cps,
                        float(sell_price),
                        opened_iso,
                        _dt_to_iso(closed_at),
                        realized,
                        0,
                        aid,
                    ),
                )
                new_open = open_qty - take
                if new_open <= 1e-12:
                    conn.execute("DELETE FROM tax_lots_open WHERE id = ?", (lot_id,))
                else:
                    conn.execute(
                        "UPDATE tax_lots_open SET qty = ? WHERE id = ?",
                        (new_open, lot_id),
                    )
                remaining -= take
            conn.commit()

        return closed

    def record_sell_lot(
        self,
        lot_id: str,
        sell_price: float,
        closed_at: datetime,
        *,
        qty: float | None = None,
        is_tlh: bool = False,
        account_id: str = "default",
    ) -> ClosedLot:
        """Close a specific open lot by id (full lot or partial qty)."""
        lid = str(lot_id).strip()
        aid = account_id.strip() or "default"
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT id, symbol, qty, cost_per_share, opened_at FROM tax_lots_open
                WHERE id = ? AND account_id = ?
                """,
                (lid, aid),
            ).fetchone()
            if row is None:
                msg = f"No open lot with id {lid!r}"
                raise ValueError(msg)
            open_qty = float(row["qty"])
            sym = str(row["symbol"]).strip().upper()
            cps = float(row["cost_per_share"])
            opened_iso = str(row["opened_at"])
            opened = _dt_from_iso(opened_iso)
            sell_qty = float(open_qty if qty is None else qty)
            if sell_qty <= 0.0:
                msg = f"Sell qty must be positive for lot {lid}"
                raise ValueError(msg)
            if sell_qty > open_qty + 1e-9:
                msg = f"Sell qty {sell_qty} exceeds open qty {open_qty} for lot {lid}"
                raise ValueError(msg)
            take = sell_qty
            realized = take * (float(sell_price) - cps)
            tlh_flag = 1 if is_tlh else 0
            conn.execute(
                """
                INSERT INTO tax_lots_closed(
                    lot_id, symbol, qty, cost_per_share, sell_price,
                    opened_at, closed_at, realized_pnl, is_tlh, account_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    lid,
                    sym,
                    take,
                    cps,
                    float(sell_price),
                    opened_iso,
                    _dt_to_iso(closed_at),
                    realized,
                    tlh_flag,
                    aid,
                ),
            )
            new_open = open_qty - take
            if new_open <= 1e-12:
                conn.execute("DELETE FROM tax_lots_open WHERE id = ?", (lid,))
            else:
                conn.execute(
                    "UPDATE tax_lots_open SET qty = ? WHERE id = ?",
                    (new_open, lid),
                )
            conn.commit()

        return ClosedLot(
            lot_id=lid,
            symbol=sym,
            qty=take,
            cost_per_share=cps,
            sell_price=float(sell_price),
            opened_at=opened,
            closed_at=closed_at,
            realized_pnl=realized,
            is_tlh=is_tlh,
            account_id=aid,
        )

    def get_open_lots(
        self,
        symbol: str | None = None,
        *,
        account_id: str | None = None,
    ) -> list[TaxLot]:
        """Open lots, optionally filtered by symbol and/or ``account_id`` (oldest first)."""
        sql = """
            SELECT id, symbol, qty, cost_per_share, opened_at, account_id FROM tax_lots_open
            WHERE 1=1
        """
        params: list[str] = []
        if symbol is not None:
            sql += " AND symbol = ?"
            params.append(symbol.strip().upper())
        if account_id is not None:
            sql += " AND account_id = ?"
            params.append(account_id.strip() or "default")
        sql += " ORDER BY opened_at ASC"
        with self._connect() as conn:
            rows = conn.execute(sql, tuple(params)).fetchall()
        rows = list(rows)
        return [
            TaxLot(
                id=str(r["id"]),
                symbol=str(r["symbol"]),
                qty=float(r["qty"]),
                cost_per_share=float(r["cost_per_share"]),
                opened_at=_dt_from_iso(str(r["opened_at"])),
                account_id=str(r["account_id"] if r["account_id"] is not None else "default"),
            )
            for r in rows
        ]

    def clear_account(self, account_id: str = "default") -> None:
        """Delete all open and closed lots for ``account_id`` (rebuild / reset)."""
        aid = account_id.strip() or "default"
        with self._connect() as conn:
            conn.execute("DELETE FROM tax_lots_open WHERE account_id = ?", (aid,))
            conn.execute("DELETE FROM tax_lots_closed WHERE account_id = ?", (aid,))
            conn.commit()

    def prune_dust_lots(
        self,
        account_id: str = "default",
        *,
        min_qty: float = 1e-8,
    ) -> int:
        """Delete open lots with qty below ``min_qty`` (fractional FIFO residue). Returns deleted count."""
        aid = account_id.strip() or "default"
        with self._connect() as conn:
            cur = conn.execute(
                "DELETE FROM tax_lots_open WHERE account_id = ? AND qty < ?",
                (aid, float(min_qty)),
            )
            conn.commit()
            return int(cur.rowcount)

    def get_closed_lots(
        self,
        symbol: str | None = None,
        *,
        limit: int | None = None,
        account_id: str | None = None,
    ) -> list[ClosedLot]:
        """Closed lots, optionally filtered by symbol, newest first.

        When ``limit`` is set, the query uses ``LIMIT ?`` (most recent rows first).
        """
        sql = """
            SELECT lot_id, symbol, qty, cost_per_share, sell_price,
                   opened_at, closed_at, realized_pnl, COALESCE(is_tlh, 0) AS is_tlh,
                   account_id
            FROM tax_lots_closed
            WHERE 1=1
        """
        params: list[str | int] = []
        if symbol is not None:
            sql += " AND symbol = ?"
            params.append(symbol.strip().upper())
        if account_id is not None:
            sql += " AND account_id = ?"
            params.append(account_id.strip() or "default")
        sql += " ORDER BY closed_at DESC"
        if limit is not None:
            sql += " LIMIT ?"
            params.append(int(limit))
        with self._connect() as conn:
            rows = conn.execute(sql, tuple(params)).fetchall()
        return [
            ClosedLot(
                lot_id=str(r["lot_id"]),
                symbol=str(r["symbol"]),
                qty=float(r["qty"]),
                cost_per_share=float(r["cost_per_share"]),
                sell_price=float(r["sell_price"]),
                opened_at=_dt_from_iso(str(r["opened_at"])),
                closed_at=_dt_from_iso(str(r["closed_at"])),
                realized_pnl=float(r["realized_pnl"]),
                is_tlh=bool(int(r["is_tlh"])),
                account_id=str(r["account_id"] if r["account_id"] is not None else "default"),
            )
            for r in rows
        ]

    def unrealized_pnl(
        self,
        symbol: str,
        current_price: float,
        *,
        account_id: str | None = None,
    ) -> float:
        """Mark-to-market unrealized P&L for open lots of ``symbol``."""
        sym = symbol.strip().upper()
        total = 0.0
        for lot in self.get_open_lots(sym, account_id=account_id):
            total += lot.qty * (float(current_price) - lot.cost_per_share)
        return total

    def realized_pnl(self, symbol: str | None = None, *, account_id: str | None = None) -> float:
        """Sum realized P&L from closed lots."""
        lots = self.get_closed_lots(symbol, account_id=account_id)
        return sum(cl.realized_pnl for cl in lots)

    def closed_lot_aggregates(
        self,
        *,
        account_id: str | None = None,
    ) -> tuple[dict[str, float], float]:
        """Return ``(realized_pnl_by_symbol, total_closed_cost_basis)`` via SQL aggregation.

        Avoids loading every ``tax_lots_closed`` row when only per-symbol sums and total
        closed cost basis are needed (e.g. position contribution).
        """
        aid_clause = ""
        params_agg: tuple[str, ...] = ()
        if account_id is not None:
            aid_clause = " WHERE account_id = ?"
            params_agg = (account_id.strip() or "default",)
        with self._connect() as conn:
            rows = conn.execute(
                f"""
                SELECT symbol, SUM(realized_pnl) AS rp
                FROM tax_lots_closed
                {aid_clause}
                GROUP BY symbol
                """,
                params_agg,
            ).fetchall()
            total_row = conn.execute(
                f"SELECT COALESCE(SUM(qty * cost_per_share), 0.0) AS cb FROM tax_lots_closed{aid_clause}",
                params_agg,
            ).fetchone()
        by_sym = {str(r["symbol"]).strip().upper(): float(r["rp"]) for r in rows}
        total_cb = float(total_row["cb"]) if total_row is not None else 0.0
        return by_sym, total_cb

    def total_cost_basis(
        self,
        symbol: str | None = None,
        *,
        account_id: str | None = None,
    ) -> float:
        """Aggregate cost basis (qty * cost_per_share) for open lots."""
        lots = self.get_open_lots(symbol, account_id=account_id)
        return sum(lot.qty * lot.cost_per_share for lot in lots)

    def detect_wash_sales(
        self,
        account_id: str | None = None,
        *,
        window_days: int = 30,
    ) -> list[WashSale]:
        """Run wash-sale detection on this ledger's open and closed lots.

        When ``account_id`` is set, only that account's lots participate (intra-account).
        When ``None`` and multiple hub accounts exist, returns merged intra-account
        results per account (cross-account patterns use ``detect_cross_account_wash_sales``).
        """
        from src.portfolio.wash_sales import detect_wash_sales

        if account_id is not None:
            return detect_wash_sales(
                self.get_closed_lots(account_id=account_id),
                self.get_open_lots(account_id=account_id),
                window_days=window_days,
            )
        closed = self.get_closed_lots(account_id=None)
        open_lots = self.get_open_lots(account_id=None)
        aids = {
            str(getattr(x, "account_id", None) or "default").strip() or "default"
            for x in (*closed, *open_lots)
        }
        if len(aids) <= 1:
            return detect_wash_sales(closed, open_lots, window_days=window_days)
        merged: list[WashSale] = []
        for aid in sorted(aids):
            merged.extend(
                detect_wash_sales(
                    self.get_closed_lots(account_id=aid),
                    self.get_open_lots(account_id=aid),
                    window_days=window_days,
                ),
            )
        merged.sort(key=lambda w: w.sell_date)
        return merged
