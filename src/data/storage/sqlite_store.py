"""SQLite metadata: symbols, last fetch times, data source registry, trade log."""

from __future__ import annotations

import json
import logging
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.automation.alerts import Alert
    from src.backtesting.engine import BacktestConfig, BacktestResult
    from src.data.regime import MarketRegime
    from src.models import OrderExecutionResult, Signal
    from src.reporting.returns import ReturnMetrics

logger = logging.getLogger(__name__)


class SQLiteStore:
    """Persist ingestion metadata for dashboards and pipeline idempotency."""

    def __init__(self, db_path: str | Path) -> None:
        self._path = Path(db_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()
        self._migrate_trade_executions_columns()
        self._migrate_backtest_trades_column()
        self._migrate_backtest_tier50_columns()
        self._migrate_backtest_tier53_columns()
        self._migrate_signal_explanation_column()
        self._migrate_execution_strategy_name()
        self._migrate_hub_account_id()
        self._migrate_equity_snapshots_add_cash()
        self._migrate_equity_snapshots_add_broker_equity()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS symbols (
                    symbol TEXT PRIMARY KEY NOT NULL,
                    source TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT (datetime('now'))
                );
                CREATE TABLE IF NOT EXISTS last_fetch (
                    symbol TEXT PRIMARY KEY NOT NULL,
                    fetched_at TEXT NOT NULL,
                    FOREIGN KEY(symbol) REFERENCES symbols(symbol)
                );
                CREATE TABLE IF NOT EXISTS data_sources (
                    id TEXT PRIMARY KEY NOT NULL,
                    description TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS trade_signals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    cycle_id TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    direction TEXT NOT NULL,
                    weight REAL NOT NULL,
                    confidence REAL NOT NULL,
                    strategy_name TEXT,
                    rationale TEXT NOT NULL,
                    account_id TEXT NOT NULL DEFAULT 'default'
                );
                CREATE TABLE IF NOT EXISTS trade_executions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    cycle_id TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    submitted INTEGER NOT NULL,
                    order_id TEXT,
                    reason TEXT,
                    side TEXT NOT NULL DEFAULT 'buy',
                    qty REAL,
                    filled_avg_price REAL,
                    strategy_name TEXT,
                    order_type TEXT DEFAULT 'market',
                    limit_price REAL,
                    stop_price REAL,
                    order_status TEXT DEFAULT 'filled',
                    filled_qty REAL,
                    account_id TEXT NOT NULL DEFAULT 'default'
                );
                CREATE INDEX IF NOT EXISTS idx_signals_cycle ON trade_signals(cycle_id);
                CREATE INDEX IF NOT EXISTS idx_executions_cycle ON trade_executions(cycle_id);
                CREATE TABLE IF NOT EXISTS submitted_client_order_ids (
                    client_order_id TEXT PRIMARY KEY,
                    submitted_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS equity_snapshots (
                    account_id TEXT NOT NULL DEFAULT 'default',
                    date TEXT NOT NULL,
                    total_market_value REAL NOT NULL,
                    total_cost_basis REAL NOT NULL,
                    unrealized_pnl REAL NOT NULL,
                    realized_pnl REAL NOT NULL,
                    total_pnl REAL NOT NULL,
                    cash REAL,
                    broker_equity REAL,
                    PRIMARY KEY (account_id, date)
                );
                CREATE TABLE IF NOT EXISTS alerts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    level TEXT NOT NULL,
                    category TEXT NOT NULL,
                    message TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS backtest_runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_at TEXT NOT NULL,
                    strategy_name TEXT NOT NULL,
                    start_date TEXT NOT NULL,
                    end_date TEXT NOT NULL,
                    initial_capital REAL NOT NULL,
                    final_equity REAL NOT NULL,
                    slippage_bps REAL NOT NULL,
                    rebalance_frequency TEXT NOT NULL,
                    total_return_pct REAL NOT NULL,
                    cagr_pct REAL NOT NULL,
                    sharpe_ratio REAL,
                    sortino_ratio REAL,
                    max_drawdown_pct REAL NOT NULL,
                    calmar_ratio REAL,
                    annual_volatility_pct REAL NOT NULL,
                    trade_count INTEGER NOT NULL,
                    equity_curve_json TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS macro_indicators (
                    date TEXT NOT NULL,
                    series_id TEXT NOT NULL,
                    value REAL NOT NULL,
                    fetched_at TEXT NOT NULL,
                    PRIMARY KEY (date, series_id)
                );
                CREATE TABLE IF NOT EXISTS market_regime (
                    date TEXT PRIMARY KEY NOT NULL,
                    overall TEXT NOT NULL,
                    yield_curve TEXT NOT NULL,
                    volatility TEXT NOT NULL,
                    vix_close REAL,
                    yield_spread REAL,
                    sizing_multiplier REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS fundamental_scores (
                    symbol TEXT NOT NULL,
                    period TEXT NOT NULL,
                    score INTEGER NOT NULL,
                    data_json TEXT NOT NULL,
                    fetched_at TEXT NOT NULL,
                    PRIMARY KEY (symbol, period)
                );
                CREATE INDEX IF NOT EXISTS idx_macro_series_date
                    ON macro_indicators(series_id, date);
                CREATE TABLE IF NOT EXISTS execution_quality (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    execution_id INTEGER NOT NULL UNIQUE,
                    cycle_id TEXT NOT NULL,
                    execution_ts TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    side TEXT NOT NULL,
                    qty REAL,
                    fill_price REAL,
                    reference_price REAL,
                    slippage_bps REAL,
                    note TEXT
                );
                CREATE TABLE IF NOT EXISTS rag_chunks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source_path TEXT NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    text TEXT NOT NULL,
                    embedding_json TEXT NOT NULL,
                    indexed_at TEXT NOT NULL,
                    UNIQUE(source_path, chunk_index)
                );
                CREATE TABLE IF NOT EXISTS ml_scores (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    cycle_id TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    score REAL NOT NULL,
                    model_version TEXT NOT NULL,
                    computed_at TEXT NOT NULL,
                    top_features_json TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_ml_scores_cycle ON ml_scores(cycle_id);
                CREATE INDEX IF NOT EXISTS idx_ml_scores_computed ON ml_scores(computed_at);
                CREATE TABLE IF NOT EXISTS loss_carryforward (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    year INTEGER NOT NULL,
                    account_id TEXT NOT NULL DEFAULT 'default',
                    short_term_carryforward REAL NOT NULL DEFAULT 0.0,
                    long_term_carryforward REAL NOT NULL DEFAULT 0.0,
                    computed_at TEXT NOT NULL,
                    UNIQUE(year, account_id)
                );
                CREATE INDEX IF NOT EXISTS idx_loss_cf_year ON loss_carryforward(year, account_id);
                CREATE TABLE IF NOT EXISTS chat_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    question TEXT NOT NULL,
                    answer TEXT NOT NULL,
                    asked_at TEXT NOT NULL,
                    account_id TEXT NOT NULL DEFAULT 'default'
                );
                CREATE INDEX IF NOT EXISTS idx_chat_history_account ON chat_history(account_id, id);
                """
            )

    def _migrate_trade_executions_columns(self) -> None:
        """Add columns for reconciliation, order type, and status (existing DBs)."""
        with self._connect() as conn:
            rows = conn.execute("PRAGMA table_info(trade_executions)").fetchall()
            names = {str(r[1]) for r in rows}
            if "qty" not in names:
                conn.execute("ALTER TABLE trade_executions ADD COLUMN qty REAL")
            if "filled_avg_price" not in names:
                conn.execute("ALTER TABLE trade_executions ADD COLUMN filled_avg_price REAL")
            if "order_type" not in names:
                conn.execute(
                    "ALTER TABLE trade_executions ADD COLUMN order_type TEXT DEFAULT 'market'",
                )
            if "limit_price" not in names:
                conn.execute("ALTER TABLE trade_executions ADD COLUMN limit_price REAL")
            if "stop_price" not in names:
                conn.execute("ALTER TABLE trade_executions ADD COLUMN stop_price REAL")
            if "order_status" not in names:
                conn.execute(
                    "ALTER TABLE trade_executions ADD COLUMN order_status TEXT DEFAULT 'filled'",
                )
            if "filled_qty" not in names:
                conn.execute("ALTER TABLE trade_executions ADD COLUMN filled_qty REAL")
            conn.commit()

    def _migrate_backtest_trades_column(self) -> None:
        """Add ``trades_json`` to ``backtest_runs`` (existing DBs)."""
        with self._connect() as conn:
            rows = conn.execute("PRAGMA table_info(backtest_runs)").fetchall()
            names = {str(r[1]) for r in rows}
            if "trades_json" not in names:
                conn.execute("ALTER TABLE backtest_runs ADD COLUMN trades_json TEXT")
            conn.commit()

    def _migrate_backtest_tier50_columns(self) -> None:
        """Add Tier 50 mode, cash yield, and benchmark columns to ``backtest_runs``."""
        additions = {
            "sizing_mode": "TEXT",
            "cash_yield_annual_pct": "REAL",
            "benchmark_symbol": "TEXT",
            "benchmark_metrics_json": "TEXT",
        }
        with self._connect() as conn:
            rows = conn.execute("PRAGMA table_info(backtest_runs)").fetchall()
            names = {str(r[1]) for r in rows}
            for col, typ in additions.items():
                if col not in names:
                    conn.execute(f"ALTER TABLE backtest_runs ADD COLUMN {col} {typ}")
            conn.commit()

    def _migrate_backtest_tier53_columns(self) -> None:
        """Add Tier 53 config snapshot, experiment label, and turnover columns."""
        additions = {
            "config_json": "TEXT",
            "experiment_label": "TEXT",
            "turnover": "REAL",
        }
        with self._connect() as conn:
            rows = conn.execute("PRAGMA table_info(backtest_runs)").fetchall()
            names = {str(r[1]) for r in rows}
            for col, typ in additions.items():
                if col not in names:
                    conn.execute(f"ALTER TABLE backtest_runs ADD COLUMN {col} {typ}")
            conn.commit()

    def _migrate_signal_explanation_column(self) -> None:
        """Add ``explanation`` to ``trade_signals`` (existing DBs)."""
        with self._connect() as conn:
            rows = conn.execute("PRAGMA table_info(trade_signals)").fetchall()
            names = {str(r[1]) for r in rows}
            if "explanation" not in names:
                conn.execute("ALTER TABLE trade_signals ADD COLUMN explanation TEXT")
            conn.commit()

    def _migrate_execution_strategy_name(self) -> None:
        """Add ``strategy_name`` to ``trade_executions`` (TLH / attribution)."""
        with self._connect() as conn:
            rows = conn.execute("PRAGMA table_info(trade_executions)").fetchall()
            names = {str(r[1]) for r in rows}
            if "strategy_name" not in names:
                conn.execute("ALTER TABLE trade_executions ADD COLUMN strategy_name TEXT")
            conn.commit()

    def _migrate_hub_account_id(self) -> None:
        """Add ``account_id`` to hub tables; rebuild ``equity_snapshots`` for composite PK."""
        with self._connect() as conn:
            for table in ("trade_signals", "trade_executions"):
                rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
                names = {str(r[1]) for r in rows}
                if names and "account_id" not in names:
                    conn.execute(
                        f"ALTER TABLE {table} ADD COLUMN account_id TEXT NOT NULL DEFAULT 'default'",
                    )
            self._migrate_equity_snapshots_account_composite(conn)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_signals_account ON trade_signals(account_id)")
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_executions_account ON trade_executions(account_id)",
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_equity_snapshots_account ON equity_snapshots(account_id)",
            )
            conn.commit()

    def _migrate_equity_snapshots_account_composite(self, conn: sqlite3.Connection) -> None:
        """Migrate legacy ``equity_snapshots`` (PK = date) to (account_id, date)."""
        row = conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name='equity_snapshots'",
        ).fetchone()
        if row is None:
            return
        ddl = str(row[0] or "")
        norm = "".join(ddl.split())
        if "PRIMARYKEY(account_id,date)" in norm or "PRIMARYKEY(date,account_id)" in norm:
            return
        conn.execute("ALTER TABLE equity_snapshots RENAME TO equity_snapshots_legacy_t30")
        conn.executescript(
            """
            CREATE TABLE equity_snapshots (
                account_id TEXT NOT NULL DEFAULT 'default',
                date TEXT NOT NULL,
                total_market_value REAL NOT NULL,
                total_cost_basis REAL NOT NULL,
                unrealized_pnl REAL NOT NULL,
                realized_pnl REAL NOT NULL,
                total_pnl REAL NOT NULL,
                PRIMARY KEY (account_id, date)
            );
            INSERT INTO equity_snapshots(
                account_id, date, total_market_value, total_cost_basis,
                unrealized_pnl, realized_pnl, total_pnl
            )
            SELECT 'default', date, total_market_value, total_cost_basis,
                   unrealized_pnl, realized_pnl, total_pnl
            FROM equity_snapshots_legacy_t30;
            DROP TABLE equity_snapshots_legacy_t30;
            """,
        )

    def _migrate_equity_snapshots_add_cash(self) -> None:
        """Add nullable ``cash`` to ``equity_snapshots`` (Tier 46). NULL = unknown."""
        with self._connect() as conn:
            rows = conn.execute("PRAGMA table_info(equity_snapshots)").fetchall()
            names = {str(r[1]) for r in rows}
            if names and "cash" not in names:
                conn.execute("ALTER TABLE equity_snapshots ADD COLUMN cash REAL")
            conn.commit()

    def _migrate_equity_snapshots_add_broker_equity(self) -> None:
        """Add nullable ``broker_equity`` to ``equity_snapshots`` (Tier 47C-2). NULL = unknown."""
        with self._connect() as conn:
            rows = conn.execute("PRAGMA table_info(equity_snapshots)").fetchall()
            names = {str(r[1]) for r in rows}
            if names and "broker_equity" not in names:
                conn.execute("ALTER TABLE equity_snapshots ADD COLUMN broker_equity REAL")
            conn.commit()

    def register_symbol(self, symbol: str, *, source: str) -> None:
        """Insert or replace a tracked symbol and its upstream source."""
        sym = symbol.strip().upper()
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO symbols(symbol, source) VALUES (?, ?) "
                "ON CONFLICT(symbol) DO UPDATE SET source = excluded.source",
                (sym, source),
            )
            conn.commit()

    def record_last_fetch(self, symbol: str, fetched_at_iso: str) -> None:
        """Record the last successful fetch timestamp (ISO-8601 string)."""
        sym = symbol.strip().upper()
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO last_fetch(symbol, fetched_at) VALUES (?, ?) "
                "ON CONFLICT(symbol) DO UPDATE SET fetched_at = excluded.fetched_at",
                (sym, fetched_at_iso),
            )
            conn.commit()

    def get_last_fetch(self, symbol: str) -> str | None:
        """Return last fetch ISO timestamp or ``None``."""
        sym = symbol.strip().upper()
        with self._connect() as conn:
            row = conn.execute(
                "SELECT fetched_at FROM last_fetch WHERE symbol = ?",
                (sym,),
            ).fetchone()
        return str(row["fetched_at"]) if row else None

    def list_symbols(self) -> list[str]:
        """Return tracked symbols sorted alphabetically."""
        with self._connect() as conn:
            rows = conn.execute("SELECT symbol FROM symbols ORDER BY symbol").fetchall()
        return [str(r["symbol"]) for r in rows]

    def register_data_source(self, source_id: str, description: str) -> None:
        """Register a human-readable description for a source id."""
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO data_sources(id, description) VALUES (?, ?) "
                "ON CONFLICT(id) DO UPDATE SET description = excluded.description",
                (source_id, description),
            )
            conn.commit()

    def list_data_sources(self) -> list[dict[str, Any]]:
        """Return all registered data sources."""
        with self._connect() as conn:
            rows = conn.execute("SELECT id, description FROM data_sources ORDER BY id").fetchall()
        return [{"id": r["id"], "description": r["description"]} for r in rows]

    def log_signal(
        self,
        cycle_id: str,
        signal: Signal,
        *,
        explanation: str | None = None,
        account_id: str = "default",
    ) -> None:
        """Append one strategy signal for a trading cycle (append-only)."""
        sig = signal
        ts = sig.timestamp.isoformat()
        aid = account_id.strip() or "default"
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO trade_signals(
                    cycle_id, timestamp, symbol, direction, weight, confidence,
                    strategy_name, rationale, explanation, account_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    cycle_id,
                    ts,
                    sig.symbol.strip().upper(),
                    sig.direction,
                    float(sig.weight),
                    float(sig.confidence),
                    sig.strategy_name,
                    sig.rationale,
                    explanation,
                    aid,
                ),
            )
            conn.commit()

    def log_execution(
        self,
        cycle_id: str,
        result: OrderExecutionResult,
        *,
        account_id: str = "default",
    ) -> int | None:
        """Append one order execution outcome (append-only). Returns new row ``id`` or ``None``."""
        r = result
        ts = r.timestamp.isoformat()
        aid = account_id.strip() or "default"
        with self._connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO trade_executions(
                    cycle_id, timestamp, symbol, submitted, order_id, reason, side,
                    qty, filled_avg_price, strategy_name,
                    order_type, limit_price, stop_price, order_status, filled_qty,
                    account_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    cycle_id,
                    ts,
                    r.symbol.strip().upper(),
                    1 if r.submitted else 0,
                    r.order_id,
                    r.reason,
                    r.side,
                    float(r.qty) if r.qty is not None else None,
                    float(r.fill_price) if r.fill_price is not None else None,
                    r.strategy_name,
                    r.order_type,
                    float(r.limit_price) if r.limit_price is not None else None,
                    float(r.stop_price) if r.stop_price is not None else None,
                    r.order_status,
                    float(r.filled_qty) if r.filled_qty is not None else None,
                    aid,
                ),
            )
            conn.commit()
            rid = cur.lastrowid
            return int(rid) if rid else None

    def has_recent_client_order_id(self, cid: str, *, since_minutes: int = 60) -> bool:
        """Return True if ``cid`` was recorded as submitted within ``since_minutes``.

        Backstop against duplicate submissions when the broker fails to dedupe by
        ``client_order_id`` (observed on Alpaca paper). Window is wide enough to
        cover retries and clock skew.
        """
        c = (cid or "").strip()
        if not c:
            return False
        cutoff = (datetime.now(UTC) - timedelta(minutes=max(1, int(since_minutes)))).isoformat()
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT 1 FROM submitted_client_order_ids
                WHERE client_order_id = ? AND submitted_at >= ?
                LIMIT 1
                """,
                (c, cutoff),
            ).fetchone()
        return row is not None

    def record_submitted_client_order_id(self, cid: str) -> None:
        """Record that ``cid`` was just submitted to the broker. Idempotent on the cid."""
        c = (cid or "").strip()
        if not c:
            return
        now = datetime.now(UTC).isoformat()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO submitted_client_order_ids(client_order_id, submitted_at)
                VALUES(?, ?)
                ON CONFLICT(client_order_id) DO UPDATE SET submitted_at = excluded.submitted_at
                """,
                (c, now),
            )
            conn.commit()

    def count_signals(self, *, account_id: str | None = None) -> int:
        """Total rows in ``trade_signals``, optionally scoped to ``account_id``."""
        sql = "SELECT COUNT(*) AS n FROM trade_signals WHERE 1=1"
        params: list[str] = []
        if account_id is not None:
            sql += " AND account_id = ?"
            params.append(account_id.strip() or "default")
        with self._connect() as conn:
            row = conn.execute(sql, params).fetchone()
        return int(row[0]) if row is not None else 0

    def count_executions(self, *, account_id: str | None = None) -> int:
        """Total rows in ``trade_executions``, optionally scoped to ``account_id``."""
        sql = "SELECT COUNT(*) AS n FROM trade_executions WHERE 1=1"
        params: list[str] = []
        if account_id is not None:
            sql += " AND account_id = ?"
            params.append(account_id.strip() or "default")
        with self._connect() as conn:
            row = conn.execute(sql, params).fetchone()
        return int(row[0]) if row is not None else 0

    def count_alerts(self) -> int:
        """Total rows in ``alerts``."""
        with self._connect() as conn:
            row = conn.execute("SELECT COUNT(*) AS n FROM alerts").fetchone()
        return int(row[0]) if row is not None else 0

    def count_backtest_runs(self) -> int:
        """Total rows in ``backtest_runs``."""
        with self._connect() as conn:
            row = conn.execute("SELECT COUNT(*) AS n FROM backtest_runs").fetchone()
        return int(row[0]) if row is not None else 0

    def count_open_tax_lots(self, *, account_id: str | None = None) -> int:
        """Count rows in ``tax_lots_open`` (same hub DB as the ledger), if the table exists."""
        with self._connect() as conn:
            try:
                if account_id is not None:
                    aid = account_id.strip() or "default"
                    row = conn.execute(
                        "SELECT COUNT(*) AS n FROM tax_lots_open WHERE account_id = ?",
                        (aid,),
                    ).fetchone()
                else:
                    row = conn.execute("SELECT COUNT(*) AS n FROM tax_lots_open").fetchone()
            except sqlite3.OperationalError:
                return 0
        return int(row["n"]) if row is not None else 0

    def log_chat(self, question: str, answer: str, *, account_id: str = "default") -> None:
        """Persist one Ask Hub Q&A row."""
        ts = datetime.now(UTC).isoformat()
        aid = account_id.strip() or "default"
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO chat_history(question, answer, asked_at, account_id)
                VALUES (?, ?, ?, ?)
                """,
                (str(question), str(answer), ts, aid),
            )
            conn.commit()

    def get_chat_history(
        self,
        *,
        limit: int = 50,
        account_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Recent chat rows, oldest first (last ``limit`` messages)."""
        lim = max(1, min(int(limit), 500))
        with self._connect() as conn:
            if account_id is None:
                rows = conn.execute(
                    """
                    SELECT id, question, answer, asked_at, account_id
                    FROM chat_history
                    ORDER BY id DESC
                    LIMIT ?
                    """,
                    (lim,),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT id, question, answer, asked_at, account_id
                    FROM chat_history
                    WHERE account_id = ?
                    ORDER BY id DESC
                    LIMIT ?
                    """,
                    (account_id.strip() or "default", lim),
                ).fetchall()
        out = [_row_to_dict(r) for r in reversed(rows)]
        return out

    def get_loss_carryforward_row(
        self,
        year: int,
        *,
        account_id: str = "default",
    ) -> dict[str, Any] | None:
        """Return stored carryforward snapshot for a tax year and hub account, if any."""
        aid = account_id.strip() or "default"
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT id, year, account_id, short_term_carryforward,
                       long_term_carryforward, computed_at
                FROM loss_carryforward
                WHERE year = ? AND account_id = ?
                """,
                (int(year), aid),
            ).fetchone()
        return _row_to_dict(row) if row else None

    def list_loss_carryforward_rows(
        self,
        *,
        account_id: str | None = None,
        limit: int = 30,
    ) -> list[dict[str, Any]]:
        """Recent carryforward snapshots, newest tax year first."""
        lim = max(1, min(int(limit), 200))
        with self._connect() as conn:
            if account_id is not None:
                aid = account_id.strip() or "default"
                rows = conn.execute(
                    """
                    SELECT id, year, account_id, short_term_carryforward,
                           long_term_carryforward, computed_at
                    FROM loss_carryforward
                    WHERE account_id = ?
                    ORDER BY year DESC
                    LIMIT ?
                    """,
                    (aid, lim),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT id, year, account_id, short_term_carryforward,
                           long_term_carryforward, computed_at
                    FROM loss_carryforward
                    ORDER BY year DESC, account_id
                    LIMIT ?
                    """,
                    (lim,),
                ).fetchall()
        return [_row_to_dict(r) for r in rows]

    def upsert_loss_carryforward(
        self,
        year: int,
        *,
        account_id: str = "default",
        short_term_carryforward: float,
        long_term_carryforward: float,
        computed_at: str | None = None,
    ) -> None:
        """Insert or replace carryforward amounts computed for a tax year."""
        aid = account_id.strip() or "default"
        ts = computed_at or datetime.now(UTC).isoformat()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO loss_carryforward(
                    year, account_id, short_term_carryforward,
                    long_term_carryforward, computed_at
                ) VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(year, account_id) DO UPDATE SET
                    short_term_carryforward = excluded.short_term_carryforward,
                    long_term_carryforward = excluded.long_term_carryforward,
                    computed_at = excluded.computed_at
                """,
                (
                    int(year),
                    aid,
                    float(short_term_carryforward),
                    float(long_term_carryforward),
                    ts,
                ),
            )
            conn.commit()

    def count_equity_snapshots(
        self,
        start: str | None = None,
        end: str | None = None,
        *,
        account_id: str | None = None,
    ) -> int:
        """Count snapshot rows, using the same date filters as :meth:`get_equity_snapshots`."""
        sql = "SELECT COUNT(*) AS n FROM equity_snapshots WHERE 1=1"
        params: list[Any] = []
        if account_id is not None:
            sql += " AND account_id = ?"
            params.append(account_id.strip() or "default")
        if start is not None:
            sql += " AND date >= ?"
            params.append(start.strip())
        if end is not None:
            sql += " AND date <= ?"
            params.append(end.strip())
        with self._connect() as conn:
            row = conn.execute(sql, params).fetchone()
        return int(row[0]) if row is not None else 0

    def get_signals(
        self,
        *,
        limit: int = 100,
        offset: int = 0,
        account_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Recent signals, newest first."""
        lim = max(1, min(int(limit), 10_000))
        off = max(0, int(offset))
        sql = """
                SELECT cycle_id, timestamp, symbol, direction, weight, confidence,
                       strategy_name, rationale, explanation, account_id
                FROM trade_signals
                WHERE 1=1
        """
        params: list[str | int] = []
        if account_id is not None:
            sql += " AND account_id = ?"
            params.append(account_id.strip() or "default")
        sql += " ORDER BY id DESC LIMIT ? OFFSET ?"
        params.extend([lim, off])
        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [_row_to_dict(r) for r in rows]

    def get_executions(
        self,
        *,
        limit: int = 100,
        offset: int = 0,
        account_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Recent executions, newest first."""
        lim = max(1, min(int(limit), 10_000))
        off = max(0, int(offset))
        sql = """
                SELECT cycle_id, timestamp, symbol, submitted, order_id, reason, side,
                       qty, filled_avg_price, strategy_name,
                       order_type, limit_price, stop_price, order_status, filled_qty,
                       account_id
                FROM trade_executions
                WHERE 1=1
        """
        params: list[str | int] = []
        if account_id is not None:
            sql += " AND account_id = ?"
            params.append(account_id.strip() or "default")
        sql += " ORDER BY id DESC LIMIT ? OFFSET ?"
        params.extend([lim, off])
        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [_row_to_dict(r) for r in rows]

    def get_executions_chronological(
        self,
        *,
        since_timestamp: str | None = None,
        account_id: str | None = None,
        submitted_only: bool = True,
    ) -> list[dict[str, Any]]:
        """Executions oldest-first (for lot-ledger rebuild / replay)."""
        sql = """
                SELECT cycle_id, timestamp, symbol, submitted, order_id, reason, side,
                       qty, filled_avg_price, strategy_name,
                       order_type, limit_price, stop_price, order_status, filled_qty,
                       account_id
                FROM trade_executions
                WHERE 1=1
        """
        params: list[str | int] = []
        if submitted_only:
            sql += " AND submitted = 1"
        if since_timestamp is not None:
            sql += " AND timestamp >= ?"
            params.append(str(since_timestamp))
        if account_id is not None:
            sql += " AND account_id = ?"
            params.append(account_id.strip() or "default")
        sql += " ORDER BY timestamp ASC, id ASC"
        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [_row_to_dict(r) for r in rows]

    def delete_equity_snapshots(
        self,
        *,
        account_id: str = "default",
        start: str | None = None,
        end: str | None = None,
    ) -> int:
        """Delete equity snapshot rows for an account (optional date bounds). Returns rowcount."""
        aid = account_id.strip() or "default"
        sql = "DELETE FROM equity_snapshots WHERE account_id = ?"
        params: list[str] = [aid]
        if start is not None:
            sql += " AND date >= ?"
            params.append(start.strip())
        if end is not None:
            sql += " AND date <= ?"
            params.append(end.strip())
        with self._connect() as conn:
            cur = conn.execute(sql, params)
            conn.commit()
            return int(cur.rowcount)

    def list_stale_pending_executions(
        self,
        *,
        before_timestamp_iso: str,
        account_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Broker orders still marked pending in SQLite and older than ``before_timestamp_iso``."""
        sql = """
                SELECT id, cycle_id, timestamp, symbol, side, qty, filled_qty,
                       order_id, order_type, order_status, account_id
                FROM trade_executions
                WHERE submitted = 1
                  AND order_id IS NOT NULL
                  AND order_id != ''
                  AND LOWER(COALESCE(order_status, '')) = 'pending'
                  AND timestamp < ?
        """
        params: list[str] = [before_timestamp_iso]
        if account_id is not None:
            sql += " AND account_id = ?"
            params.append(account_id.strip() or "default")
        sql += " ORDER BY id ASC"
        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [_row_to_dict(r) for r in rows]

    def update_trade_execution_status_by_order_id(
        self,
        order_id: str,
        *,
        order_status: str,
        filled_qty: float | None = None,
        filled_avg_price: float | None = None,
    ) -> int:
        """Update status (and optional fill fields) for rows matching ``order_id``."""
        oid = order_id.strip()
        if not oid:
            return 0
        sets = ["order_status = ?"]
        params: list[Any] = [order_status]
        if filled_qty is not None:
            sets.append("filled_qty = ?")
            params.append(float(filled_qty))
        if filled_avg_price is not None:
            sets.append("filled_avg_price = ?")
            params.append(float(filled_avg_price))
        params.append(oid)
        sql = f"UPDATE trade_executions SET {', '.join(sets)} WHERE order_id = ?"
        with self._connect() as conn:
            cur = conn.execute(sql, params)
            conn.commit()
            return int(cur.rowcount)

    def list_submitted_executions_missing_filled_qty(
        self,
        *,
        since_timestamp: str,
        account_id: str = "default",
    ) -> list[dict[str, Any]]:
        """Submitted rows with ``filled_qty IS NULL`` on or after ``since_timestamp``."""
        aid = account_id.strip() or "default"
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT id, cycle_id, timestamp, symbol, submitted, order_id, side,
                       qty, filled_avg_price, order_status, filled_qty, account_id
                FROM trade_executions
                WHERE submitted = 1
                  AND filled_qty IS NULL
                  AND timestamp >= ?
                  AND account_id = ?
                ORDER BY timestamp ASC, id ASC
                """,
                (since_timestamp, aid),
            ).fetchall()
        return [_row_to_dict(r) for r in rows]

    def update_execution_fill_fields(
        self,
        execution_id: int,
        *,
        filled_qty: float,
        filled_avg_price: float | None = None,
    ) -> int:
        """Set only ``filled_qty`` and, when known, ``filled_avg_price``.

        Does not touch ``qty``, ``side``, ``symbol``, ``submitted``, or ``order_status``.
        """
        sets = ["filled_qty = ?"]
        params: list[Any] = [float(filled_qty)]
        if filled_avg_price is not None:
            sets.append("filled_avg_price = ?")
            params.append(float(filled_avg_price))
        params.append(int(execution_id))
        sql = f"UPDATE trade_executions SET {', '.join(sets)} WHERE id = ?"
        with self._connect() as conn:
            cur = conn.execute(sql, params)
            conn.commit()
            return int(cur.rowcount)

    def write_execution_quality(self, row: dict[str, Any]) -> None:
        """Upsert slippage vs reference for one ``trade_executions`` row."""
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO execution_quality(
                    execution_id, cycle_id, execution_ts, symbol, side, qty,
                    fill_price, reference_price, slippage_bps, note
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(execution_id) DO UPDATE SET
                    cycle_id = excluded.cycle_id,
                    execution_ts = excluded.execution_ts,
                    symbol = excluded.symbol,
                    side = excluded.side,
                    qty = excluded.qty,
                    fill_price = excluded.fill_price,
                    reference_price = excluded.reference_price,
                    slippage_bps = excluded.slippage_bps,
                    note = excluded.note
                """,
                (
                    int(row["execution_id"]),
                    str(row["cycle_id"]),
                    str(row["execution_ts"]),
                    str(row["symbol"]).strip().upper(),
                    str(row["side"]),
                    float(row["qty"]) if row.get("qty") is not None else None,
                    float(row["fill_price"]) if row.get("fill_price") is not None else None,
                    float(row["reference_price"])
                    if row.get("reference_price") is not None
                    else None,
                    float(row["slippage_bps"]) if row.get("slippage_bps") is not None else None,
                    str(row.get("note") or ""),
                ),
            )
            conn.commit()

    def get_execution_quality_rows(self, *, limit: int = 100) -> list[dict[str, Any]]:
        """Recent execution quality rows (newest by ``execution_id`` desc)."""
        lim = max(1, min(int(limit), 10_000))
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT execution_id, cycle_id, execution_ts, symbol, side, qty,
                       fill_price, reference_price, slippage_bps, note
                FROM execution_quality
                ORDER BY execution_id DESC
                LIMIT ?
                """,
                (lim,),
            ).fetchall()
        return [_row_to_dict(r) for r in rows]

    def upsert_rag_chunk(
        self,
        *,
        source_path: str,
        chunk_index: int,
        text: str,
        embedding: list[float],
        indexed_at: str,
    ) -> None:
        """Insert or replace one indexed document chunk and its embedding vector."""
        emb = json.dumps(embedding)
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO rag_chunks(source_path, chunk_index, text, embedding_json, indexed_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(source_path, chunk_index) DO UPDATE SET
                    text = excluded.text,
                    embedding_json = excluded.embedding_json,
                    indexed_at = excluded.indexed_at
                """,
                (source_path, int(chunk_index), text, emb, indexed_at),
            )
            conn.commit()

    def load_rag_chunks(self) -> list[dict[str, Any]]:
        """All RAG chunks with parsed ``embedding`` list."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT source_path, chunk_index, text, embedding_json, indexed_at FROM rag_chunks",
            ).fetchall()
        out: list[dict[str, Any]] = []
        for r in rows:
            d = _row_to_dict(r)
            raw = d.pop("embedding_json", None)
            if isinstance(raw, str) and raw:
                try:
                    d["embedding"] = json.loads(raw)
                except json.JSONDecodeError:
                    d["embedding"] = []
            else:
                d["embedding"] = []
            out.append(d)
        return out

    def write_ml_score(
        self,
        *,
        cycle_id: str,
        symbol: str,
        score: float,
        model_version: str,
        computed_at: str,
        top_features_json: str | None = None,
    ) -> None:
        """Append one experimental ML score row for a symbol in a cycle."""
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO ml_scores(
                    cycle_id, symbol, score, model_version, computed_at, top_features_json
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    str(cycle_id),
                    str(symbol).strip().upper(),
                    float(score),
                    str(model_version),
                    str(computed_at),
                    top_features_json,
                ),
            )
            conn.commit()

    def get_ml_scores(self, *, limit: int = 100) -> list[dict[str, Any]]:
        """Recent ML score rows (newest first)."""
        lim = max(1, min(int(limit), 10_000))
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT id, cycle_id, symbol, score, model_version, computed_at, top_features_json
                FROM ml_scores
                ORDER BY id DESC
                LIMIT ?
                """,
                (lim,),
            ).fetchall()
        return [_row_to_dict(r) for r in rows]

    def get_latest_ml_scores_batch(self) -> list[dict[str, Any]]:
        """All ML score rows for the most recently inserted cycle (by highest row id).

        Uses one query so ``cycle_id`` cannot race a concurrent insert between reads.
        """
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT id, cycle_id, symbol, score, model_version, computed_at, top_features_json
                FROM ml_scores
                WHERE cycle_id = (
                    SELECT cycle_id FROM ml_scores ORDER BY id DESC LIMIT 1
                )
                ORDER BY symbol ASC
                """
            ).fetchall()
        return [_row_to_dict(r) for r in rows]

    def write_equity_snapshot(
        self,
        date: str,
        total_market_value: float,
        total_cost_basis: float,
        unrealized_pnl: float,
        realized_pnl: float,
        total_pnl: float,
        *,
        account_id: str = "default",
        cash: float | None = None,
        broker_equity: float | None = None,
    ) -> None:
        """Insert or replace one daily portfolio valuation row (``date`` is ``YYYY-MM-DD``).

        ``cash`` is broker cash at snapshot time. ``None`` means unknown (do not
        coerce to 0.0). ``total_market_value`` is positions-only;
        equity = MV + cash when cash is set. ``broker_equity`` is the broker's
        account equity from the same read as ``cash`` (nullable).
        """
        d = date.strip()
        aid = account_id.strip() or "default"
        cash_val: float | None = None if cash is None else float(cash)
        be_val: float | None = None if broker_equity is None else float(broker_equity)
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO equity_snapshots(
                    account_id, date, total_market_value, total_cost_basis,
                    unrealized_pnl, realized_pnl, total_pnl, cash, broker_equity
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(account_id, date) DO UPDATE SET
                    total_market_value = excluded.total_market_value,
                    total_cost_basis = excluded.total_cost_basis,
                    unrealized_pnl = excluded.unrealized_pnl,
                    realized_pnl = excluded.realized_pnl,
                    total_pnl = excluded.total_pnl,
                    cash = COALESCE(excluded.cash, equity_snapshots.cash),
                    broker_equity = COALESCE(excluded.broker_equity, equity_snapshots.broker_equity)
                """,
                (
                    aid,
                    d,
                    float(total_market_value),
                    float(total_cost_basis),
                    float(unrealized_pnl),
                    float(realized_pnl),
                    float(total_pnl),
                    cash_val,
                    be_val,
                ),
            )
            conn.commit()

    def get_equity_snapshots(
        self,
        start: str | None = None,
        end: str | None = None,
        *,
        limit: int | None = None,
        offset: int = 0,
        account_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Return equity snapshot rows ordered by date ascending, optionally filtered.

        When ``limit`` is set, applies ``LIMIT`` / ``OFFSET`` (still ascending by date).
        When ``limit`` is ``None``, returns all matching rows (backward compatible).
        """
        sql = """
            SELECT account_id, date, total_market_value, total_cost_basis,
                   unrealized_pnl, realized_pnl, total_pnl, cash, broker_equity
            FROM equity_snapshots
            WHERE 1=1
        """
        params: list[str | int] = []
        if account_id is not None:
            sql += " AND account_id = ?"
            params.append(account_id.strip() or "default")
        if start is not None:
            sql += " AND date >= ?"
            params.append(start.strip())
        if end is not None:
            sql += " AND date <= ?"
            params.append(end.strip())
        sql += " ORDER BY date ASC"
        if limit is not None:
            lim = max(1, min(int(limit), 10_000))
            off = max(0, int(offset))
            sql += " LIMIT ? OFFSET ?"
            params.extend([lim, off])
        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [_row_to_dict(r) for r in rows]

    def update_equity_snapshot_cash(
        self,
        date: str,
        cash: float | None,
        *,
        account_id: str = "default",
    ) -> int:
        """Set ``cash`` on one equity snapshot row. Returns rows updated (0 or 1)."""
        d = date.strip()
        aid = account_id.strip() or "default"
        cash_val: float | None = None if cash is None else float(cash)
        with self._connect() as conn:
            cur = conn.execute(
                """
                UPDATE equity_snapshots SET cash = ?
                WHERE account_id = ? AND date = ?
                """,
                (cash_val, aid, d),
            )
            conn.commit()
            return int(cur.rowcount)

    def write_macro_indicator(self, date: str, series_id: str, value: float) -> None:
        """Upsert one FRED macro observation (``date`` is ``YYYY-MM-DD``)."""
        d = date.strip()
        sid = series_id.strip().upper()
        fetched = datetime.now(UTC).isoformat()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO macro_indicators(date, series_id, value, fetched_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(date, series_id) DO UPDATE SET
                    value = excluded.value,
                    fetched_at = excluded.fetched_at
                """,
                (d, sid, float(value), fetched),
            )
            conn.commit()

    def get_macro_indicator(
        self,
        series_id: str,
        *,
        start: str | None = None,
        end: str | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        """Return macro rows for ``series_id``, oldest first."""
        sid = series_id.strip().upper()
        sql = "SELECT date, series_id, value FROM macro_indicators WHERE series_id = ?"
        params: list[Any] = [sid]
        if start is not None:
            sql += " AND date >= ?"
            params.append(start.strip())
        if end is not None:
            sql += " AND date <= ?"
            params.append(end.strip())
        sql += " ORDER BY date ASC"
        if limit is not None:
            sql += " LIMIT ?"
            params.append(max(1, int(limit)))
        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [_row_to_dict(r) for r in rows]

    def get_latest_macro(self, series_id: str) -> dict[str, Any] | None:
        """Most recent observation for ``series_id``, or ``None``."""
        sid = series_id.strip().upper()
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT date, series_id, value FROM macro_indicators
                WHERE series_id = ?
                ORDER BY date DESC
                LIMIT 1
                """,
                (sid,),
            ).fetchone()
        return _row_to_dict(row) if row else None

    def write_regime_snapshot(self, regime: MarketRegime) -> None:
        """Persist one ``MarketRegime`` snapshot."""
        from src.data.regime import MarketRegime as MarketRegimeClass

        if not isinstance(regime, MarketRegimeClass):
            msg = "regime must be a MarketRegime instance"
            raise TypeError(msg)
        d = regime.timestamp.astimezone(UTC).date().isoformat()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO market_regime(
                    date, overall, yield_curve, volatility,
                    vix_close, yield_spread, sizing_multiplier
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(date) DO UPDATE SET
                    overall = excluded.overall,
                    yield_curve = excluded.yield_curve,
                    volatility = excluded.volatility,
                    vix_close = excluded.vix_close,
                    yield_spread = excluded.yield_spread,
                    sizing_multiplier = excluded.sizing_multiplier
                """,
                (
                    d,
                    regime.overall.value,
                    regime.yield_curve.value,
                    regime.volatility.value,
                    float(regime.vix_close) if regime.vix_close is not None else None,
                    float(regime.yield_spread) if regime.yield_spread is not None else None,
                    float(regime.sizing_multiplier),
                ),
            )
            conn.commit()

    def get_regime_history(
        self,
        *,
        start: str | None = None,
        end: str | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        """Regime rows, newest first when ``limit`` is set."""
        sql = "SELECT date, overall, yield_curve, volatility, vix_close, yield_spread, sizing_multiplier FROM market_regime WHERE 1=1"
        params: list[Any] = []
        if start is not None:
            sql += " AND date >= ?"
            params.append(start.strip())
        if end is not None:
            sql += " AND date <= ?"
            params.append(end.strip())
        sql += " ORDER BY date DESC"
        if limit is not None:
            sql += " LIMIT ?"
            params.append(max(1, int(limit)))
        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [_row_to_dict(r) for r in rows]

    def get_latest_regime(self) -> dict[str, Any] | None:
        """Most recent regime snapshot, or ``None``."""
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT date, overall, yield_curve, volatility, vix_close, yield_spread, sizing_multiplier
                FROM market_regime
                ORDER BY date DESC
                LIMIT 1
                """,
            ).fetchone()
        return _row_to_dict(row) if row else None

    def write_fundamental_score(
        self,
        symbol: str,
        period: str,
        score: int,
        data: dict[str, Any],
    ) -> None:
        """Upsert Piotroski or related fundamental score row."""
        sym = symbol.strip().upper()
        per = period.strip()
        fetched = datetime.now(UTC).isoformat()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO fundamental_scores(symbol, period, score, data_json, fetched_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(symbol, period) DO UPDATE SET
                    score = excluded.score,
                    data_json = excluded.data_json,
                    fetched_at = excluded.fetched_at
                """,
                (sym, per, int(score), json.dumps(data), fetched),
            )
            conn.commit()

    def get_fundamental_scores(
        self,
        *,
        symbol: str | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        """Fundamental score rows, newest ``fetched_at`` first when limited."""
        sql = (
            "SELECT symbol, period, score, data_json, fetched_at FROM fundamental_scores WHERE 1=1"
        )
        params: list[Any] = []
        if symbol is not None:
            sql += " AND symbol = ?"
            params.append(symbol.strip().upper())
        sql += " ORDER BY fetched_at DESC"
        if limit is not None:
            sql += " LIMIT ?"
            params.append(max(1, int(limit)))
        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        out = [_row_to_dict(r) for r in rows]
        for row in out:
            raw = row.pop("data_json", None)
            if isinstance(raw, str) and raw:
                try:
                    row["data"] = json.loads(raw)
                except json.JSONDecodeError:
                    logger.warning(
                        "Corrupt data_json for %s/%s", row.get("symbol"), row.get("period")
                    )
                    row["data"] = {}
            else:
                row["data"] = {}
        return out

    def get_latest_fundamental_score(self, symbol: str) -> dict[str, Any] | None:
        """Latest score row for one symbol (by ``period`` desc)."""
        sym = symbol.strip().upper()
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT symbol, period, score, data_json, fetched_at FROM fundamental_scores
                WHERE symbol = ?
                ORDER BY period DESC
                LIMIT 1
                """,
                (sym,),
            ).fetchone()
        if not row:
            return None
        d = _row_to_dict(row)
        raw = d.pop("data_json", None)
        if isinstance(raw, str) and raw:
            try:
                d["data"] = json.loads(raw)
            except json.JSONDecodeError:
                logger.warning("Corrupt data_json for %s", d.get("symbol"))
                d["data"] = {}
        else:
            d["data"] = {}
        return d

    def log_alert(self, alert: Alert) -> None:
        """Append one operational alert (append-only)."""
        ts = alert.timestamp.isoformat()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO alerts(timestamp, level, category, message)
                VALUES (?, ?, ?, ?)
                """,
                (ts, alert.level.value, alert.category, alert.message),
            )
            conn.commit()

    def get_alerts(self, *, limit: int = 50, offset: int = 0) -> list[dict[str, Any]]:
        """Recent alerts, newest first."""
        lim = max(1, min(int(limit), 10_000))
        off = max(0, int(offset))
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT id, timestamp, level, category, message
                FROM alerts
                ORDER BY id DESC
                LIMIT ? OFFSET ?
                """,
                (lim, off),
            ).fetchall()
        return [_row_to_dict(r) for r in rows]

    def save_backtest_run(
        self,
        strategy_name: str,
        config: BacktestConfig,
        result: BacktestResult,
        *,
        start_date: str,
        end_date: str,
        benchmark_symbol: str | None = None,
        benchmark_metrics: ReturnMetrics | None = None,
        experiment_label: str | None = None,
    ) -> int:
        """Persist one backtest result; return new row ``id``.

        ``start_date`` / ``end_date`` are ``YYYY-MM-DD`` strings (requested backtest window).
        Persists ``config.ablation_config_dict()`` so analysis-script runs remain interpretable.
        """
        m = result.return_metrics
        run_at = datetime.now(UTC).isoformat()
        curve_json = json.dumps(result.equity_curve)
        trades_json = json.dumps([t.model_dump() for t in result.trades])
        bench_json = (
            json.dumps(benchmark_metrics.model_dump()) if benchmark_metrics is not None else None
        )
        label = experiment_label if experiment_label is not None else config.experiment_label
        if label is None:
            label = result.experiment_label
        config_json = json.dumps(config.ablation_config_dict())
        with self._connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO backtest_runs(
                    run_at, strategy_name, start_date, end_date,
                    initial_capital, final_equity, slippage_bps, rebalance_frequency,
                    total_return_pct, cagr_pct, sharpe_ratio, sortino_ratio,
                    max_drawdown_pct, calmar_ratio, annual_volatility_pct,
                    trade_count, equity_curve_json, trades_json,
                    sizing_mode, cash_yield_annual_pct, benchmark_symbol, benchmark_metrics_json,
                    config_json, experiment_label, turnover
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_at,
                    strategy_name,
                    start_date.strip(),
                    end_date.strip(),
                    float(result.initial_capital),
                    float(result.final_equity),
                    float(config.slippage_bps),
                    str(config.rebalance_frequency),
                    float(m.total_return_pct),
                    float(m.cagr_pct),
                    m.sharpe_ratio,
                    m.sortino_ratio,
                    float(m.max_drawdown_pct),
                    m.calmar_ratio,
                    float(m.annual_volatility_pct),
                    len(result.trades),
                    curve_json,
                    trades_json,
                    str(result.sizing_mode),
                    float(result.cash_yield_annual_pct),
                    benchmark_symbol,
                    bench_json,
                    config_json,
                    label,
                    float(result.turnover),
                ),
            )
            conn.commit()
            return int(cur.lastrowid)

    def get_backtest_runs(self, *, limit: int = 50, offset: int = 0) -> list[dict[str, Any]]:
        """Recent backtest runs (newest first), excluding large JSON by default."""
        lim = max(1, min(int(limit), 10_000))
        off = max(0, int(offset))
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT id, run_at, strategy_name, start_date, end_date,
                       initial_capital, final_equity, slippage_bps, rebalance_frequency,
                       total_return_pct, cagr_pct, sharpe_ratio, sortino_ratio,
                       max_drawdown_pct, calmar_ratio, annual_volatility_pct, trade_count,
                       sizing_mode, cash_yield_annual_pct, benchmark_symbol
                FROM backtest_runs
                ORDER BY id DESC
                LIMIT ? OFFSET ?
                """,
                (lim, off),
            ).fetchall()
        return [_row_to_dict(r) for r in rows]

    def get_backtest_run(self, run_id: int) -> dict[str, Any] | None:
        """Return one backtest row including ``equity_curve_json``."""
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT id, run_at, strategy_name, start_date, end_date,
                       initial_capital, final_equity, slippage_bps, rebalance_frequency,
                       total_return_pct, cagr_pct, sharpe_ratio, sortino_ratio,
                       max_drawdown_pct, calmar_ratio, annual_volatility_pct, trade_count,
                       equity_curve_json, trades_json,
                       sizing_mode, cash_yield_annual_pct, benchmark_symbol, benchmark_metrics_json,
                       config_json, experiment_label, turnover
                FROM backtest_runs
                WHERE id = ?
                """,
                (int(run_id),),
            ).fetchone()
        return _row_to_dict(row) if row else None


def _row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    # sqlite3.Row is not a plain mapping; build an explicit str-key dict.
    return {str(k): row[k] for k in row.keys()}  # noqa: SIM118
