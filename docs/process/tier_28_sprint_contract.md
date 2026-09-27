# Tier 28 Sprint Contract — Execution Depth

**Depends on:** Tier 27 complete (630+ tests green, lint clean, ensemble merged)
**Governance:** `AGENTS.md` — TDD (failing test first), ruff clean, no `print()` in production paths

---

## 1. Sprint goal

Expand order execution beyond market orders: add limit order support for reduced slippage, stop-loss orders for downside protection, and PDT (Pattern Day Trader) tracking to warn before triggering the 4-trade/5-day rule. Also add order status tracking so unfilled limit orders are visible and can be reconciled.

---

## 2. In scope

### 28A — Limit & Stop Order Support

- **Intent:** Market orders have slippage. Limit orders let strategies set a maximum price. Stop-loss orders protect positions from large drawdowns.
- **BrokerInterface additions** (`src/execution/broker_interface.py`):
  ```python
  @abstractmethod
  def submit_limit_order(
      self, symbol: str, qty: float, side: Literal["buy", "sell"],
      *, limit_price: float, time_in_force: str = "day",
  ) -> str:
      """Submit a limit order; return broker order id."""

  @abstractmethod
  def submit_stop_order(
      self, symbol: str, qty: float, side: Literal["buy", "sell"],
      *, stop_price: float, time_in_force: str = "day",
  ) -> str:
      """Submit a stop order; return broker order id."""

  @abstractmethod
  def cancel_order(self, order_id: str) -> bool:
      """Cancel an open order; return True if cancelled, False if already filled/cancelled."""

  @abstractmethod
  def get_order_status(self, order_id: str) -> dict[str, Any]:
      """Return order status dict with at least: order_id, status, filled_qty, filled_avg_price."""
  ```
- **AlpacaBrokerAdapter implementation** (`src/execution/alpaca_adapter.py`):
  - `submit_limit_order()` using `LimitOrderRequest` from alpaca-py
  - `submit_stop_order()` using `StopOrderRequest` from alpaca-py
  - `cancel_order()` using `self._client.cancel_order_by_id()`
  - `get_order_status()` using `self._client.get_order_by_id()`
  - All with existing retry policy integration
- **Config additions** (`src/config.py` → new `ExecutionConfig`):
  ```python
  class ExecutionConfig(BaseModel):
      default_order_type: Literal["market", "limit"] = "market"
      limit_offset_bps: float = Field(
          default=10.0, ge=0.0, le=100.0,
          description="Basis points above last price for limit buy orders (e.g., 10 = 0.1% above)"
      )
      stop_loss_enabled: bool = False
      stop_loss_pct: float = Field(
          default=5.0, ge=0.5, le=50.0,
          description="Stop-loss trigger as % below entry price"
      )
      unfilled_timeout_minutes: int = Field(
          default=30, ge=1, le=480,
          description="Cancel unfilled limit orders after this many minutes"
      )
  ```
- **Config in `settings.yaml`:**
  ```yaml
  execution:
    default_order_type: "market"
    limit_offset_bps: 10.0
    stop_loss_enabled: false
    stop_loss_pct: 5.0
    unfilled_timeout_minutes: 30
  ```
- **Tests:**
  - `test_submit_limit_order()` — verify LimitOrderRequest constructed with correct limit_price
  - `test_submit_stop_order()` — verify StopOrderRequest constructed with correct stop_price
  - `test_cancel_order()` — verify cancellation call
  - `test_get_order_status()` — verify status dict returned
  - `test_limit_order_price_calculation()` — last_price + limit_offset_bps = correct limit_price

### 28B — Order Manager Limit/Stop Integration

- **Intent:** `OrderManager` should use limit orders when configured, and optionally place stop-loss orders after fills.
- **Changes to `src/execution/order_manager.py`:**
  - `execute_signals()`: when `default_order_type == "limit"`, compute limit price from `last_price * (1 + limit_offset_bps / 10000)` and call `submit_limit_order()` instead of `submit_market_order()`
  - After successful fill: if `stop_loss_enabled`, place a trailing stop-loss at `fill_price * (1 - stop_loss_pct / 100)` via `submit_stop_order()` (sell side)
  - `close_position()`: always use market orders for sells (immediate execution needed for rotation/TLH)
  - Track unfilled limit orders: store `order_id` + `submitted_at` for timeout checking
- **Tests:**
  - `test_execute_signals_uses_limit_order_when_configured()` — verify limit order submitted
  - `test_execute_signals_places_stop_loss_after_fill()` — verify stop order placed
  - `test_close_position_always_uses_market()` — no change for sells
  - `test_limit_order_with_retry()` — transient failure + retry on limit submit

### 28C — Order Status Tracking & Unfilled Order Cleanup

- **Intent:** Limit orders may not fill immediately. Track order status in SQLite; cancel stale unfilled orders.
- **SQLite schema migration** (`src/data/storage/sqlite_store.py`):
  Add columns to `trade_executions`:
  ```sql
  ALTER TABLE trade_executions ADD COLUMN order_type TEXT DEFAULT 'market';
  ALTER TABLE trade_executions ADD COLUMN limit_price REAL;
  ALTER TABLE trade_executions ADD COLUMN stop_price REAL;
  ALTER TABLE trade_executions ADD COLUMN order_status TEXT DEFAULT 'filled';
  ALTER TABLE trade_executions ADD COLUMN filled_qty REAL;
  ```
- **OrderExecutionResult model extension** (`src/models.py`):
  ```python
  order_type: Literal["market", "limit", "stop", "stop_limit"] = "market"
  limit_price: float | None = None
  stop_price: float | None = None
  order_status: str = "filled"  # filled, pending, cancelled, rejected
  filled_qty: float | None = None
  ```
- **Unfilled order cleanup** (`src/execution/order_manager.py`):
  - New method: `cancel_stale_orders(self, *, timeout_minutes: int) -> list[str]`
  - Queries SQLite for orders where `order_status == 'pending'` and `timestamp < now - timeout_minutes`
  - Calls `broker.cancel_order()` for each
  - Updates `order_status` to `'cancelled'` in SQLite
  - Called at start of each `run_cycle()` in workflows.py
- **Tests:**
  - `test_cancel_stale_orders()` — pending orders older than timeout are cancelled
  - `test_cancel_stale_orders_skips_filled()` — already-filled orders untouched
  - `test_order_status_persisted_to_sqlite()` — verify new columns populated

### 28D — PDT (Pattern Day Trader) Tracking

- **Intent:** Accounts under $25k that make 4+ day trades in 5 business days get flagged as PDT by FINRA. Track round-trip counts and warn before triggering the threshold.
- **What is a day trade:** Buying and selling the same symbol on the same calendar day.
- **New module:** `src/risk/pdt_tracker.py`
  ```python
  def count_day_trades(executions: list[dict], *, lookback_days: int = 5) -> int:
      """Count round-trip day trades in the lookback window."""

  def evaluate_pdt_risk(
      day_trade_count: int,
      equity: float,
      *,
      pdt_threshold: int = 4,
      equity_floor: float = 25_000.0,
  ) -> PDTCheckResult:
      """Return warning/block if approaching or at PDT limit."""
  ```
  - `PDTCheckResult`: `(allowed: bool, warning: str | None, day_trade_count: int)`
- **Integration with pre-trade checks** (`src/risk/pre_trade_checks.py`):
  - When `pdt_protection: true` (already in config but unused), check PDT before allowing sells that would create a day trade
  - If a buy happened today for the same symbol, and we're about to sell → this would be a day trade
  - If `day_trade_count >= pdt_threshold` and `equity < equity_floor` → block with reason
  - If `day_trade_count >= pdt_threshold - 1` → warn but allow
- **Config:** `pdt_protection: true` already exists in `RiskConfig` — just wire it
- **Tests:**
  - `test_count_day_trades_same_day_buy_sell()` — 1 round-trip counted
  - `test_count_day_trades_across_days_not_counted()` — buy Monday, sell Tuesday = 0
  - `test_pdt_blocks_at_threshold()` — 4 day trades + equity < $25k → blocked
  - `test_pdt_allows_above_equity_floor()` — 4 day trades + equity > $25k → allowed
  - `test_pdt_warns_near_threshold()` — 3 day trades → warning
  - `test_pdt_protection_disabled()` — `pdt_protection: false` → always allowed

---

## 3. Out of scope (deferred beyond Tier 28)

- Trailing stop orders (TrailingStopOrderRequest) — keep it simple with fixed stop-loss first
- Stop-limit orders (StopLimitOrderRequest) — edge case, defer
- Bracket orders (OCO/OTO) — Alpaca supports these but adds significant complexity
- Streaming WebSocket fills (real-time) — poll-based status check is sufficient for swing trading
- Partial fill handling beyond tracking `filled_qty` — Alpaca handles this internally
- Per-symbol order type overrides (all symbols use same `default_order_type`)
- Extended hours trading (`extended_hours` flag)
- Multi-leg order coordination

---

## 4. Acceptance criteria

- [ ] `ruff check src/ tests/` clean
- [ ] Full `pytest` suite green; target **665+ tests** (currently 630, adding ~35)
- [ ] `default_order_type: "market"` (default) preserves exact current behavior — no regressions
- [ ] Limit orders compute correct price from `last_price + limit_offset_bps`
- [ ] Stop-loss orders placed after fill when enabled
- [ ] Unfilled limit orders cancelled after `unfilled_timeout_minutes`
- [ ] `trade_executions` table has new columns for `order_type`, `limit_price`, `stop_price`, `order_status`, `filled_qty`
- [ ] PDT tracker counts day trades correctly (same-day buy+sell = 1 day trade)
- [ ] PDT blocks trades when at threshold with equity < $25k
- [ ] `pdt_protection: true` wired to pre-trade checks
- [ ] `BUILD_STATE.md` updated with Tier 28 entry
- [ ] No `print()` in production paths; use `logging`

---

## 5. TDD task order (suggested)

**Phase 1 — Broker Interface & Adapter (28A)**

1. Write `test_submit_limit_order()` → add method to ABC + AlpacaBrokerAdapter
2. Write `test_submit_stop_order()` → add method to ABC + AlpacaBrokerAdapter
3. Write `test_cancel_order()` → add method to ABC + AlpacaBrokerAdapter
4. Write `test_get_order_status()` → add method to ABC + AlpacaBrokerAdapter
5. Add `ExecutionConfig` to `config.py`, add to `settings.yaml`

**Phase 2 — Order Manager Integration (28B)**

6. Write `test_limit_order_price_calculation()` — verify bps offset math
7. Write `test_execute_signals_uses_limit_order_when_configured()` → modify `execute_signals()`
8. Write `test_execute_signals_places_stop_loss_after_fill()` → add stop-loss logic
9. Write `test_close_position_always_uses_market()` → verify no change for sells
10. Write `test_limit_order_with_retry()` → verify retry on limit submit

**Phase 3 — Order Status & Schema (28C)**

11. Add `order_type`, `limit_price`, `stop_price`, `order_status`, `filled_qty` to `OrderExecutionResult`
12. Add SQLite migration for new columns
13. Write `test_order_status_persisted_to_sqlite()` → verify new columns populated
14. Write `test_cancel_stale_orders()` → implement `cancel_stale_orders()` method
15. Write `test_cancel_stale_orders_skips_filled()` → verify filled orders untouched
16. Wire `cancel_stale_orders()` into `run_cycle()` start

**Phase 4 — PDT Tracking (28D)**

17. Write `test_count_day_trades_same_day_buy_sell()` → implement `count_day_trades()`
18. Write `test_count_day_trades_across_days_not_counted()` → verify cross-day not counted
19. Write `test_pdt_blocks_at_threshold()` → implement `evaluate_pdt_risk()`
20. Write `test_pdt_allows_above_equity_floor()` → equity check
21. Write `test_pdt_warns_near_threshold()` → warning at N-1
22. Write `test_pdt_protection_disabled()` → config flag bypass
23. Wire PDT check into `pre_trade_checks.py`

**Phase 5 — Finalize**

24. Run full `pytest` suite → all green
25. Run `ruff check src/ tests/` → clean
26. Update `BUILD_STATE.md` with Tier 28 entry
27. Verify `default_order_type: "market"` produces identical behavior to Tier 27

---

## 6. Risks and mitigations

| Risk | Mitigation |
|------|------------|
| Limit orders don't fill → positions not taken | `unfilled_timeout_minutes` cancels stale orders; next cycle retries as market if needed |
| Stop-loss fires in a flash crash → sells at bad price | Stop orders are stop-market (not stop-limit); Alpaca handles fill. Users can disable via `stop_loss_enabled: false` |
| PDT tracker false positives from old executions | Lookback is configurable (default 5 business days); only counts same-day buy+sell pairs |
| Schema migration breaks existing SQLite databases | Use `ALTER TABLE ADD COLUMN` with defaults; existing rows get `order_type='market'`, `order_status='filled'` |
| Limit price calculation edge case (bps offset on very low-price stocks) | Floor at $0.01 minimum price; validation in order manager |
| Cancel race condition (order fills while cancel in flight) | `cancel_order()` returns `False` if already filled; log and continue |

---

## 7. Definition of done

- All 4 sub-scopes (28A–28D) implemented with passing tests
- 665+ total tests, all green, ruff clean
- Zero regressions on existing 630 tests
- `default_order_type: "market"` + `stop_loss_enabled: false` + `pdt_protection: true` (defaults) produce identical behavior to Tier 27
- Limit and stop orders work via Alpaca paper trading
- PDT tracking counts round-trips correctly and blocks/warns appropriately
- `BUILD_STATE.md` updated with Tier 28 summary

---

## Key files

| File | Change |
|------|--------|
| `src/execution/broker_interface.py` | Add `submit_limit_order()`, `submit_stop_order()`, `cancel_order()`, `get_order_status()` to ABC |
| `src/execution/alpaca_adapter.py` | Implement new methods using `LimitOrderRequest`, `StopOrderRequest` from alpaca-py |
| `src/execution/order_manager.py` | Add limit order routing, stop-loss placement, `cancel_stale_orders()` |
| `src/models.py` | Extend `OrderExecutionResult` with `order_type`, `limit_price`, `stop_price`, `order_status`, `filled_qty` |
| `src/data/storage/sqlite_store.py` | Schema migration for new `trade_executions` columns |
| `src/config.py` | Add `ExecutionConfig` class |
| `config/settings.yaml` | Add `execution:` section |
| `src/risk/pdt_tracker.py` | **NEW** — `count_day_trades()`, `evaluate_pdt_risk()` |
| `src/risk/pre_trade_checks.py` | Wire PDT check into pre-trade evaluation |
| `src/automation/workflows.py` | Call `cancel_stale_orders()` at cycle start |
| `tests/unit/execution/test_limit_stop_orders.py` | **NEW** — broker + order manager limit/stop tests |
| `tests/unit/execution/test_order_status.py` | **NEW** — status tracking + stale cancellation tests |
| `tests/unit/risk/test_pdt_tracker.py` | **NEW** — day trade counting + PDT risk evaluation tests |

---

## Verification

1. `ruff check src/ tests/` — clean
2. `pytest tests/ -x` — 665+ tests, all green
3. `financialhub once` with `default_order_type: "market"` — identical behavior to Tier 27
4. `financialhub once` with `default_order_type: "limit"` + `limit_offset_bps: 10` — verify limit orders submitted to Alpaca paper
5. Check Alpaca paper dashboard: limit orders visible with correct prices
6. Check SQLite: `trade_executions` rows have `order_type`, `limit_price`, `order_status` populated
7. `financialhub reconcile` — still works with new order types
