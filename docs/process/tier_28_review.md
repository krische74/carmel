# Tier 28 Code Review

## Scope
- **28A** — Limit & stop order support in BrokerInterface + AlpacaBrokerAdapter
- **28B** — Order manager limit routing, stop-loss placement after fill
- **28C** — Order status tracking (new SQLite columns), unfilled order cleanup
- **28D** — PDT tracking and pre-trade check integration

## Verdict: PASS with one notable issue

Limit/stop order logic is correct, PDT tracking works, SQLite migrations are backward-compatible, and all defaults preserve Tier 27 behavior. One race condition in `cancel_stale_orders()` should be addressed.

---

## Production Code

### What's correct

| Check | Status |
|-------|--------|
| Limit price calculation (last_price × (1 + bps/10000)) | CORRECT |
| Stop-loss price (fill_price × (1 - stop_loss_pct/100)) | CORRECT |
| Sells always use market orders | CORRECT |
| Limit buy price floor at $0.01 | CORRECT |
| `default_order_type: "market"` preserves Tier 27 behavior | CORRECT |
| `stop_loss_enabled: false` default | CORRECT |
| PDT counts same-day buy+sell as 1 day trade | CORRECT |
| PDT blocks at threshold when equity < $25k | CORRECT |
| PDT allows when equity ≥ $25k regardless of count | CORRECT |
| `pdt_protection: true` wired to pre-trade checks | CORRECT |
| SQLite migrations use ALTER TABLE ADD COLUMN with defaults | CORRECT — backward-compatible |
| OrderExecutionResult extended fields | CORRECT — order_type, limit_price, stop_price, order_status, filled_qty |
| Retry policy works with limit/stop orders | CORRECT |

---

## Findings

### HIGH — Race condition in stale order cancellation

#### H1: TOCTOU in `cancel_stale_orders()`
- **File:** `src/execution/order_manager.py` (~lines 179–202)
- **Problem:** The method queries SQLite for `order_status == 'pending'` orders, then calls `broker.cancel_order()`. Between the query and the cancel call, the order may have filled. When `cancel_order()` returns `False`, the code logs "may have filled" but does not verify by calling `get_order_status()`. The order is left as `pending` in SQLite, creating a status mismatch.
- **Impact:** A filled order could remain marked as "pending" in the database. Reconciliation would eventually catch this, but the SQLite state would be stale until then.
- **Fix:** After `cancel_order()` returns `False`, call `get_order_status()` to check actual state and update SQLite accordingly:
  ```python
  if not cancelled:
      status = self._broker.get_order_status(oid)
      actual = status.get("status", "unknown")
      self._sqlite.update_order_status(oid, actual)
      if actual == "filled":
          logger.info("Order %s filled while cancel in flight", oid)
  ```

---

### MEDIUM — Potential issues

#### M1: Order status classification may be fragile
- **File:** `src/execution/order_manager.py` (~lines 36–47)
- **Problem:** The `_bucket_order_status()` helper maps broker status strings to internal categories. If Alpaca changes status names or adds new ones (e.g., "queued", "held", "pending_cancel"), they would fall through to a default bucket. Currently logs a warning, but the choice of default bucket matters.
- **Fix:** Add explicit handling for known Alpaca statuses (`new`, `accepted`, `partially_filled`, `filled`, `cancelled`, `expired`, `rejected`, `pending_cancel`, `pending_replace`) and map any unknown to `"unknown"` with a WARNING log.

#### M2: PDT threshold and equity floor not configurable via YAML
- **File:** `src/risk/pdt_tracker.py` (~line 158)
- **Problem:** `pdt_threshold=4` and `equity_floor=25_000.0` are function parameter defaults, not config fields. Users can't customize via `settings.yaml`. Unlikely to need this (FINRA rules are fixed), but inconsistent with how other risk params are configurable.
- **Fix:** Add `pdt_threshold: int = 4` and `pdt_equity_floor: float = 25_000.0` to `RiskConfig`. Low priority — the defaults match FINRA rules exactly.

---

### LOW — Minor items

| ID | File | Issue |
|----|------|-------|
| L1 | `order_manager.py` ~line 300 | `assert limit_px is not None` — assertion can never fail since `_limit_buy_price()` always returns a float. Unnecessary but harmless. |
| L2 | `sqlite_store.py` | New columns added via migration but not in initial `CREATE TABLE`. New databases run the migration immediately after creation — works but slightly inefficient. |
| L3 | `test_pdt_tracker.py` | Tests only use single symbols. No test for PDT with multiple symbols traded same day (e.g., buy+sell SPY and buy+sell QQQ = 2 day trades). |
| L4 | `test_order_status.py` | No test for partial fill handling — what if order is 50% filled when cancelled? |

---

## Test Coverage

**19 new tests total:**
- `test_limit_stop_orders.py` — 8 tests: limit orders, stop-loss placement, retry, adapter methods
- `test_order_status.py` — 3 tests: status persistence, stale cancellation, skip-filled
- `test_pdt_tracker.py` — 8 tests: day trade counting, cross-day exclusion, threshold block/warn, equity check, disabled bypass

**Gaps:**
- No test for cancel race condition (order fills during cancel)
- No test for `_bucket_order_status()` with all Alpaca status strings
- No multi-symbol PDT test
- No partial fill scenario test

---

## Security

| Check | Status |
|-------|--------|
| SQL injection | Safe — parameterized queries |
| Order ID handling | Safe — stripped and validated |
| Float precision | Safe — explicit conversions |
| Price manipulation | Safe — limit price has $0.01 floor |

---

## Summary

| Category | Finding Count | Severity |
|----------|--------------|----------|
| Production code bugs | 0 | Clean |
| Race conditions | 1 (cancel_stale_orders TOCTOU) | HIGH |
| Classification fragility | 1 (status string mapping) | MEDIUM |
| Config gaps | 1 (PDT threshold not in YAML) | MEDIUM |
| Test gaps | 4 | LOW |
| Security issues | 0 | Clean |

---

## Recommended Fix Order

1. **H1** — Fix TOCTOU in `cancel_stale_orders()`: call `get_order_status()` when cancel returns False, update SQLite (15 min)
2. **M1** — Enumerate all known Alpaca order statuses in `_bucket_order_status()` (10 min)
3. **L3** — Add multi-symbol PDT test (5 min)
4. **L4** — Add partial fill cancellation test (10 min)
5. **M2, L1, L2** — Address as bandwidth allows
