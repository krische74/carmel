# Tier 29 Code Review

## Scope
- **29A** — Dashboard authentication (session-based login gate)
- **29B** — API pagination (offset param, {items, total, limit, offset} response)
- **29C** — Streamlit test foundation (tests/unit/dashboard/ with 19 tests)
- **29D** — API error response consistency (400/404/500 handlers)
- **29E** — Tier 28 review fixes (TOCTOU, status enumeration, multi-symbol PDT, partial fill)

## Verdict: PASS with one security recommendation

All features are correctly implemented. Dashboard auth covers all 8 pages, pagination works across all list endpoints, error responses return consistent JSON, and all Tier 28 fixes are verified. One security item to address.

**Test count:** 677 total (up from 665)

---

## Critical Security Finding

### S1: Password comparison is not timing-safe
- **File:** `src/dashboard/auth.py` (~line 37)
- **Problem:** Password comparison uses standard `==` operator:
  ```python
  if entered == password:
  ```
  This is vulnerable to timing attacks — an attacker can statistically infer the correct password by measuring response time differences as each character matches or fails.
- **Risk:** Medium. Streamlit is typically on localhost/internal networks, but best practice is constant-time comparison regardless.
- **Fix:** Replace with `hmac.compare_digest()`:
  ```python
  import hmac
  if hmac.compare_digest(entered, password):
  ```
  One-line change. `hmac.compare_digest()` is in Python stdlib and compares in constant time.

---

## All Features Verified

### 29A — Dashboard Auth
| Check | Status |
|-------|--------|
| `require_auth()` gates all 8 pages + home | VERIFIED — auth gate at top of every file before any data access |
| Auth disabled by default | CORRECT — `auth_enabled: false`, empty password = no gate |
| Env var override (`DASHBOARD_PASSWORD`) | CORRECT — takes precedence over YAML |
| Session persistence | CORRECT — `st.session_state` keeps auth across page navigation |
| Failed login logged | CORRECT — `logger.warning` on wrong password |
| Password never logged | CORRECT — no password value in any log statement |
| `.env.example` updated | CORRECT — `DASHBOARD_PASSWORD=` added |
| 08_ask.py silent exception fixed | CORRECT — proper exception handling with `st.error()` + logging |

### 29B — API Pagination
| Check | Status |
|-------|--------|
| `offset` param on trades/signals, trades/executions, alerts, backtests | VERIFIED |
| `count_*()` methods in SQLiteStore | VERIFIED |
| Response schema `{items, total, limit, offset}` | CORRECT |
| `offset=0` default preserves backward compatibility | CORRECT |
| Offset beyond total returns empty items (not error) | CORRECT |
| Offset clamped to max 1,000,000 | CORRECT — prevents DoS |
| Limit clamped to 1–10,000 | CORRECT |

### 29C — Dashboard Test Foundation
| Check | Status |
|-------|--------|
| `tests/unit/dashboard/` exists | YES — 5 files |
| `test_formatting.py` (8 tests) | GOOD — timestamps, side display, edge cases |
| `test_paths.py` (3 tests) | ADEQUATE — path resolution, absolute path passthrough |
| `test_charts.py` (2 tests) | MINIMAL — layout + color hex validation |
| `test_auth.py` (6 tests) | EXCELLENT — disabled, empty password, env override, cached session, wrong password, not authenticated |

### 29D — API Error Consistency
| Check | Status |
|-------|--------|
| 400 (validation error) → `{"error": "Bad request", "detail": [...]}` | CORRECT |
| 400 (ValueError) → `{"error": "Bad request", "detail": "..."}` | CORRECT |
| 404 → `{"error": "Not found"}` | CORRECT |
| 500 → `{"error": "Internal server error"}` (no detail leaked) | CORRECT — security best practice |

### 29E — Tier 28 Fixes
| ID | Status | Evidence |
|----|--------|----------|
| H1 (TOCTOU) | RESOLVED | `cancel_stale_orders()` calls `get_order_status()` when cancel returns False; test `test_cancel_stale_orders_refetches_status_when_cancel_returns_false()` verifies |
| M1 (status enum) | RESOLVED | `_bucket_order_status()` covers 19 known Alpaca statuses (filled, pending, terminal) + unknown with WARNING log; 13 parametrized test cases |
| L3 (multi-symbol PDT) | RESOLVED | `test_count_day_trades_multiple_symbols_same_day()` — SPY + QQQ = 2 day trades |
| L4 (partial fill) | RESOLVED | `test_cancel_stale_orders_partial_fill_when_cancel_false()` — qty=10, filled=5, status=cancelled with fill fields persisted |

---

## Minor Observations (not blocking)

| ID | Area | Note |
|----|------|------|
| L1 | `test_paths.py` | Only 3 tests; no coverage for missing directories or permission errors. Acceptable — paths are simple and Streamlit handles missing dirs. |
| L2 | `test_charts.py` | Only 2 tests; no invalid input handling. Acceptable — `apply_hub_layout()` is a thin Plotly wrapper. |
| L3 | Pagination | No test for `limit=0` edge case. Clamped to `max(1, ...)` in SQLiteStore, so it works but isn't explicitly tested. |
| L4 | Auth | No password reset mechanism. Acceptable — rotate via `.env` + redeploy. |

---

## Summary

| Category | Finding Count | Severity |
|----------|--------------|----------|
| Security (timing-safe comparison) | 1 | HIGH (easy fix) |
| Production code bugs | 0 | Clean |
| Auth coverage | All 8 pages verified | Complete |
| Pagination correctness | All endpoints verified | Complete |
| Error response consistency | All status codes verified | Complete |
| Tier 28 fixes | All 4 items verified | Complete |
| Test quality | 19 dashboard + 6 API tests | Good |

---

## Recommended Fix

1. **S1** — Replace `==` with `hmac.compare_digest()` in `auth.py` (1-line change, 2 min)

That's it. Everything else is clean.
