# Tier 30 Code Review

## Scope
- **30A** — Multi-account config (AccountConfig, BrokerConfig.accounts, legacy env fallback)
- **30B** — Database schema (account_id on 5 tables, LotLedger partition, migrations)
- **30C** — Broker factory (build_brokers), get_account_id(), multi-account workflow + runner
- **30D** — Per-account tax reporting, cross-account wash sales, Form 8949 per-account, API + dashboard
- **30E** — hmac.compare_digest() for dashboard auth (Tier 29 review fix)

## Verdict: PASS

This is the most complex tier to date and it's well-executed. Multi-account architecture is sound, account isolation is correct at every layer, secrets handling is secure, cross-account wash detection boundary logic is precise, and backward compatibility is fully preserved. 750+ tests passing.

---

## Critical Security Checks — All Passed

| Check | Status | Evidence |
|-------|--------|---------|
| API keys never in YAML | SECURE | `AccountConfig` stores only env var names (`api_key_env`); `.env.example` has placeholders only |
| `hmac.compare_digest()` in auth | PRESENT | Line 38 in `auth.py` — Tier 29 S1 fix confirmed |
| Secrets excluded from public config | SECURE | `public_settings_dict()` strips sensitive fields |
| SQL injection | SAFE | All parameterized queries with `?` placeholders |

## Core Architecture — Correct

| Check | Status | Evidence |
|-------|--------|---------|
| Legacy fallback (empty accounts) | CORRECT | `build_brokers()` uses `ALPACA_API_KEY`/`ALPACA_SECRET_KEY` → `{"default": adapter}` |
| account_id='default' on existing data | CORRECT | Migration adds `DEFAULT 'default'` column; all queries normalize blanks to 'default' |
| Lot ledger partition | CORRECT | All queries include `WHERE account_id = ?` when parameter set |
| Workflow account isolation | CORRECT | `log_signal()`, `log_execution()`, `record_buy()`, `record_sell()` all pass `self._account_id` |
| Runner sequential iteration | CORRECT | `for name, br in brokers.items()` loop — no async/threading |
| Cross-account wash ±30 days | CORRECT | Day 30 flagged, day 31 not flagged — boundary tests confirm |
| Tax summary per-account | CORRECT | `compute_tax_summary(account_id=...)` filters closed lots |
| Form 8949 per-account CSV | CORRECT | CSV includes account scope row; filename scoped by account |
| Dashboard account selector | CORRECT | Dropdown + cross-account wash expander |
| API account_id param | CORRECT | Tax endpoints accept `account_id` query param |

## Test Coverage — Comprehensive (47 new + 33 legacy = 80 total for this tier)

| Test Area | Tests | Quality |
|-----------|-------|---------|
| Config parsing + fallback | 4 | Good — multi-account YAML, empty accounts, env vars, defaults |
| Lot ledger partition | 5 | Excellent — isolation, merged wash, scoped wash, sorting |
| Cross-account wash sales | 11 | Excellent — within window, outside window, same account, boundary (day 30/31), gains excluded, earliest replacement, closed lot replacement |
| Broker factory | 11 | Excellent — resolve_account_id edge cases (type, whitespace, exception, empty), build_brokers (legacy, multi, validation, missing env) |
| Runner iteration | 1 | Adequate — verifies sequential iteration with correct broker per account |
| Tax reporting | 9 | Good — per-account filter, consolidated, CSV export, unknown account, whitespace |
| API endpoints | 4 | Good — filter, whitespace, consolidated, cross-account endpoint |
| SQLite store | 2 | Adequate — migration, signal filtering |
| **Legacy regression** | **33** | **All passing unchanged — no regressions** |

---

## Findings

### MEDIUM — Minor type annotation issue

#### M1: `count_equity_snapshots()` params type annotation
- **File:** `src/data/storage/sqlite_store.py` (~line 472)
- **Problem:** `params: list[str] = []` but later appends mixed types. Works at runtime but misleads static analysis.
- **Fix:** Change to `params: list[str | int] = []` or `list[Any]`.

### LOW — Documentation improvements

| ID | File | Note |
|----|------|------|
| L1 | `src/portfolio/wash_sales.py` | Module docstring should explain intra-account vs cross-account wash detection distinction |
| L2 | `src/reporting/tax_report.py` | `compute_tax_summary()` docstring should note that `account_id=None` means consolidated |
| L3 | `config/settings.yaml` | Multi-account example is commented out (correct) but could use a brief inline guide |
| L4 | `tests/unit/automation/test_multi_account_runner.py` | Only 1 test for runner iteration — could add a test for single-account-in-list (distinct from legacy fallback) |

---

## What's Particularly Well Done

1. **Secrets architecture** — env var names in YAML, actual keys in `.env`, resolved at runtime by `build_brokers()`. Clean separation.
2. **`resolve_sqlite_account_id()`** — 5 tests covering type safety, whitespace, empty string, and exception handling. Defensive programming at its best.
3. **Cross-account wash boundary tests** — day 30 IS flagged, day 31 is NOT. Explicit boundary validation is exactly what tax-sensitive code needs.
4. **Backward compatibility guarantee** — 33 legacy tests running unchanged prove the `account_id='default'` migration is safe.
5. **Graceful degradation** — missing env vars raise `ValueError` with a clear message; unknown accounts return empty results instead of errors.

---

## Summary

| Category | Finding Count | Severity |
|----------|--------------|----------|
| Security issues | 0 | All checks passed |
| Production code bugs | 0 | Clean |
| Architecture issues | 0 | Account isolation correct at every layer |
| Type annotation | 1 | MEDIUM (static analysis only) |
| Documentation gaps | 4 | LOW |
| Test regressions | 0 | 33 legacy tests unchanged and passing |

---

## Recommended Fixes

1. **M1** — Fix type annotation in `count_equity_snapshots()` (1 min)
2. **L1–L3** — Add docstrings/comments for multi-account distinction (10 min)
3. **L4** — Add single-account-in-list runner test (5 min)

Everything else is clean. This is production-ready.
