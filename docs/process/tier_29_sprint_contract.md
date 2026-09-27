# Tier 29 Sprint Contract — Dashboard & API Hardening

**Depends on:** Tier 28 complete (665+ tests green, lint clean, limit/stop orders + PDT tracking)
**Governance:** `AGENTS.md` — TDD (failing test first), ruff clean, no `print()` in production paths

---

## 1. Sprint goal

Harden the dashboard and API for real-world deployment: add dashboard authentication so pages aren't publicly accessible, add API pagination so large datasets don't choke responses, establish a Streamlit test foundation so dashboard logic is validated, fix API error response consistency, and close Tier 28 review items.

---

## 2. In scope

### 29A — Dashboard Authentication

- **Intent:** Dashboard is currently wide open — anyone on the network can view portfolio, trades, and balances. Add a session-based login gate.
- **Implementation:** Streamlit's `st.session_state` + a configurable token/password
- **New config** (`src/config.py` → `DashboardConfig`):
  ```python
  class DashboardConfig(BaseModel):
      port: int = Field(default=8501, ge=1024, le=65535)
      refresh_interval_seconds: int = Field(default=30, ge=5, le=300)
      auth_enabled: bool = False
      auth_password: str = Field(
          default="",
          description="Dashboard login password. Set via DASHBOARD_PASSWORD env var. Empty = no auth."
      )
  ```
- **New module:** `src/dashboard/auth.py`
  ```python
  def require_auth(settings: Settings) -> bool:
      """Check auth and render login form if needed. Returns True if authenticated."""
  ```
  - If `auth_enabled: false` or `auth_password` is empty → return `True` (no gate)
  - If `auth_enabled: true` → check `st.session_state.get("dashboard_authenticated")`
  - If not authenticated → render `st.text_input(type="password")` + `st.button("Login")`
  - On correct password → set `st.session_state["dashboard_authenticated"] = True`, `st.rerun()`
  - On incorrect → `st.error("Incorrect password")`
  - Return `False` to prevent page content from rendering
- **Dashboard integration** (`src/dashboard/app.py` + all pages):
  - Add at top of `app.py` and each page file:
    ```python
    from src.dashboard.auth import require_auth
    if not require_auth(settings):
        st.stop()
    ```
- **Env var:** Add `DASHBOARD_PASSWORD=` to `.env.example`
- **Config in `settings.yaml`:**
  ```yaml
  dashboard:
    port: 8501
    refresh_interval_seconds: 30
    auth_enabled: false
    auth_password: ""
  ```
- **Tests:**
  - `test_require_auth_disabled()` — returns True when auth_enabled=False
  - `test_require_auth_empty_password()` — returns True when password is empty
  - `test_require_auth_authenticated_session()` — returns True when session state has auth flag
  - `test_require_auth_not_authenticated()` — returns False when no session state

### 29B — API Pagination

- **Intent:** API routes use `limit` but no `offset` — clients can't page through large result sets. Add offset-based pagination.
- **SQLiteStore changes** (`src/data/storage/sqlite_store.py`):
  - Update these methods to accept `offset: int = 0`:
    - `get_signals(limit, offset)` → `SELECT ... ORDER BY id DESC LIMIT ? OFFSET ?`
    - `get_executions(limit, offset)` → same pattern
    - `get_alerts(limit, offset)` → same pattern
    - `get_backtest_runs(limit, offset)` → same pattern
    - `get_equity_snapshots(limit, offset)` → same pattern (if not already paginated)
  - Add `count_signals()`, `count_executions()`, `count_alerts()` for total count
- **API route changes:**
  - Add `offset: int = Query(default=0, ge=0)` to paginated endpoints
  - Return pagination metadata in response:
    ```json
    {
      "items": [...],
      "total": 1234,
      "limit": 100,
      "offset": 0
    }
    ```
  - Affected routes: `/trades/signals`, `/trades/executions`, `/alerts`, `/backtests`
- **Config:** No new config needed — limit/offset are query params
- **Tests:**
  - `test_get_signals_with_offset()` — verify correct rows returned
  - `test_get_signals_offset_beyond_total()` — empty list, not error
  - `test_api_trades_pagination_metadata()` — verify response includes total/limit/offset
  - `test_api_pagination_defaults()` — offset=0 when not specified

### 29C — Streamlit Test Foundation

- **Intent:** Zero dashboard tests exist. Establish a test foundation for the most critical pages using `streamlit.testing.v1` or by testing the data-fetching logic separately.
- **Approach:** Since Streamlit pages mix UI and data, extract testable logic where possible and test the data layer.
- **New directory:** `tests/unit/dashboard/`
- **New files:**
  - `tests/unit/dashboard/conftest.py` — fixtures with mock `SQLiteStore` and `ParquetStore`
  - `tests/unit/dashboard/test_formatting.py` — test `formatting.py` helpers (pure functions, easy to test)
  - `tests/unit/dashboard/test_paths.py` — test path resolution logic
  - `tests/unit/dashboard/test_charts.py` — test `apply_hub_layout()` returns valid Plotly layout
  - `tests/unit/dashboard/test_auth.py` — test `require_auth()` logic with mocked `st.session_state`
- **What NOT to test:** Full page rendering with Streamlit components (too brittle, requires full Streamlit runtime). Focus on testable pure functions and data queries.
- **Tests (~15):**
  - `test_format_timestamp()` — various timestamp formats
  - `test_format_side()` — "buy"→green, "sell"→red styling
  - `test_hub_sqlite_path()` — correct path resolution
  - `test_hub_parquet_path()` — correct path resolution
  - `test_apply_hub_layout()` — returns dict with expected keys (template, font, colors)
  - `test_require_auth_*()` — 4 auth tests (from 29A)
  - Plus edge cases

### 29D — API Error Response Consistency

- **Intent:** API routes return mixed error formats — some return plain strings, some return `{"detail": "..."}` (FastAPI default). Standardize to a consistent JSON error schema.
- **New error model** (`src/api/errors.py` or in `app.py`):
  ```python
  class APIError(BaseModel):
      error: str
      detail: str | None = None
      code: str | None = None
  ```
- **Exception handlers** (`src/api/app.py`):
  - `@app.exception_handler(ValueError)` → 400 with `{"error": "Bad request", "detail": str(exc)}`
  - `@app.exception_handler(404)` → 404 with `{"error": "Not found"}`
  - `@app.exception_handler(Exception)` → 500 with `{"error": "Internal server error"}` (no detail in prod to avoid leaking internals)
- **Tests:**
  - `test_api_400_returns_json_error()` — bad query param → consistent JSON
  - `test_api_404_returns_json_error()` — unknown route → consistent JSON
  - `test_api_500_returns_json_error()` — internal error → consistent JSON without stack trace

### 29E — Tier 28 Review Fixes

Close items from `tier_28_review.md`:

| ID | Fix | File |
|----|-----|------|
| H1 | Fix TOCTOU in `cancel_stale_orders()`: call `get_order_status()` when cancel returns False, update SQLite | `src/execution/order_manager.py` |
| M1 | Enumerate all known Alpaca order statuses in `_bucket_order_status()` (new, accepted, partially_filled, filled, cancelled, expired, rejected, pending_cancel, pending_replace) | `src/execution/order_manager.py` |
| L3 | Add multi-symbol PDT test (buy+sell SPY and buy+sell QQQ = 2 day trades) | `tests/unit/risk/test_pdt_tracker.py` |
| L4 | Add partial fill cancellation test | `tests/unit/execution/test_order_status.py` |

---

## 3. Out of scope (deferred beyond Tier 29)

- Multi-user dashboard (roles, per-user views)
- OAuth/SSO integration for dashboard
- API rate limiting (slowapi middleware)
- API request logging middleware
- WebSocket streaming for real-time dashboard updates
- Dashboard page-level caching (`@st.cache_data`)
- Cursor-based pagination (offset is sufficient for current scale)
- Full Streamlit page rendering tests (requires heavy test infrastructure)
- Multi-account support

---

## 4. Acceptance criteria

- [ ] `ruff check src/ tests/` clean
- [ ] Full `pytest` suite green; target **700+ tests** (currently 665, adding ~35)
- [ ] Dashboard login gate works when `auth_enabled: true` + password set
- [ ] Dashboard is unprotected (no change) when `auth_enabled: false` (default)
- [ ] API pagination returns `{items, total, limit, offset}` on paginated endpoints
- [ ] `offset=0` default preserves existing API response behavior
- [ ] Dashboard test directory exists with 15+ tests for formatting, paths, charts, auth
- [ ] API error responses return consistent JSON schema on 400/404/500
- [ ] All Tier 28 review items (H1, M1, L3, L4) resolved
- [ ] `BUILD_STATE.md` updated with Tier 29 entry
- [ ] No `print()` in production paths; use `logging`

---

## 5. TDD task order (suggested)

**Phase 1 — Tier 28 Review Fixes (29E)**

1. Fix TOCTOU in `cancel_stale_orders()` → add `get_order_status()` fallback
2. Enumerate Alpaca statuses in `_bucket_order_status()` → test with known statuses
3. Add multi-symbol PDT test → verify 2 day trades counted
4. Add partial fill cancellation test → verify status update

**Phase 2 — Dashboard Auth (29A)**

5. Write `test_require_auth_disabled()` → returns True
6. Write `test_require_auth_empty_password()` → returns True
7. Write `test_require_auth_authenticated_session()` → returns True
8. Write `test_require_auth_not_authenticated()` → returns False
9. Implement `src/dashboard/auth.py` → tests green
10. Add `DashboardConfig` to config → update `settings.yaml`
11. Wire `require_auth()` into `app.py` and all pages
12. Add `DASHBOARD_PASSWORD=` to `.env.example`

**Phase 3 — Dashboard Tests Foundation (29C)**

13. Create `tests/unit/dashboard/conftest.py` with fixtures
14. Write `test_format_timestamp()` → test formatting helpers
15. Write `test_format_side()` → test side formatting
16. Write `test_hub_sqlite_path()` → test path resolution
17. Write `test_apply_hub_layout()` → test chart layout
18. Run dashboard tests → all green

**Phase 4 — API Pagination (29B)**

19. Write `test_get_signals_with_offset()` → add `offset` param to SQLiteStore
20. Write `test_get_signals_offset_beyond_total()` → empty list
21. Write `test_api_trades_pagination_metadata()` → update route to return `{items, total, limit, offset}`
22. Write `test_api_pagination_defaults()` → verify offset=0 default
23. Add `offset` to remaining paginated routes (alerts, backtests)
24. Add `count_*()` methods to SQLiteStore

**Phase 5 — API Error Consistency (29D)**

25. Write `test_api_400_returns_json_error()` → add exception handlers
26. Write `test_api_404_returns_json_error()` → add 404 handler
27. Write `test_api_500_returns_json_error()` → add generic handler
28. Implement error handlers in `app.py`

**Phase 6 — Finalize**

29. Run full `pytest` suite → all green
30. Run `ruff check src/ tests/` → clean
31. Update `BUILD_STATE.md` with Tier 29 entry

---

## 6. Risks and mitigations

| Risk | Mitigation |
|------|------------|
| Dashboard auth password stored in plaintext in `.env` | Same pattern as `ALPACA_SECRET_KEY` — `.env` is gitignored; personal deployment; upgrade to hashed passwords later if multi-user needed |
| `st.session_state` clears on browser refresh | User re-enters password; session persists as long as tab is open (Streamlit default behavior) |
| Pagination changes break existing API clients | `offset=0` default means existing calls (no offset param) return identical results |
| Dashboard tests are fragile with Streamlit mocking | Test pure functions only (formatting, paths, charts); don't test UI rendering |
| Error handlers catch too broadly | Generic 500 handler only catches unhandled exceptions; typed handlers (ValueError, 404) take priority |
| `cancel_stale_orders` fix adds a broker call per stale order | Only fires when cancel returns False (rare); one `get_order_status()` call per stale order is acceptable |

---

## 7. Definition of done

- All 5 sub-scopes (29A–29E) implemented with passing tests
- 700+ total tests, all green, ruff clean
- Zero regressions on existing 665 tests
- Dashboard auth gate functional (manual verification: set password, verify login required)
- API pagination works on all list endpoints
- Dashboard test directory established with 15+ tests
- API error responses return consistent JSON
- Tier 28 review items resolved (TOCTOU fix, status enumeration, PDT + partial fill tests)
- `BUILD_STATE.md` updated with Tier 29 summary

---

## Key files

| File | Change |
|------|--------|
| `src/dashboard/auth.py` | **NEW** — `require_auth()` login gate |
| `src/dashboard/app.py` | Wire auth gate at top |
| `src/dashboard/pages/*.py` | Wire auth gate at top of each page |
| `src/config.py` | Add/extend `DashboardConfig` with auth fields |
| `config/settings.yaml` | Add dashboard auth section |
| `.env.example` | Add `DASHBOARD_PASSWORD=` |
| `src/data/storage/sqlite_store.py` | Add `offset` param to paginated methods; add `count_*()` methods |
| `src/api/routes/trades.py` | Add `offset` query param, return pagination metadata |
| `src/api/routes/alerts_route.py` | Same pagination pattern |
| `src/api/routes/backtests.py` | Same pagination pattern |
| `src/api/app.py` | Add exception handlers for 400/404/500 |
| `src/execution/order_manager.py` | Fix TOCTOU in `cancel_stale_orders()`, enumerate statuses |
| `tests/unit/dashboard/` | **NEW** directory — conftest, test_formatting, test_paths, test_charts, test_auth |
| `tests/unit/api/test_pagination.py` | **NEW** — pagination tests |
| `tests/unit/api/test_error_responses.py` | **NEW** — error consistency tests |
| `tests/unit/risk/test_pdt_tracker.py` | Add multi-symbol test |
| `tests/unit/execution/test_order_status.py` | Add partial fill test |

---

## Verification

1. `ruff check src/ tests/` — clean
2. `pytest tests/ -x` — 700+ tests, all green
3. Set `DASHBOARD_PASSWORD=test123` in `.env`, `auth_enabled: true` in settings → dashboard shows login form
4. Enter correct password → dashboard renders; refresh → still logged in (session persists)
5. `curl localhost:8000/api/trades/executions?limit=10&offset=0` → returns `{items: [...], total: N, limit: 10, offset: 0}`
6. `curl localhost:8000/api/trades/executions?limit=10&offset=9999` → returns `{items: [], total: N, limit: 10, offset: 9999}`
7. `curl localhost:8000/api/nonexistent` → returns `{"error": "Not found"}` (not HTML)
