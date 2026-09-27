# Tier 30 Sprint Contract — Multi-Account Foundation & Per-Account Tax

**Depends on:** Tier 29 complete (700+ tests green, lint clean, dashboard auth + pagination)
**Governance:** `AGENTS.md` — TDD (failing test first), ruff clean, no `print()` in production paths

---

## 1. Sprint goal

Add multi-account support so the platform can manage separate Alpaca accounts (e.g., individual brokerage, IRA, joint) with isolated lot tracking, per-account execution, and per-account tax reporting. Also enhance Form 8949 CSV export to support per-account filtering and cross-account wash sale detection.

This is the foundation tier — TurboTax import format, advanced tax software integration, and multi-account ensemble are deferred to Tier 31.

---

## 2. In scope

### 30A — Multi-Account Configuration

- **Intent:** Support multiple Alpaca accounts with separate API keys, each identified by name and type.
- **New config** (`src/config.py`):
  ```python
  class AccountConfig(BaseModel):
      name: str = Field(description="Human label: 'Main', 'IRA', 'Joint'")
      account_type: Literal["brokerage", "ira_traditional", "ira_roth", "joint"] = "brokerage"
      api_key_env: str = Field(description="Env var name for API key, e.g. ALPACA_ACCOUNT_MAIN_KEY")
      api_secret_env: str = Field(description="Env var name for secret, e.g. ALPACA_ACCOUNT_MAIN_SECRET")
  ```
  - Keys are NOT stored in YAML — only env var names are configured. Actual keys live in `.env`.
  - This avoids secrets in version control while supporting N accounts.
- **Updated `BrokerConfig`:**
  ```python
  class BrokerConfig(BaseModel):
      provider: str = "alpaca"
      paper_trading: bool = True
      accounts: list[AccountConfig] = Field(default_factory=list)
      default_account: str = Field(default="", description="Name of primary trading account")
  ```
- **Backward compatibility:** When `accounts` is empty, fall back to legacy `ALPACA_API_KEY` / `ALPACA_SECRET_KEY` env vars with `account_id="default"`. Existing single-account users change nothing.
- **Config in `settings.yaml`:**
  ```yaml
  broker:
    provider: "alpaca"
    paper_trading: true
    default_account: "Main"
    accounts:
      - name: "Main"
        account_type: "brokerage"
        api_key_env: "ALPACA_ACCOUNT_MAIN_KEY"
        api_secret_env: "ALPACA_ACCOUNT_MAIN_SECRET"
      # - name: "IRA"
      #   account_type: "ira_traditional"
      #   api_key_env: "ALPACA_ACCOUNT_IRA_KEY"
      #   api_secret_env: "ALPACA_ACCOUNT_IRA_SECRET"
  ```
- **`.env.example` additions:**
  ```bash
  # Multi-account (optional — leave blank to use legacy ALPACA_API_KEY)
  # ALPACA_ACCOUNT_MAIN_KEY=
  # ALPACA_ACCOUNT_MAIN_SECRET=
  # ALPACA_ACCOUNT_IRA_KEY=
  # ALPACA_ACCOUNT_IRA_SECRET=
  ```
- **Tests:**
  - `test_account_config_parses_correctly()` — YAML with 2 accounts
  - `test_empty_accounts_falls_back_to_legacy()` — backward compat
  - `test_account_keys_loaded_from_env()` — env var resolution

### 30B — Database Schema: `account_id` Column

- **Intent:** Partition all execution, lot, and snapshot data by account.
- **SQLite migrations** (`src/data/storage/sqlite_store.py`):
  Add `account_id TEXT NOT NULL DEFAULT 'default'` to:
  - `trade_signals`
  - `trade_executions`
  - `equity_snapshots`
  - `tax_lots_open`
  - `tax_lots_closed`
  Add indexes: `CREATE INDEX idx_*_account ON *(account_id)`
- **LotLedger** (`src/portfolio/tax_lots.py`):
  - All methods accept `account_id: str = "default"` parameter
  - `record_buy()`, `record_sell()`: write with account_id
  - `get_open_lots()`, `get_closed_lots()`: filter by account_id (None = all accounts)
- **SQLiteStore**:
  - `log_signal()`, `log_execution()`, `log_equity_snapshot()`: accept `account_id`
  - `get_signals()`, `get_executions()`, `get_equity_snapshots()`: accept optional `account_id` filter
  - Count methods: accept optional `account_id`
- **Backward compat:** Default value `'default'` means existing data and single-account usage work unchanged.
- **Tests:**
  - `test_lot_ledger_partitions_by_account()` — buy in account A, buy in account B, get_open_lots(A) returns only A's lots
  - `test_sqlite_signals_filtered_by_account()` — same partition pattern
  - `test_migration_adds_account_id_with_default()` — existing rows get 'default'

### 30C — Multi-Account Broker Factory & Workflow

- **Intent:** Create one `AlpacaBrokerAdapter` per account. Run trading cycles per account.
- **Broker interface addition** (`src/execution/broker_interface.py`):
  ```python
  @abstractmethod
  def get_account_id(self) -> str:
      """Return the account identifier this adapter is connected to."""
  ```
- **Broker factory** (`src/automation/runner.py` or new `src/execution/account_factory.py`):
  ```python
  def build_brokers(settings: Settings) -> dict[str, BrokerInterface]:
      """Create one broker adapter per configured account."""
  ```
  - Reads `AccountConfig.api_key_env` / `api_secret_env` from `os.environ`
  - Creates `AlpacaBrokerAdapter.create(key, secret, paper=...)` per account
  - Falls back to legacy single env vars when `accounts` is empty
- **Workflow changes** (`src/automation/workflows.py`):
  - `TradingWorkflow.__init__()` accepts `account_id: str = "default"`
  - Passes `account_id` to `lot_ledger`, `sqlite_store.log_*()`, `order_manager`
  - All logging includes `account_id` for audit trail
- **Runner changes** (`src/automation/runner.py`):
  - Build `dict[str, BrokerInterface]` from `build_brokers()`
  - Create one `TradingWorkflow` per account
  - Scheduler runs each account's cycle sequentially (not parallel — avoid race conditions on shared SQLite)
  - Log: `"Running cycle for account: %s"` before each
- **Tests:**
  - `test_build_brokers_from_accounts()` — 2 accounts → 2 adapters
  - `test_build_brokers_legacy_fallback()` — empty accounts → 1 adapter with 'default'
  - `test_workflow_uses_account_id_for_logging()` — verify account_id in logged signals/executions
  - `test_runner_iterates_accounts()` — mock 2 accounts, verify both cycle

### 30D — Per-Account Tax Reporting

- **Intent:** Tax reports should be filterable by account. Cross-account wash sales must be detectable (IRS treats household as single taxpayer).
- **Per-account tax summary** (`src/reporting/tax_report.py`):
  ```python
  def compute_tax_summary(
      year: int,
      closed_lots: list[ClosedLot],
      *,
      account_id: str | None = None,  # None = all accounts consolidated
  ) -> TaxSummary:
  ```
- **Cross-account wash sale detection** (`src/portfolio/wash_sales.py`):
  ```python
  def detect_cross_account_wash_sales(
      all_accounts_closed: dict[str, list[ClosedLot]],
      all_accounts_open: dict[str, list[TaxLot]],
      *,
      window_days: int = 30,
  ) -> list[CrossAccountWashSale]:
      """Detect losses in account A replaced by purchases in account B within ±30 days."""
  ```
  - Returns list of `CrossAccountWashSale(selling_account, buying_account, symbol, loss_amount, replacement_date)`
  - Informational — flags for user review, does not auto-adjust basis (IRS determination is complex)
- **Form 8949 per-account export:**
  - `export_form_8949_csv()` accepts optional `account_id` filter
  - When filtering, CSV header includes account name
- **Dashboard** (`src/dashboard/pages/07_taxes.py`):
  - Add account selector dropdown (or "All Accounts" consolidated)
  - Show cross-account wash sale warnings in an expander
  - Per-account CSV download buttons
- **API** (`src/api/routes/tax.py`):
  - Add `account_id: str | None = Query(default=None)` to tax endpoints
  - `GET /api/tax/summary?year=2025&account_id=Main`
  - `GET /api/tax/cross-account-washes?year=2025` — returns flagged wash sales
- **Tests:**
  - `test_tax_summary_filtered_by_account()` — only includes that account's lots
  - `test_tax_summary_all_accounts_consolidated()` — includes all lots
  - `test_cross_account_wash_detected()` — sell in A, buy in B within 30 days → flagged
  - `test_cross_account_wash_not_detected_outside_window()` — 45 days apart → not flagged
  - `test_form_8949_per_account_csv()` — CSV includes only filtered lots
  - `test_api_tax_summary_with_account_filter()` — API returns filtered summary

### 30E — Tier 29 Review Fix

| ID | Fix | File |
|----|-----|------|
| S1 | Replace `==` with `hmac.compare_digest()` for timing-safe password comparison | `src/dashboard/auth.py` |

---

## 3. Out of scope (deferred to Tier 31+)

- TurboTax `.txf` / `.tax` import file format
- H&R Block import format
- Schedule D summary generation
- 1099-B reconciliation
- IRA contribution/distribution tracking
- Roth conversion tracking
- Multi-account strategy ensemble (strategies spanning accounts)
- Parallel account cycle execution (sequential is sufficient)
- Account-specific strategy configuration (all accounts use same strategies)
- Cross-account rebalancing
- Custody / regulatory compliance features

---

## 4. Acceptance criteria

- [ ] `ruff check src/ tests/` clean
- [ ] Full `pytest` suite green; target **740+ tests** (currently 700, adding ~40)
- [ ] Empty `accounts` list preserves exact single-account behavior — no regressions
- [ ] 2-account config creates 2 broker adapters, runs 2 cycles, logs with correct `account_id`
- [ ] Lot ledger partitions by `account_id` — lots from account A invisible when querying account B
- [ ] Tax summary filterable by `account_id` — consolidated view when None
- [ ] Cross-account wash sale detection finds ±30-day loss-replacement across accounts
- [ ] Form 8949 CSV export supports per-account filtering
- [ ] Dashboard tax page has account selector
- [ ] API tax endpoints accept `account_id` query param
- [ ] `hmac.compare_digest()` used for dashboard password comparison
- [ ] `BUILD_STATE.md` updated with Tier 30 entry
- [ ] No `print()` in production paths; use `logging`

---

## 5. TDD task order (suggested)

**Phase 1 — Tier 29 Fix + Config (30A + 30E)**

1. Fix `auth.py` timing-safe comparison → verify test still passes
2. Write `test_account_config_parses_correctly()` → add `AccountConfig` + `BrokerConfig.accounts`
3. Write `test_empty_accounts_falls_back_to_legacy()` → implement fallback logic
4. Write `test_account_keys_loaded_from_env()` → env var resolution
5. Update `settings.yaml` and `.env.example`

**Phase 2 — Database Schema (30B)**

6. Write `test_migration_adds_account_id_with_default()` → add migrations
7. Write `test_lot_ledger_partitions_by_account()` → add `account_id` to LotLedger methods
8. Write `test_sqlite_signals_filtered_by_account()` → add `account_id` to SQLiteStore methods
9. Verify existing tests still pass with `account_id='default'`

**Phase 3 — Broker Factory & Workflow (30C)**

10. Write `test_build_brokers_from_accounts()` → implement broker factory
11. Write `test_build_brokers_legacy_fallback()` → verify single-account fallback
12. Write `test_workflow_uses_account_id_for_logging()` → add `account_id` to TradingWorkflow
13. Write `test_runner_iterates_accounts()` → update runner to iterate accounts
14. Add `get_account_id()` to BrokerInterface + AlpacaBrokerAdapter

**Phase 4 — Tax Reporting (30D)**

15. Write `test_tax_summary_filtered_by_account()` → add account filter to `compute_tax_summary()`
16. Write `test_tax_summary_all_accounts_consolidated()` → None = all
17. Write `test_cross_account_wash_detected()` → implement `detect_cross_account_wash_sales()`
18. Write `test_cross_account_wash_not_detected_outside_window()` → 45 days
19. Write `test_form_8949_per_account_csv()` → add account filter to export
20. Write `test_api_tax_summary_with_account_filter()` → add query param to API
21. Update dashboard tax page with account selector

**Phase 5 — Finalize**

22. Run full `pytest` suite → all green
23. Run `ruff check src/ tests/` → clean
24. Update `BUILD_STATE.md` with Tier 30 entry

---

## 6. Risks and mitigations

| Risk | Mitigation |
|------|------------|
| Adding `account_id` to existing tables breaks queries | `DEFAULT 'default'` on all migrations; existing data gets 'default'; all queries work unchanged |
| Sequential account cycles slow total cycle time | Each account cycle is ~10-30s; 2 accounts = ~1 min total. Acceptable for cron-based scheduling |
| Cross-account wash sale detection has false positives | Informational only — flags for review, does not auto-adjust cost basis. User makes final determination |
| Multiple Alpaca keys fail in parallel | Adapters created sequentially; each has its own `TradingClient` instance; no shared state |
| Config complexity confuses single-account users | Empty `accounts` list = legacy behavior; existing `.env` works unchanged; multi-account is opt-in |
| Lot ledger partition bug causes wrong P&L | All queries include `WHERE account_id = ?`; integration test verifies partition isolation |

---

## 7. Definition of done

- All 5 sub-scopes (30A–30E) implemented with passing tests
- 740+ total tests, all green, ruff clean
- Zero regressions for single-account users (empty `accounts` list)
- Multi-account config creates separate broker adapters and runs isolated cycles
- Tax reporting supports per-account and consolidated views
- Cross-account wash sales detected and flagged
- Dashboard tax page has account selector with CSV export per account
- `BUILD_STATE.md` updated with Tier 30 summary

---

## Key files

| File | Change |
|------|--------|
| `src/config.py` | Add `AccountConfig`, update `BrokerConfig` with `accounts` list |
| `config/settings.yaml` | Add `broker.accounts` section |
| `.env.example` | Add multi-account env var pattern |
| `src/execution/broker_interface.py` | Add `get_account_id()` abstract method |
| `src/execution/alpaca_adapter.py` | Accept + return `account_id` |
| `src/execution/account_factory.py` | **NEW** — `build_brokers()` factory |
| `src/data/storage/sqlite_store.py` | Add `account_id` column migrations + filter params |
| `src/portfolio/tax_lots.py` | Add `account_id` to all methods |
| `src/portfolio/wash_sales.py` | Add `detect_cross_account_wash_sales()` |
| `src/automation/workflows.py` | Accept `account_id`, pass to all storage calls |
| `src/automation/runner.py` | Build multi-account brokers, iterate account cycles |
| `src/reporting/tax_report.py` | Add `account_id` filter to summary + export |
| `src/api/routes/tax.py` | Add `account_id` query param + cross-account wash endpoint |
| `src/dashboard/pages/07_taxes.py` | Account selector, cross-account wash expander |
| `src/dashboard/auth.py` | Fix `hmac.compare_digest()` (Tier 29 review) |
| `tests/unit/portfolio/test_tax_lots_multi_account.py` | **NEW** — partition tests |
| `tests/unit/portfolio/test_cross_account_wash.py` | **NEW** — wash detection tests |
| `tests/unit/execution/test_account_factory.py` | **NEW** — broker factory tests |
| `tests/unit/automation/test_multi_account_runner.py` | **NEW** — multi-account cycle tests |
| `tests/unit/reporting/test_tax_report_multi_account.py` | **NEW** — per-account tax tests |

---

## Verification

1. `ruff check src/ tests/` — clean
2. `pytest tests/ -x` — 740+ tests, all green
3. With empty `accounts` list: `financialhub once` works identically to Tier 29 (backward compat)
4. With 2 accounts configured: `financialhub once` logs `"Running cycle for account: Main"` then `"Running cycle for account: IRA"`
5. SQLite: `SELECT DISTINCT account_id FROM trade_executions` returns both account names
6. Dashboard tax page: account selector shows "All Accounts" + individual accounts; CSV downloads per-account
7. `GET /api/tax/summary?year=2025&account_id=Main` returns only Main account lots
8. `GET /api/tax/cross-account-washes?year=2025` returns flagged wash sales (if any)
