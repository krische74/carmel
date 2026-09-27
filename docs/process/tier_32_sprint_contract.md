# Tier 32 Sprint Contract — Operational Polish & Tax Sophistication

**Depends on:** Tier 31 complete (753+ tests green, lint clean, tax exports done)
**Governance:** `AGENTS.md` — TDD (failing test first), ruff clean, no `print()` in production paths

---

## 1. Sprint goal

Polish the platform for daily unattended use: wire email notifications into the trading cycle, add weekly digest summaries, implement loss carryforward tracking for multi-year tax planning, add IRA-specific tax treatment, add a backtest warm-up period so indicators settle before signals fire, persist dashboard chat history, and add a paper-to-live transition safety checklist. Also close Tier 31 review items.

---

## 2. In scope

### 32A — Email Notifier Wiring & Digest Notifications

- **Intent:** `EmailNotifier` class exists but is never instantiated or called. Users without webhooks have no alert channel. Also, `digest_cron` is configured but no digest logic exists.
- **Email wiring** (`src/automation/runner.py`):
  - When `notification.email_enabled: true` in config, instantiate `EmailNotifier` and add to `CompositeNotifier`
  - `CompositeNotifier` already fans out to all registered notifiers — just add email alongside webhook
- **Digest accumulation** (new: `src/automation/digest.py`):
  ```python
  def generate_digest_summary(
      sqlite_store: SQLiteStore,
      *,
      lookback_hours: int = 168,  # 7 days default
      account_id: str | None = None,
  ) -> DigestSummary:
      """Aggregate recent trades, alerts, P&L for a digest email/webhook."""
  ```
  - `DigestSummary` model: trade_count, alert_count, net_pnl, top_movers, regime_summary, open_positions_count
  - Rendered as plain-text or HTML for email / webhook payload
- **Scheduler wiring** (`src/automation/runner.py`):
  - Register digest job on `settings.scheduler.digest_cron` (already in config)
  - Job calls `generate_digest_summary()` → sends via `CompositeNotifier`
- **Config:** Already exists — `notification.email_enabled`, `smtp_*`, `digest_cron`. No new config needed.
- **Tests:**
  - `test_email_notifier_instantiated_when_enabled()` — verify runner creates EmailNotifier
  - `test_digest_summary_aggregates_recent_data()` — 5 trades + 3 alerts → correct counts
  - `test_digest_summary_empty_when_no_activity()` — no trades → zero counts
  - `test_digest_cron_registered()` — scheduler has digest job when cron is set

### 32B — Loss Carryforward Tracking

- **Intent:** If a user harvests $10k in losses but only offsets $5k in gains, the remaining $5k carries forward to next year (IRS allows $3k/year against ordinary income + unlimited against future gains). Currently not tracked.
- **New SQLite table** (`src/data/storage/sqlite_store.py`):
  ```sql
  CREATE TABLE IF NOT EXISTS loss_carryforward (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      year INTEGER NOT NULL,
      account_id TEXT NOT NULL DEFAULT 'default',
      short_term_carryforward REAL NOT NULL DEFAULT 0.0,
      long_term_carryforward REAL NOT NULL DEFAULT 0.0,
      computed_at TEXT NOT NULL
  );
  ```
- **New function** (`src/reporting/tax_report.py`):
  ```python
  def compute_loss_carryforward(
      tax_summary: TaxSummary,
      prior_carryforward: LossCarryforward | None = None,
      *,
      annual_deduction_limit: float = 3_000.0,
  ) -> LossCarryforward:
      """Compute remaining loss carryforward after applying gains + $3k deduction."""
  ```
  - Logic: apply ST losses against ST gains first, then LT losses against LT gains, then cross-term netting, then $3k ordinary income deduction, remainder carries forward
- **Dashboard** (`src/dashboard/pages/07_taxes.py`):
  - New metric card: "Loss Carryforward: $X,XXX" showing available carryforward
  - Expander: year-by-year carryforward table
- **API:** `GET /api/tax/carryforward?year=2025&account_id=Main`
- **Tests:**
  - `test_carryforward_no_losses()` — all gains → zero carryforward
  - `test_carryforward_losses_exceed_gains()` — $10k loss, $5k gain, $3k deduction → $2k carries forward
  - `test_carryforward_with_prior_year()` — prior $2k + current $1k loss → $3k available
  - `test_carryforward_per_account()` — per-account isolation
  - `test_carryforward_cross_term_netting()` — ST loss offsets LT gain after same-term netting

### 32C — IRA-Specific Tax Treatment

- **Intent:** IRA accounts (traditional/Roth) have different tax treatment. Traditional IRA gains are tax-deferred (no current-year tax). Roth IRA gains are tax-free (qualified). Tax reports should distinguish these.
- **Changes to `src/reporting/tax_report.py`:**
  - `TaxSummary` gets new field: `account_type: str | None = None`
  - When `account_type` is `"ira_traditional"` or `"ira_roth"`:
    - Form 8949 CSV header includes note: "IRA — Not Reportable on Form 8949 (informational only)"
    - Schedule D excludes IRA lots from taxable totals
    - Dashboard shows IRA accounts separately with "Tax-Deferred" / "Tax-Free" label
  - Consolidated reports: IRA lots shown but clearly labeled as non-taxable
- **Config wiring:**
  - `AccountConfig.account_type` already exists (`brokerage`, `ira_traditional`, `ira_roth`, `joint`)
  - Pass `account_type` to `compute_tax_summary()` from workflow/API
- **Dashboard** (`src/dashboard/pages/07_taxes.py`):
  - IRA accounts show "(Tax-Deferred)" or "(Tax-Free)" badge next to account name
  - Consolidated view: separate section for "Taxable Accounts" and "Tax-Advantaged Accounts"
- **Tests:**
  - `test_ira_traditional_excluded_from_taxable_summary()` — IRA lots not in taxable totals
  - `test_ira_roth_marked_tax_free()` — Roth lots labeled correctly
  - `test_brokerage_included_in_taxable_summary()` — brokerage lots are taxable (no change)
  - `test_consolidated_separates_taxable_and_ira()` — consolidated view has both sections
  - `test_form_8949_ira_header_note()` — CSV includes IRA informational header

### 32D — Backtest Warm-Up Period

- **Intent:** Backtests starting on day 1 have undefined indicators (SMA-200 needs 200 bars). Currently, strategies return `None` or skip signals during warm-up, but the backtest engine records equity from day 1 (flat line during warm-up is misleading).
- **Config addition** (`src/config.py` → `BacktestConfig` or inline):
  ```python
  warmup_bars: int = Field(default=200, ge=0, le=500, description="Skip first N bars for indicator warm-up")
  ```
- **Engine change** (`src/backtesting/engine.py`):
  - After building date range, skip first `warmup_bars` trading days before starting rebalance/equity tracking
  - Equity curve starts after warm-up period (cleaner charts, realistic performance)
- **Config in `settings.yaml`:** No new section — add inline to backtest config or strategy config
- **Tests:**
  - `test_backtest_warmup_skips_initial_bars()` — equity curve starts after warmup
  - `test_backtest_warmup_zero_starts_immediately()` — warmup=0 preserves current behavior
  - `test_backtest_warmup_longer_than_data()` — warmup > data length → empty result (not crash)

### 32E — Dashboard Chat Persistence

- **Intent:** LLM Q&A page (`08_ask.py`) uses `st.session_state` for chat history — lost on browser close. Persist to SQLite for multi-session conversations.
- **New SQLite table** (`src/data/storage/sqlite_store.py`):
  ```sql
  CREATE TABLE IF NOT EXISTS chat_history (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      question TEXT NOT NULL,
      answer TEXT NOT NULL,
      asked_at TEXT NOT NULL,
      account_id TEXT NOT NULL DEFAULT 'default'
  );
  ```
- **SQLiteStore methods:**
  ```python
  def log_chat(self, question: str, answer: str, *, account_id: str = "default") -> None:
  def get_chat_history(self, *, limit: int = 50, account_id: str | None = None) -> list[dict]:
  ```
- **Dashboard change** (`src/dashboard/pages/08_ask.py`):
  - On page load: populate `st.session_state.qa_history` from SQLite (last 50)
  - On new Q&A: write to SQLite, then append to session state
  - Display: existing chat rendering (no UI change needed)
- **Tests:**
  - `test_log_chat_and_retrieve()` — write + read round-trip
  - `test_chat_history_limit()` — verify limit param
  - `test_chat_history_per_account()` — account filtering

### 32F — Paper-to-Live Transition Safety

- **Intent:** Switching `ALPACA_PAPER=false` enables real-money trading. Add a pre-flight validation command that checks for common misconfigurations before going live.
- **New CLI command** (extend `pyproject.toml` entry point or `src/automation/runner.py`):
  ```
  financialhub preflight
  ```
  Checks:
  1. Alpaca API keys are set and valid (connects to broker)
  2. Account equity > $0 (funded)
  3. Kill switch daily loss limit is set (not 100%)
  4. At least one strategy is enabled
  5. Paper trading flag matches intent (`ALPACA_PAPER` env var)
  6. PDT protection is enabled (for accounts < $25k)
  7. Notification channel is configured (webhook or email)
  8. Recent backtest exists for enabled strategies (warn if none)
  - Returns pass/warn/fail for each check
  - Blocks with clear message if any check fails
- **New module:** `src/automation/preflight.py`
  ```python
  def run_preflight_checks(settings: Settings, broker: BrokerInterface) -> PreflightReport:
      """Validate configuration before live trading."""
  ```
- **Tests:**
  - `test_preflight_passes_valid_config()` — all checks green
  - `test_preflight_fails_no_strategies()` — no strategies enabled → fail
  - `test_preflight_warns_no_notifications()` — no notifier → warn
  - `test_preflight_warns_no_backtest()` — no recent backtest → warn
  - `test_preflight_fails_kill_switch_too_high()` — 100% loss limit → fail

### 32G — Tier 31 Review Fixes

| ID | Fix | File |
|----|-----|------|
| H1 | Add cumulative wash adjustment test (two WashSale on same lot_id → sum) | `tests/unit/reporting/test_form_8949_enhanced.py` |
| H2 | Add basis_mismatch test to 1099-B reconciliation | `tests/unit/reporting/test_reconciliation_1099b.py` |
| M1 | Add empty-row CSV parsing test | `tests/unit/reporting/test_reconciliation_1099b.py` |
| M2 | Add BOM-prefixed CSV parsing test | `tests/unit/reporting/test_reconciliation_1099b.py` |
| M3 | Add multi-lot matching test | `tests/unit/reporting/test_reconciliation_1099b.py` |

---

## 3. Out of scope (deferred or not planned)

- TurboTax `.txf` binary import format
- H&R Block import format
- Roth conversion tracking / pro-rata rule
- IRA contribution limit tracking / RMD calculations
- Multi-account strategy ensemble (strategies spanning accounts)
- Parallel account cycle execution
- Fundamental scoring strategy (Piotroski integration)
- Position-level risk metrics (Greeks, drawdown contribution)
- HIFO lot selection algorithm
- PDF tax form generation
- Live market data streaming (WebSocket)
- Mobile app

---

## 4. Acceptance criteria

- [ ] `ruff check src/ tests/` clean
- [ ] Full `pytest` suite green; target **810+ tests** (currently 753, adding ~60)
- [ ] Email notifier sends alerts when `email_enabled: true`
- [ ] Digest summary aggregates recent trades/alerts and sends on `digest_cron`
- [ ] Loss carryforward computes correctly: losses - gains - $3k deduction = remainder
- [ ] IRA accounts labeled as tax-deferred/tax-free in reports; excluded from taxable Schedule D
- [ ] Backtest warm-up skips first N bars; equity curve starts clean
- [ ] Chat history persists across browser sessions via SQLite
- [ ] `financialhub preflight` validates config and reports pass/warn/fail
- [ ] All Tier 31 review items (H1, H2, M1–M3) resolved
- [ ] `BUILD_STATE.md` updated with Tier 32 entry
- [ ] No `print()` in production paths; use `logging`

---

## 5. TDD task order (suggested)

**Phase 1 — Tier 31 Review Fixes (32G)**

1. Add cumulative wash adjustment test → verify sum of two WashSale on same lot
2. Add basis_mismatch reconciliation test → verify status="basis_mismatch"
3. Add empty-row, BOM, and multi-lot CSV tests → verify parser handles edge cases

**Phase 2 — Email & Digest (32A)**

4. Write `test_email_notifier_instantiated_when_enabled()` → wire EmailNotifier in runner
5. Write `test_digest_summary_aggregates_recent_data()` → implement `generate_digest_summary()`
6. Write `test_digest_summary_empty_when_no_activity()` → handle empty case
7. Write `test_digest_cron_registered()` → wire digest job in scheduler
8. Add `DigestSummary` model

**Phase 3 — Loss Carryforward (32B)**

9. Write `test_carryforward_no_losses()` → implement `compute_loss_carryforward()`
10. Write `test_carryforward_losses_exceed_gains()` → verify $3k deduction + remainder
11. Write `test_carryforward_with_prior_year()` → chain prior + current
12. Write `test_carryforward_per_account()` → account isolation
13. Write `test_carryforward_cross_term_netting()` → ST loss offsets LT gain
14. Add SQLite table + API endpoint + dashboard card

**Phase 4 — IRA Tax Treatment (32C)**

15. Write `test_ira_traditional_excluded_from_taxable_summary()` → modify tax report
16. Write `test_ira_roth_marked_tax_free()` → label Roth correctly
17. Write `test_brokerage_included_in_taxable_summary()` → regression
18. Write `test_consolidated_separates_taxable_and_ira()` → consolidated view
19. Write `test_form_8949_ira_header_note()` → CSV header note
20. Update dashboard tax page with IRA badges

**Phase 5 — Backtest Warm-Up (32D)**

21. Write `test_backtest_warmup_skips_initial_bars()` → add warmup_bars config
22. Write `test_backtest_warmup_zero_starts_immediately()` → backward compat
23. Write `test_backtest_warmup_longer_than_data()` → graceful empty result

**Phase 6 — Chat Persistence (32E)**

24. Write `test_log_chat_and_retrieve()` → add chat_history table + methods
25. Write `test_chat_history_limit()` → verify limit param
26. Write `test_chat_history_per_account()` → account filtering
27. Update `08_ask.py` to load/persist from SQLite

**Phase 7 — Preflight Safety (32F)**

28. Write `test_preflight_passes_valid_config()` → implement `run_preflight_checks()`
29. Write `test_preflight_fails_no_strategies()` → no strategies → fail
30. Write `test_preflight_warns_no_notifications()` → warn on missing notifier
31. Write `test_preflight_warns_no_backtest()` → warn on no recent backtest
32. Write `test_preflight_fails_kill_switch_too_high()` → 100% loss limit → fail
33. Add `financialhub preflight` CLI command

**Phase 8 — Finalize**

34. Run full `pytest` suite → all green
35. Run `ruff check src/ tests/` → clean
36. Update `BUILD_STATE.md` with Tier 32 entry

---

## 6. Risks and mitigations

| Risk | Mitigation |
|------|------------|
| Email SMTP config untested in CI (no real SMTP server) | Mock SMTP in tests; manual validation with real SMTP server |
| Loss carryforward IRS rules are complex (cross-term netting order) | Follow IRS Publication 550 ordering: same-term first, then cross-term, then $3k deduction |
| IRA exclusion from Schedule D breaks consolidated totals | Separate "taxable" and "tax-advantaged" sections; totals are per-section |
| Backtest warmup changes existing backtest results | `warmup_bars=0` default option available; existing behavior preserved if set to 0 |
| Chat history grows unbounded | Default limit=50 on retrieval; add `max_chat_rows` config if needed |
| Preflight check connects to broker (may fail in CI) | Mock broker in tests; real connection only in manual `financialhub preflight` run |

---

## 7. Definition of done

- All 7 sub-scopes (32A–32G) implemented with passing tests
- 810+ total tests, all green, ruff clean
- Zero regressions on existing 753 tests
- Email alerts actually send (manual verification with SMTP)
- Digest accumulates 7 days of activity and delivers summary
- Loss carryforward matches IRS Publication 550 examples
- IRA accounts visually distinct in dashboard and excluded from taxable totals
- Backtest warm-up produces cleaner equity curves
- Chat history survives browser close
- `financialhub preflight` catches common misconfigurations
- `BUILD_STATE.md` updated with Tier 32 summary

---

## Key files

| File | Change |
|------|--------|
| `src/automation/runner.py` | Wire EmailNotifier, register digest cron job |
| `src/automation/digest.py` | **NEW** — `generate_digest_summary()` + `DigestSummary` model |
| `src/reporting/tax_report.py` | Loss carryforward + IRA tax treatment |
| `src/data/storage/sqlite_store.py` | `loss_carryforward` table, `chat_history` table |
| `src/backtesting/engine.py` | `warmup_bars` support |
| `src/dashboard/pages/07_taxes.py` | Loss carryforward card, IRA badges, taxable/tax-advantaged sections |
| `src/dashboard/pages/08_ask.py` | Chat persistence via SQLite |
| `src/automation/preflight.py` | **NEW** — `run_preflight_checks()` + `PreflightReport` model |
| `src/api/routes/tax.py` | `/carryforward` endpoint |
| `tests/unit/automation/test_digest.py` | **NEW** |
| `tests/unit/reporting/test_loss_carryforward.py` | **NEW** |
| `tests/unit/reporting/test_ira_tax_treatment.py` | **NEW** |
| `tests/unit/backtesting/test_backtest_warmup.py` | **NEW** |
| `tests/unit/data/test_chat_history.py` | **NEW** |
| `tests/unit/automation/test_preflight.py` | **NEW** |
| Various Tier 31 review test files | Fix additions |

---

## Verification

1. `ruff check src/ tests/` — clean
2. `pytest tests/ -x` — 810+ tests, all green
3. Set `email_enabled: true` with valid SMTP → receive alert email on kill switch trigger
4. Set `digest_cron: "0 9 * * 1"` → digest summary delivered Monday 9 AM
5. Tax page: loss carryforward metric shows correct amount after harvesting losses
6. Tax page: IRA account shows "(Tax-Deferred)" badge, excluded from taxable Schedule D totals
7. `financialhub backtest --strategy momentum --start 2023-01-01 --end 2024-01-01` with `warmup_bars=200` → equity curve starts ~200 bars in
8. Dashboard Ask page: close browser, reopen → previous Q&A history loaded
9. `financialhub preflight` → reports pass/warn/fail for each check
