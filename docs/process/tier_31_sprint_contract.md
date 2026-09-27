# Tier 31 Sprint Contract — Tax Export Completion & Platform Polish

**Depends on:** Tier 30 complete (750+ tests green, lint clean, multi-account foundation)
**Governance:** `AGENTS.md` — TDD (failing test first), ruff clean, no `print()` in production paths

---

## 1. Sprint goal

Complete the tax filing workflow: enhance Form 8949 CSV with IRS adjustment codes and Part I/II segregation, add Schedule D summary generation, add a 1099-B reconciliation workflow against Alpaca broker statements, make the wash-sale window configurable, and add a guided tax filing checklist to the dashboard. Also close Tier 30 review items.

---

## 2. In scope

### 31A — Form 8949 Enhancement (Part I/II + Adjustment Codes)

- **Intent:** Current CSV is a flat list. IRS Form 8949 requires Part I (short-term) and Part II (long-term) segregation, plus adjustment codes (e.g., code `W` for wash sale disallowed loss).
- **Changes to `src/reporting/tax_report.py`:**
  - Split `export_form_8949_csv()` output into two sections:
    - **Part I — Short-Term** (held < 1 year): header row + short-term lots
    - **Part II — Long-Term** (held ≥ 1 year): header row + long-term lots
  - Add IRS adjustment code column:
    - `W` — wash sale loss disallowed (when `wash_adjustment > 0`)
    - Empty — no adjustment
  - Add adjustment amount column (wash sale disallowed amount)
  - Add cost basis adjusted column: `cost_basis + wash_adjustment` (IRS requires adjusted basis)
  - CSV columns (IRS-aligned):
    ```
    (a) Description, (b) Date Acquired, (c) Date Sold, (d) Proceeds,
    (e) Cost or Other Basis, (f) Code, (g) Adjustment, (h) Gain or Loss
    ```
- **New model** (`src/reporting/tax_report.py`):
  ```python
  class Form8949Row(BaseModel):
      description: str        # "10 sh SPY"
      date_acquired: str      # MM/DD/YYYY
      date_sold: str          # MM/DD/YYYY
      proceeds: float         # (d)
      cost_basis: float       # (e) - original cost
      adjustment_code: str    # (f) - "W" or ""
      adjustment_amount: float # (g) - wash disallowed amount
      gain_or_loss: float     # (h) - proceeds - adjusted_basis
      term: str               # "short" or "long"
      account_id: str         # for per-account filtering
  ```
- **Tests:**
  - `test_form_8949_part_i_short_term_only()` — verify Part I header + short-term lots
  - `test_form_8949_part_ii_long_term_only()` — verify Part II header + long-term lots
  - `test_form_8949_wash_sale_code_w()` — lot with wash adjustment → code "W", adjustment amount populated
  - `test_form_8949_no_adjustment_empty_code()` — lot without wash → empty code
  - `test_form_8949_adjusted_basis()` — cost_basis + wash_adjustment = adjusted basis in output
  - `test_form_8949_per_account_with_parts()` — per-account export with Part I/II

### 31B — Schedule D Summary

- **Intent:** Schedule D (Form 1040) summarizes Form 8949 totals. Generate a companion CSV that users can reference when filling Schedule D.
- **New function** (`src/reporting/tax_report.py`):
  ```python
  def generate_schedule_d_summary(
      tax_summary: TaxSummary,
      *,
      account_id: str | None = None,
  ) -> ScheduleDSummary:
  ```
- **New model:**
  ```python
  class ScheduleDSummary(BaseModel):
      year: int
      account_id: str | None
      # Part I — Short-Term
      short_term_proceeds: float
      short_term_cost_basis: float
      short_term_wash_adjustments: float
      short_term_net: float
      # Part II — Long-Term
      long_term_proceeds: float
      long_term_cost_basis: float
      long_term_wash_adjustments: float
      long_term_net: float
      # Totals
      total_net_gain_or_loss: float
  ```
- **Export:** `export_schedule_d_csv()` → simple CSV with one row per line item
- **API endpoint:** `GET /api/tax/schedule-d?year=2025&account_id=Main`
- **Dashboard:** Add Schedule D summary card above the lots table on tax page; download button alongside Form 8949
- **Tests:**
  - `test_schedule_d_short_and_long_totals()` — verify proceeds/basis/net match lot sums
  - `test_schedule_d_wash_adjustments_included()` — verify wash disallowed amounts in summary
  - `test_schedule_d_per_account()` — filtered by account
  - `test_schedule_d_csv_export()` — verify CSV format

### 31C — 1099-B Reconciliation

- **Intent:** Alpaca sends a 1099-B at year-end. Users need to verify their lot ledger matches. Import the broker statement and flag discrepancies.
- **New module:** `src/reporting/reconciliation_1099b.py`
  ```python
  def parse_1099b_csv(file_path: Path) -> list[BrokerLotRecord]:
      """Parse Alpaca 1099-B CSV export into standardized records."""

  def reconcile_1099b(
      broker_records: list[BrokerLotRecord],
      closed_lots: list[ClosedLot],
      *,
      qty_tolerance: float = 1e-4,
      price_tolerance: float = 0.01,
  ) -> ReconciliationReport1099B:
      """Match broker 1099-B records to internal closed lots. Flag discrepancies."""
  ```
- **Models:**
  ```python
  class BrokerLotRecord(BaseModel):
      symbol: str
      qty: float
      date_acquired: date
      date_sold: date
      proceeds: float
      cost_basis: float

  class LotMatch(BaseModel):
      broker_record: BrokerLotRecord
      internal_lot: ClosedLot | None
      status: Literal["matched", "proceeds_mismatch", "basis_mismatch", "qty_mismatch", "missing_internal", "missing_broker"]
      variance: float  # dollar difference

  class ReconciliationReport1099B(BaseModel):
      total_broker_records: int
      total_internal_lots: int
      matched: int
      discrepancies: int
      matches: list[LotMatch]
  ```
- **Matching algorithm:**
  1. Group both lists by `(symbol, date_sold)` — primary key
  2. Within each group, match by `qty` (within tolerance)
  3. Compare `proceeds` and `cost_basis` (within tolerance)
  4. Flag unmatched records in either direction
- **API endpoint:** `POST /api/tax/reconcile-1099b` — accepts CSV upload, returns `ReconciliationReport1099B`
- **Dashboard:** New expander on tax page: "Reconcile 1099-B" → file upload → results table with match/mismatch indicators
- **Tests:**
  - `test_parse_1099b_csv()` — valid Alpaca CSV parsed correctly
  - `test_parse_1099b_csv_missing_columns()` — graceful error on bad CSV
  - `test_reconcile_all_matched()` — perfect match
  - `test_reconcile_proceeds_mismatch()` — proceeds differ by > tolerance
  - `test_reconcile_missing_internal()` — broker has lot that internal doesn't
  - `test_reconcile_missing_broker()` — internal has lot that broker doesn't
  - `test_reconcile_qty_mismatch()` — quantity differs

### 31D — Configurable Wash-Sale Window

- **Intent:** Wash-sale window is hardcoded to 30 days. Make it configurable for conservative users who want a wider buffer.
- **Config addition** (`src/config.py` → `TaxConfig`):
  ```python
  wash_sale_window_days: int = Field(default=30, ge=1, le=90)
  ```
- **Wiring:**
  - `detect_wash_sales()` in `wash_sales.py`: pass `window_days=settings.tax.wash_sale_window_days`
  - `detect_cross_account_wash_sales()`: same
  - Dashboard and API: no changes (reads from config)
- **Config in `settings.yaml`:**
  ```yaml
  tax:
    wash_sale_window_days: 30
  ```
- **Tests:**
  - `test_wash_sale_custom_window_45_days()` — replacement at day 35 flagged with 45-day window
  - `test_wash_sale_custom_window_15_days()` — replacement at day 20 NOT flagged with 15-day window
  - `test_cross_account_wash_respects_custom_window()` — cross-account uses config window

### 31E — Tax Filing Checklist (Dashboard UX)

- **Intent:** Guide users through the tax filing workflow with an interactive checklist so they don't miss steps.
- **Dashboard addition** (`src/dashboard/pages/07_taxes.py`):
  - New expander at top of page: "Tax Filing Checklist"
  - Markdown-rendered checklist with `st.checkbox()` items (session state persistence):
    ```
    □ Review all closed lots for the selected year
    □ Check wash sale flags — verify disallowed amounts are reasonable
    □ Review cross-account wash sales (if multiple accounts)
    □ Download Form 8949 CSV (Part I + Part II)
    □ Download Schedule D summary
    □ Reconcile against broker 1099-B statement
    □ Verify totals match 1099-B
    □ File with tax software or accountant
    ```
  - Progress indicator: "5 of 8 complete"
  - Session state: `st.session_state[f"tax_checklist_{year}"]` — persists during session
- **No backend changes** — purely frontend UX
- **Tests:** Manual verification (Streamlit UI testing is out of scope)

### 31F — Tier 30 Review Fixes

| ID | Fix | File |
|----|-----|------|
| M1 | Fix type annotation `list[str]` → `list[str | int]` in `count_equity_snapshots()` | `src/data/storage/sqlite_store.py` |
| L1 | Add module docstring to `wash_sales.py` explaining intra vs cross-account detection | `src/portfolio/wash_sales.py` |
| L2 | Add docstring note to `compute_tax_summary()` that `account_id=None` means consolidated | `src/reporting/tax_report.py` |
| L4 | Add single-account-in-list runner test | `tests/unit/automation/test_multi_account_runner.py` |

---

## 3. Out of scope (deferred beyond Tier 31)

- TurboTax `.txf` / `.tax` binary import format (proprietary; CSV + Schedule D is sufficient for most users)
- H&R Block specific import format
- IRA contribution/distribution tracking
- Roth conversion tracking
- Loss carryforward tracking (multi-year)
- PDF form generation (IRS forms as PDF)
- Multi-account strategy ensemble
- Parallel account cycle execution
- Automated cost basis adjustment from wash sales (informational flagging only)

---

## 4. Acceptance criteria

- [ ] `ruff check src/ tests/` clean
- [ ] Full `pytest` suite green; target **790+ tests** (currently 750, adding ~40)
- [ ] Form 8949 CSV has Part I (short-term) and Part II (long-term) sections with IRS column headers
- [ ] Wash sale lots show adjustment code `W` and adjustment amount in Form 8949
- [ ] Schedule D summary correctly aggregates proceeds, basis, and net by term
- [ ] 1099-B reconciliation parses Alpaca CSV and flags mismatches
- [ ] Wash-sale window is configurable via `tax.wash_sale_window_days`
- [ ] Tax filing checklist renders on dashboard with session-state persistence
- [ ] All Tier 30 review items (M1, L1, L2, L4) resolved
- [ ] `BUILD_STATE.md` updated with Tier 31 entry
- [ ] No `print()` in production paths; use `logging`

---

## 5. TDD task order (suggested)

**Phase 1 — Tier 30 Review Fixes (31F)**

1. Fix type annotation in `count_equity_snapshots()` → verify lint clean
2. Add docstrings to `wash_sales.py` and `tax_report.py`
3. Add single-account-in-list runner test

**Phase 2 — Configurable Wash-Sale Window (31D)**

4. Write `test_wash_sale_custom_window_45_days()` → add `wash_sale_window_days` to TaxConfig
5. Write `test_wash_sale_custom_window_15_days()` → wire config to detection
6. Write `test_cross_account_wash_respects_custom_window()` → verify cross-account uses config
7. Update `settings.yaml`

**Phase 3 — Form 8949 Enhancement (31A)**

8. Write `test_form_8949_part_i_short_term_only()` → implement Part I/II split
9. Write `test_form_8949_part_ii_long_term_only()` → verify long-term section
10. Write `test_form_8949_wash_sale_code_w()` → add adjustment code column
11. Write `test_form_8949_no_adjustment_empty_code()` → verify no-wash lots
12. Write `test_form_8949_adjusted_basis()` → verify cost + adjustment math
13. Write `test_form_8949_per_account_with_parts()` → per-account with Part I/II

**Phase 4 — Schedule D Summary (31B)**

14. Write `test_schedule_d_short_and_long_totals()` → implement ScheduleDSummary
15. Write `test_schedule_d_wash_adjustments_included()` → verify wash totals
16. Write `test_schedule_d_per_account()` → account filter
17. Write `test_schedule_d_csv_export()` → CSV format
18. Add API endpoint `GET /api/tax/schedule-d`
19. Add dashboard download button

**Phase 5 — 1099-B Reconciliation (31C)**

20. Write `test_parse_1099b_csv()` → implement `parse_1099b_csv()`
21. Write `test_parse_1099b_csv_missing_columns()` → graceful error
22. Write `test_reconcile_all_matched()` → implement `reconcile_1099b()`
23. Write `test_reconcile_proceeds_mismatch()` → flag proceeds variance
24. Write `test_reconcile_missing_internal()` → broker lot not in internal
25. Write `test_reconcile_missing_broker()` → internal lot not in broker
26. Write `test_reconcile_qty_mismatch()` → quantity variance
27. Add API endpoint `POST /api/tax/reconcile-1099b`
28. Add dashboard file upload + results expander

**Phase 6 — Tax Filing Checklist (31E)**

29. Add checklist expander to `07_taxes.py` with session state
30. Add progress indicator

**Phase 7 — Finalize**

31. Run full `pytest` suite → all green
32. Run `ruff check src/ tests/` → clean
33. Update `BUILD_STATE.md` with Tier 31 entry

---

## 6. Risks and mitigations

| Risk | Mitigation |
|------|------------|
| Form 8949 column order doesn't match IRS spec exactly | Use official IRS Form 8949 instructions (Publication 550) as reference; columns labeled (a)–(h) |
| Alpaca 1099-B CSV format changes year-to-year | Parser validates required columns on load; raises clear error on missing/renamed columns |
| Wash-sale adjustment amount calculation edge cases | Use simple `abs(realized_pnl)` for disallowed amount (matches current detection); don't prorate partial replacements |
| 1099-B matching false negatives on fractional shares | `qty_tolerance=1e-4` handles float precision; log unmatched for manual review |
| Tax filing checklist session state lost on browser close | Acceptable — checklist is guidance, not data; re-check on next session |
| Schedule D totals don't match Form 8949 lots | Schedule D is computed FROM the same TaxSummary; mathematically guaranteed to agree |

---

## 7. Definition of done

- All 6 sub-scopes (31A–31F) implemented with passing tests
- 790+ total tests, all green, ruff clean
- Zero regressions on existing 750 tests
- Form 8949 CSV has IRS-aligned columns with Part I/II sections
- Schedule D summary matches Form 8949 lot totals
- 1099-B reconciliation catches mismatches between broker and internal data
- Configurable wash-sale window works for both intra and cross-account detection
- Dashboard tax page has filing checklist + Schedule D download + 1099-B upload
- `BUILD_STATE.md` updated with Tier 31 summary

---

## Key files

| File | Change |
|------|--------|
| `src/reporting/tax_report.py` | Form 8949 Part I/II split, adjustment codes, Form8949Row model, Schedule D summary |
| `src/reporting/reconciliation_1099b.py` | **NEW** — 1099-B parser + reconciler |
| `src/portfolio/wash_sales.py` | Accept configurable `window_days`, add docstring |
| `src/config.py` | Add `wash_sale_window_days` to TaxConfig |
| `config/settings.yaml` | Add `wash_sale_window_days: 30` |
| `src/api/routes/tax.py` | Add `/schedule-d` and `/reconcile-1099b` endpoints |
| `src/dashboard/pages/07_taxes.py` | Filing checklist, Schedule D card, 1099-B upload expander |
| `src/data/storage/sqlite_store.py` | Fix type annotation (Tier 30 M1) |
| `tests/unit/reporting/test_form_8949_enhanced.py` | **NEW** — Part I/II, adjustment codes |
| `tests/unit/reporting/test_schedule_d.py` | **NEW** — Schedule D summary tests |
| `tests/unit/reporting/test_reconciliation_1099b.py` | **NEW** — 1099-B parser + reconciler tests |
| `tests/unit/portfolio/test_wash_sales_configurable.py` | **NEW** — custom window tests |

---

## Verification

1. `ruff check src/ tests/` — clean
2. `pytest tests/ -x` — 790+ tests, all green
3. Export Form 8949 → verify Part I and Part II sections, adjustment code `W` on wash lots
4. Export Schedule D → verify short/long totals match Form 8949 lot sums
5. Upload sample 1099-B CSV → verify match/mismatch counts are correct
6. Set `wash_sale_window_days: 45` → verify day-35 replacement is flagged (was not flagged at 30)
7. Dashboard tax page → checklist renders, checkboxes persist in session, Schedule D downloads
