# Tier 24 Sprint Contract — ML Integration, Tier 23 Polish & Test Hardening

**Depends on:** Tier 23 complete (514 tests green, lint clean)
**Governance:** `AGENTS.md` — TDD (failing test first), ruff clean, no `print()` in production paths

---

## 1. Sprint goal

Wire ML scores into order execution (`affect_orders` gate), fix all Tier 23 review findings (HIGH + MEDIUM), close critical test coverage gaps, and clean up dead dependencies. This tier transforms ML from diagnostic-only to optionally actionable.

---

## 2. In scope

### 2A — Tier 23 Review Fixes (prerequisite polish)

All findings from `tier_23_review.md`:

| ID | Fix | File(s) |
|----|-----|---------|
| **H1** | Merge ML scores expander from `trades.py` into `03_trades.py`; delete `trades.py` | `src/dashboard/pages/03_trades.py`, `src/dashboard/pages/trades.py` |
| **H2** | Add ML train→infer round-trip test on synthetic data; add SHAP summary test | `tests/unit/ml/test_train.py`, `tests/unit/ml/test_explain.py` |
| **M1** | Collapse `get_latest_ml_scores_batch()` into single correlated subquery | `src/data/storage/sqlite_store.py` |
| **M2** | Replace local `get_sqlite_store` with import from `src/api/dependencies` | `src/api/routes/regime.py` |
| **M3** | Add `logger.debug(...)` at each `return None` path in `build_feature_vector()` | `src/ml/features.py` |
| **M4** | Remove dead `affect_orders` stub log; replace with real implementation (2B below) | `src/automation/workflows.py` |
| **M5** | Add column-name validation guard at entry of `build_feature_vector()` | `src/ml/features.py` |
| **M6** | Assert `out.is_relative_to(PROJECT_ROOT)` before `joblib.dump` | `src/ml/train.py` |
| **L1** | Change `len(close) > 5` → `>= 6`, `> 10` → `>= 11` for clarity | `src/ml/features.py` |
| **L2** | Change SHAP failure log level from `DEBUG` → `WARNING` | `src/ml/explain.py` |
| **L3** | Add comment: `# 50-bar SMA window + 5-bar buffer` to `_MIN_BARS = 55` | `src/ml/features.py` |
| **L4** | Add negative test: storage exception returns 500 | `tests/unit/api/test_ml_endpoint.py` |

### 2B — ML Order Gating (`affect_orders`)

- **Intent:** When `ml.affect_orders: true`, ML confidence scores filter or reweight signals before execution. Symbols below a configurable threshold are excluded from the order batch.
- **Config additions** (`config/settings.yaml` → `ml:` section):
  ```yaml
  ml:
    affect_orders: false          # existing, currently dead
    score_threshold: 0.5          # NEW: minimum ML score to pass signal through
    missing_score_action: "pass"  # NEW: "pass" | "block" — what to do when no ML score exists for a symbol
  ```
- **Config model** (`src/config.py` → `MLConfig`):
  ```python
  score_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
  missing_score_action: Literal["pass", "block"] = "pass"
  ```
- **Gating logic** (`src/automation/workflows.py`):
  - After signal generation, before `execute_signals()`, if `ml.affect_orders` is `True`:
    1. Load latest ML scores batch from SQLite
    2. For each signal, look up symbol's ML score
    3. If score < `score_threshold` → drop signal, log reason
    4. If no score exists → apply `missing_score_action` policy
    5. Log summary: `"ML gate: N of M signals passed (threshold=X)"`
  - When `ml.affect_orders` is `False` (default): no change to current behavior
- **Storage:** No new tables; reads existing `ml_scores`
- **Tests:** Unit test the gating logic with mock signals + mock scores covering: all pass, all blocked, mixed, missing scores with pass/block policy

### 2C — Test Coverage Hardening

Fill the most critical gaps identified in the codebase analysis:

| Gap | New test file | What to cover |
|-----|---------------|---------------|
| ML train pipeline | `tests/unit/ml/test_train.py` | `train_momentum_model_bundle()` on small synthetic Parquet; verify bundle keys, model type, feature name alignment |
| ML inference | `tests/unit/ml/test_infer.py` | `load_ml_bundle()` with valid/missing/corrupt file; `score_positive_proba()` with known model |
| ML SHAP | `tests/unit/ml/test_explain.py` | `shap_summary_text()` with small tree model; verify output format |
| Strategy wiring | `tests/unit/automation/test_strategy_wiring.py` | `wire_strategies()` returns correct strategy instances from config |
| Backtesting DCA | `tests/unit/backtesting/test_backtest_dca.py` | Walk-forward backtest with DCAStrategy on fixture data |
| Backtesting MeanRev | `tests/unit/backtesting/test_backtest_mean_reversion.py` | Walk-forward backtest with MeanReversionStrategy on fixture data |
| ML API error path | `tests/unit/api/test_ml_endpoint.py` | Storage exception → 500; NaN score handling |

### 2D — Dead Dependency Cleanup

| Extra | Dependency | Action | Reason |
|-------|-----------|--------|--------|
| `[llm]` | `langchain`, `langchain-community` | **Remove** from `pyproject.toml` | Never imported; Ollama uses raw `httpx` |
| `[backtest]` | `backtrader` | **Remove** from `pyproject.toml` | Backtesting engine is custom walk-forward; backtrader unused |

- Verify no imports reference these packages: `grep -r "langchain\|backtrader" src/`
- Remove the extras groups from `pyproject.toml`
- Update `BUILD_STATE.md` noting the cleanup

---

## 3. Out of scope (deferred beyond Tier 24)

- Strategy ensemble (combining multiple strategy signals into weighted votes)
- Regime-aware DCA / mean reversion (only momentum uses regime today)
- Dashboard authentication
- Limit/stop order types
- Multi-period Brinson attribution
- Intraday equity snapshots
- ML retraining automation / scheduled retraining
- Fractional quantity rounding per-symbol
- Dashboard unit tests (Streamlit testing framework needed)

---

## 4. Acceptance criteria

- [ ] `ruff check src/ tests/` clean
- [ ] Full `pytest` suite green; target **560+ tests** (currently 514, adding ~50)
- [ ] All Tier 23 review items (H1, H2, M1–M6, L1–L4) resolved
- [ ] `ml.affect_orders` gating logic works: verified by unit tests with mock signals/scores
- [ ] `ml.affect_orders: false` (default) preserves exact current behavior — no regressions
- [ ] Dead extras (`[llm]`, `[backtest]`) removed; `pip install .[dev,ml]` still works
- [ ] `BUILD_STATE.md` updated with Tier 24 entry
- [ ] No `print()` in production paths; use `logging`
- [ ] All new tests use fixtures, not live APIs

---

## 5. TDD task order (suggested)

**Phase 1 — Tier 23 Polish (H/M/L fixes)**

1. **H1:** Merge ML expander into `03_trades.py`, delete `trades.py`, verify dashboard loads
2. **M1:** Write test for `get_latest_ml_scores_batch()` correctness → fix to single subquery
3. **M2:** Fix `regime.py` import → run existing API tests
4. **M3 + M5 + L1 + L3:** Add logging + validation + clarity fixes to `features.py` → verify existing feature tests pass
5. **L2:** Bump SHAP log level → verify explain module behavior
6. **M6:** Write test: model path outside project raises `ValueError` → add assertion in `train.py`

**Phase 2 — ML Test Coverage (H2 + 2C)**

7. Write `test_train.py`: synthetic data → `train_momentum_model_bundle()` → assert bundle structure
8. Write `test_infer.py`: load the bundle from step 7 → `score_positive_proba()` → assert 0 ≤ score ≤ 1
9. Write `test_explain.py`: small tree model → `shap_summary_text()` → assert non-empty string
10. Write `test_strategy_wiring.py`: config with all 3 strategies → `wire_strategies()` → assert correct types
11. Write `test_backtest_dca.py`: DCA strategy + fixture OHLCV → walk-forward → assert trade count > 0
12. Write `test_backtest_mean_reversion.py`: MeanReversion + fixture OHLCV → walk-forward → assert equity series length

**Phase 3 — ML Order Gating (2B)**

13. Add `score_threshold` and `missing_score_action` to `MLConfig` in `config.py`
14. Add config entries to `settings.yaml`
15. Write `test_ml_order_gate.py`:
    - Test: all signals pass when scores above threshold
    - Test: signals blocked when scores below threshold
    - Test: missing score + `pass` policy → signal passes
    - Test: missing score + `block` policy → signal blocked
    - Test: `affect_orders: false` → no filtering
16. Implement gating logic in `workflows.py` (replace the TODO stub)
17. Write `test_ml_endpoint.py` negative case: storage exception → 500

**Phase 4 — Cleanup (2D)**

18. Grep for `langchain` and `backtrader` imports → confirm zero hits
19. Remove `[llm]` and `[backtest]` extras from `pyproject.toml`
20. Run full `pytest` suite → green
21. Update `BUILD_STATE.md` with Tier 24 completion

---

## 6. Risks and mitigations

| Risk | Mitigation |
|------|------------|
| ML gating accidentally blocks all orders when scores are stale | Default `missing_score_action: "pass"` ensures no blocking without fresh scores; log loudly when scores are older than 24h |
| Removing `[llm]`/`[backtest]` extras breaks CI or other tooling | Grep codebase first; CI installs `.[dev,ml]` only — neither extra is referenced |
| H1 dashboard merge introduces layout regressions | Manual visual check after merge; existing trade page tests still pass |
| Train test is flaky on small synthetic data | Use fixed random seed and known-good fixture with clear signal |
| `score_threshold` misconfigured to 1.0 blocks everything | Validate `0.0 ≤ threshold ≤ 1.0` in Pydantic model; log warning if threshold > 0.9 |

---

## 7. Definition of done

- All 12 Tier 23 review items (H1–L4) are resolved and verified
- `ml.affect_orders` gating is functional with tests for all 5 scenarios (above/below threshold, missing score with pass/block, disabled)
- ~50 new tests added, total suite at 560+, all green
- Dead dependency extras removed from `pyproject.toml`
- `BUILD_STATE.md` updated with Tier 24 summary
- `ruff check` and `ruff format --check` clean
- No regressions in existing 514 tests
