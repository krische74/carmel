# Tier 24 Code Review

## Scope
- **2A** — Tier 23 review fixes (H1–L4)
- **2B** — ML order gating (`affect_orders`, `score_threshold`, `missing_score_action`)
- **2C** — Test coverage hardening (ML train/infer/explain, strategy wiring, backtest DCA/mean reversion)
- **2D** — Dead dependency cleanup (`[llm]`, `[backtest]` extras removed)

## Verdict: PASS

Tier 24 is solid. The ML gating feature is well-designed, all Tier 23 fixes are verified in place, and dependency cleanup is clean. The main area for improvement is test assertion depth — several new tests verify that results exist but don't check correctness.

---

## Tier 23 Fixes — All Verified

| ID | Status | Evidence |
|----|--------|----------|
| H1 | RESOLVED | `trades.py` deleted; ML expander merged into `03_trades.py` |
| H2 | RESOLVED | `test_train.py`, `test_infer.py`, `test_explain.py` exist |
| M1 | RESOLVED | `get_latest_ml_scores_batch()` uses single correlated subquery |
| M2 | RESOLVED | `regime.py` imports from `src.api.dependencies` |
| M3 | RESOLVED | `features.py` has `logger.debug()` at all return-None paths |
| M4 | RESOLVED | Dead stub replaced with real ML gating in `workflows.py` |
| M5 | RESOLVED | Column validation + case-mismatch detection in `features.py` |
| M6 | RESOLVED | `is_relative_to(PROJECT_ROOT)` check before `joblib.dump` |
| L1 | RESOLVED | Bounds use `>= 6` and `>= 11` |
| L2 | RESOLVED | SHAP failures at `WARNING` level |
| L3 | RESOLVED | `_MIN_BARS = 55` has buffer comment |
| L4 | RESOLVED | NaN/Inf filtering test + storage error 500 test added |

---

## ML Order Gating (2B) — Clean Implementation

### What works well
- `src/risk/ml_order_gate.py` is well-isolated with no external dependencies beyond `models.Signal`
- Defensive: handles missing scores, non-numeric scores, and NaN explicitly
- Symbol normalization (`.strip().upper()`) is consistent across gate, workflows, and API
- Threshold boundary uses `<` (not `<=`), so score == threshold passes — correct
- `_apply_ml_order_gate()` in workflows has proper guards: early return when disabled, warning when SQLite missing
- Two call sites (line ~207 for original signals, line ~330 for replacement signals) — both after logging, before execution
- Pydantic validation on `score_threshold` bounds `[0.0, 1.0]` and `Literal["pass", "block"]`
- Lazy import of gate module (only when `affect_orders=True`)

### No bugs found in the gating logic.

---

## Findings — New Tests (2C)

The new test coverage addresses the critical gaps from Tier 23, but assertion depth varies significantly across files.

### HIGH — Tests that need stronger assertions

#### H1: `test_explain.py` has a single weak assertion
- **File:** `tests/unit/ml/test_explain.py`
- **Problem:** The only assertion is `text.startswith("SHAP:")`. This passes even if the output contains garbage after the prefix. Doesn't verify feature names appear, doesn't verify sort order (largest SHAP value first), doesn't verify format (`feature: +0.1234; feature: -0.5678`).
- **Fix:** Add assertions:
  ```python
  assert "sma_50" in text or "rsi_14" in text  # known feature names appear
  assert text.count(";") >= 1                    # multiple features separated
  assert "+" in text or "-" in text              # signed values present
  ```

#### H2: Backtest tests only check existence, not correctness
- **Files:** `tests/unit/backtesting/test_backtest_dca.py`, `tests/unit/backtesting/test_backtest_mean_reversion.py`
- **Problem:** Both tests assert `len(res.equity_curve) >= N` and `len(res.trades) > 0` but never verify:
  - Trade details (prices, quantities, sides)
  - Equity curve progression makes sense (monotonic for DCA in rising market?)
  - Initial capital matches config
  - Return metrics are populated
  - DCA weight allocation matches target weights
  - Mean reversion entries correspond to BB/RSI conditions
- **Fix:** Add at minimum:
  ```python
  assert res.initial_capital == 10_000
  assert res.final_equity > 0
  assert all(e > 0 for e in res.equity_curve.values())  # no negative equity
  # For DCA: verify trades include both target symbols
  symbols_traded = {t.symbol for t in res.trades}
  assert "SPY" in symbols_traded and "BND" in symbols_traded
  ```

---

### MEDIUM — Missing edge case coverage

#### M1: No test for insufficient training data
- **File:** `tests/unit/ml/test_train.py`
- **Problem:** `train_momentum_model_bundle()` raises `ValueError` when `len(feat_rows) < min_train_rows`, but this path is never tested.
- **Fix:** Add test with data too short to produce `min_train_rows` valid feature vectors.

#### M2: No test for empty signal list in gate
- **File:** `tests/unit/risk/test_ml_order_gate.py`
- **Problem:** `filter_signals_by_ml_scores([])` is never tested. Should return `([], 0, 0)`.
- **Fix:** One-liner parametrized case.

#### M3: No error path tests for strategy wiring
- **File:** `tests/unit/automation/test_strategy_wiring.py`
- **Problem:** Missing tests for: empty `enabled` list (should raise ValueError), unknown strategy name (should skip with warning), duplicate names (should deduplicate).
- **Fix:** Add 2–3 negative tests.

#### M4: No test for malformed feature dict in inference
- **File:** `tests/unit/ml/test_infer.py`
- **Problem:** `score_positive_proba()` catches `(TypeError, ValueError, AttributeError)` but no test triggers these. Missing feature key, wrong feature count, or NaN feature value should be tested.
- **Fix:** Add test with `feature_dict` missing a required key.

---

### LOW — Minor quality items

| ID | File | Issue |
|----|------|-------|
| L1 | `test_backtest_dca.py` | Linear price fixture (100→105) is unrealistically smooth; add noise for realism |
| L2 | `test_backtest_mean_reversion.py` | Perfect sine wave is artificial; consider adding random noise to the oscillation |
| L3 | `test_train.py` | Doesn't verify `X_bg` is correctly extracted from last 200 rows |
| L4 | `test_infer.py` | Round-trip test doesn't verify loaded model produces same score as in-memory model |

---

## Dependency Cleanup (2D) — Verified

- `[llm]` extra (langchain, langchain-community): **Removed**
- `[backtest]` extra (backtrader): **Removed**
- Remaining extras: `[dev]`, `[ml]`, `[fundamental]` — all actively used
- Ruff exception for `src/risk/ml_order_gate.py` TC001 is justified (runtime type annotation)

---

## Security & Architecture — No Issues

| Check | Status |
|-------|--------|
| SQL injection | Parameterized queries throughout |
| Path traversal | `is_relative_to(PROJECT_ROOT)` on model write |
| Score coercion | `try/except (TypeError, ValueError)` + `isnan` |
| Config validation | Pydantic bounds + Literal types |
| Symbol normalization | `.strip().upper()` at all use points |
| Signal flow ordering | Signals logged before gating, gating before execution |

---

## Summary

| Category | Finding Count | Severity |
|----------|--------------|----------|
| Tier 23 fixes | 12/12 verified | All resolved |
| ML gating bugs | 0 | Clean |
| Test assertion gaps | 2 HIGH, 4 MEDIUM, 4 LOW | Correctness not verified |
| Security issues | 0 | Clean |
| Architecture issues | 0 | Clean |

---

## Recommended Fix Order
1. **H1** — Strengthen `test_explain.py` assertions (5 min)
2. **H2** — Add correctness assertions to both backtest tests (15 min)
3. **M1** — Add insufficient-data test for `test_train.py` (10 min)
4. **M2** — Add empty signal list test to gate (2 min)
5. **M3** — Add error path tests for strategy wiring (10 min)
6. **M4** — Add malformed feature dict test for inference (5 min)
7. **L1–L4** — Polish as bandwidth allows
