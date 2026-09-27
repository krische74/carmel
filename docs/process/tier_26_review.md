# Tier 26 Code Review

## Scope
- **26A** — Regime-aware DCA: `effective_dca_amount()` helper, per-regime dollar amount multipliers
- **26B** — Regime-aware mean reversion: `effective_mean_reversion_max_positions()` helper, per-regime position caps
- **26C** — Workflow simplification: removed isinstance checks, pass `market_regime` to all strategies uniformly
- **26D** — Backtest parity: removed isinstance guard, pass regime to all strategies in backtests

## Verdict: PASS

Clean implementation. All production code is correct, well-structured, and follows the established momentum pattern exactly. The isinstance checks are properly removed from both workflows and backtest engine. No bugs found. The only gap is test coverage depth — only None and CRISIS regimes are tested; CAUTIOUS and DEFENSIVE are untested.

**Test count:** 595 total (up from 582)

---

## Production Code — No Issues Found

All four sub-scopes are correctly implemented:

| Check | Status | Evidence |
|-------|--------|---------|
| `effective_dca_amount()` helper | CORRECT | Returns base when `overall is None`; applies per-regime multiplier; epsilon comparison for rationale |
| `effective_mean_reversion_max_positions()` helper | CORRECT | Returns base when `overall is None`; per-regime cap; `k <= 0` guard |
| Strategy ABC updated | CORRECT | `market_regime: MarketRegime | None = None` keyword-only param |
| DCA accepts + uses regime | CORRECT | Scales amount, appends rationale when adjusted |
| Mean reversion accepts + uses regime | CORRECT | Caps candidates by depth, weights = 1/k |
| Momentum unaffected | CORRECT | Already had market_regime; no changes needed |
| isinstance checks removed (workflows) | VERIFIED | Lines 173–179: uniform `market_regime=cycle_regime` for all strategies |
| isinstance checks removed (backtest) | VERIFIED | Lines 215–227: uniform regime passing for all strategies |
| Config fields validated | CORRECT | Pydantic bounds: multipliers [0.0, 2.0], positions [0, 10] |
| settings.yaml aligned | CORRECT | All 8 new fields present with sensible defaults |
| BUILD_STATE.md updated | VERIFIED | Tier 26 entry present |

### What's particularly well done
- All three strategies follow identical regime extraction pattern: `overall = market_regime.overall if market_regime is not None else None`
- Float comparison uses epsilon (`abs(amount - base_amt) > 1e-9`) to avoid spurious rationale notes
- Mean reversion `k <= 0` guard gracefully handles edge case of zero max positions
- Backtest `_CaptureDCA` and `_CaptureMeanReversion` subclasses are a clever testing pattern

---

## Findings — Test Coverage Gaps

### HIGH — Missing intermediate regime tests

#### H1: CAUTIOUS and DEFENSIVE regimes are completely untested
- **Files:** `tests/unit/strategy/test_regime_params.py`, `test_dca.py`, `test_mean_reversion.py`
- **Problem:** Only None (base), RISK_ON (partial), and CRISIS are tested. The CAUTIOUS (0.75×) and DEFENSIVE (0.5×) multipliers are defined in config and implemented in code but never validated by any test. A typo in `regime_params.py` (e.g., swapping cautious and defensive values) would go undetected.
- **Impact:** 50% of regime levels untested
- **Fix:** Add parametrized tests covering all four regime levels for both helpers:

  For `test_regime_params.py`:
  ```python
  @pytest.mark.parametrize("regime,expected_mult", [
      (OverallRegime.RISK_ON, 1.0),
      (OverallRegime.CAUTIOUS, 0.75),
      (OverallRegime.DEFENSIVE, 0.5),
      (OverallRegime.CRISIS, 0.25),
  ])
  def test_effective_dca_amount_all_regimes(settings, regime, expected_mult):
      result = effective_dca_amount(settings, regime)
      assert result == pytest.approx(settings.strategy.dca.amount * expected_mult)
  ```

  Same pattern for `effective_mean_reversion_max_positions`:
  ```python
  @pytest.mark.parametrize("regime,expected_max", [
      (OverallRegime.RISK_ON, 4),
      (OverallRegime.CAUTIOUS, 3),
      (OverallRegime.DEFENSIVE, 2),
      (OverallRegime.CRISIS, 1),
  ])
  ```

#### H2: Backtest tests verify regime is *passed* but not that it *affects* output
- **Files:** `tests/unit/backtesting/test_backtest_dca.py`, `test_backtest_mean_reversion.py`
- **Problem:** Tests use `_CaptureDCA`/`_CaptureMeanReversion` subclasses to verify `market_regime is not None`, but never check that the regime actually changed the DCA amount or capped positions during the backtest.
- **Fix:** Add one test per strategy that compares output under crisis vs. no-regime:
  ```python
  def test_backtest_dca_crisis_reduces_equity():
      # Run backtest with crisis VIX data → smaller equity growth
      # Run backtest without regime → larger equity growth
      # Assert crisis final_equity < no_regime final_equity
  ```

---

### MEDIUM — Missing edge case tests

#### M1: No test for multiplier = 0.0
- **File:** `tests/unit/strategy/test_regime_params.py`
- **Problem:** Config allows `ge=0.0` on regime multipliers. If a user sets `regime_amount_crisis: 0.0`, DCA would generate $0 contributions. This is arguably valid (skip DCA in crisis) but should be explicitly tested.
- **Fix:** Add one test with multiplier override to 0.0, verify DCA signals have $0 amount or are omitted.

#### M2: No test for RISK_ON identity (regime produces same result as no-regime)
- **File:** `tests/unit/strategy/test_dca.py`, `test_mean_reversion.py`
- **Problem:** RISK_ON multiplier is 1.0 (DCA) and max_positions is 4 (MR) — same as base. This should be tested to confirm regime doesn't accidentally alter behavior when regime is "normal."
- **Fix:** Add tests that verify RISK_ON signals are identical to no-regime signals.

---

### LOW — Minor items

| ID | File | Issue |
|----|------|-------|
| L1 | `src/backtesting/engine.py` | Config field `use_regime_for_momentum` is now misleading (applies to all strategies). Not a bug but a naming debt. |
| L2 | `src/strategy/dca.py`, `mean_reversion.py` | Local imports inside methods (`from src.strategy.regime_params import ...`). Follows existing momentum pattern, so consistent, but non-standard Python style. |
| L3 | `test_workflows.py` | Regime-passing tests verify regime is non-None but don't spy on the exact regime values passed (e.g., VIX level, overall classification). |

---

## Summary

| Category | Finding Count | Severity |
|----------|--------------|----------|
| Production code bugs | 0 | Clean |
| Architecture issues | 0 | isinstance checks properly removed |
| Config issues | 0 | Pydantic validated, defaults aligned |
| Test coverage gaps | 2 HIGH, 2 MEDIUM, 3 LOW | Intermediate regimes untested |
| Security issues | 0 | Clean |

---

## Recommended Fix Order

1. **H1** — Add parametrized regime tests for all 4 levels in `test_regime_params.py` (10 min)
2. **H1** — Add CAUTIOUS/DEFENSIVE strategy signal tests in `test_dca.py` and `test_mean_reversion.py` (15 min)
3. **M2** — Add RISK_ON identity test (verify same-as-base behavior) (5 min)
4. **H2** — Add backtest output comparison test (crisis vs no-regime equity curve) (20 min)
5. **M1** — Add zero-multiplier edge case test (5 min)
6. **L1–L3** — Address as bandwidth allows
