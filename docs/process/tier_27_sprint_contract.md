# Tier 27 Sprint Contract — Strategy Ensemble

**Depends on:** Tier 26 complete (595+ tests green, lint clean, all strategies regime-aware)
**Governance:** `AGENTS.md` — TDD (failing test first), ruff clean, no `print()` in production paths

---

## 1. Sprint goal

Add a weighted-vote ensemble that merges tactical signals from multiple strategies before execution. When momentum and mean reversion agree on a symbol, the combined signal is stronger. When momentum signals "cash" (defensive), tactical longs from other strategies are penalized. DCA signals pass through untouched — they are scheduled contributions, not tactical views.

---

## 2. In scope

### 27A — Ensemble Module

- **Intent:** When multiple strategies target the same symbol, merge into one signal with a blended weight rather than submitting duplicate orders.
- **New module:** `src/strategy/ensemble.py`
- **Core function:**
  ```python
  def merge_signals(
      signals: list[Signal],
      *,
      ensemble_config: EnsembleConfig,
      timestamp: datetime,
  ) -> list[Signal]:
  ```
- **Algorithm:**
  1. **Partition** signals into DCA (pass-through) and tactical (momentum, mean reversion, etc.)
  2. **Detect** momentum "cash" signal (direction="cash") → set `cash_penalty` flag
  3. **Group** tactical long signals by symbol
  4. **For each symbol group**, compute blended weight:
     ```
     blended = sum(signal.weight * strategy_weight[signal.strategy_name])
              / sum(strategy_weight[signal.strategy_name])
     ```
     Where `strategy_weight` comes from config (e.g., momentum=0.5, mean_reversion=0.5)
  5. **Apply cash penalty**: if momentum signaled cash, multiply all tactical blended weights by `cash_signal_penalty` (default 0.5)
  6. **Filter**: drop signals with blended weight < `min_blended_weight` (default 0.05)
  7. **Emit** one merged `Signal` per symbol:
     - `direction="long"`, `weight=blended_weight`
     - `confidence=max(input confidences)`
     - `rationale="Ensemble: MomentumRotation(0.6×0.5) + MeanReversion(0.25×0.5) = 0.425"`
     - `strategy_name="Ensemble"`
  8. **Pass through** momentum cash signal if no tactical longs survive
  9. **Return** merged tactical + DCA signals + flat signals (informational)
- **Design decisions:**
  - DCA signals are **never** modified — `OrderManager` checks `strategy_name == "DCAStrategy"` for the budget path
  - Single-strategy signals (no overlap) still get their strategy weight applied — a single momentum signal at weight=1.0 with strategy_weight=0.5 becomes blended=1.0 (no penalty for being alone; weight is normalized by contributing strategies only)
  - `confidence` is NOT used in blending (values are hardcoded constants today); reserved for future ML confidence
  - Flat signals pass through unchanged (informational from mean reversion)

### 27B — Ensemble Configuration

- **New config class** (`src/config.py`):
  ```python
  class EnsembleConfig(BaseModel):
      enabled: bool = Field(default=False, description="Enable ensemble signal merging")
      strategy_weights: dict[str, float] = Field(
          default_factory=lambda: {
              "MomentumRotationStrategy": 0.5,
              "MeanReversionStrategy": 0.5,
          },
          description="Per-strategy weight for blending. Keys are strategy_name values."
      )
      cash_signal_penalty: float = Field(
          default=0.5, ge=0.0, le=1.0,
          description="Multiply tactical weights by this when momentum signals cash."
      )
      min_blended_weight: float = Field(
          default=0.05, ge=0.0, le=1.0,
          description="Drop merged signals with blended weight below this."
      )
  ```
- **Add to `StrategyConfig`:**
  ```python
  ensemble: EnsembleConfig = Field(default_factory=EnsembleConfig)
  ```
- **Config in `settings.yaml`:**
  ```yaml
  strategy:
    ensemble:
      enabled: false
      strategy_weights:
        MomentumRotationStrategy: 0.5
        MeanReversionStrategy: 0.5
      cash_signal_penalty: 0.5
      min_blended_weight: 0.05
  ```

### 27C — Workflow Integration

- **Intent:** Insert ensemble merge into the live/paper trading cycle, after per-signal logging (audit trail) but before ML gate (risk filter).
- **File:** `src/automation/workflows.py`
- **Signal flow after change:**
  ```
  Strategies generate signals → flat list
      ↓
  Per-signal logging to SQLite (audit: what each strategy said)
      ↓
  [NEW] ensemble.merge_signals() if ensemble.enabled
      ↓
  ML order gate (risk filter on merged signals)
      ↓
  Momentum rotation sells
      ↓
  Tax-loss harvest + replacement signals
      ↓
  OrderManager.execute_signals()
  ```
- **Critical invariant:** Momentum rotation sell logic (lines ~219–253) looks for `strategy_name == "MomentumRotationStrategy"` signals. After ensemble merge, tactical signals have `strategy_name="Ensemble"`. Solution: save `raw_signals = list(signals)` before merge; use `raw_signals` for momentum rotation target extraction. The merged `signals` list feeds into execution.
- **Implementation** (~10 lines inserted after signal logging, before ML gate):
  ```python
  raw_signals = list(signals)  # preserve for rotation logic
  if self._settings.strategy.ensemble.enabled:
      from src.strategy.ensemble import merge_signals
      signals = merge_signals(
          signals,
          ensemble_config=self._settings.strategy.ensemble,
          timestamp=when,
      )
  ```
  Then change momentum rotation line from `signals` to `raw_signals` for finding the momentum target.

### 27D — Tier 26 Review Test Fixes

Close the test gaps from `tier_26_review.md`:

| ID | Fix | File |
|----|-----|------|
| H1 | Add parametrized tests for all 4 regime levels (risk_on, cautious, defensive, crisis) in `effective_dca_amount()` and `effective_mean_reversion_max_positions()` | `tests/unit/strategy/test_regime_params.py` |
| H2 | Add backtest output comparison test: crisis vs no-regime equity curve differs | `tests/unit/backtesting/test_backtest_dca.py` or `test_backtest_mean_reversion.py` |
| M1 | Add test for multiplier = 0.0 edge case | `tests/unit/strategy/test_regime_params.py` |
| M2 | Add RISK_ON identity test (same result as no-regime) | `tests/unit/strategy/test_dca.py`, `test_mean_reversion.py` |

---

## 3. Out of scope (deferred beyond Tier 27)

- Multi-strategy backtesting (backtester remains single-strategy)
- Dynamic confidence weighting (confidence values are static constants today)
- Per-symbol strategy weight overrides
- Ensemble as a `Strategy` subclass (it's a post-processing step, not a signal generator)
- New API endpoints for ensemble (signals log as `strategy_name="Ensemble"` — existing API shows them)
- Dashboard changes for ensemble visualization
- Kelly-style position sizing

---

## 4. Acceptance criteria

- [ ] `ruff check src/ tests/` clean
- [ ] Full `pytest` suite green; target **630+ tests** (currently 595, adding ~35)
- [ ] `ensemble.enabled: false` (default) preserves exact current behavior — no regressions
- [ ] DCA signals pass through ensemble untouched (verified by test)
- [ ] Same-symbol signals from two strategies produce one merged signal (verified by test)
- [ ] Momentum cash signal penalizes tactical longs by `cash_signal_penalty` factor (verified by test)
- [ ] Signals below `min_blended_weight` are dropped (verified by test)
- [ ] Momentum rotation sell logic still works after ensemble (uses raw pre-merge signals)
- [ ] All Tier 26 review test gaps closed (H1, H2, M1, M2)
- [ ] `BUILD_STATE.md` updated with Tier 27 entry
- [ ] No `print()` in production paths; use `logging`

---

## 5. TDD task order (suggested)

**Phase 1 — Tier 26 Review Fixes (27D)**

1. Add parametrized regime tests for all 4 levels in `test_regime_params.py` → run, verify green
2. Add RISK_ON identity tests for DCA and mean reversion → run, verify green
3. Add multiplier=0.0 edge case test → run, verify green
4. Add backtest crisis vs no-regime comparison test → run, verify green

**Phase 2 — Ensemble Config (27B)**

5. Add `EnsembleConfig` to `config.py` → verify Pydantic validation
6. Add `ensemble` field to `StrategyConfig`
7. Add ensemble section to `settings.yaml`

**Phase 3 — Ensemble Core Logic (27A)**

8. Write `test_dca_signals_pass_through_unchanged()` → DCA signals returned as-is
9. Write `test_single_strategy_signal_unchanged()` → lone momentum signal weight unmodified
10. Write `test_two_strategies_same_symbol_blended()` → momentum(0.6) + MR(0.25) → weighted average
11. Write `test_cash_signal_penalizes_tactical_longs()` → blended weights reduced by `cash_signal_penalty`
12. Write `test_cash_signal_passes_through_when_no_longs_survive()` → cash signal emitted
13. Write `test_signals_below_min_weight_dropped()` → filtered out
14. Write `test_flat_signals_pass_through()` → informational signals preserved
15. Write `test_ensemble_disabled_returns_signals_unchanged()` → no-op when disabled
16. Implement `merge_signals()` in `src/strategy/ensemble.py` → all tests green

**Phase 4 — Workflow Integration (27C)**

17. Write `test_workflow_calls_ensemble_when_enabled()` → verify merge_signals called
18. Write `test_workflow_skips_ensemble_when_disabled()` → signals unchanged
19. Write `test_workflow_momentum_rotation_uses_raw_signals()` → rotation target found from pre-merge list
20. Implement workflow changes → all tests green

**Phase 5 — Finalize**

21. Run full `pytest` suite → all green
22. Run `ruff check src/ tests/` → clean
23. Update `BUILD_STATE.md` with Tier 27 entry

---

## 6. Risks and mitigations

| Risk | Mitigation |
|------|------------|
| Ensemble changes DCA behavior | DCA signals partitioned first; never modified; test verifies pass-through |
| Momentum rotation breaks after merge (can't find target symbol) | `raw_signals` preserved before merge; rotation logic uses pre-merge list |
| Single-strategy users see different sizing | Single-strategy blended weight = original weight (normalized by 1 strategy); no change |
| Ensemble enabled with only DCA → no tactical signals to merge | Ensemble returns DCA signals unchanged; no error |
| Unknown `strategy_name` not in `strategy_weights` | Treat unknown strategies with weight=0 (signal dropped from blend); log warning |
| Tax-loss replacement signals generated after ensemble | Replacement signals have `strategy_name="TaxLossHarvester"`; ensemble runs before TLH; replacements are not re-merged |

---

## 7. Definition of done

- All 4 sub-scopes (27A–27D) implemented with passing tests
- 630+ total tests, all green, ruff clean
- Zero regressions on existing 595 tests
- `ensemble.enabled: false` (default) produces identical behavior to Tier 26
- DCA signals verified untouched by ensemble
- Momentum rotation verified working with ensemble enabled
- `BUILD_STATE.md` updated with Tier 27 summary
- `ensemble.py` is a pure-function module with no side effects — easy to test and reason about

---

## Key files

| File | Change |
|------|--------|
| `src/strategy/ensemble.py` | **NEW** — `merge_signals()` core logic (~90 lines) |
| `src/config.py` | Add `EnsembleConfig` class + field on `StrategyConfig` |
| `config/settings.yaml` | Add `ensemble:` section under `strategy:` |
| `src/automation/workflows.py` | Insert ensemble merge step; preserve `raw_signals` for rotation |
| `tests/unit/strategy/test_ensemble.py` | **NEW** — 8–10 test functions (~150 lines) |
| `tests/unit/strategy/test_regime_params.py` | Tier 26 review fixes (parametrized regime tests) |
| `tests/unit/strategy/test_dca.py` | Tier 26 review fix (RISK_ON identity test) |
| `tests/unit/strategy/test_mean_reversion.py` | Tier 26 review fix (RISK_ON identity test) |
| `tests/unit/backtesting/test_backtest_dca.py` or `test_backtest_mean_reversion.py` | Tier 26 review fix (crisis vs no-regime comparison) |

---

## Verification

1. `ruff check src/ tests/` — clean
2. `pytest tests/ -x` — 630+ tests, all green
3. `financialhub once` with `ensemble.enabled: false` — identical behavior to Tier 26
4. `financialhub once` with `ensemble.enabled: true` and both momentum + mean_reversion enabled — verify merged signals in SQLite log with `strategy_name="Ensemble"`
5. Check SQLite: individual strategy signals still logged (audit trail), followed by ensemble-merged signals passed to execution
