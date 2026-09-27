# Tier 26 Sprint Contract — Regime-Aware DCA & Mean Reversion

**Depends on:** Tier 25 complete (582 tests green, lint clean, all prior review items resolved)
**Governance:** `AGENTS.md` — TDD (failing test first), ruff clean, no `print()` in production paths

---

## 1. Sprint goal

Make DCA and mean reversion strategies regime-aware, matching the pattern already established by momentum. DCA scales its dollar allocation down in defensive/crisis regimes. Mean reversion reduces max open positions. Both strategies receive `MarketRegime` in live cycles and backtests.

---

## 2. In scope

### 26A — Regime-Aware DCA

- **Intent:** In a crisis regime, continuing to DCA the full configured amount increases drawdown risk. Scale the DCA dollar amount by regime.
- **Pattern:** Follow `effective_adx_threshold()` in `src/strategy/regime_params.py`
- **New helper** (`src/strategy/regime_params.py`):
  ```python
  def effective_dca_amount(settings: Settings, overall: OverallRegime | None) -> float:
  ```
  - Returns `settings.strategy.dca.amount` when `overall is None`
  - Applies per-regime multiplier: risk_on=1.0, cautious=0.75, defensive=0.5, crisis=0.25
  - Multipliers are configurable via new `DCAConfig` fields
- **Config additions** (`src/config.py` → `DCAConfig`):
  ```python
  regime_amount_risk_on: float = Field(default=1.0, ge=0.0, le=2.0)
  regime_amount_cautious: float = Field(default=0.75, ge=0.0, le=2.0)
  regime_amount_defensive: float = Field(default=0.5, ge=0.0, le=2.0)
  regime_amount_crisis: float = Field(default=0.25, ge=0.0, le=2.0)
  ```
- **Strategy change** (`src/strategy/dca.py` → `generate_signals()`):
  - Add `market_regime: MarketRegime | None = None` parameter
  - Call `effective_dca_amount(settings, overall)` to get regime-adjusted amount
  - Use adjusted amount when computing per-target allocation weights
  - Signal `rationale` should note regime adjustment when active
- **Config in `settings.yaml`:**
  ```yaml
  dca:
    frequency: "weekly"
    amount: 100.0
    regime_amount_risk_on: 1.0
    regime_amount_cautious: 0.75
    regime_amount_defensive: 0.5
    regime_amount_crisis: 0.25
  ```
- **Tests:**
  - `test_effective_dca_amount_no_regime()` — returns base amount
  - `test_effective_dca_amount_crisis()` — returns 25% of base
  - `test_effective_dca_amount_risk_on()` — returns 100% of base
  - `test_dca_signals_with_crisis_regime()` — signal weights reflect reduced amount
  - `test_dca_signals_without_regime_unchanged()` — no regression

### 26B — Regime-Aware Mean Reversion

- **Intent:** In defensive/crisis regimes, mean reversion signals are riskier (volatility spikes cause false oversold readings). Reduce max open positions to limit exposure.
- **Pattern:** Follow `effective_adx_threshold()` in `src/strategy/regime_params.py`
- **New helper** (`src/strategy/regime_params.py`):
  ```python
  def effective_mean_reversion_max_positions(settings: Settings, overall: OverallRegime | None) -> int:
  ```
  - Returns `settings.strategy.mean_reversion.max_positions` when `overall is None`
  - Per-regime overrides: risk_on=4, cautious=3, defensive=2, crisis=1
  - Configurable via new `MeanReversionConfig` fields
- **Config additions** (`src/config.py` → `MeanReversionConfig`):
  ```python
  regime_max_positions_risk_on: int = Field(default=4, ge=0, le=10)
  regime_max_positions_cautious: int = Field(default=3, ge=0, le=10)
  regime_max_positions_defensive: int = Field(default=2, ge=0, le=10)
  regime_max_positions_crisis: int = Field(default=1, ge=0, le=10)
  ```
- **Strategy change** (`src/strategy/mean_reversion.py` → `generate_signals()`):
  - Add `market_regime: MarketRegime | None = None` parameter
  - Call `effective_mean_reversion_max_positions(settings, overall)` to get regime-adjusted limit
  - Use adjusted max_positions when slicing top candidates
  - Signal `rationale` should note regime adjustment when active
- **Config in `settings.yaml`:**
  ```yaml
  mean_reversion:
    # ... existing fields ...
    regime_max_positions_risk_on: 4
    regime_max_positions_cautious: 3
    regime_max_positions_defensive: 2
    regime_max_positions_crisis: 1
  ```
- **Tests:**
  - `test_effective_max_positions_no_regime()` — returns base (4)
  - `test_effective_max_positions_crisis()` — returns 1
  - `test_mean_reversion_signals_capped_by_regime()` — 4 candidates, crisis → only 1 signal
  - `test_mean_reversion_signals_without_regime_unchanged()` — no regression

### 26C — Workflow Integration

- **Intent:** Pass `market_regime` to DCA and mean reversion in the live trading cycle (currently only momentum receives it).
- **File:** `src/automation/workflows.py` (lines 174–184)
- **Change:** Replace the `isinstance(strat, MomentumRotationStrategy)` special case with passing `market_regime` to all strategies:
  ```python
  for strat in self._strategies:
      uni = strat.get_universe()
      sub = {s: combined[s] for s in uni if s in combined}
      signals.extend(
          strat.generate_signals(sub, as_of=when, market_regime=cycle_regime),
      )
  ```
  This works because all three strategies now accept `market_regime` as an optional kwarg. When `cycle_regime is None` (regime disabled), behavior is unchanged.
- **Strategy ABC update** (`src/strategy/base.py`):
  - Add `market_regime: MarketRegime | None = None` to the ABC `generate_signals()` signature so the interface is uniform
- **Tests:**
  - `test_workflow_passes_regime_to_dca()` — mock DCA, verify `market_regime=` kwarg received
  - `test_workflow_passes_regime_to_mean_reversion()` — same for mean reversion
  - `test_workflow_regime_none_when_disabled()` — regime disabled → all strategies get `None`

### 26D — Backtesting Integration

- **Intent:** Backtest engine should pass regime to DCA and mean reversion, matching live behavior (regime parity).
- **File:** `src/backtesting/engine.py` (lines 216–233)
- **Change:** Remove the `isinstance(strategy, MomentumRotationStrategy)` guard. Build regime for all strategies when `cfg.use_regime_for_momentum` is True (rename config field to `use_regime` or keep backward-compatible):
  ```python
  if settings is not None and cfg.use_regime_for_momentum:
      # Build regime snapshot for any strategy
      vix_sym = settings.regime.vix_symbol.strip().upper()
      vix_df = normalized.get(vix_sym)
      vx = vix_close_as_of(vix_df, as_of) if vix_df is not None else None
      regime = build_market_regime_snapshot(
          settings, as_of=as_of, vix_close=vx, yield_spread=None,
      )
      signals = strategy.generate_signals(data, as_of=as_of, market_regime=regime)
  else:
      signals = strategy.generate_signals(data, as_of=as_of)
  ```
- **Tests:**
  - `test_backtest_dca_with_regime()` — DCA backtest with regime data → signals reflect adjusted amount
  - `test_backtest_mean_reversion_with_regime()` — mean reversion backtest with regime → positions capped

---

## 3. Out of scope (deferred beyond Tier 26)

- Strategy ensemble / weighted signal voting
- Regime-aware RSI thresholds for mean reversion (tighter oversold in crisis)
- Limit/stop order types
- Streaming broker connection
- Dashboard authentication
- Multi-account support
- Renaming `use_regime_for_momentum` config field (keep backward-compatible for now)

---

## 4. Acceptance criteria

- [ ] `ruff check src/ tests/` clean
- [ ] Full `pytest` suite green; target **610+ tests** (currently 582, adding ~30)
- [ ] `market_regime` parameter added to Strategy ABC, DCA, and mean reversion
- [ ] `effective_dca_amount()` and `effective_mean_reversion_max_positions()` helpers in `regime_params.py`
- [ ] Workflow passes `market_regime` to all strategies uniformly (no more isinstance check)
- [ ] Backtest engine passes regime to all strategies (not just momentum)
- [ ] Default config values preserve exact current behavior (multiplier=1.0 for risk_on, base max_positions=4)
- [ ] Regime disabled (`regime.enabled: false`) → all strategies get `market_regime=None` → identical behavior to Tier 25
- [ ] `BUILD_STATE.md` updated with Tier 26 entry
- [ ] No `print()` in production paths; use `logging`

---

## 5. TDD task order (suggested)

**Phase 1 — Regime Parameter Helpers (26A + 26B config)**

1. Write `test_effective_dca_amount_no_regime()` → returns base amount
2. Write `test_effective_dca_amount_crisis()` → returns 25%
3. Implement `effective_dca_amount()` in `regime_params.py` → tests green
4. Write `test_effective_max_positions_no_regime()` → returns 4
5. Write `test_effective_max_positions_crisis()` → returns 1
6. Implement `effective_mean_reversion_max_positions()` in `regime_params.py` → tests green
7. Add config fields to `DCAConfig` and `MeanReversionConfig` in `config.py`
8. Add config values to `settings.yaml`

**Phase 2 — Strategy Changes (26A + 26B signals)**

9. Update Strategy ABC: add `market_regime: MarketRegime | None = None` to `generate_signals()`
10. Write `test_dca_signals_with_crisis_regime()` → signal reflects reduced amount
11. Write `test_dca_signals_without_regime_unchanged()` → no regression
12. Implement `market_regime` parameter in `dca.py` `generate_signals()` → tests green
13. Write `test_mean_reversion_signals_capped_by_regime()` → 4 candidates, crisis → 1 signal
14. Write `test_mean_reversion_signals_without_regime_unchanged()` → no regression
15. Implement `market_regime` parameter in `mean_reversion.py` `generate_signals()` → tests green

**Phase 3 — Workflow & Backtest Integration (26C + 26D)**

16. Write `test_workflow_passes_regime_to_dca()` → verify kwarg
17. Write `test_workflow_passes_regime_to_mean_reversion()` → verify kwarg
18. Simplify `workflows.py` — remove isinstance check, pass `market_regime` to all strategies
19. Write `test_backtest_dca_with_regime()` → regime-adjusted DCA in backtest
20. Write `test_backtest_mean_reversion_with_regime()` → regime-capped positions in backtest
21. Update `engine.py` — remove isinstance guard for regime passing
22. Run full `pytest` suite → all green
23. Update `BUILD_STATE.md` with Tier 26 entry

---

## 6. Risks and mitigations

| Risk | Mitigation |
|------|------------|
| Changing Strategy ABC signature breaks existing strategy subclasses | `market_regime` is keyword-only with default `None`; existing callers unaffected |
| DCA amount=0 in crisis blocks all contributions | Multiplier floor is 0.25 (25%), never zero; config validation enforces `ge=0.0` |
| Mean reversion max_positions=0 blocks all signals | Default crisis value is 1, not 0; config allows `ge=0` but default is safe |
| Removing isinstance check in workflows breaks momentum-specific logic | Momentum already accepts `market_regime`; uniform calling works for all strategies |
| Backtest regime change produces different results than prior runs | Only when `use_regime_for_momentum=True`; old backtests without regime are unchanged |

---

## 7. Definition of done

- All 4 sub-scopes (26A–26D) implemented with passing tests
- 610+ total tests, all green, ruff clean
- Zero regressions on existing 582 tests
- Default config produces identical behavior to Tier 25 (regime multipliers default to current behavior)
- `BUILD_STATE.md` updated with Tier 26 summary
- `regime_params.py` has three helper functions following the same pattern (ADX, DCA amount, max positions)
- Workflow passes regime uniformly — no strategy-specific isinstance checks

---

## Key files to modify

| File | Change |
|------|--------|
| `src/strategy/base.py` | Add `market_regime` kwarg to ABC |
| `src/strategy/dca.py` | Accept + use `market_regime` |
| `src/strategy/mean_reversion.py` | Accept + use `market_regime` |
| `src/strategy/regime_params.py` | Add `effective_dca_amount()` and `effective_mean_reversion_max_positions()` |
| `src/config.py` | Add regime fields to `DCAConfig` and `MeanReversionConfig` |
| `config/settings.yaml` | Add regime multiplier values to dca and mean_reversion sections |
| `src/automation/workflows.py` | Remove isinstance check; pass regime to all strategies |
| `src/backtesting/engine.py` | Remove isinstance guard; pass regime to all strategies |
| `tests/unit/strategy/test_regime_params.py` | New tests for DCA + mean reversion helpers |
| `tests/unit/strategy/test_dca.py` | Add regime-aware signal tests |
| `tests/unit/strategy/test_mean_reversion.py` | Add regime-capped position tests |
| `tests/unit/automation/test_workflows.py` | Add regime-passing verification tests |
| `tests/unit/backtesting/test_backtest_dca.py` | Add regime backtest test |
| `tests/unit/backtesting/test_backtest_mean_reversion.py` | Add regime backtest test |

---

## Verification

1. `ruff check src/ tests/` — clean
2. `pytest tests/ -x` — 610+ tests, all green
3. `financialhub once` with `regime.enabled: true` — verify DCA and mean reversion signals log regime adjustment
4. `financialhub backtest --strategy dca --start 2023-01-01 --end 2024-01-01` — completes with regime-adjusted equity curve
5. `financialhub backtest --strategy mean_reversion --start 2023-01-01 --end 2024-01-01` — completes with regime-capped positions
