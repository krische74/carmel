# Tier 27 Code Review

## Scope
- **27A** — `src/strategy/ensemble.py`: `merge_signals()` weighted-vote signal merging
- **27B** — `EnsembleConfig` (enabled, strategy_weights, cash_signal_penalty, min_blended_weight)
- **27C** — Workflow integration: ensemble after logging, before ML gate; raw_signals for rotation
- **27D** — Tier 26 review test fixes (parametrized regime tests, RISK_ON identity, zero multiplier)

## Verdict: PASS — Clean

Zero bugs found. The ensemble logic is mathematically correct, workflow integration properly preserves raw signals for momentum rotation, and all Tier 26 review items are resolved.

---

## Production Code — No Issues

| Check | Status |
|-------|--------|
| Weighted blending math | CORRECT — `sum(weight × strategy_weight) / sum(strategy_weights)` verified: (0.6×0.5 + 0.25×0.5) / 1.0 = 0.425 |
| DCA pass-through | CORRECT — DCA signals partitioned first, never modified |
| Cash signal penalty | CORRECT — applied only when momentum signals cash; multiplies blended weights |
| Min weight filtering | CORRECT — epsilon comparison (`blended + 1e-12 < min_w`) prevents float noise |
| Cash fallback | CORRECT — cash signal passes through when all tactical longs drop below min |
| Unknown strategy handling | CORRECT — logs WARNING, skips with `continue` |
| Ensemble disabled (default) | CORRECT — `list(signals)` shallow copy returned unchanged |
| raw_signals for rotation | CORRECT — `raw_signals = list(signals)` before merge; rotation uses `raw_signals` |
| Signal logging audit trail | CORRECT — individual strategy signals logged before merge |
| Protected symbols includes Ensemble | CORRECT — TLH won't harvest ensemble longs |
| Weight rounding | CORRECT — 6 decimal places |
| Config validation | CORRECT — Pydantic bounds on all fields |

## Tier 26 Review Items — All Resolved

| ID | Status | Evidence |
|----|--------|----------|
| H1 | RESOLVED | `@pytest.mark.parametrize` for all 4 regime levels on both DCA and MR helpers |
| H2 | RESOLVED | Backtest comparison tests present |
| M1 | RESOLVED | `test_dca_amount_zero_crisis_multiplier()` exists |
| M2 | RESOLVED | `test_dca_signals_risk_on_matches_no_regime()` and MR equivalent exist |

## Test Coverage — Comprehensive

**10 ensemble unit tests** covering: disabled no-op, DCA pass-through, single strategy, two-strategy blend, cash penalty, cash fallback, min weight filter, flat pass-through, unknown strategy warning, DCA-only no error.

**2 workflow integration tests** covering: ensemble called when enabled, momentum rotation uses raw_signals.

## Minor Observations (not actionable)

| Item | Note |
|------|------|
| Cash signals checked twice in `ensemble.py` | Once for `cash_penalty` bool, once for `momentum_cash_signals` list. Slightly verbose but correct. |
| No test for 3+ strategies simultaneously | Edge case; pattern is clear from 2-strategy test. |
| No test for config validation edge cases | Pydantic handles this; not needed. |

---

## Summary: No fixes needed. Tier 27 is production-ready.
