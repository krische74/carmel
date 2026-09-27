# AGENTS.md — AI Co-Development Guide for Carmel

This file is the authoritative guide for any AI coding agent working in this repository. Read it fully before writing any code or tests.

---

## Project Summary

**Carmel** is an automated personal investing platform for a married couple, built in Python. The system ingests market data from free sources, applies quantitative analysis, executes swing trades and multi-strategy investments via Alpaca, and explains every decision in plain language.

Personal use only — not a fund, no regulatory registration needed. Starting with limited capital, scaling sophistication as capital grows.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.12+ (type hints required) |
| Package Manager | uv |
| Configuration | pydantic-settings + YAML + .env |
| Data Sources | yfinance, fredapi/FRED, edgartools (SEC) |
| Data Storage | SQLite (metadata/state) + Parquet (prices) + DuckDB (queries) |
| Broker | Alpaca via alpaca-py (paper trading by default) |
| Scheduling | APScheduler |
| API | FastAPI |
| Dashboard | Streamlit (MVP) |
| Backtesting | Backtrader |
| ML (future) | scikit-learn, XGBoost/LightGBM, SHAP |
| LLM (future) | Ollama (local), LangChain/LlamaIndex (RAG) |
| Testing | pytest + pytest-cov + pytest-mock |
| Linting | ruff (replaces flake8 + black + isort) |
| CI | GitHub Actions |
| Containers | Docker + Docker Compose |

---

## Non-Negotiable Rules

1. **Never write production code without a failing test first.** No exceptions.
2. **Never commit with failing tests.** Keep the suite green at every checkpoint.
3. **Never use `git add .`**. Stage each file individually.
4. **`git add .` is banned.** (Repeated because it matters.)
5. **Commit at each tier boundary** on a green suite — stage files individually, review
   the diff, no secrets or `data/cache/*.sqlite`. Tag pre-live milestones
   (e.g. `pre-live-tier-49`).
6. **No scope creep.** If a task does not require a change, do not make it.
7. **Evidence over assumption.** Say "I verified X" or "I believe X (unverified)". Never confabulate.
8. **No secrets in code.** All API keys, tokens, and credentials go in `.env` (gitignored). Never hardcode them.
9. **Paper trading by default.** Live trading requires explicit configuration change. Never default to live.
10. **All broker interactions go through `BrokerInterface`.** No direct alpaca-py calls from strategies or risk modules.
11. **All data access goes through adapters.** No direct yfinance/FRED calls from strategies. Strategies talk to storage, not APIs.
12. **No lookahead bias.** Signal code must never access future data. Use point-in-time data only.
13. **Data validation before calculation.** Never compute indicators on unvalidated data.
14. **Risk checks before every order.** No order reaches the broker without passing pre-trade checks.
15. **Never end a task with an unraised question.** If you were unsure about something, say so in
    the response and route it. See **Open Questions Protocol**.

---

## Repository Layout

```
Carmel/
  .github/workflows/ci.yml
  config/
    settings.yaml             # Non-secret configuration
    logging.yaml              # Logging configuration
  src/
    __init__.py
    config.py                 # pydantic-settings, loads YAML + .env
    models.py                 # Shared domain models (Pydantic)
    data/
      adapters/               # MarketDataAdapter ABC + implementations
        base.py
        yfinance_adapter.py
        fred_adapter.py
      storage/                # Parquet writer, DuckDB queries, SQLite
        base.py
        parquet_store.py
        sqlite_store.py
        duckdb_queries.py
      validation.py
      pipeline.py             # Orchestrates fetch -> validate -> store
    strategy/
      base.py                 # Strategy ABC, Signal model
      indicators.py           # Technical indicator calculations
      momentum.py             # ETF momentum rotation
      dca.py                  # Dollar-cost averaging
    risk/
      position_sizing.py
      pre_trade_checks.py
      kill_switch.py
      portfolio_limits.py
    execution/
      broker_interface.py     # BrokerInterface ABC
      alpaca_adapter.py
      order_manager.py
    portfolio/
      state.py                # Holdings, cash, P&L
      rebalancer.py
      tax_lots.py             # Lot-level tracking
    automation/
      scheduler.py            # APScheduler jobs
      workflows.py            # End-to-end pipeline
      health.py               # Heartbeat, dead-man switch
    dashboard/
      app.py                  # Streamlit main
      pages/
    ai/
      explainer.py            # Decision explanations
      rag.py                  # RAG over portfolio data
    reporting/
      performance.py          # QuantStats / pyfolio integration
      attribution.py          # Brinson-style attribution
  tests/
    conftest.py               # Shared fixtures
    unit/
      data/
      strategy/
      risk/
      execution/
      portfolio/
    integration/
    fixtures/                 # Sample data files
  docs/
    research/                 # Deep research notebooks
    architecture/             # ADRs
  scripts/
  AGENTS.md
  BUILD_STATE.md
  Dockerfile
  docker-compose.yml
  pyproject.toml
  .env.example
```

---

## Architecture: Data Flow

```
Market Data Sources (yfinance, FRED, SEC)
  └─ [1] Adapters fetch raw data
  └─ [2] Validator checks OHLCV consistency
  └─ [3] Storage persists to Parquet/SQLite
  └─ [4] Strategy reads from storage, generates Signals
  └─ [5] Risk module checks each Signal (sizing, limits, kill switch)
  └─ [6] Execution sends approved orders to broker
  └─ [7] Portfolio state updates from fill confirmations
  └─ [8] Reporting computes metrics
  └─ [9] Dashboard displays everything
```

Key invariant: data flows one direction. Strategies never call APIs. Risk never calls the broker. Each module has a single responsibility and a clean interface.

---

## Known Non-Blocking Alerts

Discord alerts that fire periodically during paper trading but are **not** indicators of broken code. Don't chase unless the user asks or they escalate.

- **`reconciliation` qty_mismatch warnings** — SQLite execution ledger vs broker position drift. Triggered by manual trades placed outside the daemon (e.g. a manual rebalance on Alpaca's dashboard). Self-heals over subsequent cycles as trade executions sync. If persistent across >5 cycles, run `carmel reconcile` (if wired) or investigate.
- **`lot_reconciliation` (Tier 46/47)** — tax-lot open qty vs broker positions, and ledger equity (lot MV + cash) vs broker equity. Per-symbol qty mismatch is WARNING; total equity beyond **~3% relative / $50 absolute** (`max($50, 3% × broker_equity)`) is CRITICAL. On a small account the absolute floor can govern. Run `carmel rebuild-lots --since YYYY-MM-DD` if CRITICAL after a known reset; do not ignore CRITICAL equity mismatches. Day-to-day parquet-close vs Alpaca-mark divergence is expected — do not tighten below that floor.
- **`ingest_failure` for single symbols** — yfinance occasionally returns NaN OHLCV rows for individual ETFs (IVV observed 2026-04-18). Transient. Only act if the same symbol fails two consecutive days.
- **`pre_trade_reject` for concentration** — the portfolio-limits guardrail ([src/execution/order_manager.py](src/execution/order_manager.py)) rejecting orders that would breach `risk.max_position_pct`. **As of Tier 43 this is no longer routine noise.** Fixed-weight orders are now sized on the *delta* to target, so a position already at target sizes to 0 and is skipped below the min-order floor (an `order_skipped_below_floor` DEBUG line), not rejected. If a `pre_trade_reject` for concentration *still* fires post-Tier 43, treat it as a **genuine signal worth investigating** (e.g. a target that actually exceeds cap after a config change, or DCA contributions stacking past the cap) — the old "expected, don't chase" guidance no longer applies.
- **`pre_trade_reject` for cash reserve (Tier 47)** — a strategy entry failed to fund. After 47B this should self-heal via unsweep; if it still fires with the ordinary reserve reason (not `insufficient cash after partial unsweep`), treat as **RED** per `LIVE_RUNBOOK.md`.
- **`leverage` CRITICAL (Tier 48)** — cash < 0 or Alpaca reports margin borrowing. Every risk control assumes an unlevered account; treat as **RED** per `LIVE_RUNBOOK.md`. Negative-cash sweep holds log WARNING (degenerate guard); the end-of-cycle invariant fires independently of the sweep path.
- **`order_rejected` info-level alerts** — operational info, not an error. Summarized in the cycle report.
- **`CashSweep` buy/sell orders (Tier 44/45)** — routine cash management (INFO), not strategy signals. The sweep parks idle cash above the reserve+buffer band in a short-term T-bill ETF (**BIL** by default) and sells it back on a **two-sided thermostat** (Tier 45A): hard breach when cash < reserve, and soft refill when cash sits below the mid-band low-water (expected ~every 1–2 weeks under current drain). A sweep-symbol position **well above `risk.max_position_pct`** is **by design** — `sweep_buy` deliberately bypasses the position cap (T-bills at half the account is not the concentration risk the cap protects against). It still respects the kill switch, cash reserve, and the $5 order floor. The sweep symbol is held outside the momentum universe so rotation never touches it.

## Known Operational Caveats

Not alerts, but gotchas that have surfaced during paper trading:

- **Account cache + fill latency + settlement latency (Tier 47 / 48)** — `AlpacaBrokerAdapter` caches cash/equity until the next **order submit** (invalidate-on-submit) or an explicit `refresh_account()`. Positions are fetched live. Any code path that reads account state after placing an order in the same cycle must account for **cache staleness**, **fill latency** (`get_order_status() == filled`), and **settlement latency** (cash debits/credits landing after `filled` — 8/19 proved status leads cash on buys). Poll with `refresh_account()` on every iteration (`OrderManager.wait_for_fill`); for **buys**, exit on cash decrease, not on `filled` alone. Do not add a TTL; do not refresh inside `snapshot_from_broker`. Size same-cycle sweeps with `unsettled_pending_buy_notional()`.
- **Equity snapshots (Tier 46/47)** — `equity_snapshots.total_market_value` is **positions-only**. Account equity is `total_market_value + cash`, or `broker_equity` when present (same broker read as cash). `cash` / `broker_equity` are nullable: NULL means unknown (never treat as 0.0). Daily returns prefer `broker_equity` and skip the first basis-change pair.
- **Cash reserve + pending limit orders** — `remaining_cash` only decrements on fill in [src/execution/order_manager.py](src/execution/order_manager.py). Multiple pending limit orders in one cycle can appear "cash rich" for later legs. Same class of issue as the original three-QQQ-orders bug, on the cash axis. Deferred fix; be aware when reviewing order-manager changes.
- **`list_recent_orders(limit=100)` tail truncation** — the portfolio-limits guardrail queries the 100 most recent orders regardless of status. On busy accounts, old pending orders can fall off the tail and be missed. Documented in [_pending_buy_notional_by_symbol](src/execution/order_manager.py) docstring; fix is a separate `status=open` broker API extension.
- **Paper trading cadence (as of Tier 45B, 2026-08-17; MR disabled Tier 44)** — **weekly DCA on Mondays** at 2.5% of equity (same weekly deployment rate as the prior 0.5%/day drip; legs clear the $5 floor at small account sizes), weekly momentum rebalance. **Mean reversion is disabled** as of Tier 44A (config block retained in `settings.yaml` but removed from `strategy.enabled`) — see BUILD_STATE.md tech debt for the two structural bugs and the redesign analysis that must be resolved before re-enabling. Expect far fewer orders/month than the daily-DCA era (~a handful of DCA legs + occasional CashSweep refill sells). Previous defaults were daily DCA 0.5% / weekly momentum; older decision docs may reflect the old cadence.
- **Log contamination from pytest / unittest.mock** — test runs have bled into production log files in past sessions, causing Claude to misdiagnose a Mock broker traceback as a production fallback. Always confirm log provenance (is this a test run? are there `unittest.mock` markers in surrounding lines? what process wrote this log?) before root-causing log-based alerts.
- **Backtest window (Tier 51/52)** — a claim about trend-following (momentum) is incomplete without the date window. A bull-only slice (e.g. 2023–2026) measures the insurance *premium* and cannot measure the *payout*. State start/end, constrained vs `--raw-signal`, and **momentum-only vs full-system** (DCA starts 2011-01-28). Cash yield is FRED DTB3 (not a flat 4%). Do not rank different-exposure books on Calmar. See `BUILD_STATE.md` Tier 52.

---

## Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what the current task or sprint contract asks for.
- No abstractions for single-use code (no `Strategy` base class for one strategy variant, no `SizingPolicy` ABC for one sizing rule).
- No "flexibility" or "configurability" that wasn't requested (no extra config knobs, no parameterization for hypothetical future use).
- No error handling for impossible scenarios (don't validate inputs that Pydantic has already validated, don't wrap calls that cannot raise).
- If you wrote 200 lines and 50 would do, rewrite it before declaring GREEN.

The test: "Would a senior engineer call this overcomplicated?" If yes, simplify before moving on.

Pattern to avoid:

```python
# Overengineered for a single sizing rule
class SizingStrategy(ABC):
    @abstractmethod
    def size(self, equity: float, signal: Signal) -> float: ...

class FixedFractionSizing(SizingStrategy):
    def __init__(self, fraction: float, min_position: float, max_position: float, ...):
        ...

# Simple. Refactor when a second sizing rule actually appears.
def size_position(equity: float, fraction: float = 0.005) -> float:
    return equity * fraction
```

When complexity is genuinely needed (multiple broker adapters, multiple data sources), follow the existing project patterns (`BrokerInterface`, `MarketDataAdapter`). Do not invent new patterns for one-off cases.

---

## Surgical Edits

**Touch only what you must. Match existing style. Clean up only your own mess.**

When editing existing code:

- Only change lines that trace directly to the current task. Every changed line should map to a requested behavior change.
- Match the existing style — quote style, type-hint density, docstring presence, naming. If the surrounding file uses one pattern, follow it even if you would do it differently in a greenfield context.
- Do not refactor things that aren't broken. Do not "improve" adjacent code, comments, or docstrings while you happen to be in the file.
- If you notice unrelated dead code or a real refactoring opportunity, **mention it in the response — do not act on it.** It either becomes a separate task or goes into `BUILD_STATE.md` tech debt.

When your changes create orphans:

- Remove imports, variables, helpers, or types that **your** changes made unused.
- Do not remove pre-existing dead code unless explicitly asked.

The test: every line in the diff should be defensible as a direct consequence of the current task. If you cannot point at the task and explain why this line had to change, revert it.

This rule matters extra in this codebase: changes to `src/risk/`, `src/strategy/`, or `src/execution/` that aren't traceable to the task at hand are the highest-stakes regressions possible — real money is downstream. Drive-by reformatting in those directories is forbidden.

---

## Coding Conventions

### Python style
- Python 3.12+ features are encouraged (type hints, `|` union syntax, f-strings)
- Type hints on all public functions and methods
- Docstrings on all public classes and functions (Google style)
- Use `logging` module — never `print()` for debugging
- Use Pydantic models for all data structures that cross module boundaries

### Naming
- Files: `snake_case.py` (e.g., `yfinance_adapter.py`, `kill_switch.py`)
- Classes: `PascalCase` (e.g., `MomentumRotationStrategy`, `AlpacaAdapter`)
- Functions/variables: `snake_case`
- Constants: `SCREAMING_SNAKE_CASE`
- Environment variables: `SCREAMING_SNAKE_CASE`
- Config keys (YAML): `snake_case`
- Pydantic models: `PascalCase` suffixed with purpose (e.g., `RiskConfig`, `Signal`)

### Error handling
- Use typed exceptions — never bare `except:` or `except Exception:`
- Log errors with context (symbol, timestamp, order ID, etc.)
- Financial operations must fail loudly — never swallow errors silently
- Broker errors: distinguish transient (retry) from terminal (log and alert)

### Imports
- Group imports: stdlib, then third-party, then `src.*`
- ruff handles import sorting automatically
- Use absolute imports from `src` (e.g., `from src.config import Settings`)

---

## Test Conventions

### Where tests live
All tests are in `tests/`. One test file per module.
- `tests/unit/` — fast, isolated, no external dependencies
- `tests/integration/` — tests that cross module boundaries
- `tests/fixtures/` — sample data files shared across tests

### Running the suite
```bash
# Full suite
.venv\Scripts\python -m pytest tests/ -v

# Single file
.venv\Scripts\python -m pytest tests/unit/test_config.py -v

# With coverage
.venv\Scripts\python -m pytest tests/ --cov=src --cov-report=term-missing

# Lint
.venv\Scripts\ruff check src/ tests/
```

### Test naming
Use descriptive names that state the behavior under test:
```python
# Good
def test_momentum_strategy_rotates_to_cash_when_all_below_sma():
def test_kill_switch_blocks_orders_after_daily_loss_limit():
def test_alpaca_adapter_retries_on_network_timeout():

# Bad
def test_strategy():
def test_it_works():
```

### Arrange / Act / Assert
Every test has a clear three-part structure. One behavior per test.

### Mocking external services
- Unit tests NEVER call real APIs (yfinance, Alpaca, FRED)
- Mock at the adapter boundary — the adapter interface is the seam
- Use deterministic fixture data (from `tests/conftest.py` or `tests/fixtures/`)
- Integration tests may use Alpaca paper trading API when explicitly marked

---

## TDD Workflow (Required)

This project follows strict red/green/refactor TDD.

### Before writing any code

1. Break the feature into the **smallest independently testable behavior**.
2. State what you expect the test to demonstrate.
3. Write the test. Run it. Verify it **fails for the right reason** — not a missing import, not a syntax error, but missing implementation.

### RED -> GREEN -> REFACTOR cycle

#### RED: Write the failing test
- Write a single test for the next smallest behavior.
- Verify pytest reports failure with the expected error.
- Do not write another test until the current one is green.

#### GREEN: Minimal implementation
- Write the simplest code that makes the test pass.
- Hardcoding a return value is acceptable in GREEN — the next test will force generalization.
- Run the full suite: new test must pass, nothing else must break.

#### REFACTOR: Improve while green
- Remove duplication, improve naming, extract helpers.
- Run the suite after **every small change**. Do not batch refactoring steps.
- Do not add features during refactoring.

### Mandatory checkpoints
- After each RED/GREEN/REFACTOR cycle: run the full suite, log the result.
- After every 5 actions without running tests: run tests immediately.
- Before any `git commit`: full suite must be green.

### STOP rules — halt and report if:
- A test failure cannot be explained or fixed within 2 attempts.
- More than ~20 lines of production code written without a failing test driving them.
- About to refactor while tests are failing.
- Reality contradicts expectation — state what you expected, what happened, why your model was wrong.
- Scope is expanding beyond the current tier.

---

## Explicit Reasoning Protocol

Use this for any action that could fail:

```
DOING: <exact action>
EXPECT: <specific predicted outcome>
```

After:
```
RESULT: <what actually happened>
MATCHES: yes / no
THEREFORE: <conclusion — or STOP if unexpected>
```

If `MATCHES: no` -> stop, state the discrepancy, form a theory, propose a fix, get confirmation before retrying.

---

## Open Questions Protocol

**Standing principle (Keith, 2026-08-20):** *"I'll always lean towards building things as sound as
possible every step of the way. If you ever have questions let's bring them up and see about where we
can find an answer."*

Uncertainty is a work item, not something to route around. A question raised costs a paragraph. A
question deferred costs a tier.

### Recognize when you have a question

All of these count, and every one must be surfaced in the response rather than absorbed into a
summary:

- You made a choice where a defensible alternative existed and nothing in the task, this file, or the
  code decided it for you.
- You wrote a comment, docstring, or contract line that **argues** a case rather than **asserting** a
  verified fact.
- You are relying on a behavior you have not observed — the broker's, a library's, an API's.
- A number, threshold, or parameter got picked because it seemed reasonable.
- Something contradicted what you expected and you moved on anyway.

State it as three things: **the question, what it depends on, and what would settle it.**

### Route it to whatever can actually answer it

Take the cheapest rung that can close it. Do not skip up the ladder — most questions die on rung 1.

1. **The code or the data.** Read it, grep it, query it, run it. Most "I think X" is one command away
   from "I verified X."
2. **A test.** When the behavior can be constructed. **Mandatory when the question concerns timing or
   ordering between two operations** — write the test, not the argument. Twice-earned here: Tier 44's "rotation sells are self-funding" caused the 8/18 sweep
   lockout, and Tier 47's buy-vs-sell scoping gap caused the 8/19 leverage event. Both were arguments
   in a sprint contract's risk table that read as sound and were never tested.
3. **An experiment.** An ablation run, a backtest over a different window, a rebalance-weekday
   dispersion run. Cheap. Run it rather than reasoning about what it would show.
4. **External deep research.** For what this codebase structurally cannot answer: prior art, whether
   an approach is known to fail, legal/tax/regulatory treatment, what the literature says about a
   design choice. Write the research prompt as part of the deliverable — do not offer to write one
   later. Deep research is treated as free here and is expected to be used.
5. **Keith.** Preference, risk tolerance, priority, scope — judgment calls, not facts. One focused
   question.

### Verify state claims before you make them

There is a difference between *asking* whether something is broken and *asserting* that it is. Before
saying that something is broken, stale, missing, unimplemented, or currently doing X, check the most
recent evidence for that same signal and **state what you found there, including when it agrees with
you.** One observation about one date is never a claim about current state.

This is a separate failure from an unraised question: the ladder above only helps with uncertainty
you actually feel, and the errors this rule exists to prevent were confident claims that never
registered as questions. In every recorded instance the disconfirming data was already in the output
being read. The tell is always the same — a strong signal appears, and reading stops.

### Nothing gets deferred silently

- A question that cannot close in this task goes into [OPEN_QUESTIONS.md](OPEN_QUESTIONS.md) with the
  date, why it matters, and its rung. Not into a summary paragraph that scrolls away.
- **"Worth revisiting later" is not a resolution.** Either it is written down with a route, or it is
  closed now.
- When an answer lands, record it next to the question — and say plainly when the answer was not what
  you expected. Those are the entries worth keeping.

---

## Harness Patterns (Multi-Agent Workflow)

This project uses a three-agent workflow:

| Role | Agent | Responsibility |
|------|-------|----------------|
| Planner | Opus 4.6 | Sprint contracts, architecture decisions, code review |
| Builder | Composer 2.0 | TDD implementation against sprint contracts |
| Evaluator | Opus 4.6 | Code review at tier completion, grading against criteria |

### Context Reset Protocol

Each new Composer session starts by reading these files in order:
1. `AGENTS.md` (this file) — coding conventions and rules
2. `BUILD_STATE.md` — current tier, what is done, what is next
3. The current tier's sprint contract (if separate)
4. Run `pytest` to confirm green baseline

### BUILD_STATE.md

A living document maintained in the repo root. Updated after every tier. Contains:
- Current tier and status
- What has been built (module summaries)
- What is next
- Known issues / tech debt
- Key decisions made and why

---

## Tier Completion Criteria

A tier is done when:

- [ ] All tests pass (green bar, no skips on critical paths)
- [ ] New behavior is covered by tests at unit and integration levels
- [ ] ruff reports no errors (`ruff check src/ tests/`)
- [ ] Type hints on all public interfaces
- [ ] Docstrings on all public classes and functions
- [ ] No commented-out code or incomplete TODOs in production files
- [ ] No hardcoded secrets or `print()` debugging
- [ ] Architecture principles upheld (abstractions, single responsibility)
- [ ] Financial domain constraints respected:
  - No lookahead bias in signal code
  - Data validation present before any calculation
  - Risk checks enforced before any order
  - Paper trading mode is the default
- [ ] BUILD_STATE.md updated
