# Carmel: How It Works

A complete reference for understanding the system — from data ingestion through trade execution to tax filing.

---

## Table of Contents

1. [Overview](#1-overview)
2. [Configuration](#2-configuration)
3. [CLI Commands](#3-cli-commands)
4. [The Trading Cycle](#4-the-trading-cycle)
5. [Strategies](#5-strategies)
6. [Market Regime Detection](#6-market-regime-detection)
7. [Risk Management & Safety](#7-risk-management--safety)
8. [Execution & Broker Interface](#8-execution--broker-interface)
9. [Data Storage](#9-data-storage)
10. [Tax Lot Tracking & Reporting](#10-tax-lot-tracking--reporting)
11. [Dashboard & API](#11-dashboard--api)
12. [Backtesting](#12-backtesting)
13. [Optional Features (ML, LLM, Fundamentals)](#13-optional-features)
14. [Deployment](#14-deployment)
15. [Monitoring & Alerts](#15-monitoring--alerts)
16. [Common Workflows](#16-common-workflows)
17. [Architecture Principles](#17-architecture-principles)
18. [File Reference](#18-file-reference)

---

## 1. Overview

Carmel is an automated personal investing platform. It ingests free market data, applies quantitative strategies, enforces multi-layer risk controls, executes trades via Alpaca, tracks tax lots, and displays analytics via a Streamlit dashboard.

**Core data flow:**

```
Market Data (yfinance, FRED)
  → Validation & Storage (Parquet + SQLite)
    → Strategy Signals (momentum, DCA, mean reversion)
      → Ensemble Merge (optional weighted voting)
        → ML Gate (optional score filter)
          → Risk Checks (kill switch, position limits, PDT)
            → Order Execution (Alpaca — market or limit)
              → Lot Tracking (FIFO, wash sales)
                → Reporting & Dashboard
```

**Key invariant:** Data flows one direction. Strategies never call the broker. Risk checks gate every order. Paper trading is the default.

---

## 2. Configuration

Settings load from three sources (later overrides earlier):

| Source | Contains | Committed to Git? |
|--------|----------|-------------------|
| `config/settings.yaml` | All non-secret parameters | Yes |
| Environment variables | Overrides for any field | No |
| `.env` file | Secrets (API keys, passwords) | No (gitignored) |

### Environment variable patterns

**Flat** (top-level credentials): `ALPACA_API_KEY`, `FRED_API_KEY`, `DASHBOARD_PASSWORD`, etc. These map directly to single-segment `Settings` fields.

**Nested** (configuration sections): use `__` as the delimiter so pydantic-settings can populate nested models:

- `NOTIFICATION__WEBHOOK_URL` → `settings.notification.webhook_url`
- `NOTIFICATION__ENABLED=true` → `settings.notification.enabled = True`
- `BROKER__PAPER_TRADING=false` → `settings.broker.paper_trading = False`

Environment variables always override YAML values. Keep credentials in `.env`; keep non-secret tuning in `config/settings.yaml`.

For **nested** settings, if a field name appears explicitly in YAML (for example `notification.enabled: false`), that value is treated as fixed and the matching `NOTIFICATION__ENABLED` env var will not override it. Omit keys you want to control only from `.env`.

### Key Config Sections

**`broker:`** — Alpaca connection, paper/live toggle, multi-account definitions

**`strategy:`** — Which strategies are enabled, their parameters, ensemble settings

**`risk:`** — Position limits, kill switch threshold, PDT protection, sizing method

**`execution:`** — Order type (market/limit), limit offset, stop-loss, unfilled timeout

**`regime:`** — VIX/yield curve thresholds, sizing multipliers per regime

**`tax:`** — Harvest thresholds, lot selection method (FIFO/HIFO), wash-sale window, replacement map

**`scheduler:`** — Cron expressions for each job, timezone, retry policy, log rotation

**`notification:`** — Webhook URL, email SMTP settings

**`ml:`** — Scoring threshold, model path, affect_orders toggle

**`dashboard:`** — Port, auth toggle, password

### Symbol dependency resolution (OHLCV)

Tickers can appear in several YAML places: `data.universe`, `data.dca_targets`, `strategy.momentum.cash_symbol` (cash when momentum rotates defensive), `strategy.mean_reversion.universe`, and both keys and values of `tax.replacement_map` (TLH substitutes). **`resolve_required_symbols(settings)`** in `src/data/symbol_resolver.py` unions all of these into one normalized set. `carmel ingest` (with no `--symbols`), the trading-cycle ingest, setup wizard backfill, and preflight’s Parquet coverage check all use this resolver so a symbol like **SHV** is never skipped just because it only appears as the momentum cash ETF. Pass explicit `--symbols A,B` to ingest only those tickers. **VIX** (`regime.vix_symbol`) is not part of that union; regime handling adds it on ingest paths that need it.

### Multi-Account Setup

For multiple Alpaca accounts (brokerage + IRA), configure `broker.accounts` in YAML. Each account references env var *names* — actual keys stay in `.env`:

```yaml
broker:
  accounts:
    - name: "Main"
      account_type: "brokerage"
      api_key_env: "ALPACA_ACCOUNT_MAIN_KEY"
      api_secret_env: "ALPACA_ACCOUNT_MAIN_SECRET"
    - name: "IRA"
      account_type: "ira_traditional"
      api_key_env: "ALPACA_ACCOUNT_IRA_KEY"
      api_secret_env: "ALPACA_ACCOUNT_IRA_SECRET"
```

When `accounts` is empty, the system falls back to the legacy `ALPACA_API_KEY` / `ALPACA_SECRET_KEY` env vars with a single "default" account.

---

## 3. CLI Commands

| Command | What It Does |
|---------|-------------|
| `carmel setup` | Interactive first-time wizard — creates `.env`, validates Alpaca connection |
| `carmel config` | Print non-secret settings to terminal |
| `carmel once` | Run one complete trading cycle and exit |
| `carmel run` | Start the background daemon (scheduler + all cron jobs) |
| `carmel backtest` | Run historical backtest with walk-forward + Monte Carlo |
| `carmel reconcile` | Compare local execution log to broker fills |
| `carmel serve` | Start FastAPI REST server (port 8000) |
| `carmel ml-train` | Train momentum ML classifier from Parquet data |
| `carmel index-rag` | Build document embeddings for LLM-assisted Q&A |
| `carmel preflight` | Pre-live validation — checks config, account, risk settings |

---

## 4. The Trading Cycle

`TradingWorkflow.run_cycle()` is the heart of the system. One cycle does everything:

### Step-by-step

1. **Refresh account** — query Alpaca for latest equity, cash, positions
2. **Cancel stale orders** — limit orders unfilled past timeout get cancelled
3. **Ingest OHLCV** — fetch latest price bars for all symbols via yfinance, store in Parquet
4. **Detect market regime** (if enabled) — classify VIX + yield curve into RISK_ON / CAUTIOUS / DEFENSIVE / CRISIS
5. **Generate signals** — each enabled strategy reads Parquet data and produces Signal objects (symbol, direction, weight, confidence, rationale)
6. **Log signals** — write each signal to SQLite (audit trail — preserves what each strategy said)
7. **Ensemble merge** (if enabled) — blend tactical signals by configurable strategy weights; DCA passes through untouched
8. **ML gate** (if enabled) — drop signals below ML confidence threshold
9. **Risk checks** — for each signal: kill switch, position limit, cash reserve, PDT check
10. **Execute orders** — submit to Alpaca via OrderManager (market or limit); place stop-loss if configured
11. **Record lots** — update tax lot ledger (FIFO open lots for buys, close lots for sells)
12. **Reconcile** — compare this cycle's orders to broker fills, flag discrepancies
13. **Snapshot equity** — record portfolio value to SQLite for equity curve
14. **Generate alerts** — kill switch, reconciliation warnings, ML staleness, etc.
15. **Notify** — send alerts via webhook/email if configured

**Duration:** 30–60 seconds per cycle. Cycles never overlap — the next one waits for the current one to finish.

**Multi-account:** The runner iterates accounts sequentially — one full cycle per account, same shared SQLite database partitioned by `account_id`.

---

## 5. Strategies

### Momentum Rotation

Rotates capital among ETFs based on absolute momentum, with a cash fallback.

- **Universe:** SPY, QQQ, TLT, GLD (configurable)
- **Logic:** Rank symbols by blended return (1/3/6/12 month lookbacks). Pick the top performer that is above its 200-day SMA and has ADX ≥ threshold (confirming a real trend). If nothing qualifies, signal cash (SHV).
- **Rebalance:** Monthly
- **Regime effect:** ADX threshold loosens in RISK_ON (allow weaker trends), tightens in CRISIS (require strong confirmation)

### Dollar-Cost Averaging (DCA)

Scheduled recurring purchases of long-term targets.

- **Targets:** Configured symbol/weight pairs (e.g., VOO 60%, VXUS 30%, BND 10%)
- **Frequency:** Weekly or monthly
- **Amount:** Fixed dollar amount per cycle (e.g., $100)
- **Regime effect:** Amount scales down — 100% in risk-on, 75% cautious, 50% defensive, 25% crisis

### Mean Reversion

Opportunistic entry into oversold assets using Bollinger Bands + RSI.

- **Entry:** Price below lower Bollinger Band AND RSI < 30 AND price above 200-day SMA
- **Exit:** Oversold condition clears (price rises back above lower band or RSI normalizes)
- **Max positions:** 4 simultaneous (configurable)
- **Regime effect:** Max positions cap scales — 4 in risk-on, 3 cautious, 2 defensive, 1 crisis

### Ensemble Merger (optional)

When `strategy.ensemble.enabled: true`, tactical long signals from multiple strategies are blended before execution:

- DCA signals pass through unchanged (they use a separate dollar-budget sizing path)
- Tactical longs are grouped by symbol and weighted-averaged using configurable `strategy_weights`
- If momentum signals cash (defensive), a penalty multiplier reduces all tactical weights
- Signals below `min_blended_weight` are dropped
- Result: one merged signal per symbol with `strategy_name="Ensemble"`

---

## 6. Market Regime Detection

Classifies the macro environment from two inputs:

| Input | Source | Levels |
|-------|--------|--------|
| **VIX** (volatility index) | yfinance `^VIX` | Low (<15), Normal (15-20), Elevated (20-30), Crisis (>30) |
| **Yield curve** (10Y-2Y spread) | FRED API | Normal (>+0.5%), Flat (±0.5%), Inverted (<-0.5%) |

**Combined classification:**

| Overall Regime | Sizing Multiplier | When |
|---------------|-------------------|------|
| RISK_ON | 1.0× | Low/normal VIX + normal yield curve |
| CAUTIOUS | 0.75× | Elevated VIX or flat yield curve |
| DEFENSIVE | 0.5× | High VIX + flat/inverted yield curve |
| CRISIS | 0.25× | Very high VIX + inverted yield curve |

The sizing multiplier scales all non-DCA position sizes. Each strategy also has its own regime-specific parameter adjustments (ADX threshold, DCA amount, max positions).

---

## 7. Risk Management & Safety

Multiple layers, all enforced before any order reaches the broker:

### Kill Switch
- Tracks daily P&L as percentage of equity
- If loss exceeds `daily_loss_limit_pct` (default 3%), halts ALL trading
- Latching — stays halted until manually reviewed (no auto-reset)

### Position Sizing
- **Fixed weight:** Signal weight × equity × regime multiplier (capped by max position %)
- **ATR risk parity:** Size so that one ATR move = `risk_pct` of portfolio (volatility-normalized)

### Position Limits
- **Max per symbol:** 25% of portfolio (no single stock dominates)
- **Min cash reserve:** 5% always held back
- All limits enforced per-order before submission

### PDT Protection
- Counts same-day buy+sell round-trips over rolling 5 business days
- Warns at 3 day trades, blocks at 4 when equity < $25k (FINRA PDT rule)
- Disabled for accounts ≥ $25k

### Pre-Trade Checks (every signal)
1. Kill switch halted? → reject
2. Position would exceed max %? → reject
3. Cash reserve violated? → reject
4. PDT threshold breached? → warn/reject
5. Notional and price valid? → reject if not

---

## 8. Execution & Broker Interface

### Order Types

| Type | When Used | Behavior |
|------|-----------|----------|
| **Market** (default) | `execution.default_order_type: "market"` | Immediate fill at current price |
| **Limit** | `execution.default_order_type: "limit"` | Fill only at specified price or better; offset from last price by `limit_offset_bps` |
| **Stop** | After fill, if `stop_loss_enabled: true` | Sell triggers when price drops to `fill_price × (1 - stop_loss_pct/100)` |

- **Sells always use market orders** — rotation and tax-loss harvesting need immediate execution
- **Unfilled limit orders** are cancelled after `unfilled_timeout_minutes` (default 30)
- **Retry policy:** Transient failures (network timeout) retry with exponential backoff (configurable)

### Broker Adapter (Alpaca)

The `AlpacaBrokerAdapter` wraps `alpaca-py`. Paper and live use identical code — only the API keys differ. The abstract `BrokerInterface` means a different broker (e.g., Interactive Brokers) could be added without changing strategy or risk code.

---

## 9. Data Storage

### Parquet (time-series prices)

```
data/parquet/SPY.parquet    ← OHLCV for SPY
data/parquet/QQQ.parquet    ← OHLCV for QQQ
...
```

Columnar format — fast reads, 50-80% compression vs CSV. One file per symbol.

### SQLite (operational state)

Single database (`hub_metadata.sqlite`) with tables for:

| Table | Contents |
|-------|----------|
| `trade_signals` | Every signal generated (audit trail) |
| `trade_executions` | Every order submitted (fills, prices, order type, status) |
| `equity_snapshots` | Daily portfolio value snapshots |
| `alerts` | Operational alerts (kill switch, reconciliation, ML staleness) |
| `tax_lots_open` / `tax_lots_closed` | Open and closed tax lots |
| `ml_scores` | ML model predictions per symbol |
| `market_regime` | Regime snapshots (VIX, yield curve, classification) |
| `loss_carryforward` | Multi-year loss carryforward tracking |
| `chat_history` | LLM Q&A history |

All tables partitioned by `account_id` for multi-account support.

---

## 10. Tax Lot Tracking & Reporting

### FIFO Lot Ledger

Every share purchase creates an "open lot" with cost basis and date. When shares are sold, lots are matched in FIFO order (oldest first). Realized gain/loss = (sell price - cost basis) × qty.

### Wash Sale Detection

IRS rule: if you sell at a loss and buy the same (or substantially identical) security within ±30 days, the loss is disallowed.

- **Intra-account:** Detected within each account
- **Cross-account:** Detected across accounts (IRS treats household as single taxpayer)
- **Window:** Configurable via `tax.wash_sale_window_days` (default 30)
- **Replacement map:** `tax.replacement_map` defines equivalents (SPY ↔ IVV)
- Flagged as informational — user reviews before filing

### Tax Exports

| Export | Format | What |
|--------|--------|------|
| **Form 8949** | CSV | Part I (short-term) + Part II (long-term), IRS columns (a)-(h), wash sale code "W" |
| **Schedule D** | CSV | Aggregate proceeds, basis, net by term |
| **1099-B Reconciliation** | Upload + report | Match broker 1099-B against internal lots, flag discrepancies |

### Loss Carryforward

Tracks unused losses across tax years. If losses exceed gains + $3k ordinary income deduction, the remainder carries forward to the next year.

### IRA Treatment

IRA accounts (traditional/Roth) are labeled in reports:
- **Traditional IRA:** Tax-deferred — gains not reported on Form 8949 (informational only)
- **Roth IRA:** Tax-free — qualified gains excluded from taxable totals

---

## 11. Dashboard & API

### Streamlit Dashboard (8 pages)

`http://localhost:8501` — protected by optional password auth.

| Page | Shows |
|------|-------|
| **Home** | Equity curve, balance summary, recent trades, regime status |
| **Portfolio** | Positions, allocation donut, open/closed lots, wash sales |
| **Performance** | Returns vs benchmark, drawdown, rolling Sharpe, Brinson attribution |
| **Trades** | Signal + execution log with filters, ML scores expander |
| **Backtests** | Run history, compare equity curves, trade lists |
| **Alerts** | Timeline of operational alerts with level/date filters |
| **Decisions** | Decision journal — signal rationale + LLM explanations |
| **Taxes** | Filing checklist, Form 8949/Schedule D download, 1099-B upload, loss carryforward, cross-account wash flags |
| **Ask** | LLM-powered Q&A about your portfolio (persistent chat history) |

### REST API

`http://localhost:8000` — optional bearer token auth, OpenAPI docs at `/docs`.

Paginated endpoints return `{items: [...], total: N, limit: M, offset: O}`.

Key endpoints: `/api/portfolio`, `/api/trades/signals`, `/api/trades/executions`, `/api/tax/summary`, `/api/tax/export`, `/api/tax/schedule-d`, `/api/tax/reconcile-1099b`, `/api/backtests`, `/api/alerts`, `/api/ai/ask`, `/api/ml/latest`, `/api/market/regime`.

---

## 12. Backtesting

Walk-forward backtesting with regime parity:

1. Load historical OHLCV from Parquet
2. Skip `warmup_bars` (default 200) for indicator stabilization
3. Day-by-day: generate signals using only data available at that point (no lookahead)
4. Simulate execution with configurable slippage (fixed bps)
5. Record equity curve, trades, return metrics

**Variants:**
- **Walk-forward:** Train/test splits for parameter validation
- **Monte Carlo:** Bootstrap random return paths for confidence intervals
- **Regime-aware:** Build regime from historical VIX/yield data, apply strategy adjustments

```bash
carmel backtest --strategy momentum --start 2023-01-01 --end 2024-01-01 \
    --capital 10000 --slippage-bps 5 --rebalance monthly
```

---

## 13. Optional Features

### ML Momentum Classifier

Train a gradient-boosted classifier on historical returns to predict 5-day forward performance. When `ml.affect_orders: true`, signals with low ML scores are filtered out before execution. SHAP explanations show which features drove the prediction.

### LLM Integration (Ollama)

Local Ollama instance provides natural-language explanations of trades, signal rationale, and portfolio Q&A. Falls back to rule-based templates when LLM is unavailable. Completely optional — all core logic works without it.

### SEC Fundamentals (Edgar)

Optional Piotroski F-Score calculation from SEC 10-K filings. Requires `pip install ".[fundamental]"` and `EDGAR_IDENTITY` env var.

---

## 14. Deployment

### Local (development)

```bash
uv pip install -e ".[dev]"
carmel setup
carmel run
```

### Docker Compose (production)

```bash
docker compose up -d
```

Four services: **trader** (daemon), **dashboard** (Streamlit:8501), **api** (FastAPI:8000), **ollama** (optional LLM). Healthcheck via heartbeat file — auto-restart on stale.

---

## 15. Monitoring & Alerts

- **Heartbeat:** Written every cycle to `logs/hub.heartbeat`; Docker healthcheck restarts if stale >5 min
- **Alerts:** Kill switch, reconciliation discrepancies, ML staleness, PDT warnings → logged to SQLite + sent via webhook/email
- **Weekly digest:** Aggregates recent trades, alerts, P&L into a summary delivered on `digest_cron`
- **Logs:** Rolling file (`logs/hub.log`) with configurable size/backup count

### Webhook providers

`WebhookNotifier` auto-detects the provider from the URL host and formats the payload accordingly:

| Provider | Detection | Payload format |
|----------|-----------|----------------|
| Discord | `discord.com` or `discordapp.com` | `embeds` with color-coded title |
| Slack | `hooks.slack.com` | `text` + `blocks` with emoji |
| Telegram | `api.telegram.org` | `text` + `parse_mode`; requires `NOTIFICATION__TELEGRAM_CHAT_ID` |
| Generic | Anything else | `{timestamp, level, category, message}` |

After configuring, run `carmel notify-test` to verify delivery; the output labels each webhook with the detected provider, for example `webhook (discord):`.

---

## 16. Common Workflows

### First-time setup
```bash
carmel setup          # Creates .env, validates Alpaca
carmel once           # Test one cycle
streamlit run src/dashboard/app.py  # View dashboard
```

### Enable a new strategy
1. Add strategy name to `strategy.enabled` list in `settings.yaml`
2. Backtest: `carmel backtest --strategy mean_reversion --start 2023-01-01 --end 2024-01-01`
3. If results look good: `carmel run`

### Go live
1. `carmel preflight` — validates everything
2. Paper trade for 2-4 weeks
3. Change `.env` to live API keys, set `broker.paper_trading: false`
4. Start with DCA only (safest strategy)
5. Monitor dashboard and alerts daily

### Export taxes
1. Dashboard → Taxes page → select year
2. Download Form 8949 CSV (Part I + Part II)
3. Download Schedule D summary
4. Upload broker 1099-B for reconciliation
5. Follow the filing checklist

---

## 17. Architecture Principles

1. **One-way data flow** — sources → storage → strategies → execution → feedback
2. **No lookahead bias** — strategies only see data up to `as_of` timestamp
3. **All orders gated by risk checks** — nothing reaches the broker unchecked
4. **Paper trading by default** — live trading requires explicit opt-in
5. **Secrets never in code** — `.env` only, gitignored
6. **Fail loudly** — errors are logged and alerted, never silently swallowed
7. **Test everything** — 800+ tests, TDD workflow, 80%+ coverage

---

## 18. File Reference

| Purpose | Location |
|---------|----------|
| Configuration | `config/settings.yaml` |
| Secrets template | `.env.example` |
| Entry point / CLI | `src/automation/runner.py` |
| Trading cycle | `src/automation/workflows.py` |
| Strategies | `src/strategy/{momentum,dca,mean_reversion,ensemble}.py` |
| Risk controls | `src/risk/{kill_switch,pre_trade_checks,pdt_tracker,position_sizing}.py` |
| Broker interface | `src/execution/{broker_interface,alpaca_adapter,order_manager}.py` |
| Tax lots | `src/portfolio/{tax_lots,wash_sales}.py` |
| Tax reports | `src/reporting/{tax_report,reconciliation_1099b}.py` |
| Data storage | `src/data/storage/{parquet_store,sqlite_store}.py` |
| Regime detection | `src/data/regime.py` |
| Dashboard | `src/dashboard/app.py` + `pages/*.py` |
| REST API | `src/api/app.py` + `routes/*.py` |
| ML module | `src/ml/{features,train,infer,explain}.py` |
| LLM integration | `src/ai/{llm_client,llm_explainer,portfolio_qa,rag_index}.py` |
| Tests | `tests/{unit,integration}/` |
| Docker | `Dockerfile`, `docker-compose.yml` |
| Logs | `logs/hub.log`, `logs/hub.heartbeat` |
| Data | `data/parquet/`, `data/cache/`, `data/ml/` |
