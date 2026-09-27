# Carmel Operator Runbook

Quick reference for triaging alerts and running common operational tasks. One screen, scannable. Last updated 2026-05-22.

## 30-second daily check

1. Discord/webhook: most recent `cycle_summary` (every cycle), most recent `liveness` (daily, default 06:00 PDT).
2. `docker compose ps trader` → `Up` and healthy.
3. No `cycle_failed` / `duplicate_scheduler` / `heartbeat_stale` in the last 24h.

If all three pass, walk away.

## Alert response table

| Category | Level | What it means | Action |
|---|---|---|---|
| `cycle_summary` | INFO | A cycle completed (every run) | None — its presence is the signal |
| `liveness` | INFO | Daily "I'm alive" ping | None — absence is the signal |
| `weekly_digest` | INFO | Weekly performance summary | Read for context |
| `order_rejected` | INFO | One or more orders not submitted (sizing cap, dedupe, pending-notional) | Read the reason. Almost always benign — guardrails working |
| `notify_test` | INFO | You manually triggered `carmel notify-test` | Confirms delivery path works |
| `daily_loss_warning` | WARNING | P&L within 80% of `daily_loss_limit_pct` | Watch — `kill_switch` likely follows on continued drawdown |
| `ingest_failure` | WARNING | OHLCV ingest failed for one or more symbols | If persists 2+ cycles, run `carmel ingest --symbols X --years 1` |
| `market_regime` (defensive) | WARNING | Regime shift detected, sizing auto-reduced | None — automatic |
| `ml_model_stale` | WARNING | ML classifier age exceeded threshold | `carmel ml-train` — only matters if ML is on (off by default) |
| `reconciliation` | WARNING | SQLite ↔ broker mismatch detected | Run `carmel reconcile` for details. Known-minor fill-qty drift is safe to ignore during paper soak |
| `kill_switch` | CRITICAL | Daily loss limit breached, trading halted | See **Kill switch tripped** below |
| `heartbeat_stale` | CRITICAL | Daemon stopped writing heartbeat (>300s) | See **Daemon down** below |
| `duplicate_scheduler` | CRITICAL | A second carmel daemon detected | See **Duplicate scheduler** below |
| `cycle_failed` | CRITICAL | Unhandled exception in a cycle | See **Cycle failed** below |
| `alpaca_credentials` | CRITICAL | Bad / missing / rejected Alpaca keys | Check `.env`, then `carmel preflight` |
| `market_regime` (crisis) | CRITICAL | Crisis regime detected, sizing at minimum | None — automatic; review weekly |

## Incident procedures

### Kill switch tripped (`kill_switch`)
1. Pull the most recent `cycle_summary` to see what hit. Most common cause: market down ≥3% in your direction.
2. `KillSwitch` is **in-process state** (src/risk/kill_switch.py:6) — no persistence. Options to clear:
   - **Wait** for the next trading day. P&L denominator resets; the switch self-clears on the first cycle.
   - **Restart** the daemon (`docker compose restart trader`) if you want to resume the same day after manual review. ⚠️ It will re-arm immediately on the next cycle if today's P&L is still below `-daily_loss_limit_pct`.
3. Threshold: `config/settings.yaml` → `risk.daily_loss_limit_pct` (default `0.03`).

### Daemon down (`heartbeat_stale`)
1. `docker compose ps trader` — is it actually running?
2. If stopped: `docker compose logs --tail=200 trader` — find the exit cause. Common: Alpaca 401, Python exception, OOM.
3. `docker compose up -d trader`
4. Confirm with the next `cycle_summary` in Discord.

### Duplicate scheduler (`duplicate_scheduler`)
Something else is writing `logs/hub.heartbeat`. Almost always: a `carmel run` started on the Windows host while the container is also up.
1. Find rogue host python: PowerShell `tasklist | findstr python`
2. Kill it (`taskkill /PID <pid> /F`), or `docker compose stop trader` if the host process is the one you want to keep.
3. After exactly one is running: `docker compose up -d trader` (or leave the host process running).
4. Defense A is now observation-based — it tolerates routine restarts. If you still see a storm, delete the stale file: `rm logs/hub.heartbeat`.

### Cycle failed (`cycle_failed`)
1. `docker compose logs --tail=300 trader` — find the traceback.
2. Transient (network, broker 5xx): next cycle retries. Watch for the next `cycle_summary`.
3. Persistent: capture the traceback, hand to Cursor for a fix branch. Don't restart blindly.

### Reconciliation drift (`reconciliation`)
1. `docker compose exec trader carmel reconcile` — structured details
2. During paper soak, known minor fill-qty mismatches are expected. Investigate if they're growing or appear on new symbols.

## Common manual operations

```powershell
# One-shot cycle (does not start the daemon)
docker compose exec trader carmel once

# Validate config + broker connectivity
docker compose exec trader carmel preflight

# Send a test alert through your notification channels
docker compose exec trader carmel notify-test

# Backfill missing market data
docker compose exec trader carmel ingest --symbols QQQ,VOO,VXUS,BND --years 1

# Compare SQLite execution log to broker orders
docker compose exec trader carmel reconcile

# Print non-secret config (sanity check what's actually loaded)
docker compose exec trader carmel config
```

### Safe rebuild after code change
The trader runs from the Docker image — **source edits do nothing until you rebuild**:
```powershell
docker compose build trader
docker compose up -d trader
```
Verify the fix actually landed in the running container:
```powershell
docker compose exec trader grep <pattern> /app/src/<path>.py
```

### Reseeding the paper account
1. Alpaca dashboard: delete old paper account, create new, regenerate keys.
2. Update `ALPACA_API_KEY` / `ALPACA_SECRET_KEY` in `.env`.
3. Back up + wipe local state: rename `data/cache/hub_metadata.sqlite` to `.backup-YYYYMMDD`.
4. `docker compose restart trader` and confirm with the next `cycle_summary`.

## Escalation rules of thumb

- One CRITICAL alert: triage immediately, don't wait for a pattern.
- Two WARNING alerts of the same category in 24h: dig in.
- Anything new (a category not in the table above): treat as CRITICAL until you've read the source — update this runbook with the new row.
