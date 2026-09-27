"""Typed configuration for Carmel.

Loads settings from config/settings.yaml with environment variable overrides.
Secrets (API keys) come exclusively from .env / environment variables.
"""

from __future__ import annotations

import logging
import logging.config
import os
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator
from pydantic_settings import BaseSettings

PROJECT_ROOT = Path(__file__).resolve().parent.parent

logger = logging.getLogger(__name__)


def _load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML file and return its contents as a dict."""
    resolved = path if path.is_absolute() else PROJECT_ROOT / path
    if not resolved.exists():
        return {}
    with open(resolved) as f:
        return yaml.safe_load(f) or {}


class AppConfig(BaseModel):
    """Top-level application metadata."""

    name: str = "Carmel"
    version: str = "0.1.0"
    environment: str = "development"


class DCATarget(BaseModel):
    """A single DCA allocation target."""

    symbol: str
    weight: float


class DataConfig(BaseModel):
    """Data source and storage configuration."""

    default_provider: str = "yfinance"
    cache_dir: str = "data/cache"
    parquet_dir: str = "data/parquet"
    universe: list[str] = Field(default_factory=lambda: ["SPY", "QQQ", "TLT", "GLD"])
    dca_targets: list[DCATarget] = Field(default_factory=list)
    ohlcv_outlier_zscore_threshold: float | None = Field(
        default=4.0,
        description=(
            "If set, log a warning (not an ingest error) when close is extreme vs prior "
            "mean. None disables."
        ),
    )
    ohlcv_min_prior_for_zscore: int = Field(
        default=20,
        ge=5,
        le=200,
        description="Minimum prior closes before outlier z-score warning applies.",
    )
    ohlcv_max_abs_daily_return: float = Field(
        default=0.5,
        ge=0.05,
        le=2.0,
        description="Hard reject if any daily |close[t]/close[t-1]-1| exceeds this (feed errors).",
    )
    validation_exempt_symbols: list[str] = Field(
        default_factory=lambda: ["^VIX"],
        description=(
            "Symbols that skip the max_abs_daily_return check during OHLCV validation. "
            "Volatility indices like ^VIX routinely make 50%+ daily moves that are not "
            "feed errors. Other hard checks (NaN, Inf, non-positive prices, OHLC consistency) "
            "still apply."
        ),
    )
    initial_backfill_years: int = Field(
        default=3,
        ge=0,
        le=20,
        description="Years of history to fetch on first ingest of a symbol (0 to disable).",
    )


class AccountConfig(BaseModel):
    """One Alpaca account: credentials referenced by env var names only (no secrets in YAML)."""

    name: str = Field(..., description="Human label: Main, IRA, Joint")
    account_type: Literal["brokerage", "ira_traditional", "ira_roth", "joint"] = "brokerage"
    api_key_env: str = Field(
        ...,
        description="Env var for API key, e.g. ALPACA_ACCOUNT_MAIN_KEY",
    )
    api_secret_env: str = Field(
        ...,
        description="Env var for secret, e.g. ALPACA_ACCOUNT_MAIN_SECRET",
    )


class BrokerConfig(BaseModel):
    """Broker connection settings."""

    provider: str = "alpaca"
    paper_trading: bool = True
    accounts: list[AccountConfig] = Field(default_factory=list)
    default_account: str = Field(
        default="",
        description="Primary account name when multiple are configured (logging / digest).",
    )


class MomentumConfig(BaseModel):
    """Momentum rotation strategy parameters."""

    lookback_months: list[int] = Field(default_factory=lambda: [1, 3, 6, 12])
    sma_filter_period: int = 200
    cash_symbol: str = "SHV"
    rebalance_frequency: str = "monthly"
    adx_filter_period: int = 14
    adx_threshold: float = 25.0
    # Regime-aware ADX (optional absolute overrides; None → use deltas below vs base).
    adx_threshold_risk_on: float | None = None
    adx_threshold_cautious: float | None = None
    adx_threshold_defensive: float | None = None
    adx_threshold_crisis: float | None = None
    # When overrides are None: risk_on uses base - risk_on_delta; defensive/crisis add deltas.
    adx_threshold_risk_on_delta: float = Field(default=2.0, ge=0.0)
    adx_threshold_defensive_delta: float = Field(default=3.0, ge=0.0)
    adx_threshold_crisis_delta: float = Field(default=5.0, ge=0.0)


class DCAConfig(BaseModel):
    """Dollar-cost averaging parameters."""

    frequency: str = "weekly"
    amount: float = 100.0
    percent_of_equity: float | None = Field(
        default=None,
        ge=0.0,
        le=0.5,
        description=(
            "When set, per-cycle DCA contribution = equity * percent_of_equity. "
            "When None, falls back to amount."
        ),
    )
    regime_amount_risk_on: float = Field(default=1.0, ge=0.0, le=2.0)
    regime_amount_cautious: float = Field(default=0.75, ge=0.0, le=2.0)
    regime_amount_defensive: float = Field(default=0.5, ge=0.0, le=2.0)
    regime_amount_crisis: float = Field(default=0.25, ge=0.0, le=2.0)


class MeanReversionConfig(BaseModel):
    """Mean reversion (Bollinger + RSI) parameters."""

    bb_period: int = 20
    bb_num_std: float = 2.0
    rsi_period: int = 14
    rsi_oversold: float = 30.0
    rsi_overbought: float = 70.0
    sma_trend_period: int = 200
    max_positions: int = 4
    regime_max_positions_risk_on: int = Field(default=4, ge=0, le=10)
    regime_max_positions_cautious: int = Field(default=3, ge=0, le=10)
    regime_max_positions_defensive: int = Field(default=2, ge=0, le=10)
    regime_max_positions_crisis: int = Field(default=1, ge=0, le=10)
    universe: list[str] = Field(default_factory=lambda: ["SPY", "QQQ", "TLT", "GLD"])


class EnsembleConfig(BaseModel):
    """Weighted merge of tactical signals from multiple strategies before execution."""

    enabled: bool = Field(default=False, description="Enable ensemble signal merging")
    strategy_weights: dict[str, float] = Field(
        default_factory=lambda: {
            "MomentumRotationStrategy": 0.5,
            "MeanReversionStrategy": 0.5,
        },
        description="Per-strategy weight for blending. Keys are strategy_name values.",
    )
    cash_signal_penalty: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Multiply tactical blended weights by this when momentum signals cash.",
    )
    min_blended_weight: float = Field(
        default=0.05,
        ge=0.0,
        le=1.0,
        description="Drop merged signals with blended weight below this.",
    )


class StrategyConfig(BaseModel):
    """Strategy-level configuration."""

    enabled: list[str] = Field(
        default_factory=lambda: ["momentum", "dca"],
        min_length=1,
        description="Strategies for live/paper cycles: momentum, dca, mean_reversion (case-insensitive).",
    )
    momentum: MomentumConfig = Field(default_factory=MomentumConfig)
    dca: DCAConfig = Field(default_factory=DCAConfig)
    mean_reversion: MeanReversionConfig = Field(default_factory=MeanReversionConfig)
    ensemble: EnsembleConfig = Field(default_factory=EnsembleConfig)

    @field_validator("enabled", mode="before")
    @classmethod
    def _normalize_enabled(cls, v: Any) -> Any:
        if isinstance(v, list):
            cleaned = [str(x).strip().lower() for x in v if str(x).strip()]
            if not cleaned:
                raise ValueError(
                    "strategy.enabled must contain at least one strategy name "
                    "(momentum, dca, mean_reversion).",
                )
            return cleaned
        return v


class RiskConfig(BaseModel):
    """Risk management parameters."""

    max_position_pct: float = 0.25
    max_portfolio_risk_pct: float = 0.02
    daily_loss_limit_pct: float = 0.03
    min_cash_reserve_pct: float = 0.05
    pdt_protection: bool = True
    pdt_threshold: int = Field(
        default=4,
        ge=1,
        le=20,
        description="Day trades in 5 business days that trigger PDT rules (FINRA default 4).",
    )
    pdt_equity_floor: float = Field(
        default=25_000.0,
        gt=0.0,
        description="Equity above this USD amount bypasses PDT day-trade count limits.",
    )
    sizing_method: Literal["fixed_weight", "atr_risk_parity"] = "fixed_weight"
    atr_risk_pct: float = Field(default=0.01, gt=0.0, le=1.0)
    atr_period: int = Field(default=14, ge=2)
    min_order_notional_usd: float = Field(
        default=5.0,
        description=(
            "Minimum dollar notional for a live buy order. Orders sized below this "
            "(including post-delta top-ups) are skipped rather than submitted — avoids "
            "sub-$5 fractional order noise. Does not apply to full-position sells "
            "(close_position)."
        ),
    )


class CashSweepConfig(BaseModel):
    """Idle-cash sweep into a short-term T-bill ETF. Cash management, not a strategy."""

    enabled: bool = True
    symbol: str = Field(
        default="BIL",
        description="Sweep vehicle. Must NOT be a momentum-universe symbol (see validator).",
    )
    buffer_pct: float = Field(
        default=0.02,
        gt=0.0,
        le=0.2,
        description=(
            "Cash buffer above risk.min_cash_reserve_pct kept unswept to fund routine "
            "DCA legs between sweeps."
        ),
    )
    min_trade_usd: float = Field(
        default=50.0,
        gt=0.0,
        description=(
            "Minimum sweep buy/sell size, to avoid daily dribble. Overridden downward "
            "only when selling to restore the hard reserve."
        ),
    )


class ExecutionConfig(BaseModel):
    """Order types, limit offsets, stop-loss, and stale limit handling."""

    default_order_type: Literal["market", "limit"] = "market"
    limit_offset_bps: float = Field(
        default=10.0,
        ge=0.0,
        le=100.0,
        description="Basis points above last price for limit buy orders (e.g. 10 = 0.1% above).",
    )
    stop_loss_enabled: bool = False
    stop_loss_pct: float = Field(
        default=5.0,
        ge=0.5,
        le=50.0,
        description="Stop-loss trigger as percent below fill price.",
    )
    unfilled_timeout_minutes: int = Field(
        default=30,
        ge=1,
        le=480,
        description="Cancel unfilled limit orders older than this many minutes.",
    )


class ExecutionRetryConfig(BaseModel):
    """Broker order submission retries on transient connection failures."""

    max_attempts: int = Field(default=2, ge=1, le=5)
    backoff_base_seconds: float = Field(default=0.5, ge=0.1, le=5.0)
    use_jitter: bool = True


class LogRotationConfig(BaseModel):
    """Size-based log file rotation (applied to ``logging.yaml`` file handler)."""

    max_bytes: int = Field(default=10_485_760, ge=1_048_576, le=1_073_741_824)
    backup_count: int = Field(default=5, ge=1, le=20)


class SchedulerConfig(BaseModel):
    """Job scheduling configuration."""

    data_ingestion_cron: str = "0 17 * * 1-5"
    signal_generation_cron: str = "30 17 * * 1-5"
    rebalance_check_cron: str = "0 18 * * 5"
    heartbeat_interval_seconds: int = 60
    reconcile_cron: str = ""  # empty = disabled; e.g. "0 19 * * 5" for Friday 7 PM
    digest_cron: str = ""  # empty = disabled; e.g. "0 9 * * 1" for Monday 9 AM weekly digest
    timezone: str = Field(
        default="UTC",
        description="IANA timezone for cron scheduling (e.g. US/Eastern, Europe/London).",
    )
    ml_retraining_cron: str = ""  # empty = disabled; e.g. "0 2 * * 0" for Sunday 2 AM
    liveness_cron: str = ""  # empty = disabled; e.g. "0 13 * * *" for daily 13:00 in scheduler TZ
    execution_retry: ExecutionRetryConfig = Field(default_factory=ExecutionRetryConfig)
    log_rotation: LogRotationConfig = Field(default_factory=LogRotationConfig)
    reconciliation_warning_threshold: int = Field(default=1, ge=0)
    reconciliation_critical_threshold: int = Field(default=5, ge=1)

    @field_validator("timezone")
    @classmethod
    def _validate_timezone(cls, v: Any) -> str:
        """Reject unknown IANA zones at load time; fall back to UTC with a warning."""
        from zoneinfo import available_timezones

        raw = str(v).strip() if v is not None else ""
        name = raw or "UTC"
        if name not in available_timezones():
            logging.getLogger(__name__).warning(
                "Invalid scheduler.timezone %r; falling back to UTC",
                v,
            )
            return "UTC"
        return name


class DashboardConfig(BaseModel):
    """Dashboard display configuration."""

    port: int = Field(default=8501, ge=1024, le=65535)
    refresh_interval_seconds: int = Field(default=30, ge=5, le=300)
    auth_enabled: bool = False
    auth_password: str = Field(
        default="",
        description="Dashboard login password; prefer DASHBOARD_PASSWORD in .env. Empty disables gate even if auth_enabled.",
    )


_VALID_REGIME_KEYS = frozenset({"risk_on", "cautious", "defensive", "crisis"})


class RegimeConfig(BaseModel):
    """Market regime detection and adjustment settings."""

    enabled: bool = True
    vix_symbol: str = "^VIX"
    fred_series: list[str] = Field(default_factory=lambda: ["DGS10", "DGS2"])
    yield_curve_flat_threshold: float = 0.5
    yield_curve_inverted_threshold: float = -0.5
    vix_low: float = 15.0
    vix_normal: float = 20.0
    vix_elevated: float = 30.0
    sizing_adjustment: dict[str, float] = Field(
        default_factory=lambda: {
            "risk_on": 1.0,
            "cautious": 0.75,
            "defensive": 0.5,
            "crisis": 0.25,
        },
    )

    @field_validator("sizing_adjustment")
    @classmethod
    def _validate_sizing_keys(cls, v: dict[str, float]) -> dict[str, float]:
        invalid = set(v.keys()) - _VALID_REGIME_KEYS
        if invalid:
            raise ValueError(
                f"Invalid sizing_adjustment key(s): {sorted(invalid)}. "
                f"Allowed: {sorted(_VALID_REGIME_KEYS)}"
            )
        return v

    @model_validator(mode="after")
    def _validate_vix_ordering(self) -> RegimeConfig:
        if not (self.vix_low < self.vix_normal < self.vix_elevated):
            raise ValueError(
                f"VIX thresholds must satisfy vix_low < vix_normal < vix_elevated, "
                f"got {self.vix_low} / {self.vix_normal} / {self.vix_elevated}"
            )
        return self


class FundamentalConfig(BaseModel):
    """SEC/Edgar fundamental data settings."""

    enabled: bool = False
    refresh_interval_days: int = Field(default=30, ge=1)
    max_symbols_per_cycle: int = Field(default=12, ge=1, le=500)
    skip_symbols: list[str] = Field(
        default_factory=lambda: [
            "SPY",
            "QQQ",
            "IWM",
            "DIA",
            "VTI",
            "VOO",
            "VXUS",
            "BND",
            "AGG",
            "TLT",
            "GLD",
            "SLV",
            "SHV",
            "IVV",
            "QQQM",
            "VGLT",
        ],
        description="Uppercase tickers to skip (ETFs / funds with no 10-K fundamentals).",
    )


class BacktestConfigSection(BaseModel):
    """CLI / engine backtest defaults (non-strategy)."""

    warmup_bars: int = Field(
        default=0,
        ge=0,
        le=500,
        description="Skip first N union-calendar trading days before equity tracking and rebalances.",
    )
    min_coverage_ratio: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description=(
            "Minimum fraction of expected trading days that must fall in the requested window. "
            "Below this, BacktestDataMissingError is raised. Set 0.0 to disable."
        ),
    )
    cash_yield_annual_pct: float = Field(
        default=4.0,
        ge=0.0,
        description=(
            "Fallback annualized yield on uninvested cash when the historical "
            "short-rate series is missing (logged, never silent)."
        ),
    )
    cash_yield_series_id: str = Field(
        default="DTB3",
        description=(
            "FRED series id for backtest cash yield and excess Sharpe (e.g. DTB3). "
            "Empty string uses cash_yield_annual_pct only."
        ),
    )
    benchmark_symbol: str = Field(
        default="SPY",
        description="Default buy-and-hold benchmark ticker for CLI backtests.",
    )


class TaxConfig(BaseModel):
    """Tax-loss harvesting and lot selection settings."""

    harvest_enabled: bool = False
    harvest_threshold_pct: float = Field(default=0.05, ge=0.0, le=1.0)
    harvest_min_loss_dollars: float = Field(default=50.0, ge=0.0)
    lot_selection_method: Literal["fifo", "hifo"] = "fifo"
    replacement_map: dict[str, str] = Field(default_factory=dict)
    wash_sale_window_days: int = Field(
        default=30,
        ge=1,
        le=90,
        description="Calendar-day ±window for wash-sale detection (intra- and cross-account).",
    )


class LLMConfig(BaseModel):
    """Local LLM settings (Ollama)."""

    enabled: bool = False
    base_url: str = "http://localhost:11434"
    model: str = "llama3.2"
    timeout_seconds: int = 30
    max_tokens: int = 512
    temperature: float = Field(default=0.3, ge=0.0, le=2.0)


class MLConfig(BaseModel):
    """Optional experimental ML overlay (offline train; scores logged unless gated later)."""

    enabled: bool = False
    affect_orders: bool = Field(
        default=False,
        description="When False (default), ML scores are diagnostic only and never change execution.",
    )
    score_threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Minimum ML score (P(class=1)) required for a signal to pass the gate.",
    )
    missing_score_action: Literal["pass", "block"] = Field(
        default="pass",
        description="When no ML score exists for a symbol: pass keeps the signal; block drops it.",
    )
    model_path: str = "data/ml/momentum_model.joblib"
    label_horizon_days: int = Field(default=5, ge=1, le=60)
    min_train_rows: int = Field(default=80, ge=20)
    model_version: str = "1"
    retraining_lookback_days: int = Field(default=365, ge=30, le=1825)
    max_model_age_days: int = Field(default=30, ge=1, le=365)


class RAGConfig(BaseModel):
    """Document RAG over whitelisted repo paths (Ollama embeddings + SQLite chunks)."""

    enabled: bool = False
    index_paths: list[str] = Field(
        default_factory=lambda: ["docs", "AGENTS.md"],
        description="Paths relative to project root; only these subtrees/files are indexed.",
    )
    top_k: int = Field(default=5, ge=1, le=20)
    embedding_model: str = ""


class NotificationConfig(BaseModel):
    """Webhook notification settings (URL may point to Telegram, Discord, Slack, etc.)."""

    webhook_url: str = ""
    telegram_chat_id: str = Field(
        default="",
        description="Telegram chat ID (required when webhook_url points to api.telegram.org).",
    )
    enabled: bool = False
    timeout_seconds: int = 10
    max_retries: int = 2
    backoff_seconds: float = 1.0
    email_enabled: bool = False
    smtp_host: str = ""
    smtp_port: int = 587
    smtp_use_tls: bool = True
    email_from: str = ""
    email_to: list[str] = Field(default_factory=list)


class Settings(BaseSettings):
    """Root settings object. Loads from YAML then overlays env vars.

    YAML provides non-secret defaults. Environment variables override
    any YAML value and are the ONLY source for secrets (API keys).
    """

    app: AppConfig = Field(default_factory=AppConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    broker: BrokerConfig = Field(default_factory=BrokerConfig)
    strategy: StrategyConfig = Field(default_factory=StrategyConfig)
    risk: RiskConfig = Field(default_factory=RiskConfig)
    execution: ExecutionConfig = Field(default_factory=ExecutionConfig)
    cash_sweep: CashSweepConfig = Field(default_factory=CashSweepConfig)
    scheduler: SchedulerConfig = Field(default_factory=SchedulerConfig)
    dashboard: DashboardConfig = Field(default_factory=DashboardConfig)
    notification: NotificationConfig = Field(default_factory=NotificationConfig)
    tax: TaxConfig = Field(default_factory=TaxConfig)
    backtest: BacktestConfigSection = Field(default_factory=BacktestConfigSection)
    regime: RegimeConfig = Field(default_factory=RegimeConfig)
    fundamental: FundamentalConfig = Field(default_factory=FundamentalConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    rag: RAGConfig = Field(default_factory=RAGConfig)
    ml: MLConfig = Field(default_factory=MLConfig)

    # Secrets — env vars only, never in YAML
    alpaca_api_key: str = ""
    alpaca_secret_key: str = ""
    alpaca_paper: bool = True
    fred_api_key: str = ""
    smtp_username: str = ""
    smtp_password: str = ""
    log_level: str = "INFO"
    api_token: str = ""
    dashboard_password: str = Field(
        default="",
        description="Dashboard password from env DASHBOARD_PASSWORD (overrides dashboard.auth_password).",
    )

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "env_nested_delimiter": "__",
        # Allow NOTIFICATION__* env vars to merge with YAML ``notification:`` block.
        "nested_model_default_partial_update": True,
        "extra": "ignore",
    }

    def __init__(self, _yaml_path: Path | None = None, _env_file: str | None = "", **kwargs: Any):
        yaml_data = {}
        if _yaml_path is not None:
            yaml_data = _load_yaml(_yaml_path)
        else:
            yaml_data = _load_yaml(Path("config/settings.yaml"))

        if _env_file == "":
            pass  # use default from model_config
        elif _env_file is None:
            kwargs["_env_file"] = None

        merged = {**yaml_data, **kwargs}
        super().__init__(**merged)

        if self.alpaca_paper is False:
            self.broker.paper_trading = False

        self._warn_on_yaml_secrets()

    @model_validator(mode="after")
    def _validate_cash_sweep_symbol(self) -> Settings:
        """Pin the rotation-safety property: the sweep symbol must not be a momentum-universe symbol.

        Momentum rotation sells every momentum-universe position that isn't the current
        target, without checking ownership. A swept position on such a symbol would be
        liquidated every cycle, so reject the config outright.
        """
        if self.cash_sweep.enabled:
            sym = self.cash_sweep.symbol.strip().upper()
            uni = momentum_universe_symbols(self)
            if sym in uni:
                msg = (
                    f"cash_sweep.symbol {sym!r} is in the momentum universe {sorted(uni)}; "
                    "the rotation step would liquidate a swept position every cycle. Choose a "
                    "symbol outside data.universe and strategy.momentum.cash_symbol (e.g. BIL)."
                )
                raise ValueError(msg)
        return self

    def _warn_on_yaml_secrets(self) -> None:
        """Log warnings when known credentials are present in YAML defaults."""
        url = (self.notification.webhook_url or "").strip()
        if not url:
            return
        if _notification_webhook_url_set_via_env():
            return
        logger.warning(
            "notification.webhook_url is set in YAML — recommended to move to "
            ".env as NOTIFICATION__WEBHOOK_URL (webhook URLs are credentials).",
        )


def _notification_webhook_url_set_via_env() -> bool:
    """True if NOTIFICATION__WEBHOOK_URL is set in OS env or project-root ``.env`` (non-empty).

    Pydantic-settings reads ``.env`` into the model without always mutating ``os.environ``,
    so we mirror that check for the YAML-secret warning.
    """
    direct = (os.environ.get("NOTIFICATION__WEBHOOK_URL") or "").strip()
    if direct:
        return True
    env_path = PROJECT_ROOT / ".env"
    if not env_path.is_file():
        return False
    try:
        text = env_path.read_text(encoding="utf-8")
    except OSError:
        return False
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, _, val = line.partition("=")
        if key.strip().upper() == "NOTIFICATION__WEBHOOK_URL" and val.strip():
            return True
    return False


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached singleton Settings instance."""
    return Settings()


PUBLIC_SETTINGS_EXCLUDE = frozenset(
    {
        "alpaca_api_key",
        "alpaca_secret_key",
        "fred_api_key",
        "smtp_password",
        "smtp_username",
        "api_token",
        "dashboard_password",
    },
)


def public_settings_dict(settings: Settings) -> dict[str, Any]:
    """Return settings safe for JSON logging and the REST ``/api/config`` endpoint."""
    return settings.model_dump(exclude=PUBLIC_SETTINGS_EXCLUDE)


def momentum_universe_symbols(settings: Settings) -> set[str]:
    """Normalized symbols the momentum rotation tracks: ``data.universe`` + its cash ETF.

    This is the set the rotation step will sell out of when they are not the current
    target, so the cash-sweep vehicle must stay outside it (enforced in ``Settings``).
    """
    uni = {str(s).strip().upper() for s in settings.data.universe if str(s).strip()}
    cash = str(settings.strategy.momentum.cash_symbol).strip().upper()
    if cash:
        uni.add(cash)
    return uni


def parquet_dir(settings: Settings) -> Path:
    """Parquet OHLCV root: absolute path, or under project root when relative."""
    p = Path(settings.data.parquet_dir)
    return p if p.is_absolute() else PROJECT_ROOT / p


def hub_sqlite_path(settings: Settings) -> Path:
    """Hub metadata SQLite (signals, executions, snapshots, tax lots)."""
    cache = Path(settings.data.cache_dir)
    base = cache if cache.is_absolute() else PROJECT_ROOT / cache
    return base / "hub_metadata.sqlite"


def ml_model_path(settings: Settings) -> Path:
    """Resolved path to persisted sklearn model bundle (joblib)."""
    p = Path(settings.ml.model_path)
    return p if p.is_absolute() else PROJECT_ROOT / p


def parse_as_of_iso(raw: str | None) -> datetime | None:
    """Parse optional ISO-8601 simulation time (CLI ``--as-of`` / API cycle body).

    Naive datetimes are treated as UTC. Returns ``None`` when ``raw`` is empty.
    """
    if raw is None or not str(raw).strip():
        return None
    s = str(raw).strip().replace("Z", "+00:00")
    out = datetime.fromisoformat(s)
    if out.tzinfo is None:
        out = out.replace(tzinfo=UTC)
    return out


def setup_logging(
    config_path: Path | None = None,
    *,
    log_rotation: LogRotationConfig | None = None,
) -> None:
    """Configure logging from YAML config file.

    When ``log_rotation`` is set, overrides ``maxBytes`` / ``backupCount`` on the
    ``file`` handler before applying :func:`logging.config.dictConfig`.

    If the environment variable ``CARMEL_LOG_DIR`` is set to an absolute or
    project-relative directory, the rotating file handler writes
    ``carmel.log`` there instead of under ``<project>/logs/``. Pytest sets this
    automatically in ``tests/conftest.py`` so the suite does not append to the
    operator's production log file.
    """
    path = config_path or PROJECT_ROOT / "config" / "logging.yaml"
    if path.exists():
        config = _load_yaml(path)

        override = (os.environ.get("CARMEL_LOG_DIR") or "").strip()
        if override:
            log_dir = Path(override).expanduser()
            if not log_dir.is_absolute():
                log_dir = PROJECT_ROOT / log_dir
        else:
            log_dir = PROJECT_ROOT / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)

        file_handler = config.get("handlers", {}).get("file", {})
        if "filename" in file_handler:
            file_handler["filename"] = str(log_dir / Path(file_handler["filename"]).name)
        if log_rotation is not None:
            file_handler["maxBytes"] = int(log_rotation.max_bytes)
            file_handler["backupCount"] = int(log_rotation.backup_count)

        logging.config.dictConfig(config)
    else:
        logging.basicConfig(level=logging.INFO)
