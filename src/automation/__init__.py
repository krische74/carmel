"""Scheduling, workflows, and health signals.

Import ``create_trading_workflow`` / ``main`` from ``src.automation.runner`` to avoid
heavy imports when the package is loaded (e.g. ``python -m src.automation.runner``).
"""

from src.automation.alerts import (
    Alert,
    AlertLevel,
    check_heartbeat_staleness,
    evaluate_cycle_alerts,
)
from src.automation.notifier import (
    CompositeNotifier,
    EmailNotifier,
    LogOnlyNotifier,
    Notifier,
    WebhookNotifier,
)
from src.automation.scheduler import HubScheduler, cron_trigger_from_expression
from src.automation.workflows import TradingCycleResult, TradingWorkflow

__all__ = [
    "Alert",
    "AlertLevel",
    "CompositeNotifier",
    "EmailNotifier",
    "HubScheduler",
    "LogOnlyNotifier",
    "Notifier",
    "TradingCycleResult",
    "TradingWorkflow",
    "WebhookNotifier",
    "check_heartbeat_staleness",
    "cron_trigger_from_expression",
    "evaluate_cycle_alerts",
]
