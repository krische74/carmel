"""Tier 30: CLI runs a cycle per configured broker account."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock

from src.automation import runner as runner_mod
from src.config import Settings

if TYPE_CHECKING:
    from pathlib import Path

    import pytest


def test_runner_iterates_accounts_on_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``carmel once`` calls ``run_cycle`` once per entry from ``build_brokers``."""
    br_a = MagicMock()
    br_b = MagicMock()
    monkeypatch.setattr(
        runner_mod,
        "build_brokers",
        lambda _s: {"Main": br_a, "IRA": br_b},
    )

    workflows: list[MagicMock] = []
    brokers_passed: list[object] = []

    def _make_wf(*_a, broker=None, **_k):
        wf = MagicMock()
        workflows.append(wf)
        brokers_passed.append(broker)
        return wf

    monkeypatch.setattr(runner_mod, "create_trading_workflow", _make_wf)
    monkeypatch.setattr(runner_mod, "SQLiteStore", MagicMock(return_value=MagicMock()))
    monkeypatch.setattr(runner_mod, "LotLedger", MagicMock(return_value=MagicMock()))
    monkeypatch.setattr(runner_mod, "setup_logging", lambda *a, **k: None)
    yml = tmp_path / "settings.yaml"
    yml.write_text(
        """
app: {name: "X", version: "0.1.0", environment: "development"}
data: {universe: ["SPY"]}
risk: {max_position_pct: 0.25}
""",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        runner_mod,
        "get_settings",
        lambda: Settings(
            _yaml_path=yml,
            _env_file=None,
            alpaca_api_key="k",
            alpaca_secret_key="s",
        ),
    )

    code = runner_mod.main(["once"])
    assert code == 0
    assert len(workflows) == 2
    assert brokers_passed == [br_a, br_b]
    workflows[0].run_cycle.assert_called_once_with(as_of=None)
    workflows[1].run_cycle.assert_called_once_with(as_of=None)


def test_runner_once_single_entry_multi_account_dict_one_cycle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One broker key (e.g. YAML with a single ``accounts:`` entry) still runs one cycle.

    Distinct from legacy fallback: ``build_brokers`` may return a named key like ``SoloIRA``
    instead of ``default``/``default_account``, but the ``once`` loop behaves the same
    for a one-element map.
    """
    br_only = MagicMock()
    monkeypatch.setattr(
        runner_mod,
        "build_brokers",
        lambda _s: {"SoloIRA": br_only},
    )

    workflows: list[MagicMock] = []
    brokers_passed: list[object] = []

    def _make_wf(*_a, broker=None, **_k):
        wf = MagicMock()
        workflows.append(wf)
        brokers_passed.append(broker)
        return wf

    monkeypatch.setattr(runner_mod, "create_trading_workflow", _make_wf)
    monkeypatch.setattr(runner_mod, "SQLiteStore", MagicMock(return_value=MagicMock()))
    monkeypatch.setattr(runner_mod, "LotLedger", MagicMock(return_value=MagicMock()))
    monkeypatch.setattr(runner_mod, "setup_logging", lambda *a, **k: None)
    yml = tmp_path / "settings.yaml"
    yml.write_text(
        """
app: {name: "X", version: "0.1.0", environment: "development"}
data: {universe: ["SPY"]}
risk: {max_position_pct: 0.25}
""",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        runner_mod,
        "get_settings",
        lambda: Settings(
            _yaml_path=yml,
            _env_file=None,
            alpaca_api_key="k",
            alpaca_secret_key="s",
        ),
    )

    code = runner_mod.main(["once"])
    assert code == 0
    assert len(workflows) == 1
    assert brokers_passed == [br_only]
    workflows[0].run_cycle.assert_called_once_with(as_of=None)
