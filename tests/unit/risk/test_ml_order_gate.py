"""ML order gate (signal filtering by score)."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Literal

import pytest

from src.models import Signal
from src.risk.ml_order_gate import filter_signals_by_ml_scores


def _sig(sym: str) -> Signal:
    return Signal(
        symbol=sym,
        direction="long",
        weight=0.25,
        confidence=0.8,
        rationale="test",
        timestamp=datetime(2026, 1, 15, 12, 0, tzinfo=UTC),
        strategy_name="TestStrat",
    )


@pytest.mark.parametrize("missing_score_action", ["pass", "block"])
def test_ml_gate_empty_signal_list_returns_empty_counts(
    missing_score_action: Literal["pass", "block"],
) -> None:
    out, passed, n = filter_signals_by_ml_scores(
        [],
        {},
        threshold=0.5,
        missing_score_action=missing_score_action,
    )
    assert out == [] and passed == 0 and n == 0


def test_ml_gate_all_pass_when_scores_above_threshold() -> None:
    sigs = [_sig("SPY"), _sig("QQQ")]
    scores = {"SPY": 0.9, "QQQ": 0.8}
    out, p, n = filter_signals_by_ml_scores(
        sigs,
        scores,
        threshold=0.5,
        missing_score_action="block",
    )
    assert n == 2 and p == 2
    assert len(out) == 2


def test_ml_gate_drops_below_threshold() -> None:
    sigs = [_sig("SPY"), _sig("QQQ")]
    scores = {"SPY": 0.2, "QQQ": 0.9}
    out, p, n = filter_signals_by_ml_scores(
        sigs,
        scores,
        threshold=0.5,
        missing_score_action="block",
    )
    assert n == 2 and p == 1
    assert [s.symbol for s in out] == ["QQQ"]


def test_ml_gate_missing_score_pass_policy() -> None:
    sigs = [_sig("SPY")]
    scores: dict[str, float] = {}
    out, p, _n = filter_signals_by_ml_scores(
        sigs,
        scores,
        threshold=0.5,
        missing_score_action="pass",
    )
    assert p == 1 and len(out) == 1


def test_ml_gate_missing_score_block_policy() -> None:
    sigs = [_sig("SPY")]
    scores = {}
    out, p, _n = filter_signals_by_ml_scores(
        sigs,
        scores,
        threshold=0.5,
        missing_score_action="block",
    )
    assert p == 0 and out == []


def test_ml_gate_nan_score_treated_as_missing() -> None:
    sigs = [_sig("SPY")]
    scores = {"SPY": float("nan")}
    out_pass, p1, _ = filter_signals_by_ml_scores(
        sigs,
        scores,
        threshold=0.5,
        missing_score_action="pass",
    )
    assert p1 == 1 and len(out_pass) == 1
    out_block, p2, _ = filter_signals_by_ml_scores(
        sigs,
        scores,
        threshold=0.5,
        missing_score_action="block",
    )
    assert p2 == 0 and out_block == []


@pytest.mark.parametrize(
    ("score", "threshold", "expect_kept"),
    [
        (0.0, 0.0, True),
        (0.0, 0.01, False),
        (0.5, 0.5, True),
        (0.49, 0.5, False),
        (0.51, 0.5, True),
        (1.0, 1.0, True),
        (0.99, 1.0, False),
        (1.0, 0.99, True),
    ],
)
def test_ml_gate_threshold_boundary_behavior(
    score: float,
    threshold: float,
    expect_kept: bool,
) -> None:
    sigs = [_sig("SPY")]
    scores = {"SPY": score}
    out, p, _ = filter_signals_by_ml_scores(
        sigs,
        scores,
        threshold=threshold,
        missing_score_action="block",
    )
    assert (p == 1) is expect_kept
    assert (len(out) == 1) is expect_kept


def test_ml_gate_mixed_missing_and_threshold() -> None:
    sigs = [_sig("SPY"), _sig("QQQ"), _sig("GLD")]
    scores = {"SPY": 0.9, "QQQ": 0.1}
    out, p, n = filter_signals_by_ml_scores(
        sigs,
        scores,
        threshold=0.5,
        missing_score_action="pass",
    )
    assert n == 3
    assert p == 2
    syms = {s.symbol.upper() for s in out}
    assert syms == {"SPY", "GLD"}
