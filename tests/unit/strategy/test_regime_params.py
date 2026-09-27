"""Regime-derived ADX threshold helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from src.data.regime import OverallRegime
from src.strategy.regime_params import (
    effective_adx_threshold,
    effective_dca_amount,
    effective_mean_reversion_max_positions,
)

if TYPE_CHECKING:
    from src.config import Settings


def test_effective_adx_none_uses_base(settings: Settings) -> None:
    assert effective_adx_threshold(settings, None) == float(
        settings.strategy.momentum.adx_threshold
    )


def test_effective_adx_risk_on_lower_than_base(settings: Settings) -> None:
    base = float(settings.strategy.momentum.adx_threshold)
    v = effective_adx_threshold(settings, OverallRegime.RISK_ON)
    assert v < base
    assert v >= 5.0


def test_effective_adx_cautious_equals_base(settings: Settings) -> None:
    base = float(settings.strategy.momentum.adx_threshold)
    assert effective_adx_threshold(settings, OverallRegime.CAUTIOUS) == base


def test_effective_adx_defensive_higher_than_base(settings: Settings) -> None:
    base = float(settings.strategy.momentum.adx_threshold)
    assert effective_adx_threshold(settings, OverallRegime.DEFENSIVE) > base


def test_effective_adx_crisis_highest(settings: Settings) -> None:
    base = float(settings.strategy.momentum.adx_threshold)
    d = effective_adx_threshold(settings, OverallRegime.DEFENSIVE)
    c = effective_adx_threshold(settings, OverallRegime.CRISIS)
    assert c > base
    assert c >= d


def test_effective_adx_override_risk_on(settings: Settings) -> None:
    settings.strategy.momentum.adx_threshold_risk_on = 10.0
    assert effective_adx_threshold(settings, OverallRegime.RISK_ON) == 10.0


def test_effective_dca_amount_no_regime(settings: Settings) -> None:
    assert effective_dca_amount(settings, None) == float(settings.strategy.dca.amount)


@pytest.mark.parametrize(
    ("regime", "expected_mult"),
    [
        (OverallRegime.RISK_ON, 1.0),
        (OverallRegime.CAUTIOUS, 0.75),
        (OverallRegime.DEFENSIVE, 0.5),
        (OverallRegime.CRISIS, 0.25),
    ],
)
def test_effective_dca_amount_all_regimes(
    settings: Settings,
    regime: OverallRegime,
    expected_mult: float,
) -> None:
    base = float(settings.strategy.dca.amount)
    assert effective_dca_amount(settings, regime) == pytest.approx(base * expected_mult)


def test_effective_dca_amount_zero_crisis_multiplier(settings: Settings) -> None:
    dca = settings.strategy.dca.model_copy(update={"regime_amount_crisis": 0.0})
    strat_cfg = settings.strategy.model_copy(update={"dca": dca})
    s = settings.model_copy(update={"strategy": strat_cfg})
    assert effective_dca_amount(s, OverallRegime.CRISIS) == 0.0


def test_effective_dca_amount_uses_percent_of_equity_when_set(settings: Settings) -> None:
    dca = settings.strategy.dca.model_copy(
        update={"amount": 100.0, "percent_of_equity": 0.005},
    )
    strat_cfg = settings.strategy.model_copy(update={"dca": dca})
    s = settings.model_copy(update={"strategy": strat_cfg})
    assert effective_dca_amount(s, None, equity=100_000.0) == pytest.approx(500.0)


def test_effective_dca_amount_percent_of_equity_cautious_regime(settings: Settings) -> None:
    dca = settings.strategy.dca.model_copy(
        update={
            "amount": 100.0,
            "percent_of_equity": 0.005,
            "regime_amount_cautious": 0.75,
        },
    )
    strat_cfg = settings.strategy.model_copy(update={"dca": dca})
    s = settings.model_copy(update={"strategy": strat_cfg})
    assert effective_dca_amount(s, OverallRegime.CAUTIOUS, equity=100_000.0) == pytest.approx(
        375.0
    )


def test_effective_max_positions_no_regime(settings: Settings) -> None:
    assert effective_mean_reversion_max_positions(settings, None) == int(
        settings.strategy.mean_reversion.max_positions,
    )


@pytest.mark.parametrize(
    ("regime", "expected_max"),
    [
        (OverallRegime.RISK_ON, 4),
        (OverallRegime.CAUTIOUS, 3),
        (OverallRegime.DEFENSIVE, 2),
        (OverallRegime.CRISIS, 1),
    ],
)
def test_effective_mean_reversion_max_positions_all_regimes(
    settings: Settings,
    regime: OverallRegime,
    expected_max: int,
) -> None:
    assert effective_mean_reversion_max_positions(settings, regime) == expected_max
