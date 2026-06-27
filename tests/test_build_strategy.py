import pytest

from pipeline.fitness import build_strategy, BeginnerMix, SpeedOnlyBaseline, STRATEGIES


def test_registry_only_keeps_beginner_mix_and_two_baselines():
    assert set(STRATEGIES) == {"beginner_mix", "speed_only_baseline", "progress_only"}


def test_build_beginner_mix_applies_params():
    strat = build_strategy("beginner_mix", {"rewards": {"progress": 40}, "penalties": {"stall": 60}})
    assert isinstance(strat, BeginnerMix)
    assert strat.rewards == {"progress": 40.0}
    assert strat.penalties == {"stall": 60.0}


def test_build_baseline_ignores_params():
    strat = build_strategy("speed_only_baseline", {"rewards": {"progress": 40}})
    assert isinstance(strat, SpeedOnlyBaseline)


def test_unknown_strategy_raises():
    with pytest.raises(ValueError):
        build_strategy("does_not_exist", {})
