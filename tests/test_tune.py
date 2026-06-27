import json

from pipeline.config import ExperimentConfig
from pipeline.tune import neighbor_recipes, write_winner_config


def test_neighbors_perturb_each_active_slider_up_and_down_clamped():
    base = {"rewards": {"progress": 50, "speed": 90}, "penalties": {"crash": 100}}
    neighbors = neighbor_recipes(base, step=15)
    labels = {label for label, _ in neighbors}
    assert "progress_up" in labels and "progress_down" in labels
    recipes = dict(neighbors)
    assert recipes["speed_up"]["rewards"]["speed"] == 100.0
    assert recipes["speed_down"]["rewards"]["speed"] == 75.0
    assert recipes["crash_down"]["penalties"]["crash"] == 85.0
    # crash_up would be 115 -> clamps to 100 == current -> dropped
    assert "crash_up" not in labels


def test_neighbors_are_deterministic():
    base = {"rewards": {"progress": 50}, "penalties": {"stall": 40}}
    assert neighbor_recipes(base, step=10) == neighbor_recipes(base, step=10)


def test_write_winner_config_is_rerunnable(tmp_path):
    # The tuned winner must serialize back into a config that from_path can re-run.
    cfg = ExperimentConfig.from_dict({
        "run_name": "tune",
        "strategies": [{"name": "base", "strategy": "beginner_mix",
                        "params": {"rewards": {"progress": 50}, "penalties": {"crash": 80}}}],
    })
    out = write_winner_config(cfg, tmp_path / "auto_winner.json")
    assert out.exists()
    reloaded = ExperimentConfig.from_path(out)
    assert reloaded.strategies[0].name == "base"
    assert reloaded.strategies[0].strategy == "beginner_mix"
    assert reloaded.strategies[0].params == {"rewards": {"progress": 50}, "penalties": {"crash": 80}}
