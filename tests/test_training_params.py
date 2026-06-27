from pipeline.config import ExperimentConfig
from pipeline.training import _evaluate_network
from pipeline.nn import NeuralNetwork
import numpy as np


def _net():
    return NeuralNetwork.random([6, 6, 4], np.random.default_rng(0))


def test_evaluate_network_accepts_strategy_config_and_uses_params():
    cfg = ExperimentConfig.from_dict({
        "train_seeds": [101], "validation_seeds": [202],
        "generations": 1, "population_size": 2,
        "strategies": [{"name": "progress_first", "params": {"rewards": {"progress": 50}}}],
    })
    strat_cfg = cfg.strategies[0]
    fitness, summary, episodes = _evaluate_network(_net(), cfg.train_seeds, strat_cfg, cfg)
    assert isinstance(fitness, float)
    assert "avg_max_track_progress" in summary
