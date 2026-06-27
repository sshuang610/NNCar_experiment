from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any

from .paths import resolve_project_path


_BASELINE_NAMES = {"speed_only_baseline", "progress_only"}


@dataclass
class StrategyConfig:
    name: str
    strategy: str = "beginner_mix"
    params: dict[str, Any] = field(default_factory=dict)


def _default_strategy_type(name: str) -> str:
    return name if name in _BASELINE_NAMES else "beginner_mix"


@dataclass
class ExperimentConfig:
    run_name: str = "baseline"
    output_dir: str = "artifacts/runs"
    architecture: list[int] = field(default_factory=lambda: [6, 6, 4])
    population_size: int = 50
    generations: int = 10
    mutation_rate: int = 90
    train_seeds: list[int] = field(default_factory=lambda: [101])
    validation_seeds: list[int] = field(default_factory=lambda: [202, 203, 204])
    time_limit_seconds: float = 30.0
    fps: int = 30
    strategies: list[StrategyConfig] = field(
        default_factory=lambda: [StrategyConfig(name="speed_only_baseline")]
    )
    parallel_workers: int = 1
    master_seed: int = 1234
    retry_generation: int = 10
    retry_min_avg_max_track_progress: float = 0.2
    max_seed_retries: int = 0
    track_cell_size: int = 120
    track_half_width: float = 34.0

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ExperimentConfig":
        strategies = [
            StrategyConfig(
                name=item["name"],
                strategy=item.get("strategy", _default_strategy_type(item["name"])),
                params=item.get("params", {}),
            )
            for item in data.get("strategies", [{"name": "speed_only_baseline"}])
        ]
        return cls(
            run_name=data.get("run_name", "baseline"),
            output_dir=data.get("output_dir", "artifacts/runs"),
            architecture=list(data.get("architecture", [6, 6, 4])),
            population_size=int(data.get("population_size", 50)),
            generations=int(data.get("generations", 10)),
            mutation_rate=int(data.get("mutation_rate", 90)),
            train_seeds=[int(seed) for seed in data.get("train_seeds", [101])],
            validation_seeds=[
                int(seed) for seed in data.get("validation_seeds", [202, 203, 204])
            ],
            time_limit_seconds=float(data.get("time_limit_seconds", 30.0)),
            fps=int(data.get("fps", 30)),
            strategies=strategies,
            parallel_workers=int(data.get("parallel_workers", 1)),
            master_seed=int(data.get("master_seed", 1234)),
            retry_generation=int(data.get("retry_generation", 10)),
            retry_min_avg_max_track_progress=float(
                data.get(
                    "retry_min_avg_max_track_progress",
                    data.get("min_completion_rate", 0.2),
                )
            ),
            max_seed_retries=int(data.get("max_seed_retries", 0)),
            track_cell_size=int(data.get("track_cell_size", 120)),
            track_half_width=float(data.get("track_half_width", 34.0)),
        )

    @classmethod
    def from_path(cls, path: str | Path) -> "ExperimentConfig":
        with resolve_project_path(path).open("r", encoding="utf-8") as handle:
            return cls.from_dict(json.load(handle))
