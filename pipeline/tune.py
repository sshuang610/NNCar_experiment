from __future__ import annotations

import argparse
import copy
import json
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

from .config import ExperimentConfig, StrategyConfig
from .paths import resolve_project_path
from .training import run_experiment


def _clamp(value: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return max(lo, min(hi, value))


def neighbor_recipes(base: dict[str, Any], step: float) -> list[tuple[str, dict]]:
    """Coordinate search: perturb each active slider +/- step (clamped 0..100).

    Variants equal to the base after clamping are dropped. Deterministic order:
    rewards first (insertion order) then penalties, each as <name>_up, <name>_down.
    """
    out: list[tuple[str, dict]] = []
    for group in ("rewards", "penalties"):
        for slider, current in base.get(group, {}).items():
            for direction, delta in (("up", step), ("down", -step)):
                new_value = _clamp(float(current) + delta)
                if new_value == float(current):
                    continue
                variant = copy.deepcopy(base)
                variant[group][slider] = new_value
                out.append((f"{slider}_{direction}", variant))
    return out


def _base_strategy(config: ExperimentConfig) -> StrategyConfig:
    for strat in config.strategies:
        if strat.strategy == "beginner_mix":
            return strat
    raise ValueError("Base config has no beginner_mix strategy to tune")


def _rank_tuple_from_result(result: dict) -> tuple:
    validation = result["best_validation"]
    finish = int(validation["finish_count"])
    finish_time = validation["avg_finish_time"]
    time_score = -(float(finish_time) if finish_time is not None else 1e9)
    return (finish, time_score, float(validation["avg_max_track_progress"]))


def _pick_winner(run_dir: Path, config: ExperimentConfig) -> ExperimentConfig:
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    best = max(summary["strategies"], key=_rank_tuple_from_result)
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    name_to_params = {s["name"]: s["params"] for s in manifest["strategies"]}
    params = name_to_params[best["strategy_name"]]
    return replace(
        config,
        strategies=[StrategyConfig(name="base", strategy="beginner_mix", params=params)],
    )


def run_round(config: ExperimentConfig, step: float) -> ExperimentConfig:
    """Build a config of the base plus its neighbors, run it, return the winner config."""
    base = _base_strategy(config)
    strategies = [StrategyConfig(name="base", strategy="beginner_mix", params=base.params)]
    for label, recipe in neighbor_recipes(base.params, step):
        strategies.append(StrategyConfig(name=label, strategy="beginner_mix", params=recipe))
    round_config = replace(config, strategies=strategies)
    run_dir = run_experiment(round_config)
    return _pick_winner(run_dir, config)


def write_winner_config(config: ExperimentConfig, out_path: str | Path) -> Path:
    """Serialize the tuned winner ExperimentConfig to a re-runnable config JSON."""
    path = resolve_project_path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(config), indent=2), encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Coordinate-search auto-tune for BeginnerMix recipes.")
    parser.add_argument("--base-config", required=True)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--step", type=float, default=15.0)
    parser.add_argument(
        "--out",
        default=None,
        help="Write the final tuned winner config JSON here (re-runnable by run_experiment).",
    )
    args = parser.parse_args()

    config = ExperimentConfig.from_path(args.base_config)
    for round_idx in range(1, args.rounds + 1):
        config = run_round(config, args.step)
        winner = _base_strategy(config)
        print(f"round {round_idx} winner params: {json.dumps(winner.params)}")

    if args.out:
        out_path = write_winner_config(config, args.out)
        print(f"winner config written to {out_path}")


if __name__ == "__main__":
    main()
