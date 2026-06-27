from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import csv
from dataclasses import asdict
import multiprocessing as mp
from pathlib import Path
import random
import subprocess
from typing import Any

import numpy as np

from .config import ExperimentConfig, StrategyConfig
from .fitness import build_strategy
from .nn import NeuralNetwork, mutate, uniform_crossover
from .paths import resolve_project_path
from .simulator import Simulator
from .storage import append_jsonl, ensure_dir, save_model, utc_timestamp, write_json
from .track import generate_track


def _rank_key(validation_summary: dict[str, Any]) -> tuple[int, float, float]:
    finish_count = int(validation_summary["finish_count"])
    avg_finish_time = validation_summary["avg_finish_time"]
    finish_time_score = -(avg_finish_time if avg_finish_time is not None else 999999.0)
    return finish_count, finish_time_score, validation_summary["avg_max_track_progress"]


def _stable_strategy_seed(master_seed: int, strategy_name: str) -> int:
    return master_seed + sum(ord(char) for char in strategy_name)


def _evaluate_network(
    network: NeuralNetwork,
    seeds: list[int],
    strategy_config: StrategyConfig,
    config: ExperimentConfig,
) -> tuple[float, dict[str, float], list[dict[str, Any]]]:
    strategy = build_strategy(strategy_config.strategy, strategy_config.params)
    episode_metrics: list[dict[str, Any]] = []
    training_fitness: list[float] = []
    finish_times: list[float] = []
    progresses: list[float] = []
    collisions: list[int] = []
    stalls: list[float] = []
    spins: list[float] = []

    for seed in seeds:
        track = generate_track(
            seed=seed,
            cell_size=config.track_cell_size,
            half_width=config.track_half_width,
        )
        simulator = Simulator(track=track, fps=config.fps, time_limit_seconds=config.time_limit_seconds)
        result = simulator.run_episode(network, strategy)
        metrics = asdict(result.metrics)
        metrics["seed"] = seed
        episode_metrics.append(metrics)
        training_fitness.append(result.metrics.training_fitness)
        progresses.append(result.metrics.max_track_progress)
        collisions.append(result.metrics.collision_count)
        stalls.append(result.metrics.stall_time)
        spins.append(result.metrics.spin_time)
        if result.metrics.finish_time is not None:
            finish_times.append(result.metrics.finish_time)

    summary = {
        "avg_training_fitness": float(np.mean(training_fitness)),
        "avg_finish_time": float(np.mean(finish_times)) if finish_times else None,
        "avg_max_track_progress": float(np.mean(progresses)),
        "avg_collision_count": float(np.mean(collisions)),
        "avg_stall_time": float(np.mean(stalls)),
        "avg_spin_time": float(np.mean(spins)),
        "finish_count": int(sum(1 for item in episode_metrics if item["finished_within_30s"])),
    }
    return summary["avg_training_fitness"], summary, episode_metrics


def _render_payload(
    network: NeuralNetwork,
    strategy_config: StrategyConfig,
    seed: int,
    config: ExperimentConfig,
) -> dict[str, Any]:
    track = generate_track(
        seed=seed,
        cell_size=config.track_cell_size,
        half_width=config.track_half_width,
    )
    simulator = Simulator(track=track, fps=config.fps, time_limit_seconds=config.time_limit_seconds)
    result = simulator.run_episode(network, build_strategy(strategy_config.strategy, strategy_config.params))
    stride = max(1, len(result.trajectory) // 200)
    return {
        "seed": seed,
        "track_polyline": track.polyline,
        "canvas_size": track.canvas_size,
        "track_half_width": track.half_width,
        "trajectory": result.trajectory[::stride] or result.trajectory,
        "car_position": result.trajectory[-1],
        "metrics": asdict(result.metrics),
    }


def _breed_next_generation(
    scored_population: list[tuple[NeuralNetwork, float, dict[str, float]]],
    population_size: int,
    mutation_rate: int,
    rng: random.Random,
) -> tuple[list[NeuralNetwork], tuple[NeuralNetwork, NeuralNetwork]]:
    sorted_population = sorted(scored_population, key=lambda item: item[1], reverse=True)
    parent_a = sorted_population[0][0].clone()
    parent_b = sorted_population[1][0].clone()
    next_population = [parent_a.clone(), parent_b.clone()]

    while len(next_population) < population_size:
        child_a, child_b = uniform_crossover(parent_a, parent_b)
        mutate(child_a, mutation_rate, rng)
        mutate(child_b, mutation_rate, rng)
        next_population.append(child_a)
        if len(next_population) < population_size:
            next_population.append(child_b)

    return next_population, (parent_a, parent_b)


def _train_strategy(
    strategy_config: StrategyConfig,
    config: ExperimentConfig,
    run_dir: Path,
    progress_queue: mp.Queue | None = None,
) -> dict[str, Any]:
    strategy_dir = ensure_dir(run_dir / "strategies" / strategy_config.name)
    base_strategy_seed = _stable_strategy_seed(config.master_seed, strategy_config.name)
    train_log_path = strategy_dir / "train_log.jsonl"
    best_model: NeuralNetwork | None = None
    best_validation_summary: dict[str, Any] | None = None
    best_metadata: dict[str, Any] | None = None
    best_rank: tuple[int, float, float] | None = None
    attempt_seeds: list[int] = []
    total_completed_generations = 0
    retry_triggered = False

    for attempt in range(1, config.max_seed_retries + 2):
        evolution_seed = base_strategy_seed + (attempt - 1)
        attempt_seeds.append(evolution_seed)
        rng = np.random.default_rng(evolution_seed)
        py_rng = random.Random(evolution_seed)
        population = [
            NeuralNetwork.random(config.architecture, rng)
            for _ in range(config.population_size)
        ]
        attempt_best_completion_rate = 0.0
        attempt_best_avg_max_track_progress = 0.0
        retry_this_attempt = False

        for generation in range(1, config.generations + 1):
            scored_population: list[tuple[NeuralNetwork, float, dict[str, float]]] = []
            for network in population:
                train_score, train_summary, _ = _evaluate_network(
                    network=network,
                    seeds=config.train_seeds,
                    strategy_config=strategy_config,
                    config=config,
                )
                scored_population.append((network, train_score, train_summary))

            scored_population.sort(key=lambda item: item[1], reverse=True)
            best_train_network = scored_population[0][0].clone()
            _, validation_summary, validation_episodes = _evaluate_network(
                network=best_train_network,
                seeds=config.validation_seeds,
                strategy_config=strategy_config,
                config=config,
            )
            total_completed_generations += 1
            completion_rate = validation_summary["finish_count"] / len(config.validation_seeds)
            attempt_best_completion_rate = max(attempt_best_completion_rate, completion_rate)
            attempt_best_avg_max_track_progress = max(
                attempt_best_avg_max_track_progress,
                float(validation_summary["avg_max_track_progress"]),
            )
            current_rank = _rank_key(validation_summary)
            if best_rank is None or current_rank > best_rank:
                best_rank = current_rank
                best_model = best_train_network.clone()
                best_validation_summary = {
                    **validation_summary,
                    "episodes": validation_episodes,
                    "attempt": attempt,
                    "evolution_seed": evolution_seed,
                }
                best_metadata = {
                    "strategy_name": strategy_config.name,
                    "strategy": strategy_config.strategy,
                    "strategy_params": strategy_config.params,
                    "architecture": config.architecture,
                    "generation": generation,
                    "attempt": attempt,
                    "evolution_seed": evolution_seed,
                    "train_seeds": config.train_seeds,
                    "validation_seeds": config.validation_seeds,
                    "mutation_rate": config.mutation_rate,
                    "time_limit_seconds": config.time_limit_seconds,
                    "fps": config.fps,
                    "track_cell_size": config.track_cell_size,
                    "track_half_width": config.track_half_width,
                    "model_architecture_version": "v1",
                }

            if (
                generation == config.retry_generation
                and attempt_best_avg_max_track_progress < config.retry_min_avg_max_track_progress
                and attempt <= config.max_seed_retries
            ):
                retry_this_attempt = True
                retry_triggered = True

            append_jsonl(
                train_log_path,
                {
                    "attempt": attempt,
                    "evolution_seed": evolution_seed,
                    "generation": generation,
                    "strategy_name": strategy_config.name,
                    "best_training_fitness": scored_population[0][1],
                    "completion_rate": completion_rate,
                    "attempt_best_completion_rate": attempt_best_completion_rate,
                    "attempt_best_avg_max_track_progress": attempt_best_avg_max_track_progress,
                    "retry_scheduled": retry_this_attempt,
                    "parent_selection": "top_2_by_fitness",
                    "train_summary": scored_population[0][2],
                    "validation_summary": validation_summary,
                },
            )
            if progress_queue is not None:
                progress_queue.put(
                    {
                        "type": "progress",
                        "strategy_name": strategy_config.name,
                        "attempt": attempt,
                        "evolution_seed": evolution_seed,
                        "generation": generation,
                        "completed_generations": config.generations,
                        "best_training_fitness": scored_population[0][1],
                        "best_validation_generation": (
                            best_metadata["generation"] if best_metadata else generation
                        ),
                        "validation_summary": validation_summary,
                        "render": _render_payload(
                            network=best_train_network,
                            strategy_config=strategy_config,
                            seed=config.train_seeds[0],
                            config=config,
                        ),
                    }
                )
            if retry_this_attempt:
                break

            population, parents = _breed_next_generation(
                scored_population=scored_population,
                population_size=config.population_size,
                mutation_rate=config.mutation_rate,
                rng=py_rng,
            )

        if not retry_this_attempt:
            break

    if best_model is None or best_validation_summary is None or best_metadata is None:
        raise RuntimeError(f"No model produced for strategy {strategy_config.name}")

    save_model(strategy_dir / "best_model.npz", best_model, best_metadata)
    write_json(strategy_dir / "validation.json", best_validation_summary)
    result = {
        "strategy_name": strategy_config.name,
        "best_generation": best_metadata["generation"],
        "best_attempt": best_metadata["attempt"],
        "evolution_seed": best_metadata["evolution_seed"],
        "attempts_completed": len(attempt_seeds),
        "attempt_seeds": attempt_seeds,
        "retry_triggered": retry_triggered,
        "completed_generations": total_completed_generations,
        "best_validation": best_validation_summary,
        "best_model_path": str(strategy_dir / "best_model.npz"),
    }
    if progress_queue is not None:
        progress_queue.put({"type": "result", "result": result})
    return result


def _train_strategy_worker(
    strategy_config: StrategyConfig,
    config: ExperimentConfig,
    run_dir: Path,
    progress_queue: mp.Queue,
) -> None:
    _train_strategy(
        strategy_config=strategy_config,
        config=config,
        run_dir=run_dir,
        progress_queue=progress_queue,
    )


def _write_summary(run_dir: Path, run_id: str, results: list[dict[str, Any]]) -> None:
    write_json(run_dir / "summary.json", {"run_id": run_id, "strategies": results})
    with (run_dir / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "strategy_name",
                "best_generation",
                "best_attempt",
                "evolution_seed",
                "attempts_completed",
                "retry_triggered",
                "completed_generations",
                "validation_finish_count",
                "finished_within_30s_any_validation",
                "avg_finish_time",
                "avg_max_track_progress",
                "avg_collision_count",
                "avg_stall_time",
                "avg_spin_time",
                "best_model_path",
            ],
        )
        writer.writeheader()
        for result in results:
            validation = result["best_validation"]
            writer.writerow(
                {
                    "strategy_name": result["strategy_name"],
                    "best_generation": result["best_generation"],
                    "best_attempt": result["best_attempt"],
                    "evolution_seed": result["evolution_seed"],
                    "attempts_completed": result["attempts_completed"],
                    "retry_triggered": result["retry_triggered"],
                    "completed_generations": result["completed_generations"],
                    "validation_finish_count": validation["finish_count"],
                    "finished_within_30s_any_validation": validation["finish_count"] > 0,
                    "avg_finish_time": validation["avg_finish_time"],
                    "avg_max_track_progress": validation["avg_max_track_progress"],
                    "avg_collision_count": validation["avg_collision_count"],
                    "avg_stall_time": validation["avg_stall_time"],
                    "avg_spin_time": validation["avg_spin_time"],
                    "best_model_path": result["best_model_path"],
                }
            )


def _git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=resolve_project_path("."),
            capture_output=True, text=True, check=True,
        )
        return out.stdout.strip()
    except Exception:
        return "unknown"


def run_experiment(config: ExperimentConfig, render: bool = False) -> Path:
    run_id = f"{utc_timestamp()}_{config.run_name}"
    run_dir = ensure_dir(resolve_project_path(config.output_dir) / run_id)
    ensure_dir(run_dir / "strategies")
    manifest = {
        "run_id": run_id,
        "run_name": config.run_name,
        "git_commit": _git_commit(),
        "architecture": config.architecture,
        "population_size": config.population_size,
        "generations": config.generations,
        "mutation_rate": config.mutation_rate,
        "train_seeds": config.train_seeds,
        "validation_seeds": config.validation_seeds,
        "time_limit_seconds": config.time_limit_seconds,
        "fps": config.fps,
        "parallel_workers": config.parallel_workers,
        "strategy_randomization": "master_seed plus stable strategy-name offset",
        "retry_generation": config.retry_generation,
        "retry_min_avg_max_track_progress": config.retry_min_avg_max_track_progress,
        "max_seed_retries": config.max_seed_retries,
        "seed_retry_behavior": "stop the current attempt at retry_generation when best validation avg_max_track_progress stays below threshold, then rerun with the next evolution seed",
        "strategies": [asdict(strategy) for strategy in config.strategies],
        "model_architecture_version": "v1",
        "parent_selection": "top_2_by_fitness",
        "fitness_baseline": "score += velocity",
        "best_generation_meaning": "best by validation finish count, then finish time, then progress",
    }
    write_json(run_dir / "manifest.json", manifest)

    if render:
        from .visualize import run_dashboard

        ctx = mp.get_context("spawn")
        progress_queue: mp.Queue = ctx.Queue()
        processes = [
            ctx.Process(
                target=_train_strategy_worker,
                args=(strategy, config, run_dir, progress_queue),
            )
            for strategy in config.strategies
        ]
        for process in processes:
            process.start()
        try:
            results = run_dashboard(
                run_name=config.run_name,
                dashboard_path=run_dir / "dashboard.html",
                strategies=[strategy.name for strategy in config.strategies],
                progress_queue=progress_queue,
                processes=processes,
            )
        finally:
            for process in processes:
                process.join()
    elif config.parallel_workers > 1 and len(config.strategies) > 1:
        with ProcessPoolExecutor(max_workers=config.parallel_workers) as pool:
            futures = [
                pool.submit(_train_strategy, strategy, config, run_dir)
                for strategy in config.strategies
            ]
            results = [future.result() for future in futures]
    else:
        results = [_train_strategy(strategy, config, run_dir) for strategy in config.strategies]

    _write_summary(run_dir, run_id, results)
    return run_dir
