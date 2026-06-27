from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from .paths import resolve_project_path
from .storage import load_model, write_json


def export_final_goal_model(npz_path: str | Path, group_id: str = "0",
                            username: str = "player") -> dict[str, Any]:
    """Flatten a saved model into the shared final_goal model JSON format."""
    network, _ = load_model(npz_path)
    weights = [weight.reshape(-1).tolist() for weight in network.weights]
    biases = [bias.reshape(-1).tolist() for bias in network.biases]
    return {"group_id": str(group_id), "username": username,
            "weights": weights, "biases": biases}


def promote_template(
    run_dir: str | Path,
    strategy_name: str,
    template_name: str,
    templates_root: str | Path = "templates",
    group_id: str = "0",
    username: str = "player",
) -> Path:
    """Package one strategy from a finished run into a committed template folder."""
    run_dir = resolve_project_path(run_dir)
    strat_dir = run_dir / "strategies" / strategy_name
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    validation = json.loads((strat_dir / "validation.json").read_text(encoding="utf-8"))
    _, metadata = load_model(strat_dir / "best_model.npz")

    out_root = resolve_project_path(templates_root)
    out_dir = out_root / template_name
    out_dir.mkdir(parents=True, exist_ok=True)

    recipe = metadata.get("strategy_params", {})
    write_json(out_dir / "recipe.json", recipe)
    write_json(out_dir / "reproduce.json", {
        "template_name": template_name,
        "strategy_name": strategy_name,
        "strategy": metadata.get("strategy", "beginner_mix"),
        "params": recipe,
        "git_commit": manifest.get("git_commit", "unknown"),
        "run_id": manifest.get("run_id"),
        "architecture": manifest.get("architecture"),
        "population_size": manifest.get("population_size"),
        "generations": manifest.get("generations"),
        "mutation_rate": manifest.get("mutation_rate"),
        "train_seeds": manifest.get("train_seeds"),
        "validation_seeds": manifest.get("validation_seeds"),
        "time_limit_seconds": manifest.get("time_limit_seconds"),
        "fps": manifest.get("fps"),
        "master_seed": manifest.get("master_seed"),
        "track_cell_size": manifest.get("track_cell_size"),
        "track_half_width": manifest.get("track_half_width"),
        "evolution_seed": metadata.get("evolution_seed"),
    })
    write_json(out_dir / "result.json", {
        "finish_count": validation.get("finish_count"),
        "avg_finish_time": validation.get("avg_finish_time"),
        "avg_max_track_progress": validation.get("avg_max_track_progress"),
        "avg_collision_count": validation.get("avg_collision_count"),
        "avg_stall_time": validation.get("avg_stall_time"),
        "avg_spin_time": validation.get("avg_spin_time"),
    })
    shutil.copy2(strat_dir / "best_model.npz", out_dir / "best_model.npz")
    write_json(out_dir / "model.json",
               export_final_goal_model(out_dir / "best_model.npz", group_id, username))

    _update_index(out_root, template_name, validation, recipe)
    return out_dir


def _update_index(out_root: Path, template_name: str, validation: dict, recipe: dict) -> None:
    index_path = out_root / "index.json"
    if index_path.exists():
        index = json.loads(index_path.read_text(encoding="utf-8"))
    else:
        index = {"templates": []}
    index["templates"] = [t for t in index["templates"] if t["name"] != template_name]
    index["templates"].append({
        "name": template_name,
        "recipe": recipe,
        "finish_count": validation.get("finish_count"),
        "avg_finish_time": validation.get("avg_finish_time"),
        "avg_max_track_progress": validation.get("avg_max_track_progress"),
    })
    write_json(index_path, index)
