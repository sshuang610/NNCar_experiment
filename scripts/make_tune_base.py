"""Build configs/tune/auto_base.json from a Stage-1 preset run dir.

Local helper for the CLI workflow (the Colab notebook does this inline).
Picks the best non-baseline strategy from a finished preset run and writes a
single-strategy tune base config.

Usage:
    uv run python scripts/make_tune_base.py <run_dir> [parallel_workers]

Example:
    uv run python scripts/make_tune_base.py artifacts/runs/20260627T..._beginner_mix_presets 4
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

BASELINES = {"speed_only_baseline", "progress_only"}


def _rank(result: dict) -> tuple:
    validation = result["best_validation"]
    finish_time = validation["avg_finish_time"]
    time_score = -(float(finish_time) if finish_time is not None else 1e9)
    return (
        int(validation["finish_count"]),
        time_score,
        float(validation["avg_max_track_progress"]),
    )


def main() -> None:
    if len(sys.argv) < 2:
        sys.exit("usage: make_tune_base.py <run_dir> [parallel_workers]")
    run_dir = Path(sys.argv[1])
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 4

    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))

    candidates = [s for s in summary["strategies"] if s["strategy_name"] not in BASELINES]
    winner = max(candidates, key=_rank)
    params = {s["name"]: s["params"] for s in manifest["strategies"]}[winner["strategy_name"]]

    config = {
        **{
            key: manifest[key]
            for key in (
                "architecture", "population_size", "generations", "mutation_rate",
                "train_seeds", "validation_seeds", "time_limit_seconds", "fps", "master_seed",
            )
        },
        "run_name": "tune",
        "output_dir": "artifacts/runs",
        "max_seed_retries": 0,
        "parallel_workers": workers,
        "strategies": [{"name": "base", "strategy": "beginner_mix", "params": params}],
    }

    os.makedirs("configs/tune", exist_ok=True)
    out = Path("configs/tune/auto_base.json")
    out.write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(f"winner = {winner['strategy_name']}  ->  {out}  (parallel_workers={workers})")
    print(json.dumps(params))


if __name__ == "__main__":
    main()
