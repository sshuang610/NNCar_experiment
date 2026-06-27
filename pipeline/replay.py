from __future__ import annotations

import argparse
from pathlib import Path

from .fitness import build_strategy
from .render import write_replay_svg
from .paths import resolve_project_path
from .simulator import Simulator
from .storage import load_model, write_json
from .track import generate_track


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay a saved model on a generated track.")
    parser.add_argument("--model", required=True, help="Path to best_model.npz")
    parser.add_argument("--seed", type=int, required=True, help="Track seed for replay")
    parser.add_argument("--output-dir", default="artifacts/replays", help="Where replay outputs are written")
    args = parser.parse_args()

    model_path = resolve_project_path(args.model)
    network, metadata = load_model(model_path)
    track = generate_track(
        seed=args.seed,
        cell_size=int(metadata.get("track_cell_size", 120)),
        half_width=float(metadata.get("track_half_width", 34.0)),
    )
    simulator = Simulator(
        track=track,
        fps=int(metadata.get("fps", 30)),
        time_limit_seconds=float(metadata.get("time_limit_seconds", 30.0)),
    )
    strategy = build_strategy(
        metadata.get("strategy", metadata.get("strategy_name", "speed_only_baseline")),
        metadata.get("strategy_params"),
    )
    result = simulator.run_episode(network, strategy)

    output_dir = resolve_project_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    strategy_name = model_path.parent.name
    base_name = f"{strategy_name}_{model_path.stem}_seed_{args.seed}"
    svg_path = output_dir / f"{base_name}.svg"
    write_replay_svg(track, result.trajectory, svg_path, result.trajectory[-1])
    write_json(output_dir / f"{base_name}.json", {"metadata": metadata, "metrics": result.metrics.__dict__})


if __name__ == "__main__":
    main()
