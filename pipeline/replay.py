from __future__ import annotations

import argparse
from pathlib import Path

from .fitness import build_strategy
from .render import write_replay_svg
from .simulator import Simulator
from .storage import load_model, write_json
from .track import generate_track


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay a saved model on a generated track.")
    parser.add_argument("--model", required=True, help="Path to best_model.npz")
    parser.add_argument("--seed", type=int, required=True, help="Track seed for replay")
    parser.add_argument("--output-dir", default="artifacts/replays", help="Where replay outputs are written")
    args = parser.parse_args()

    network, metadata = load_model(args.model)
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
    strategy = build_strategy(metadata.get("strategy_name", "speed_only_baseline"))
    result = simulator.run_episode(network, strategy)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = Path(args.model)
    strategy_name = model_path.parent.name
    base_name = f"{strategy_name}_{model_path.stem}_seed_{args.seed}"
    svg_path = output_dir / f"{base_name}.svg"
    write_replay_svg(track, result.trajectory, svg_path, result.trajectory[-1])
    write_json(output_dir / f"{base_name}.json", {"metadata": metadata, "metrics": result.metrics.__dict__})


if __name__ == "__main__":
    main()
