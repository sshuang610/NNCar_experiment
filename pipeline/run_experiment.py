from __future__ import annotations

import argparse

from .config import ExperimentConfig
from .training import run_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description="Run headless GA experiments for neural-network cars.")
    parser.add_argument("--config", required=True, help="Path to a JSON experiment config")
    parser.add_argument(
        "--render",
        action="store_true",
        help="Write a live multi-strategy dashboard HTML while training runs",
    )
    args = parser.parse_args()
    config = ExperimentConfig.from_path(args.config)
    run_dir = run_experiment(config, render=args.render)
    print(run_dir)


if __name__ == "__main__":
    main()
