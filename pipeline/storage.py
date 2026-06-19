from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, UTC
import json
from pathlib import Path
from typing import Any

import numpy as np

from .nn import NeuralNetwork


def utc_timestamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def append_jsonl(path: Path, data: Any) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(data, sort_keys=True) + "\n")


def save_model(path: Path, network: NeuralNetwork, metadata: dict[str, Any]) -> None:
    arrays = {
        "sizes": np.array(network.sizes, dtype=int),
        "metadata_json": np.array(json.dumps(metadata), dtype=object),
    }
    for idx, weight in enumerate(network.weights):
        arrays[f"weight_{idx}"] = weight
    for idx, bias in enumerate(network.biases):
        arrays[f"bias_{idx}"] = bias
    np.savez(path, **arrays)


def load_model(path: str | Path) -> tuple[NeuralNetwork, dict[str, Any]]:
    payload = np.load(Path(path), allow_pickle=True)
    sizes = payload["sizes"].tolist()
    weights = []
    biases = []
    weight_idx = 0
    while f"weight_{weight_idx}" in payload:
        weights.append(payload[f"weight_{weight_idx}"])
        weight_idx += 1
    bias_idx = 0
    while f"bias_{bias_idx}" in payload:
        biases.append(payload[f"bias_{bias_idx}"])
        bias_idx += 1
    metadata = json.loads(str(payload["metadata_json"].item()))
    return NeuralNetwork(sizes=sizes, weights=weights, biases=biases), metadata

