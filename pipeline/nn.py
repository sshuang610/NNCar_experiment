from __future__ import annotations

from dataclasses import dataclass
import math
import random
from typing import Iterable

import numpy as np


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -60.0, 60.0)))


@dataclass
class NeuralNetwork:
    sizes: list[int]
    weights: list[np.ndarray]
    biases: list[np.ndarray]

    @classmethod
    def random(cls, sizes: list[int], rng: np.random.Generator) -> "NeuralNetwork":
        weights = [rng.standard_normal((y, x)) for x, y in zip(sizes[:-1], sizes[1:])]
        biases = [rng.standard_normal((y, 1)) for y in sizes[1:]]
        return cls(sizes=list(sizes), weights=weights, biases=biases)

    def clone(self) -> "NeuralNetwork":
        return NeuralNetwork(
            sizes=list(self.sizes),
            weights=[weight.copy() for weight in self.weights],
            biases=[bias.copy() for bias in self.biases],
        )

    def feedforward(self, inputs: Iterable[float]) -> np.ndarray:
        activations = np.array(list(inputs), dtype=float).reshape(-1, 1)
        for bias, weight in zip(self.biases, self.weights):
            activations = sigmoid((weight @ activations) + bias)
        return activations


def _flatten(arrays: list[np.ndarray]) -> np.ndarray:
    return np.concatenate([array.reshape(-1) for array in arrays])


def _unflatten(template: list[np.ndarray], flat: np.ndarray) -> list[np.ndarray]:
    arrays: list[np.ndarray] = []
    offset = 0
    for array in template:
        size = math.prod(array.shape)
        arrays.append(flat[offset : offset + size].reshape(array.shape).copy())
        offset += size
    return arrays


def uniform_crossover(parent_a: NeuralNetwork, parent_b: NeuralNetwork) -> tuple[NeuralNetwork, NeuralNetwork]:
    child_a = parent_a.clone()
    child_b = parent_b.clone()

    for attr in ("weights", "biases"):
        flat_a = _flatten(getattr(parent_a, attr))
        flat_b = _flatten(getattr(parent_b, attr))
        mixed_a = flat_a.copy()
        mixed_b = flat_b.copy()
        swap = True
        for idx in range(len(flat_a)):
            if swap:
                mixed_a[idx] = flat_b[idx]
                mixed_b[idx] = flat_a[idx]
            swap = not swap
        setattr(child_a, attr, _unflatten(getattr(child_a, attr), mixed_a))
        setattr(child_b, attr, _unflatten(getattr(child_b, attr), mixed_b))

    return child_a, child_b


def mutate(network: NeuralNetwork, mutation_rate: int, rng: random.Random) -> None:
    weight_flat = _flatten(network.weights)
    bias_flat = _flatten(network.biases)
    if len(weight_flat) == 0 or len(bias_flat) == 0:
        return

    for _ in range(mutation_rate):
        weight_idx = rng.randrange(len(weight_flat))
        bias_idx = rng.randrange(len(bias_flat))
        weight_flat[weight_idx] *= rng.uniform(0.8, 1.2)
        bias_flat[bias_idx] *= rng.uniform(0.8, 1.2)

    network.weights = _unflatten(network.weights, weight_flat)
    network.biases = _unflatten(network.biases, bias_flat)
