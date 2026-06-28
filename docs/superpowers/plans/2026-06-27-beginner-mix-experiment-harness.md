# BeginnerMix 10-param Experiment Harness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Convert `NNCars-Fitness-Experiments` from ~40 hardcoded fitness classes to a single parameterized `BeginnerMix` block-fitness, then run a presets→auto-tune sweep that emits reproducible "template" param combos (config + seeds + trained model in final_goal format).

**Architecture:** A `BeginnerMix` strategy implements the §9 block model (auto-normalized rewards + independent per-second penalties + one-shot crash + fixed finish bonus). The `params` field is finally plumbed through `build_strategy → training → replay` and saved in model metadata. A two-stage flow (`configs/presets/` then `pipeline/tune.py` coordinate search) finds winners, which `pipeline/export.py` promotes into committed `templates/`. A Colab notebook drives the whole thing; old artifacts/configs/game files move to `archive/`.

**Tech Stack:** Python 3.10+, numpy, `uv` for env, pytest (run via `uv run --with pytest`).

**Spec:** [docs/superpowers/specs/2026-06-27-beginner-mix-experiment-harness-design.md](../specs/2026-06-27-beginner-mix-experiment-harness-design.md)

**Conventions:**
- Run tests with: `uv run --with pytest pytest <path> -v` (keeps pytest out of project deps).
- Commit messages end with a `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>` trailer.
- All paths are relative to the repo root `NNCars-Fitness-Experiments/`.

---

## Task 1: `BeginnerMix` block-fitness model

**Files:**
- Modify: `pipeline/fitness.py` (add constants + `_split_params` + `BeginnerMix`)
- Test: `tests/test_beginner_mix.py` (create)

- [ ] **Step 1: Write a test helper + first failing test (back-compat reward shape)**

Create `tests/test_beginner_mix.py`:

```python
from __future__ import annotations

from pipeline.fitness import BeginnerMix, FINISH_BONUS, B_CRASH
from pipeline.simulator import StepContext


def ctx(**overrides) -> StepContext:
    base = dict(
        velocity=0.0,
        progress_delta=0.0,
        progress_ratio=0.0,
        center_offset=0.0,
        normalized_center_offset=0.0,
        heading_alignment=1.0,
        front_clearance=100.0,
        min_clearance=100.0,
        side_clearance_balance=0.0,
        turn_amount=0.0,
        collided=False,
        finished=False,
        is_stalled=False,
        is_spinning=False,
        frame=30,
        time_elapsed=1.0,  # dt = time_elapsed / frame = 1/30
    )
    base.update(overrides)
    return StepContext(**base)


def test_flat_dict_is_treated_as_rewards():
    strat = BeginnerMix()
    strat.configure({"speed": 30, "progress": 40, "centered": 10, "alignment": 10, "safety": 10})
    assert strat.rewards == {"speed": 30.0, "progress": 40.0, "centered": 10.0,
                             "alignment": 10.0, "safety": 10.0}
    assert strat.penalties == {}
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run --with pytest pytest tests/test_beginner_mix.py::test_flat_dict_is_treated_as_rewards -v`
Expected: FAIL — `ImportError: cannot import name 'BeginnerMix'`.

- [ ] **Step 3: Implement constants, `_split_params`, and `BeginnerMix`**

Add to `pipeline/fitness.py` (after the `_event_bonus` helper, before `SpeedOnlyBaseline`):

```python
B = 10.0
CRASH_SECONDS = 15.0
B_CRASH = B * CRASH_SECONDS          # 150.0
FINISH_SECONDS = 300.0
FINISH_BONUS = B * FINISH_SECONDS    # 3000.0

REWARD_BLOCKS = ("speed", "progress", "centered", "alignment", "safety")
PERSTEP_PENALTY_BLOCKS = ("stall", "spin", "wrong_way", "time")


def _split_params(params: dict) -> tuple[dict, dict]:
    """Accept both the new {rewards, penalties} shape and a flat rewards-only dict."""
    if "rewards" in params or "penalties" in params:
        return dict(params.get("rewards", {})), dict(params.get("penalties", {}))
    return dict(params), {}


class BeginnerMix(FitnessStrategy):
    name = "beginner_mix"

    def __init__(self) -> None:
        self.rewards: dict[str, float] = {}
        self.penalties: dict[str, float] = {}

    def configure(self, params: dict) -> None:
        rewards, penalties = _split_params(params)
        self.rewards = {k: float(v) for k, v in rewards.items() if k in REWARD_BLOCKS}
        self.penalties = {
            k: max(0.0, float(v))
            for k, v in penalties.items()
            if k in PERSTEP_PENALTY_BLOCKS or k == "crash"
        }

    def _reward_factors(self, c: StepContext) -> dict[str, float]:
        return {
            "speed": c.velocity / 10.0,
            "progress": min(c.progress_delta / 10.0, 1.0),
            "centered": max(0.0, 1.0 - c.normalized_center_offset),
            "alignment": max(0.0, c.heading_alignment),
            "safety": min(c.min_clearance, 90.0) / 90.0,
        }

    def _perstep_penalty_factors(self, c: StepContext) -> dict[str, float]:
        return {
            "stall": 1.0 if c.is_stalled else 0.0,
            "spin": 1.0 if c.is_spinning else 0.0,
            "wrong_way": 1.0 if c.heading_alignment < 0.0 else 0.0,
            "time": 1.0,
        }

    def score_step(self, context: StepContext) -> float:
        dt = context.time_elapsed / context.frame if context.frame else 0.0

        reward = 0.0
        weight_sum = sum(self.rewards.values())
        if weight_sum > 0.0:
            factors = self._reward_factors(context)
            weighted = sum(self.rewards[k] * factors[k] for k in self.rewards)
            reward = (weighted / weight_sum) * B * dt

        penalty = 0.0
        pfactors = self._perstep_penalty_factors(context)
        for key, weight in self.penalties.items():
            if key == "crash":
                continue
            penalty += (weight / 100.0) * B * pfactors[key] * dt

        step = reward - penalty
        if context.collided:
            step -= (self.penalties.get("crash", 0.0) / 100.0) * B_CRASH
        if context.finished:
            step += FINISH_BONUS
        return step
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run --with pytest pytest tests/test_beginner_mix.py::test_flat_dict_is_treated_as_rewards -v`
Expected: PASS.

- [ ] **Step 5: Add calibration, regression, fps-independence, crash, finish, clamp tests**

Append to `tests/test_beginner_mix.py`:

```python
def test_penalty_at_100_cancels_full_reward_for_that_frame():
    strat = BeginnerMix()
    strat.configure({"rewards": {"progress": 50}, "penalties": {"stall": 100}})
    # progress factor saturates to 1.0 (progress_delta/10 capped), stall active.
    step = strat.score_step(ctx(progress_delta=10.0, is_stalled=True))
    assert abs(step) < 1e-9


def test_extra_reward_blocks_do_not_dilute_penalties():
    a = BeginnerMix(); a.configure({"rewards": {"progress": 40}, "penalties": {"stall": 60}})
    b = BeginnerMix(); b.configure({"rewards": {"progress": 40, "speed": 30, "safety": 30},
                                    "penalties": {"stall": 60}})
    # On a frame with zero reward factors, reward=0 for both; penalty must be identical.
    c = ctx(is_stalled=True)  # velocity 0, progress 0, etc.
    assert a.score_step(c) == b.score_step(c)
    assert a.score_step(c) < 0.0  # pure stall penalty


def test_perstep_pieces_scale_with_dt_so_per_second_is_fps_invariant():
    strat = BeginnerMix()
    strat.configure({"rewards": {"speed": 100}, "penalties": {"time": 50}})
    c30 = ctx(velocity=5.0, frame=30, time_elapsed=1.0)   # dt = 1/30
    c60 = ctx(velocity=5.0, frame=60, time_elapsed=1.0)   # dt = 1/60
    assert abs(strat.score_step(c30) - 2.0 * strat.score_step(c60)) < 1e-9


def test_crash_is_one_shot_and_independent_of_dt():
    strat = BeginnerMix(); strat.configure({"penalties": {"crash": 100}})
    c30 = ctx(collided=True, frame=30, time_elapsed=1.0)
    c60 = ctx(collided=True, frame=60, time_elapsed=1.0)
    assert strat.score_step(c30) == strat.score_step(c60) == -B_CRASH


def test_finish_adds_fixed_bonus():
    strat = BeginnerMix(); strat.configure({})
    assert strat.score_step(ctx(finished=True)) == FINISH_BONUS


def test_negative_penalty_is_clamped_to_zero():
    strat = BeginnerMix(); strat.configure({"penalties": {"stall": -50}})
    assert strat.penalties["stall"] == 0.0
```

- [ ] **Step 6: Run the full test file**

Run: `uv run --with pytest pytest tests/test_beginner_mix.py -v`
Expected: all 7 tests PASS.

- [ ] **Step 7: Commit**

```bash
git add pipeline/fitness.py tests/test_beginner_mix.py
git commit -m "feat: add BeginnerMix block fitness (§9 model)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 2: Slim `STRATEGIES` registry and plumb params into `build_strategy`

**Files:**
- Modify: `pipeline/fitness.py` (replace `STRATEGIES` dict + `build_strategy`; delete the ~38 unused classes)
- Test: `tests/test_build_strategy.py` (create)

- [ ] **Step 1: Write failing test**

Create `tests/test_build_strategy.py`:

```python
import pytest

from pipeline.fitness import build_strategy, BeginnerMix, SpeedOnlyBaseline, STRATEGIES


def test_registry_only_keeps_beginner_mix_and_two_baselines():
    assert set(STRATEGIES) == {"beginner_mix", "speed_only_baseline", "progress_only"}


def test_build_beginner_mix_applies_params():
    strat = build_strategy("beginner_mix", {"rewards": {"progress": 40}, "penalties": {"stall": 60}})
    assert isinstance(strat, BeginnerMix)
    assert strat.rewards == {"progress": 40.0}
    assert strat.penalties == {"stall": 60.0}


def test_build_baseline_ignores_params():
    strat = build_strategy("speed_only_baseline", {"rewards": {"progress": 40}})
    assert isinstance(strat, SpeedOnlyBaseline)


def test_unknown_strategy_raises():
    with pytest.raises(ValueError):
        build_strategy("does_not_exist", {})
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run --with pytest pytest tests/test_build_strategy.py -v`
Expected: FAIL — registry still has ~40 keys / `build_strategy` signature mismatch.

- [ ] **Step 3: Delete unused strategy classes and replace registry + builder**

In `pipeline/fitness.py`, delete every `FitnessStrategy` subclass EXCEPT `SpeedOnlyBaseline`, `ProgressOnly`, and `BeginnerMix`. Then replace the `STRATEGIES` dict and `build_strategy` at the bottom of the file with:

```python
STRATEGIES: dict[str, type[FitnessStrategy]] = {
    BeginnerMix.name: BeginnerMix,
    SpeedOnlyBaseline.name: SpeedOnlyBaseline,
    ProgressOnly.name: ProgressOnly,
}


def build_strategy(strategy_type: str, params: dict | None = None) -> FitnessStrategy:
    try:
        strategy_cls = STRATEGIES[strategy_type]
    except KeyError as exc:
        raise ValueError(f"Unknown fitness strategy: {strategy_type}") from exc
    strategy = strategy_cls()
    configure = getattr(strategy, "configure", None)
    if params and callable(configure):
        configure(params)
    return strategy
```

Keep the `_event_bonus` helper only if `SpeedOnlyBaseline`/`ProgressOnly` use it (they do not — they can stay as-is). `_event_bonus` is now unused; delete it.

- [ ] **Step 4: Run both fitness test files**

Run: `uv run --with pytest pytest tests/test_beginner_mix.py tests/test_build_strategy.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add pipeline/fitness.py tests/test_build_strategy.py
git commit -m "refactor: slim fitness registry to BeginnerMix + 2 baselines, plumb params into build_strategy

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 3: Add `strategy` field to `StrategyConfig`

**Files:**
- Modify: `pipeline/config.py:11-15` (`StrategyConfig`) and `pipeline/config.py:41-45` (`from_dict` strategies parsing)
- Test: `tests/test_config.py` (create)

- [ ] **Step 1: Write failing test**

Create `tests/test_config.py`:

```python
from pipeline.config import ExperimentConfig


def test_strategy_type_defaults_to_beginner_mix():
    cfg = ExperimentConfig.from_dict({"strategies": [{"name": "progress_first",
                                                      "params": {"rewards": {"progress": 50}}}]})
    s = cfg.strategies[0]
    assert s.name == "progress_first"
    assert s.strategy == "beginner_mix"
    assert s.params == {"rewards": {"progress": 50}}


def test_known_baseline_name_resolves_to_its_own_strategy():
    cfg = ExperimentConfig.from_dict({"strategies": [{"name": "speed_only_baseline"}]})
    assert cfg.strategies[0].strategy == "speed_only_baseline"


def test_explicit_strategy_field_wins():
    cfg = ExperimentConfig.from_dict({"strategies": [{"name": "weird", "strategy": "progress_only"}]})
    assert cfg.strategies[0].strategy == "progress_only"
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run --with pytest pytest tests/test_config.py -v`
Expected: FAIL — `StrategyConfig` has no `strategy` attribute.

- [ ] **Step 3: Implement**

Replace `StrategyConfig` in `pipeline/config.py`:

```python
_BASELINE_NAMES = {"speed_only_baseline", "progress_only"}


@dataclass
class StrategyConfig:
    name: str
    strategy: str = "beginner_mix"
    params: dict[str, Any] = field(default_factory=dict)


def _default_strategy_type(name: str) -> str:
    return name if name in _BASELINE_NAMES else "beginner_mix"
```

Replace the strategies list-comprehension inside `from_dict`:

```python
        strategies = [
            StrategyConfig(
                name=item["name"],
                strategy=item.get("strategy", _default_strategy_type(item["name"])),
                params=item.get("params", {}),
            )
            for item in data.get("strategies", [{"name": "speed_only_baseline"}])
        ]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run --with pytest pytest tests/test_config.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add pipeline/config.py tests/test_config.py
git commit -m "feat: add strategy type field to StrategyConfig

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 4: Thread strategy type + params through training and replay

**Files:**
- Modify: `pipeline/training.py` (`_evaluate_network`, `_render_payload`, `_train_strategy`, `run_experiment` manifest)
- Modify: `pipeline/replay.py:33`
- Test: `tests/test_training_params.py` (create)

- [ ] **Step 1: Write failing test (params reach the simulator and change scoring)**

Create `tests/test_training_params.py`:

```python
from pipeline.config import ExperimentConfig
from pipeline.training import _evaluate_network
from pipeline.nn import NeuralNetwork
import numpy as np


def _net():
    return NeuralNetwork.random([6, 6, 4], np.random.default_rng(0))


def test_evaluate_network_accepts_strategy_config_and_uses_params():
    cfg = ExperimentConfig.from_dict({
        "train_seeds": [101], "validation_seeds": [202],
        "generations": 1, "population_size": 2,
        "strategies": [{"name": "progress_first", "params": {"rewards": {"progress": 50}}}],
    })
    strat_cfg = cfg.strategies[0]
    fitness, summary, episodes = _evaluate_network(_net(), cfg.train_seeds, strat_cfg, cfg)
    assert isinstance(fitness, float)
    assert "avg_max_track_progress" in summary
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run --with pytest pytest tests/test_training_params.py -v`
Expected: FAIL — `_evaluate_network` currently takes `strategy_name: str`, not a `StrategyConfig`.

- [ ] **Step 3: Update `_evaluate_network` signature and body**

In `pipeline/training.py`, change `_evaluate_network` to take `strategy_config: StrategyConfig` instead of `strategy_name: str`. Its first line becomes:

```python
def _evaluate_network(
    network: NeuralNetwork,
    seeds: list[int],
    strategy_config: StrategyConfig,
    config: ExperimentConfig,
) -> tuple[float, dict[str, float], list[dict[str, Any]]]:
    strategy = build_strategy(strategy_config.strategy, strategy_config.params)
```

- [ ] **Step 4: Update `_render_payload` and all call sites**

In `pipeline/training.py`:
- `_render_payload`: change param `strategy_name: str` → `strategy_config: StrategyConfig`, and its `build_strategy(strategy_name)` call → `build_strategy(strategy_config.strategy, strategy_config.params)`.
- In `_train_strategy`, the two `_evaluate_network(...)` calls currently pass `strategy_name=strategy_config.name`; change them to pass `strategy_config=strategy_config`.
- The `_render_payload(...)` call inside the progress-queue block: pass `strategy_config=strategy_config` instead of `strategy_name=strategy_config.name`.
- In the `best_metadata` dict, add two keys:

```python
                    "strategy": strategy_config.strategy,
                    "strategy_params": strategy_config.params,
```

(keep the existing `"strategy_name": strategy_config.name` key for the folder/label.)

- [ ] **Step 5: Add git commit to the manifest**

At the top of `pipeline/training.py`, add:

```python
import subprocess
```

Add this helper above `run_experiment`:

```python
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
```

In `run_experiment`, add to the `manifest` dict:

```python
        "git_commit": _git_commit(),
```

- [ ] **Step 6: Update replay.py**

In `pipeline/replay.py`, replace line 33:

```python
    strategy = build_strategy(
        metadata.get("strategy", metadata.get("strategy_name", "speed_only_baseline")),
        metadata.get("strategy_params"),
    )
```

- [ ] **Step 7: Run tests + a smoke run**

Run: `uv run --with pytest pytest tests/ -v`
Expected: all PASS.

Then verify an end-to-end run still works (uses the preset config built in Task 5; if running before Task 5, temporarily craft a 1-strategy inline config). Defer the full smoke to Task 5 Step 4.

- [ ] **Step 8: Commit**

```bash
git add pipeline/training.py pipeline/replay.py tests/test_training_params.py
git commit -m "feat: thread strategy type+params through training and replay, record git commit in manifest

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 5: Starter presets config

**Files:**
- Create: `configs/presets/starter_presets.json`

- [ ] **Step 1: Write the presets config**

Create `configs/presets/starter_presets.json`:

```json
{
  "run_name": "beginner_mix_presets",
  "output_dir": "artifacts/runs",
  "architecture": [6, 6, 4],
  "population_size": 20,
  "generations": 30,
  "mutation_rate": 90,
  "train_seeds": [101],
  "validation_seeds": [202, 203, 204],
  "time_limit_seconds": 30.0,
  "fps": 30,
  "parallel_workers": 6,
  "master_seed": 1234,
  "retry_generation": 15,
  "retry_min_avg_max_track_progress": 0.2,
  "max_seed_retries": 1,
  "track_cell_size": 120,
  "track_half_width": 34.0,
  "strategies": [
    { "name": "progress_first", "strategy": "beginner_mix",
      "params": { "rewards": {"progress": 50, "speed": 20, "alignment": 30},
                  "penalties": {"stall": 60, "crash": 80} } },
    { "name": "speed_first", "strategy": "beginner_mix",
      "params": { "rewards": {"speed": 50, "alignment": 30, "progress": 20},
                  "penalties": {"crash": 80, "time": 30} } },
    { "name": "stable_generalist", "strategy": "beginner_mix",
      "params": { "rewards": {"progress": 35, "speed": 20, "centered": 15, "alignment": 15, "safety": 15},
                  "penalties": {"stall": 50, "spin": 50, "crash": 80} } },
    { "name": "safe_centerline", "strategy": "beginner_mix",
      "params": { "rewards": {"progress": 30, "safety": 30, "centered": 25, "alignment": 15},
                  "penalties": {"crash": 90, "spin": 40} } },
    { "name": "anti_wrong_way", "strategy": "beginner_mix",
      "params": { "rewards": {"progress": 40, "alignment": 35, "speed": 25},
                  "penalties": {"wrong_way": 60, "stall": 50, "crash": 80} } },
    { "name": "speed_only_baseline", "strategy": "speed_only_baseline" },
    { "name": "progress_only", "strategy": "progress_only" }
  ]
}
```

- [ ] **Step 2: Create a fast smoke variant for verification**

Create `configs/presets/smoke.json` (tiny, for CI/sanity — 2 strategies, 2 generations):

```json
{
  "run_name": "beginner_mix_smoke",
  "output_dir": "artifacts/runs",
  "architecture": [6, 6, 4],
  "population_size": 6,
  "generations": 2,
  "mutation_rate": 90,
  "train_seeds": [101],
  "validation_seeds": [202],
  "time_limit_seconds": 10.0,
  "fps": 30,
  "parallel_workers": 2,
  "master_seed": 1234,
  "max_seed_retries": 0,
  "track_cell_size": 120,
  "track_half_width": 34.0,
  "strategies": [
    { "name": "progress_first", "strategy": "beginner_mix",
      "params": { "rewards": {"progress": 50, "speed": 20}, "penalties": {"stall": 60, "crash": 80} } },
    { "name": "speed_only_baseline", "strategy": "speed_only_baseline" }
  ]
}
```

- [ ] **Step 3: Run the smoke config end-to-end**

Run: `uv run python -m pipeline.run_experiment --config configs/presets/smoke.json`
Expected: prints a `artifacts/runs/<timestamp>_beginner_mix_smoke` path; that dir contains `summary.csv`, `summary.json`, `manifest.json` (with a `git_commit` field), and `strategies/progress_first/best_model.npz`.

- [ ] **Step 4: Verify replay works with saved params**

Run: `uv run python -m pipeline.replay --model artifacts/runs/<that_run>/strategies/progress_first/best_model.npz --seed 202`
Expected: writes an `.svg` + `.json` under `artifacts/replays/` without error (confirms `strategy_params` round-trip through metadata).

- [ ] **Step 5: Commit**

```bash
git add configs/presets/
git commit -m "feat: add BeginnerMix starter presets + smoke config

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 6: `export.py` — final_goal model export + template promotion

**Files:**
- Create: `pipeline/export.py`
- Test: `tests/test_export.py` (create)

- [ ] **Step 1: Write failing test for final_goal export shape**

Create `tests/test_export.py`:

```python
import json
import numpy as np

from pipeline.nn import NeuralNetwork
from pipeline.storage import save_model
from pipeline.export import export_final_goal_model, promote_template


def test_export_final_goal_model_has_correct_flattened_shapes(tmp_path):
    net = NeuralNetwork.random([6, 6, 4], np.random.default_rng(0))
    npz = tmp_path / "best_model.npz"
    save_model(npz, net, {"strategy": "beginner_mix"})
    model = export_final_goal_model(npz, group_id="7", username="alice")
    assert model["group_id"] == "7"
    assert model["username"] == "alice"
    assert [len(model["weights"][0]), len(model["weights"][1])] == [36, 24]
    assert [len(model["biases"][0]), len(model["biases"][1])] == [6, 4]
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run --with pytest pytest tests/test_export.py::test_export_final_goal_model_has_correct_flattened_shapes -v`
Expected: FAIL — `pipeline.export` does not exist.

- [ ] **Step 3: Implement `export.py`**

Create `pipeline/export.py`:

```python
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
    shutil.cop2 = shutil.copy2  # noqa: keep explicit
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
```

Note: remove the stray `shutil.copy2 = shutil.copy2` line — it is a typo guard; the real call is the `shutil.copy2(...)` below it. Final code should contain only the single `shutil.copy2(strat_dir / "best_model.npz", out_dir / "best_model.npz")` call.

- [ ] **Step 4: Add a promote round-trip test**

Append to `tests/test_export.py`:

```python
def test_promote_template_writes_five_files(tmp_path):
    # Build a minimal fake run dir.
    run = tmp_path / "run"
    strat = run / "strategies" / "progress_first"
    strat.mkdir(parents=True)
    (run / "manifest.json").write_text(json.dumps({"run_id": "r1", "git_commit": "abc",
        "architecture": [6, 6, 4], "master_seed": 1234}), encoding="utf-8")
    (strat / "validation.json").write_text(json.dumps({"finish_count": 1,
        "avg_finish_time": 12.3, "avg_max_track_progress": 1.0,
        "avg_collision_count": 0, "avg_stall_time": 0, "avg_spin_time": 0}), encoding="utf-8")
    net = NeuralNetwork.random([6, 6, 4], np.random.default_rng(1))
    save_model(strat / "best_model.npz", net,
               {"strategy": "beginner_mix",
                "strategy_params": {"rewards": {"progress": 50}, "penalties": {"crash": 80}}})

    out = promote_template(run, "progress_first", "progress_first_v1",
                           templates_root=tmp_path / "templates")
    for fname in ("recipe.json", "reproduce.json", "result.json", "best_model.npz", "model.json"):
        assert (out / fname).exists()
    index = json.loads((tmp_path / "templates" / "index.json").read_text(encoding="utf-8"))
    assert index["templates"][0]["name"] == "progress_first_v1"
```

- [ ] **Step 5: Run the export tests**

Run: `uv run --with pytest pytest tests/test_export.py -v`
Expected: both tests PASS.

- [ ] **Step 6: Commit**

```bash
git add pipeline/export.py tests/test_export.py
git commit -m "feat: add final_goal model export and template promotion

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 7: `tune.py` — coordinate-search auto-tune

**Files:**
- Create: `pipeline/tune.py`
- Test: `tests/test_tune.py` (create)

- [ ] **Step 1: Write failing test for neighbor generation**

Create `tests/test_tune.py`:

```python
from pipeline.tune import neighbor_recipes


def test_neighbors_perturb_each_active_slider_up_and_down_clamped():
    base = {"rewards": {"progress": 50, "speed": 90}, "penalties": {"crash": 100}}
    neighbors = neighbor_recipes(base, step=15)
    labels = {label for label, _ in neighbors}
    # progress: 50±15 -> 65 and 35 ; speed: 90+15 clamps to 100, 90-15=75 ;
    # crash: 100+15 clamps to 100 (dropped as duplicate of base), 100-15=85
    assert "progress_up" in labels and "progress_down" in labels
    recipes = dict(neighbors)
    assert recipes["speed_up"]["rewards"]["speed"] == 100.0
    assert recipes["speed_down"]["rewards"]["speed"] == 75.0
    assert recipes["crash_down"]["penalties"]["crash"] == 85.0


def test_neighbors_are_deterministic():
    base = {"rewards": {"progress": 50}, "penalties": {"stall": 40}}
    assert neighbor_recipes(base, step=10) == neighbor_recipes(base, step=10)
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run --with pytest pytest tests/test_tune.py -v`
Expected: FAIL — `pipeline.tune` does not exist.

- [ ] **Step 3: Implement `tune.py`**

Create `pipeline/tune.py`:

```python
from __future__ import annotations

import argparse
import copy
import json
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
    rewards first (insertion order), then penalties, each as <name>_up, <name>_down.
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


def run_round(config: ExperimentConfig, step: float) -> ExperimentConfig:
    """Build a config whose strategies are the base plus its neighbors, then run it."""
    base = _base_strategy(config)
    strategies = [StrategyConfig(name="base", strategy="beginner_mix", params=base.params)]
    for label, recipe in neighbor_recipes(base.params, step):
        strategies.append(StrategyConfig(name=label, strategy="beginner_mix", params=recipe))
    round_config = copy.replace(config, strategies=strategies) if hasattr(copy, "replace") else None
    if round_config is None:
        from dataclasses import replace
        round_config = replace(config, strategies=strategies)
    run_dir = run_experiment(round_config)
    return _pick_winner(run_dir, config)


def _rank_tuple(row: dict) -> tuple:
    finish = int(row["validation_finish_count"])
    finish_time = row["avg_finish_time"]
    time_score = -(float(finish_time) if finish_time not in (None, "") else 1e9)
    return (finish, time_score, float(row["avg_max_track_progress"]))


def _pick_winner(run_dir: Path, config: ExperimentConfig) -> ExperimentConfig:
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    best = max(summary["strategies"], key=lambda r: _rank_tuple_from_result(r))
    from dataclasses import replace
    winner_params = best["best_validation"]  # placeholder; replaced below
    # Re-read params from the run's manifest strategies by name.
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    name_to_params = {s["name"]: s["params"] for s in manifest["strategies"]}
    params = name_to_params[best["strategy_name"]]
    return replace(config, strategies=[StrategyConfig(name="base", strategy="beginner_mix", params=params)])


def _rank_tuple_from_result(result: dict) -> tuple:
    v = result["best_validation"]
    finish = int(v["finish_count"])
    ft = v["avg_finish_time"]
    time_score = -(float(ft) if ft is not None else 1e9)
    return (finish, time_score, float(v["avg_max_track_progress"]))


def main() -> None:
    parser = argparse.ArgumentParser(description="Coordinate-search auto-tune for BeginnerMix recipes.")
    parser.add_argument("--base-config", required=True)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--step", type=float, default=15.0)
    args = parser.parse_args()

    config = ExperimentConfig.from_path(args.base_config)
    for round_idx in range(1, args.rounds + 1):
        config = run_round(config, args.step)
        winner = _base_strategy(config)
        print(f"round {round_idx} winner params: {json.dumps(winner.params)}")


if __name__ == "__main__":
    main()
```

Simplify `run_round` to drop the dead `copy.replace` probe — use `from dataclasses import replace` at module top and call `replace(config, strategies=strategies)` directly. Likewise delete the unused `_rank_tuple`/`winner_params` placeholder lines in `_pick_winner`; keep only the manifest-based param lookup and `_rank_tuple_from_result`. Final `_pick_winner`:

```python
def _pick_winner(run_dir: Path, config: ExperimentConfig) -> ExperimentConfig:
    from dataclasses import replace
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    best = max(summary["strategies"], key=_rank_tuple_from_result)
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    name_to_params = {s["name"]: s["params"] for s in manifest["strategies"]}
    params = name_to_params[best["strategy_name"]]
    return replace(config, strategies=[StrategyConfig(name="base", strategy="beginner_mix", params=params)])
```

- [ ] **Step 4: Run the neighbor tests**

Run: `uv run --with pytest pytest tests/test_tune.py -v`
Expected: both tests PASS.

- [ ] **Step 5: Integration smoke for the tuner**

Create `configs/tune/smoke_base.json`:

```json
{
  "run_name": "tune_smoke",
  "output_dir": "artifacts/runs",
  "architecture": [6, 6, 4],
  "population_size": 6,
  "generations": 2,
  "train_seeds": [101],
  "validation_seeds": [202],
  "time_limit_seconds": 10.0,
  "fps": 30,
  "parallel_workers": 3,
  "master_seed": 1234,
  "max_seed_retries": 0,
  "strategies": [
    { "name": "base", "strategy": "beginner_mix",
      "params": { "rewards": {"progress": 50, "speed": 30}, "penalties": {"crash": 80} } }
  ]
}
```

Run: `uv run python -m pipeline.tune --base-config configs/tune/smoke_base.json --rounds 1 --step 15`
Expected: prints `round 1 winner params: {...}` without error.

- [ ] **Step 6: Commit**

```bash
git add pipeline/tune.py tests/test_tune.py configs/tune/
git commit -m "feat: add coordinate-search auto-tune driver

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 8: Restructure repo + archive legacy + gitignore artifacts

**Files:**
- Create: `archive/` (move targets)
- Modify: `.gitignore`
- Move: old runs, configs, tmp_configs, legacy game files

- [ ] **Step 1: Create archive layout and move old artifacts/configs**

```bash
mkdir -p archive/runs archive/configs archive/legacy_game
git mv artifacts/runs archive/runs/old_runs 2>/dev/null || mv artifacts/runs archive/runs/old_runs
git mv configs/experiment_architecture_ablation_6124.json archive/configs/ 2>/dev/null || true
git mv configs/experiment_architecture_ablation_664.json archive/configs/ 2>/dev/null || true
git mv configs/experiment_architecture_ablation_684.json archive/configs/ 2>/dev/null || true
git mv configs/experiment_architecture_ablation_6884.json archive/configs/ 2>/dev/null || true
git mv configs/experiment_baseline.json archive/configs/ 2>/dev/null || true
git mv configs/experiment_focused.json archive/configs/ 2>/dev/null || true
git mv tmp_configs archive/configs/tmp_configs 2>/dev/null || mv tmp_configs archive/configs/tmp_configs
```

- [ ] **Step 2: Move legacy game files**

```bash
git mv nnCarGame.py archive/legacy_game/ 2>/dev/null || mv nnCarGame.py archive/legacy_game/
git mv mapGen.py archive/legacy_game/ 2>/dev/null || mv mapGen.py archive/legacy_game/
git mv Images archive/legacy_game/Images 2>/dev/null || mv Images archive/legacy_game/Images
git mv bg4.png archive/legacy_game/ 2>/dev/null || mv bg4.png archive/legacy_game/
git mv bg7.png archive/legacy_game/ 2>/dev/null || mv bg7.png archive/legacy_game/
git mv randomGeneratedTrackBack.png archive/legacy_game/ 2>/dev/null || mv randomGeneratedTrackBack.png archive/legacy_game/
git mv randomGeneratedTrackFront.png archive/legacy_game/ 2>/dev/null || mv randomGeneratedTrackFront.png archive/legacy_game/
```

- [ ] **Step 3: Update `.gitignore` to treat run outputs as ephemeral**

Append to `.gitignore`:

```gitignore
# Ephemeral experiment outputs (templates/ is the committed deliverable)
artifacts/runs/
artifacts/replays/

# Keep archived history out of normal diffs but in the tree
# (archive/ is committed once, rarely touched)
```

- [ ] **Step 4: Confirm core imports + tests still pass after the move**

Run: `uv run --with pytest pytest tests/ -v`
Expected: all PASS (nothing in `pipeline/` or `tests/` referenced the moved files).

Run: `uv run python -c "import pipeline.training, pipeline.tune, pipeline.export"`
Expected: no output, exit 0.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "chore: restructure repo, archive legacy game + old runs/configs, gitignore run outputs

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 9: Colab notebook

**Files:**
- Create: `notebooks/run_experiments.ipynb`

- [ ] **Step 1: Author the notebook**

Create `notebooks/run_experiments.ipynb` with these cells (use the NotebookEdit tool or write valid `.ipynb` JSON). Cell contents:

**Cell 1 (markdown):**
```markdown
# NNCars BeginnerMix — 自動化實驗 (Colab)
Stage 1 presets → Stage 2 auto-tune → promote templates。
填入你自己的 repo URL，執行各 cell。
```

**Cell 2 (code) — config + clone:**
```python
REPO_URL = "https://github.com/sshuang610/NNCar_experiment.git"  # 你的新 repo
REPO_BRANCH = "main"
import os, subprocess, sys
REPO_DIR = REPO_URL.rstrip("/").split("/")[-1].replace(".git", "")
if not os.path.isdir(REPO_DIR):
    subprocess.run(["git", "clone", "--branch", REPO_BRANCH, REPO_URL], check=True)
os.chdir(REPO_DIR)
print("cwd:", os.getcwd())
```

**Cell 3 (code) — install deps (numpy only for core):**
```python
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "numpy", "pandas"], check=True)
```

**Cell 4 (code) — Stage 1 presets:**
```python
import subprocess
out = subprocess.run([sys.executable, "-m", "pipeline.run_experiment",
                      "--config", "configs/presets/starter_presets.json"],
                     capture_output=True, text=True, check=True)
RUN_DIR = out.stdout.strip().splitlines()[-1]
print("run dir:", RUN_DIR)
```

**Cell 5 (code) — show summary table:**
```python
import pandas as pd
df = pd.read_csv(f"{RUN_DIR}/summary.csv")
df.sort_values(["validation_finish_count", "avg_finish_time", "avg_max_track_progress"],
               ascending=[False, True, False])
```

**Cell 6 (code) — Stage 2 auto-tune around the winner:**
```python
import json
summary = json.load(open(f"{RUN_DIR}/summary.json"))
manifest = json.load(open(f"{RUN_DIR}/manifest.json"))
def rank(r):
    v = r["best_validation"]; ft = v["avg_finish_time"]
    return (v["finish_count"], -(ft if ft is not None else 1e9), v["avg_max_track_progress"])
winner = max([s for s in summary["strategies"]
              if s["strategy_name"] not in ("speed_only_baseline", "progress_only")], key=rank)
params = {s["name"]: s["params"] for s in manifest["strategies"]}[winner["strategy_name"]]
base_cfg = {**{k: manifest[k] for k in ("architecture","population_size","generations","mutation_rate",
              "train_seeds","validation_seeds","time_limit_seconds","fps","master_seed")},
            "run_name": "tune", "output_dir": "artifacts/runs", "max_seed_retries": 0,
            "parallel_workers": 6,
            "strategies": [{"name": "base", "strategy": "beginner_mix", "params": params}]}
os.makedirs("configs/tune", exist_ok=True)
json.dump(base_cfg, open("configs/tune/auto_base.json", "w"), indent=2)
subprocess.run([sys.executable, "-m", "pipeline.tune",
                "--base-config", "configs/tune/auto_base.json", "--rounds", "2", "--step", "15"],
               check=True)
```

**Cell 7 (code) — promote winner to a template:**
```python
# Re-run the final base recipe once to get a clean run dir, then promote it.
final = subprocess.run([sys.executable, "-m", "pipeline.run_experiment",
                        "--config", "configs/tune/auto_base.json"],
                       capture_output=True, text=True, check=True)
FINAL_DIR = final.stdout.strip().splitlines()[-1]
from pipeline.export import promote_template
out_dir = promote_template(FINAL_DIR, "base", "winner_v1", group_id="0", username="ga_research")
print("template at:", out_dir)
print(open(f"{out_dir}/result.json").read())
```

**Cell 8 (code) — zip templates for download:**
```python
import shutil
shutil.make_archive("templates_export", "zip", "templates")
try:
    from google.colab import files; files.download("templates_export.zip")
except Exception:
    print("templates_export.zip ready in working dir")
```

- [ ] **Step 2: Validate the notebook JSON is well-formed**

Run: `uv run python -c "import json,nbformat" 2>/dev/null || uv run --with nbformat python -c "import nbformat; nbformat.read('notebooks/run_experiments.ipynb', as_version=4); print('ok')"`
Expected: prints `ok` (notebook parses).

- [ ] **Step 3: Commit**

```bash
git add notebooks/run_experiments.ipynb
git commit -m "feat: add Colab notebook driving presets -> auto-tune -> template export

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 10: Docs — BeginnerMix how-to + README refresh

**Files:**
- Create: `docs/beginner_mix.md`
- Modify: `README.md`

- [ ] **Step 1: Write `docs/beginner_mix.md`**

Create `docs/beginner_mix.md` covering: the 10 blocks (rewards `speed/progress/centered/alignment/safety`, penalties `stall/spin/wrong_way/time/crash`), the §9 scoring formula, the `{rewards, penalties}` config shape, the symptom→slider tuning table (port §6 of the design doc), and the reproduce-from-template workflow (`reproduce.json` → re-run same config/seed/commit).

- [ ] **Step 2: Refresh `README.md`**

Replace the "目前結論 / 多元策略" sections (which describe the deleted strategies) with: the new BeginnerMix-based workflow, the two-stage presets→auto-tune commands, the `templates/` deliverable layout, and a pointer to `docs/beginner_mix.md` and the Colab notebook. Keep the install (`uv sync`) and replay sections.

Concretely, the new "跑實驗" section commands:

```bash
# Stage 1: presets
uv run python -m pipeline.run_experiment --config configs/presets/starter_presets.json
# Stage 2: auto-tune around a winner
uv run python -m pipeline.tune --base-config configs/tune/auto_base.json --rounds 2 --step 15
```

- [ ] **Step 3: Commit**

```bash
git add docs/beginner_mix.md README.md
git commit -m "docs: BeginnerMix how-to + README refresh for new workflow

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 11: End-to-end acceptance — produce the first templates

**Files:**
- Create: `templates/<name>/...` (generated artifacts, committed)

- [ ] **Step 1: Run Stage 1 presets (real settings)**

Run: `uv run python -m pipeline.run_experiment --config configs/presets/starter_presets.json`
Expected: prints a run dir; `summary.csv` ranks the 7 strategies. Note the top non-baseline strategy name.

- [ ] **Step 2: Auto-tune the winner**

Build `configs/tune/auto_base.json` from the winner's params (copy from the run's `manifest.json`), then:
Run: `uv run python -m pipeline.tune --base-config configs/tune/auto_base.json --rounds 2 --step 15`
Expected: prints improving/stable winner params per round.

- [ ] **Step 3: Promote 1–2 winners into `templates/`**

Run a final clean run of `auto_base.json`, then:
```bash
uv run python -c "from pipeline.export import promote_template; promote_template('<FINAL_RUN_DIR>', 'base', 'progress_first_v1', group_id='0', username='ga_research')"
```
Expected: `templates/progress_first_v1/` has all 5 files; `templates/index.json` lists it.

- [ ] **Step 4: Verify the template model.json is final_goal-shaped**

Run: `uv run python -c "import json; m=json.load(open('templates/progress_first_v1/model.json')); print([len(m['weights'][0]), len(m['weights'][1]), len(m['biases'][0]), len(m['biases'][1])])"`
Expected: `[36, 24, 6, 4]`.

- [ ] **Step 5: Commit the templates**

```bash
git add templates/
git commit -m "feat: add first reproducible BeginnerMix templates

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 12: Publish to the new repo

**Files:** none (git remote + push)

**Target:** `https://github.com/sshuang610/NNCar_experiment`

- [ ] **Step 1: Point a remote at the new repo**

The current `origin` points at `NCTU-CS-Camp/NNCars-Fitness-Experiments`. Add the new repo as a separate remote so the org repo is untouched:

```bash
git remote add newrepo https://github.com/sshuang610/NNCar_experiment.git
git remote -v
```

- [ ] **Step 2: Push the branch to the new repo**

```bash
git push newrepo HEAD:main
```

Expected: all commits (code, configs, templates, notebook, docs) land on `sshuang610/NNCar_experiment` `main`. If the push is rejected for auth, the user supplies a token / does the push; if the remote already has history, push to a branch (`HEAD:beginner-mix`) and open a PR instead.

- [ ] **Step 3: Confirm the notebook clones the published repo**

The notebook `REPO_URL` already defaults to `https://github.com/sshuang610/NNCar_experiment.git`. Sanity-check the repo is reachable (public) or that Colab will have a token for it.

---

## Self-Review notes (for the implementer)

- **Spec coverage:** §2 fitness → Task 1; §2.4 slim + §3 plumbing → Tasks 2–4; §4 presets/auto-tune → Tasks 5,7; §5 templates → Task 6,11; §6 Colab → Task 9; §7 restructure → Task 8; §8 tests → embedded in Tasks 1–7; §10 acceptance → Task 11.
- **Type consistency:** `build_strategy(strategy_type, params)`, `StrategyConfig(name, strategy, params)`, `_evaluate_network(network, seeds, strategy_config, config)`, `promote_template(run_dir, strategy_name, template_name, ...)`, `neighbor_recipes(base, step)` are used identically across tasks.
- **Known cleanups flagged inline:** Task 6 Step 3 (remove the `shutil.copy2 = shutil.copy2` typo guard) and Task 7 Step 3 (drop the `copy.replace` probe + dead placeholder in `_pick_winner`) — implement the simplified final versions shown.
```
