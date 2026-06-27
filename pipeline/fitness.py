from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class StepContext:
    velocity: float
    progress_delta: float
    progress_ratio: float
    center_offset: float
    normalized_center_offset: float
    heading_alignment: float
    front_clearance: float
    min_clearance: float
    side_clearance_balance: float
    turn_amount: float
    collided: bool
    finished: bool
    is_stalled: bool
    is_spinning: bool
    frame: int
    time_elapsed: float


class FitnessStrategy:
    name = "base"

    def reset(self) -> None:
        return

    def score_step(self, context: StepContext) -> float:
        raise NotImplementedError


B = 10.0
CRASH_SECONDS = 15.0
B_CRASH = B * CRASH_SECONDS          # 150.0
FINISH_SECONDS = 300.0
FINISH_BONUS = B * FINISH_SECONDS    # 3000.0
CHECKPOINT_SECONDS = 5.0
B_CHECKPOINT = B * CHECKPOINT_SECONDS      # 50.0
DEFAULT_CHECKPOINTS = (0.2, 0.4, 0.6, 0.8, 0.95)

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
        self._checkpoints = list(DEFAULT_CHECKPOINTS)
        self._next_checkpoint = 0

    def reset(self) -> None:
        self._next_checkpoint = 0

    def configure(self, params: dict) -> None:
        rewards, penalties = _split_params(params)
        self.rewards = {
            k: float(v) for k, v in rewards.items()
            if k in REWARD_BLOCKS or k == "checkpoint"
        }
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
        per_frame = {k: v for k, v in self.rewards.items() if k in REWARD_BLOCKS}
        weight_sum = sum(per_frame.values())
        if weight_sum > 0.0:
            factors = self._reward_factors(context)
            weighted = sum(per_frame[k] * factors[k] for k in per_frame)
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

        cp_weight = self.rewards.get("checkpoint", 0.0)
        if cp_weight > 0.0:
            while (
                self._next_checkpoint < len(self._checkpoints)
                and context.progress_ratio >= self._checkpoints[self._next_checkpoint]
            ):
                step += (cp_weight / 100.0) * B_CHECKPOINT
                self._next_checkpoint += 1
        return step


class SpeedOnlyBaseline(FitnessStrategy):
    name = "speed_only_baseline"

    def score_step(self, context: StepContext) -> float:
        return context.velocity


class ProgressOnly(FitnessStrategy):
    name = "progress_only"

    def score_step(self, context: StepContext) -> float:
        return context.progress_delta


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
