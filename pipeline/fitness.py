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


def _event_bonus(context: StepContext, finish_bonus: float, collision_penalty: float) -> float:
    if context.finished:
        return finish_bonus
    if context.collided:
        return -collision_penalty
    return 0.0


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


class SpeedOnlyBaseline(FitnessStrategy):
    name = "speed_only_baseline"

    def score_step(self, context: StepContext) -> float:
        return context.velocity


class ProgressOnly(FitnessStrategy):
    name = "progress_only"

    def score_step(self, context: StepContext) -> float:
        return context.progress_delta


class SpeedProgress(FitnessStrategy):
    name = "speed_progress"

    def score_step(self, context: StepContext) -> float:
        return (context.progress_delta * 2.0) + (context.velocity * 0.5)


class ForwardMomentum(FitnessStrategy):
    name = "forward_momentum"

    def score_step(self, context: StepContext) -> float:
        aligned_speed = context.velocity * max(0.0, context.heading_alignment)
        return (context.progress_delta * 4.0) + aligned_speed - (20.0 if context.collided else 0.0)


class CenterlineProgress(FitnessStrategy):
    name = "centerline_progress"

    def score_step(self, context: StepContext) -> float:
        center_penalty = context.normalized_center_offset * 2.5
        return (
            (context.progress_delta * 4.0)
            + (context.heading_alignment * 0.8)
            - center_penalty
            + _event_bonus(context, finish_bonus=2500.0, collision_penalty=350.0)
        )


class SafeFast(FitnessStrategy):
    name = "safe_fast"

    def score_step(self, context: StepContext) -> float:
        clearance_bonus = min(context.min_clearance, 90.0) * 0.08
        return (
            (context.progress_delta * 3.0)
            + (context.velocity * max(0.2, context.heading_alignment))
            + clearance_bonus
            - (context.normalized_center_offset * 1.5)
            + _event_bonus(context, finish_bonus=3500.0, collision_penalty=500.0)
        )


class RaceMetricProxy(FitnessStrategy):
    name = "race_metric_proxy"

    def score_step(self, context: StepContext) -> float:
        time_penalty = context.time_elapsed * 0.03
        behavior_penalty = 5.0 if context.is_stalled else 0.0
        behavior_penalty += 4.0 if context.is_spinning else 0.0
        return (
            (context.progress_delta * 6.0)
            + (context.velocity * 0.4)
            + (context.progress_ratio * 0.5)
            - time_penalty
            - behavior_penalty
            + _event_bonus(context, finish_bonus=10000.0, collision_penalty=700.0)
        )


class RaceMetricProgressFocus(FitnessStrategy):
    name = "race_metric_progress_focus"

    def score_step(self, context: StepContext) -> float:
        behavior_penalty = 5.0 if context.is_stalled else 0.0
        behavior_penalty += 4.0 if context.is_spinning else 0.0
        return (
            (context.progress_delta * 7.5)
            + (context.velocity * 0.4)
            + (context.progress_ratio * 0.5)
            - (context.time_elapsed * 0.03)
            - behavior_penalty
            + _event_bonus(context, finish_bonus=10000.0, collision_penalty=700.0)
        )


class RaceMetricSpeedFocus(FitnessStrategy):
    name = "race_metric_speed_focus"

    def score_step(self, context: StepContext) -> float:
        behavior_penalty = 5.0 if context.is_stalled else 0.0
        behavior_penalty += 4.0 if context.is_spinning else 0.0
        return (
            (context.progress_delta * 6.0)
            + (context.velocity * 0.5)
            + (context.progress_ratio * 0.5)
            - (context.time_elapsed * 0.03)
            - behavior_penalty
            + _event_bonus(context, finish_bonus=10000.0, collision_penalty=700.0)
        )


class RaceMetricLinearTime(FitnessStrategy):
    name = "race_metric_linear_time"

    def score_step(self, context: StepContext) -> float:
        behavior_penalty = 5.0 if context.is_stalled else 0.0
        behavior_penalty += 4.0 if context.is_spinning else 0.0
        return (
            (context.progress_delta * 6.0)
            + (context.velocity * 0.4)
            + (context.progress_ratio * 0.5)
            - 0.45
            - behavior_penalty
            + _event_bonus(context, finish_bonus=10000.0, collision_penalty=700.0)
        )


class RaceMetricForwardEfficiency(FitnessStrategy):
    name = "race_metric_forward_efficiency"

    def score_step(self, context: StepContext) -> float:
        forward_ratio = min(1.0, context.progress_delta / max(context.velocity, 0.1))
        effective_speed = context.velocity * (0.25 + (0.75 * forward_ratio))
        behavior_penalty = 5.0 if context.is_stalled else 0.0
        behavior_penalty += 4.0 if context.is_spinning else 0.0
        return (
            (context.progress_delta * 6.0)
            + (effective_speed * 0.4)
            + (context.progress_ratio * 0.5)
            - (context.time_elapsed * 0.03)
            - behavior_penalty
            + _event_bonus(context, finish_bonus=10000.0, collision_penalty=700.0)
        )


class RaceMetricCollisionGuard(FitnessStrategy):
    name = "race_metric_collision_guard"

    def score_step(self, context: StepContext) -> float:
        behavior_penalty = 5.0 if context.is_stalled else 0.0
        behavior_penalty += 4.0 if context.is_spinning else 0.0
        return (
            (context.progress_delta * 6.0)
            + (context.velocity * 0.4)
            + (context.progress_ratio * 0.5)
            - (context.time_elapsed * 0.03)
            - behavior_penalty
            + _event_bonus(context, finish_bonus=10000.0, collision_penalty=1400.0)
        )


class RaceMetricClearanceGuard(FitnessStrategy):
    name = "race_metric_clearance_guard"

    def score_step(self, context: StepContext) -> float:
        danger_penalty = max(0.0, 12.0 - context.min_clearance) * 0.15
        behavior_penalty = 5.0 if context.is_stalled else 0.0
        behavior_penalty += 4.0 if context.is_spinning else 0.0
        return (
            (context.progress_delta * 6.0)
            + (context.velocity * 0.4)
            + (context.progress_ratio * 0.5)
            - (context.time_elapsed * 0.03)
            - danger_penalty
            - behavior_penalty
            + _event_bonus(context, finish_bonus=10000.0, collision_penalty=700.0)
        )


class RaceMetricFinishReward(FitnessStrategy):
    name = "race_metric_finish_reward"

    def score_step(self, context: StepContext) -> float:
        behavior_penalty = 5.0 if context.is_stalled else 0.0
        behavior_penalty += 4.0 if context.is_spinning else 0.0
        return (
            (context.progress_delta * 6.0)
            + (context.velocity * 0.4)
            + (context.progress_ratio * 0.5)
            - (context.time_elapsed * 0.03)
            - behavior_penalty
            + _event_bonus(context, finish_bonus=20000.0, collision_penalty=700.0)
        )


class FrontierExplorer(FitnessStrategy):
    name = "frontier_explorer"

    def reset(self) -> None:
        self._max_progress_ratio = 0.0

    def score_step(self, context: StepContext) -> float:
        frontier_gain = max(0.0, context.progress_ratio - self._max_progress_ratio)
        self._max_progress_ratio = max(self._max_progress_ratio, context.progress_ratio)
        return (
            (frontier_gain * 30000.0)
            + (context.velocity * 0.15)
            - (5.0 if context.is_stalled else 0.0)
            - (4.0 if context.is_spinning else 0.0)
            + _event_bonus(context, finish_bonus=15000.0, collision_penalty=800.0)
        )


class RiskAdjustedPace(FitnessStrategy):
    name = "risk_adjusted_pace"

    def score_step(self, context: StepContext) -> float:
        clearance_factor = min(1.0, max(0.0, (context.front_clearance - 10.0) / 60.0))
        safe_speed = context.velocity * clearance_factor
        return (
            (context.progress_delta * 4.0)
            + (safe_speed * 0.8)
            - (6.0 if context.is_stalled else 0.0)
            - (5.0 if context.is_spinning else 0.0)
            + _event_bonus(context, finish_bonus=12000.0, collision_penalty=1200.0)
        )


class CenterlinePace(FitnessStrategy):
    name = "centerline_pace"

    def score_step(self, context: StepContext) -> float:
        center_confidence = max(0.0, 1.0 - context.normalized_center_offset)
        forward_alignment = max(0.0, context.heading_alignment)
        quality_speed = (
            context.velocity
            * forward_alignment
            * (0.25 + (0.75 * center_confidence))
        )
        return (
            (context.progress_delta * 4.0)
            + (quality_speed * 0.8)
            - (6.0 if context.is_stalled else 0.0)
            + _event_bonus(context, finish_bonus=12000.0, collision_penalty=1000.0)
        )


class TwoPhaseRacer(FitnessStrategy):
    name = "two_phase_racer"

    def score_step(self, context: StepContext) -> float:
        if context.progress_ratio < 0.45:
            clearance_bonus = min(context.min_clearance, 30.0) * 0.03
            pace_score = (
                (context.progress_delta * 5.0)
                + (context.velocity * 0.2)
                + clearance_bonus
            )
        else:
            pace_score = (
                (context.progress_delta * 7.0)
                + (context.velocity * 0.6)
                - 0.35
            )
        return (
            pace_score
            - (6.0 if context.is_stalled else 0.0)
            - (5.0 if context.is_spinning else 0.0)
            + _event_bonus(context, finish_bonus=15000.0, collision_penalty=1000.0)
        )


class SmoothControlPace(FitnessStrategy):
    name = "smooth_control_pace"

    def score_step(self, context: StepContext) -> float:
        excessive_turn = max(0.0, context.turn_amount - 2.5)
        return (
            (context.progress_delta * 5.0)
            + (context.velocity * 0.35)
            - (excessive_turn * 0.25)
            - (6.0 if context.is_stalled else 0.0)
            - (8.0 if context.is_spinning else 0.0)
            + _event_bonus(context, finish_bonus=12000.0, collision_penalty=900.0)
        )


class SparseOutcome(FitnessStrategy):
    name = "sparse_outcome"

    def score_step(self, context: StepContext) -> float:
        return (
            (context.progress_delta * 8.0)
            - 0.2
            - (3.0 if context.is_stalled else 0.0)
            + _event_bonus(context, finish_bonus=30000.0, collision_penalty=1000.0)
        )


class SensorBalancePace(FitnessStrategy):
    name = "sensor_balance_pace"

    def score_step(self, context: StepContext) -> float:
        front_space_bonus = min(context.front_clearance, 80.0) * 0.025
        side_imbalance_penalty = min(context.side_clearance_balance, 100.0) * 0.02
        return (
            (context.progress_delta * 4.5)
            + (context.velocity * 0.4)
            + front_space_bonus
            - side_imbalance_penalty
            - (6.0 if context.is_stalled else 0.0)
            + _event_bonus(context, finish_bonus=12000.0, collision_penalty=1000.0)
        )


class RaceMetricProxyV2(FitnessStrategy):
    name = "race_metric_proxy_v2"

    def score_step(self, context: StepContext) -> float:
        aligned_speed = context.velocity * max(0.0, context.heading_alignment)
        behavior_penalty = 6.0 if context.is_stalled else 0.0
        behavior_penalty += 5.0 if context.is_spinning else 0.0
        center_penalty = max(0.0, context.normalized_center_offset) * 0.6
        return (
            (context.progress_delta * 7.0)
            + (context.velocity * 0.25)
            + (aligned_speed * 0.45)
            + (context.progress_ratio * 1.0)
            - (context.time_elapsed * 0.035)
            - center_penalty
            - behavior_penalty
            + _event_bonus(context, finish_bonus=15000.0, collision_penalty=750.0)
        )


class RaceMetricProxyFinishPush(FitnessStrategy):
    name = "race_metric_proxy_finish_push"

    def score_step(self, context: StepContext) -> float:
        closing_bonus = 0.0
        if context.progress_ratio > 0.70:
            closing_bonus += (context.progress_ratio - 0.70) * 45.0
        if context.progress_ratio > 0.90:
            closing_bonus += (context.progress_ratio - 0.90) * 250.0
        aligned_speed = context.velocity * max(0.0, context.heading_alignment)
        return (
            (context.progress_delta * 7.0)
            + (aligned_speed * 0.55)
            + (context.velocity * 0.25)
            + (context.progress_ratio * 0.8)
            + closing_bonus
            - (context.time_elapsed * 0.035)
            - (8.0 if context.is_stalled else 0.0)
            - (6.0 if context.is_spinning else 0.0)
            + _event_bonus(context, finish_bonus=18000.0, collision_penalty=800.0)
        )


class RaceMetricProxyCheckpointPush(FitnessStrategy):
    name = "race_metric_proxy_checkpoint_push"

    def reset(self) -> None:
        self._next_checkpoint_idx = 0
        self._checkpoints = [0.20, 0.40, 0.60, 0.80, 0.92, 0.98]

    def score_step(self, context: StepContext) -> float:
        checkpoint_bonus = 0.0
        while (
            self._next_checkpoint_idx < len(self._checkpoints)
            and context.progress_ratio >= self._checkpoints[self._next_checkpoint_idx]
        ):
            checkpoint = self._checkpoints[self._next_checkpoint_idx]
            checkpoint_bonus += 250.0 + (checkpoint * 900.0)
            self._next_checkpoint_idx += 1
        behavior_penalty = 6.0 if context.is_stalled else 0.0
        behavior_penalty += 5.0 if context.is_spinning else 0.0
        return (
            (context.progress_delta * 6.5)
            + (context.velocity * 0.35)
            + (context.progress_ratio * 0.8)
            + checkpoint_bonus
            - (context.time_elapsed * 0.035)
            - behavior_penalty
            + _event_bonus(context, finish_bonus=17000.0, collision_penalty=800.0)
        )


class RaceMetricProxyAlignmentPush(FitnessStrategy):
    name = "race_metric_proxy_alignment_push"

    def score_step(self, context: StepContext) -> float:
        aligned_speed = context.velocity * context.heading_alignment
        wrong_way_penalty = 12.0 if context.heading_alignment < 0.0 else 0.0
        return (
            (context.progress_delta * 7.2)
            + (aligned_speed * 0.75)
            + (context.progress_ratio * 0.9)
            - wrong_way_penalty
            - (context.normalized_center_offset * 0.7)
            - (context.time_elapsed * 0.04)
            - (7.0 if context.is_stalled else 0.0)
            - (6.0 if context.is_spinning else 0.0)
            + _event_bonus(context, finish_bonus=16000.0, collision_penalty=850.0)
        )


class AntiStallProgress(FitnessStrategy):
    name = "anti_stall_progress"

    def score_step(self, context: StepContext) -> float:
        stall_penalty = 12.0 if context.is_stalled else 0.0
        spin_penalty = 6.0 if context.is_spinning else 0.0
        return (
            (context.progress_delta * 5.0)
            + (context.velocity * 0.7)
            - stall_penalty
            - spin_penalty
            + _event_bonus(context, finish_bonus=3000.0, collision_penalty=450.0)
        )


class SmoothProgress(FitnessStrategy):
    name = "smooth_progress"

    def score_step(self, context: StepContext) -> float:
        turn_penalty = context.turn_amount * 0.12
        return (
            (context.progress_delta * 4.5)
            + (context.heading_alignment * 1.2)
            - turn_penalty
            - (8.0 if context.is_spinning else 0.0)
            + _event_bonus(context, finish_bonus=2800.0, collision_penalty=400.0)
        )


class WallClearanceProgress(FitnessStrategy):
    name = "wall_clearance_progress"

    def score_step(self, context: StepContext) -> float:
        clearance = min(context.front_clearance, 120.0) * 0.04
        return (
            (context.progress_delta * 3.0)
            + clearance
            - (context.normalized_center_offset * 1.0)
            + _event_bonus(context, finish_bonus=2500.0, collision_penalty=550.0)
        )


class BalancedDriver(FitnessStrategy):
    name = "balanced_driver"

    def score_step(self, context: StepContext) -> float:
        penalties = context.normalized_center_offset * 2.0
        penalties += 5.0 if context.is_stalled else 0.0
        penalties += 5.0 if context.is_spinning else 0.0
        penalties += context.side_clearance_balance * 0.02
        return (
            (context.progress_delta * 4.0)
            + (context.velocity * 0.4)
            + (context.heading_alignment * 1.0)
            - penalties
            + _event_bonus(context, finish_bonus=4000.0, collision_penalty=500.0)
        )


class FinishSeeker(FitnessStrategy):
    name = "finish_seeker"

    def score_step(self, context: StepContext) -> float:
        finish_pressure = context.progress_ratio * context.progress_ratio * 3.0
        return (
            (context.progress_delta * 5.0)
            + finish_pressure
            - (context.time_elapsed * 0.02)
            + _event_bonus(context, finish_bonus=12000.0, collision_penalty=650.0)
        )


class RaceMetricSprint(FitnessStrategy):
    name = "race_metric_sprint"

    def score_step(self, context: StepContext) -> float:
        aligned_speed = context.velocity * max(0.0, context.heading_alignment)
        behavior_penalty = 10.0 if context.is_stalled else 0.0
        behavior_penalty += 8.0 if context.is_spinning else 0.0
        return (
            (context.progress_delta * 8.0)
            + (aligned_speed * 0.9)
            + (context.progress_ratio * 1.5)
            - (context.time_elapsed * 0.06)
            - behavior_penalty
            + _event_bonus(context, finish_bonus=20000.0, collision_penalty=900.0)
        )


class RaceMetricLateBonus(FitnessStrategy):
    name = "race_metric_late_bonus"

    def score_step(self, context: StepContext) -> float:
        late_bonus = 0.0
        if context.progress_ratio > 0.80:
            late_bonus = (context.progress_ratio - 0.80) * 40.0
        if context.progress_ratio > 0.93:
            late_bonus += (context.progress_ratio - 0.93) * 120.0
        behavior_penalty = 5.0 if context.is_stalled else 0.0
        behavior_penalty += 4.0 if context.is_spinning else 0.0
        return (
            (context.progress_delta * 6.0)
            + (context.velocity * 0.4)
            + (context.progress_ratio * 0.5)
            + late_bonus
            - (context.time_elapsed * 0.03)
            - behavior_penalty
            + _event_bonus(context, finish_bonus=12000.0, collision_penalty=700.0)
        )


class RaceMetricAligned(FitnessStrategy):
    name = "race_metric_aligned"

    def score_step(self, context: StepContext) -> float:
        aligned_speed = context.velocity * max(0.0, context.heading_alignment)
        behavior_penalty = 5.0 if context.is_stalled else 0.0
        behavior_penalty += 4.0 if context.is_spinning else 0.0
        return (
            (context.progress_delta * 6.0)
            + (context.velocity * 0.25)
            + (aligned_speed * 0.25)
            + (context.progress_ratio * 0.5)
            - (context.time_elapsed * 0.03)
            - behavior_penalty
            + _event_bonus(context, finish_bonus=11000.0, collision_penalty=700.0)
        )


class RaceMetricSoftCheckpoints(FitnessStrategy):
    name = "race_metric_soft_checkpoints"

    def reset(self) -> None:
        self._next_checkpoint_idx = 0
        self._checkpoints = [0.25, 0.50, 0.75, 0.90, 0.97]

    def score_step(self, context: StepContext) -> float:
        checkpoint_bonus = 0.0
        while (
            self._next_checkpoint_idx < len(self._checkpoints)
            and context.progress_ratio >= self._checkpoints[self._next_checkpoint_idx]
        ):
            checkpoint_bonus += 150.0 + (self._checkpoints[self._next_checkpoint_idx] * 450.0)
            self._next_checkpoint_idx += 1
        behavior_penalty = 5.0 if context.is_stalled else 0.0
        behavior_penalty += 4.0 if context.is_spinning else 0.0
        return (
            (context.progress_delta * 6.0)
            + (context.velocity * 0.4)
            + (context.progress_ratio * 0.5)
            + checkpoint_bonus
            - (context.time_elapsed * 0.03)
            - behavior_penalty
            + _event_bonus(context, finish_bonus=12000.0, collision_penalty=700.0)
        )


class RaceMetricNoStall(FitnessStrategy):
    name = "race_metric_no_stall"

    def score_step(self, context: StepContext) -> float:
        behavior_penalty = 9.0 if context.is_stalled else 0.0
        behavior_penalty += 5.0 if context.is_spinning else 0.0
        return (
            (context.progress_delta * 6.0)
            + (context.velocity * 0.45)
            + (context.progress_ratio * 0.5)
            - (context.time_elapsed * 0.03)
            - behavior_penalty
            + _event_bonus(context, finish_bonus=12000.0, collision_penalty=700.0)
        )


class RaceMetricFinishPressure(FitnessStrategy):
    name = "race_metric_finish_pressure"

    def score_step(self, context: StepContext) -> float:
        finish_pressure = context.progress_ratio * context.progress_ratio * 1.4
        if context.progress_ratio > 0.90:
            finish_pressure += (context.progress_ratio - 0.90) * 80.0
        behavior_penalty = 5.0 if context.is_stalled else 0.0
        behavior_penalty += 4.0 if context.is_spinning else 0.0
        return (
            (context.progress_delta * 6.0)
            + (context.velocity * 0.4)
            + finish_pressure
            - (context.time_elapsed * 0.03)
            - behavior_penalty
            + _event_bonus(context, finish_bonus=14000.0, collision_penalty=700.0)
        )


class LateProgressSprint(FitnessStrategy):
    name = "late_progress_sprint"

    def score_step(self, context: StepContext) -> float:
        late_multiplier = 5.0 + (context.progress_ratio * 8.0)
        aligned_speed = context.velocity * max(0.0, context.heading_alignment)
        return (
            (context.progress_delta * late_multiplier)
            + (aligned_speed * (0.3 + context.progress_ratio))
            - (context.normalized_center_offset * 1.0)
            - (12.0 if context.is_stalled else 0.0)
            - (10.0 if context.is_spinning else 0.0)
            + _event_bonus(context, finish_bonus=24000.0, collision_penalty=850.0)
        )


class CheckpointSprint(FitnessStrategy):
    name = "checkpoint_sprint"

    def reset(self) -> None:
        self._next_checkpoint_idx = 0
        self._checkpoints = [0.10, 0.25, 0.50, 0.75, 0.90, 0.97]

    def score_step(self, context: StepContext) -> float:
        checkpoint_bonus = 0.0
        while (
            self._next_checkpoint_idx < len(self._checkpoints)
            and context.progress_ratio >= self._checkpoints[self._next_checkpoint_idx]
        ):
            checkpoint = self._checkpoints[self._next_checkpoint_idx]
            checkpoint_bonus += 500.0 + (checkpoint * 2500.0)
            self._next_checkpoint_idx += 1
        return (
            (context.progress_delta * 6.5)
            + (context.velocity * max(0.1, context.heading_alignment) * 0.5)
            + checkpoint_bonus
            - (context.time_elapsed * 0.04)
            - (12.0 if context.is_stalled else 0.0)
            - (10.0 if context.is_spinning else 0.0)
            + _event_bonus(context, finish_bonus=26000.0, collision_penalty=800.0)
        )


class FinishLineHunter(FitnessStrategy):
    name = "finish_line_hunter"

    def score_step(self, context: StepContext) -> float:
        closing_bonus = 0.0
        if context.progress_ratio > 0.80:
            closing_bonus = (context.progress_ratio - 0.80) * 200.0
        if context.progress_ratio > 0.92:
            closing_bonus += (context.progress_ratio - 0.92) * 900.0
        return (
            (context.progress_delta * 7.0)
            + closing_bonus
            + (context.velocity * max(0.0, context.heading_alignment) * 0.6)
            - (context.time_elapsed * 0.03)
            - (15.0 if context.is_stalled else 0.0)
            - (8.0 if context.is_spinning else 0.0)
            + _event_bonus(context, finish_bonus=30000.0, collision_penalty=700.0)
        )


class TimeAttackAlignment(FitnessStrategy):
    name = "time_attack_alignment"

    def score_step(self, context: StepContext) -> float:
        wrong_way_penalty = 20.0 if context.heading_alignment < 0.0 else 0.0
        return (
            (context.progress_delta * 7.5)
            + (context.velocity * context.heading_alignment * 1.2)
            - wrong_way_penalty
            - (context.normalized_center_offset * 1.2)
            - (context.time_elapsed * 0.08)
            + _event_bonus(context, finish_bonus=22000.0, collision_penalty=900.0)
        )


class NoSpinRaceMetric(FitnessStrategy):
    name = "no_spin_race_metric"

    def score_step(self, context: StepContext) -> float:
        return (
            (context.progress_delta * 7.0)
            + (context.velocity * 0.5)
            + (context.progress_ratio * 0.8)
            - (20.0 if context.is_spinning else 0.0)
            - (14.0 if context.is_stalled else 0.0)
            - (context.time_elapsed * 0.04)
            + _event_bonus(context, finish_bonus=18000.0, collision_penalty=900.0)
        )


class ConservativeSurvivor(FitnessStrategy):
    name = "conservative_survivor"

    def score_step(self, context: StepContext) -> float:
        survival_bonus = 0.1 if not context.collided else 0.0
        return (
            (context.progress_delta * 2.5)
            + survival_bonus
            + (min(context.min_clearance, 120.0) * 0.06)
            - (context.velocity * context.normalized_center_offset * 0.15)
            + _event_bonus(context, finish_bonus=2200.0, collision_penalty=700.0)
        )


STRATEGIES: dict[str, type[FitnessStrategy]] = {
    SpeedOnlyBaseline.name: SpeedOnlyBaseline,
    ProgressOnly.name: ProgressOnly,
    SpeedProgress.name: SpeedProgress,
    ForwardMomentum.name: ForwardMomentum,
    CenterlineProgress.name: CenterlineProgress,
    SafeFast.name: SafeFast,
    RaceMetricProxy.name: RaceMetricProxy,
    RaceMetricProgressFocus.name: RaceMetricProgressFocus,
    RaceMetricSpeedFocus.name: RaceMetricSpeedFocus,
    RaceMetricLinearTime.name: RaceMetricLinearTime,
    RaceMetricForwardEfficiency.name: RaceMetricForwardEfficiency,
    RaceMetricCollisionGuard.name: RaceMetricCollisionGuard,
    RaceMetricClearanceGuard.name: RaceMetricClearanceGuard,
    RaceMetricFinishReward.name: RaceMetricFinishReward,
    FrontierExplorer.name: FrontierExplorer,
    RiskAdjustedPace.name: RiskAdjustedPace,
    CenterlinePace.name: CenterlinePace,
    TwoPhaseRacer.name: TwoPhaseRacer,
    SmoothControlPace.name: SmoothControlPace,
    SparseOutcome.name: SparseOutcome,
    SensorBalancePace.name: SensorBalancePace,
    RaceMetricProxyV2.name: RaceMetricProxyV2,
    RaceMetricProxyFinishPush.name: RaceMetricProxyFinishPush,
    RaceMetricProxyCheckpointPush.name: RaceMetricProxyCheckpointPush,
    RaceMetricProxyAlignmentPush.name: RaceMetricProxyAlignmentPush,
    AntiStallProgress.name: AntiStallProgress,
    SmoothProgress.name: SmoothProgress,
    WallClearanceProgress.name: WallClearanceProgress,
    BalancedDriver.name: BalancedDriver,
    FinishSeeker.name: FinishSeeker,
    RaceMetricSprint.name: RaceMetricSprint,
    RaceMetricLateBonus.name: RaceMetricLateBonus,
    RaceMetricAligned.name: RaceMetricAligned,
    RaceMetricSoftCheckpoints.name: RaceMetricSoftCheckpoints,
    RaceMetricNoStall.name: RaceMetricNoStall,
    RaceMetricFinishPressure.name: RaceMetricFinishPressure,
    LateProgressSprint.name: LateProgressSprint,
    CheckpointSprint.name: CheckpointSprint,
    FinishLineHunter.name: FinishLineHunter,
    TimeAttackAlignment.name: TimeAttackAlignment,
    NoSpinRaceMetric.name: NoSpinRaceMetric,
    ConservativeSurvivor.name: ConservativeSurvivor,
}


def build_strategy(name: str) -> FitnessStrategy:
    try:
        return STRATEGIES[name]()
    except KeyError as exc:
        raise ValueError(f"Unknown fitness strategy: {name}") from exc
