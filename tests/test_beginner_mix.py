from __future__ import annotations

from pipeline.fitness import BeginnerMix, FINISH_BONUS, B_CRASH, B_CHECKPOINT
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


def test_penalty_at_100_cancels_full_reward_for_that_frame():
    strat = BeginnerMix()
    strat.configure({"rewards": {"progress": 50}, "penalties": {"stall": 100}})
    step = strat.score_step(ctx(progress_delta=10.0, is_stalled=True))
    assert abs(step) < 1e-9


def test_extra_reward_blocks_do_not_dilute_penalties():
    a = BeginnerMix(); a.configure({"rewards": {"progress": 40}, "penalties": {"stall": 60}})
    b = BeginnerMix(); b.configure({"rewards": {"progress": 40, "speed": 30, "safety": 30},
                                    "penalties": {"stall": 60}})
    # min_clearance=0 zeroes the safety reward factor so the comparison isolates
    # the penalty side; otherwise extra reward blocks contribute their own reward.
    c = ctx(is_stalled=True, min_clearance=0.0)
    assert a.score_step(c) == b.score_step(c)
    assert a.score_step(c) < 0.0


def test_perstep_pieces_scale_with_dt_so_per_second_is_fps_invariant():
    strat = BeginnerMix()
    strat.configure({"rewards": {"speed": 100}, "penalties": {"time": 50}})
    c30 = ctx(velocity=5.0, frame=30, time_elapsed=1.0)
    c60 = ctx(velocity=5.0, frame=60, time_elapsed=1.0)
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


def test_checkpoint_excluded_from_normalized_reward():
    a = BeginnerMix(); a.configure({"rewards": {"progress": 50}})
    b = BeginnerMix(); b.configure({"rewards": {"progress": 50, "checkpoint": 100}})
    a.reset(); b.reset()
    # progress_ratio below first milestone -> no checkpoint award; per-frame reward identical
    c = ctx(progress_delta=5.0, progress_ratio=0.05)
    assert a.score_step(c) == b.score_step(c)


def test_checkpoint_awards_once_per_milestone():
    strat = BeginnerMix(); strat.configure({"rewards": {"checkpoint": 100}}); strat.reset()
    # default marks (0.2, 0.4, 0.6, 0.8, 0.95): crossing 0.45 awards for 0.2 and 0.4
    step = strat.score_step(ctx(progress_ratio=0.45))
    assert abs(step - 2 * B_CHECKPOINT) < 1e-9
    # same ratio next step -> no further award
    assert abs(strat.score_step(ctx(progress_ratio=0.45))) < 1e-9


def test_checkpoint_reset_reawards():
    strat = BeginnerMix(); strat.configure({"rewards": {"checkpoint": 100}}); strat.reset()
    strat.score_step(ctx(progress_ratio=0.25))   # awards milestone 0.2
    strat.reset()
    assert abs(strat.score_step(ctx(progress_ratio=0.25)) - B_CHECKPOINT) < 1e-9
