# Simulator Speedup (A) + Checkpoint Fitness (D) — Implementation Plan

> **For agentic workers:** TDD per task. Run tests with `uv run --with pytest pytest <path> -v`.

**Goal:** (A) Make the headless sim ~50–100× faster by fixing the sensor raycast + on-track test without changing behavior; (D) add a one-shot `checkpoint` reward block so the GA gets a stepped "go further" navigation signal.

**Branch:** `beginner-mix-harness`.

---

## Task A1: Fast on-track test (early-exit) + coarse-then-fine raycast

**Files:** Modify `pipeline/track.py` (`is_on_track`), `pipeline/simulator.py` (sensor loop + new `_ray_track_distance`). Test: `tests/test_raycast.py` (create).

**Context:** `Track.is_on_track(point)` currently calls `project()` which loops ALL polyline segments and returns the min distance ≤ half_width. The sensor loop in `SimCar.update_geometry` marches a point outward +4 up to 1000×/sensor (5 sensors), calling `is_on_track` each step. Both are the hot path.

**A1a — early-exit `is_on_track`** (behavior-identical: same boolean):

Replace `Track.is_on_track` in `pipeline/track.py` with:

```python
    def is_on_track(self, point: Point) -> bool:
        hw_sq = self.half_width * self.half_width
        px, py = point
        for start, end in zip(self.polyline[:-1], self.polyline[1:]):
            vx = end[0] - start[0]
            vy = end[1] - start[1]
            seg_len_sq = (vx * vx) + (vy * vy)
            if seg_len_sq == 0.0:
                dx = px - start[0]
                dy = py - start[1]
                if (dx * dx) + (dy * dy) <= hw_sq:
                    return True
                continue
            ux = px - start[0]
            uy = py - start[1]
            t = max(0.0, min(1.0, ((ux * vx) + (uy * vy)) / seg_len_sq))
            cx = start[0] + (vx * t)
            cy = start[1] + (vy * t)
            dx = px - cx
            dy = py - cy
            if (dx * dx) + (dy * dy) <= hw_sq:
                return True
        return False
```

**A1b — coarse-then-fine raycast** in `pipeline/simulator.py`. Add a module function (near the top, after `_move`):

```python
def _ray_track_distance(
    track: Track,
    origin: Point,
    angle: float,
    start: float = 10.0,
    max_range: float = 4000.0,
    coarse: float = 12.0,
) -> float:
    """Distance from origin along `angle` to the track-tube boundary.

    Mirrors the old +4/-1 marcher's tube-exit semantics but uses a coarse march
    then a binary search to ~1u precision, so it is far cheaper.
    """
    if not track.is_on_track(_move(origin, angle, start)):
        return start - 1.0
    distance = start
    while distance < max_range:
        nxt = distance + coarse
        if not track.is_on_track(_move(origin, angle, nxt)):
            lo, hi = distance, nxt
            while hi - lo > 1.0:
                mid = (lo + hi) / 2.0
                if track.is_on_track(_move(origin, angle, mid)):
                    lo = mid
                else:
                    hi = mid
            return lo
        distance = nxt
    return max_range
```

Then replace the sensor loop in `SimCar.update_geometry` (the `for sensor_angle in (0, 45, -45, 90, -90):` block) with:

```python
        self.sensor_points = []
        self.sensor_distances = []
        for sensor_angle in (0, 45, -45, 90, -90):
            ray_angle = self.angle + sensor_angle
            distance = _ray_track_distance(self.track, (self.x, self.y), ray_angle)
            self.sensor_points.append(_move((self.x, self.y), ray_angle, distance))
            self.sensor_distances.append(distance)
```

(Note: `sensor_distances` was previously `math.dist(center, point)`; since `point = _move(center, angle, distance)`, that distance equals `distance` — same value.)

**Tests — `tests/test_raycast.py`:** keep a reference copy of the OLD marcher and assert the new raycast matches within tolerance, on a real track.

```python
import math
from pipeline.track import generate_track
from pipeline.simulator import _move, _ray_track_distance


def _old_marcher(track, origin, angle):
    point = _move(origin, angle, 10)
    for _ in range(1000):
        if not track.is_on_track(point):
            point = _move(point, angle, -1)
            break
        point = _move(point, angle, 4)
    return math.dist(origin, point)


def test_new_raycast_matches_old_within_tolerance():
    track = generate_track(seed=202, cell_size=120, half_width=34.0)
    # sample positions along the track centerline, several ray angles each
    max_diff = 0.0
    for prog in [i * track.total_length / 40 for i in range(40)]:
        origin = track.point_at_progress(prog)
        base = track.heading_at_progress(prog)
        for sensor_angle in (0, 45, -45, 90, -90):
            a = base + sensor_angle
            new = _ray_track_distance(track, origin, a)
            old = _old_marcher(track, origin, a)
            max_diff = max(max_diff, abs(new - old))
    assert max_diff <= 5.0, f"max sensor distance diff {max_diff} exceeds tolerance"


def test_is_on_track_boundary():
    track = generate_track(seed=202, cell_size=120, half_width=34.0)
    center = track.point_at_progress(track.total_length / 3)
    assert track.is_on_track(center) is True
    far = (center[0] + 1000.0, center[1] + 1000.0)
    assert track.is_on_track(far) is False
```

**Steps:** write tests → run (the raycast import fails) → implement A1a + A1b → run tests → run full suite `uv run --with pytest pytest tests/ -v` (all green) → commit `perf: coarse-then-fine raycast + early-exit on-track test`.

---

## Task A2: Timing sanity (no code, just measure)

After A1, run the smoke config and confirm it is dramatically faster than before:
```
uv run python -m pipeline.run_experiment --config configs/presets/smoke.json
```
Delete the resulting run dir afterward (artifacts ignored). Record the wall time in the commit/PR notes. No commit needed unless config changes.

---

## Task D1: `checkpoint` one-shot reward block

**Files:** Modify `pipeline/fitness.py` (`BeginnerMix`). Test: extend `tests/test_beginner_mix.py`.

**Context:** Mirror how `crash` is a one-shot penalty: `checkpoint` is a one-shot reward, excluded from the per-frame normalized reward sum. It awards `(w/100) * B_CHECKPOINT` each time `progress_ratio` crosses the next milestone.

Add constants near the existing `B`/`FINISH_BONUS` block in `pipeline/fitness.py`:

```python
CHECKPOINT_SECONDS = 5.0
B_CHECKPOINT = B * CHECKPOINT_SECONDS      # 50.0
DEFAULT_CHECKPOINTS = (0.2, 0.4, 0.6, 0.8, 0.95)
```

Modify `BeginnerMix`:
- `__init__`: after setting `rewards`/`penalties`, add `self._checkpoints = list(DEFAULT_CHECKPOINTS)` and `self._next_checkpoint = 0`.
- Add a `reset` method:

```python
    def reset(self) -> None:
        self._next_checkpoint = 0
```

- `configure`: allow `checkpoint` through the rewards filter — change the rewards comprehension to keep keys in `REWARD_BLOCKS` **or** `== "checkpoint"`:

```python
        self.rewards = {
            k: float(v) for k, v in rewards.items()
            if k in REWARD_BLOCKS or k == "checkpoint"
        }
```

- `score_step`: compute the normalized per-frame reward from **only `REWARD_BLOCKS`** (exclude `checkpoint`), then add the checkpoint one-shot. Replace the reward block of `score_step` with:

```python
        reward = 0.0
        per_frame = {k: v for k, v in self.rewards.items() if k in REWARD_BLOCKS}
        weight_sum = sum(per_frame.values())
        if weight_sum > 0.0:
            factors = self._reward_factors(context)
            weighted = sum(per_frame[k] * factors[k] for k in per_frame)
            reward = (weighted / weight_sum) * B * dt
```

And after the `if context.finished:` line (end of method, before `return step`), insert the checkpoint award:

```python
        cp_weight = self.rewards.get("checkpoint", 0.0)
        if cp_weight > 0.0:
            while (
                self._next_checkpoint < len(self._checkpoints)
                and context.progress_ratio >= self._checkpoints[self._next_checkpoint]
            ):
                step += (cp_weight / 100.0) * B_CHECKPOINT
                self._next_checkpoint += 1
```

**Tests — append to `tests/test_beginner_mix.py`** (the `ctx(**overrides)` helper already exists there):

```python
from pipeline.fitness import B_CHECKPOINT


def test_checkpoint_excluded_from_normalized_reward():
    a = BeginnerMix(); a.configure({"rewards": {"progress": 50}})
    b = BeginnerMix(); b.configure({"rewards": {"progress": 50, "checkpoint": 100}})
    a.reset(); b.reset()
    # progress_ratio below first milestone -> no checkpoint award; per-frame reward identical
    c = ctx(progress_delta=5.0, progress_ratio=0.05)
    assert a.score_step(c) == b.score_step(c)


def test_checkpoint_awards_once_per_milestone():
    strat = BeginnerMix(); strat.configure({"rewards": {"checkpoint": 100}}); strat.reset()
    # cross 0.2 and 0.4 in one step -> two awards; default marks (0.2,0.4,0.6,0.8,0.95)
    step = strat.score_step(ctx(progress_ratio=0.45))
    assert abs(step - 2 * B_CHECKPOINT) < 1e-9
    # next step at same ratio -> no further award
    assert abs(strat.score_step(ctx(progress_ratio=0.45))) < 1e-9


def test_checkpoint_reset_reawards():
    strat = BeginnerMix(); strat.configure({"rewards": {"checkpoint": 100}}); strat.reset()
    strat.score_step(ctx(progress_ratio=0.25))   # awards 0.2
    strat.reset()
    assert abs(strat.score_step(ctx(progress_ratio=0.25)) - B_CHECKPOINT) < 1e-9
```

**Steps:** write tests → run (fail) → implement → run `tests/test_beginner_mix.py` then full suite → commit `feat: add checkpoint one-shot reward block to BeginnerMix`.

---

## Task D2: checkpoint-enabled motion preset (config only)

Add one strategy to `configs/presets/starter_presets_fast.json` (and the full `starter_presets.json`) that uses `checkpoint`, e.g.:
```json
{ "name": "checkpoint_racer", "strategy": "beginner_mix",
  "params": { "rewards": {"progress": 45, "speed": 35, "checkpoint": 60},
              "penalties": {"stall": 40, "crash": 15} } }
```
Validate it loads. Commit `feat: add checkpoint_racer preset`.

---

## Acceptance
- Full test suite green (incl. new raycast + checkpoint tests).
- Smoke run wall-time confirms large speedup.
- `checkpoint` weight=0 / absent ⇒ identical to pre-D behavior (back-compat tests).
