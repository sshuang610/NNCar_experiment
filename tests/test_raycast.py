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
