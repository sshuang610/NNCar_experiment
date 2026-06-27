import math
import random

from pipeline.track import generate_track


def _reference_on_track(track, point):
    # Brute-force predicate: within half_width of ANY polyline segment.
    px, py = point
    hw = track.half_width
    best = float("inf")
    for s, e in zip(track.polyline[:-1], track.polyline[1:]):
        vx = e[0] - s[0]
        vy = e[1] - s[1]
        L = (vx * vx) + (vy * vy)
        if L == 0.0:
            d = math.dist(point, s)
        else:
            u = max(0.0, min(1.0, (((px - s[0]) * vx) + ((py - s[1]) * vy)) / L))
            d = math.dist(point, (s[0] + vx * u, s[1] + vy * u))
        best = min(best, d)
    return best <= hw


def test_is_on_track_matches_reference_across_seeds():
    for seed in (202, 203, 204):
        track = generate_track(seed, 120, 34.0)
        w, h = track.canvas_size
        rng = random.Random(seed)
        for _ in range(3000):
            p = (rng.uniform(0, w), rng.uniform(0, h))
            assert track.is_on_track(p) == _reference_on_track(track, p)
        for i in range(50):
            on = track.point_at_progress(i * track.total_length / 50)
            assert track.is_on_track(on) is True


def test_is_on_track_far_point_false():
    track = generate_track(202, 120, 34.0)
    assert track.is_on_track((-99999.0, -99999.0)) is False
