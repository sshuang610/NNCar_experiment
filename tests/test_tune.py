from pipeline.tune import neighbor_recipes


def test_neighbors_perturb_each_active_slider_up_and_down_clamped():
    base = {"rewards": {"progress": 50, "speed": 90}, "penalties": {"crash": 100}}
    neighbors = neighbor_recipes(base, step=15)
    labels = {label for label, _ in neighbors}
    assert "progress_up" in labels and "progress_down" in labels
    recipes = dict(neighbors)
    assert recipes["speed_up"]["rewards"]["speed"] == 100.0
    assert recipes["speed_down"]["rewards"]["speed"] == 75.0
    assert recipes["crash_down"]["penalties"]["crash"] == 85.0
    # crash_up would be 115 -> clamps to 100 == current -> dropped
    assert "crash_up" not in labels


def test_neighbors_are_deterministic():
    base = {"rewards": {"progress": 50}, "penalties": {"stall": 40}}
    assert neighbor_recipes(base, step=10) == neighbor_recipes(base, step=10)
