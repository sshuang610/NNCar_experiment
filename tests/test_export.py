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
    np.testing.assert_allclose(model["weights"][0], net.weights[0].reshape(-1))
    np.testing.assert_allclose(model["weights"][1], net.weights[1].reshape(-1))
    np.testing.assert_allclose(model["biases"][0], net.biases[0].reshape(-1))
    np.testing.assert_allclose(model["biases"][1], net.biases[1].reshape(-1))


def test_promote_template_writes_five_files(tmp_path):
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
