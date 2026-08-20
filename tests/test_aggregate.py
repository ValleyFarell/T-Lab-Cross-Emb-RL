
from evaluation.aggregate import aggregate


def test_aggregate():
    out = aggregate([
        {"success": True, "steps": 10, "path_length": 5.0, "final_distance": 0.1},
        {"success": False, "steps": 100, "path_length": 20.0, "final_distance": 2.0},
    ])

    assert out["number_of_runs"] == 2
    assert out["successes"] == 1
    assert out["success_rate"] == 0.5
