
from evaluation.aggregate import aggregate


def test_aggregate():
    out = aggregate([
        {"success": True, "steps": 10, "path_length": 5.0, "final_distance": 0.1},
        {"success": False, "steps": 100, "path_length": 20.0, "final_distance": 2.0},
    ])

    assert out["overall"]["number_of_runs"] == 2
    assert out["overall"]["successes"] == 1
    assert out["overall"]["success_rate"] == 0.5
