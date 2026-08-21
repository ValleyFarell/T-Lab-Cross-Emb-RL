
import pytest

from evaluation.aggregate import aggregate, pair_runs


def test_aggregate():
    out = aggregate([
        {
            "method": "baseline",
            "task_id": 1,
            "environment_seed": 0,
            "controller_seed": 0,
            "success": True,
            "steps": 10,
            "path_length": 5.0,
            "final_distance": 0.1,
        },
        {
            "method": "baseline",
            "task_id": 1,
            "environment_seed": 1,
            "controller_seed": 0,
            "success": False,
            "steps": 100,
            "path_length": 20.0,
            "final_distance": 2.0,
        },
    ])

    overall = out["per_method"]["baseline"]["overall"]
    task = out["per_method"]["baseline"]["per_task"]["1"]

    assert overall["number_of_runs"] == 2
    assert overall["successes"] == 1
    assert overall["success_rate"] == 0.5
    assert task == overall


def test_pair_runs_requires_identical_scenarios():
    runs = [
        {
            "method": "baseline",
            "task_id": 1,
            "environment_seed": 0,
            "controller_seed": 0,
        },
        {
            "method": "h0",
            "task_id": 1,
            "environment_seed": 1,
            "controller_seed": 0,
        },
    ]

    with pytest.raises(ValueError, match="different scenario sets"):
        pair_runs(runs, "baseline", "h0")
