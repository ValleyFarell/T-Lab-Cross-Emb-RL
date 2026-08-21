import json

import pytest

from evaluation.paired import collect_paired_records, compare_paired


def make_record(scenario_id, seed, success, steps, path_length, final_distance):
    return {
        "scenario_id": scenario_id,
        "task_id": 1,
        "start_ij": None,
        "goal_ij": None,
        "environment_seed": seed,
        "controller_seed": 0,
        "temperature": 0.0,
        "success": success,
        "steps": steps,
        "path_length": path_length,
        "final_distance": final_distance,
    }


def key(record):
    return (
        record["scenario_id"],
        record["task_id"],
        None,
        None,
        record["environment_seed"],
        record["controller_seed"],
        record["temperature"],
    )


def test_paired_comparison_uses_candidate_minus_baseline():
    baseline_items = [
        make_record("task-1", 0, False, 1000, 20.0, 5.0),
        make_record("task-1", 1, True, 200, 10.0, 0.4),
    ]
    candidate_items = [
        make_record("task-1", 0, True, 300, 12.0, 0.4),
        make_record("task-1", 1, True, 150, 8.0, 0.3),
    ]

    result = compare_paired(
        {key(item): item for item in baseline_items},
        {key(item): item for item in candidate_items},
        bootstrap_samples=100,
    )

    overall = result["overall"]
    assert overall["number_of_pairs"] == 2
    assert overall["success_rate_delta"] == 0.5
    assert overall["discordant_pairs"]["candidate_only_success"] == 1
    assert overall["steps_delta_both_success"]["mean"] == -50.0
    assert overall["path_length_delta_both_success"]["mean"] == -2.0


def test_paired_comparison_reports_unmatched_runs():
    baseline = make_record("task-1", 0, True, 10, 1.0, 0.1)
    candidate = make_record("task-1", 1, True, 10, 1.0, 0.1)
    common_candidate = make_record("task-1", 0, True, 10, 1.0, 0.1)

    result = compare_paired(
        {key(baseline): baseline},
        {key(candidate): candidate, key(common_candidate): common_candidate},
        bootstrap_samples=10,
    )

    assert result["unmatched"] == {"baseline": 0, "candidate": 1}


def test_paired_comparison_requires_common_scenario():
    baseline = make_record("task-1", 0, True, 10, 1.0, 0.1)
    candidate = make_record("task-2", 0, True, 10, 1.0, 0.1)

    with pytest.raises(ValueError, match="No matching scenarios"):
        compare_paired(
            {key(baseline): baseline},
            {key(candidate): candidate},
            bootstrap_samples=10,
        )


def test_collection_rejects_duplicate_scenario_seed(tmp_path):
    record = make_record("task-1", 0, True, 10, 1.0, 0.1)

    for run_id in ("000001", "000002"):
        run_dir = tmp_path / "task" / "runs" / run_id
        run_dir.mkdir(parents=True)
        with (run_dir / "summary.json").open("w", encoding="utf-8") as file:
            json.dump(record, file)
        with (run_dir / "scenario.json").open("w", encoding="utf-8") as file:
            json.dump(record, file)

    with pytest.raises(ValueError, match="Duplicate scenario"):
        collect_paired_records(tmp_path)
