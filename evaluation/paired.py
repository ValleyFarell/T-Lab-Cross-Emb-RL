"""Paired comparison of two methods on identical evaluation scenarios."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _scenario_key(record: dict) -> tuple:
    """Return the fields that must be identical for a valid pair."""

    return (
        record.get("scenario_id"),
        record.get("task_id"),
        tuple(record["start_ij"]) if record.get("start_ij") is not None else None,
        tuple(record["goal_ij"]) if record.get("goal_ij") is not None else None,
        record.get("environment_seed"),
        record.get("controller_seed"),
        record.get("temperature", 0.0),
    )


def collect_paired_records(results_dir: str | Path) -> dict[tuple, dict]:
    """Load one record per scenario, rejecting accidental duplicate runs."""

    records = {}
    for summary_path in sorted(Path(results_dir).rglob("summary.json")):
        if summary_path.parent == Path(results_dir):
            continue

        summary = _load_json(summary_path)
        if not {"success", "steps", "path_length", "final_distance"}.issubset(summary):
            continue

        scenario_path = summary_path.with_name("scenario.json")
        scenario = _load_json(scenario_path) if scenario_path.exists() else {}
        record = {**scenario, **summary}
        record["run_dir"] = str(summary_path.parent)
        key = _scenario_key(record)

        if key in records:
            raise ValueError(
                "Duplicate scenario in one method directory: "
                f"{key}. Keep exactly one run per scenario and seed."
            )
        records[key] = record

    return records


def _mean(values) -> float | None:
    return float(np.mean(values)) if values else None


def _median(values) -> float | None:
    return float(np.median(values)) if values else None


def _bootstrap_mean_ci(
    values,
    *,
    seed: int,
    samples: int,
) -> list[float] | None:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return None
    if values.size == 1:
        value = float(values[0])
        return [value, value]

    rng = np.random.default_rng(seed)
    indices = rng.integers(0, values.size, size=(samples, values.size))
    bootstrap_means = values[indices].mean(axis=1)
    low, high = np.quantile(bootstrap_means, [0.025, 0.975])
    return [float(low), float(high)]


def _mcnemar_exact_p(baseline_only: int, candidate_only: int) -> float | None:
    """Two-sided exact McNemar test over discordant success pairs."""

    discordant = baseline_only + candidate_only
    if discordant == 0:
        return 1.0

    smaller = min(baseline_only, candidate_only)
    tail = sum(math.comb(discordant, k) for k in range(smaller + 1))
    probability = tail / (2**discordant)
    return float(min(1.0, 2.0 * probability))


def _summarize_pairs(pairs: list[dict], *, bootstrap_seed: int, bootstrap_samples: int) -> dict:
    success_deltas = [
        int(pair["candidate"]["success"]) - int(pair["baseline"]["success"])
        for pair in pairs
    ]
    baseline_only = sum(delta == -1 for delta in success_deltas)
    candidate_only = sum(delta == 1 for delta in success_deltas)
    both_success = [
        pair
        for pair in pairs
        if pair["baseline"]["success"] and pair["candidate"]["success"]
    ]
    both_failure = sum(
        not pair["baseline"]["success"] and not pair["candidate"]["success"]
        for pair in pairs
    )

    steps_deltas = [
        pair["candidate"]["steps"] - pair["baseline"]["steps"]
        for pair in both_success
    ]
    path_deltas = [
        pair["candidate"]["path_length"] - pair["baseline"]["path_length"]
        for pair in both_success
    ]
    distance_deltas = [
        pair["candidate"]["final_distance"] - pair["baseline"]["final_distance"]
        for pair in pairs
    ]

    count = len(pairs)
    baseline_successes = sum(pair["baseline"]["success"] for pair in pairs)
    candidate_successes = sum(pair["candidate"]["success"] for pair in pairs)

    return {
        "number_of_pairs": count,
        "baseline_success_rate": baseline_successes / count if count else None,
        "candidate_success_rate": candidate_successes / count if count else None,
        "success_rate_delta": _mean(success_deltas),
        "success_rate_delta_ci95": _bootstrap_mean_ci(
            success_deltas,
            seed=bootstrap_seed,
            samples=bootstrap_samples,
        ),
        "discordant_pairs": {
            "baseline_only_success": baseline_only,
            "candidate_only_success": candidate_only,
            "mcnemar_exact_p": _mcnemar_exact_p(baseline_only, candidate_only),
        },
        "both_success": len(both_success),
        "both_failure": both_failure,
        "steps_delta_both_success": {
            "mean": _mean(steps_deltas),
            "median": _median(steps_deltas),
            "ci95_mean": _bootstrap_mean_ci(
                steps_deltas,
                seed=bootstrap_seed + 1,
                samples=bootstrap_samples,
            ),
        },
        "path_length_delta_both_success": {
            "mean": _mean(path_deltas),
            "median": _median(path_deltas),
            "ci95_mean": _bootstrap_mean_ci(
                path_deltas,
                seed=bootstrap_seed + 2,
                samples=bootstrap_samples,
            ),
        },
        "final_distance_delta_all": {
            "mean": _mean(distance_deltas),
            "median": _median(distance_deltas),
            "ci95_mean": _bootstrap_mean_ci(
                distance_deltas,
                seed=bootstrap_seed + 3,
                samples=bootstrap_samples,
            ),
        },
    }


def compare_paired(
    baseline_records: dict[tuple, dict],
    candidate_records: dict[tuple, dict],
    *,
    bootstrap_seed: int = 0,
    bootstrap_samples: int = 10_000,
) -> dict:
    """Compare two methods using only their exact scenario intersection."""

    if bootstrap_samples <= 0:
        raise ValueError("bootstrap_samples must be positive.")

    baseline_keys = set(baseline_records)
    candidate_keys = set(candidate_records)
    common_keys = sorted(baseline_keys & candidate_keys, key=repr)

    if not common_keys:
        raise ValueError("No matching scenarios were found between the two methods.")

    pairs = [
        {
            "scenario_key": list(key),
            "baseline": baseline_records[key],
            "candidate": candidate_records[key],
        }
        for key in common_keys
    ]

    per_scenario = {}
    for pair in pairs:
        scenario_id = pair["baseline"].get("scenario_id", "unknown")
        per_scenario.setdefault(scenario_id, []).append(pair)

    return {
        "pairing_fields": [
            "scenario_id",
            "task_id",
            "start_ij",
            "goal_ij",
            "environment_seed",
            "controller_seed",
            "temperature",
        ],
        "unmatched": {
            "baseline": len(baseline_keys - candidate_keys),
            "candidate": len(candidate_keys - baseline_keys),
        },
        "overall": _summarize_pairs(
            pairs,
            bootstrap_seed=bootstrap_seed,
            bootstrap_samples=bootstrap_samples,
        ),
        "per_scenario": {
            scenario_id: _summarize_pairs(
                scenario_pairs,
                bootstrap_seed=bootstrap_seed,
                bootstrap_samples=bootstrap_samples,
            )
            for scenario_id, scenario_pairs in sorted(per_scenario.items())
        },
        "pairs": pairs,
    }
