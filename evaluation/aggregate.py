
from pathlib import Path
import csv
import json
import numpy as np


REQUIRED_KEYS = {
    "method",
    "task_id",
    "environment_seed",
    "controller_seed",
    "success",
    "steps",
    "path_length",
    "final_distance",
}


def collect_runs(results_dir):
    runs = []

    for path in Path(results_dir).rglob("summary.json"):
        with path.open("r", encoding="utf-8") as f:
            item = json.load(f)

        scenario_path = path.with_name("scenario.json")
        if scenario_path.exists():
            with scenario_path.open("r", encoding="utf-8") as f:
                scenario = json.load(f)
            for key, value in scenario.items():
                item.setdefault(key, value)

        # Recover task id from directory structure if it is not stored
        # in the episode summary.
        if "task_id" not in item:
            for parent in path.parents:
                if parent.name.startswith("baseline_task_"):
                    item["task_id"] = int(
                        parent.name.replace("baseline_task_", "")
                    )
                    break

        if not REQUIRED_KEYS.issubset(item.keys()):
            continue

        runs.append(item)

    return runs


def _mean(values):
    return float(np.mean(values)) if values else None


def _median(values):
    return float(np.median(values)) if values else None


def summarize_group(runs):
    success = [r for r in runs if r["success"]]
    failure = [r for r in runs if not r["success"]]

    return {
        "number_of_runs": len(runs),
        "successes": len(success),
        "success_rate": len(success) / len(runs) if runs else None,
        "mean_steps_success": _mean([r["steps"] for r in success]),
        "median_steps_success": _median([r["steps"] for r in success]),
        "mean_path_length_success": _mean([r["path_length"] for r in success]),
        "median_path_length_success": _median([r["path_length"] for r in success]),
        "mean_final_distance_success": _mean(
            [r["final_distance"] for r in success]
        ),
        "mean_final_distance_failure": _mean(
            [r["final_distance"] for r in failure]
        ),
    }


def aggregate(runs):
    per_method = {}

    for run in runs:
        method = str(run["method"])
        per_method.setdefault(method, []).append(run)

    result = {}
    for method, method_runs in sorted(per_method.items()):
        per_task = {}
        for run in method_runs:
            task = str(run["task_id"])
            per_task.setdefault(task, []).append(run)

        result[method] = {
            "overall": summarize_group(method_runs),
            "per_task": {
                task: summarize_group(items)
                for task, items in sorted(
                    per_task.items(),
                    key=lambda pair: int(pair[0]),
                )
            },
        }

    return {"per_method": result}


def scenario_key(run):
    return (
        int(run["task_id"]),
        int(run["environment_seed"]),
        int(run["controller_seed"]),
    )


def _index_method_runs(runs, method):
    index = {}
    for run in runs:
        if run["method"] != method:
            continue
        key = scenario_key(run)
        if key in index:
            raise ValueError(
                f"Duplicate scenario for method {method!r}: {key}"
            )
        index[key] = run
    return index


def pair_runs(runs, method_a, method_b):
    """Return aligned run pairs and reject unfair comparisons."""

    index_a = _index_method_runs(runs, method_a)
    index_b = _index_method_runs(runs, method_b)

    keys_a = set(index_a)
    keys_b = set(index_b)
    if keys_a != keys_b:
        missing_a = sorted(keys_b - keys_a)
        missing_b = sorted(keys_a - keys_b)
        raise ValueError(
            "Methods use different scenario sets. "
            f"Missing for {method_a!r}: {missing_a}; "
            f"missing for {method_b!r}: {missing_b}."
        )

    return [
        (index_a[key], index_b[key])
        for key in sorted(keys_a)
    ]


def save_csv(path, rows):
    if not rows:
        return

    with Path(path).open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        fieldnames = sorted({key for row in rows for key in row})
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
