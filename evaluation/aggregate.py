
from pathlib import Path
import csv
import json
import numpy as np


REQUIRED_KEYS = {
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

        if not REQUIRED_KEYS.issubset(item.keys()):
            continue

        # Recover task id from directory structure if it is not stored
        # in the episode summary.
        if "task_id" not in item:
            for parent in path.parents:
                if parent.name.startswith("baseline_task_"):
                    item["task_id"] = int(
                        parent.name.replace("baseline_task_", "")
                    )
                    break

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
    per_task = {}

    for run in runs:
        task = str(run.get("task_id", "unknown"))
        per_task.setdefault(task, []).append(run)

    return {
        "overall": summarize_group(runs),
        "per_task": {
            task: summarize_group(items)
            for task, items in sorted(per_task.items())
        },
    }


def save_csv(path, rows):
    if not rows:
        return

    with Path(path).open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=rows[0].keys(),
        )
        writer.writeheader()
        writer.writerows(rows)
