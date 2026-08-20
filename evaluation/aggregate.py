
from pathlib import Path
import json
import csv
import numpy as np


def collect_runs(results_dir):
    runs = []
    for path in Path(results_dir).rglob("summary.json"):
        with path.open("r", encoding="utf-8") as f:
            item = json.load(f)
        item["run_dir"] = str(path.parent)
        runs.append(item)
    return runs


def aggregate(runs):
    if not runs:
        return {}

    success = [bool(x["success"]) for x in runs]
    success_steps = [x["steps"] for x in runs if x["success"]]
    success_paths = [x["path_length"] for x in runs if x["success"]]

    return {
        "number_of_runs": len(runs),
        "successes": int(sum(success)),
        "success_rate": float(np.mean(success)),
        "mean_steps_on_success": float(np.mean(success_steps)) if success_steps else None,
        "median_steps_on_success": float(np.median(success_steps)) if success_steps else None,
        "mean_path_length_on_success": float(np.mean(success_paths)) if success_paths else None,
        "mean_final_distance": float(np.mean([x["final_distance"] for x in runs])),
    }


def save_csv(path, rows):
    if not rows:
        return
    with Path(path).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
