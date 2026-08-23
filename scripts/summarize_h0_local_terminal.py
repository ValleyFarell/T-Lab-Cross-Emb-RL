"""Сводка успехов, падений, резервных переходов и поведения у финиша."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def collect(results_dir: Path):
    records = []
    for summary_path in sorted(results_dir.rglob("summary.json")):
        trajectory_path = summary_path.with_name("trajectory.npz")
        if not trajectory_path.exists():
            continue
        with summary_path.open("r", encoding="utf-8") as handle:
            summary = json.load(handle)
        with np.load(trajectory_path, allow_pickle=False) as trajectory:
            if "diagnostic_h0lt_mode" not in trajectory:
                continue
            mode = np.asarray(trajectory["diagnostic_h0lt_mode"])
            goal_distance = np.asarray(trajectory["diagnostic_h0lt_goal_distance"])
            observations = np.asarray(trajectory["observations"])
            terminal = np.flatnonzero(np.isin(mode, [2, 3]))
            local_counts = np.asarray(trajectory["diagnostic_h0lt_local_candidate_count"])
            eta_one = np.asarray(trajectory["diagnostic_selected_eta1"])
            record = {
                "task_id": summary.get("task_id"),
                "environment_seed": summary.get("environment_seed"),
                "success": bool(summary["success"]),
                "steps": int(summary["steps"]),
                "path_length": float(summary["path_length"]),
                "final_distance": float(summary["final_distance"]),
                "minimum_distance_before_final_step": float(goal_distance.min()),
                "direct_zero_steps": int(np.sum(mode == 0)),
                "two_switch_steps": int(np.sum(mode == 1)),
                "terminal_direct_steps": int(np.sum(mode == 2)),
                "terminal_baseline_steps": int(np.sum(mode == 3)),
                "entered_terminal": bool(len(terminal)),
                "first_terminal_step": int(terminal[0]) if len(terminal) else None,
                "fell_below_0_3": bool(
                    observations.shape[1] > 2 and np.any(observations[:, 2] < 0.3)
                ),
                "minimum_body_height": float(observations[:, 2].min())
                if observations.shape[1] > 2
                else None,
                "mean_local_candidate_count": float(local_counts.mean()),
                "selected_eta_one_fraction": float(
                    np.mean(np.isfinite(eta_one) & np.isclose(eta_one, 1.0))
                ),
                "run_dir": str(summary_path.parent),
            }
            records.append(record)
    return records


def summarize(records):
    if not records:
        return {"runs": 0}
    successful = [row for row in records if row["success"]]
    entered = [row for row in records if row["entered_terminal"]]
    return {
        "runs": len(records),
        "successes": len(successful),
        "success_rate": len(successful) / len(records),
        "falls": sum(row["fell_below_0_3"] for row in records),
        "fall_rate": float(
            np.mean([row["fell_below_0_3"] for row in records])
        ),
        "entered_terminal": len(entered),
        "success_after_entering_terminal": (
            float(np.mean([row["success"] for row in entered])) if entered else None
        ),
        "mean_steps_success": (
            float(np.mean([row["steps"] for row in successful])) if successful else None
        ),
        "mean_direct_zero_steps": float(
            np.mean([row["direct_zero_steps"] for row in records])
        ),
        "mean_two_switch_steps": float(
            np.mean([row["two_switch_steps"] for row in records])
        ),
        "mean_terminal_steps": float(
            np.mean(
                [
                    row["terminal_direct_steps"] + row["terminal_baseline_steps"]
                    for row in records
                ]
            )
        ),
    }


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, required=True, help='Каталог сохранения эпизодов.')
    parser.add_argument("--output-dir", type=Path, default=None, help='Каталог сохранения моделей, оценок и промежуточных данных.')
    args = parser.parse_args(argv)
    if not args.results_dir.is_dir():
        parser.error(f"results directory does not exist: {args.results_dir}")

    rows = collect(args.results_dir)
    if not rows:
        parser.error("no local-terminal H0 trajectories were found")
    output_dir = args.output_dir or args.results_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    groups = {}
    for row in rows:
        groups.setdefault(str(row["task_id"]), []).append(row)
    result = {
        "overall": summarize(rows),
        "per_task": {task: summarize(items) for task, items in sorted(groups.items())},
    }
    json_path = output_dir / "h0_local_terminal_diagnostics.json"
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    csv_path = output_dir / "h0_local_terminal_runs.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(result, indent=2))
    print(f"saved_json: {json_path}")
    print(f"saved_csv: {csv_path}")
    return result


if __name__ == "__main__":
    main()
