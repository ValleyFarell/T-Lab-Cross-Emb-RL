
"""Серия запусков исходного агента по нескольким задачам и инициализациям."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--tasks",
        nargs="+",
        type=int,
        default=[1, 2, 3, 4, 5],
    help='Список задач в evaluate_baseline.',
    )

    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[0, 1, 2, 3, 4],
    help='Список инициализаций среды для серии экспериментов.',
    )

    parser.add_argument(
        "--controller-seed",
        type=int,
        default=0,
    help='Число, определяющее воспроизводимую случайность контроллера.',
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
    help='Случайность выбора действий; ноль означает детерминированный режим.',
    )

    parser.add_argument(
        "--log-file",
        type=Path,
        default=Path("results/evaluation_commands.json"),
    help='Файл журнала с параметрами группового запуска.',
    )

    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results"),
    help='Каталог сохранения эпизодов.',
    )

    return parser.parse_args()


def main():
    args = parse_args()

    commands = []

    for task_id in args.tasks:
        for env_seed in args.seeds:
            command = [
                sys.executable,
                "-m",
                "scripts.run_baseline",
                "--task-id",
                str(task_id),
                "--environment-seed",
                str(env_seed),
                "--controller-seed",
                str(args.controller_seed),
                "--temperature",
                str(args.temperature),
                "--results-dir",
                str(args.results_dir),
            ]

            print("\nRunning:")
            print(" ".join(command))

            subprocess.run(
                command,
                check=True,
            )

            commands.append(
                {
                    "task_id": task_id,
                    "environment_seed": env_seed,
                    "controller_seed": args.controller_seed,
                    "temperature": args.temperature,
                    "results_dir": str(args.results_dir),
                }
            )

    args.log_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with args.log_file.open("w", encoding="utf-8") as f:
        json.dump(
            commands,
            f,
            indent=2,
        )

    print("\nFinished.")
    print(f"Executed runs: {len(commands)}")


if __name__ == "__main__":
    main()
