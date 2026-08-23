"""Сохранение контрольной сводки исходного метода."""

import argparse
import json
from pathlib import Path

from evaluation.baseline_reference import save_baseline_reference


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, default=Path("results/summary.json"), help='Путь к ранее рассчитанной итоговой сводке.')
    parser.add_argument("--results-dir", type=Path, default=Path("results"), help='Каталог сохранения эпизодов.')
    args = parser.parse_args()

    with args.summary.open("r", encoding="utf-8") as f:
        summary = json.load(f)

    metadata = {
        "environment": "ogbench-antmaze-medium-navigate-v0",
        "checkpoint_frozen": True,
        "controller": "BaselineController",
        "temperature": 0.0,
        "evaluation_runs": 25,
        "tasks": [1, 2, 3, 4, 5],
        "seeds": [0, 1, 2, 3, 4],
    }

    print(
        save_baseline_reference(
            args.results_dir,
            summary,
            metadata,
        )
    )


if __name__ == "__main__":
    main()
