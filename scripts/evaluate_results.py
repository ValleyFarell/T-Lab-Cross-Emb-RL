"""Построение общей сводки по сохранённым результатам метода."""


import argparse
import json
from pathlib import Path

from evaluation.aggregate import collect_runs, aggregate, save_csv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=Path("results"), help='Каталог сохранения эпизодов.')
    args = parser.parse_args()

    runs = collect_runs(args.results_dir)
    summary = aggregate(runs)

    with (args.results_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    save_csv(args.results_dir / "summary.csv", runs)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
