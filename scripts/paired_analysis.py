"""Compare a candidate method with the baseline on matched scenario seeds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from evaluation.paired import collect_paired_records, compare_paired


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/paired_comparison.json"),
    )
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    return parser.parse_args()


def main():
    args = parse_args()
    baseline = collect_paired_records(args.baseline_dir)
    candidate = collect_paired_records(args.candidate_dir)
    comparison = compare_paired(
        baseline,
        candidate,
        bootstrap_seed=args.bootstrap_seed,
        bootstrap_samples=args.bootstrap_samples,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as file:
        json.dump(comparison, file, indent=2)

    print(json.dumps({key: value for key, value in comparison.items() if key != "pairs"}, indent=2))
    print(f"saved_to: {args.output}")


if __name__ == "__main__":
    main()

