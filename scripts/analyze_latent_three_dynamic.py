"""Парное сравнение исходного метода и динамического 4D-планировщика."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=Path("results_latent_three_dynamic"), help='Каталог сохранения эпизодов.')
    parser.add_argument("--baseline", default="fbpiswitch_baseline", help='Название исходного метода внутри общего каталога результатов.')
    parser.add_argument("--method", default="latent_three_dynamic_decoded", help='Название сравниваемого метода внутри общего каталога результатов.')
    parser.add_argument("--bootstrap-samples", type=int, default=10_000, help='Число повторных случайных выборок для доверительного интервала.')
    parser.add_argument("--seed", type=int, default=0, help='Воспроизводимая инициализация обучения и разбиения данных.')
    parser.add_argument("--output", type=Path, default=None, help='Путь к сохраняемому изображению или итоговому JSON-файлу.')
    args = parser.parse_args(argv)
    if args.bootstrap_samples <= 0:
        parser.error("--bootstrap-samples must be positive")
    return args


def load_results(directory: Path):
    results = {}
    for path in directory.rglob("summary.json"):
        with path.open("r", encoding="utf-8") as stream:
            record = json.load(stream)
        key = (
            str(record["method"]),
            int(record["task_id"]),
            int(record["environment_seed"]),
        )
        results[key] = record
    return results


def _mcnemar_exact(wins: int, losses: int) -> float:
    total = wins + losses
    if total == 0:
        return 1.0
    threshold = min(wins, losses)
    tail = sum(math.comb(total, count) for count in range(threshold + 1)) / 2**total
    return float(min(1.0, 2.0 * tail))


def paired_report(results, baseline, method, *, bootstrap_samples, seed):
    baseline_keys = {(task, episode_seed) for name, task, episode_seed in results if name == baseline}
    method_keys = {(task, episode_seed) for name, task, episode_seed in results if name == method}
    shared = sorted(baseline_keys & method_keys)
    if not shared:
        raise ValueError(
            f"no paired scenarios found for baseline={baseline!r} and method={method!r}"
        )
    baseline_success = np.asarray(
        [bool(results[(baseline, task, episode_seed)]["success"]) for task, episode_seed in shared],
        dtype=np.float64,
    )
    method_success = np.asarray(
        [bool(results[(method, task, episode_seed)]["success"]) for task, episode_seed in shared],
        dtype=np.float64,
    )
    difference = method_success - baseline_success
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(shared), size=(bootstrap_samples, len(shared)))
    bootstrap = difference[indices].mean(axis=1)
    wins = int(np.count_nonzero(difference > 0))
    losses = int(np.count_nonzero(difference < 0))
    tasks = {}
    for task in sorted({task for task, _ in shared}):
        positions = np.asarray([index for index, pair in enumerate(shared) if pair[0] == task])
        tasks[str(task)] = {
            "paired_count": int(len(positions)),
            "baseline_success_rate": float(baseline_success[positions].mean()),
            "method_success_rate": float(method_success[positions].mean()),
            "success_delta": float(difference[positions].mean()),
        }
    jointly_successful_steps = []
    for position, (task, episode_seed) in enumerate(shared):
        if baseline_success[position] and method_success[position]:
            jointly_successful_steps.append(
                results[(method, task, episode_seed)]["steps"]
                - results[(baseline, task, episode_seed)]["steps"]
            )
    return {
        "baseline": baseline,
        "method": method,
        "paired_count": int(len(shared)),
        "unpaired_baseline_count": int(len(baseline_keys - method_keys)),
        "unpaired_method_count": int(len(method_keys - baseline_keys)),
        "baseline_success_rate": float(baseline_success.mean()),
        "method_success_rate": float(method_success.mean()),
        "success_delta": float(difference.mean()),
        "bootstrap_ci95": [float(np.quantile(bootstrap, 0.025)), float(np.quantile(bootstrap, 0.975))],
        "discordant_method_wins": wins,
        "discordant_baseline_wins": losses,
        "mcnemar_exact_pvalue": _mcnemar_exact(wins, losses),
        "jointly_successful_count": int(len(jointly_successful_steps)),
        "jointly_successful_median_step_delta": (
            float(np.median(jointly_successful_steps)) if jointly_successful_steps else None
        ),
        "tasks": tasks,
    }


def main(argv=None):
    args = parse_args(argv)
    report = paired_report(
        load_results(args.results_dir),
        args.baseline,
        args.method,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )
    rendered = json.dumps(report, indent=2)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as stream:
            stream.write(rendered + "\n")


if __name__ == "__main__":
    main()
