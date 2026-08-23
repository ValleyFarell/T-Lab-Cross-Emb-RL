"""Проверка зависимости FB-ценности от физических и скрытых координат."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys


def _configure_runtime_before_jax(argv: list[str]) -> None:
    """Применяет настройки памяти и устройства до первого импорта JAX."""

    device = "cpu"
    for index, argument in enumerate(argv):
        if argument == "--device" and index + 1 < len(argv):
            device = argv[index + 1]
        elif argument.startswith("--device="):
            device = argument.partition("=")[2]
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    if device == "cpu":
        os.environ["JAX_PLATFORMS"] = "cpu"
    elif device in {"auto", "gpu"}:
        # Сохраняем явно выбранное пользователем устройство для автоматического режима.
        if os.environ.get("JAX_PLATFORMS") == "cpu":
            os.environ.pop("JAX_PLATFORMS", None)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            'Проверка зависимости FB-ценности от физических и скрытых координат.'
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("checkpoints/antmaze-medium-navigate-v0"),
    help='Каталог замороженного агента с params.pkl и flags.json.',
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("artifacts/value_geometry")
    , help='Каталог сохранения моделей, оценок и промежуточных данных.')
    parser.add_argument(
        "--target-mode",
        choices=("xy-goal", "state-goal", "fixed-task"),
        default="xy-goal",
        help=(
            'Тип предсказываемой задачи: пространственная цель, полное состояние или фиксированная задача.'
        ),
    )
    parser.add_argument(
        "--task-id", type=int, default=4,
        help='Номер официальной задачи OGBench.',
    )
    parser.add_argument(
        "--device",
        choices=("cpu", "auto", "gpu"),
        default="cpu",
        help='Устройство замороженного JAX-критика; вспомогательное обучение выполняется на процессоре.',
    )
    parser.add_argument("--seed", type=int, default=0, help='Воспроизводимая инициализация выбора данных или обучения.')
    parser.add_argument(
        "--model-seeds", type=int, nargs="+", default=[0],
        help='Независимые начальные инициализации вспомогательных моделей.',
    )
    parser.add_argument("--max-states", type=int, default=18_000, help='Верхняя граница числа используемых офлайн-состояний.')
    parser.add_argument("--train-pairs", type=int, default=20_000, help='Число обучающих пар «начальное состояние — цель».')
    parser.add_argument("--goal-count", type=int, default=64, help='Число различных целевых состояний.')
    parser.add_argument("--goal-variants", type=int, default=2, help='Число разных полных состояний возле близких координат цели.')
    parser.add_argument("--pose-radius", type=float, default=0.20, help='Радиус, в котором целевые положения считаются пространственно близкими.')
    parser.add_argument("--candidates-per-start", type=int, default=8, help='Число целевых состояний для одного начального состояния.')
    parser.add_argument("--teacher-batch-size", type=int, default=128, help='Размер блока запросов к замороженному FB-критику.')
    parser.add_argument("--reference-samples", type=int, default=100_000, help='Число состояний для построения представления конечной награды.')
    parser.add_argument("--disagreement-penalty", type=float, default=0.5, help='Штраф за расхождение участников FB-ансамбля.')
    parser.add_argument(
        "--distance-mode", choices=("auto", "maze", "euclidean"), default="auto"
    , help='maze, euclidean или auto: расстояние по коридорам, по прямой либо автоматический выбор.')
    parser.add_argument("--distance-bins", type=int, default=4, help='Число диапазонов расстояния для отдельного анализа.')
    parser.add_argument(
        "--models", nargs="+", default=["xy", "full", "latent2", "latent4"],
        help='Список сравниваемых архитектур вспомогательных моделей.',
    )
    parser.add_argument("--hidden-width", type=int, default=64, help='Ширина скрытых слоёв вспомогательной модели.')
    parser.add_argument("--hidden-layers", type=int, default=2, help='Число скрытых слоёв.')
    parser.add_argument("--batch-size", type=int, default=256, help='Размер обучающего блока.')
    parser.add_argument("--epochs", type=int, default=80, help='Максимальное число проходов обучения модели геометрии.')
    parser.add_argument("--patience", type=int, default=12, help='Число проходов без улучшения до остановки.')
    parser.add_argument("--learning-rate", type=float, default=2e-3, help='Размер шага оптимизации.')
    parser.add_argument("--weight-decay", type=float, default=1e-5, help='Сила регуляризации весов.')
    parser.add_argument("--gradient-clip", type=float, default=5.0, help='Верхняя граница нормы градиента.')
    parser.add_argument("--log-every", type=int, default=5, help='Как часто печатать ход обучения.')
    parser.add_argument(
        "--quick", action="store_true",
        help='Уменьшает размер эксперимента для быстрой предварительной проверки.',
    )
    parser.add_argument(
        "--synthetic", action="store_true",
        help='Проверяет весь конвейер на искусственных данных без чекпоинта.',
    )
    parser.add_argument("--synthetic-states", type=int, default=3_000, help='Количество искусственных состояний.')
    parser.add_argument(
        "--resume", action="store_true",
        help='Переиспользует ранее сохранённые пары и оценки замороженного критика.',
    )
    parser.add_argument("--no-plots", action="store_true", help='Отключает сохранение графиков.')
    args = parser.parse_args(argv)
    positive = (
        "max_states", "train_pairs", "goal_count", "goal_variants",
        "candidates_per_start", "teacher_batch_size", "reference_samples",
        "distance_bins", "hidden_width", "hidden_layers", "batch_size",
        "epochs", "patience", "log_every", "synthetic_states",
    )
    for name in positive:
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.pose_radius < 0:
        parser.error("--pose-radius must be non-negative")
    if args.learning_rate <= 0 or args.gradient_clip <= 0:
        parser.error("--learning-rate and --gradient-clip must be positive")
    if args.weight_decay < 0 or args.disagreement_penalty < 0:
        parser.error("--weight-decay and --disagreement-penalty must be non-negative")
    if args.quick:
        args.max_states = min(args.max_states, 6_000)
        args.train_pairs = min(args.train_pairs, 4_000)
        args.goal_count = min(args.goal_count, 32)
        args.epochs = min(args.epochs, 30)
        args.patience = min(args.patience, 8)
        args.hidden_width = min(args.hidden_width, 48)
        args.synthetic_states = min(args.synthetic_states, 1_800)
    return args


def main(argv: list[str] | None = None):
    arguments = list(sys.argv[1:] if argv is None else argv)
    _configure_runtime_before_jax(arguments)
    args = parse_args(arguments)
    from hypotheses.value_geometry.experiment import ExperimentConfig, run_experiment
    from hypotheses.value_geometry.models import TrainingConfig

    training = TrainingConfig(
        hidden_width=args.hidden_width,
        hidden_layers=args.hidden_layers,
        batch_size=args.batch_size,
        max_epochs=args.epochs,
        patience=args.patience,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        gradient_clip=args.gradient_clip,
        log_every=args.log_every,
    )
    config = ExperimentConfig(
        checkpoint=str(args.checkpoint),
        output_dir=str(args.output_dir),
        target_mode=args.target_mode,
        task_id=args.task_id,
        seed=args.seed,
        model_seeds=tuple(args.model_seeds),
        max_states=args.max_states,
        train_pairs=args.train_pairs,
        goal_count=args.goal_count,
        goal_variants=args.goal_variants,
        pose_radius=args.pose_radius,
        candidates_per_start=args.candidates_per_start,
        teacher_batch_size=args.teacher_batch_size,
        reference_samples=args.reference_samples,
        disagreement_penalty=args.disagreement_penalty,
        distance_mode=args.distance_mode,
        distance_bins=args.distance_bins,
        models=tuple(dict.fromkeys(args.models)),
        training=training,
        synthetic=args.synthetic,
        synthetic_states=args.synthetic_states,
        resume=args.resume,
        plots=not args.no_plots,
    )
    return run_experiment(config)


if __name__ == "__main__":
    main()
