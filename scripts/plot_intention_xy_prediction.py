"""Сравнение истинных координат состояния с предсказанием декодера."""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

np.in1d = np.isin

from baseline.frozen_fb import FrozenFB, load_checkpoint_config
from evaluation.visualization import _draw_maze_walls
from probes.intention_xy import IntentionXYDecoder
from utils.datasets import Dataset
from utils.env_utils import make_env_and_datasets


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            'Сравнение истинных координат состояния с предсказанием декодера.'
        )
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("checkpoints/antmaze-medium-navigate-v0"),
    help='Каталог замороженного агента с params.pkl и flags.json.',
    )
    parser.add_argument(
        "--decoder",
        type=Path,
        default=Path("artifacts/intention_xy_decoder_bmirror"),
        help='Каталог обученного декодера или путь к decoder.npz.',
    )
    parser.add_argument(
        "--dataset",
        choices=("train", "validation"),
        default="train",
    help='Часть офлайн-набора: train либо validation.',
    )
    parser.add_argument(
        "--dataset-index",
        type=int,
        default=None,
        help='Точный индекс состояния; без него выбор определяется параметром --seed.',
    )
    parser.add_argument("--seed", type=int, default=0, help='Воспроизводимая инициализация обучения и разбиения данных.')
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/intention_xy_decoder_bmirror/diagnostics"),
    help='Каталог сохранения моделей, оценок и промежуточных данных.',
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help='Дополнительно открывает окно графика после сохранения.',
    )
    return parser.parse_args(argv)


def _select_index(size: int, explicit_index: int | None, seed: int) -> int:
    if size <= 0:
        raise ValueError("the selected dataset is empty")
    if explicit_index is None:
        return int(np.random.default_rng(seed).integers(size))
    if explicit_index < 0 or explicit_index >= size:
        raise ValueError(
            f"dataset index {explicit_index} is outside the valid range [0, {size})"
        )
    return int(explicit_index)


def _load_runtime(checkpoint: Path, dataset_name: str):
    config, saved_flags = load_checkpoint_config(checkpoint)
    env_name = saved_flags["env_name"]
    env, raw_train, raw_validation = make_env_and_datasets(
        env_name,
        frame_stack=config["frame_stack"],
        add_info=True,
    )
    train_dataset = Dataset.create(**raw_train)
    validation_dataset = (
        None if raw_validation is None else Dataset.create(**raw_validation)
    )

    dataset_module = importlib.import_module("utils.datasets")
    dataset_class = getattr(dataset_module, config["dataset_class"])
    train_for_agent = dataset_class(train_dataset, config)
    frozen_fb = FrozenFB.from_checkpoint(
        checkpoint,
        train_for_agent.sample(1),
        config=config,
    )

    if dataset_name == "train":
        selected_dataset = train_dataset
    else:
        if validation_dataset is None:
            raise ValueError(f"environment {env_name!r} has no validation dataset")
        selected_dataset = validation_dataset
    return env, selected_dataset, frozen_fb, env_name


def _plot_comparison(env, true_xy, predicted_xy, output_path: Path, *, title: str):
    fig, ax = plt.subplots(figsize=(8, 8))
    bounds = _draw_maze_walls(ax, env)

    ax.plot(
        [true_xy[0], predicted_xy[0]],
        [true_xy[1], predicted_xy[1]],
        linestyle="--",
        linewidth=2.0,
        color="#6c757d",
        label="decoding error",
        zorder=3,
    )
    ax.scatter(
        [true_xy[0]],
        [true_xy[1]],
        s=120,
        marker="o",
        color="#1f77b4",
        edgecolor="white",
        linewidth=1.0,
        label="true state XY",
        zorder=4,
    )
    ax.scatter(
        [predicted_xy[0]],
        [predicted_xy[1]],
        s=150,
        marker="X",
        color="#ff7f0e",
        edgecolor="white",
        linewidth=1.0,
        label="decoded XY",
        zorder=5,
    )

    error = float(np.linalg.norm(predicted_xy - true_xy))
    midpoint = (true_xy + predicted_xy) / 2.0
    ax.annotate(
        f"error = {error:.3f}",
        xy=midpoint,
        xytext=(8, 8),
        textcoords="offset points",
        zorder=6,
    )

    maze_unit = bounds["maze_unit"]
    margin = 0.75 * maze_unit
    ax.set_xlim(
        min(bounds["xmin"], true_xy[0], predicted_xy[0]) - margin,
        max(bounds["xmax"], true_xy[0], predicted_xy[0]) + margin,
    )
    ax.set_ylim(
        min(bounds["ymin"], true_xy[1], predicted_xy[1]) - margin,
        max(bounds["ymax"], true_xy[1], predicted_xy[1]) + margin,
    )
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title)
    ax.grid(False)
    ax.legend(loc="upper left")
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    return fig


def main(argv=None):
    args = parse_args(argv)
    env, dataset, frozen_fb, env_name = _load_runtime(
        args.checkpoint,
        args.dataset,
    )
    decoder = IntentionXYDecoder.load(args.decoder)
    if decoder.latent_dim != frozen_fb.latent_dim:
        raise ValueError(
            f"decoder latent_dim={decoder.latent_dim} does not match "
            f"checkpoint latent_dim={frozen_fb.latent_dim}"
        )

    observations = np.asarray(dataset["observations"], dtype=np.float32)
    if observations.ndim != 2 or observations.shape[1] < 2:
        raise ValueError(f"unexpected observation array shape: {observations.shape}")
    index = _select_index(len(observations), args.dataset_index, args.seed)
    observation = observations[index]

    raw_backward = frozen_fb.backward_repr(jnp.asarray(observation)[None])[0]
    intention = frozen_fb.normalize_latent(raw_backward)
    predicted_xy = np.asarray(decoder.predict(intention), dtype=np.float64)
    true_xy = np.asarray(observation[:2], dtype=np.float64)
    if predicted_xy.shape != (2,) or not np.all(np.isfinite(predicted_xy)):
        raise RuntimeError(f"decoder returned invalid XY: {predicted_xy}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{args.dataset}_index_{index}"
    image_path = args.output_dir / f"{stem}.png"
    json_path = args.output_dir / f"{stem}.json"
    error = float(np.linalg.norm(predicted_xy - true_xy))
    record = {
        "environment": env_name,
        "checkpoint": str(args.checkpoint),
        "decoder": str(args.decoder),
        "dataset": args.dataset,
        "dataset_index": index,
        "true_xy": true_xy.tolist(),
        "decoded_xy": predicted_xy.tolist(),
        "euclidean_error": error,
        "raw_backward_norm": float(np.linalg.norm(np.asarray(raw_backward))),
        "intention_norm": float(np.linalg.norm(np.asarray(intention))),
    }
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(record, f, indent=2)

    fig = _plot_comparison(
        env,
        true_xy,
        predicted_xy,
        image_path,
        title=f"{args.dataset}[{index}]: true XY vs decoded normalize(B(s))",
    )
    print(json.dumps(record, indent=2))
    print(f"image: {image_path}")
    print(f"data: {json_path}")
    if args.show:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()
