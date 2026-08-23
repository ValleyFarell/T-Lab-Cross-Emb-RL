"""Обучение декодера нормализованных намерений в координаты карты."""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np

np.in1d = np.isin

from baseline.frozen_fb import FrozenFB, load_checkpoint_config
from probes.intention_xy import (
    IntentionXYDecoder,
    fit_decoder,
    regression_metrics,
    split_dataset_indices,
)
from utils.datasets import Dataset
from utils.env_utils import make_env_and_datasets


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description='Обучение декодера нормализованных намерений в координаты карты.'
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("checkpoints/antmaze-medium-navigate-v0"),
    help='Каталог замороженного агента с params.pkl и flags.json.',
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/intention_xy_decoder_bmirror"),
    help='Каталог сохранения моделей, оценок и промежуточных данных.',
    )
    parser.add_argument("--seed", type=int, default=0, help='Воспроизводимая инициализация обучения и разбиения данных.')
    parser.add_argument("--max-samples", type=int, default=300_000, help='Максимальное число офлайн-состояний для обучения.')
    parser.add_argument("--encoding-batch-size", type=int, default=4096, help='Размер блока вычисления B(s) замороженным агентом.')
    parser.add_argument("--hidden-dims", nargs="+", type=int, default=(512, 512, 512), help='Размер каждого скрытого слоя декодера.')
    parser.add_argument("--batch-size", type=int, default=1024, help='Размер обучающего блока.')
    parser.add_argument("--learning-rate", type=float, default=3e-4, help='Размер шага оптимизации.')
    parser.add_argument("--max-epochs", type=int, default=500, help='Максимальное число проходов по обучающим данным.')
    parser.add_argument("--patience", type=int, default=50, help='Число проходов без улучшения до остановки.')
    parser.add_argument("--weight-decay", type=float, default=1e-5, help='Сила регуляризации весов.')
    parser.add_argument("--warmup-epochs", type=int, default=5, help='Число начальных проходов плавного увеличения шага обучения.')
    parser.add_argument("--gradient-clip-norm", type=float, default=1.0, help='Верхняя граница нормы градиента.')
    parser.add_argument(
        "--target-rmse",
        type=float,
        default=0.3,
        help='Желаемая среднеквадратичная ошибка одной координаты на отложенных траекториях.',
    )
    args = parser.parse_args(argv)
    for name in ("max_samples", "encoding_batch_size", "batch_size", "max_epochs", "patience"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.learning_rate <= 0:
        parser.error("--learning-rate must be positive")
    if args.weight_decay < 0:
        parser.error("--weight-decay must be non-negative")
    if any(width <= 0 for width in args.hidden_dims):
        parser.error("--hidden-dims values must be positive")
    if args.warmup_epochs < 0 or args.warmup_epochs >= args.max_epochs:
        parser.error("--warmup-epochs must be non-negative and below --max-epochs")
    if args.gradient_clip_norm <= 0:
        parser.error("--gradient-clip-norm must be positive")
    if args.target_rmse <= 0:
        parser.error("--target-rmse must be positive")
    return args


def _encode_intentions(frozen_fb, observations, *, batch_size: int) -> np.ndarray:
    encoded = []
    for start in range(0, len(observations), batch_size):
        batch = jnp.asarray(observations[start : start + batch_size])
        backward = frozen_fb.backward_repr(batch)
        intentions = frozen_fb.normalize_latent(backward)
        encoded.append(np.asarray(intentions, dtype=np.float32))
    result = np.concatenate(encoded, axis=0)
    if result.ndim != 2 or result.shape[1] != frozen_fb.latent_dim:
        raise RuntimeError(f"unexpected encoded intention shape: {result.shape}")
    if not np.all(np.isfinite(result)):
        raise RuntimeError("encoded intentions contain non-finite values")
    return result


def _save_json(path: Path, value) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(value, f, indent=2)


def main(argv=None):
    args = parse_args(argv)
    config, saved_flags = load_checkpoint_config(args.checkpoint)
    env_name = saved_flags["env_name"]
    _, raw_train_dataset, _ = make_env_and_datasets(
        env_name,
        frame_stack=config["frame_stack"],
        add_info=True,
    )
    train_dataset = Dataset.create(**raw_train_dataset)

    dataset_module = importlib.import_module("utils.datasets")
    dataset_class = getattr(dataset_module, config["dataset_class"])
    dataset_for_agent = dataset_class(train_dataset, config)
    example_batch = dataset_for_agent.sample(1)
    frozen_fb = FrozenFB.from_checkpoint(
        args.checkpoint,
        example_batch,
        config=config,
    )
    if frozen_fb.latent_dim != 128:
        raise ValueError(
            "this requested decoder expects a 128-dimensional intention, but "
            "the checkpoint has "
            f"latent_dim={frozen_fb.latent_dim}"
        )

    observations = np.asarray(train_dataset["observations"], dtype=np.float32)
    if observations.ndim != 2 or observations.shape[1] < 2:
        raise ValueError(f"unexpected observation array shape: {observations.shape}")
    terminals = train_dataset.get("terminals")
    splits, split_strategy = split_dataset_indices(
        len(observations),
        terminals=terminals,
        seed=args.seed,
        max_samples=args.max_samples,
    )

    split_data = {}
    for split_name, indices in splits.items():
        split_observations = observations[indices]
        split_data[split_name] = {
            "z": _encode_intentions(
                frozen_fb,
                split_observations,
                batch_size=args.encoding_batch_size,
            ),
            "xy": split_observations[:, :2],
        }

    decoder, training = fit_decoder(
        split_data["train"]["z"],
        split_data["train"]["xy"],
        split_data["validation"]["z"],
        split_data["validation"]["xy"],
        seed=args.seed,
        hidden_dims=tuple(args.hidden_dims),
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        patience=args.patience,
        weight_decay=args.weight_decay,
        warmup_epochs=args.warmup_epochs,
        gradient_clip_norm=args.gradient_clip_norm,
    )

    metrics = {
        name: regression_metrics(decoder, data["z"], data["xy"])
        for name, data in split_data.items()
    }
    metrics["target"] = {
        "selection_metric": "validation.rmse_xy",
        "threshold": args.target_rmse,
        "validation_observed": metrics["validation"]["rmse_xy"],
        "validation_reached": bool(
            metrics["validation"]["rmse_xy"] <= args.target_rmse
        ),
        "test_observed_final_audit": metrics["test"]["rmse_xy"],
        "test_reached_final_audit": bool(
            metrics["test"]["rmse_xy"] <= args.target_rmse
        ),
    }
    metadata = {
        "model": "intention_xy_probe",
        "architecture": [128, *args.hidden_dims, 2],
        "hidden_block_1": "dense_layer_norm_tanh",
        "remaining_hidden_blocks": "dense_gelu",
        "output_block": "dense_linear",
        "latent_definition": "normalize(B(observation))",
        "target_definition": "observation[:2]",
        "checkpoint": str(args.checkpoint),
        "environment": env_name,
        "seed": args.seed,
        "split_strategy": split_strategy,
        "split_sizes": {name: int(len(indices)) for name, indices in splits.items()},
        "optimizer": "adam_with_global_gradient_clipping",
        "learning_rate": args.learning_rate,
        "learning_rate_schedule": "linear_warmup_then_cosine_decay",
        "batch_size": args.batch_size,
        "max_epochs": args.max_epochs,
        "patience": args.patience,
        "weight_decay": args.weight_decay,
        "warmup_epochs": args.warmup_epochs,
        "gradient_clip_norm": args.gradient_clip_norm,
        "target_rmse_xy": args.target_rmse,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    decoder.save(args.output_dir, metadata=metadata)
    _save_json(args.output_dir / "metrics.json", metrics)
    _save_json(args.output_dir / "training_history.json", training)

    reloaded = IntentionXYDecoder.load(args.output_dir)
    if not np.allclose(
        np.asarray(reloaded.predict(split_data["test"]["z"][:32])),
        np.asarray(decoder.predict(split_data["test"]["z"][:32])),
        atol=1e-6,
        rtol=1e-6,
    ):
        raise RuntimeError("saved decoder failed the reload equivalence check")

    print(json.dumps({"metadata": metadata, "metrics": metrics}, indent=2))
    print(f"saved_to: {args.output_dir}")


if __name__ == "__main__":
    main()
