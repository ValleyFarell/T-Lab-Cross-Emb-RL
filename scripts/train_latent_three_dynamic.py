"""Обучение четырёхмерной геометрии и декодера намерений."""

from __future__ import annotations

import argparse
from datetime import datetime
import importlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from hypotheses.latent_three_dynamic.geometry import (
    LatentGeometryModel,
    LatentIntentionDecoder,
    blocked_split,
    normalize_intentions,
)


def _log(message: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {message}", flush=True)


def configure_device(device: str) -> None:
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")
    if device == "cpu":
        os.environ["JAX_PLATFORM_NAME"] = "cpu"
        os.environ["JAX_PLATFORMS"] = "cpu"
    elif device == "gpu":
        os.environ["JAX_PLATFORM_NAME"] = "gpu"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/antmaze-medium-navigate-v0"), help='Каталог замороженного агента с params.pkl и flags.json.')
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/latent_three_dynamic"), help='Каталог сохранения моделей, оценок и промежуточных данных.')
    parser.add_argument("--geometry-model", type=Path, default=None, help='Уже обученная совместимая модель геометрии.')
    parser.add_argument("--cache", type=Path, default=None, help='Ранее сохранённые оценки замороженного критика.')
    parser.add_argument("--force-recompute", action="store_true", help='Принудительно пересчитывает сохранённые оценки критика.')
    parser.add_argument("--device", choices=("cpu", "gpu", "auto"), default="cpu", help='Устройство вычислений: cpu, gpu или auto, если скрипт поддерживает эти варианты.')
    parser.add_argument("--max-states", type=int, default=40_000, help='Верхняя граница числа используемых офлайн-состояний.')
    parser.add_argument("--train-pairs", type=int, default=80_000, help='Число обучающих пар «начальное состояние — цель».')
    parser.add_argument("--goal-count", type=int, default=256, help='Число различных целевых состояний.')
    parser.add_argument("--teacher-batch-size", type=int, default=128, help='Размер блока запросов к замороженному FB-критику.')
    parser.add_argument("--hidden-dim", type=int, default=96, help='Ширина основной модели ценности.')
    parser.add_argument("--decoder-hidden-dim", type=int, default=128, help='Ширина скрытых слоёв декодера намерений.')
    parser.add_argument("--epochs", type=int, default=60, help='Максимальное число проходов обучения модели геометрии.')
    parser.add_argument("--decoder-epochs", type=int, default=80, help='Максимальное число проходов обучения декодера.')
    parser.add_argument("--batch-size", type=int, default=512, help='Размер обучающего блока.')
    parser.add_argument("--learning-rate", type=float, default=0.001, help='Размер шага оптимизации.')
    parser.add_argument("--patience", type=int, default=12, help='Число проходов без улучшения до остановки.')
    parser.add_argument("--validation-fraction", type=float, default=0.15, help='Доля данных, отложенная для проверки во время обучения.')
    parser.add_argument("--block-size", type=int, default=128, help='Размер соседних временных блоков при разделении данных.')
    parser.add_argument("--seed", type=int, default=0, help='Воспроизводимая инициализация обучения и разбиения данных.')
    args = parser.parse_args(argv)
    for name in (
        "max_states", "train_pairs", "goal_count", "teacher_batch_size", "hidden_dim",
        "decoder_hidden_dim", "epochs", "decoder_epochs", "batch_size", "patience", "block_size",
    ):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if not np.isfinite(args.learning_rate) or args.learning_rate <= 0:
        parser.error("--learning-rate must be positive and finite")
    if not 0 < args.validation_fraction < 0.5:
        parser.error("--validation-fraction must be between 0 and 0.5")
    return args


def load_frozen_components(checkpoint: Path):
    np.in1d = np.isin
    from baseline.frozen_fb import FrozenFB, load_checkpoint_config
    from utils.datasets import Dataset
    from utils.env_utils import make_env_and_datasets

    config, saved_flags = load_checkpoint_config(checkpoint)
    environment_name = saved_flags["env_name"]
    _log(f"Loading environment and offline dataset: {environment_name}")
    env, raw_train, _ = make_env_and_datasets(
        environment_name,
        frame_stack=config["frame_stack"],
        add_info=True,
    )
    train_dataset = Dataset.create(**raw_train)
    dataset_module = importlib.import_module("utils.datasets")
    dataset_class = getattr(dataset_module, config["dataset_class"])
    agent_dataset = dataset_class(train_dataset, config)
    _log("Restoring frozen FB checkpoint")
    frozen_fb = FrozenFB.from_checkpoint(checkpoint, agent_dataset.sample(1), config=config)
    if hasattr(env, "close"):
        env.close()
    return frozen_fb, train_dataset, environment_name


def _forward_teacher(frozen_fb, starts, intentions):
    forward = np.asarray(frozen_fb.forward_repr(starts, intentions), dtype=np.float32)
    if forward.ndim == 2:
        forward = forward[None, :, :]
    if forward.ndim != 3 or forward.shape[1] != len(starts):
        raise RuntimeError(f"unexpected frozen forward shape: {forward.shape}")
    values = np.einsum("end,nd->en", forward, intentions)
    return values.mean(axis=0).astype(np.float32)


def build_teacher_cache(args, frozen_fb, dataset) -> dict[str, np.ndarray]:
    all_states = np.asarray(dataset["observations"], dtype=np.float32)
    if all_states.ndim != 2 or len(all_states) < 16:
        raise ValueError("offline dataset must contain at least 16 state vectors")
    if len(all_states) > args.max_states:
        source_indices = np.linspace(0, len(all_states) - 1, args.max_states, dtype=np.int64)
    else:
        source_indices = np.arange(len(all_states), dtype=np.int64)
    states = all_states[source_indices]
    train_states, validation_states = blocked_split(
        len(states),
        seed=args.seed,
        validation_fraction=args.validation_fraction,
        block_size=args.block_size,
    )
    _log(
        f"Selected {len(states)} states; train={len(train_states)}, "
        f"validation={len(validation_states)}, intention_dim={frozen_fb.latent_dim}"
    )

    intentions = np.empty((len(states), int(frozen_fb.latent_dim)), dtype=np.float32)
    for start in range(0, len(states), args.teacher_batch_size):
        stop = min(start + args.teacher_batch_size, len(states))
        backward = frozen_fb.backward_repr(states[start:stop])
        intentions[start:stop] = np.asarray(frozen_fb.normalize_latent(backward), dtype=np.float32)
        if start == 0 or stop == len(states) or stop % 5000 < args.teacher_batch_size:
            _log(f"Prepared frozen B intentions: {stop}/{len(states)}")
    intentions = normalize_intentions(intentions, int(frozen_fb.latent_dim))

    rng = np.random.default_rng(args.seed)
    validation_pair_count = max(1, int(round(args.train_pairs * args.validation_fraction)))
    train_pair_count = args.train_pairs - validation_pair_count
    if train_pair_count <= 0:
        raise ValueError("--train-pairs is too small for the configured validation fraction")

    train_goal_count = min(len(train_states), max(1, args.goal_count))
    validation_goal_count = min(len(validation_states), max(1, int(round(args.goal_count * args.validation_fraction))))
    train_goals = rng.choice(train_states, size=train_goal_count, replace=False)
    validation_goals = rng.choice(validation_states, size=validation_goal_count, replace=False)
    start_indices = np.concatenate(
        [
            rng.choice(train_states, size=train_pair_count, replace=True),
            rng.choice(validation_states, size=validation_pair_count, replace=True),
        ]
    ).astype(np.int64)
    goal_indices = np.concatenate(
        [
            rng.choice(train_goals, size=train_pair_count, replace=True),
            rng.choice(validation_goals, size=validation_pair_count, replace=True),
        ]
    ).astype(np.int64)
    validation_mask = np.zeros(len(start_indices), dtype=bool)
    validation_mask[train_pair_count:] = True

    values = np.empty(len(start_indices), dtype=np.float32)
    for start in range(0, len(values), args.teacher_batch_size):
        stop = min(start + args.teacher_batch_size, len(values))
        values[start:stop] = _forward_teacher(
            frozen_fb,
            states[start_indices[start:stop]],
            intentions[goal_indices[start:stop]],
        )
        if start == 0 or stop == len(values) or stop % 5000 < args.teacher_batch_size:
            _log(f"Prepared frozen FB value labels: {stop}/{len(values)}")

    finite = np.isfinite(values)
    if not np.all(finite):
        raise RuntimeError(f"frozen critic produced {(~finite).sum()} non-finite teacher values")
    return {
        "observations": states,
        "source_indices": source_indices,
        "intentions": intentions,
        "start_indices": start_indices,
        "goal_indices": goal_indices,
        "teacher_values": values,
        "validation_mask": validation_mask,
        "train_state_indices": train_states,
        "validation_state_indices": validation_states,
    }


def _load_cache(path: Path) -> dict[str, np.ndarray]:
    required = {
        "observations", "source_indices", "intentions", "start_indices", "goal_indices",
        "teacher_values", "validation_mask", "train_state_indices", "validation_state_indices",
    }
    with np.load(path, allow_pickle=False) as loaded:
        missing = required.difference(loaded.files)
        if missing:
            raise ValueError("teacher cache is missing arrays: " + ", ".join(sorted(missing)))
        return {name: np.asarray(loaded[name]) for name in required}


def evaluate_decoder(decoder, embeddings, intentions, validation_indices):
    predicted = decoder.predict(embeddings[validation_indices])
    expected = normalize_intentions(intentions[validation_indices], decoder.latent_dim)
    cosine = np.sum(predicted * expected, axis=-1) / decoder.latent_dim
    errors = np.linalg.norm(predicted - expected, axis=-1)
    return {
        "validation_count": int(len(validation_indices)),
        "mean_cosine": float(cosine.mean()),
        "median_cosine": float(np.median(cosine)),
        "p10_cosine": float(np.quantile(cosine, 0.10)),
        "mean_l2_error": float(errors.mean()),
        "p90_l2_error": float(np.quantile(errors, 0.90)),
        "predicted_norm_mean": float(np.linalg.norm(predicted, axis=-1).mean()),
        "expected_norm": float(np.sqrt(decoder.latent_dim)),
    }


def main(argv=None):
    args = parse_args(argv)
    configure_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cache_path = args.cache if args.cache is not None else args.output_dir / "teacher_cache.npz"

    if cache_path.exists() and not args.force_recompute:
        _log(f"Reusing existing frozen-FB teacher cache: {cache_path}")
        cache = _load_cache(cache_path)
        environment_name = "cached"
    else:
        frozen_fb, dataset, environment_name = load_frozen_components(args.checkpoint)
        cache = build_teacher_cache(args, frozen_fb, dataset)
        np.savez_compressed(cache_path, **cache)
        _log(f"Saved teacher cache: {cache_path}")

    if args.geometry_model is not None:
        _log(f"Reusing compatible 4D geometry model: {args.geometry_model}")
        geometry = LatentGeometryModel.load(args.geometry_model)
        if geometry.observation_dim != cache["observations"].shape[1]:
            raise ValueError("geometry observation dimension does not match the teacher cache")
        geometry_report = {"source": str(args.geometry_model), "retrained": False}
    else:
        _log("Training shared four-dimensional encoder and pairwise value head on CPU/NumPy")
        geometry, fitted = LatentGeometryModel.fit(
            cache["observations"],
            cache["start_indices"],
            cache["goal_indices"],
            cache["teacher_values"],
            validation_mask=cache["validation_mask"],
            hidden_dim=args.hidden_dim,
            value_hidden_dim=args.hidden_dim,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            patience=args.patience,
            seed=args.seed,
            progress=_log,
        )
        geometry_report = fitted.as_dict()
        geometry_report["retrained"] = True

    embeddings = geometry.encode(cache["observations"])
    _log("Training 4D-to-128D decoder on frozen B targets")
    decoder, decoder_fit = LatentIntentionDecoder.fit(
        embeddings,
        cache["intentions"],
        train_indices=cache["train_state_indices"],
        validation_indices=cache["validation_state_indices"],
        hidden_dim=args.decoder_hidden_dim,
        epochs=args.decoder_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        patience=args.patience,
        seed=args.seed,
        progress=_log,
    )

    metadata: dict[str, Any] = {
        "environment": environment_name,
        "checkpoint": str(args.checkpoint),
        "observation_dim": int(cache["observations"].shape[1]),
        "embedding_dim": int(geometry.embedding_dim),
        "intention_dim": int(decoder.latent_dim),
        "state_count": int(len(cache["observations"])),
        "pair_count": int(len(cache["teacher_values"])),
        "seed": int(args.seed),
        "validation_split": "contiguous_state_blocks",
        "teacher_target": "mean_ensemble(F(s, normalize(B(g)))) dot normalize(B(g))",
        "frozen_components": ["F", "B", "low_level_policy"],
    }
    geometry.save(args.output_dir, metadata)
    decoder.save(args.output_dir, metadata)
    metrics = {
        "metadata": metadata,
        "geometry": geometry_report,
        "decoder_fit": decoder_fit.as_dict(),
        "decoder_validation": evaluate_decoder(
            decoder,
            embeddings,
            cache["intentions"],
            cache["validation_state_indices"],
        ),
    }
    with (args.output_dir / "training_metrics.json").open("w", encoding="utf-8") as stream:
        json.dump(metrics, stream, indent=2)
    _log(
        "Training complete: "
        f"geometry_val_r2={geometry_report.get('validation_r2', 'reused')}, "
        f"decoder_mean_cosine={metrics['decoder_validation']['mean_cosine']:.4f}, "
        f"output={args.output_dir}"
    )


if __name__ == "__main__":
    main()
