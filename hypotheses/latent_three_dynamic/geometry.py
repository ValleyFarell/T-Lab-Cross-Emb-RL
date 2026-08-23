"""Небольшие NumPy-модели четырёхмерной геометрии и декодера намерений."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np


EMBEDDING_DIM = 4
EPSILON = 1e-8


def _as_matrix(value: Any, *, name: str, width: int | None = None) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim != 2 or not len(array):
        raise ValueError(f"{name} must be a non-empty matrix, got {array.shape}")
    if width is not None and array.shape[1] != width:
        raise ValueError(f"{name} must have width {width}, got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains non-finite values")
    return array


def _batch_matrix(value: Any, *, width: int, name: str) -> tuple[np.ndarray, bool]:
    array = np.asarray(value, dtype=np.float32)
    single = array.ndim == 1
    if single:
        array = array[None, :]
    return _as_matrix(array, name=name, width=width), single


def normalize_intentions(values: Any, latent_dim: int | None = None) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    if array.ndim not in (1, 2) or not array.shape[-1]:
        raise ValueError(f"intentions must have shape (D,) or (N,D), got {array.shape}")
    dimension = int(latent_dim if latent_dim is not None else array.shape[-1])
    if array.shape[-1] != dimension:
        raise ValueError(f"expected intention dimension {dimension}, got {array.shape}")
    norm = np.linalg.norm(array, axis=-1, keepdims=True)
    if not np.all(np.isfinite(array)) or np.any(norm <= EPSILON):
        raise ValueError("intentions must be finite and have non-zero norm")
    return (array / norm * np.sqrt(dimension)).astype(np.float32)


def _glorot(rng: np.random.Generator, fan_in: int, fan_out: int) -> np.ndarray:
    limit = np.sqrt(6.0 / (fan_in + fan_out))
    return rng.uniform(-limit, limit, (fan_in, fan_out)).astype(np.float32)


def _adam_step(
    params: dict[str, np.ndarray],
    gradients: Mapping[str, np.ndarray],
    moments: dict[str, tuple[np.ndarray, np.ndarray]],
    step: int,
    *,
    learning_rate: float,
    weight_decay: float,
) -> None:
    beta1, beta2 = 0.9, 0.999
    for name, parameter in params.items():
        gradient = np.asarray(gradients[name], dtype=np.float32)
        if name.endswith("_w"):
            gradient = gradient + weight_decay * parameter
        first, second = moments.setdefault(
            name,
            (np.zeros_like(parameter), np.zeros_like(parameter)),
        )
        first *= beta1
        first += (1.0 - beta1) * gradient
        second *= beta2
        second += (1.0 - beta2) * np.square(gradient)
        first_corrected = first / (1.0 - beta1**step)
        second_corrected = second / (1.0 - beta2**step)
        parameter -= learning_rate * first_corrected / (
            np.sqrt(second_corrected) + 1e-8
        )


def blocked_split(
    size: int,
    *,
    seed: int = 0,
    validation_fraction: float = 0.15,
    block_size: int = 128,
) -> tuple[np.ndarray, np.ndarray]:
    """Сохраняет соседние состояния вместе, уменьшая временную утечку данных."""

    if size < 4:
        raise ValueError("at least four states are required for a blocked split")
    if not 0.0 < validation_fraction < 0.5:
        raise ValueError("validation_fraction must be between 0 and 0.5")
    block_size = max(1, min(int(block_size), max(1, size // 4)))
    block_ids = np.arange(size, dtype=np.int64) // block_size
    unique_blocks = np.unique(block_ids)
    if len(unique_blocks) < 2:
        unique_blocks = np.arange(size, dtype=np.int64)
        block_ids = unique_blocks.copy()
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(unique_blocks)
    validation_blocks = max(1, int(round(len(shuffled) * validation_fraction)))
    selected = shuffled[:validation_blocks]
    validation_mask = np.isin(block_ids, selected)
    train = np.flatnonzero(~validation_mask)
    validation = np.flatnonzero(validation_mask)
    if not len(train) or not len(validation):
        raise RuntimeError("blocked split produced an empty partition")
    return train, validation


@dataclass(frozen=True)
class TrainingReport:
    best_epoch: int
    epochs_ran: int
    training_loss: float
    validation_loss: float
    validation_r2: float
    history: tuple[dict[str, float], ...]
    validation_metric_name: str = "r2"

    def as_dict(self) -> dict[str, Any]:
        payload = {
            "best_epoch": self.best_epoch,
            "epochs_ran": self.epochs_ran,
            "training_loss": self.training_loss,
            "validation_loss": self.validation_loss,
            "history": list(self.history),
        }
        payload[f"validation_{self.validation_metric_name}"] = self.validation_r2
        return payload


@dataclass
class LatentGeometryModel:
    """Сжимает состояние и предсказывает значения замороженного FB-критика."""

    params: dict[str, np.ndarray]
    observation_mean: np.ndarray
    observation_scale: np.ndarray
    value_mean: float
    value_scale: float

    @property
    def observation_dim(self) -> int:
        return int(self.observation_mean.shape[0])

    @property
    def embedding_dim(self) -> int:
        return int(self.params["encoder_out_b"].shape[0])

    def _encode_batch(self, observations: np.ndarray):
        standardized = (observations - self.observation_mean) / self.observation_scale
        hidden = np.tanh(standardized @ self.params["encoder_w"] + self.params["encoder_b"])
        embedding = hidden @ self.params["encoder_out_w"] + self.params["encoder_out_b"]
        return embedding, (standardized, hidden)

    def encode(self, observations: Any) -> np.ndarray:
        matrix, single = _batch_matrix(
            observations,
            width=self.observation_dim,
            name="observations",
        )
        embedding, _ = self._encode_batch(matrix)
        result = embedding.astype(np.float32)
        return result[0] if single else result

    def _predict_standardized(self, starts: np.ndarray, goals: np.ndarray):
        features = np.concatenate(
            [starts, goals, starts - goals, starts * goals], axis=-1
        )
        hidden = np.tanh(features @ self.params["value_w"] + self.params["value_b"])
        prediction = (hidden @ self.params["value_out_w"] + self.params["value_out_b"]).ravel()
        return prediction, (features, hidden)

    def predict_value(self, starts: Any, goals: Any) -> np.ndarray | float:
        start_matrix, start_single = _batch_matrix(
            starts, width=self.embedding_dim, name="start_embeddings"
        )
        goal_matrix, goal_single = _batch_matrix(
            goals, width=self.embedding_dim, name="goal_embeddings"
        )
        if len(start_matrix) == 1 and len(goal_matrix) > 1:
            start_matrix = np.repeat(start_matrix, len(goal_matrix), axis=0)
        elif len(goal_matrix) == 1 and len(start_matrix) > 1:
            goal_matrix = np.repeat(goal_matrix, len(start_matrix), axis=0)
        if len(start_matrix) != len(goal_matrix):
            raise ValueError("start and goal embedding batches must have matching lengths")
        standardized, _ = self._predict_standardized(start_matrix, goal_matrix)
        values = standardized * self.value_scale + self.value_mean
        return float(values[0]) if start_single and goal_single else values.astype(np.float32)

    def save(self, output_dir: str | Path, metadata: Mapping[str, Any] | None = None) -> Path:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / "geometry.npz"
        arrays = {name: np.asarray(value) for name, value in self.params.items()}
        arrays.update(
            observation_mean=self.observation_mean,
            observation_scale=self.observation_scale,
            value_mean=np.asarray(self.value_mean, dtype=np.float32),
            value_scale=np.asarray(self.value_scale, dtype=np.float32),
        )
        np.savez_compressed(path, **arrays)
        if metadata is not None:
            with (output_dir / "geometry_config.json").open("w", encoding="utf-8") as stream:
                json.dump(dict(metadata), stream, indent=2)
        return path

    @classmethod
    def load(cls, path: str | Path) -> "LatentGeometryModel":
        path = Path(path)
        if path.is_dir():
            path = path / "geometry.npz"
        if not path.exists():
            raise FileNotFoundError(
                f"4D geometry model not found: {path}. Run "
                "python -m scripts.train_latent_three_dynamic first."
            )
        required = {
            "encoder_w", "encoder_b", "encoder_out_w", "encoder_out_b",
            "value_w", "value_b", "value_out_w", "value_out_b",
            "observation_mean", "observation_scale", "value_mean", "value_scale",
        }
        with np.load(path, allow_pickle=False) as loaded:
            missing = required.difference(loaded.files)
            if missing:
                raise ValueError(
                    "Incompatible geometry artifact; missing arrays: "
                    + ", ".join(sorted(missing))
                )
            params = {
                key: np.asarray(loaded[key], dtype=np.float32)
                for key in required
                if key.endswith("_w") or key.endswith("_b")
            }
            model = cls(
                params=params,
                observation_mean=np.asarray(loaded["observation_mean"], dtype=np.float32),
                observation_scale=np.asarray(loaded["observation_scale"], dtype=np.float32),
                value_mean=float(loaded["value_mean"]),
                value_scale=float(loaded["value_scale"]),
            )
        if model.embedding_dim != EMBEDDING_DIM:
            raise ValueError(f"expected a 4D geometry model, got {model.embedding_dim}D")
        return model

    @classmethod
    def fit(
        cls,
        observations: Any,
        start_indices: Any,
        goal_indices: Any,
        target_values: Any,
        *,
        validation_mask: Any,
        hidden_dim: int = 96,
        value_hidden_dim: int = 96,
        epochs: int = 60,
        batch_size: int = 512,
        learning_rate: float = 1e-3,
        patience: int = 12,
        seed: int = 0,
        progress: Callable[[str], None] | None = None,
    ) -> tuple["LatentGeometryModel", TrainingReport]:
        states = _as_matrix(observations, name="observations")
        starts = np.asarray(start_indices, dtype=np.int64).ravel()
        goals = np.asarray(goal_indices, dtype=np.int64).ravel()
        values = np.asarray(target_values, dtype=np.float32).ravel()
        validation = np.asarray(validation_mask, dtype=bool).ravel()
        if not (len(starts) == len(goals) == len(values) == len(validation)):
            raise ValueError("pair indices, values, and validation_mask must have equal lengths")
        if np.any(starts < 0) or np.any(goals < 0) or np.any(starts >= len(states)) or np.any(goals >= len(states)):
            raise ValueError("pair indices are outside the observation bank")
        if not np.all(np.isfinite(values)):
            raise ValueError("teacher values contain non-finite entries")
        train_pairs = np.flatnonzero(~validation)
        validation_pairs = np.flatnonzero(validation)
        if not len(train_pairs) or not len(validation_pairs):
            raise ValueError("training and validation pairs must both be non-empty")

        rng = np.random.default_rng(seed)
        train_state_indices = np.unique(np.concatenate([starts[train_pairs], goals[train_pairs]]))
        observation_mean = states[train_state_indices].mean(axis=0)
        observation_scale = np.maximum(states[train_state_indices].std(axis=0), 1e-4)
        value_mean = float(values[train_pairs].mean())
        value_scale = max(float(values[train_pairs].std()), 1e-4)
        params = {
            "encoder_w": _glorot(rng, states.shape[1], hidden_dim),
            "encoder_b": np.zeros(hidden_dim, dtype=np.float32),
            "encoder_out_w": _glorot(rng, hidden_dim, EMBEDDING_DIM),
            "encoder_out_b": np.zeros(EMBEDDING_DIM, dtype=np.float32),
            "value_w": _glorot(rng, EMBEDDING_DIM * 4, value_hidden_dim),
            "value_b": np.zeros(value_hidden_dim, dtype=np.float32),
            "value_out_w": _glorot(rng, value_hidden_dim, 1),
            "value_out_b": np.zeros(1, dtype=np.float32),
        }
        model = cls(params, observation_mean, observation_scale, value_mean, value_scale)
        moments: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        best_params = {name: value.copy() for name, value in params.items()}
        best_validation = float("inf")
        best_epoch = 0
        stalled = 0
        steps = 0
        history: list[dict[str, float]] = []

        def evaluate(indices: np.ndarray) -> tuple[float, float]:
            prediction_parts = []
            for offset in range(0, len(indices), max(batch_size, 2048)):
                selected = indices[offset : offset + max(batch_size, 2048)]
                prediction_parts.append(
                    np.asarray(
                        model.predict_value(
                            model.encode(states[starts[selected]]),
                            model.encode(states[goals[selected]]),
                        )
                    )
                )
            prediction = np.concatenate(prediction_parts)
            residual = prediction - values[indices]
            mse = float(np.mean(np.square(residual)))
            total = float(np.mean(np.square(values[indices] - values[indices].mean())))
            return mse, float(1.0 - mse / max(total, 1e-12))

        for epoch in range(1, epochs + 1):
            shuffled = rng.permutation(train_pairs)
            for offset in range(0, len(shuffled), batch_size):
                selected = shuffled[offset : offset + batch_size]
                start_embedding, start_cache = model._encode_batch(states[starts[selected]])
                goal_embedding, goal_cache = model._encode_batch(states[goals[selected]])
                predicted, value_cache = model._predict_standardized(start_embedding, goal_embedding)
                targets = (values[selected] - value_mean) / value_scale
                derivative = (2.0 / len(selected)) * (predicted - targets)
                features, value_hidden = value_cache
                derivative_column = derivative[:, None]
                gradients: dict[str, np.ndarray] = {
                    "value_out_w": value_hidden.T @ derivative_column,
                    "value_out_b": derivative_column.sum(axis=0),
                }
                value_pre = (derivative_column @ params["value_out_w"].T) * (1.0 - value_hidden**2)
                gradients["value_w"] = features.T @ value_pre
                gradients["value_b"] = value_pre.sum(axis=0)
                feature_derivative = value_pre @ params["value_w"].T
                dimension = EMBEDDING_DIM
                start_derivative = (
                    feature_derivative[:, :dimension]
                    + feature_derivative[:, 2 * dimension : 3 * dimension]
                    + feature_derivative[:, 3 * dimension :] * goal_embedding
                )
                goal_derivative = (
                    feature_derivative[:, dimension : 2 * dimension]
                    - feature_derivative[:, 2 * dimension : 3 * dimension]
                    + feature_derivative[:, 3 * dimension :] * start_embedding
                )
                start_standardized, start_hidden = start_cache
                goal_standardized, goal_hidden = goal_cache
                gradients["encoder_out_w"] = (
                    start_hidden.T @ start_derivative + goal_hidden.T @ goal_derivative
                )
                gradients["encoder_out_b"] = start_derivative.sum(axis=0) + goal_derivative.sum(axis=0)
                start_pre = (start_derivative @ params["encoder_out_w"].T) * (1.0 - start_hidden**2)
                goal_pre = (goal_derivative @ params["encoder_out_w"].T) * (1.0 - goal_hidden**2)
                gradients["encoder_w"] = start_standardized.T @ start_pre + goal_standardized.T @ goal_pre
                gradients["encoder_b"] = start_pre.sum(axis=0) + goal_pre.sum(axis=0)
                steps += 1
                _adam_step(
                    params,
                    gradients,
                    moments,
                    steps,
                    learning_rate=learning_rate,
                    weight_decay=1e-5,
                )

            train_subset = train_pairs[: min(len(train_pairs), 8192)]
            train_loss, _ = evaluate(train_subset)
            validation_loss, validation_r2 = evaluate(validation_pairs)
            history.append(
                {
                    "epoch": float(epoch),
                    "train_mse": train_loss,
                    "validation_mse": validation_loss,
                    "validation_r2": validation_r2,
                }
            )
            if progress is not None and (epoch == 1 or epoch % 5 == 0):
                progress(
                    f"geometry epoch={epoch} train_mse={train_loss:.5f} "
                    f"val_mse={validation_loss:.5f} val_r2={validation_r2:.4f}"
                )
            if validation_loss < best_validation - 1e-7:
                best_validation = validation_loss
                best_epoch = epoch
                best_params = {name: value.copy() for name, value in params.items()}
                stalled = 0
            else:
                stalled += 1
                if stalled >= patience:
                    break

        model.params = best_params
        final_train, _ = evaluate(train_pairs[: min(len(train_pairs), 8192)])
        final_validation, final_r2 = evaluate(validation_pairs)
        report = TrainingReport(
            best_epoch=best_epoch,
            epochs_ran=len(history),
            training_loss=final_train,
            validation_loss=final_validation,
            validation_r2=final_r2,
            history=tuple(history),
        )
        return model, report


@dataclass
class LatentIntentionDecoder:
    """Переводит четырёхмерное состояние в намерение низкоуровневой политики."""

    params: dict[str, np.ndarray]
    embedding_mean: np.ndarray
    embedding_scale: np.ndarray

    @property
    def embedding_dim(self) -> int:
        return int(self.embedding_mean.shape[0])

    @property
    def latent_dim(self) -> int:
        return int(self.params["out_b"].shape[0])

    def _forward(self, embeddings: np.ndarray):
        standardized = (embeddings - self.embedding_mean) / self.embedding_scale
        first = np.tanh(standardized @ self.params["first_w"] + self.params["first_b"])
        second = np.tanh(first @ self.params["second_w"] + self.params["second_b"])
        raw = second @ self.params["out_w"] + self.params["out_b"]
        return raw, (standardized, first, second)

    def predict(self, embeddings: Any) -> np.ndarray:
        matrix, single = _batch_matrix(
            embeddings, width=self.embedding_dim, name="embeddings"
        )
        raw, _ = self._forward(matrix)
        norms = np.linalg.norm(raw, axis=-1, keepdims=True)
        if np.any(norms <= EPSILON) or not np.all(np.isfinite(raw)):
            raise RuntimeError("decoder produced a non-finite or zero intention")
        normalized = raw / norms * np.sqrt(self.latent_dim)
        normalized = normalized.astype(np.float32)
        return normalized[0] if single else normalized

    decode = predict

    def save(self, output_dir: str | Path, metadata: Mapping[str, Any] | None = None) -> Path:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / "intention_decoder.npz"
        arrays = {name: np.asarray(value) for name, value in self.params.items()}
        arrays["embedding_mean"] = self.embedding_mean
        arrays["embedding_scale"] = self.embedding_scale
        np.savez_compressed(path, **arrays)
        if metadata is not None:
            with (output_dir / "intention_decoder_config.json").open("w", encoding="utf-8") as stream:
                json.dump(dict(metadata), stream, indent=2)
        return path

    @classmethod
    def load(cls, path: str | Path) -> "LatentIntentionDecoder":
        path = Path(path)
        if path.is_dir():
            path = path / "intention_decoder.npz"
        if not path.exists():
            raise FileNotFoundError(
                f"intention decoder not found: {path}. Run "
                "python -m scripts.train_latent_three_dynamic first."
            )
        required = {
            "first_w", "first_b", "second_w", "second_b", "out_w", "out_b",
            "embedding_mean", "embedding_scale",
        }
        with np.load(path, allow_pickle=False) as loaded:
            missing = required.difference(loaded.files)
            if missing:
                raise ValueError("decoder artifact is missing arrays: " + ", ".join(sorted(missing)))
            params = {
                key: np.asarray(loaded[key], dtype=np.float32)
                for key in required
                if key.endswith("_w") or key.endswith("_b")
            }
            decoder = cls(
                params=params,
                embedding_mean=np.asarray(loaded["embedding_mean"], dtype=np.float32),
                embedding_scale=np.asarray(loaded["embedding_scale"], dtype=np.float32),
            )
        if decoder.embedding_dim != EMBEDDING_DIM:
            raise ValueError(f"expected decoder input width 4, got {decoder.embedding_dim}")
        return decoder

    @classmethod
    def fit(
        cls,
        embeddings: Any,
        intentions: Any,
        *,
        train_indices: Any,
        validation_indices: Any,
        hidden_dim: int = 128,
        epochs: int = 80,
        batch_size: int = 512,
        learning_rate: float = 1e-3,
        patience: int = 15,
        seed: int = 0,
        progress: Callable[[str], None] | None = None,
    ) -> tuple["LatentIntentionDecoder", TrainingReport]:
        latent_states = _as_matrix(embeddings, name="embeddings", width=EMBEDDING_DIM)
        targets = normalize_intentions(_as_matrix(intentions, name="intentions"))
        if len(latent_states) != len(targets):
            raise ValueError("embeddings and intentions must have equal lengths")
        train = np.asarray(train_indices, dtype=np.int64).ravel()
        validation = np.asarray(validation_indices, dtype=np.int64).ravel()
        if not len(train) or not len(validation):
            raise ValueError("training and validation indices must be non-empty")
        rng = np.random.default_rng(seed)
        mean = latent_states[train].mean(axis=0)
        scale = np.maximum(latent_states[train].std(axis=0), 1e-4)
        dimension = targets.shape[1]
        params = {
            "first_w": _glorot(rng, EMBEDDING_DIM, hidden_dim),
            "first_b": np.zeros(hidden_dim, dtype=np.float32),
            "second_w": _glorot(rng, hidden_dim, hidden_dim),
            "second_b": np.zeros(hidden_dim, dtype=np.float32),
            "out_w": _glorot(rng, hidden_dim, dimension),
            "out_b": targets[train].mean(axis=0).astype(np.float32),
        }
        decoder = cls(params, mean, scale)
        moments: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        best_params = {name: value.copy() for name, value in params.items()}
        best_validation = float("inf")
        best_epoch = 0
        stalled = 0
        steps = 0
        history: list[dict[str, float]] = []

        def evaluate(indices: np.ndarray) -> tuple[float, float]:
            predictions = decoder.predict(latent_states[indices])
            mse = float(np.mean(np.square(predictions - targets[indices])))
            cosine = np.sum(predictions * targets[indices], axis=-1) / dimension
            return mse, float(np.mean(cosine))

        for epoch in range(1, epochs + 1):
            shuffled = rng.permutation(train)
            for offset in range(0, len(shuffled), batch_size):
                selected = shuffled[offset : offset + batch_size]
                raw, cache = decoder._forward(latent_states[selected])
                expected = targets[selected]
                derivative = (2.0 / (len(selected) * dimension)) * (raw - expected)

                raw_norm = np.maximum(np.linalg.norm(raw, axis=-1, keepdims=True), 1e-6)
                expected_norm = np.sqrt(dimension)
                cosine = np.sum(raw * expected, axis=-1, keepdims=True) / (
                    raw_norm * expected_norm
                )
                cosine_gradient = (
                    cosine * raw / np.square(raw_norm)
                    - expected / (raw_norm * expected_norm)
                ) / len(selected)
                derivative += 0.25 * cosine_gradient

                standardized, first, second = cache
                gradients = {
                    "out_w": second.T @ derivative,
                    "out_b": derivative.sum(axis=0),
                }
                second_pre = (derivative @ params["out_w"].T) * (1.0 - second**2)
                gradients["second_w"] = first.T @ second_pre
                gradients["second_b"] = second_pre.sum(axis=0)
                first_pre = (second_pre @ params["second_w"].T) * (1.0 - first**2)
                gradients["first_w"] = standardized.T @ first_pre
                gradients["first_b"] = first_pre.sum(axis=0)
                steps += 1
                _adam_step(
                    params,
                    gradients,
                    moments,
                    steps,
                    learning_rate=learning_rate,
                    weight_decay=1e-5,
                )

            train_subset = train[: min(len(train), 8192)]
            train_mse, train_cosine = evaluate(train_subset)
            validation_mse, validation_cosine = evaluate(validation)
            history.append(
                {
                    "epoch": float(epoch),
                    "train_mse": train_mse,
                    "train_cosine": train_cosine,
                    "validation_mse": validation_mse,
                    "validation_cosine": validation_cosine,
                }
            )
            if progress is not None and (epoch == 1 or epoch % 5 == 0):
                progress(
                    f"decoder epoch={epoch} train_cosine={train_cosine:.4f} "
                    f"val_cosine={validation_cosine:.4f} val_mse={validation_mse:.5f}"
                )
            if validation_mse < best_validation - 1e-7:
                best_validation = validation_mse
                best_epoch = epoch
                best_params = {name: value.copy() for name, value in params.items()}
                stalled = 0
            else:
                stalled += 1
                if stalled >= patience:
                    break

        decoder.params = best_params
        train_mse, _ = evaluate(train[: min(len(train), 8192)])
        validation_mse, validation_cosine = evaluate(validation)
        report = TrainingReport(
            best_epoch=best_epoch,
            epochs_ran=len(history),
            training_loss=train_mse,
            validation_loss=validation_mse,
            validation_r2=validation_cosine,
            history=tuple(history),
            validation_metric_name="cosine",
        )
        return decoder, report
