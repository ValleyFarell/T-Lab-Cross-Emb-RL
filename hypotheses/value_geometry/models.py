"""Вспомогательные модели ценности, обучаемые средствами NumPy."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np

from .data import PairSplit


@dataclass(frozen=True)
class TrainingConfig:
    hidden_width: int = 64
    hidden_layers: int = 2
    batch_size: int = 256
    max_epochs: int = 80
    patience: int = 12
    learning_rate: float = 2e-3
    weight_decay: float = 1e-5
    gradient_clip: float = 5.0
    huber_delta: float = 1.0
    prediction_batch_size: int = 4_096
    min_improvement: float = 1e-5
    log_every: int = 5

    def validate(self) -> None:
        for name in (
            "hidden_width",
            "hidden_layers",
            "batch_size",
            "max_epochs",
            "patience",
            "prediction_batch_size",
            "log_every",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or int(value) != value or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("learning_rate", "gradient_clip", "huber_delta"):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive and finite")
        if self.weight_decay < 0 or not np.isfinite(self.weight_decay):
            raise ValueError("weight_decay must be finite and non-negative")


def latent_dimension(model_name: str) -> int | None:
    if model_name.startswith("latent"):
        suffix = model_name[len("latent") :]
        if suffix.isdigit() and int(suffix) > 0:
            return int(suffix)
    if model_name in {"xy", "full", "pose"}:
        return None
    raise ValueError(
        f"unknown model {model_name!r}; use xy, full, pose or latent<dimension>"
    )


def _init_mlp(
    parameters: dict[str, np.ndarray],
    prefix: str,
    dimensions: tuple[int, ...],
    rng: np.random.Generator,
) -> None:
    for layer, (fan_in, fan_out) in enumerate(
        zip(dimensions[:-1], dimensions[1:])
    ):
        limit = np.sqrt(6.0 / (fan_in + fan_out))
        parameters[f"{prefix}.w{layer}"] = rng.uniform(
            -limit, limit, size=(fan_in, fan_out)
        ).astype(np.float32)
        parameters[f"{prefix}.b{layer}"] = np.zeros(fan_out, dtype=np.float32)


def _layer_count(parameters: Mapping[str, np.ndarray], prefix: str) -> int:
    count = 0
    while f"{prefix}.w{count}" in parameters:
        count += 1
    if count == 0:
        raise ValueError(f"missing parameters for network {prefix!r}")
    return count


def _forward_mlp(
    parameters: Mapping[str, np.ndarray],
    prefix: str,
    inputs: np.ndarray,
    *,
    cache: bool,
) -> tuple[np.ndarray, list[tuple[np.ndarray, np.ndarray | None]] | None]:
    output = np.asarray(inputs, dtype=np.float32)
    number_of_layers = _layer_count(parameters, prefix)
    caches = [] if cache else None
    for layer in range(number_of_layers):
        current_input = output
        output = (
            current_input @ parameters[f"{prefix}.w{layer}"]
            + parameters[f"{prefix}.b{layer}"]
        )
        if layer + 1 < number_of_layers:
            output = np.tanh(output)
            activation = output
        else:
            activation = None
        if caches is not None:
            caches.append((current_input, activation))
    return output.astype(np.float32, copy=False), caches


def _backward_mlp(
    parameters: Mapping[str, np.ndarray],
    prefix: str,
    caches: list[tuple[np.ndarray, np.ndarray | None]],
    output_gradient: np.ndarray,
    gradients: dict[str, np.ndarray],
) -> np.ndarray:
    gradient = np.asarray(output_gradient, dtype=np.float32)
    for layer in reversed(range(len(caches))):
        current_input, activation = caches[layer]
        if activation is not None:
            gradient = gradient * (1.0 - activation * activation)
        weight_name = f"{prefix}.w{layer}"
        bias_name = f"{prefix}.b{layer}"
        weight_gradient = current_input.T @ gradient
        bias_gradient = gradient.sum(axis=0)
        if weight_name in gradients:
            gradients[weight_name] += weight_gradient
            gradients[bias_name] += bias_gradient
        else:
            gradients[weight_name] = weight_gradient.astype(np.float32, copy=False)
            gradients[bias_name] = bias_gradient.astype(np.float32, copy=False)
        gradient = gradient @ parameters[weight_name].T
    return gradient


def _huber_loss(
    prediction: np.ndarray,
    target: np.ndarray,
    delta: float,
) -> tuple[float, np.ndarray]:
    residual = prediction - target
    absolute = np.abs(residual)
    loss = np.where(
        absolute <= delta,
        0.5 * residual * residual,
        delta * (absolute - 0.5 * delta),
    )
    gradient = np.clip(residual, -delta, delta) / max(1, residual.size)
    return float(loss.mean()), gradient.astype(np.float32)


def _standardizer(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=np.float32)
    mean = values.mean(axis=0, dtype=np.float64).astype(np.float32)
    scale = values.std(axis=0, dtype=np.float64).astype(np.float32)
    scale = np.where(scale < 1e-6, 1.0, scale).astype(np.float32)
    return mean, scale


class _Adam:
    def __init__(self, parameters: Mapping[str, np.ndarray]):
        self.first = {name: np.zeros_like(value) for name, value in parameters.items()}
        self.second = {name: np.zeros_like(value) for name, value in parameters.items()}
        self.step_number = 0

    def update(
        self,
        parameters: dict[str, np.ndarray],
        gradients: Mapping[str, np.ndarray],
        *,
        learning_rate: float,
        weight_decay: float,
        gradient_clip: float,
    ) -> float:
        squared_norm = sum(
            float(np.sum(gradient * gradient, dtype=np.float64))
            for gradient in gradients.values()
        )
        gradient_norm = float(np.sqrt(squared_norm))
        scale = min(1.0, gradient_clip / (gradient_norm + 1e-12))
        self.step_number += 1
        bias1 = 1.0 - 0.9**self.step_number
        bias2 = 1.0 - 0.999**self.step_number
        for name, parameter in parameters.items():
            gradient = gradients[name] * scale
            first = self.first[name]
            second = self.second[name]
            first *= 0.9
            first += 0.1 * gradient
            second *= 0.999
            second += 0.001 * gradient * gradient
            if ".w" in name and weight_decay:
                parameter *= 1.0 - learning_rate * weight_decay
            update = (first / bias1) / (np.sqrt(second / bias2) + 1e-8)
            parameter -= learning_rate * update
        return gradient_norm


@dataclass
class TrainedValueModel:
    name: str
    parameters: dict[str, np.ndarray]
    state_mean: np.ndarray
    state_scale: np.ndarray
    target_mean: float
    target_scale: float
    config: TrainingConfig
    history: list[dict[str, Any]]
    best_epoch: int
    seed: int

    @property
    def embedding_dimension(self) -> int | None:
        return latent_dimension(self.name)

    def _standardize_states(self, states: np.ndarray) -> np.ndarray:
        return (
            (np.asarray(states, dtype=np.float32) - self.state_mean)
            / self.state_scale
        ).astype(np.float32, copy=False)

    def encode(self, states: np.ndarray, *, batch_size: int | None = None) -> np.ndarray:
        if self.embedding_dimension is None:
            raise ValueError(f"model {self.name!r} has no learned state encoder")
        states = np.asarray(states, dtype=np.float32)
        squeeze = states.ndim == 1
        states = states[None, :] if squeeze else states
        if states.ndim != 2 or states.shape[1] != len(self.state_mean):
            raise ValueError("states has the wrong observation dimension")
        batch_size = batch_size or self.config.prediction_batch_size
        chunks = []
        for first in range(0, len(states), batch_size):
            standardized = self._standardize_states(states[first : first + batch_size])
            encoded, _ = _forward_mlp(
                self.parameters, "encoder", standardized, cache=False
            )
            chunks.append(encoded)
        result = np.concatenate(chunks, axis=0)
        return result[0] if squeeze else result

    def predict(
        self,
        observations: np.ndarray,
        start_indices: np.ndarray,
        goal_indices: np.ndarray,
        *,
        batch_size: int | None = None,
    ) -> np.ndarray:
        observations = np.asarray(observations, dtype=np.float32)
        starts = np.asarray(start_indices, dtype=np.int64)
        goals = np.asarray(goal_indices, dtype=np.int64)
        if starts.shape != goals.shape or starts.ndim != 1:
            raise ValueError("start_indices and goal_indices must be aligned vectors")
        if len(starts) == 0:
            return np.empty(0, dtype=np.float32)
        batch_size = batch_size or self.config.prediction_batch_size
        chunks = []
        for first in range(0, len(starts), batch_size):
            batch_starts = self._standardize_states(
                observations[starts[first : first + batch_size]]
            )
            batch_goals = self._standardize_states(
                observations[goals[first : first + batch_size]]
            )
            prediction, _ = _forward_model(
                self.name, self.parameters, batch_starts, batch_goals, cache=False
            )
            chunks.append(
                prediction.reshape(-1) * self.target_scale + self.target_mean
            )
        return np.concatenate(chunks).astype(np.float32)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays = {
            name.replace(".", "__"): value
            for name, value in self.parameters.items()
        }
        arrays.update(
            state_mean=self.state_mean,
            state_scale=self.state_scale,
            target_mean=np.asarray(self.target_mean, dtype=np.float32),
            target_scale=np.asarray(self.target_scale, dtype=np.float32),
            model_name=np.asarray(self.name),
            best_epoch=np.asarray(self.best_epoch, dtype=np.int64),
        )
        np.savez_compressed(path, **arrays)


def _forward_model(
    model_name: str,
    parameters: Mapping[str, np.ndarray],
    start_states: np.ndarray,
    goal_states: np.ndarray,
    *,
    cache: bool,
) -> tuple[np.ndarray, dict[str, Any] | None]:
    embedding_dim = latent_dimension(model_name)
    caches = {} if cache else None
    if embedding_dim is not None:
        start_embedding, start_cache = _forward_mlp(
            parameters, "encoder", start_states, cache=cache
        )
        goal_embedding, goal_cache = _forward_mlp(
            parameters, "encoder", goal_states, cache=cache
        )
        inputs = np.concatenate((start_embedding, goal_embedding), axis=1)
        if caches is not None:
            caches["start_encoder"] = start_cache
            caches["goal_encoder"] = goal_cache
            caches["embedding_dim"] = embedding_dim
    elif model_name == "xy":
        inputs = np.concatenate((start_states[:, :2], goal_states[:, :2]), axis=1)
    elif model_name == "pose":
        inputs = np.concatenate((start_states[:, 2:], goal_states[:, 2:]), axis=1)
    elif model_name == "full":
        # Явно включаем модель по координатам в контрольную модель полного состояния.
        # Обычная сеть на 58 признаках может переобучиться на несущественных особенностях шага
        # и ошибочно показаться хуже модели по координатам на конечной выборке.
        inputs = np.concatenate((start_states[:, :2], goal_states[:, :2]), axis=1)
    else:
        inputs = np.concatenate((start_states, goal_states), axis=1)

    output, head_cache = _forward_mlp(parameters, "head", inputs, cache=cache)
    if caches is not None:
        caches["head"] = head_cache
    if model_name == "full":
        residual_input = np.concatenate((start_states, goal_states), axis=1)
        residual, residual_cache = _forward_mlp(
            parameters, "residual", residual_input, cache=cache
        )
        output = output + np.float32(0.20) * residual
        if caches is not None:
            caches["residual"] = residual_cache
    return output, caches


def _backward_model(
    model_name: str,
    parameters: Mapping[str, np.ndarray],
    caches: Mapping[str, Any],
    loss_gradient: np.ndarray,
) -> dict[str, np.ndarray]:
    gradients: dict[str, np.ndarray] = {}
    head_gradient = _backward_mlp(
        parameters, "head", caches["head"], loss_gradient, gradients
    )
    if model_name == "full":
        _backward_mlp(
            parameters,
            "residual",
            caches["residual"],
            np.float32(0.20) * loss_gradient,
            gradients,
        )
    embedding_dim = latent_dimension(model_name)
    if embedding_dim is not None:
        _backward_mlp(
            parameters,
            "encoder",
            caches["start_encoder"],
            head_gradient[:, :embedding_dim],
            gradients,
        )
        _backward_mlp(
            parameters,
            "encoder",
            caches["goal_encoder"],
            head_gradient[:, embedding_dim:],
            gradients,
        )
    return gradients


def _copy_parameters(parameters: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {name: value.copy() for name, value in parameters.items()}


def fit_value_model(
    model_name: str,
    observations: np.ndarray,
    train_pairs: PairSplit,
    validation_pairs: PairSplit,
    *,
    train_state_indices: np.ndarray,
    config: TrainingConfig,
    seed: int,
    progress: Callable[[str], None] | None = None,
) -> TrainedValueModel:
    """Обучает вспомогательную модель ценности по подготовленным парам состояний."""

    config.validate()
    observations = np.asarray(observations, dtype=np.float32)
    if observations.ndim != 2 or observations.shape[1] < 3:
        raise ValueError("observations must have shape [N, observation_dim >= 3]")
    if train_pairs.values is None or validation_pairs.values is None:
        raise ValueError("both train and validation pairs need teacher values")
    embedding_dim = latent_dimension(model_name)
    train_state_indices = np.asarray(train_state_indices, dtype=np.int64)
    if len(train_state_indices) == 0:
        raise ValueError("train_state_indices cannot be empty")

    state_mean, state_scale = _standardizer(observations[train_state_indices])
    target_mean = float(np.mean(train_pairs.values, dtype=np.float64))
    target_scale = float(np.std(train_pairs.values, dtype=np.float64))
    if not np.isfinite(target_mean) or not np.isfinite(target_scale):
        raise ValueError("teacher values contain non-finite numbers")
    if target_scale < 1e-8:
        raise ValueError(
            "teacher values are nearly constant; the requested experiment is uninformative"
        )

    hidden = (config.hidden_width,) * config.hidden_layers
    rng = np.random.default_rng(seed)
    parameters: dict[str, np.ndarray] = {}
    if embedding_dim is not None:
        _init_mlp(
            parameters,
            "encoder",
            (observations.shape[1], *hidden, embedding_dim),
            rng,
        )
        head_input = embedding_dim * 2
    elif model_name == "xy":
        head_input = 4
    elif model_name == "pose":
        head_input = 2 * (observations.shape[1] - 2)
    elif model_name == "full":
        head_input = 4
        _init_mlp(
            parameters,
            "residual",
            (2 * observations.shape[1], *hidden, 1),
            rng,
        )
    else:
        head_input = 2 * observations.shape[1]
    _init_mlp(parameters, "head", (head_input, *hidden, 1), rng)
    optimizer = _Adam(parameters)

    result = TrainedValueModel(
        name=model_name,
        parameters=parameters,
        state_mean=state_mean,
        state_scale=state_scale,
        target_mean=target_mean,
        target_scale=target_scale,
        config=config,
        history=[],
        best_epoch=0,
        seed=seed,
    )
    best_parameters = _copy_parameters(parameters)
    best_validation = float("inf")
    stale_epochs = 0

    for epoch in range(1, config.max_epochs + 1):
        permutation = rng.permutation(len(train_pairs))
        losses: list[float] = []
        norms: list[float] = []
        # Плавное косинусное уменьшение шага сохраняет устойчивость обучения
        # без специального разгона и зависимости от компиляции вычислительного движка.
        progress_ratio = (epoch - 1) / max(1, config.max_epochs)
        learning_rate = config.learning_rate * (
            0.15 + 0.85 * 0.5 * (1.0 + np.cos(np.pi * progress_ratio))
        )
        for first in range(0, len(permutation), config.batch_size):
            rows = permutation[first : first + config.batch_size]
            starts = (
                observations[train_pairs.start_indices[rows]] - state_mean
            ) / state_scale
            goals = (
                observations[train_pairs.goal_indices[rows]] - state_mean
            ) / state_scale
            targets = (
                (train_pairs.values[rows] - target_mean) / target_scale
            ).reshape(-1, 1)
            prediction, caches = _forward_model(
                model_name,
                parameters,
                starts.astype(np.float32, copy=False),
                goals.astype(np.float32, copy=False),
                cache=True,
            )
            loss, loss_gradient = _huber_loss(
                prediction, targets.astype(np.float32), config.huber_delta
            )
            gradients = _backward_model(
                model_name, parameters, caches, loss_gradient
            )
            norm = optimizer.update(
                parameters,
                gradients,
                learning_rate=float(learning_rate),
                weight_decay=config.weight_decay,
                gradient_clip=config.gradient_clip,
            )
            losses.append(loss)
            norms.append(norm)

        validation_prediction = result.predict(
            observations,
            validation_pairs.start_indices,
            validation_pairs.goal_indices,
        )
        validation_rmse = float(
            np.sqrt(np.mean((validation_prediction - validation_pairs.values) ** 2))
        )
        entry = {
            "epoch": epoch,
            "train_huber": float(np.mean(losses)),
            "validation_rmse": validation_rmse,
            "gradient_norm": float(np.mean(norms)),
            "learning_rate": float(learning_rate),
        }
        result.history.append(entry)
        if validation_rmse < best_validation - config.min_improvement * target_scale:
            best_validation = validation_rmse
            best_parameters = _copy_parameters(parameters)
            result.best_epoch = epoch
            stale_epochs = 0
        else:
            stale_epochs += 1

        if progress and (
            epoch == 1 or epoch % config.log_every == 0 or stale_epochs >= config.patience
        ):
            progress(
                f"[{model_name}] epoch {epoch}/{config.max_epochs} "
                f"train_huber={entry['train_huber']:.5f} "
                f"val_rmse={validation_rmse:.6f} "
                f"best_epoch={result.best_epoch}"
            )
        if stale_epochs >= config.patience:
            break

    result.parameters = best_parameters
    return result
