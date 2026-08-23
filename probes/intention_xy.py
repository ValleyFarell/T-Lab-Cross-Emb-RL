"""Декодер 128-мерных намерений в физические координаты лабиринта."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

try:
    import jax
    import jax.numpy as jnp
except ImportError:  # Сохранённые диагностические модели должны работать без пакетов обучения.
    jax = None
    jnp = None


DEFAULT_HIDDEN_DIMS = (512, 512, 512)
HIDDEN_DIM = DEFAULT_HIDDEN_DIMS[0]
OUTPUT_DIM = 2
_SCALE_EPSILON = 1e-6
_LAYER_NORM_EPSILON = 1e-6


def _require_jax() -> None:
    if jax is None or jnp is None:
        raise ImportError(
            "Training the intention-to-XY decoder requires JAX. Install the "
            "project dependencies first; loading and prediction work with NumPy."
        )


def _array(value):
    """Использует JAX при наличии и NumPy как совместимый запасной вариант."""

    if jnp is not None:
        return jnp.asarray(value, dtype=jnp.float32)
    return np.asarray(value, dtype=np.float32)


def _matrix(value, *, name: str, width: int | None = None) -> np.ndarray:
    try:
        matrix = np.asarray(value, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a numeric rank-2 array") from exc
    if matrix.ndim != 2:
        raise ValueError(f"{name} must be a rank-2 array, got {matrix.shape}")
    if width is not None and matrix.shape[1] != width:
        raise ValueError(f"{name} must have width {width}, got {matrix.shape}")
    if matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError(f"{name} must not be empty")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} contains non-finite values")
    return matrix


def _vector(value, *, name: str, width: int) -> np.ndarray:
    try:
        vector = np.asarray(value, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a numeric vector") from exc
    if vector.shape != (width,):
        raise ValueError(f"{name} must have shape ({width},), got {vector.shape}")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} contains non-finite values")
    return vector


def _layer_count(params: Mapping[str, Any]) -> int:
    weight_indices = sorted(
        int(name[1:])
        for name in params
        if isinstance(name, str) and name.startswith("w") and name[1:].isdigit()
    )
    if len(weight_indices) < 2:
        raise ValueError("decoder must contain at least one hidden and one output layer")
    if weight_indices != list(range(1, len(weight_indices) + 1)):
        raise ValueError("decoder weight layers are missing or non-consecutive")
    bias_indices = sorted(
        int(name[1:])
        for name in params
        if isinstance(name, str) and name.startswith("b") and name[1:].isdigit()
    )
    if bias_indices != weight_indices:
        missing = sorted(set(weight_indices) - set(bias_indices))
        if missing:
            raise ValueError(f"decoder file is missing b{missing[0]}")
        raise ValueError("decoder contains a bias without a matching weight layer")
    for index in weight_indices:
        if f"b{index}" not in params:
            raise ValueError(f"decoder file is missing b{index}")
    return len(weight_indices)


def _validate_params(params: Mapping[str, Any]) -> tuple[int, ...]:
    number_of_layers = _layer_count(params)
    widths: list[int] = []
    previous_width: int | None = None

    for index in range(1, number_of_layers + 1):
        weight = _matrix(params[f"w{index}"], name=f"w{index}")
        if previous_width is not None and weight.shape[0] != previous_width:
            raise ValueError(
                f"w{index} expects input width {weight.shape[0]}, but the "
                f"previous layer outputs width {previous_width}"
            )
        _vector(params[f"b{index}"], name=f"b{index}", width=weight.shape[1])
        if not widths:
            widths.append(int(weight.shape[0]))
        widths.append(int(weight.shape[1]))
        previous_width = int(weight.shape[1])

    if widths[-1] != OUTPUT_DIM:
        raise ValueError(
            f"decoder output width must be {OUTPUT_DIM}, got {widths[-1]}"
        )

    has_scale = "ln_scale1" in params
    has_bias = "ln_bias1" in params
    if has_scale != has_bias:
        raise ValueError("decoder must contain both ln_scale1 and ln_bias1 or neither")
    if has_scale:
        _vector(params["ln_scale1"], name="ln_scale1", width=widths[1])
        _vector(params["ln_bias1"], name="ln_bias1", width=widths[1])

    return tuple(widths)


def _init_params(key, input_dim: int, hidden_dims: tuple[int, ...]) -> dict[str, Any]:
    _require_jax()
    dimensions = (input_dim, *hidden_dims, OUTPUT_DIM)
    keys = jax.random.split(key, len(dimensions) - 1)
    params = {}
    for index, (fan_in, fan_out, layer_key) in enumerate(
        zip(dimensions[:-1], dimensions[1:], keys),
        start=1,
    ):
        limit = np.sqrt(6.0 / (fan_in + fan_out))
        params[f"w{index}"] = jax.random.uniform(
            layer_key,
            (fan_in, fan_out),
            minval=-limit,
            maxval=limit,
            dtype=jnp.float32,
        )
        params[f"b{index}"] = jnp.zeros((fan_out,), dtype=jnp.float32)

    # Повторяем устройство первого скрытого блока замороженного обратного представления.
    params["ln_scale1"] = jnp.ones((hidden_dims[0],), dtype=jnp.float32)
    params["ln_bias1"] = jnp.zeros((hidden_dims[0],), dtype=jnp.float32)
    return params


def _gelu_numpy(values: np.ndarray) -> np.ndarray:
    """Воспроизводит используемое JAX приближение функции активации GELU."""

    values = np.asarray(values, dtype=np.float32)
    cubic = values + np.float32(0.044715) * values**3
    factor = np.float32(np.sqrt(2.0 / np.pi))
    return np.float32(0.5) * values * (1.0 + np.tanh(factor * cubic))


def _network(params: Mapping[str, Any], standardized_z):
    output = standardized_z
    number_of_layers = _layer_count(params)
    use_jax = jnp is not None
    xp = jnp if use_jax else np

    for index in range(1, number_of_layers + 1):
        output = output @ params[f"w{index}"] + params[f"b{index}"]
        if index == number_of_layers:
            continue
        if index == 1 and "ln_scale1" in params:
            mean = xp.mean(output, axis=-1, keepdims=True)
            variance = xp.mean(xp.square(output - mean), axis=-1, keepdims=True)
            output = (output - mean) / xp.sqrt(variance + _LAYER_NORM_EPSILON)
            output = output * params["ln_scale1"] + params["ln_bias1"]
            output = xp.tanh(output)
        elif use_jax:
            output = jax.nn.gelu(output)
        else:
            output = _gelu_numpy(output)
    return output


@dataclass(frozen=True)
class IntentionXYDecoder:
    """Восстанавливает координаты лабиринта по нормализованному намерению."""

    params: Mapping[str, Any]
    input_mean: Any
    input_scale: Any
    target_mean: Any
    target_scale: Any

    def __post_init__(self) -> None:
        architecture = _validate_params(self.params)
        input_width = architecture[0]
        input_mean = _vector(self.input_mean, name="input_mean", width=input_width)
        input_scale = _vector(self.input_scale, name="input_scale", width=input_width)
        target_mean = _vector(self.target_mean, name="target_mean", width=OUTPUT_DIM)
        target_scale = _vector(self.target_scale, name="target_scale", width=OUTPUT_DIM)

        if np.any(input_scale <= 0):
            raise ValueError("input_scale must contain strictly positive values")
        if np.any(target_scale <= 0):
            raise ValueError("target_scale must contain strictly positive values")

        # Принимаем массивы NumPy, сохраняя единый вычислительный механизм.
        object.__setattr__(self, "params", {
            name: _array(value) for name, value in self.params.items()
        })
        object.__setattr__(self, "input_mean", _array(input_mean))
        object.__setattr__(self, "input_scale", _array(input_scale))
        object.__setattr__(self, "target_mean", _array(target_mean))
        object.__setattr__(self, "target_scale", _array(target_scale))

    @property
    def latent_dim(self) -> int:
        return int(np.asarray(self.params["w1"]).shape[0])

    @property
    def architecture(self) -> tuple[int, ...]:
        return _validate_params(self.params)

    def _validated_intentions(self, intentions):
        try:
            values = np.asarray(intentions, dtype=np.float32)
        except (TypeError, ValueError) as exc:
            raise ValueError("intentions must be numeric") from exc
        if values.ndim not in (1, 2):
            raise ValueError(
                f"intentions must have shape (D,) or (N, D), got {values.shape}"
            )
        if values.shape[-1] != self.latent_dim:
            raise ValueError(
                f"expected latent dimension {self.latent_dim}, got {values.shape}"
            )
        if values.ndim == 2 and values.shape[0] == 0:
            raise ValueError("intentions batch must not be empty")
        if not np.all(np.isfinite(values)):
            raise ValueError("intentions contain non-finite values")
        return _array(values)

    def predict(self, intentions):
        """Восстанавливает координаты одного намерения или блока намерений."""

        values = self._validated_intentions(intentions)
        standardized = (values - self.input_mean) / self.input_scale
        prediction = _network(self.params, standardized)
        result = prediction * self.target_scale + self.target_mean
        if not np.all(np.isfinite(np.asarray(result))):
            raise RuntimeError("decoder produced non-finite XY coordinates")
        return result

    def predict_with_diagnostics(
        self,
        intentions,
        *,
        max_abs_z_score: float = 6.0,
        max_rms_z_score: float = 2.5,
        max_relative_norm_error: float = 0.15,
    ) -> tuple[Any, dict[str, Any]]:
        """Восстанавливает координаты и отмечает признаки выхода за обучающее распределение.

        Диагностика предупреждает о возможном выходе за обучающее распределение, но не доказывает корректность произвольного намерения.
        """

        for name, threshold in (
            ("max_abs_z_score", max_abs_z_score),
            ("max_rms_z_score", max_rms_z_score),
            ("max_relative_norm_error", max_relative_norm_error),
        ):
            if not np.isfinite(threshold) or threshold <= 0:
                raise ValueError(f"{name} must be positive and finite")

        values = self._validated_intentions(intentions)
        values_np = np.asarray(values, dtype=np.float32)
        standardized = (
            values_np - np.asarray(self.input_mean)
        ) / np.asarray(self.input_scale)
        maximum = np.max(np.abs(standardized), axis=-1)
        rms = np.sqrt(np.mean(np.square(standardized), axis=-1))
        expected_norm = float(np.sqrt(self.latent_dim))
        actual_norm = np.linalg.norm(values_np, axis=-1)
        relative_norm_error = np.abs(actual_norm - expected_norm) / expected_norm
        out_of_distribution = (
            (maximum > max_abs_z_score)
            | (rms > max_rms_z_score)
            | (relative_norm_error > max_relative_norm_error)
        )

        def python_scalar_or_array(value):
            array = np.asarray(value)
            return array.item() if array.ndim == 0 else array

        diagnostics = {
            "latent_norm": python_scalar_or_array(actual_norm),
            "expected_latent_norm": expected_norm,
            "relative_norm_error": python_scalar_or_array(relative_norm_error),
            "max_abs_z_score": python_scalar_or_array(maximum),
            "rms_z_score": python_scalar_or_array(rms),
            "out_of_distribution": python_scalar_or_array(out_of_distribution),
        }
        return self.predict(values_np), diagnostics

    def save(self, output_dir: str | Path, *, metadata: Mapping[str, Any]) -> None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        arrays = {name: np.asarray(value) for name, value in self.params.items()}
        arrays.update(
            input_mean=np.asarray(self.input_mean),
            input_scale=np.asarray(self.input_scale),
            target_mean=np.asarray(self.target_mean),
            target_scale=np.asarray(self.target_scale),
        )
        np.savez_compressed(output_dir / "decoder.npz", **arrays)
        with (output_dir / "decoder_config.json").open("w", encoding="utf-8") as file:
            json.dump(dict(metadata), file, indent=2)

    @classmethod
    def load(cls, model_path: str | Path) -> "IntentionXYDecoder":
        model_path = Path(model_path)
        if model_path.is_dir():
            model_path = model_path / "decoder.npz"
        if not model_path.is_file():
            raise FileNotFoundError(
                f"Decoder artifact not found: {model_path}. The supplied project "
                "contains artifacts/intention_xy_decoder_deep/decoder.npz; pass "
                "that directory explicitly if the bmirror artifact was not trained."
            )

        with np.load(model_path, allow_pickle=False) as data:
            required = {"input_mean", "input_scale", "target_mean", "target_scale"}
            missing = required - set(data.files)
            if missing:
                raise ValueError(
                    "decoder file is missing arrays: " + ", ".join(sorted(missing))
                )

            weight_names = {
                name
                for name in data.files
                if name.startswith("w") and name[1:].isdigit()
            }
            bias_names = {
                name
                for name in data.files
                if name.startswith("b") and name[1:].isdigit()
            }
            param_names = weight_names | bias_names
            param_names.update(
                name for name in ("ln_scale1", "ln_bias1") if name in data.files
            )
            params = {name: np.array(data[name], copy=True) for name in param_names}
            return cls(
                params=params,
                input_mean=np.array(data["input_mean"], copy=True),
                input_scale=np.array(data["input_scale"], copy=True),
                target_mean=np.array(data["target_mean"], copy=True),
                target_scale=np.array(data["target_scale"], copy=True),
            )


def regression_metrics(decoder: IntentionXYDecoder, z, xy) -> dict[str, float]:
    z = _matrix(z, name="z", width=decoder.latent_dim)
    xy = _matrix(xy, name="xy", width=OUTPUT_DIM)
    if len(z) != len(xy):
        raise ValueError("z and xy must contain the same number of examples")
    error = np.asarray(decoder.predict(z)) - xy
    distance = np.linalg.norm(error, axis=1)
    return {
        "rmse_xy": float(np.sqrt(np.mean(np.square(error)))),
        "rmse_euclidean": float(np.sqrt(np.mean(np.sum(np.square(error), axis=1)))),
        "mae_x": float(np.mean(np.abs(error[:, 0]))),
        "mae_y": float(np.mean(np.abs(error[:, 1]))),
        "mean_euclidean_error": float(np.mean(distance)),
        "median_euclidean_error": float(np.median(distance)),
        "p90_euclidean_error": float(np.quantile(distance, 0.9)),
        "fraction_within_0_3": float(np.mean(distance <= 0.3)),
        "fraction_within_0_5": float(np.mean(distance <= 0.5)),
    }


def fit_decoder(
    train_z,
    train_xy,
    validation_z,
    validation_xy,
    *,
    seed: int = 0,
    hidden_dims: tuple[int, ...] = DEFAULT_HIDDEN_DIMS,
    learning_rate: float = 3e-4,
    batch_size: int = 1024,
    max_epochs: int = 500,
    patience: int = 50,
    weight_decay: float = 1e-5,
    warmup_epochs: int = 5,
    gradient_clip_norm: float = 1.0,
    input_noise_std: float = 0.0,
) -> tuple[IntentionXYDecoder, dict[str, Any]]:
    """Обучает декодер координат с регуляризацией и ранней остановкой."""

    _require_jax()
    train_z = _matrix(train_z, name="train_z")
    train_xy = _matrix(train_xy, name="train_xy", width=OUTPUT_DIM)
    validation_z = _matrix(validation_z, name="validation_z", width=train_z.shape[1])
    validation_xy = _matrix(validation_xy, name="validation_xy", width=OUTPUT_DIM)

    if len(train_z) != len(train_xy) or len(validation_z) != len(validation_xy):
        raise ValueError("each latent split must align with its XY targets")
    if any(
        isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value <= 0
        for value in (batch_size, max_epochs, patience)
    ):
        raise ValueError("batch_size, max_epochs, and patience must be positive integers")
    if not hidden_dims or any(
        isinstance(width, bool)
        or not isinstance(width, (int, np.integer))
        or width <= 0
        for width in hidden_dims
    ):
        raise ValueError("hidden_dims must contain one or more positive integer widths")
    hidden_dims = tuple(int(width) for width in hidden_dims)
    if not np.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("learning_rate must be positive and finite")
    if not np.isfinite(weight_decay) or weight_decay < 0:
        raise ValueError("weight_decay must be non-negative and finite")
    if (
        isinstance(warmup_epochs, bool)
        or not isinstance(warmup_epochs, (int, np.integer))
        or warmup_epochs < 0
        or warmup_epochs >= max_epochs
    ):
        raise ValueError("warmup_epochs must be non-negative and below max_epochs")
    if not np.isfinite(gradient_clip_norm) or gradient_clip_norm <= 0:
        raise ValueError("gradient_clip_norm must be positive and finite")
    if not np.isfinite(input_noise_std) or input_noise_std < 0:
        raise ValueError("input_noise_std must be non-negative and finite")

    input_mean = train_z.mean(axis=0, dtype=np.float64).astype(np.float32)
    input_scale = train_z.std(axis=0, dtype=np.float64).astype(np.float32)
    input_scale = np.where(input_scale < _SCALE_EPSILON, 1.0, input_scale)
    target_mean = train_xy.mean(axis=0, dtype=np.float64).astype(np.float32)
    target_scale = train_xy.std(axis=0, dtype=np.float64).astype(np.float32)
    target_scale = np.where(target_scale < _SCALE_EPSILON, 1.0, target_scale)

    x_train = jnp.asarray((train_z - input_mean) / input_scale, dtype=jnp.float32)
    y_train = jnp.asarray((train_xy - target_mean) / target_scale, dtype=jnp.float32)
    x_val = jnp.asarray((validation_z - input_mean) / input_scale, dtype=jnp.float32)
    y_val = jnp.asarray((validation_xy - target_mean) / target_scale, dtype=jnp.float32)

    initialization_key = jax.random.PRNGKey(seed)
    params = _init_params(initialization_key, train_z.shape[1], hidden_dims)
    first_moment = jax.tree_util.tree_map(jnp.zeros_like, params)
    second_moment = jax.tree_util.tree_map(jnp.zeros_like, params)

    steps_per_epoch = int(np.ceil(len(x_train) / batch_size))
    total_steps = max_epochs * steps_per_epoch
    warmup_steps = warmup_epochs * steps_per_epoch

    def scheduled_learning_rate(step):
        step = jnp.asarray(step, dtype=jnp.float32)
        if warmup_steps > 0:
            warmup_scale = jnp.minimum(step / warmup_steps, 1.0)
        else:
            warmup_scale = jnp.asarray(1.0, dtype=jnp.float32)
        decay_steps = max(1, total_steps - warmup_steps)
        progress = jnp.clip((step - warmup_steps) / decay_steps, 0.0, 1.0)
        cosine_scale = 0.5 * (1.0 + jnp.cos(jnp.pi * progress))
        scale = jnp.where(step <= warmup_steps, warmup_scale, cosine_scale)
        return learning_rate * scale

    def loss_fn(current_params, features, targets):
        residual = _network(current_params, features) - targets
        return jnp.mean(jnp.square(residual))

    @jax.jit
    def train_step(current_params, old_first, old_second, step, features, targets):
        if input_noise_std > 0:
            noise_key = jax.random.fold_in(initialization_key, step)
            features = features + input_noise_std * jax.random.normal(
                noise_key,
                features.shape,
                dtype=features.dtype,
            )

        loss, gradients = jax.value_and_grad(loss_fn)(current_params, features, targets)
        gradient_norm = jnp.sqrt(
            sum(jnp.sum(jnp.square(value)) for value in gradients.values())
        )
        clipping_scale = jnp.minimum(
            1.0,
            gradient_clip_norm / (gradient_norm + 1e-8),
        )
        gradients = jax.tree_util.tree_map(
            lambda gradient: gradient * clipping_scale,
            gradients,
        )

        beta1, beta2 = 0.9, 0.999
        first = jax.tree_util.tree_map(
            lambda previous, gradient: beta1 * previous + (1.0 - beta1) * gradient,
            old_first,
            gradients,
        )
        second = jax.tree_util.tree_map(
            lambda previous, gradient: (
                beta2 * previous + (1.0 - beta2) * jnp.square(gradient)
            ),
            old_second,
            gradients,
        )
        first_corrected = jax.tree_util.tree_map(
            lambda value: value / (1.0 - beta1**step),
            first,
        )
        second_corrected = jax.tree_util.tree_map(
            lambda value: value / (1.0 - beta2**step),
            second,
        )
        current_lr = scheduled_learning_rate(step)

        updated = {}
        for name, parameter in current_params.items():
            adam_update = first_corrected[name] / (
                jnp.sqrt(second_corrected[name]) + 1e-8
            )
            # AdamW отделяет регуляризацию весов от адаптивной нормализации градиента
            # и не применяет её к смещениям и параметрам нормализации слоя.
            decay = weight_decay * parameter if name.startswith("w") else 0.0
            updated[name] = parameter - current_lr * (adam_update + decay)

        return updated, first, second, loss, gradient_norm, current_lr

    rng = np.random.default_rng(seed)
    best_params = jax.tree_util.tree_map(lambda value: np.asarray(value).copy(), params)
    best_val_rmse_xy = float("inf")
    best_val_standardized_rmse = float("inf")
    best_epoch = 0
    stale_epochs = 0
    optimizer_step = 0
    history: list[dict[str, float | int]] = []

    for epoch in range(1, max_epochs + 1):
        permutation = rng.permutation(len(x_train))
        batch_losses = []
        gradient_norms = []
        current_lr = 0.0

        for start in range(0, len(permutation), batch_size):
            indices = permutation[start : start + batch_size]
            optimizer_step += 1
            (
                params,
                first_moment,
                second_moment,
                loss,
                gradient_norm,
                current_lr,
            ) = train_step(
                params,
                first_moment,
                second_moment,
                optimizer_step,
                x_train[indices],
                y_train[indices],
            )
            batch_losses.append(float(loss))
            gradient_norms.append(float(gradient_norm))

        residual = np.asarray(_network(params, x_val) - y_val)
        if not np.all(np.isfinite(residual)):
            raise FloatingPointError(
                f"decoder validation became non-finite at epoch {epoch}; "
                "reduce the learning rate or inspect the training data"
            )
        standardized_rmse = float(np.sqrt(np.mean(np.square(residual))))
        rmse_xy = float(np.sqrt(np.mean(np.square(residual * target_scale))))

        history.append({
            "epoch": epoch,
            "train_loss": float(np.mean(batch_losses)),
            "validation_rmse_xy": rmse_xy,
            "validation_standardized_rmse": standardized_rmse,
            "mean_gradient_norm_before_clipping": float(np.mean(gradient_norms)),
            "learning_rate": float(current_lr),
        })

        if rmse_xy < best_val_rmse_xy - 1e-6:
            best_val_rmse_xy = rmse_xy
            best_val_standardized_rmse = standardized_rmse
            best_epoch = epoch
            best_params = jax.tree_util.tree_map(
                lambda value: np.asarray(value).copy(),
                params,
            )
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break

    decoder = IntentionXYDecoder(
        params=best_params,
        input_mean=input_mean,
        input_scale=input_scale,
        target_mean=target_mean,
        target_scale=target_scale,
    )
    return decoder, {
        "architecture": [train_z.shape[1], *hidden_dims, OUTPUT_DIM],
        "optimizer": "adamw_with_global_gradient_clipping",
        "input_noise_std": float(input_noise_std),
        "best_epoch": best_epoch,
        "epochs_ran": len(history),
        "best_validation_rmse_xy": best_val_rmse_xy,
        "best_validation_standardized_rmse": best_val_standardized_rmse,
        "history": history,
    }


def split_dataset_indices(
    size: int,
    *,
    terminals=None,
    seed: int = 0,
    max_samples: int | None = 300_000,
    train_fraction: float = 0.8,
    validation_fraction: float = 0.1,
) -> tuple[dict[str, np.ndarray], str]:
    """Разделяет целые траектории до отбора отдельных состояний."""

    if isinstance(size, bool) or not isinstance(size, (int, np.integer)) or size < 3:
        raise ValueError("dataset must contain at least three observations")
    if (
        not np.isfinite(train_fraction)
        or not np.isfinite(validation_fraction)
        or not 0 < train_fraction < 1
        or not 0 < validation_fraction < 1
    ):
        raise ValueError("split fractions must lie strictly between zero and one")
    if train_fraction + validation_fraction >= 1:
        raise ValueError("train_fraction + validation_fraction must be below one")
    if max_samples is not None and (
        isinstance(max_samples, bool)
        or not isinstance(max_samples, (int, np.integer))
        or max_samples < 3
    ):
        raise ValueError("max_samples must be an integer of at least three or None")

    rng = np.random.default_rng(seed)
    groups = None
    strategy = "random_state"

    if terminals is not None:
        markers = np.asarray(terminals).reshape(-1)
        if len(markers) != size:
            raise ValueError("terminals length does not match dataset size")
        if not np.all(np.isfinite(markers)):
            raise ValueError("terminals contain non-finite values")
        groups = np.concatenate([
            np.zeros(1, dtype=np.int64),
            np.cumsum(markers[:-1] > 0, dtype=np.int64),
        ])
        if len(np.unique(groups)) < 3:
            groups = None

    if groups is not None:
        strategy = "trajectory"
        unique_groups = np.unique(groups)
        rng.shuffle(unique_groups)
        count = len(unique_groups)
        train_end = max(1, int(np.floor(count * train_fraction)))
        validation_count = max(1, int(np.floor(count * validation_fraction)))
        validation_end = min(count - 1, train_end + validation_count)
        if validation_end <= train_end:
            train_end = max(1, count - 2)
            validation_end = count - 1
        group_splits = {
            "train": unique_groups[:train_end],
            "validation": unique_groups[train_end:validation_end],
            "test": unique_groups[validation_end:],
        }
        splits = {
            name: np.flatnonzero(np.isin(groups, selected))
            for name, selected in group_splits.items()
        }
    else:
        shuffled = rng.permutation(size)
        train_end = max(1, int(np.floor(size * train_fraction)))
        validation_end = max(
            train_end + 1,
            int(np.floor(size * (train_fraction + validation_fraction))),
        )
        validation_end = min(size - 1, validation_end)
        splits = {
            "train": shuffled[:train_end],
            "validation": shuffled[train_end:validation_end],
            "test": shuffled[validation_end:],
        }

    if any(len(indices) == 0 for indices in splits.values()):
        raise ValueError("could not create three non-empty dataset splits")

    if max_samples is not None and sum(map(len, splits.values())) > max_samples:
        fractions = {
            "train": train_fraction,
            "validation": validation_fraction,
            "test": 1.0 - train_fraction - validation_fraction,
        }
        targets = {
            name: float(max_samples) * fraction
            for name, fraction in fractions.items()
        }
        desired = {
            name: min(len(splits[name]), max(1, int(np.floor(target))))
            for name, target in targets.items()
        }
        while sum(desired.values()) > max_samples:
            removable = [name for name in desired if desired[name] > 1]
            name = max(
                removable,
                key=lambda candidate: desired[candidate] - targets[candidate],
            )
            desired[name] -= 1
        while sum(desired.values()) < max_samples:
            available = [
                name for name in desired if desired[name] < len(splits[name])
            ]
            if not available:
                break
            name = max(
                available,
                key=lambda candidate: targets[candidate] - desired[candidate],
            )
            desired[name] += 1
        splits = {
            name: rng.choice(
                indices,
                size=min(len(indices), desired[name]),
                replace=False,
            )
            for name, indices in splits.items()
        }

    return {
        name: np.sort(np.asarray(indices, dtype=np.int64))
        for name, indices in splits.items()
    }, strategy