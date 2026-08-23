"""Проверка поправок на качество начального и целевого состояния."""

from __future__ import annotations

import argparse
import copy
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


MODELS = ("xy", "xy_start", "xy_goal", "xy_both", "xy_additive", "full")
SPLIT_ALIASES = {"train": ("train", "training"), "val": ("val", "valid", "validation"), "test": ("test", "testing")}


@dataclass
class PairData:
    starts: np.ndarray
    goals: np.ndarray
    values: np.ndarray
    start_ids: np.ndarray | None = None

    def __post_init__(self) -> None:
        self.starts = np.asarray(self.starts, dtype=np.float32)
        self.goals = np.asarray(self.goals, dtype=np.float32)
        self.values = np.asarray(self.values, dtype=np.float32).reshape(-1)
        if self.starts.ndim != 2 or self.goals.ndim != 2 or self.starts.shape != self.goals.shape:
            raise ValueError(f"starts and goals must have identical matrix shapes: {self.starts.shape}, {self.goals.shape}")
        if self.starts.shape[0] != len(self.values) or self.starts.shape[1] < 3:
            raise ValueError("pair observations and scalar values have incompatible shapes")
        if not np.isfinite(self.starts).all() or not np.isfinite(self.goals).all() or not np.isfinite(self.values).all():
            raise ValueError("pair data contain NaN or infinity")
        if self.start_ids is not None:
            self.start_ids = np.asarray(self.start_ids).reshape(-1)
            if len(self.start_ids) != len(self.values):
                raise ValueError("start_ids must contain one id per pair")

    def take(self, indices: np.ndarray) -> "PairData":
        return PairData(self.starts[indices], self.goals[indices], self.values[indices], None if self.start_ids is None else self.start_ids[indices])


def _name_score(name: str, aliases: tuple[str, ...], kinds: tuple[str, ...]) -> int:
    key = name.lower().replace("-", "_")
    split = max((20 if key == alias or key.startswith(alias + "_") or key.endswith("_" + alias) or "_" + alias + "_" in key else 0) for alias in aliases)
    kind = max((10 if word in key else 0) for word in kinds)
    return split + kind if split and kind else 0


def _candidate(arrays: dict[str, np.ndarray], aliases: tuple[str, ...], kinds: tuple[str, ...], *, ndim: tuple[int, ...], expected: int | None = None) -> np.ndarray | None:
    candidates: list[tuple[int, str, np.ndarray]] = []
    for name, array in arrays.items():
        score = _name_score(name, aliases, kinds)
        if score and array.ndim in ndim and (expected is None or len(array) == expected):
            candidates.append((score, name, array))
    return max(candidates, key=lambda item: (item[0], -len(item[1])))[2] if candidates else None


def _load_one_split(arrays: dict[str, np.ndarray], split: str) -> PairData | None:
    aliases = SPLIT_ALIASES[split]
    starts = _candidate(arrays, aliases, ("starts", "start_states", "start_observations", "source_states", "source_observations", "obs_s"), ndim=(2,))
    goals = _candidate(arrays, aliases, ("goals", "goal_states", "goal_observations", "target_states", "target_observations", "obs_g"), ndim=(2,))
    values = _candidate(arrays, aliases, ("values", "labels", "targets", "scores", "value", "target", "y"), ndim=(1, 2), expected=None if starts is None else len(starts))
    ids = _candidate(arrays, aliases, ("start_indices", "start_idx", "start_ids", "source_indices", "source_idx"), ndim=(1,), expected=None if starts is None else len(starts))

    # Плотные пары содержат старт из 29 чисел, цель из 29 чисел и оценку либо индексы.
    pairs = _candidate(arrays, aliases, ("pairs", "pair_data", "examples"), ndim=(2,))
    observations = _candidate(arrays, aliases, ("observations", "states", "obs"), ndim=(2,))
    if observations is None:
        shared = [(name, arr) for name, arr in arrays.items() if arr.ndim == 2 and arr.shape[1] >= 3 and any(word in name.lower() for word in ("observations", "states", "obs"))]
        if shared:
            observations = max(shared, key=lambda item: len(item[1]))[1]
    if (starts is None or goals is None) and pairs is not None:
        if pairs.shape[1] == 2 and observations is not None and np.issubdtype(pairs.dtype, np.integer):
            starts, goals = observations[pairs[:, 0]], observations[pairs[:, 1]]
            ids = pairs[:, 0]
        elif pairs.shape[1] >= 7 and pairs.shape[1] % 2 == 1:
            dim = (pairs.shape[1] - 1) // 2
            starts, goals = pairs[:, :dim], pairs[:, dim : 2 * dim]
            if values is None:
                values = pairs[:, -1]
    start_indices = _candidate(arrays, aliases, ("start_indices", "start_idx", "source_indices", "source_idx"), ndim=(1,))
    goal_indices = _candidate(arrays, aliases, ("goal_indices", "goal_idx", "target_indices", "target_idx"), ndim=(1,))
    if (starts is None or goals is None) and observations is not None and start_indices is not None and goal_indices is not None and len(start_indices) == len(goal_indices):
        starts, goals, ids = observations[start_indices], observations[goal_indices], start_indices
    if starts is None or goals is None or values is None:
        return None
    if values.ndim == 2 and values.shape[1] != 1:
        return None
    return PairData(starts, goals, values, ids)


def inspect_cache(directory: Path) -> list[dict[str, Any]]:
    files = sorted(directory.rglob("*.npz")) if directory.is_dir() else [directory]
    descriptions = []
    for path in files:
        try:
            with np.load(path, allow_pickle=False) as archive:
                descriptions.append({"path": str(path), "arrays": {key: {"shape": list(archive[key].shape), "dtype": str(archive[key].dtype)} for key in archive.files}})
        except (OSError, ValueError, KeyError) as exc:
            descriptions.append({"path": str(path), "error": str(exc)})
    return descriptions


def load_cached_splits(directory: Path) -> dict[str, PairData]:
    files = sorted(directory.rglob("*.npz")) if directory.is_dir() else [directory]
    if not files:
        raise FileNotFoundError(f"no .npz files were found in {directory}")
    merged: dict[str, np.ndarray] = {}
    for path in files:
        with np.load(path, allow_pickle=False) as archive:
            stem = path.stem.lower()
            parent = path.parent.name.lower()
            for key in archive.files:
                array = archive[key]
                merged[key] = array
                merged[f"{stem}_{key}"] = array
                merged[f"{parent}_{key}"] = array
                for split, aliases in SPLIT_ALIASES.items():
                    if stem in aliases or parent in aliases:
                        merged[f"{split}_{key}"] = array
    result = {name: _load_one_split(merged, name) for name in SPLIT_ALIASES}
    missing = [name for name, value in result.items() if value is None]
    if missing:
        available = ", ".join(sorted(merged)[:35])
        raise ValueError(f"could not locate complete {missing} splits in {directory}; available array names include: {available}. Run with --inspect-cache, or omit --data-dir to calculate fresh labels.")
    return {name: value for name, value in result.items() if value is not None}


def split_observation_indices(size: int, terminals: np.ndarray | None, rng: np.random.Generator) -> dict[str, np.ndarray]:
    if terminals is not None and len(terminals) == size and np.count_nonzero(terminals) >= 4:
        boundaries = np.flatnonzero(np.asarray(terminals).astype(bool)) + 1
        edges = np.unique(np.r_[0, boundaries, size])
        trajectories = [np.arange(left, right) for left, right in zip(edges[:-1], edges[1:]) if right > left]
        rng.shuffle(trajectories)
        counts = np.cumsum([len(item) for item in trajectories])
        first = max(1, int(np.searchsorted(counts, 0.70 * size)))
        second = max(first + 1, int(np.searchsorted(counts, 0.85 * size)))
        second = min(second, len(trajectories) - 1)
        return {"train": np.concatenate(trajectories[:first]), "val": np.concatenate(trajectories[first:second]), "test": np.concatenate(trajectories[second:])}
    order = rng.permutation(size)
    first, second = int(0.70 * size), int(0.85 * size)
    return {"train": order[:first], "val": order[first:second], "test": order[second:]}


def build_checkpoint_splits(args: argparse.Namespace) -> dict[str, PairData]:
    if args.device == "cpu":
        os.environ.setdefault("JAX_PLATFORMS", "cpu")
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    try:
        import importlib
        import jax.numpy as jnp
        from baseline.frozen_fb import FrozenFB, load_checkpoint_config
        from utils.datasets import Dataset
        from utils.env_utils import make_env_and_datasets
    except ImportError as exc:
        raise RuntimeError("checkpoint mode requires the existing project's JAX, Flax and OGBench dependencies. Run from the project root, or use --data-dir to reuse saved pair labels.") from exc

    config, saved_flags = load_checkpoint_config(args.checkpoint)
    _, raw_train, _ = make_env_and_datasets(saved_flags["env_name"], frame_stack=config["frame_stack"], add_info=True)
    dataset = Dataset.create(**raw_train)
    dataset_class = getattr(importlib.import_module("utils.datasets"), config["dataset_class"])
    frozen = FrozenFB.from_checkpoint(args.checkpoint, dataset_class(dataset, config).sample(1), config=config)
    observations = np.asarray(dataset["observations"], dtype=np.float32)
    terminals = dataset.get("terminals")
    rng = np.random.default_rng(args.seed)
    groups = split_observation_indices(len(observations), terminals, rng)
    output: dict[str, PairData] = {}
    fractions = {"train": 1.0, "val": 0.20, "test": 0.20}
    for name, indices in groups.items():
        state_cap = max(128, int(args.max_states * (0.7 if name == "train" else 0.15)))
        if len(indices) > state_cap:
            indices = rng.choice(indices, size=state_cap, replace=False)
        goal_count = min(len(indices), max(8, int(args.goal_count * (1.0 if name == "train" else 0.35))))
        goal_pool = rng.choice(indices, size=goal_count, replace=False)
        pair_count = int(args.train_pairs * fractions[name])
        group_size = max(2, args.candidates_per_start)
        start_pool = rng.choice(indices, size=int(np.ceil(pair_count / group_size)), replace=True)
        start_indices = np.repeat(start_pool, group_size)[:pair_count]
        goal_indices = rng.choice(goal_pool, size=pair_count, replace=True)
        starts, goals = observations[start_indices], observations[goal_indices]

        # Воспроизводим исходную целевую величину: вектор награды B(g),
        # нормализованное намерение normalize(B(g)) и минимум оценок ансамбля.
        goal_reprs: dict[int, np.ndarray] = {}
        unique_goals = np.unique(goal_indices)
        for begin in range(0, len(unique_goals), args.critic_batch_size):
            batch_ids = unique_goals[begin : begin + args.critic_batch_size]
            raw = np.asarray(frozen.backward_repr(jnp.asarray(observations[batch_ids])), dtype=np.float32)
            for idx, vector in zip(batch_ids, raw):
                goal_reprs[int(idx)] = vector
        labels = np.empty(pair_count, dtype=np.float32)
        for begin in range(0, pair_count, args.critic_batch_size):
            end = min(pair_count, begin + args.critic_batch_size)
            reward_vectors = np.stack([goal_reprs[int(index)] for index in goal_indices[begin:end]])
            intentions = frozen.normalize_latent(jnp.asarray(reward_vectors))
            ensemble = np.asarray(frozen.forward_repr(jnp.asarray(starts[begin:end]), intentions))
            if ensemble.ndim == 2:
                ensemble = ensemble[None, :, :]
            member_values = np.einsum("ebd,bd->eb", ensemble, reward_vectors)
            labels[begin:end] = np.min(member_values, axis=0)
        output[name] = PairData(starts, goals, labels, start_indices)
        print(f"prepared {name}: pairs={len(labels):,}, unique_starts={len(np.unique(start_indices)):,}, unique_goals={len(unique_goals):,}", flush=True)
    return output


def synthetic_splits(seed: int, count: int = 6000) -> dict[str, PairData]:
    rng = np.random.default_rng(seed)
    result = {}
    for name, size in (("train", count), ("val", max(800, count // 5)), ("test", max(800, count // 5))):
        starts = rng.normal(size=(size, 29)).astype(np.float32)
        goals = rng.normal(size=(size, 29)).astype(np.float32)
        starts[:, :2] *= 3.0
        goals[:, :2] *= 3.0
        distance = np.linalg.norm(starts[:, :2] - goals[:, :2], axis=1)
        spatial = 4.0 + 16.0 * np.exp(-distance / 5.0)
        start_quality = 0.20 + 0.80 / (1.0 + np.exp(-2.2 * starts[:, 2]))
        goal_quality = 0.25 + 0.75 / (1.0 + np.exp(-1.8 * goals[:, 3]))
        values = -8.0 + spatial * start_quality * goal_quality + rng.normal(0.0, 0.15, size=size)
        result[name] = PairData(starts, goals, values.astype(np.float32), np.repeat(np.arange((size + 3) // 4), 4)[:size])
    return result


class MLP:
    def __init__(self, sizes: list[int], rng: np.random.Generator, *, zero_output: bool = False):
        self.weights = [rng.normal(0.0, np.sqrt(2.0 / max(1, size)), size=(size, next_size)).astype(np.float32) for size, next_size in zip(sizes[:-1], sizes[1:])]
        self.biases = [np.zeros(size, dtype=np.float32) for size in sizes[1:]]
        if zero_output:
            self.weights[-1] *= 0.01
        self.mw = [np.zeros_like(weight) for weight in self.weights]
        self.vw = [np.zeros_like(weight) for weight in self.weights]
        self.mb = [np.zeros_like(bias) for bias in self.biases]
        self.vb = [np.zeros_like(bias) for bias in self.biases]
        self.t = 0

    def forward(self, values: np.ndarray) -> tuple[np.ndarray, tuple[list[np.ndarray], list[np.ndarray]]]:
        activations = [values.astype(np.float32, copy=False)]
        preactivations = []
        for index, (weight, bias) in enumerate(zip(self.weights, self.biases)):
            raw = activations[-1] @ weight + bias
            preactivations.append(raw)
            activations.append(np.maximum(raw, 0.0) if index < len(self.weights) - 1 else raw)
        return activations[-1].reshape(-1), (activations, preactivations)

    def backward(self, grad: np.ndarray, cache: tuple[list[np.ndarray], list[np.ndarray]], *, learning_rate: float, weight_decay: float, clip: float) -> None:
        activations, preactivations = cache
        delta = grad.reshape(-1, 1).astype(np.float32)
        gradients_w, gradients_b = [], []
        for index in range(len(self.weights) - 1, -1, -1):
            gradients_w.append(activations[index].T @ delta + weight_decay * self.weights[index])
            gradients_b.append(delta.sum(axis=0))
            if index:
                delta = (delta @ self.weights[index].T) * (preactivations[index - 1] > 0.0)
        gradients_w.reverse()
        gradients_b.reverse()
        norm = float(np.sqrt(sum(float(np.sum(item * item)) for item in gradients_w + gradients_b)))
        scale = min(1.0, clip / (norm + 1e-12))
        self.t += 1
        for index, (gw, gb) in enumerate(zip(gradients_w, gradients_b)):
            gw, gb = gw * scale, gb * scale
            self.mw[index] = 0.9 * self.mw[index] + 0.1 * gw
            self.vw[index] = 0.999 * self.vw[index] + 0.001 * gw * gw
            self.mb[index] = 0.9 * self.mb[index] + 0.1 * gb
            self.vb[index] = 0.999 * self.vb[index] + 0.001 * gb * gb
            mw = self.mw[index] / (1.0 - 0.9**self.t)
            vw = self.vw[index] / (1.0 - 0.999**self.t)
            mb = self.mb[index] / (1.0 - 0.9**self.t)
            vb = self.vb[index] / (1.0 - 0.999**self.t)
            self.weights[index] -= learning_rate * mw / (np.sqrt(vw) + 1e-7)
            self.biases[index] -= learning_rate * mb / (np.sqrt(vb) + 1e-7)


def sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.clip(values, -35.0, 35.0)
    return (1.0 / (1.0 + np.exp(-values))).astype(np.float32)


def softplus(values: np.ndarray) -> np.ndarray:
    return (np.log1p(np.exp(-np.abs(values))) + np.maximum(values, 0.0)).astype(np.float32)


class FactorModel:
    def __init__(self, name: str, train: PairData, *, seed: int, hidden: int, quality_hidden: int, gate_regularization: float):
        self.name = name
        self.gate_regularization = gate_regularization
        rng = np.random.default_rng(seed)
        all_states = np.concatenate([train.starts, train.goals], axis=0)
        self.state_mean = all_states.mean(axis=0).astype(np.float32)
        self.state_scale = np.maximum(all_states.std(axis=0), 1e-4).astype(np.float32)
        self.offset = float(train.values.min() - max(0.25 * float(train.values.std()), 1e-3))
        self.target_scale = float(max(train.values.std(), 1e-4))
        dim = train.starts.shape[1]
        main_dim = 2 * dim if name == "full" else 4
        self.main = MLP([main_dim, hidden, hidden, 1], rng)
        self.start = MLP([dim - 2, quality_hidden, quality_hidden, 1], rng, zero_output=True) if name in ("xy_start", "xy_both", "xy_additive") else None
        self.goal = MLP([dim - 2, quality_hidden, quality_hidden, 1], rng, zero_output=True) if name in ("xy_goal", "xy_both", "xy_additive") else None

    def _inputs(self, data: PairData, indices: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        starts = data.starts if indices is None else data.starts[indices]
        goals = data.goals if indices is None else data.goals[indices]
        norm_s = (starts - self.state_mean) / self.state_scale
        norm_g = (goals - self.state_mean) / self.state_scale
        main = np.concatenate([norm_s, norm_g], axis=1) if self.name == "full" else np.concatenate([norm_s[:, :2], norm_g[:, :2]], axis=1)
        return main.astype(np.float32), norm_s[:, 2:].astype(np.float32), norm_g[:, 2:].astype(np.float32)

    def _forward(self, data: PairData, indices: np.ndarray | None = None) -> dict[str, Any]:
        main_input, start_input, goal_input = self._inputs(data, indices)
        main_raw, main_cache = self.main.forward(main_input)
        output: dict[str, Any] = {"main_raw": main_raw, "main_cache": main_cache}
        if self.name in ("xy", "full"):
            output["prediction_normalized"] = main_raw
        elif self.name == "xy_additive":
            start_raw, start_cache = self.start.forward(start_input)
            goal_raw, goal_cache = self.goal.forward(goal_input)
            output.update(start_raw=start_raw, goal_raw=goal_raw, start_cache=start_cache, goal_cache=goal_cache)
            output["prediction_normalized"] = main_raw + start_raw + goal_raw
        else:
            spatial = softplus(main_raw)
            start_raw, start_cache = self.start.forward(start_input) if self.start is not None else (np.zeros(len(main_raw), dtype=np.float32), None)
            goal_raw, goal_cache = self.goal.forward(goal_input) if self.goal is not None else (np.zeros(len(main_raw), dtype=np.float32), None)
            start_gate = 2.0 * sigmoid(start_raw) if self.start is not None else np.ones(len(main_raw), dtype=np.float32)
            goal_gate = 2.0 * sigmoid(goal_raw) if self.goal is not None else np.ones(len(main_raw), dtype=np.float32)
            output.update(spatial=spatial, start_raw=start_raw, goal_raw=goal_raw, start_gate=start_gate, goal_gate=goal_gate, start_cache=start_cache, goal_cache=goal_cache)
            output["prediction_normalized"] = spatial * start_gate * goal_gate
        return output

    def predict(self, data: PairData, *, batch_size: int = 8192) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        predictions, factors = [], {"start_quality": [], "goal_quality": [], "spatial_value": []}
        for begin in range(0, len(data.values), batch_size):
            indices = np.arange(begin, min(begin + batch_size, len(data.values)))
            output = self._forward(data, indices)
            if self.name in ("xy", "full", "xy_additive"):
                prediction = self.offset + self.target_scale * output["prediction_normalized"]
            else:
                prediction = self.offset + self.target_scale * output["prediction_normalized"]
                factors["start_quality"].append(output["start_gate"])
                factors["goal_quality"].append(output["goal_gate"])
                factors["spatial_value"].append(output["spatial"])
            predictions.append(prediction.astype(np.float32))
        return np.concatenate(predictions), {key: np.concatenate(value) for key, value in factors.items() if value}

    def train_batch(self, data: PairData, indices: np.ndarray, *, learning_rate: float, weight_decay: float, clip: float) -> float:
        output = self._forward(data, indices)
        target = (data.values[indices] - self.offset) / self.target_scale
        residual = output["prediction_normalized"] - target
        grad = 2.0 * residual / len(indices)
        loss = float(np.mean(residual * residual))
        if self.name in ("xy", "full"):
            self.main.backward(grad, output["main_cache"], learning_rate=learning_rate, weight_decay=weight_decay, clip=clip)
        elif self.name == "xy_additive":
            self.main.backward(grad, output["main_cache"], learning_rate=learning_rate, weight_decay=weight_decay, clip=clip)
            self.start.backward(grad, output["start_cache"], learning_rate=learning_rate, weight_decay=weight_decay, clip=clip)
            self.goal.backward(grad, output["goal_cache"], learning_rate=learning_rate, weight_decay=weight_decay, clip=clip)
        else:
            spatial, start_gate, goal_gate = output["spatial"], output["start_gate"], output["goal_gate"]
            grad_main = grad * start_gate * goal_gate * sigmoid(output["main_raw"])
            self.main.backward(grad_main, output["main_cache"], learning_rate=learning_rate, weight_decay=weight_decay, clip=clip)
            if self.start is not None:
                derivative = start_gate * (1.0 - start_gate / 2.0)
                regularizer = 2.0 * self.gate_regularization * (start_gate - 1.0) * derivative / len(indices)
                self.start.backward(grad * spatial * goal_gate * derivative + regularizer, output["start_cache"], learning_rate=learning_rate, weight_decay=weight_decay, clip=clip)
                loss += self.gate_regularization * float(np.mean((start_gate - 1.0) ** 2))
            if self.goal is not None:
                derivative = goal_gate * (1.0 - goal_gate / 2.0)
                regularizer = 2.0 * self.gate_regularization * (goal_gate - 1.0) * derivative / len(indices)
                self.goal.backward(grad * spatial * start_gate * derivative + regularizer, output["goal_cache"], learning_rate=learning_rate, weight_decay=weight_decay, clip=clip)
                loss += self.gate_regularization * float(np.mean((goal_gate - 1.0) ** 2))
        return loss


def regression_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    actual = np.asarray(actual, dtype=np.float64)
    predicted = np.asarray(predicted, dtype=np.float64)
    mse = float(np.mean((actual - predicted) ** 2))
    variance = float(np.var(actual))
    return {"count": int(len(actual)), "mse": mse, "rmse": float(np.sqrt(mse)), "mae": float(np.mean(np.abs(actual - predicted))), "r2": float(1.0 - mse / variance) if variance > 1e-12 else float("nan")}


def rank_metrics(data: PairData, predicted: np.ndarray) -> dict[str, float | int]:
    if data.start_ids is None:
        return {"groups": 0}
    order = np.argsort(data.start_ids, kind="stable")
    sorted_ids = data.start_ids[order]
    boundaries = np.r_[0, np.flatnonzero(sorted_ids[1:] != sorted_ids[:-1]) + 1, len(order)]
    matches, regrets, correlations = [], [], []
    for left, right in zip(boundaries[:-1], boundaries[1:]):
        if right - left < 2:
            continue
        indices = order[left:right]
        truth, estimate = data.values[indices], predicted[indices]
        selected = int(np.argmax(estimate))
        matches.append(float(selected == int(np.argmax(truth))))
        regrets.append(float(np.max(truth) - truth[selected]))
        if np.std(truth) > 1e-8 and np.std(estimate) > 1e-8:
            truth_rank = np.argsort(np.argsort(truth)).astype(np.float64)
            estimate_rank = np.argsort(np.argsort(estimate)).astype(np.float64)
            correlations.append(float(np.corrcoef(truth_rank, estimate_rank)[0, 1]))
    return {"groups": len(matches), "top1_accuracy": float(np.mean(matches)) if matches else float("nan"), "selection_regret": float(np.mean(regrets)) if regrets else float("nan"), "rank_correlation": float(np.mean(correlations)) if correlations else float("nan")}


def distance_metrics(data: PairData, predicted: np.ndarray, edges: np.ndarray) -> dict[str, dict[str, float]]:
    distance = np.linalg.norm(data.starts[:, :2] - data.goals[:, :2], axis=1)
    output = {}
    for index, name in enumerate(("near", "medium", "far", "very_far")):
        mask = (distance >= edges[index]) & (distance < edges[index + 1] if index < 3 else distance <= edges[index + 1])
        if mask.any():
            output[name] = regression_metrics(data.values[mask], predicted[mask])
    return output


def train_model(name: str, splits: dict[str, PairData], args: argparse.Namespace, seed: int) -> tuple[FactorModel, list[dict[str, float]]]:
    model = FactorModel(name, splits["train"], seed=seed, hidden=args.hidden_dim, quality_hidden=args.quality_hidden_dim, gate_regularization=args.gate_regularization)
    rng = np.random.default_rng(seed + 83)
    history, best_model, best_loss, stale = [], None, float("inf"), 0
    for epoch in range(args.epochs):
        order = rng.permutation(len(splits["train"].values))
        losses = []
        for begin in range(0, len(order), args.batch_size):
            losses.append(model.train_batch(splits["train"], order[begin : begin + args.batch_size], learning_rate=args.learning_rate, weight_decay=args.weight_decay, clip=args.gradient_clip))
        val_prediction, _ = model.predict(splits["val"])
        val = regression_metrics(splits["val"].values, val_prediction)
        history.append({"epoch": epoch + 1, "train_loss_normalized": float(np.mean(losses)), "validation_mse": val["mse"], "validation_r2": val["r2"]})
        if val["mse"] < best_loss - max(1e-8, best_loss * 1e-5 if np.isfinite(best_loss) else 0.0):
            best_loss, best_model, stale = val["mse"], copy.deepcopy(model), 0
        else:
            stale += 1
            if stale >= args.patience:
                break
    return best_model if best_model is not None else model, history


def save_weights(model: FactorModel, path: Path) -> None:
    arrays: dict[str, np.ndarray] = {"state_mean": model.state_mean, "state_scale": model.state_scale, "target_offset": np.asarray(model.offset), "target_scale": np.asarray(model.target_scale)}
    for prefix in ("main", "start", "goal"):
        network = getattr(model, prefix)
        if network is not None:
            for index, (weight, bias) in enumerate(zip(network.weights, network.biases)):
                arrays[f"{prefix}_weight_{index}"], arrays[f"{prefix}_bias_{index}"] = weight, bias
    np.savez_compressed(path, **arrays)


def save_plot(metrics: dict[str, Any], output_dir: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    names = list(metrics["aggregate"])
    overall = [metrics["aggregate"][name]["test_r2_mean"] for name in names]
    far = [metrics["aggregate"][name].get("very_far_r2_mean", float("nan")) for name in names]
    indices = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(max(8, len(names) * 1.3), 4.5))
    ax.bar(indices - 0.19, overall, width=0.38, label="All held-out pairs")
    ax.bar(indices + 0.19, far, width=0.38, label="Very far pairs")
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xticks(indices, names, rotation=20, ha="right")
    ax.set_ylabel("Held-out R²")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "factor_comparison.png", dpi=170)
    plt.close(fig)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Проверка поправок на качество начального и целевого состояния.')
    parser.add_argument("--data-dir", type=Path, help='Каталог ранее сохранённых пар состояний и оценок критика.')
    parser.add_argument("--inspect-cache", action="store_true", help='Показывает названия и размеры сохранённых массивов без запуска обучения.')
    parser.add_argument("--synthetic", action="store_true", help='Проверяет весь конвейер на искусственных данных без чекпоинта.')
    parser.add_argument("--quick", action="store_true", help='Уменьшает размер эксперимента для быстрой предварительной проверки.')
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/antmaze-medium-navigate-v0"), help='Каталог замороженного агента с params.pkl и flags.json.')
    parser.add_argument("--device", choices=("cpu", "auto"), default="cpu", help='Устройство вычислений: cpu, gpu или auto, если скрипт поддерживает эти варианты.')
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/value_state_factors"), help='Каталог сохранения моделей, оценок и промежуточных данных.')
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS), help='Архитектуры: xy, xy_start, xy_goal, xy_both, xy_additive, full.')
    parser.add_argument("--model-seeds", nargs="+", type=int, default=[0, 1, 2], help='Независимые начальные инициализации вспомогательных моделей.')
    parser.add_argument("--seed", type=int, default=42, help='Воспроизводимая инициализация обучения и разбиения данных.')
    parser.add_argument("--max-states", type=int, default=80_000, help='Верхняя граница числа используемых офлайн-состояний.')
    parser.add_argument("--train-pairs", type=int, default=160_000, help='Число обучающих пар «начальное состояние — цель».')
    parser.add_argument("--goal-count", type=int, default=384, help='Число различных целевых состояний.')
    parser.add_argument("--candidates-per-start", type=int, default=8, help='Число целевых состояний для одного начального состояния.')
    parser.add_argument("--critic-batch-size", type=int, default=128, help='Размер блока запросов к замороженному критику.')
    parser.add_argument("--epochs", type=int, default=100, help='Максимальное число проходов обучения модели геометрии.')
    parser.add_argument("--patience", type=int, default=20, help='Число проходов без улучшения до остановки.')
    parser.add_argument("--batch-size", type=int, default=512, help='Размер обучающего блока.')
    parser.add_argument("--hidden-dim", type=int, default=96, help='Ширина основной модели ценности.')
    parser.add_argument("--quality-hidden-dim", type=int, default=48, help='Ширина сетей, оценивающих качество состояния.')
    parser.add_argument("--learning-rate", type=float, default=8e-4, help='Размер шага оптимизации.')
    parser.add_argument("--weight-decay", type=float, default=1e-5, help='Сила регуляризации весов.')
    parser.add_argument("--gate-regularization", type=float, default=2e-3, help='Сила ограничения поправочных коэффициентов.')
    parser.add_argument("--gradient-clip", type=float, default=2.0, help='Верхняя граница нормы градиента.')
    args = parser.parse_args(argv)
    if args.quick:
        args.train_pairs = min(args.train_pairs, 6000)
        args.max_states = min(args.max_states, 5000)
        args.goal_count = min(args.goal_count, 48)
        args.epochs = min(args.epochs, 18)
        args.patience = min(args.patience, 8)
        args.model_seeds = args.model_seeds[:1]
    if args.inspect_cache and args.data_dir is None:
        parser.error("--inspect-cache requires --data-dir")
    for name in ("max_states", "train_pairs", "goal_count", "candidates_per_start", "critic_batch_size", "epochs", "patience", "batch_size", "hidden_dim", "quality_hidden_dim"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    return args


def main(argv: list[str] | None = None) -> dict[str, Any] | None:
    args = parse_args(argv)
    if args.inspect_cache:
        print(json.dumps(inspect_cache(args.data_dir), indent=2, ensure_ascii=False))
        return None
    if args.synthetic:
        splits = synthetic_splits(args.seed, count=min(args.train_pairs, 6000) if args.quick else min(args.train_pairs, 20_000))
        source = "synthetic"
    elif args.data_dir is not None:
        splits, source = load_cached_splits(args.data_dir), str(args.data_dir)
    else:
        splits, source = build_checkpoint_splits(args), f"frozen-checkpoint:{args.checkpoint}"

    args.output_dir.mkdir(parents=True, exist_ok=True)
    cache_arrays = {}
    for name, data in splits.items():
        cache_arrays.update({f"{name}_starts": data.starts, f"{name}_goals": data.goals, f"{name}_values": data.values})
        if data.start_ids is not None:
            cache_arrays[f"{name}_start_ids"] = data.start_ids
    np.savez_compressed(args.output_dir / "pairs.npz", **cache_arrays)
    train_distance = np.linalg.norm(splits["train"].starts[:, :2] - splits["train"].goals[:, :2], axis=1)
    edges = np.quantile(train_distance, [0.0, 0.25, 0.50, 0.75, 1.0])
    edges[0], edges[-1] = 0.0, float("inf")

    result: dict[str, Any] = {"source": source, "split_sizes": {name: len(data.values) for name, data in splits.items()}, "distance_metric": "euclidean_xy", "target_definition_for_fresh_labels": "min_ensemble dot(F(start, normalize(B(goal))), B(goal))", "quality_inputs": "full observation excluding observation[:2]", "factorized_formula": "offset + target_scale * softplus(V_xy) * (2*sigmoid(Q_start)) * (2*sigmoid(Q_goal))", "models": {}, "aggregate": {}}
    for name in args.models:
        result["models"][name] = []
        for seed in args.model_seeds:
            print(f"training model={name} seed={seed}", flush=True)
            model, history = train_model(name, splits, args, seed)
            prediction, factors = model.predict(splits["test"])
            validation, _ = model.predict(splits["val"])
            run = {"seed": seed, "epochs_completed": len(history), "best_validation": regression_metrics(splits["val"].values, validation), "test": regression_metrics(splits["test"].values, prediction), "distance_groups": distance_metrics(splits["test"], prediction, edges), "ranking": rank_metrics(splits["test"], prediction), "factors": {key: {"mean": float(np.mean(value)), "std": float(np.std(value)), "p05": float(np.quantile(value, 0.05)), "p95": float(np.quantile(value, 0.95))} for key, value in factors.items()}}
            result["models"][name].append(run)
            (args.output_dir / f"history_{name}_seed{seed}.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
            np.savez_compressed(args.output_dir / f"predictions_{name}_seed{seed}.npz", actual=splits["test"].values, predicted=prediction, **factors)
            save_weights(model, args.output_dir / f"weights_{name}_seed{seed}.npz")
            print(f"  epochs={len(history)} test_R2={run['test']['r2']:.4f} very_far_R2={run['distance_groups'].get('very_far', {}).get('r2', float('nan')):.4f}", flush=True)
        runs = result["models"][name]
        result["aggregate"][name] = {"test_r2_mean": float(np.mean([run["test"]["r2"] for run in runs])), "test_r2_std": float(np.std([run["test"]["r2"] for run in runs])), "very_far_r2_mean": float(np.mean([run["distance_groups"].get("very_far", {}).get("r2", float("nan")) for run in runs])), "top1_accuracy_mean": float(np.mean([run["ranking"].get("top1_accuracy", float("nan")) for run in runs])), "selection_regret_mean": float(np.mean([run["ranking"].get("selection_regret", float("nan")) for run in runs]))}
    (args.output_dir / "metrics.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    save_plot(result, args.output_dir)
    print(json.dumps(result["aggregate"], indent=2))
    print(f"saved_to: {args.output_dir}")
    return result


if __name__ == "__main__":
    main()
