"""Метрики предсказания ценности и восстановления физических координат."""

from __future__ import annotations

from collections import defaultdict
from itertools import combinations_with_replacement
from typing import Any, Mapping

import numpy as np

from .data import PairSplit
from .models import TrainedValueModel


def _finite_vector(values: np.ndarray, name: str) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if len(values) == 0 or not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must contain finite non-empty values")
    return values


def regression_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, Any]:
    actual = _finite_vector(actual, "actual")
    predicted = _finite_vector(predicted, "predicted")
    if len(actual) != len(predicted):
        raise ValueError("actual and predicted lengths differ")
    residual = predicted - actual
    mse = float(np.mean(residual * residual))
    variance = float(np.var(actual))
    rmse = float(np.sqrt(mse))
    return {
        "count": int(len(actual)),
        "mse": mse,
        "rmse": rmse,
        "mae": float(np.mean(np.abs(residual))),
        "target_mean": float(np.mean(actual)),
        "target_std": float(np.sqrt(variance)),
        "nrmse": float(rmse / np.sqrt(variance)) if variance > 1e-15 else None,
        "r2": float(1.0 - mse / variance) if variance > 1e-15 else None,
    }


def distance_stratified_metrics(
    actual: np.ndarray,
    predicted: np.ndarray,
    distance_bins: np.ndarray,
) -> dict[str, dict[str, Any]]:
    distance_bins = np.asarray(distance_bins)
    if len(distance_bins) != len(actual):
        raise ValueError("distance_bins and values must be aligned")
    result = {}
    for group in np.unique(distance_bins):
        selected = distance_bins == group
        result[str(int(group))] = regression_metrics(
            np.asarray(actual)[selected], np.asarray(predicted)[selected]
        )
    return result


def _rank(values: np.ndarray) -> np.ndarray:
    """Вычисляет устойчивые ранги с корректной обработкой одинаковых значений."""

    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="stable")
    sorted_values = values[order]
    ranks = np.empty(len(values), dtype=np.float64)
    first = 0
    while first < len(order):
        stop = first + 1
        while stop < len(order) and sorted_values[stop] == sorted_values[first]:
            stop += 1
        ranks[order[first:stop]] = 0.5 * (first + stop - 1)
        first = stop
    return ranks


def ranking_metrics(
    pairs: PairSplit,
    predicted: np.ndarray,
    *,
    minimum_candidates: int = 3,
) -> dict[str, Any]:
    if pairs.values is None:
        raise ValueError("pairs have no teacher values")
    predicted = _finite_vector(predicted, "predicted")
    if len(predicted) != len(pairs):
        raise ValueError("predicted has a different pair count")

    groups: dict[int, list[int]] = defaultdict(list)
    for row, start in enumerate(pairs.start_indices):
        groups[int(start)].append(row)

    correlations = []
    regrets = []
    relative_regrets = []
    top1 = []
    top3 = []
    for indices in groups.values():
        if len(indices) < minimum_candidates:
            continue
        indices = np.asarray(indices, dtype=np.int64)
        truth = np.asarray(pairs.values[indices], dtype=np.float64)
        estimate = predicted[indices]
        span = float(truth.max() - truth.min())
        chosen = int(np.argmax(estimate))
        optimal = int(np.argmax(truth))
        regret = float(truth[optimal] - truth[chosen])
        regrets.append(regret)
        top1.append(chosen == optimal)
        strongest = np.argsort(estimate)[-min(3, len(indices)) :]
        top3.append(optimal in strongest)
        if span > 1e-12:
            relative_regrets.append(regret / span)
        truth_rank = _rank(truth)
        estimate_rank = _rank(estimate)
        denominator = np.linalg.norm(truth_rank - truth_rank.mean()) * np.linalg.norm(
            estimate_rank - estimate_rank.mean()
        )
        if denominator > 1e-12:
            correlations.append(
                float(
                    np.dot(
                        truth_rank - truth_rank.mean(),
                        estimate_rank - estimate_rank.mean(),
                    )
                    / denominator
                )
            )

    if not regrets:
        return {"groups": 0, "note": "not enough repeated starts"}
    return {
        "groups": len(regrets),
        "mean_spearman": float(np.mean(correlations)) if correlations else None,
        "top1_agreement": float(np.mean(top1)),
        "top3_contains_teacher_best": float(np.mean(top3)),
        "mean_value_regret": float(np.mean(regrets)),
        "median_value_regret": float(np.median(regrets)),
        "mean_relative_regret": (
            float(np.mean(relative_regrets)) if relative_regrets else None
        ),
    }


def _coordinate_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, Any]:
    actual = np.asarray(actual, dtype=np.float64)
    predicted = np.asarray(predicted, dtype=np.float64)
    if actual.shape != predicted.shape or actual.ndim != 2:
        raise ValueError("actual and predicted coordinates must be aligned matrices")
    residual = predicted - actual
    distance = np.linalg.norm(residual, axis=1)
    sse = float(np.sum(residual * residual))
    centered = actual - actual.mean(axis=0, keepdims=True)
    total = float(np.sum(centered * centered))
    return {
        "count": int(len(actual)),
        "rmse_coordinate": float(np.sqrt(np.mean(residual * residual))),
        "mean_euclidean_error": float(np.mean(distance)),
        "median_euclidean_error": float(np.median(distance)),
        "p90_euclidean_error": float(np.quantile(distance, 0.90)),
        "fraction_within_0_3": float(np.mean(distance <= 0.3)),
        "fraction_within_0_5": float(np.mean(distance <= 0.5)),
        "r2": float(1.0 - sse / total) if total > 1e-15 else None,
    }


def affine_probe(
    train_features: np.ndarray,
    train_targets: np.ndarray,
    test_features: np.ndarray,
    test_targets: np.ndarray,
) -> dict[str, Any]:
    """Обучает линейное восстановление координат без изменения исходного энкодера."""

    train_features = np.asarray(train_features, dtype=np.float64)
    train_targets = np.asarray(train_targets, dtype=np.float64)
    test_features = np.asarray(test_features, dtype=np.float64)
    test_targets = np.asarray(test_targets, dtype=np.float64)
    if train_features.ndim != 2 or train_targets.ndim != 2:
        raise ValueError("probe inputs and outputs must be matrices")
    if len(train_features) != len(train_targets):
        raise ValueError("training probe arrays are not aligned")
    augmented = np.concatenate(
        (train_features, np.ones((len(train_features), 1))), axis=1
    )
    coefficients, _, _, _ = np.linalg.lstsq(augmented, train_targets, rcond=1e-8)
    test_augmented = np.concatenate(
        (test_features, np.ones((len(test_features), 1))), axis=1
    )
    prediction = test_augmented @ coefficients
    return {
        "metrics": _coordinate_metrics(test_targets, prediction),
        "coefficients": coefficients.tolist(),
        "predictions": prediction.astype(np.float32),
    }


def _polynomial_features(features: np.ndarray, degree: int) -> np.ndarray:
    features = np.asarray(features, dtype=np.float64)
    terms = [np.ones(len(features), dtype=np.float64)]
    for current_degree in range(1, degree + 1):
        for columns in combinations_with_replacement(
            range(features.shape[1]), current_degree
        ):
            terms.append(np.prod(features[:, columns], axis=1))
    return np.column_stack(terms)


def polynomial_probe(
    train_features: np.ndarray,
    train_targets: np.ndarray,
    validation_features: np.ndarray,
    validation_targets: np.ndarray,
    test_features: np.ndarray,
    test_targets: np.ndarray,
    *,
    degree: int = 3,
) -> dict[str, Any]:
    """Восстанавливает координаты нелинейной моделью с выбором регуляризации."""

    train_features = np.asarray(train_features, dtype=np.float64)
    mean = train_features.mean(axis=0)
    scale = np.where(train_features.std(axis=0) < 1e-6, 1.0, train_features.std(axis=0))

    def transform(values: np.ndarray) -> np.ndarray:
        return _polynomial_features((np.asarray(values) - mean) / scale, degree)

    train_x = transform(train_features)
    validation_x = transform(validation_features)
    test_x = transform(test_features)
    train_y = np.asarray(train_targets, dtype=np.float64)
    validation_y = np.asarray(validation_targets, dtype=np.float64)
    best = None
    gram = train_x.T @ train_x
    cross = train_x.T @ train_y
    for ridge in (1e-6, 1e-4, 1e-2, 1.0, 100.0):
        penalty = np.eye(gram.shape[0]) * ridge
        penalty[0, 0] = 0.0
        try:
            coefficients = np.linalg.solve(gram + penalty, cross)
        except np.linalg.LinAlgError:
            coefficients = np.linalg.lstsq(gram + penalty, cross, rcond=1e-8)[0]
        error = float(np.mean((validation_x @ coefficients - validation_y) ** 2))
        if best is None or error < best[0]:
            best = (error, ridge, coefficients)
    prediction = test_x @ best[2]
    return {
        "degree": int(degree),
        "ridge": float(best[1]),
        "validation_mse": float(best[0]),
        "metrics": _coordinate_metrics(test_targets, prediction),
        "predictions": prediction.astype(np.float32),
    }


def embedding_diagnostics(
    model: TrainedValueModel,
    observations: np.ndarray,
    positions: np.ndarray,
    split_indices: Mapping[str, np.ndarray],
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    if model.embedding_dimension is None:
        return {}, {}
    encoded = {
        name: model.encode(observations[np.asarray(indices, dtype=np.int64)])
        for name, indices in split_indices.items()
    }
    xy = {name: positions[indices] for name, indices in split_indices.items()}
    forward_affine = affine_probe(
        encoded["train"], xy["train"], encoded["test"], xy["test"]
    )
    reverse_affine = affine_probe(
        xy["train"], encoded["train"], xy["test"], encoded["test"]
    )
    forward_polynomial = polynomial_probe(
        encoded["train"],
        xy["train"],
        encoded["validation"],
        xy["validation"],
        encoded["test"],
        xy["test"],
    )
    reverse_polynomial = polynomial_probe(
        xy["train"],
        encoded["train"],
        xy["validation"],
        encoded["validation"],
        xy["test"],
        encoded["test"],
    )
    covariance = np.cov(encoded["test"], rowvar=False)
    covariance = np.atleast_2d(covariance)
    eigenvalues = np.linalg.eigvalsh(covariance)[::-1]
    result = {
        "embedding_dimension": int(model.embedding_dimension),
        "coordinate_readout_affine": forward_affine["metrics"],
        "coordinate_readout_polynomial": forward_polynomial["metrics"],
        "embedding_from_xy_affine": reverse_affine["metrics"],
        "embedding_from_xy_polynomial": reverse_polynomial["metrics"],
        "coordinate_probe_polynomial_degree": forward_polynomial["degree"],
        "coordinate_probe_ridge": forward_polynomial["ridge"],
        "test_embedding_covariance_eigenvalues": eigenvalues.tolist(),
    }
    arrays = {
        "test_embeddings": encoded["test"].astype(np.float32),
        "test_xy": xy["test"].astype(np.float32),
        "test_xy_affine_prediction": forward_affine["predictions"],
        "test_xy_polynomial_prediction": forward_polynomial["predictions"],
    }
    return result, arrays


def matched_goal_pose_diagnostics(
    pairs: PairSplit,
    positions: np.ndarray,
    predicted: np.ndarray,
    *,
    bucket_radius: float = 0.25,
) -> dict[str, Any]:
    """Сравнивает состояния с одинаковым стартом и близкими координатами цели."""

    if pairs.values is None:
        raise ValueError("pairs have no teacher values")
    if bucket_radius <= 0:
        raise ValueError("bucket_radius must be positive")
    goal_xy = np.asarray(positions[pairs.goal_indices], dtype=np.float64)
    starts: dict[int, list[int]] = defaultdict(list)
    for index, start in enumerate(pairs.start_indices):
        starts[int(start)].append(index)

    spreads = []
    prediction_spreads = []
    disagreements = []
    distances = []
    coordinate_spans = []
    for rows in starts.values():
        remaining = list(dict.fromkeys(rows))
        while remaining:
            anchor = remaining.pop(0)
            nearby = [
                index
                for index in remaining
                if np.linalg.norm(goal_xy[index] - goal_xy[anchor]) <= bucket_radius
            ]
            selected = np.asarray([anchor, *nearby], dtype=np.int64)
            if len(selected) < 2 or len(np.unique(pairs.goal_indices[selected])) < 2:
                continue
            nearby_set = set(nearby)
            remaining = [index for index in remaining if index not in nearby_set]
            truth = np.asarray(pairs.values[selected], dtype=np.float64)
            estimate = np.asarray(predicted[selected], dtype=np.float64)
            spreads.append(float(truth.max() - truth.min()))
            prediction_spreads.append(float(estimate.max() - estimate.min()))
            distances.append(float(np.mean(pairs.distances[selected])))
            coordinate_spans.append(
                float(np.max(np.linalg.norm(goal_xy[selected] - goal_xy[anchor], axis=1)))
            )
            if pairs.ensemble_values is not None:
                members = np.asarray(pairs.ensemble_values[selected], dtype=np.float64)
                disagreements.append(
                    float(np.mean(members.max(axis=1) - members.min(axis=1)))
                )
    if not spreads:
        return {
            "groups": 0,
            "bucket_radius": float(bucket_radius),
            "note": "no repeated-start, matched-XY candidate groups were sampled",
        }
    return {
        "groups": len(spreads),
        "bucket_radius": float(bucket_radius),
        "teacher_mean_pose_spread": float(np.mean(spreads)),
        "teacher_median_pose_spread": float(np.median(spreads)),
        "teacher_p90_pose_spread": float(np.quantile(spreads, 0.90)),
        "student_mean_pose_spread": float(np.mean(prediction_spreads)),
        "mean_group_distance": float(np.mean(distances)),
        "mean_goal_xy_span": float(np.mean(coordinate_spans)),
        "max_goal_xy_span": float(np.max(coordinate_spans)),
        "mean_ensemble_disagreement": (
            float(np.mean(disagreements)) if disagreements else None
        ),
    }


def compare_model_signals(
    model_metrics: Mapping[str, Mapping[str, Any]],
    *,
    far_bin: int,
) -> dict[str, Any]:
    result: dict[str, Any] = {"far_distance_bin": int(far_bin)}
    full = model_metrics.get("full", {})
    xy = model_metrics.get("xy", {})
    latent = model_metrics.get("latent2", {})

    def far_r2(entry: Mapping[str, Any]) -> float | None:
        return entry.get("by_distance", {}).get(str(far_bin), {}).get("r2")

    full_r2 = far_r2(full)
    xy_r2 = far_r2(xy)
    latent_r2 = far_r2(latent)
    result.update(
        full_far_r2=full_r2,
        xy_far_r2=xy_r2,
        latent2_far_r2=latent_r2,
        full_minus_xy_far_r2=(
            float(full_r2 - xy_r2) if full_r2 is not None and xy_r2 is not None else None
        ),
        full_minus_latent2_far_r2=(
            float(full_r2 - latent_r2)
            if full_r2 is not None and latent_r2 is not None
            else None
        ),
    )
    latent_embedding = latent.get("embedding", {})
    readout = latent_embedding.get("coordinate_readout_polynomial", {})
    result["latent2_xy_readout_r2"] = readout.get("r2")
    result["latent2_xy_readout_p90"] = readout.get("p90_euclidean_error")
    return result
