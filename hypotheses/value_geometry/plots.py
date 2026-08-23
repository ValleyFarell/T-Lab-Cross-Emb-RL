"""Построение диагностических графиков эксперимента геометрии."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np


def generate_plots(
    output_dir: str | Path,
    *,
    metrics: Mapping[str, Any],
    predictions: Mapping[str, np.ndarray],
    values: np.ndarray,
    distance_bins: np.ndarray,
    embedding_arrays: Mapping[str, Mapping[str, np.ndarray]],
    histories: Mapping[str, list[Mapping[str, Any]]],
    seed: int = 0,
) -> list[str]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return []

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    created: list[str] = []
    palette = {
        "xy": "#11998e",
        "full": "#284b9b",
        "latent1": "#ee9b00",
        "latent2": "#c63d6c",
        "latent4": "#7354a3",
        "latent8": "#a75224",
        "pose": "#777777",
    }

    figure, axes = plt.subplots(1, 2, figsize=(12, 4.7), constrained_layout=True)
    for name, result in metrics.items():
        by_distance = result.get("by_distance", {})
        bins = sorted((int(index) for index in by_distance), key=int)
        scores = [by_distance[str(index)].get("r2") for index in bins]
        normalized = [by_distance[str(index)].get("nrmse") for index in bins]
        axes[0].plot(bins, scores, "o-", label=name, color=palette.get(name))
        axes[1].plot(bins, normalized, "o-", label=name, color=palette.get(name))
    axes[0].set(title="Held-out value prediction", xlabel="Maze-distance bin", ylabel="R²")
    axes[1].set(
        title="Error relative to target variation",
        xlabel="Maze-distance bin",
        ylabel="Normalized RMSE",
    )
    for axis in axes:
        axis.grid(alpha=0.20)
        axis.legend(frameon=False, fontsize=9)
    path = output / "value_quality_by_distance.png"
    figure.savefig(path, dpi=170, facecolor="white")
    plt.close(figure)
    created.append(path.name)

    names = list(predictions)
    if names:
        count = len(names)
        columns = min(3, count)
        rows = int(np.ceil(count / columns))
        figure, axes = plt.subplots(
            rows, columns, figsize=(4.6 * columns, 4.2 * rows), squeeze=False,
            constrained_layout=True,
        )
        maximum = min(len(values), 4_000)
        subset = rng.choice(len(values), size=maximum, replace=False)
        for axis, name in zip(axes.ravel(), names):
            axis.scatter(
                values[subset],
                predictions[name][subset],
                c=distance_bins[subset],
                s=7,
                alpha=0.42,
                cmap="viridis",
                linewidths=0,
            )
            low = min(float(np.min(values[subset])), float(np.min(predictions[name][subset])))
            high = max(float(np.max(values[subset])), float(np.max(predictions[name][subset])))
            axis.plot([low, high], [low, high], "--", color="#444444", linewidth=1)
            axis.set(title=name, xlabel="Frozen FB teacher", ylabel="Student prediction")
            axis.grid(alpha=0.15)
        for axis in axes.ravel()[len(names) :]:
            axis.axis("off")
        path = output / "teacher_vs_prediction.png"
        figure.savefig(path, dpi=170, facecolor="white")
        plt.close(figure)
        created.append(path.name)

    if histories:
        figure, axis = plt.subplots(figsize=(7.6, 4.8), constrained_layout=True)
        for name, history in histories.items():
            axis.plot(
                [entry["epoch"] for entry in history],
                [entry["validation_rmse"] for entry in history],
                label=name,
                color=palette.get(name),
            )
        axis.set(
            title="Validation learning curves", xlabel="Epoch", ylabel="Validation RMSE"
        )
        axis.grid(alpha=0.20)
        axis.legend(frameon=False)
        path = output / "training_curves.png"
        figure.savefig(path, dpi=170, facecolor="white")
        plt.close(figure)
        created.append(path.name)

    for name, arrays in embedding_arrays.items():
        embeddings = np.asarray(arrays["test_embeddings"])
        xy = np.asarray(arrays["test_xy"])
        if embeddings.ndim != 2 or embeddings.shape[1] < 2:
            continue
        subset = rng.choice(len(xy), size=min(len(xy), 6_000), replace=False)
        figure, axes = plt.subplots(2, 2, figsize=(10, 9), constrained_layout=True)
        scatter = axes[0, 0].scatter(
            embeddings[subset, 0], embeddings[subset, 1],
            c=xy[subset, 0], cmap="turbo", s=9, alpha=0.70, linewidths=0,
        )
        axes[0, 0].set(title="Embedding colored by physical x", xlabel="e₁", ylabel="e₂")
        figure.colorbar(scatter, ax=axes[0, 0], label="x")
        scatter = axes[0, 1].scatter(
            embeddings[subset, 0], embeddings[subset, 1],
            c=xy[subset, 1], cmap="turbo", s=9, alpha=0.70, linewidths=0,
        )
        axes[0, 1].set(title="Embedding colored by physical y", xlabel="e₁", ylabel="e₂")
        figure.colorbar(scatter, ax=axes[0, 1], label="y")
        scatter = axes[1, 0].scatter(
            xy[subset, 0], xy[subset, 1],
            c=embeddings[subset, 0], cmap="coolwarm", s=9, alpha=0.70, linewidths=0,
        )
        axes[1, 0].set(title="Physical maze colored by e₁", xlabel="x", ylabel="y")
        axes[1, 0].set_aspect("equal", adjustable="box")
        figure.colorbar(scatter, ax=axes[1, 0], label="e₁")
        scatter = axes[1, 1].scatter(
            xy[subset, 0], xy[subset, 1],
            c=embeddings[subset, 1], cmap="coolwarm", s=9, alpha=0.70, linewidths=0,
        )
        axes[1, 1].set(title="Physical maze colored by e₂", xlabel="x", ylabel="y")
        axes[1, 1].set_aspect("equal", adjustable="box")
        figure.colorbar(scatter, ax=axes[1, 1], label="e₂")
        path = output / f"{name}_embedding_geometry.png"
        figure.savefig(path, dpi=170, facecolor="white")
        plt.close(figure)
        created.append(path.name)

        prediction = np.asarray(arrays["test_xy_polynomial_prediction"])
        error = np.linalg.norm(prediction - xy, axis=1)
        figure, axes = plt.subplots(1, 2, figsize=(10, 4.4), constrained_layout=True)
        scatter = axes[0].scatter(
            xy[subset, 0], xy[subset, 1], c=error[subset],
            cmap="magma", s=10, linewidths=0,
        )
        axes[0].set(title="Post-hoc XY readout error", xlabel="x", ylabel="y")
        axes[0].set_aspect("equal", adjustable="box")
        figure.colorbar(scatter, ax=axes[0], label="Euclidean error")
        axes[1].hist(error, bins=40, color=palette.get(name, "#7354a3"), alpha=0.85)
        axes[1].axvline(0.5, color="#d1495b", linestyle="--", label="goal radius 0.5")
        axes[1].set(title="Held-out coordinate errors", xlabel="Euclidean error", ylabel="States")
        axes[1].legend(frameon=False)
        path = output / f"{name}_xy_readout_error.png"
        figure.savefig(path, dpi=170, facecolor="white")
        plt.close(figure)
        created.append(path.name)
    return created
