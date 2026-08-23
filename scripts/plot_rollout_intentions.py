"""Отрисовка выполненных намерений вдоль сохранённой траектории."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import numpy as np

from evaluation.visualization import _draw_maze_walls
from probes.intention_xy import IntentionXYDecoder
from utils.env_utils import make_env_and_datasets


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            'Отрисовка выполненных намерений вдоль сохранённой траектории.'
        )
    )
    parser.add_argument(
        "--trajectory",
        type=Path,
        required=True,
        help='Путь к сохранённому файлу trajectory.npz.',
    )
    parser.add_argument(
        "--decoder",
        type=Path,
        default=Path("artifacts/intention_xy_decoder_deep"),
        help='Каталог обученного декодера или путь к decoder.npz.',
    )
    parser.add_argument(
        "--env-name",
        default="ogbench-antmaze-medium-navigate-v0",
    help='Имя используемой среды OGBench.',
    )
    parser.add_argument(
        "--step-size",
        type=int,
        default=10,
        help='Интервал шагов между отображаемыми намерениями.',
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help='Путь сохраняемого изображения или итогового файла.',
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help='Дополнительно открывает окно графика после сохранения.',
    )
    args = parser.parse_args(argv)
    if args.step_size <= 0:
        parser.error("--step-size must be positive")
    return args


def _load_rollout(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        missing = {"positions", "intentions"} - set(data.files)
        if missing:
            raise KeyError(
                f"{path} is missing arrays: {', '.join(sorted(missing))}; "
                f"available arrays: {', '.join(data.files)}"
            )
        positions = np.asarray(data["positions"], dtype=np.float64)
        intentions = np.asarray(data["intentions"], dtype=np.float32)

    if positions.ndim != 2 or positions.shape[1] != 2:
        raise ValueError(f"positions must have shape (T, 2), got {positions.shape}")
    if intentions.ndim == 3 and intentions.shape[1] == 1:
        intentions = intentions[:, 0, :]
    if intentions.ndim != 2:
        raise ValueError(
            f"intentions must have shape (T, D), got {intentions.shape}"
        )
    if len(positions) != len(intentions):
        raise ValueError(
            "positions and intentions must have equal time length, got "
            f"{len(positions)} and {len(intentions)}"
        )
    if len(positions) == 0:
        raise ValueError("trajectory contains no rollout steps")
    if not np.all(np.isfinite(positions)) or not np.all(np.isfinite(intentions)):
        raise ValueError("trajectory positions or intentions contain non-finite values")
    return positions, intentions


def _output_paths(trajectory: Path, explicit_output: Path | None, step_size: int):
    if explicit_output is None:
        image_path = trajectory.parent / f"decoded_intentions_step_{step_size}.png"
    else:
        image_path = explicit_output
        if image_path.suffix.lower() != ".png":
            raise ValueError("--output must have a .png suffix")
    image_path.parent.mkdir(parents=True, exist_ok=True)
    return image_path, image_path.with_suffix(".npz"), image_path.with_suffix(".json")


def _plot(
    env,
    positions,
    sampled_steps,
    sampled_positions,
    decoded_xy,
    image_path: Path,
    *,
    method_label: str,
):
    fig, ax = plt.subplots(figsize=(9, 8))
    bounds = _draw_maze_walls(ax, env)

    ax.plot(
        positions[:, 0],
        positions[:, 1],
        color="#4c78a8",
        linewidth=2.0,
        alpha=0.8,
        label="Ant trajectory",
        zorder=2,
    )

    norm = Normalize(vmin=float(sampled_steps[0]), vmax=float(sampled_steps[-1] or 1))
    colors = plt.get_cmap("viridis")(norm(sampled_steps))
    delta = decoded_xy - sampled_positions
    ax.quiver(
        sampled_positions[:, 0],
        sampled_positions[:, 1],
        delta[:, 0],
        delta[:, 1],
        color=colors,
        angles="xy",
        scale_units="xy",
        scale=1.0,
        width=0.004,
        headwidth=4.0,
        headlength=5.0,
        alpha=0.8,
        zorder=3,
    )
    ax.scatter(
        sampled_positions[:, 0],
        sampled_positions[:, 1],
        c=sampled_steps,
        cmap="viridis",
        norm=norm,
        s=32,
        marker="o",
        edgecolor="white",
        linewidth=0.5,
        label="Ant at sampled step",
        zorder=4,
    )
    ax.scatter(
        decoded_xy[:, 0],
        decoded_xy[:, 1],
        c=sampled_steps,
        cmap="viridis",
        norm=norm,
        s=58,
        marker="X",
        edgecolor="white",
        linewidth=0.6,
        label="Decoded intention XY",
        zorder=5,
    )
    ax.scatter(
        [positions[0, 0]],
        [positions[0, 1]],
        s=100,
        marker="o",
        color="#2ca02c",
        label="Start",
        zorder=6,
    )
    ax.scatter(
        [positions[-1, 0]],
        [positions[-1, 1]],
        s=110,
        marker="s",
        color="#d62728",
        label="Last saved position",
        zorder=6,
    )

    all_x = np.concatenate((positions[:, 0], decoded_xy[:, 0]))
    all_y = np.concatenate((positions[:, 1], decoded_xy[:, 1]))
    margin = 0.75 * bounds["maze_unit"]
    ax.set_xlim(min(bounds["xmin"], np.min(all_x)) - margin, max(bounds["xmax"], np.max(all_x)) + margin)
    ax.set_ylim(min(bounds["ymin"], np.min(all_y)) - margin, max(bounds["ymax"], np.max(all_y)) + margin)
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(method_label)
    ax.grid(False)
    ax.legend(loc="upper left")
    colorbar = fig.colorbar(ScalarMappable(norm=norm, cmap="viridis"), ax=ax)
    colorbar.set_label("rollout step")
    fig.savefig(image_path, dpi=220, bbox_inches="tight")
    return fig


def main(argv=None):
    args = parse_args(argv)
    positions, intentions = _load_rollout(args.trajectory)
    decoder = IntentionXYDecoder.load(args.decoder)
    if intentions.shape[1] != decoder.latent_dim:
        raise ValueError(
            f"trajectory intention dim {intentions.shape[1]} does not match "
            f"decoder latent dim {decoder.latent_dim}"
        )

    sampled_steps = np.arange(0, len(intentions), args.step_size, dtype=np.int64)
    sampled_positions = positions[sampled_steps]
    sampled_intentions = intentions[sampled_steps]
    decoded_xy = np.asarray(decoder.predict(sampled_intentions), dtype=np.float64)
    if decoded_xy.shape != (len(sampled_steps), 2) or not np.all(np.isfinite(decoded_xy)):
        raise RuntimeError(f"decoder returned invalid array with shape {decoded_xy.shape}")

    image_path, data_path, summary_path = _output_paths(
        args.trajectory,
        args.output,
        args.step_size,
    )
    env = make_env_and_datasets(args.env_name, env_only=True)
    figure = _plot(
        env,
        positions,
        sampled_steps,
        sampled_positions,
        decoded_xy,
        image_path,
        method_label=(
            f"Executed intentions every {args.step_size} steps "
            f"({args.trajectory.parent.name})"
        ),
    )

    intention_norms = np.linalg.norm(sampled_intentions, axis=1)
    decoded_displacements = np.linalg.norm(decoded_xy - sampled_positions, axis=1)
    np.savez_compressed(
        data_path,
        steps=sampled_steps,
        positions=sampled_positions,
        intentions=sampled_intentions,
        decoded_xy=decoded_xy,
        intention_norms=intention_norms,
        decoded_displacements=decoded_displacements,
    )
    summary = {
        "trajectory": str(args.trajectory),
        "decoder": str(args.decoder),
        "environment": args.env_name,
        "total_steps": int(len(intentions)),
        "step_size": args.step_size,
        "number_of_decoded_intentions": int(len(sampled_steps)),
        "sampled_steps": sampled_steps.tolist(),
        "intention_norm": {
            "min": float(np.min(intention_norms)),
            "mean": float(np.mean(intention_norms)),
            "max": float(np.max(intention_norms)),
        },
        "decoded_displacement": {
            "min": float(np.min(decoded_displacements)),
            "mean": float(np.mean(decoded_displacements)),
            "median": float(np.median(decoded_displacements)),
            "max": float(np.max(decoded_displacements)),
        },
    }
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))
    print(f"image: {image_path}")
    print(f"data: {data_path}")
    print(f"summary: {summary_path}")
    if args.show:
        plt.show()
    plt.close(figure)


if __name__ == "__main__":
    main()
