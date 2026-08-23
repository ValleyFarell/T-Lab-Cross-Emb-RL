"""Проверка ориентации карты лабиринта по сохранённой траектории."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from utils.env_utils import make_env_and_datasets


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--trajectory",
        type=Path,
        required=True,
        help='Путь к сохранённому файлу trajectory.npz.',
    )
    parser.add_argument(
        "--env-name",
        type=str,
        default="ogbench-antmaze-medium-navigate-v0",
    help='Имя используемой среды OGBench.',
    )
    parser.add_argument(
        "--goal",
        nargs=2,
        type=float,
        default=None,
        metavar=("X", "Y"),
        help='Необязательная цель; без неё используется соседний summary.json.',
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("maze_orientation_check.png"),
    help='Путь к сохраняемому изображению или итоговому JSON-файлу.',
    )
    return parser.parse_args()


def load_route(trajectory_path: Path):
    data = np.load(trajectory_path)

    if "positions" not in data.files:
        raise KeyError(
            f"'positions' is missing from {trajectory_path}. "
            f"Available arrays: {data.files}"
        )

    positions = np.asarray(data["positions"], dtype=float)

    if positions.ndim != 2 or positions.shape[1] != 2:
        raise ValueError(
            f"Expected positions with shape [T, 2], got {positions.shape}"
        )

    if len(positions) == 0:
        raise ValueError("The saved route contains no positions.")

    return positions


def load_goal(trajectory_path: Path, explicit_goal, env_name=None):
    if explicit_goal is not None:
        return np.asarray(explicit_goal, dtype=float)

    summary_path = trajectory_path.parent / "summary.json"

    if summary_path.exists():
        with summary_path.open("r", encoding="utf-8") as f:
            summary = json.load(f)

        if "goal_xy" in summary:
            return np.asarray(summary["goal_xy"], dtype=float)

    if env_name == "ogbench-antmaze-medium-navigate-v0":
        return np.asarray([20.0, 20.0], dtype=float)

    raise ValueError(
        "Goal is unavailable. Pass --goal X Y or provide summary.json."
    )


def maze_variants(maze_map):
    """Перечисляет симметрии карты для проверки её ориентации."""
    transposed = maze_map.T

    return [
        ("original", maze_map),
        ("flip left-right", np.fliplr(maze_map)),
        ("flip up-down", np.flipud(maze_map)),
        ("rotate 180", np.flipud(np.fliplr(maze_map))),
        ("transpose", transposed),
        ("transpose + flip L-R", np.fliplr(transposed)),
        ("transpose + flip U-D", np.flipud(transposed)),
        ("transpose + rotate 180", np.flipud(np.fliplr(transposed))),
    ]


def cell_center_to_world(row, col, maze_shape, maze_unit):
    """Переводит центр клетки в мировые координаты по соглашению визуализации."""
    height, _ = maze_shape
    x_center = (col - 1) * maze_unit
    y_center = ((height - 2) - row) * maze_unit
    return x_center, y_center


def draw_walls(ax, maze_map, maze_unit):
    half = maze_unit / 2.0

    for row in range(maze_map.shape[0]):
        for col in range(maze_map.shape[1]):
            if maze_map[row, col] != 1:
                continue

            x_center, y_center = cell_center_to_world(
                row,
                col,
                maze_map.shape,
                maze_unit,
            )

            ax.add_patch(
                plt.Rectangle(
                    (x_center - half, y_center - half),
                    maze_unit,
                    maze_unit,
                    facecolor="#444444",
                    edgecolor="#111111",
                    linewidth=0.8,
                    zorder=1,
                )
            )


def fixed_world_bounds(maze_shape, maze_unit):
    """Возвращает неизменные границы изображения лабиринта."""
    height, width = maze_shape
    half = maze_unit / 2.0

    xmin = (0 - 1) * maze_unit - half
    xmax = ((width - 1) - 1) * maze_unit + half

    y_top_center = ((height - 2) - 0) * maze_unit
    y_bottom_center = ((height - 2) - (height - 1)) * maze_unit

    ymin = y_bottom_center - half
    ymax = y_top_center + half

    margin = 0.75 * maze_unit

    return (
        xmin - margin,
        xmax + margin,
        ymin - margin,
        ymax + margin,
    )


def main():
    args = parse_args()

    positions = load_route(args.trajectory)
    goal_xy = load_goal(args.trajectory, args.goal, args.env_name)
    start_xy = positions[0]

    env = make_env_and_datasets(
        args.env_name,
        env_only=True,
    )

    maze_map = np.asarray(env.unwrapped.maze_map)
    maze_unit = float(env.unwrapped._maze_unit)

    if maze_map.ndim != 2:
        raise ValueError(f"Expected 2D maze_map, got {maze_map.shape}")

    unique = set(np.unique(maze_map).tolist())
    if not unique.issubset({0, 1}):
        raise ValueError(f"Unexpected maze values: {sorted(unique)}")

    variants = maze_variants(maze_map)

    xmin, xmax, ymin, ymax = fixed_world_bounds(
        maze_map.shape,
        maze_unit,
    )

    fig, axes = plt.subplots(
        2,
        4,
        figsize=(18, 9),
        sharex=True,
        sharey=True,
    )

    for ax, (name, transformed_map) in zip(axes.flat, variants):
        draw_walls(
            ax,
            transformed_map,
            maze_unit,
        )

        ax.plot(
            positions[:, 0],
            positions[:, 1],
            linewidth=2.2,
            color="#1f77b4",
            label="agent path",
            zorder=3,
        )

        ax.scatter(
            [start_xy[0]],
            [start_xy[1]],
            s=55,
            color="#2ca02c",
            marker="o",
            label="start",
            zorder=4,
        )

        ax.scatter(
            [goal_xy[0]],
            [goal_xy[1]],
            s=130,
            color="#d62728",
            marker="*",
            label="goal",
            zorder=4,
        )

        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(name)
        ax.grid(False)

    axes[0, 0].legend(loc="upper left")

    fig.suptitle(
        f"Same trajectory, 8 maze-map orientations\n"
        f"{args.trajectory}",
        fontsize=14,
    )

    fig.supxlabel("MuJoCo x")
    fig.supylabel("MuJoCo y")
    fig.tight_layout(rect=(0.02, 0.02, 1.0, 0.94))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        args.out,
        dpi=220,
        bbox_inches="tight",
    )
    plt.close(fig)

    print(f"positions shape: {positions.shape}")
    print(f"start_xy: {start_xy.tolist()}")
    print(f"goal_xy: {goal_xy.tolist()}")
    print(f"maze shape: {maze_map.shape}")
    print(f"maze_unit: {maze_unit}")
    print(f"fixed limits: x=({xmin}, {xmax}), y=({ymin}, {ymax})")
    print(f"saved_to: {args.out}")


if __name__ == "__main__":
    main()
