from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _get_maze_spec(env):
    """
    Read the true maze specification from OGBench AntMaze.

    Facts established from the environment:
    - env.unwrapped.maze_map is an 8x8 discrete occupancy grid
    - values are {0, 1}
    - 1 denotes wall, 0 denotes free space
    - env.unwrapped._maze_unit is the spacing between neighboring free-cell centers
    """
    u = env.unwrapped

    maze_map = np.asarray(u.maze_map)
    maze_unit = float(u._maze_unit)

    if maze_map.ndim != 2:
        raise ValueError(f"Expected a 2D maze map, got shape {maze_map.shape}")

    unique = set(np.unique(maze_map).tolist())
    if not unique.issubset({0, 1}):
        raise ValueError(f"Unexpected maze values: {sorted(unique)}")

    return maze_map, maze_unit


def _cell_center_to_world(row, col, maze_shape, maze_unit):
    """
    Convert OGBench maze_map indices to MuJoCo world coordinates.

    maze_map row index already follows the same vertical orientation
    as the rendered world maze.
    """
    x_center = (col - 1) * maze_unit
    y_center = (row - 1) * maze_unit

    return x_center, y_center


def _wall_rectangle(row, col, maze_shape, maze_unit):
    """
    Return the lower-left corner and size of the wall rectangle in world coordinates.
    """
    x_center, y_center = _cell_center_to_world(row, col, maze_shape, maze_unit)
    half = maze_unit / 2.0
    return x_center - half, y_center - half, maze_unit, maze_unit


def _draw_maze_walls(ax, env):
    maze_map, maze_unit = _get_maze_spec(env)

    xs_min, xs_max = [], []
    ys_min, ys_max = [], []

    for row in range(maze_map.shape[0]):
        for col in range(maze_map.shape[1]):
            if maze_map[row, col] == 1:
                x, y, w, h = _wall_rectangle(row, col, maze_map.shape, maze_unit)

                ax.add_patch(
                    plt.Rectangle(
                        (x, y),
                        w,
                        h,
                        facecolor="#444444",
                        edgecolor="#111111",
                        linewidth=1.0,
                        zorder=1,
                    )
                )

                xs_min.append(x)
                xs_max.append(x + w)
                ys_min.append(y)
                ys_max.append(y + h)

    if not xs_min:
        raise RuntimeError("No walls were drawn from maze_map.")

    bounds = {
        "xmin": min(xs_min),
        "xmax": max(xs_max),
        "ymin": min(ys_min),
        "ymax": max(ys_max),
        "maze_unit": maze_unit,
    }
    return bounds


def plot_path(
    positions,
    start_xy,
    goal_xy,
    success,
    output_file,
    env,
):
    """
    Draw the Ant trajectory on top of the true maze geometry.

    This function is for visualization only.
    The maze is NOT passed into the controller/policy.
    """
    output_file = Path(output_file)

    positions = np.asarray(positions, dtype=float)
    start_xy = np.asarray(start_xy, dtype=float)
    goal_xy = np.asarray(goal_xy, dtype=float)

    fig, ax = plt.subplots(figsize=(8, 8))

    bounds = _draw_maze_walls(ax, env)

    ax.plot(
        positions[:, 0],
        positions[:, 1],
        linewidth=2.5,
        color="#1f77b4",
        label="agent path",
        zorder=3,
    )

    ax.scatter(
        [start_xy[0]],
        [start_xy[1]],
        s=90,
        color="#2ca02c",
        marker="o",
        label="start",
        zorder=4,
    )

    ax.scatter(
        [goal_xy[0]],
        [goal_xy[1]],
        s=220,
        color="#d62728",
        marker="*",
        label="goal",
        zorder=4,
    )

    margin = 0.75 * bounds["maze_unit"]
    ax.set_xlim(bounds["xmin"] - margin, bounds["xmax"] + margin)
    ax.set_ylim(bounds["ymin"] - margin, bounds["ymax"] + margin)

    ax.set_aspect("equal")
    ax.grid(False)
    ax.legend(loc="upper left")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"success={success}, steps={len(positions)}")

    fig.savefig(output_file, dpi=220, bbox_inches="tight")
    plt.close(fig)
