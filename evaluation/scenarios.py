"""Scenario definitions and validation shared by all controllers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


GridCell = tuple[int, int]


@dataclass(frozen=True)
class Scenario:
    scenario_id: str
    task_id: int | None
    environment_seed: int | None = None
    controller_seed: int = 0
    start_ij: GridCell | None = None
    goal_ij: GridCell | None = None

    def __post_init__(self) -> None:
        has_custom_start = self.start_ij is not None
        has_custom_goal = self.goal_ij is not None

        if has_custom_start != has_custom_goal:
            raise ValueError("start_ij and goal_ij must be provided together.")
        if self.task_id is None and not has_custom_start:
            raise ValueError("A scenario needs either task_id or custom grid cells.")
        if self.task_id is not None and has_custom_start:
            raise ValueError("task_id and custom grid cells are mutually exclusive.")

    @property
    def is_custom(self) -> bool:
        return self.start_ij is not None

    def reset_options(self) -> dict:
        if self.is_custom:
            return {
                "task_info": {
                    "init_ij": tuple(self.start_ij),
                    "goal_ij": tuple(self.goal_ij),
                }
            }
        return {"task_id": int(self.task_id)}


def xy_to_free_grid_cell(env, xy, *, name: str) -> GridCell:
    """Convert an exact free-cell center ``(x, y)`` to OGBench ``(i, j)``."""

    xy = np.asarray(xy, dtype=np.float64)
    if xy.shape != (2,) or not np.all(np.isfinite(xy)):
        raise ValueError(f"{name} must contain two finite coordinates, got {xy!r}.")

    base_env = env.unwrapped
    ij = tuple(int(value) for value in base_env.xy_to_ij(xy))
    maze_map = np.asarray(base_env.maze_map)

    if not (0 <= ij[0] < maze_map.shape[0] and 0 <= ij[1] < maze_map.shape[1]):
        raise ValueError(f"{name}={xy.tolist()} is outside the maze.")

    center_xy = np.asarray(base_env.ij_to_xy(ij), dtype=np.float64)
    if not np.allclose(xy, center_xy, atol=1e-8, rtol=0.0):
        raise ValueError(
            f"{name}={xy.tolist()} is not a cell center; nearest center is "
            f"{center_xy.tolist()} (grid cell {ij})."
        )

    if maze_map[ij] != 0:
        raise ValueError(f"{name}={xy.tolist()} points to a wall (grid cell {ij}).")

    return ij
