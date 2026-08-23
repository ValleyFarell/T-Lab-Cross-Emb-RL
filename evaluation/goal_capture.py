"""Вычисление метрик продолжений из сохранённых состояний возле цели."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class EntryState:
    """Хранит сохранённое физическое состояние для ветвления возле цели."""

    observation_index: int
    observation: np.ndarray
    distance: float


def find_first_entry_state(
    observations,
    goal_xy,
    *,
    entry_radius: float = 1.0,
    success_radius: float = 0.5,
) -> EntryState | None:
    """Находит первое сохранённое состояние внутри входного радиуса до достижения цели."""

    observations = np.asarray(observations)
    goal_xy = np.asarray(goal_xy, dtype=np.float64)

    if observations.ndim != 2 or observations.shape[1] < 2:
        raise ValueError(
            "observations must have shape [steps, observation_dim >= 2]"
        )
    if goal_xy.shape != (2,):
        raise ValueError(f"goal_xy must have shape (2,), got {goal_xy.shape}")
    if not 0.0 < success_radius < entry_radius:
        raise ValueError("expected 0 < success_radius < entry_radius")

    distances = np.linalg.norm(observations[:, :2] - goal_xy, axis=1)
    candidates = np.flatnonzero(
        (distances < float(entry_radius))
        & (distances > float(success_radius))
    )
    if candidates.size == 0:
        return None

    index = int(candidates[0])
    return EntryState(
        observation_index=index,
        observation=np.asarray(observations[index]).copy(),
        distance=float(distances[index]),
    )


def velocity_components(xy, xy_velocity, goal_xy) -> tuple[float, float]:
    """Разделяет скорость на направление к цели и касательную составляющую."""

    xy = np.asarray(xy, dtype=np.float64)
    xy_velocity = np.asarray(xy_velocity, dtype=np.float64)
    goal_xy = np.asarray(goal_xy, dtype=np.float64)
    delta = goal_xy - xy
    distance = float(np.linalg.norm(delta))

    if distance <= np.finfo(np.float64).eps:
        return 0.0, float(np.linalg.norm(xy_velocity))

    radial = float(np.dot(xy_velocity, delta / distance))
    tangential_sq = max(0.0, float(np.dot(xy_velocity, xy_velocity)) - radial**2)
    return radial, float(np.sqrt(tangential_sq))


def count_radius_exits(distances, *, radius: float = 1.0) -> int:
    """Считает выходы траектории за пределы заданного радиуса."""

    distances = np.asarray(distances, dtype=np.float64)
    if distances.ndim != 1:
        raise ValueError("distances must be one-dimensional")
    if distances.size < 2:
        return 0
    return int(np.count_nonzero((distances[:-1] <= radius) & (distances[1:] > radius)))


def summarize_branch(
    positions,
    radial_velocities,
    tangential_speeds,
    torso_heights,
    goal_xy,
    *,
    success_radius: float = 0.5,
    entry_radius: float = 1.0,
) -> dict:
    """Вычисляет итоговые метрики одного продолжения возле цели."""

    positions = np.asarray(positions, dtype=np.float64)
    goal_xy = np.asarray(goal_xy, dtype=np.float64)
    radial_velocities = np.asarray(radial_velocities, dtype=np.float64)
    tangential_speeds = np.asarray(tangential_speeds, dtype=np.float64)
    torso_heights = np.asarray(torso_heights, dtype=np.float64)

    if positions.ndim != 2 or positions.shape[1] != 2 or len(positions) == 0:
        raise ValueError("positions must have non-empty shape [states, 2]")

    distances = np.linalg.norm(positions - goal_xy, axis=1)
    hits = np.flatnonzero(distances <= success_radius)
    path_length = float(np.linalg.norm(np.diff(positions, axis=0), axis=1).sum())
    closest_index = int(np.argmin(distances))

    return {
        "hit_success_radius": bool(hits.size),
        # Нулевое состояние является точкой ветвления, поэтому первое состояние
        # после действия имеет номер один, совпадающий с индексом массива.
        "hit_step": int(hits[0]) if hits.size else None,
        "initial_distance": float(distances[0]),
        "minimum_distance": float(distances[closest_index]),
        "closest_state_index": closest_index,
        "final_distance": float(distances[-1]),
        "path_length": path_length,
        "radius_exits": count_radius_exits(distances, radius=entry_radius),
        "left_entry_radius": bool(np.any(distances > entry_radius)),
        "mean_radial_velocity": (
            float(radial_velocities.mean()) if radial_velocities.size else None
        ),
        "mean_tangential_speed": (
            float(tangential_speeds.mean()) if tangential_speeds.size else None
        ),
        "radial_velocity_at_closest_pre_state": (
            float(radial_velocities[min(closest_index, radial_velocities.size - 1)])
            if radial_velocities.size
            else None
        ),
        "minimum_torso_height": (
            float(torso_heights.min()) if torso_heights.size else None
        ),
        "fell_below_0_3": bool(
            torso_heights.size and np.any(torso_heights < 0.3)
        ),
    }
