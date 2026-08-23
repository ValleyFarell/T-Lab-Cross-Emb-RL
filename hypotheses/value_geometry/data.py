"""Разделение траекторий и построение пар без утечки данных."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np


SPLIT_NAMES = ("train", "validation", "test")


def _integer(value: int, *, name: str, minimum: int = 1) -> int:
    if isinstance(value, (bool, np.bool_)) or int(value) != value:
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return value


def trajectory_ids(size: int, terminals: np.ndarray | None) -> np.ndarray:
    """Присваивает каждому состоянию идентификатор исходной траектории."""

    size = _integer(size, name="size")
    if terminals is None:
        return np.arange(size, dtype=np.int64)
    terminals = np.asarray(terminals).reshape(-1)
    if len(terminals) != size:
        raise ValueError("terminals length does not match the state count")
    return np.concatenate(
        (
            np.zeros(1, dtype=np.int64),
            np.cumsum(terminals[:-1] > 0, dtype=np.int64),
        )
    )


# Сначала разделяем целые траектории, чтобы соседние состояния не просочились в тест.
def split_state_indices(
    size: int,
    *,
    terminals: np.ndarray | None = None,
    max_states: int | None = 30_000,
    seed: int = 0,
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
) -> tuple[dict[str, np.ndarray], np.ndarray, str]:
    """Разделяет целые траектории без пересечения обучения и проверки."""

    size = _integer(size, name="size", minimum=3)
    if max_states is not None:
        max_states = _integer(max_states, name="max_states", minimum=3)
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must lie in (0, 1)")
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must lie in (0, 1)")
    if train_fraction + validation_fraction >= 1.0:
        raise ValueError("train_fraction + validation_fraction must be < 1")

    rng = np.random.default_rng(seed)
    groups = trajectory_ids(size, terminals)
    unique_groups = np.unique(groups)
    if terminals is not None and len(unique_groups) >= 3:
        shuffled = rng.permutation(unique_groups)
        number = len(shuffled)
        train_end = min(number - 2, max(1, int(number * train_fraction)))
        validation_count = max(1, int(number * validation_fraction))
        validation_end = min(number - 1, train_end + validation_count)
        selected_groups = {
            "train": shuffled[:train_end],
            "validation": shuffled[train_end:validation_end],
            "test": shuffled[validation_end:],
        }
        splits = {
            name: np.flatnonzero(np.isin(groups, selected))
            for name, selected in selected_groups.items()
        }
        strategy = "trajectory"
    else:
        shuffled = rng.permutation(size)
        train_end = min(size - 2, max(1, int(size * train_fraction)))
        validation_end = min(
            size - 1,
            max(train_end + 1, int(size * (train_fraction + validation_fraction))),
        )
        splits = {
            "train": shuffled[:train_end],
            "validation": shuffled[train_end:validation_end],
            "test": shuffled[validation_end:],
        }
        strategy = "random_state"

    if max_states is not None and sum(map(len, splits.values())) > max_states:
        fractions = {
            "train": train_fraction,
            "validation": validation_fraction,
            "test": 1.0 - train_fraction - validation_fraction,
        }
        requested = {
            name: max(1, int(max_states * fraction))
            for name, fraction in fractions.items()
        }
        requested["train"] += max_states - sum(requested.values())
        splits = {
            name: rng.choice(
                indices,
                size=min(len(indices), requested[name]),
                replace=False,
            )
            for name, indices in splits.items()
        }

    if any(len(indices) == 0 for indices in splits.values()):
        raise ValueError("could not construct three non-empty state splits")
    return {
        name: np.sort(np.asarray(indices, dtype=np.int64))
        for name, indices in splits.items()
    }, groups, strategy


@dataclass(frozen=True)
class StatePool:
    """Хранит непересекающийся набор состояний, доступных по индексам пар."""

    observations: np.ndarray
    positions: np.ndarray
    original_indices: np.ndarray
    trajectory_groups: np.ndarray
    split_indices: Mapping[str, np.ndarray]
    split_strategy: str


def build_state_pool(
    observations: np.ndarray,
    *,
    positions: np.ndarray | None = None,
    terminals: np.ndarray | None = None,
    max_states: int = 30_000,
    seed: int = 0,
) -> StatePool:
    observations = np.asarray(observations, dtype=np.float32)
    if observations.ndim != 2 or observations.shape[1] < 3:
        raise ValueError("observations must have shape [N, observation_dim >= 3]")
    if not np.all(np.isfinite(observations)):
        raise ValueError("observations contain non-finite values")
    positions = (
        observations[:, :2]
        if positions is None
        else np.asarray(positions, dtype=np.float32)
    )
    if positions.shape != (len(observations), 2):
        raise ValueError("positions must have shape [N, 2]")

    original_splits, groups, strategy = split_state_indices(
        len(observations),
        terminals=terminals,
        max_states=max_states,
        seed=seed,
    )
    ordered_original = np.concatenate(
        [original_splits[name] for name in SPLIT_NAMES]
    ).astype(np.int64, copy=False)
    compact_splits: dict[str, np.ndarray] = {}
    offset = 0
    for name in SPLIT_NAMES:
        count = len(original_splits[name])
        compact_splits[name] = np.arange(offset, offset + count, dtype=np.int64)
        offset += count

    return StatePool(
        observations=np.ascontiguousarray(observations[ordered_original]),
        positions=np.ascontiguousarray(positions[ordered_original]),
        original_indices=ordered_original,
        trajectory_groups=np.ascontiguousarray(groups[ordered_original]),
        split_indices=compact_splits,
        split_strategy=strategy,
    )


@dataclass(frozen=True)
class MazeGeometry:
    """Вычисляет расстояния между свободными клетками лабиринта."""

    centers: np.ndarray
    cell_distances: np.ndarray
    state_cells: np.ndarray
    cell_scale: float
    source: str

    @classmethod
    def from_environment(cls, env: Any, positions: np.ndarray) -> "MazeGeometry":
        base = getattr(env, "unwrapped", env)
        maze_map = np.asarray(getattr(base, "maze_map"))
        if maze_map.ndim != 2:
            raise ValueError("maze_map must be a rank-2 grid")
        free = np.argwhere(maze_map == 0)
        if len(free) < 2:
            raise ValueError("maze_map contains fewer than two free cells")
        centers = np.asarray(
            [np.asarray(base.ij_to_xy(tuple(map(int, ij))), dtype=np.float32) for ij in free],
            dtype=np.float32,
        )
        mapping = {tuple(map(int, ij)): index for index, ij in enumerate(free)}
        distances = np.full((len(free), len(free)), -1, dtype=np.int32)
        edge_lengths: list[float] = []
        for source, ij in enumerate(free):
            start = tuple(map(int, ij))
            distances[source, source] = 0
            queue: deque[tuple[int, int]] = deque([start])
            while queue:
                current = queue.popleft()
                current_id = mapping[current]
                for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    neighbor = (current[0] + di, current[1] + dj)
                    neighbor_id = mapping.get(neighbor)
                    if neighbor_id is None:
                        continue
                    if source == current_id:
                        edge_lengths.append(
                            float(np.linalg.norm(centers[source] - centers[neighbor_id]))
                        )
                    if distances[source, neighbor_id] >= 0:
                        continue
                    distances[source, neighbor_id] = distances[source, current_id] + 1
                    queue.append(neighbor)

        positions = np.asarray(positions, dtype=np.float32)
        cells = np.empty(len(positions), dtype=np.int32)
        for first in range(0, len(positions), 2_048):
            batch = positions[first : first + 2_048]
            squared = np.sum(
                (batch[:, None, :] - centers[None, :, :]) ** 2,
                axis=-1,
            )
            cells[first : first + len(batch)] = np.argmin(squared, axis=1)

        scale = float(np.median(edge_lengths)) if edge_lengths else 1.0
        return cls(
            centers=centers,
            cell_distances=distances,
            state_cells=cells,
            cell_scale=scale,
            source="static_maze_map_for_diagnostics_only",
        )

    def distance(
        self,
        start_indices: np.ndarray,
        goal_indices: np.ndarray,
        positions: np.ndarray,
    ) -> np.ndarray:
        start_indices = np.asarray(start_indices, dtype=np.int64)
        goal_indices = np.asarray(goal_indices, dtype=np.int64)
        cell_steps = self.cell_distances[
            self.state_cells[start_indices],
            self.state_cells[goal_indices],
        ]
        euclidean = np.linalg.norm(
            positions[start_indices] - positions[goal_indices], axis=-1
        )
        return np.where(
            cell_steps >= 0,
            cell_steps.astype(np.float32) * self.cell_scale,
            euclidean,
        ).astype(np.float32)


def pair_distances(
    pool: StatePool,
    start_indices: np.ndarray,
    goal_indices: np.ndarray,
    geometry: MazeGeometry | None,
) -> np.ndarray:
    if geometry is not None:
        return geometry.distance(start_indices, goal_indices, pool.positions)
    return np.linalg.norm(
        pool.positions[np.asarray(start_indices)]
        - pool.positions[np.asarray(goal_indices)],
        axis=-1,
    ).astype(np.float32)


def select_goal_indices(
    pool: StatePool,
    split_name: str,
    *,
    count: int,
    seed: int,
    pose_radius: float = 0.20,
    variants_per_location: int = 2,
) -> np.ndarray:
    """Подбирает пространственно разнообразные цели и близкие варианты позы."""

    if split_name not in SPLIT_NAMES:
        raise ValueError(f"unknown split {split_name!r}")
    count = _integer(count, name="count")
    variants_per_location = _integer(
        variants_per_location, name="variants_per_location"
    )
    if not np.isfinite(pose_radius) or pose_radius < 0:
        raise ValueError("pose_radius must be finite and non-negative")

    available = np.asarray(pool.split_indices[split_name], dtype=np.int64)
    count = min(count, len(available))
    rng = np.random.default_rng(seed)
    candidate_limit = min(len(available), max(2_000, count * 30))
    candidate_ids = rng.choice(available, size=candidate_limit, replace=False)
    candidate_xy = pool.positions[candidate_ids]
    selected: list[int] = []
    selected_set: set[int] = set()
    nearest_selected_sq = np.full(candidate_limit, np.inf, dtype=np.float64)

    while len(selected) < count:
        if not selected:
            anchor_position = int(rng.integers(candidate_limit))
        else:
            anchor_position = int(np.argmax(nearest_selected_sq))
        anchor_id = int(candidate_ids[anchor_position])
        anchor_xy = pool.positions[anchor_id]
        distances_sq = np.sum((candidate_xy - anchor_xy[None, :]) ** 2, axis=1)
        nearest_selected_sq = np.minimum(nearest_selected_sq, distances_sq)
        nearby = np.flatnonzero(distances_sq <= pose_radius**2 + 1e-12)
        if len(nearby) > 1:
            rng.shuffle(nearby)
            nearby = np.concatenate(
                (
                    np.asarray([anchor_position]),
                    nearby[nearby != anchor_position],
                )
            )
        else:
            nearby = np.asarray([anchor_position], dtype=np.int64)

        added = 0
        used_groups: set[int] = set()
        for location in nearby:
            goal_id = int(candidate_ids[int(location)])
            group = int(pool.trajectory_groups[goal_id])
            if goal_id in selected_set:
                continue
            if group in used_groups and len(nearby) > variants_per_location:
                continue
            selected.append(goal_id)
            selected_set.add(goal_id)
            used_groups.add(group)
            nearest_selected_sq[int(location)] = -1.0
            added += 1
            if len(selected) >= count or added >= variants_per_location:
                break

        if added == 0:
            remaining = [int(x) for x in candidate_ids if int(x) not in selected_set]
            if not remaining:
                break
            selected.append(remaining[0])
            selected_set.add(remaining[0])
            nearest_selected_sq[np.flatnonzero(candidate_ids == remaining[0])] = -1

    return np.asarray(selected[:count], dtype=np.int64)


def estimate_distance_edges(
    pool: StatePool,
    start_indices: np.ndarray,
    goal_indices: np.ndarray,
    *,
    geometry: MazeGeometry | None,
    number_of_bins: int = 4,
    seed: int = 0,
) -> np.ndarray:
    number_of_bins = _integer(number_of_bins, name="number_of_bins")
    rng = np.random.default_rng(seed)
    sample_size = min(20_000, max(2_000, len(goal_indices) * 30))
    starts = rng.choice(start_indices, size=sample_size, replace=True)
    goals = rng.choice(goal_indices, size=sample_size, replace=True)
    distances = pair_distances(pool, starts, goals, geometry)
    edges = np.quantile(
        distances, np.linspace(0.0, 1.0, number_of_bins + 1)[1:-1]
    )
    # Квантили дискретных расстояний могут совпадать; одинаковые границы
    # создали бы пустые интервалы, в которые невозможно поместить пример.
    return np.unique(np.asarray(edges, dtype=np.float32))


@dataclass(frozen=True)
class PairSplit:
    start_indices: np.ndarray
    goal_indices: np.ndarray
    distances: np.ndarray
    distance_bins: np.ndarray
    values: np.ndarray | None = None
    ensemble_values: np.ndarray | None = None

    def __post_init__(self) -> None:
        length = len(self.start_indices)
        if length == 0:
            raise ValueError("a pair split cannot be empty")
        for name in ("goal_indices", "distances", "distance_bins"):
            if len(getattr(self, name)) != length:
                raise ValueError(f"{name} has a different pair count")
        if self.values is not None and len(self.values) != length:
            raise ValueError("values has a different pair count")
        if self.ensemble_values is not None and len(self.ensemble_values) != length:
            raise ValueError("ensemble_values has a different pair count")

    def __len__(self) -> int:
        return len(self.start_indices)

    def with_values(
        self, values: np.ndarray, ensemble_values: np.ndarray | None = None
    ) -> "PairSplit":
        return PairSplit(
            start_indices=self.start_indices,
            goal_indices=self.goal_indices,
            distances=self.distances,
            distance_bins=self.distance_bins,
            values=np.asarray(values, dtype=np.float32),
            ensemble_values=(
                None
                if ensemble_values is None
                else np.asarray(ensemble_values, dtype=np.float32)
            ),
        )


def sample_pairs(
    pool: StatePool,
    split_name: str,
    goal_indices: np.ndarray,
    *,
    number_of_pairs: int,
    distance_edges: np.ndarray,
    geometry: MazeGeometry | None,
    seed: int,
    candidates_per_start: int = 8,
    matched_goal_radius: float = 0.25,
) -> PairSplit:
    """Формирует пары состояний с контролем распределения расстояний."""

    number_of_pairs = _integer(number_of_pairs, name="number_of_pairs")
    candidates_per_start = _integer(
        candidates_per_start, name="candidates_per_start"
    )
    starts_available = np.asarray(pool.split_indices[split_name], dtype=np.int64)
    goals = np.unique(np.asarray(goal_indices, dtype=np.int64))
    if len(goals) == 0:
        raise ValueError("goal_indices cannot be empty")
    if not np.all(np.isin(goals, starts_available)):
        raise ValueError("goal bank contains states from a different split")
    if not np.isfinite(matched_goal_radius) or matched_goal_radius < 0:
        raise ValueError("matched_goal_radius must be finite and non-negative")

    rng = np.random.default_rng(seed)
    edges = np.asarray(distance_edges, dtype=np.float32)
    number_of_bins = len(edges) + 1
    start_result = np.empty(number_of_pairs, dtype=np.int64)
    goal_result = np.empty(number_of_pairs, dtype=np.int64)
    distance_result = np.empty(number_of_pairs, dtype=np.float32)
    bins_result = np.empty(number_of_pairs, dtype=np.int16)
    cursor = 0
    goal_xy = pool.positions[goals]
    neighbor_distance = np.linalg.norm(
        goal_xy[:, None, :] - goal_xy[None, :, :], axis=-1
    )
    np.fill_diagonal(neighbor_distance, np.inf)
    matched_neighbors = [
        np.flatnonzero(row <= matched_goal_radius) for row in neighbor_distance
    ]
    matched_anchors = np.asarray(
        [index for index, neighbors in enumerate(matched_neighbors) if len(neighbors)],
        dtype=np.int64,
    )

    while cursor < number_of_pairs:
        start = int(rng.choice(starts_available))
        repeated = np.full(len(goals), start, dtype=np.int64)
        distances = pair_distances(pool, repeated, goals, geometry)
        bins = np.searchsorted(edges, distances, side="right")
        number_here = min(candidates_per_start, number_of_pairs - cursor)
        requested = (np.arange(number_here) + cursor) % number_of_bins
        rng.shuffle(requested)
        already_used: set[int] = set()
        forced_choices: list[int] = []
        if len(matched_anchors) and number_here >= 2:
            anchor = int(rng.choice(matched_anchors))
            neighbor = int(rng.choice(matched_neighbors[anchor]))
            forced_choices = [anchor, neighbor]
        for requested_bin in requested:
            if forced_choices:
                choice = forced_choices.pop(0)
            else:
                matches = np.flatnonzero(bins == requested_bin)
                if len(matches) == 0:
                    closest = np.abs(bins - requested_bin)
                    matches = np.flatnonzero(closest == closest.min())
                unused = np.asarray(
                    [index for index in matches if int(index) not in already_used],
                    dtype=np.int64,
                )
                choices = unused if len(unused) else matches
                choice = int(rng.choice(choices))
            already_used.add(choice)
            start_result[cursor] = start
            goal_result[cursor] = int(goals[choice])
            distance_result[cursor] = float(distances[choice])
            bins_result[cursor] = int(bins[choice])
            cursor += 1

    return PairSplit(
        start_indices=start_result,
        goal_indices=goal_result,
        distances=distance_result,
        distance_bins=bins_result,
    )


def split_summary(pool: StatePool, pairs: Mapping[str, PairSplit]) -> dict[str, Any]:
    summary: dict[str, Any] = {"split_strategy": pool.split_strategy}
    for name in SPLIT_NAMES:
        split = pairs[name]
        unique_bins, counts = np.unique(split.distance_bins, return_counts=True)
        summary[name] = {
            "states": int(len(pool.split_indices[name])),
            "trajectories": int(
                len(np.unique(pool.trajectory_groups[pool.split_indices[name]]))
            ),
            "pairs": int(len(split)),
            "unique_starts": int(len(np.unique(split.start_indices))),
            "unique_goals": int(len(np.unique(split.goal_indices))),
            "distance_bins": {
                str(int(index)): int(count)
                for index, count in zip(unique_bins, counts)
            },
        }
    return summary
