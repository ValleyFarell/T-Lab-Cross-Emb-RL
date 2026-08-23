"""Двухточечное планирование с локальной первой подцелью."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from numbers import Integral
from typing import Any, Mapping

import numpy as np


@dataclass(frozen=True)
class LocalTerminalSelection:
    """Хранит выбранное намерение и согласованный набор диагностических полей."""

    intention: Any
    diagnostics: Mapping[str, Any]


def _positive_int(value, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _finite_float(value, *, name: str, allow_zero: bool = False) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    minimum_ok = parsed >= 0.0 if allow_zero else parsed > 0.0
    if not np.isfinite(parsed) or not minimum_ok:
        qualifier = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{name} must be {qualifier} and finite")
    return parsed


def _finite_vector(value, *, name: str, length: int | None = None) -> np.ndarray:
    vector = np.asarray(value)
    if vector.ndim != 1 or vector.dtype.kind not in "iuf" or vector.size == 0:
        raise ValueError(f"{name} must be a non-empty numeric vector")
    if length is not None and vector.size != length:
        raise ValueError(f"{name} must contain exactly {length} values")
    vector = vector.astype(np.float32, copy=False)
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must contain only finite values")
    return vector


class LocalTwoSwitchPlanner:
    """Оценивает две подцели, локальную первую точку и прямой резервный вариант."""

    MODE_DIRECT = 0
    MODE_TWO_SWITCH = 1
    MODE_TERMINAL_DIRECT = 2
    MODE_TERMINAL_BASELINE = 3

    def __init__(
        self,
        frozen_fb,
        offline_dataset,
        goal_xy,
        *,
        candidates_per_cell: int = 10,
        grid_cell_size: float = 4.0,
        local_radius: float = 5.0,
        max_local_candidates: int = 32,
        finish_radius: float = 2.0,
        finish_mode: str = "direct",
        direct_latent_mode: str = "raw",
        pair_batch_size: int = 1024,
        eta_epsilon: float = 1e-6,
        switch_margin: float = 0.0,
        enable_zero_level: bool = True,
    ):
        self.frozen_fb = frozen_fb
        self.goal_xy = _finite_vector(goal_xy, name="goal_xy", length=2).astype(
            np.float64
        )
        self.candidates_per_cell = _positive_int(
            candidates_per_cell, name="candidates_per_cell"
        )
        self.grid_cell_size = _finite_float(
            grid_cell_size, name="grid_cell_size"
        )
        self.local_radius = _finite_float(local_radius, name="local_radius")
        self.max_local_candidates = _positive_int(
            max_local_candidates, name="max_local_candidates"
        )
        self.finish_radius = _finite_float(
            finish_radius, name="finish_radius", allow_zero=True
        )
        self.pair_batch_size = _positive_int(
            pair_batch_size, name="pair_batch_size"
        )
        self.eta_epsilon = _finite_float(eta_epsilon, name="eta_epsilon")
        self.switch_margin = _finite_float(
            switch_margin, name="switch_margin", allow_zero=True
        )
        if finish_mode not in {"direct", "baseline"}:
            raise ValueError("finish_mode must be 'direct' or 'baseline'")
        if direct_latent_mode not in {"raw", "normalized"}:
            raise ValueError("direct_latent_mode must be 'raw' or 'normalized'")
        if not isinstance(enable_zero_level, (bool, np.bool_)):
            raise ValueError("enable_zero_level must be boolean")
        self.finish_mode = finish_mode
        self.direct_latent_mode = direct_latent_mode
        self.enable_zero_level = bool(enable_zero_level)

        observations = np.asarray(offline_dataset["observations"])
        if (
            observations.ndim != 2
            or observations.shape[0] == 0
            or observations.shape[1] < 2
            or observations.dtype.kind not in "iuf"
            or not np.all(np.isfinite(observations))
        ):
            raise ValueError(
                "offline observations must be a non-empty, finite numeric 2D array"
            )
        self.source_candidate_count = int(observations.shape[0])
        if "qpos" in offline_dataset:
            qpos = np.asarray(offline_dataset["qpos"])
            if (
                qpos.ndim != 2
                or qpos.shape[0] != observations.shape[0]
                or qpos.shape[1] < 2
                or not np.all(np.isfinite(qpos[:, :2]))
            ):
                raise ValueError(
                    "offline qpos must contain finite XY aligned with observations"
                )
            all_xy = qpos[:, :2].astype(np.float64, copy=False)
            self.xy_source = "offline_dataset.qpos[:, :2]"
        else:
            all_xy = observations[:, :2].astype(np.float64, copy=False)
            self.xy_source = "offline_dataset.observations[:, :2]"

        source_indices, cell_coordinates, per_cell_counts = self._stratify_cells(
            all_xy
        )
        self.source_indices = source_indices
        self.cell_coordinates = cell_coordinates
        self.per_cell_counts = per_cell_counts
        self.candidates = np.ascontiguousarray(
            observations[source_indices].astype(np.float32, copy=False)
        )
        self.candidate_xy = np.ascontiguousarray(all_xy[source_indices])
        self.candidate_count = int(len(source_indices))

        checksum = hashlib.sha256()
        checksum.update(self.candidates.tobytes())
        checksum.update(self.source_indices.tobytes())
        self.candidate_checksum = checksum.hexdigest()

        latent = frozen_fb.backward_repr(self.candidates)
        self.candidate_latents = np.asarray(
            frozen_fb.normalize_latent(latent), dtype=np.float32
        )
        if (
            self.candidate_latents.ndim != 2
            or self.candidate_latents.shape[0] != self.candidate_count
            or not np.all(np.isfinite(self.candidate_latents))
        ):
            raise ValueError("B must produce one finite latent for each candidate")
        self.latent_dim = int(self.candidate_latents.shape[1])

        self.self_forward = self._mean_forward(
            self.candidates, self.candidate_latents
        )
        self.self_measure = np.einsum(
            "ij,ij->i", self.self_forward, self.candidate_latents, optimize=True
        )
        self._goal_key: bytes | None = None
        self._goal_reward: np.ndarray | None = None
        self._goal_policy: np.ndarray | None = None
        self._self_goal: np.ndarray | None = None
        self._terminal_goal: np.ndarray | None = None
        self._pair_goal: np.ndarray | None = None
        self._pair_eta: np.ndarray | None = None
        self._pair_valid: np.ndarray | None = None
        self._pair_clipped: np.ndarray | None = None
        self._row_cached: np.ndarray | None = None

    def _stratify_cells(self, xy: np.ndarray):
        """Равномерно выбирает реальные состояния из разных клеток лабиринта."""

        keys = np.rint(xy / self.grid_cell_size).astype(np.int64)
        unique_cells, inverse, counts = np.unique(
            keys, axis=0, return_inverse=True, return_counts=True
        )
        order = np.argsort(inverse, kind="stable")
        starts = np.concatenate(([0], np.cumsum(counts[:-1])))
        selected = []
        selected_counts = []
        for cell, start, count in zip(unique_cells, starts, counts):
            members = order[int(start) : int(start + count)]
            take = min(self.candidates_per_cell, len(members))
            center = cell.astype(np.float64) * self.grid_cell_size
            distance = np.linalg.norm(xy[members] - center[None, :], axis=1)
            center_member = int(members[int(np.argmin(distance))])
            if take == 1:
                chosen = np.asarray([center_member], dtype=np.int64)
            else:
                positions = np.linspace(
                    0, len(members) - 1, take, dtype=np.int64
                )
                sampled = members[positions].tolist()
                if center_member not in sampled:
                    sampled[-1] = center_member
                chosen = np.asarray(sorted(set(sampled)), dtype=np.int64)
                if len(chosen) < take:
                    missing = take - len(chosen)
                    extras = [i for i in members if int(i) not in set(chosen)]
                    chosen = np.sort(
                        np.concatenate(
                            [chosen, np.asarray(extras[:missing], dtype=np.int64)]
                        )
                    )
            selected.append(chosen)
            selected_counts.append(len(chosen))
        return (
            np.concatenate(selected).astype(np.int64, copy=False),
            unique_cells.astype(np.int64, copy=False),
            np.asarray(selected_counts, dtype=np.int64),
        )

    def _mean_forward(self, observations, intentions) -> np.ndarray:
        observations = np.asarray(observations, dtype=np.float32)
        intentions = np.asarray(intentions, dtype=np.float32)
        if observations.ndim != 2 or intentions.ndim != 2:
            raise ValueError("forward inputs must be batched matrices")
        if observations.shape[0] != intentions.shape[0]:
            raise ValueError("forward observations and intentions must align")
        value = np.asarray(self.frozen_fb.forward_repr(observations, intentions))
        if value.ndim == 2:
            value = value[None, :, :]
        expected = (observations.shape[0], intentions.shape[1])
        if value.ndim != 3 or tuple(value.shape[1:]) != expected:
            raise ValueError(
                "forward_repr must return (ensemble, batch, latent_dim), "
                f"got {value.shape}"
            )
        return value.astype(np.float64, copy=False).mean(axis=0)

    def _safe_eta(self, numerator, denominator):
        numerator = np.asarray(numerator, dtype=np.float64)
        denominator = np.asarray(denominator, dtype=np.float64)
        valid = (
            np.isfinite(numerator)
            & np.isfinite(denominator)
            & (np.abs(denominator) >= self.eta_epsilon)
        )
        raw = np.divide(
            numerator,
            denominator,
            out=np.zeros(np.broadcast_shapes(numerator.shape, denominator.shape)),
            where=valid,
        )
        clipped = np.clip(raw, 0.0, 1.0)
        changed = valid & (np.abs(raw - clipped) > 1e-7)
        return clipped, valid, changed

    def direct_intention(self, task_latent) -> np.ndarray:
        reward = _finite_vector(
            task_latent, name="task_latent", length=self.latent_dim
        )
        if self.direct_latent_mode == "raw":
            return reward
        normalized = np.asarray(
            self.frozen_fb.normalize_latent(reward), dtype=np.float32
        )
        if normalized.shape != reward.shape or not np.all(np.isfinite(normalized)):
            raise ValueError("normalizing the task latent produced an invalid vector")
        return normalized

    def _prepare_goal(self, task_latent) -> None:
        reward = _finite_vector(
            task_latent, name="task_latent", length=self.latent_dim
        )
        key = np.ascontiguousarray(reward).tobytes()
        if self._goal_key == key:
            return
        policy = self.direct_intention(reward)
        repeated_policy = np.repeat(policy[None, :], self.candidate_count, axis=0)
        goal_forward = self._mean_forward(self.candidates, repeated_policy)
        self._goal_key = key
        self._goal_reward = reward
        self._goal_policy = policy
        self._self_goal = self.self_forward @ reward
        self._terminal_goal = goal_forward @ reward
        shape = (self.candidate_count, self.candidate_count)
        self._pair_goal = np.full(shape, np.nan, dtype=np.float64)
        self._pair_eta = np.full(shape, np.nan, dtype=np.float64)
        self._pair_valid = np.zeros(shape, dtype=bool)
        self._pair_clipped = np.zeros(shape, dtype=bool)
        self._row_cached = np.zeros(self.candidate_count, dtype=bool)

    def _ensure_pair_rows(self, rows: np.ndarray) -> None:
        missing = np.asarray(rows, dtype=np.int64)[~self._row_cached[rows]]
        if not len(missing):
            return
        total = len(missing) * self.candidate_count
        for start in range(0, total, self.pair_batch_size):
            stop = min(start + self.pair_batch_size, total)
            flat = np.arange(start, stop, dtype=np.int64)
            i = missing[flat // self.candidate_count]
            j = flat % self.candidate_count
            forward = self._mean_forward(
                self.candidates[i], self.candidate_latents[j]
            )
            pair_goal = forward @ self._goal_reward
            numerator = np.einsum(
                "ij,ij->i", forward, self.candidate_latents[j], optimize=True
            )
            eta, valid, clipped = self._safe_eta(numerator, self.self_measure[j])
            self._pair_goal[i, j] = pair_goal
            self._pair_eta[i, j] = eta
            self._pair_valid[i, j] = valid & np.isfinite(pair_goal)
            self._pair_clipped[i, j] = clipped
        self._row_cached[missing] = True

    def local_candidate_indices(self, observation):
        observation = _finite_vector(
            observation, name="observation", length=self.candidates.shape[1]
        )
        distance = np.linalg.norm(
            self.candidate_xy - observation[None, :2], axis=1
        )
        indexes = np.flatnonzero(distance <= self.local_radius)
        used_fallback = len(indexes) == 0
        if used_fallback:
            indexes = np.asarray([int(np.argmin(distance))], dtype=np.int64)
        ranking = np.argsort(distance[indexes], kind="stable")
        indexes = indexes[ranking[: self.max_local_candidates]]
        return indexes.astype(np.int64, copy=False), distance, used_fallback

    def _direct_value(self, observation, task_latent) -> float:
        policy = self.direct_intention(task_latent)
        forward = self._mean_forward(observation[None, :], policy[None, :])[0]
        return float(forward @ np.asarray(task_latent, dtype=np.float32))

    def _empty_diagnostics(self, *, goal_distance: float, direct_value: float):
        return {
            "h0lt_mode": np.int8(self.MODE_DIRECT),
            "h0lt_selected_depth": np.int8(0),
            "h0lt_terminal_active": False,
            "h0lt_goal_distance": np.float64(goal_distance),
            "h0lt_direct_value": np.float64(direct_value),
            "h0lt_best_two_value": np.float64(np.nan),
            "h0lt_advantage_over_direct": np.float64(np.nan),
            "h0lt_local_candidate_count": np.int64(0),
            "h0lt_global_candidate_count": np.int64(self.candidate_count),
            "h0lt_local_used_fallback": False,
            "h0lt_cached_pair_rows": np.int64(
                int(self._row_cached.sum()) if self._row_cached is not None else 0
            ),
            "h0lt_eta1_clipped_count": np.int64(0),
            "h0lt_eta2_clipped_count": np.int64(0),
            "h0lt_eta1_invalid_count": np.int64(0),
            "h0lt_eta2_invalid_count": np.int64(0),
            "h0lt_selected_w1_distance": np.float64(np.nan),
            "h0lt_selected_w1_xy": np.full(2, np.nan, dtype=np.float64),
            "h0lt_selected_w2_xy": np.full(2, np.nan, dtype=np.float64),
            "w1_index": np.int64(-1),
            "w2_index": np.int64(-1),
            "w1_source_index": np.int64(-1),
            "w2_source_index": np.int64(-1),
            "selected_eta1": np.float64(np.nan),
            "selected_eta2": np.float64(np.nan),
        }

    def select(
        self,
        observation,
        task_latent,
        *,
        force_terminal: bool = False,
        baseline_intention=None,
    ) -> LocalTerminalSelection:
        observation = _finite_vector(
            observation, name="observation", length=self.candidates.shape[1]
        )
        reward = _finite_vector(
            task_latent, name="task_latent", length=self.latent_dim
        )
        goal_distance = float(np.linalg.norm(observation[:2] - self.goal_xy))
        direct_value = self._direct_value(observation, reward)
        diagnostics = self._empty_diagnostics(
            goal_distance=goal_distance, direct_value=direct_value
        )
        terminal = bool(force_terminal) or (
            self.finish_radius > 0 and goal_distance <= self.finish_radius
        )
        if terminal:
            diagnostics["h0lt_terminal_active"] = True
            if self.finish_mode == "baseline":
                if baseline_intention is None:
                    raise ValueError("terminal baseline mode requires baseline_intention")
                intention = _finite_vector(
                    baseline_intention,
                    name="baseline_intention",
                    length=self.latent_dim,
                )
                diagnostics["h0lt_mode"] = np.int8(self.MODE_TERMINAL_BASELINE)
            else:
                intention = self.direct_intention(reward)
                diagnostics["h0lt_mode"] = np.int8(self.MODE_TERMINAL_DIRECT)
            return LocalTerminalSelection(intention=intention, diagnostics=diagnostics)

        self._prepare_goal(reward)
        root_indices, root_distance, used_fallback = self.local_candidate_indices(
            observation
        )
        self._ensure_pair_rows(root_indices)
        current_forward = self._mean_forward(
            np.repeat(observation[None, :], len(root_indices), axis=0),
            self.candidate_latents[root_indices],
        )
        eta_numerator = np.einsum(
            "ij,ij->i",
            current_forward,
            self.candidate_latents[root_indices],
            optimize=True,
        )
        eta1, eta1_valid, eta1_clipped = self._safe_eta(
            eta_numerator, self.self_measure[root_indices]
        )
        current_goal = current_forward @ reward
        pair_goal = self._pair_goal[root_indices, :]
        eta2 = self._pair_eta[root_indices, :]
        values = (
            current_goal[:, None]
            + eta1[:, None]
            * (pair_goal - self._self_goal[root_indices, None])
            + eta1[:, None]
            * eta2
            * (self._terminal_goal[None, :] - self._self_goal[None, :])
        )
        valid = (
            eta1_valid[:, None]
            & self._pair_valid[root_indices, :]
            & np.isfinite(values)
        )
        values = np.where(valid, values, -np.inf)
        diagnostics.update(
            {
                "h0lt_local_candidate_count": np.int64(len(root_indices)),
                "h0lt_local_used_fallback": bool(used_fallback),
                "h0lt_cached_pair_rows": np.int64(self._row_cached.sum()),
                "h0lt_eta1_clipped_count": np.int64(eta1_clipped.sum()),
                "h0lt_eta2_clipped_count": np.int64(
                    self._pair_clipped[root_indices, :].sum()
                ),
                "h0lt_eta1_invalid_count": np.int64((~eta1_valid).sum()),
                "h0lt_eta2_invalid_count": np.int64(
                    (~self._pair_valid[root_indices, :]).sum()
                ),
            }
        )
        if not np.any(np.isfinite(values)):
            if not self.enable_zero_level:
                raise RuntimeError("no finite local two-switch plan is available")
            return LocalTerminalSelection(
                intention=self.direct_intention(reward), diagnostics=diagnostics
            )

        local_i, j = np.unravel_index(int(np.argmax(values)), values.shape)
        i = int(root_indices[local_i])
        j = int(j)
        best_value = float(values[local_i, j])
        diagnostics["h0lt_best_two_value"] = np.float64(best_value)
        diagnostics["h0lt_advantage_over_direct"] = np.float64(
            best_value - direct_value
        )
        if self.enable_zero_level and best_value <= direct_value + self.switch_margin:
            return LocalTerminalSelection(
                intention=self.direct_intention(reward), diagnostics=diagnostics
            )

        diagnostics.update(
            {
                "h0lt_mode": np.int8(self.MODE_TWO_SWITCH),
                "h0lt_selected_depth": np.int8(2),
                "h0lt_selected_w1_distance": np.float64(root_distance[i]),
                "h0lt_selected_w1_xy": self.candidate_xy[i].copy(),
                "h0lt_selected_w2_xy": self.candidate_xy[j].copy(),
                "w1_index": np.int64(i),
                "w2_index": np.int64(j),
                "w1_source_index": np.int64(self.source_indices[i]),
                "w2_source_index": np.int64(self.source_indices[j]),
                "selected_eta1": np.float64(eta1[local_i]),
                "selected_eta2": np.float64(eta2[local_i, j]),
            }
        )
        return LocalTerminalSelection(
            intention=self.candidate_latents[i], diagnostics=diagnostics
        )

    def experiment_config(self) -> dict[str, Any]:
        return {
            "hypothesis": "h0_local_terminal",
            "candidate_source": "offline_train_dataset",
            "candidate_xy_source": self.xy_source,
            "source_candidate_count": self.source_candidate_count,
            "candidate_selection": "stratified_maze_cells_with_center_anchor",
            "grid_cell_size": self.grid_cell_size,
            "candidates_per_cell": self.candidates_per_cell,
            "occupied_cell_count": int(len(self.cell_coordinates)),
            "occupied_cells": self.cell_coordinates.tolist(),
            "selected_candidates_per_cell": self.per_cell_counts.tolist(),
            "candidate_count": self.candidate_count,
            "candidate_source_indices": self.source_indices.tolist(),
            "candidate_checksum_sha256": self.candidate_checksum,
            "root_candidate_selection": "nearest_candidates_inside_local_radius",
            "local_radius": self.local_radius,
            "max_local_candidates": self.max_local_candidates,
            "second_subgoal_candidates": "all_spatially_stratified_states",
            "pair_batch_size": self.pair_batch_size,
            "pair_cache": "lazy_goal_conditioned_root_rows",
            "eta_epsilon": self.eta_epsilon,
            "eta_range": [0.0, 1.0],
            "enable_zero_level": self.enable_zero_level,
            "direct_latent_mode": self.direct_latent_mode,
            "switch_margin": self.switch_margin,
            "finish_radius": self.finish_radius,
            "finish_mode": self.finish_mode,
            "goal_xy": self.goal_xy.tolist(),
            "mode_codes": {
                "direct_zero": self.MODE_DIRECT,
                "two_switch": self.MODE_TWO_SWITCH,
                "terminal_direct": self.MODE_TERMINAL_DIRECT,
                "terminal_baseline": self.MODE_TERMINAL_BASELINE,
            },
        }
