"""Выбор реального состояния рядом с декодированной подцелью."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from typing import Any, Mapping

import jax.numpy as jnp
import numpy as np


@dataclass(frozen=True)
class ProjectedSubgoalSelection:
    intention: Any
    diagnostics: Mapping[str, Any]


def _finite_vector(value, *, name: str) -> np.ndarray:
    value = np.asarray(value, dtype=np.float32)
    if value.ndim != 1 or value.size == 0:
        raise ValueError(f"{name} must be a non-empty vector, got {value.shape}")
    if not np.all(np.isfinite(value)):
        raise ValueError(f"{name} contains non-finite values")
    return value


class _XYBucketIndex:
    """Ускоряет повторный поиск офлайн-состояний в заданном пространственном радиусе."""

    def __init__(self, xy: np.ndarray, bucket_size: float):
        self.xy = xy
        self.bucket_size = float(bucket_size)
        keys = np.floor(xy / self.bucket_size).astype(np.int64)
        buckets: dict[tuple[int, int], list[int]] = {}
        for index, key in enumerate(keys):
            buckets.setdefault((int(key[0]), int(key[1])), []).append(index)
        self.buckets = {
            key: np.asarray(indices, dtype=np.int64)
            for key, indices in buckets.items()
        }

    def query_radius(self, center: np.ndarray, radius: float) -> np.ndarray:
        center_key = np.floor(center / self.bucket_size).astype(np.int64)
        extent = int(np.ceil(radius / self.bucket_size))
        parts = []
        for dx in range(-extent, extent + 1):
            for dy in range(-extent, extent + 1):
                part = self.buckets.get(
                    (int(center_key[0] + dx), int(center_key[1] + dy))
                )
                if part is not None:
                    parts.append(part)
        if not parts:
            return np.empty(0, dtype=np.int64)
        indices = np.concatenate(parts)
        distance = np.linalg.norm(self.xy[indices] - center[None, :], axis=1)
        return np.sort(indices[distance <= radius])


class DecodedDatasetSubgoalPlanner:
    """Привязывает декодированную подцель к реальному офлайн-состоянию."""

    method_name = "decoded_dataset_subgoal_vmax_finish"

    def __init__(
        self,
        frozen_fb,
        decoder,
        offline_dataset,
        *,
        goal_xy,
        candidate_radius: float = 0.5,
        max_candidates: int = 64,
        disagreement_penalty: float = 0.5,
        selection_mode: str = "max-v",
    ):
        if not np.isfinite(candidate_radius) or candidate_radius <= 0:
            raise ValueError("candidate_radius must be positive and finite")
        if (
            isinstance(max_candidates, bool)
            or not isinstance(max_candidates, Integral)
            or max_candidates <= 0
        ):
            raise ValueError("max_candidates must be a positive integer")
        if not np.isfinite(disagreement_penalty) or disagreement_penalty < 0:
            raise ValueError("disagreement_penalty must be finite and non-negative")
        if selection_mode not in {"max-v", "nearest-xy"}:
            raise ValueError("selection_mode must be 'max-v' or 'nearest-xy'")

        observations = np.asarray(offline_dataset["observations"], dtype=np.float32)
        if observations.ndim != 2 or observations.shape[0] == 0 or observations.shape[1] < 2:
            raise ValueError(
                "offline observations must have shape (N, observation_dim>=2)"
            )
        if not np.all(np.isfinite(observations)):
            raise ValueError("offline observations contain non-finite values")
        if int(decoder.latent_dim) != int(frozen_fb.latent_dim):
            raise ValueError(
                f"decoder latent_dim={decoder.latent_dim} does not match "
                f"checkpoint latent_dim={frozen_fb.latent_dim}"
            )

        self.frozen_fb = frozen_fb
        self.decoder = decoder
        self.observations = observations
        self.xy = observations[:, :2].astype(np.float64)
        self.candidate_radius = float(candidate_radius)
        self.max_candidates = int(max_candidates)
        self.disagreement_penalty = float(disagreement_penalty)
        self.selection_mode = selection_mode
        self.spatial_index = _XYBucketIndex(self.xy, self.candidate_radius)
        self.goal_xy = _finite_vector(goal_xy, name="goal_xy").astype(np.float64)
        if self.goal_xy.shape != (2,):
            raise ValueError(f"goal_xy must have shape (2,), got {self.goal_xy.shape}")
        self.goal_indices, self.goal_used_fallback = self._candidate_indices(
            self.goal_xy
        )
        self.goal_candidate_intentions = self._candidate_intentions(
            self.goal_indices
        )

    def experiment_config(self) -> dict[str, Any]:
        return {
            "hypothesis": "decoded_dataset_subgoal",
            "selection_mode": self.selection_mode,
            "candidate_source": "offline_train_dataset",
            "candidate_radius": self.candidate_radius,
            "max_candidates": self.max_candidates,
            "candidate_subsample": "linspace_source_order",
            "disagreement_penalty": self.disagreement_penalty,
            "empty_radius_fallback": "global_nearest_xy",
            "finish_selection": "v-max_each_hierarchical_replan",
            "finish_goal_xy": self.goal_xy.tolist(),
            "finish_candidate_count": int(len(self.goal_indices)),
            "finish_reward_latent": "original_xy_task_latent",
            "finish_policy_latent": "normalize(B(selected_goal_dataset_state))",
            "decoder_architecture": list(self.decoder.architecture),
            "execute": "normalize(B(selected_dataset_state))",
        }

    def _candidate_indices(self, decoded_xy: np.ndarray):
        indices = self.spatial_index.query_radius(decoded_xy, self.candidate_radius)
        used_fallback = len(indices) == 0
        if used_fallback:
            distance = np.linalg.norm(self.xy - decoded_xy[None, :], axis=1)
            indices = np.asarray([int(np.argmin(distance))], dtype=np.int64)
        elif len(indices) > self.max_candidates:
            positions = np.linspace(
                0,
                len(indices) - 1,
                self.max_candidates,
                dtype=np.int64,
            )
            indices = indices[positions]
        return indices, used_fallback

    def _candidate_intentions(self, indices: np.ndarray):
        raw = self.frozen_fb.backward_repr(jnp.asarray(self.observations[indices]))
        intentions = self.frozen_fb.normalize_latent(raw)
        values = np.asarray(intentions)
        if values.shape != (len(indices), self.frozen_fb.latent_dim):
            raise RuntimeError(f"unexpected candidate intention shape: {values.shape}")
        if not np.all(np.isfinite(values)):
            raise RuntimeError("candidate intentions contain non-finite values")
        return intentions

    def _max_v_scores(self, observation, task_latent, candidate_intentions):
        observation = _finite_vector(observation, name="observation")
        reward_latent = _finite_vector(task_latent, name="task_latent")
        if reward_latent.size != self.frozen_fb.latent_dim:
            raise ValueError("task_latent dimension does not match the checkpoint")
        count = int(candidate_intentions.shape[0])
        repeated_observations = jnp.repeat(
            jnp.asarray(observation)[None, :],
            count,
            axis=0,
        )
        forward = self.frozen_fb.forward_repr(
            repeated_observations,
            candidate_intentions,
        )
        forward = jnp.asarray(forward)
        if forward.ndim == 2:
            forward = forward[None, ...]
        expected_tail = (count, self.frozen_fb.latent_dim)
        if forward.ndim != 3 or tuple(forward.shape[-2:]) != expected_tail:
            raise RuntimeError(f"unexpected forward representation shape: {forward.shape}")
        values = jnp.sum(forward * jnp.asarray(reward_latent)[None, None, :], axis=-1)
        mean_value = jnp.mean(values, axis=0)
        ensemble_range = jnp.max(values, axis=0) - jnp.min(values, axis=0)
        score = mean_value - self.disagreement_penalty * ensemble_range
        score = jnp.where(jnp.isfinite(score), score, -jnp.inf)
        return np.asarray(score), np.asarray(mean_value), np.asarray(ensemble_range)

    def select_finish(self, observation, task_latent) -> ProjectedSubgoalSelection:
        """Выбирает реальное состояние около цели по осторожной FB-оценке."""

        score, mean_value, ensemble_range = self._max_v_scores(
            observation,
            task_latent,
            self.goal_candidate_intentions,
        )
        if not np.any(np.isfinite(score)):
            raise RuntimeError("no final-goal candidate has a finite value score")
        local_index = int(np.argmax(score))
        dataset_index = int(self.goal_indices[local_index])
        return ProjectedSubgoalSelection(
            intention=self.goal_candidate_intentions[local_index],
            diagnostics={
                "finish_dataset_xy": self.xy[dataset_index],
                "finish_dataset_index": np.int64(dataset_index),
                "finish_candidate_count": np.int64(len(self.goal_indices)),
                "finish_used_nearest_fallback": bool(self.goal_used_fallback),
                "finish_score": np.float64(score[local_index]),
                "finish_mean_value": np.float64(mean_value[local_index]),
                "finish_ensemble_range": np.float64(
                    ensemble_range[local_index]
                ),
            },
        )

    def select(self, observation, task_latent, high_intention) -> ProjectedSubgoalSelection:
        high_intention = _finite_vector(high_intention, name="high_intention")
        if high_intention.size != self.frozen_fb.latent_dim:
            raise ValueError("high_intention dimension does not match the checkpoint")
        decoded_xy = np.asarray(self.decoder.predict(high_intention), dtype=np.float64)
        if decoded_xy.shape != (2,) or not np.all(np.isfinite(decoded_xy)):
            raise RuntimeError(f"decoder returned invalid XY: {decoded_xy}")

        indices, used_fallback = self._candidate_indices(decoded_xy)
        candidate_intentions = self._candidate_intentions(indices)
        candidate_distance = np.linalg.norm(self.xy[indices] - decoded_xy[None, :], axis=1)

        # Геометрически ближайший вариант нужен как контроль без использования FB-ценности.
        if self.selection_mode == "nearest-xy":
            local_index = int(np.argmin(candidate_distance))
            score = np.full(len(indices), np.nan, dtype=np.float64)
            mean_value = score.copy()
            ensemble_range = score.copy()
        else:
            score, mean_value, ensemble_range = self._max_v_scores(
                observation,
                task_latent,
                candidate_intentions,
            )
            if not np.any(np.isfinite(score)):
                raise RuntimeError("no decoded-subgoal candidate has a finite value score")
            local_index = int(np.argmax(score))

        # Итоговое намерение всегда строится из реально наблюдавшегося офлайн-состояния.
        dataset_index = int(indices[local_index])
        selected_score = float(score[local_index]) if np.isfinite(score[local_index]) else np.nan
        selected_mean = (
            float(mean_value[local_index]) if np.isfinite(mean_value[local_index]) else np.nan
        )
        selected_range = (
            float(ensemble_range[local_index])
            if np.isfinite(ensemble_range[local_index])
            else np.nan
        )
        return ProjectedSubgoalSelection(
            intention=candidate_intentions[local_index],
            diagnostics={
                "decoded_subgoal_xy": decoded_xy,
                "projected_dataset_xy": self.xy[dataset_index],
                "projected_dataset_index": np.int64(dataset_index),
                "projection_xy_error": np.float64(candidate_distance[local_index]),
                "projection_candidate_count": np.int64(len(indices)),
                "projection_used_nearest_fallback": bool(used_fallback),
                "projection_score": np.float64(selected_score),
                "projection_mean_value": np.float64(selected_mean),
                "projection_ensemble_range": np.float64(selected_range),
            },
        )