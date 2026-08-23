"""Синтетическое и реальное целевое состояние без дополнительного обучения."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from numbers import Integral
from typing import Any, Mapping

import jax.numpy as jnp
import numpy as np


@dataclass(frozen=True)
class GoalEurSelection:
    intention: Any
    diagnostics: Mapping[str, Any]


def _finite_vector(value, *, name: str) -> np.ndarray:
    vector = np.asarray(value)
    if vector.ndim != 1 or vector.size == 0 or vector.dtype.kind not in "iuf":
        raise ValueError(f"{name} must be a non-empty numeric vector")
    vector = vector.astype(np.float32, copy=False)
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must contain only finite values")
    return vector


def _goal_xy(value) -> np.ndarray:
    goal = np.asarray(value, dtype=np.float64)
    if goal.shape != (2,) or not np.all(np.isfinite(goal)):
        raise ValueError("goal_xy must contain exactly two finite values")
    return goal


def _positive_int(value, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


class SyntheticCurrentGoalPlanner:
    """Подставляет координаты цели в текущее полное состояние робота."""

    method_name = "h_goal_eur_synthetic_current"

    def __init__(self, frozen_fb, goal_xy):
        self.frozen_fb = frozen_fb
        self.goal_xy = _goal_xy(goal_xy)

    def experiment_config(self) -> dict[str, Any]:
        return {
            "hypothesis": "H_goal_eur",
            "variant": "synthetic_current",
            "goal_xy": self.goal_xy.tolist(),
            "target_construction": "[goal_xy, current_observation[2:]]",
            "candidate_source": "synthetic_state",
        }

    def select(self, observation, task_latent) -> GoalEurSelection:
        del task_latent
        observation = _finite_vector(observation, name="observation")
        if observation.size < 3:
            raise ValueError("observation must contain XY and non-XY state components")

        target = observation.copy()
        target[:2] = self.goal_xy
        raw_latent = self.frozen_fb.backward_repr(jnp.asarray(target))
        intention = self.frozen_fb.normalize_latent(raw_latent)
        intention_np = np.asarray(intention)
        if intention_np.ndim != 1 or not np.all(np.isfinite(intention_np)):
            raise ValueError("synthetic goal produced an invalid intention")

        return GoalEurSelection(
            intention=intention,
            diagnostics={
                "hge_variant": np.int8(0),
                "hge_synthetic_target": target,
            },
        )


class DatasetMaxValueGoalPlanner:
    """Выбирает реальное состояние возле цели по осторожной оценке FB-ансамбля."""

    method_name = "h_goal_eur_dataset_max_v"

    def __init__(
        self,
        frozen_fb,
        offline_dataset,
        goal_xy,
        *,
        candidate_radius: float = 0.5,
        max_candidates: int = 64,
        disagreement_penalty: float = 0.5,
    ):
        self.frozen_fb = frozen_fb
        self.goal_xy = _goal_xy(goal_xy)
        self.max_candidates = _positive_int(max_candidates, name="max_candidates")

        self.candidate_radius = float(candidate_radius)
        if not np.isfinite(self.candidate_radius) or self.candidate_radius <= 0.0:
            raise ValueError("candidate_radius must be a positive finite number")
        self.disagreement_penalty = float(disagreement_penalty)
        if (
            not np.isfinite(self.disagreement_penalty)
            or self.disagreement_penalty < 0.0
        ):
            raise ValueError("disagreement_penalty must be finite and non-negative")

        observations = np.asarray(offline_dataset["observations"])
        qpos = np.asarray(offline_dataset["qpos"])
        if observations.ndim != 2 or observations.shape[0] == 0:
            raise ValueError("offline observations must be a non-empty 2D array")
        if observations.dtype.kind not in "iuf" or not np.all(np.isfinite(observations)):
            raise ValueError("offline observations must contain only finite numbers")
        if (
            qpos.ndim != 2
            or qpos.shape[0] != observations.shape[0]
            or qpos.shape[1] < 2
            or not np.all(np.isfinite(qpos[:, :2]))
        ):
            raise ValueError("offline qpos must be aligned with observations and contain finite XY")

        distance = np.linalg.norm(qpos[:, :2] - self.goal_xy[None], axis=-1)
        matching = np.flatnonzero(distance <= self.candidate_radius)
        if matching.size == 0:
            raise ValueError(
                "No train-dataset states lie inside the requested goal region: "
                f"goal_xy={self.goal_xy.tolist()}, radius={self.candidate_radius}"
            )

        if matching.size > self.max_candidates:
            positions = np.linspace(
                0,
                matching.size - 1,
                self.max_candidates,
                dtype=np.int64,
            )
            source_indices = matching[positions]
            selection_strategy = "deterministic_linspace_over_goal_matches"
        else:
            source_indices = matching
            selection_strategy = "all_goal_matches"

        selected = np.ascontiguousarray(
            observations[source_indices].astype(np.float32, copy=False)
        )
        selected_xy = np.ascontiguousarray(
            qpos[source_indices, :2].astype(np.float64, copy=False)
        )
        self.source_observation_count = int(observations.shape[0])
        self.goal_match_count = int(matching.size)
        self.source_indices = np.asarray(source_indices, dtype=np.int64)
        self.selection_strategy = selection_strategy
        self.candidates = jnp.asarray(selected)
        self.candidate_xy = selected_xy
        self.candidate_count = int(selected.shape[0])

        digest = hashlib.sha256()
        digest.update(selected.tobytes())
        digest.update(self.source_indices.tobytes())
        self.candidate_checksum = digest.hexdigest()

        raw_latents = self.frozen_fb.backward_repr(self.candidates)
        candidate_latents = self.frozen_fb.normalize_latent(raw_latents)
        candidate_latents_np = np.asarray(candidate_latents)
        if (
            candidate_latents_np.ndim != 2
            or candidate_latents_np.shape[0] != self.candidate_count
            or not np.all(np.isfinite(candidate_latents_np))
        ):
            raise ValueError("backward_repr must produce one finite latent per candidate")
        self.candidate_latents = jnp.asarray(candidate_latents)

    def experiment_config(self) -> dict[str, Any]:
        return {
            "hypothesis": "H_goal_eur",
            "variant": "dataset_max_v",
            "goal_xy": self.goal_xy.tolist(),
            "candidate_source": "train_dataset_goal_region",
            "source_observation_count": self.source_observation_count,
            "goal_match_count": self.goal_match_count,
            "candidate_radius": self.candidate_radius,
            "max_candidates": self.max_candidates,
            "candidate_count": self.candidate_count,
            "candidate_selection": self.selection_strategy,
            "candidate_source_indices": self.source_indices.tolist(),
            "candidate_checksum_sha256": self.candidate_checksum,
            "disagreement_penalty": self.disagreement_penalty,
            "score": "ensemble_mean-downstream_penalty*ensemble_range",
            "execute": "normalize(B(selected_dataset_state))",
        }

    def _validate_query(self, observation, task_latent):
        observation = _finite_vector(observation, name="observation")
        task_latent = _finite_vector(task_latent, name="task_latent")
        if observation.shape[0] != self.candidates.shape[1]:
            raise ValueError("observation dimension does not match candidate states")
        if task_latent.shape[0] != self.candidate_latents.shape[1]:
            raise ValueError("task_latent dimension does not match candidate intentions")
        return jnp.asarray(observation), jnp.asarray(task_latent)

    # Штраф за размах ансамбля снижает оценку кандидатов с неуверенным прогнозом.
    def score_candidates(self, observation, task_latent):
        """Вычисляет устойчивые оценки и разброс ансамбля для всех кандидатов."""

        observation, reward_latent = self._validate_query(observation, task_latent)
        observations = jnp.repeat(
            observation[None],
            self.candidate_count,
            axis=0,
        )
        forward = jnp.asarray(
            self.frozen_fb.forward_repr(observations, self.candidate_latents)
        )
        if (
            forward.ndim != 3
            or forward.shape[1] != self.candidate_count
            or forward.shape[2] != reward_latent.shape[0]
        ):
            raise ValueError(
                "forward_repr must have shape (ensemble, candidates, latent_dim), "
                f"got {forward.shape}"
            )

        values = jnp.sum(forward * reward_latent[None, None, :], axis=-1)
        mean_value = jnp.mean(values, axis=0)
        ensemble_range = jnp.max(values, axis=0) - jnp.min(values, axis=0)
        score = mean_value - self.disagreement_penalty * ensemble_range
        score = jnp.where(jnp.isfinite(score), score, -jnp.inf)
        return score, mean_value, ensemble_range

    def select(self, observation, task_latent) -> GoalEurSelection:
        score, mean_value, ensemble_range = self.score_candidates(
            observation,
            task_latent,
        )
        score_np = np.asarray(score)
        if not np.any(np.isfinite(score_np)):
            raise RuntimeError("H_goal_eur has no candidate with a finite value score")
        index = int(np.argmax(score_np))

        return GoalEurSelection(
            intention=self.candidate_latents[index],
            diagnostics={
                "hge_variant": np.int8(1),
                "hge_candidate_index": np.int64(index),
                "hge_dataset_index": np.int64(self.source_indices[index]),
                "hge_candidate_xy": self.candidate_xy[index],
                "hge_score": np.float64(score_np[index]),
                "hge_mean_value": np.float64(np.asarray(mean_value)[index]),
                "hge_ensemble_range": np.float64(
                    np.asarray(ensemble_range)[index]
                ),
            },
        )
