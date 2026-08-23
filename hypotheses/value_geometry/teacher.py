"""Получение обучающих оценок от замороженного FB-критика малыми блоками."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping

import numpy as np

from .data import PairSplit, StatePool


Progress = Callable[[str], None]


def aggregate_ensemble(
    forward: np.ndarray,
    reward_latents: np.ndarray,
    *,
    disagreement_penalty: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Воспроизводит оценку ансамбля со штрафом за разброс предсказаний."""

    forward = np.asarray(forward, dtype=np.float32)
    reward_latents = np.asarray(reward_latents, dtype=np.float32)
    if forward.ndim == 2:
        forward = forward[None, :, :]
    if forward.ndim != 3 or reward_latents.ndim != 2:
        raise ValueError("forward must be [E,N,D] and reward_latents must be [N,D]")
    if tuple(forward.shape[1:]) != tuple(reward_latents.shape):
        raise ValueError("forward and reward_latents have incompatible dimensions")
    if not np.isfinite(disagreement_penalty) or disagreement_penalty < 0:
        raise ValueError("disagreement_penalty must be non-negative and finite")
    member_values = np.sum(
        forward * reward_latents[None, :, :], axis=-1, dtype=np.float32
    )
    mean = member_values.mean(axis=0)
    spread = member_values.max(axis=0) - member_values.min(axis=0)
    score = mean - np.float32(disagreement_penalty) * spread
    return score.astype(np.float32), member_values.T.astype(np.float32)


def _is_memory_failure(exc: BaseException) -> bool:
    message = str(exc).lower()
    return any(
        phrase in message
        for phrase in (
            "out of memory",
            "resource exhausted",
            "resource_exhausted",
            "failed to allocate",
            "cuda_error_out_of_memory",
            "std::bad_alloc",
        )
    )


def _batched_backward(
    frozen_fb: Any,
    observations: np.ndarray,
    *,
    batch_size: int,
    progress: Progress | None = None,
    label: str = "B",
) -> np.ndarray:
    observations = np.asarray(observations, dtype=np.float32)
    result = np.empty((len(observations), int(frozen_fb.latent_dim)), dtype=np.float32)
    current_batch = max(1, int(batch_size))
    cursor = 0
    previous_percent = -1
    while cursor < len(observations):
        stop = min(len(observations), cursor + current_batch)
        try:
            encoded = np.asarray(
                frozen_fb.backward_repr(observations[cursor:stop]), dtype=np.float32
            )
        except Exception as exc:
            if not _is_memory_failure(exc) or current_batch == 1:
                raise
            current_batch = max(1, current_batch // 2)
            if progress:
                progress(f"{label}: reducing batch size to {current_batch} after memory failure")
            continue
        expected = (stop - cursor, int(frozen_fb.latent_dim))
        if encoded.shape != expected:
            raise RuntimeError(f"backward_repr returned {encoded.shape}, expected {expected}")
        result[cursor:stop] = encoded
        cursor = stop
        percent = int(100 * cursor / max(1, len(observations)))
        if progress and (percent == 100 or percent // 20 > previous_percent // 20):
            progress(f"{label}: {cursor}/{len(observations)} states ({percent}%)")
            previous_percent = percent
    if not np.all(np.isfinite(result)):
        raise RuntimeError("backward representations contain non-finite values")
    return result


def exact_binary_reward_raw_latent(
    backward_sum: np.ndarray,
    *,
    total_samples: int,
    positive_samples: int,
    reward_temperature: float,
) -> np.ndarray:
    """Воспроизводит исходное представление задачи для двоичной награды."""

    if total_samples < 1 or not 0 < positive_samples <= total_samples:
        raise ValueError("invalid total_samples/positive_samples")
    negative_samples = total_samples - positive_samples
    positive_logmass = np.log(float(positive_samples)) + reward_temperature
    if negative_samples:
        denominator_log = np.logaddexp(
            np.log(float(negative_samples)), positive_logmass
        )
    else:
        denominator_log = positive_logmass
    positive_weight = np.exp(reward_temperature - denominator_log)
    coefficient = positive_weight / float(total_samples)
    return (
        np.asarray(backward_sum, dtype=np.float64) * coefficient
    ).astype(np.float32)


def _radius_supports(
    positions: np.ndarray, goals: np.ndarray, radius: float
) -> list[np.ndarray]:
    try:
        from scipy.spatial import cKDTree

        tree = cKDTree(np.asarray(positions, dtype=np.float64))
        return [
            np.asarray(sorted(indices), dtype=np.int64)
            for indices in tree.query_ball_point(
                np.asarray(goals, dtype=np.float64), r=float(radius)
            )
        ]
    except ImportError:
        result = []
        squared_radius = radius * radius
        for goal in goals:
            squared = np.sum((positions - goal[None, :]) ** 2, axis=1)
            result.append(np.flatnonzero(squared <= squared_radius).astype(np.int64))
        return result


@dataclass(frozen=True)
class GoalBank:
    goal_indices: np.ndarray
    policy_intentions: np.ndarray
    reward_latents: np.ndarray
    support_sizes: np.ndarray
    reward_positions: np.ndarray
    target_mode: str


class OfflineFBTeacher:
    """Строит целевые намерения и размечает пары замороженным FB-критиком."""

    def __init__(
        self,
        frozen_fb: Any,
        pool: StatePool,
        reference_observations: np.ndarray,
        reference_positions: np.ndarray,
        *,
        goal_tolerance: float,
        target_mode: str = "xy-goal",
        fixed_goal_xy: np.ndarray | None = None,
        disagreement_penalty: float = 0.5,
        batch_size: int = 128,
        reference_samples: int = 100_000,
        progress: Progress | None = None,
    ):
        if target_mode not in {"xy-goal", "state-goal", "fixed-task"}:
            raise ValueError("unknown target_mode")
        if not np.isfinite(goal_tolerance) or goal_tolerance <= 0:
            raise ValueError("goal_tolerance must be positive and finite")
        if batch_size <= 0 or reference_samples <= 0:
            raise ValueError("batch_size and reference_samples must be positive")
        reference_observations = np.asarray(reference_observations, dtype=np.float32)
        reference_positions = np.asarray(reference_positions, dtype=np.float32)
        if len(reference_observations) != len(reference_positions):
            raise ValueError("reference observations and positions are not aligned")
        if reference_positions.ndim != 2 or reference_positions.shape[1] != 2:
            raise ValueError("reference_positions must have shape [N, 2]")

        count = min(int(reference_samples), len(reference_observations))
        if count < 1:
            raise ValueError("reference dataset is empty")
        self.frozen_fb = frozen_fb
        self.pool = pool
        self.reference_observations = reference_observations[:count]
        self.reference_positions = reference_positions[:count]
        self.goal_tolerance = float(goal_tolerance)
        self.target_mode = target_mode
        self.fixed_goal_xy = (
            None
            if fixed_goal_xy is None
            else np.asarray(fixed_goal_xy, dtype=np.float32).reshape(2)
        )
        if target_mode == "fixed-task" and self.fixed_goal_xy is None:
            raise ValueError("fixed-task mode requires fixed_goal_xy")
        self.disagreement_penalty = float(disagreement_penalty)
        self.batch_size = int(batch_size)
        self.progress = progress
        config = frozen_fb.config
        self.reward_temperature = float(config.get("reward_temperature", 0.0))
        self.normalize_reward_latent = bool(config.get("normalize_latent", True))

    def prepare_goal_banks(
        self, split_goals: Mapping[str, np.ndarray]
    ) -> dict[str, GoalBank]:
        all_goals = np.unique(
            np.concatenate([np.asarray(goals, dtype=np.int64) for goals in split_goals.values()])
        )
        if len(all_goals) == 0:
            raise ValueError("all goal banks are empty")
        goal_observations = self.pool.observations[all_goals]
        raw_goal_backward = _batched_backward(
            self.frozen_fb,
            goal_observations,
            batch_size=self.batch_size,
            progress=self.progress,
            label="goal B(s)",
        )
        intentions = np.asarray(
            self.frozen_fb.normalize_latent(raw_goal_backward), dtype=np.float32
        )

        if self.target_mode == "state-goal":
            reward_latents = intentions.copy()
            support_sizes = np.ones(len(all_goals), dtype=np.int64)
            reward_positions = self.pool.positions[all_goals].copy()
        else:
            reward_positions = (
                np.repeat(self.fixed_goal_xy[None, :], len(all_goals), axis=0)
                if self.target_mode == "fixed-task"
                else self.pool.positions[all_goals].copy()
            )
            reward_latents, support_sizes = self._reward_latents(reward_positions)

        supported = support_sizes > 0
        if self.progress and not np.all(supported):
            self.progress(
                f"dropping {int((~supported).sum())} goals without offline reward support"
            )
        lookup = {int(goal): index for index, goal in enumerate(all_goals)}
        result: dict[str, GoalBank] = {}
        for split_name, goals in split_goals.items():
            locations = np.asarray([lookup[int(goal)] for goal in goals], dtype=np.int64)
            locations = locations[supported[locations]]
            if len(locations) < 2:
                raise RuntimeError(
                    f"split {split_name!r} has fewer than two supported goals; "
                    "increase --reference-samples, --goal-count or use --target-mode state-goal"
                )
            result[split_name] = GoalBank(
                goal_indices=all_goals[locations],
                policy_intentions=intentions[locations],
                reward_latents=reward_latents[locations],
                support_sizes=support_sizes[locations],
                reward_positions=reward_positions[locations],
                target_mode=self.target_mode,
            )
        return result

    def _reward_latents(self, reward_positions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.target_mode == "fixed-task":
            unique_positions = reward_positions[:1]
            inverse = np.zeros(len(reward_positions), dtype=np.int64)
        else:
            unique_positions, inverse = np.unique(
                np.asarray(reward_positions, dtype=np.float32), axis=0, return_inverse=True
            )

        supports = _radius_supports(
            self.reference_positions, unique_positions, self.goal_tolerance
        )
        supported_arrays = [indices for indices in supports if len(indices)]
        unique_reference = (
            np.unique(np.concatenate(supported_arrays))
            if supported_arrays
            else np.empty(0, dtype=np.int64)
        )
        if self.progress:
            self.progress(
                f"reward support: {len(unique_reference)} unique reference states "
                f"for {len(unique_positions)} goal positions "
                f"from {len(self.reference_observations)} official-order samples"
            )

        reference_backward = _batched_backward(
            self.frozen_fb,
            self.reference_observations[unique_reference],
            batch_size=self.batch_size,
            progress=self.progress,
            label="reward-support B(s)",
        )
        inverse_reference = np.full(
            len(self.reference_observations), -1, dtype=np.int64
        )
        inverse_reference[unique_reference] = np.arange(len(unique_reference))
        raw = np.zeros((len(unique_positions), self.frozen_fb.latent_dim), dtype=np.float32)
        support_sizes = np.asarray([len(indices) for indices in supports], dtype=np.int64)
        for index, support in enumerate(supports):
            if not len(support):
                continue
            selected = reference_backward[inverse_reference[support]]
            backward_sum = selected.sum(axis=0, dtype=np.float64)
            raw[index] = exact_binary_reward_raw_latent(
                backward_sum,
                total_samples=len(self.reference_observations),
                positive_samples=len(support),
                reward_temperature=self.reward_temperature,
            )
        if self.normalize_reward_latent:
            normalized = np.asarray(
                self.frozen_fb.normalize_latent(raw), dtype=np.float32
            )
        else:
            normalized = raw
        return normalized[inverse], support_sizes[inverse]

    def score_pairs(self, pairs: PairSplit, bank: GoalBank) -> PairSplit:
        lookup = np.full(len(self.pool.observations), -1, dtype=np.int64)
        lookup[bank.goal_indices] = np.arange(len(bank.goal_indices))
        positions = lookup[pairs.goal_indices]
        if np.any(positions < 0):
            raise ValueError("pair split references a goal absent from its goal bank")
        scores = np.empty(len(pairs), dtype=np.float32)
        ensemble = None
        cursor = 0
        batch_size = self.batch_size
        previous_percent = -1
        while cursor < len(pairs):
            stop = min(len(pairs), cursor + batch_size)
            goal_rows = positions[cursor:stop]
            observations = self.pool.observations[pairs.start_indices[cursor:stop]]
            intentions = bank.policy_intentions[goal_rows]
            rewards = bank.reward_latents[goal_rows]
            try:
                forward = self.frozen_fb.forward_repr(observations, intentions)
                batch_scores, members = aggregate_ensemble(
                    forward,
                    rewards,
                    disagreement_penalty=self.disagreement_penalty,
                )
            except Exception as exc:
                if not _is_memory_failure(exc) or batch_size == 1:
                    raise
                batch_size = max(1, batch_size // 2)
                if self.progress:
                    self.progress(f"F teacher: reducing batch size to {batch_size}")
                continue
            if ensemble is None:
                ensemble = np.empty((len(pairs), members.shape[1]), dtype=np.float32)
            scores[cursor:stop] = batch_scores
            ensemble[cursor:stop] = members
            cursor = stop
            percent = int(100 * cursor / len(pairs))
            if self.progress and (percent == 100 or percent // 10 > previous_percent // 10):
                self.progress(f"F teacher: {cursor}/{len(pairs)} pairs ({percent}%)")
                previous_percent = percent
        if not np.all(np.isfinite(scores)):
            raise RuntimeError("FB teacher returned non-finite value labels")
        return pairs.with_values(scores, ensemble)

    def description(self) -> dict[str, Any]:
        return {
            "target_mode": self.target_mode,
            "formula": (
                "mean_e <F_e(s, normalize(B(g))), z_reward> "
                "- disagreement_penalty * range_e"
            ),
            "disagreement_penalty": self.disagreement_penalty,
            "default_two_member_penalty_is_exact_minimum": bool(
                np.isclose(self.disagreement_penalty, 0.5)
            ),
            "goal_tolerance": self.goal_tolerance,
            "reference_samples": len(self.reference_observations),
            "reward_temperature": self.reward_temperature,
            "normalize_reward_latent": self.normalize_reward_latent,
            "teacher_batch_size": self.batch_size,
        }
