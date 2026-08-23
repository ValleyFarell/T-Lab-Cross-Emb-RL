"""Динамическое исполнение первого намерения трёхточечного маршрута."""

from __future__ import annotations

from numbers import Integral

import numpy as np

from .base import HighLevelController, IntentionSelection


class DynamicLatentThreeController(HighLevelController):
    """Перестраивает маршрут из трёх точек и исполняет только первое намерение."""

    method_name = "latent_three_dynamic"

    def __init__(
        self,
        frozen_fb,
        planner,
        *,
        replan_interval: int = 10,
        finish_radius: float = 0.75,
        finish_mode: str = "baseline",
        fallback_mode: str = "baseline",
        replan_on_arrival: bool = True,
        waypoint_replan_radius: float = 0.15,
    ):
        if isinstance(replan_interval, bool) or not isinstance(replan_interval, Integral) or replan_interval <= 0:
            raise ValueError("replan_interval must be a positive integer")
        if not np.isfinite(finish_radius) or finish_radius < 0:
            raise ValueError("finish_radius must be finite and non-negative")
        if not np.isfinite(waypoint_replan_radius) or waypoint_replan_radius < 0:
            raise ValueError("waypoint_replan_radius must be finite and non-negative")
        if finish_mode not in {"baseline", "task-latent"}:
            raise ValueError("finish_mode must be 'baseline' or 'task-latent'")
        if fallback_mode not in {"baseline", "task-latent"}:
            raise ValueError("fallback_mode must be 'baseline' or 'task-latent'")
        self.frozen_fb = frozen_fb
        self.planner = planner
        self.replan_interval = int(replan_interval)
        self.finish_radius = float(finish_radius)
        self.finish_mode = finish_mode
        self.fallback_mode = fallback_mode
        self.replan_on_arrival = bool(replan_on_arrival)
        self.waypoint_replan_radius = float(waypoint_replan_radius)
        self.reset()

    def reset(self, scenario=None) -> None:
        del scenario
        self._step = 0
        self._last_plan_step = -1
        self._replan_count = 0
        self._cached = None

    def experiment_config(self):
        configuration = dict(self.planner.experiment_config())
        configuration.update(
            {
                "replan_interval": self.replan_interval,
                "finish_radius": self.finish_radius,
                "finish_mode": self.finish_mode,
                "fallback_mode": self.fallback_mode,
                "replan_on_arrival": self.replan_on_arrival,
                "waypoint_replan_radius": self.waypoint_replan_radius,
                "goal_selection": "original_task_latent_fixed_for_episode",
            }
        )
        return configuration

    @staticmethod
    def _empty_planner_diagnostics():
        return {
            "latent3_route_indices": np.full(3, -1, dtype=np.int64),
            "latent3_source_indices": np.full(3, -1, dtype=np.int64),
            "latent3_waypoints": np.full((3, 4), np.nan, dtype=np.float32),
            "latent3_waypoint_xy": np.full((3, 2), np.nan, dtype=np.float32),
            "latent3_support_distance": np.full(3, np.nan, dtype=np.float32),
            "latent3_grid_flags": np.zeros(3, dtype=np.int64),
            "latent3_etas": np.full(3, np.nan, dtype=np.float32),
            "latent3_plan_score": np.float64(np.nan),
            "latent3_direct_score": np.float64(np.nan),
            "latent3_score_spread": np.float64(np.nan),
            "latent3_surrogate_score": np.float64(np.nan),
            "latent3_valid_routes": np.int64(0),
            "latent3_considered_routes": np.int64(0),
            "latent3_clipped_etas": np.int64(0),
            "latent3_fallback_reason": np.int64(0),
            "latent3_planning_seconds": np.float64(0),
        }

    def _task_intention(self, observation, task_latent, *, rng, temperature, mode):
        if mode == "baseline":
            intention, _ = self.frozen_fb.baseline_high_intention(
                observation,
                task_latent,
                seed=rng,
                temperature=temperature,
            )
            return intention
        return self.frozen_fb.normalize_latent(task_latent)

    def _arrived(self, observation) -> bool:
        if not self.replan_on_arrival or self._cached is None or not self._cached.route_indices:
            return False
        first = self._cached.route_indices[0]
        current = np.asarray(self.planner.geometry.encode(observation), dtype=np.float32)
        waypoint = self.planner.candidate_embeddings[first]
        difference = (current - waypoint) / self.planner.embedding_scale
        return float(np.linalg.norm(difference)) <= self.waypoint_replan_radius

    def select_intention(self, observation, task_latent, *, rng, temperature):
        observation = np.asarray(observation, dtype=np.float32)
        goal_distance = float(np.linalg.norm(observation[:2] - self.planner.goal_xy))
        finish = goal_distance <= self.finish_radius
        replanned = False

        # Возле цели используем отдельный режим, не изменяя исходное представление задачи.
        if finish:
            intention = self._task_intention(
                observation,
                task_latent,
                rng=rng,
                temperature=temperature,
                mode=self.finish_mode,
            )
            diagnostics = (
                dict(self._cached.diagnostics)
                if self._cached is not None
                else self._empty_planner_diagnostics()
            )
        else:
            interval_elapsed = (
                self._cached is None
                or self._step - self._last_plan_step >= self.replan_interval
            )
            arrived = self._arrived(observation)
            # Новый маршрут требуется по таймеру либо при фактическом достижении первой точки.
            if interval_elapsed or arrived:
                self._cached = self.planner.select(observation, task_latent)
                self._last_plan_step = self._step
                self._replan_count += 1
                replanned = True
            diagnostics = dict(self._cached.diagnostics)
            if self._cached.fallback:
                intention = self._task_intention(
                    observation,
                    task_latent,
                    rng=rng,
                    temperature=temperature,
                    mode=self.fallback_mode,
                )
            else:
                intention = self._cached.intention

        diagnostics["latent3_replanned"] = bool(replanned)
        diagnostics["latent3_finish"] = bool(finish)
        diagnostics["latent3_replan_count"] = np.int64(self._replan_count)
        diagnostics["latent3_goal_distance"] = np.float64(goal_distance)
        diagnostics["latent3_step"] = np.int64(self._step)
        self._step += 1
        return IntentionSelection(intention=intention, diagnostics=diagnostics)
