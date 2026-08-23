"""Общий контроллер двух способов выбора состояния около цели."""

from __future__ import annotations

from numbers import Integral

from .base import HighLevelController, IntentionSelection


class GoalEurController(HighLevelController):
    """Исполняет намерение, выбранное одной из стратегий около цели."""

    def __init__(self, planner, *, replan_interval: int = 1):
        if (
            isinstance(replan_interval, bool)
            or not isinstance(replan_interval, Integral)
            or replan_interval <= 0
        ):
            raise ValueError("replan_interval must be a positive integer")
        self.planner = planner
        self.replan_interval = int(replan_interval)
        self.method_name = planner.method_name
        self.reset()

    def reset(self, scenario=None) -> None:
        del scenario
        self._step = 0
        self._cached = None

    def experiment_config(self):
        config = dict(self.planner.experiment_config())
        config["replan_interval"] = self.replan_interval
        return config

    def select_intention(
        self,
        observation,
        task_latent,
        *,
        rng,
        temperature,
    ) -> IntentionSelection:
        del rng, temperature
        replanned = self._cached is None or self._step % self.replan_interval == 0
        if replanned:
            self._cached = self.planner.select(observation, task_latent)

        diagnostics = dict(self._cached.diagnostics)
        diagnostics["hge_replanned"] = bool(replanned)
        result = IntentionSelection(
            intention=self._cached.intention,
            diagnostics=diagnostics,
        )
        self._step += 1
        return result
