"""Controller adapter for the H0 two-switch planner."""

from __future__ import annotations

from numbers import Integral

from .base import HighLevelController, IntentionSelection


class TwoSwitchController(HighLevelController):
    method_name = "fbpiswitch_h0_two_switch"
    replanned_diagnostic = "h0_replanned"

    def __init__(self, planner, *, replan_interval: int = 1):
        if (
            isinstance(replan_interval, bool)
            or not isinstance(replan_interval, Integral)
            or replan_interval <= 0
        ):
            raise ValueError("replan_interval must be a positive integer")
        self.planner = planner
        self.replan_interval = int(replan_interval)
        self.reset()

    def reset(self, scenario=None) -> None:
        del scenario
        self._step = 0
        self._cached = None

    def experiment_config(self):
        config = dict(self.planner.experiment_config())
        config.update(
            {
                "execution_semantics": "execute_w1_then_replan",
                "replan_interval": self.replan_interval,
            }
        )
        return config

    def select_intention(
        self,
        observation,
        task_latent,
        *,
        rng,
        temperature,
    ):
        # H0 selection is a deterministic argmax.  These parameters still
        # control the low-level action in EpisodeRunner.
        del rng, temperature

        replanned = self._cached is None or self._step % self.replan_interval == 0
        if replanned:
            self._cached = self.planner.select(observation, task_latent)

        diagnostics = dict(self._cached.diagnostics)
        diagnostics[self.replanned_diagnostic] = bool(replanned)
        result = IntentionSelection(
            intention=self._cached.intention,
            diagnostics=diagnostics,
        )
        self._step += 1
        return result
