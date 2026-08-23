"""Локальный H0 с резервным прямым управлением и отдельным финишем."""

from __future__ import annotations

from numbers import Integral

import numpy as np

from .base import HighLevelController, IntentionSelection


class LocalTerminalController(HighLevelController):
    """Исполняет локальную подцель и переключает управление возле финиша."""

    method_name = "fbpiswitch_h0_local_terminal"

    def __init__(
        self,
        frozen_fb,
        planner,
        *,
        replan_interval: int = 1,
        latch_finish: bool = True,
    ):
        if (
            isinstance(replan_interval, bool)
            or not isinstance(replan_interval, Integral)
            or replan_interval <= 0
        ):
            raise ValueError("replan_interval must be a positive integer")
        if not isinstance(latch_finish, (bool, np.bool_)):
            raise ValueError("latch_finish must be boolean")
        self.frozen_fb = frozen_fb
        self.planner = planner
        self.replan_interval = int(replan_interval)
        self.latch_finish = bool(latch_finish)
        self.reset()

    def reset(self, scenario=None) -> None:
        del scenario
        self._step = 0
        self._cached = None
        self._finish_latched = False

    def experiment_config(self):
        config = dict(self.planner.experiment_config())
        config.update(
            {
                "execution_semantics": "execute_local_w1_then_replan",
                "replan_interval": self.replan_interval,
                "finish_latched_until_episode_end": self.latch_finish,
            }
        )
        return config

    # Контроллер сохраняет единый формат диагностики независимо от режима управления.
    def select_intention(self, observation, task_latent, *, rng, temperature):
        goal_distance = float(
            np.linalg.norm(np.asarray(observation)[:2] - self.planner.goal_xy)
        )
        inside_finish = (
            self.planner.finish_radius > 0
            and goal_distance <= self.planner.finish_radius
        )
        was_latched = self._finish_latched
        if inside_finish and self.latch_finish:
            self._finish_latched = True
        terminal = bool(inside_finish or self._finish_latched)
        previous_terminal = bool(
            self._cached is not None
            and self._cached.diagnostics["h0lt_terminal_active"]
        )
        replanned = (
            self._cached is None
            or self._step % self.replan_interval == 0
            or terminal != previous_terminal
            or (self._finish_latched and not was_latched)
        )

        if replanned:
            baseline_intention = None
            if terminal and self.planner.finish_mode == "baseline":
                baseline_intention, _ = self.frozen_fb.baseline_high_intention(
                    observation,
                    task_latent,
                    seed=rng,
                    temperature=temperature,
                )
            self._cached = self.planner.select(
                observation,
                task_latent,
                force_terminal=terminal,
                baseline_intention=baseline_intention,
            )

        diagnostics = dict(self._cached.diagnostics)
        diagnostics["h0lt_replanned"] = bool(replanned)
        diagnostics["h0lt_finish_latched"] = bool(self._finish_latched)
        diagnostics["h0lt_goal_distance"] = np.float64(goal_distance)
        self._step += 1
        return IntentionSelection(
            intention=self._cached.intention, diagnostics=diagnostics
        )
