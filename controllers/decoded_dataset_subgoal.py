"""Контроллер проекции декодированной подцели на реальное состояние."""

from __future__ import annotations

from numbers import Integral

from .base import HighLevelController, IntentionSelection


class DecodedDatasetSubgoalController(HighLevelController):
    method_name = "decoded_dataset_subgoal_vmax_finish"

    def __init__(
        self,
        frozen_fb,
        planner,
        *,
        replan_interval: int = 5,
        finish_mode: str = "dynamic-v-max",
    ):
        if (
            isinstance(replan_interval, bool)
            or not isinstance(replan_interval, Integral)
            or replan_interval <= 0
        ):
            raise ValueError("replan_interval must be a positive integer")
        if finish_mode not in {
            "task-latent",
            "fixed-v-max",
            "dynamic-v-max",
        }:
            raise ValueError(
                "finish_mode must be 'task-latent', 'fixed-v-max', "
                "or 'dynamic-v-max'"
            )
        self.frozen_fb = frozen_fb
        self.planner = planner
        self.replan_interval = int(replan_interval)
        self.finish_mode = finish_mode
        if finish_mode == "task-latent":
            self.method_name = "decoded_dataset_subgoal_task_latent_finish"
        elif finish_mode == "fixed-v-max":
            self.method_name = "decoded_dataset_subgoal_fixed_vmax_finish"
        else:
            self.method_name = "decoded_dataset_subgoal_vmax_finish"
        self.reset()

    def reset(self, scenario=None) -> None:
        del scenario
        self._step = 0
        self._cached = None
        self._finish = None

    def experiment_config(self):
        config = dict(self.planner.experiment_config())
        config["replan_interval"] = self.replan_interval
        config["finish_mode"] = self.finish_mode
        if self.finish_mode == "task-latent":
            config["finish_selection"] = "original_task_latent_fixed_for_episode"
        elif self.finish_mode == "fixed-v-max":
            config["finish_selection"] = "v-max_once_at_episode_start"
        else:
            config["finish_selection"] = "v-max_each_hierarchical_replan"
        config["source_high_policy"] = "frozen_fbpiswitch_high_actor"
        return config

    def select_intention(self, observation, task_latent, *, rng, temperature):
        replanned = self._cached is None or self._step % self.replan_interval == 0
        finish_replanned = False
        if replanned:
            # Координаты задачи неизменны; режим определяет только представителя полного состояния цели.
            if self.finish_mode == "task-latent":
                finish_intention = task_latent
                finish_diagnostics = {
                    "finish_source": "original_task_latent",
                }
            else:
                if self._finish is None or self.finish_mode == "dynamic-v-max":
                    self._finish = self.planner.select_finish(
                        observation,
                        task_latent,
                    )
                    finish_replanned = True
                finish_intention = self._finish.intention
                finish_diagnostics = self._finish.diagnostics
            high_intention, raw_high = self.frozen_fb.baseline_high_intention(
                observation,
                finish_intention,
                seed=rng,
                temperature=temperature,
            )
            # Исполняем намерение реального офлайн-состояния, а не координаты декодера напрямую.
            projected = self.planner.select(
                observation,
                task_latent,
                high_intention,
            )
            diagnostics = dict(finish_diagnostics)
            diagnostics.update(projected.diagnostics)
            diagnostics["selected_finish_policy_intention"] = finish_intention
            diagnostics["raw_high_actor_output"] = raw_high
            diagnostics["original_high_intention"] = high_intention
            self._cached = IntentionSelection(
                intention=projected.intention,
                diagnostics=diagnostics,
            )

        diagnostics = dict(self._cached.diagnostics)
        diagnostics["projection_replanned"] = bool(replanned)
        diagnostics["finish_replanned"] = bool(finish_replanned)
        self._step += 1
        return IntentionSelection(
            intention=self._cached.intention,
            diagnostics=diagnostics,
        )
