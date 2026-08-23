"""Точное воспроизведение исходного контроллера FB π-Switch."""

from __future__ import annotations

from baseline.frozen_fb import FrozenFB

from .base import HighLevelController, IntentionSelection


class BaselineController(HighLevelController):
    method_name = "fbpiswitch_baseline"

    def __init__(self, frozen_fb: FrozenFB):
        self.frozen_fb = frozen_fb

    def select_intention(
        self,
        observation,
        task_latent,
        *,
        rng,
        temperature: float,
    ) -> IntentionSelection:
        intention, raw = self.frozen_fb.baseline_high_intention(
            observation,
            task_latent,
            seed=rng,
            temperature=temperature,
        )
        return IntentionSelection(
            intention=intention,
            diagnostics={"raw_high_actor_output": raw},
        )
