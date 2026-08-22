"""Diagnostic controller that bypasses the learned high-level actor."""

from __future__ import annotations

from baseline.frozen_fb import FrozenFB

from .base import HighLevelController, IntentionSelection


class DirectGoalController(HighLevelController):
    """Pass the normalized downstream-task latent directly to the low actor."""

    method_name = "fb_direct_goal"

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
        del observation, rng, temperature
        intention = task_latent
        return IntentionSelection(intention=intention)

