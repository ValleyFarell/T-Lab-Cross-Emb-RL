"""Диагностический контроллер прямого целевого намерения."""

from __future__ import annotations

from baseline.frozen_fb import FrozenFB

from .base import HighLevelController, IntentionSelection


class DirectGoalController(HighLevelController):
    """Передаёт целевое намерение напрямую низкоуровневой политике."""

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

