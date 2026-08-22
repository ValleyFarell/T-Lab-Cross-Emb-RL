"""Common high-level controller interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True)
class IntentionSelection:
    intention: Any
    diagnostics: Mapping[str, Any] = field(default_factory=dict)


class HighLevelController(ABC):
    """Stable boundary between a hypothesis and the evaluation pipeline.

    A controller may expose any numeric per-step diagnostics, but it must keep
    the same diagnostic keys for every step in an episode.  Static,
    JSON-serializable method parameters belong in ``experiment_config``.
    """

    method_name: str

    def reset(self, scenario=None) -> None:
        """Reset method-specific episode state."""

    def experiment_config(self) -> Mapping[str, Any]:
        """Return method-specific, JSON-serializable experiment metadata."""

        return {}

    @abstractmethod
    def select_intention(
        self,
        observation,
        task_latent,
        *,
        rng,
        temperature: float,
    ) -> IntentionSelection:
        """Return the one intention that should be executed now."""

