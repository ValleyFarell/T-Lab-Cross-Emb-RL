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
    method_name: str

    def reset(self, scenario=None) -> None:
        """Reset method-specific episode state."""

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
