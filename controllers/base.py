"""Общий интерфейс контроллеров и формат выбранного намерения."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True)
class IntentionSelection:
    intention: Any
    diagnostics: Mapping[str, Any] = field(default_factory=dict)


class HighLevelController(ABC):
    """Определяет общий контракт между планировщиком и запуском эпизода."""

    method_name: str

    def reset(self, scenario=None) -> None:
        """Сбрасывает внутреннее состояние контроллера перед новым эпизодом."""

    def experiment_config(self) -> Mapping[str, Any]:
        """Возвращает сохраняемые параметры и происхождение данных метода."""

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
        """Возвращает единственное намерение, которое должно исполняться сейчас."""

