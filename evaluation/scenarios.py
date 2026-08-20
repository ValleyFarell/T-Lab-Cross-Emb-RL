"""Scenario definitions used by every high-level controller."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Scenario:
    scenario_id: str
    task_id: int
    environment_seed: int | None = None
    controller_seed: int = 0
