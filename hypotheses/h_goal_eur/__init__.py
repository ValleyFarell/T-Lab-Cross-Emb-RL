"""Способы выбора полного состояния около конечной цели."""

from .planner import DatasetMaxValueGoalPlanner, SyntheticCurrentGoalPlanner

__all__ = ["DatasetMaxValueGoalPlanner", "SyntheticCurrentGoalPlanner"]
