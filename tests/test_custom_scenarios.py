"""Проверки корректности компонента custom scenarios и его взаимодействия со стендом."""

import numpy as np
import pytest

from evaluation.scenarios import Scenario, xy_to_free_grid_cell


class FakeMaze:
    def __init__(self):
        self.unwrapped = self
        self.maze_map = np.array(
            [
                [1, 1, 1, 1],
                [1, 0, 1, 1],
                [1, 0, 0, 1],
                [1, 1, 1, 1],
            ]
        )

    @staticmethod
    def xy_to_ij(xy):
        return int((xy[1] + 6) / 4), int((xy[0] + 6) / 4)

    @staticmethod
    def ij_to_xy(ij):
        return 4 * ij[1] - 4, 4 * ij[0] - 4


def test_custom_scenario_builds_task_info():
    scenario = Scenario(
        scenario_id="custom",
        task_id=None,
        environment_seed=3,
        controller_seed=7,
        start_ij=(1, 1),
        goal_ij=(2, 2),
    )

    assert scenario.is_custom
    assert scenario.reset_options() == {
        "task_info": {"init_ij": (1, 1), "goal_ij": (2, 2)}
    }


def test_scenario_rejects_mixed_standard_and_custom_modes():
    with pytest.raises(ValueError):
        Scenario(
            scenario_id="invalid",
            task_id=1,
            start_ij=(1, 1),
            goal_ij=(2, 2),
        )


def test_xy_validation_accepts_free_center_and_rejects_wall():
    env = FakeMaze()

    assert xy_to_free_grid_cell(env, (0, 0), name="start") == (1, 1)
    with pytest.raises(ValueError, match="wall"):
        xy_to_free_grid_cell(env, (4, 0), name="goal")


def test_xy_validation_rejects_non_center():
    with pytest.raises(ValueError, match="not a cell center"):
        xy_to_free_grid_cell(FakeMaze(), (0.25, 0), name="start")

