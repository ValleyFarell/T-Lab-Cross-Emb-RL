"""Проверки корректности компонента goal capture и его взаимодействия со стендом."""

import numpy as np

from evaluation.goal_capture import (
    count_radius_exits,
    find_first_entry_state,
    summarize_branch,
    velocity_components,
)


def test_first_entry_excludes_already_successful_states():
    observations = np.zeros((5, 4))
    observations[:, :2] = [[2.0, 0.0], [0.9, 0.0], [0.4, 0.0], [0.8, 0.0], [2.0, 0.0]]
    entry = find_first_entry_state(observations, [0.0, 0.0])
    assert entry.observation_index == 1
    assert np.isclose(entry.distance, 0.9)


def test_velocity_components_use_positive_inward_sign():
    radial, tangential = velocity_components([1.0, 0.0], [-2.0, 3.0], [0.0, 0.0])
    assert np.isclose(radial, 2.0)
    assert np.isclose(tangential, 3.0)


def test_radius_exits_count_only_inside_to_outside_crossings():
    assert count_radius_exits([0.8, 1.1, 1.2, 0.9, 1.01, 0.7]) == 2


def test_summary_counts_hit_and_keeps_final_state():
    positions = [[0.9, 0.0], [1.1, 0.0], [0.7, 0.0], [0.4, 0.0]]
    result = summarize_branch(
        positions,
        radial_velocities=[-0.2, 0.4, 0.3],
        tangential_speeds=[1.0, 0.8, 0.2],
        torso_heights=[0.5, 0.5, 0.5],
        goal_xy=[0.0, 0.0],
    )
    assert result["hit_success_radius"] is True
    assert result["hit_step"] == 3
    assert result["radius_exits"] == 1
    assert np.isclose(result["final_distance"], 0.4)
