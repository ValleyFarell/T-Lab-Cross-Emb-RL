"""Проверки корректности компонента h goal eur и его взаимодействия со стендом."""

from __future__ import annotations

import numpy as np

from controllers import GoalEurController
from hypotheses.h_goal_eur import (
    DatasetMaxValueGoalPlanner,
    SyntheticCurrentGoalPlanner,
)
from scripts import run_baseline, run_h_goal_eur


class FakeFB:
    def __init__(self):
        self.last_backward_input = None

    def backward_repr(self, observations):
        observations = np.asarray(observations)
        self.last_backward_input = observations.copy()
        return observations[..., 2:4]

    @staticmethod
    def normalize_latent(latent):
        latent = np.asarray(latent, dtype=np.float64)
        norm = np.linalg.norm(latent, axis=-1, keepdims=True)
        return latent / np.maximum(norm, 1e-12)

    @staticmethod
    def forward_repr(observations, intentions):
        del observations
        intentions = np.asarray(intentions)
        first_candidate = intentions[:, 0] > intentions[:, 1]
        # Кандидат 0: оценки ансамбля [3, -1], среднее 1, размах 4.
        # Кандидат 1: оценки ансамбля [1, 1], среднее 1, размах 0.
        head_0 = np.where(first_candidate, 3.0, 1.0)
        head_1 = np.where(first_candidate, -1.0, 1.0)
        zeros = np.zeros_like(head_0)
        return np.stack(
            [
                np.stack([head_0, zeros], axis=-1),
                np.stack([head_1, zeros], axis=-1),
            ],
            axis=0,
        )


def _offline_dataset():
    observations = np.zeros((4, 29), dtype=np.float64)
    observations[0, 2:4] = [1.0, 0.0]
    observations[1, 2:4] = [0.0, 1.0]
    observations[2, 2:4] = [1.0, 1.0]
    observations[3, 2:4] = [-1.0, 0.0]
    qpos = np.zeros((4, 15), dtype=np.float64)
    qpos[:, :2] = [[4.0, 4.0], [4.1, 4.0], [8.0, 8.0], [9.0, 9.0]]
    return {"observations": observations, "qpos": qpos}


def test_synthetic_current_replaces_only_xy():
    frozen_fb = FakeFB()
    planner = SyntheticCurrentGoalPlanner(frozen_fb, [4.0, 5.0])
    observation = np.arange(29, dtype=np.float64)
    selection = planner.select(observation, [1.0, 0.0])

    expected = observation.astype(np.float32)
    expected[:2] = [4.0, 5.0]
    np.testing.assert_array_equal(frozen_fb.last_backward_input, expected)
    np.testing.assert_array_equal(selection.diagnostics["hge_synthetic_target"], expected)
    np.testing.assert_allclose(selection.intention, [2.0, 3.0] / np.sqrt(13.0))


def test_dataset_max_v_uses_downstream_task_value_and_conservative_score():
    planner = DatasetMaxValueGoalPlanner(
        FakeFB(),
        _offline_dataset(),
        [4.0, 4.0],
        candidate_radius=0.5,
        max_candidates=64,
        disagreement_penalty=0.5,
    )
    selection = planner.select(np.zeros(29), np.asarray([1.0, 0.0]))

    # Для двух оценщиков и штрафа 0.5 итог равен меньшей из двух оценок.
    # Кандидат 0 получает -1, кандидат 1 получает 1.
    assert selection.diagnostics["hge_candidate_index"] == 1
    assert selection.diagnostics["hge_dataset_index"] == 1
    assert selection.diagnostics["hge_score"] == 1.0
    assert selection.diagnostics["hge_ensemble_range"] == 0.0
    np.testing.assert_allclose(selection.intention, [0.0, 1.0])


def test_dataset_candidates_are_real_goal_region_states_and_reproducible():
    first = DatasetMaxValueGoalPlanner(
        FakeFB(),
        _offline_dataset(),
        [4.0, 4.0],
        max_candidates=1,
    )
    second = DatasetMaxValueGoalPlanner(
        FakeFB(),
        _offline_dataset(),
        [4.0, 4.0],
        max_candidates=1,
    )
    config = first.experiment_config()
    assert config["goal_match_count"] == 2
    assert config["candidate_count"] == 1
    assert config["candidate_source_indices"] == [0]
    assert config["candidate_checksum_sha256"] == second.candidate_checksum


def test_controller_reuses_cached_selection_with_stable_diagnostics():
    class SpyPlanner:
        method_name = "h_goal_eur_spy"

        def __init__(self):
            self.calls = 0

        def select(self, observation, task_latent):
            self.calls += 1
            return type(
                "Selection",
                (),
                {
                    "intention": np.asarray([1.0, 0.0]),
                    "diagnostics": {"hge_score": 2.0},
                },
            )()

        @staticmethod
        def experiment_config():
            return {"variant": "spy"}

    planner = SpyPlanner()
    controller = GoalEurController(planner, replan_interval=2)
    first = controller.select_intention(None, None, rng=None, temperature=0.0)
    second = controller.select_intention(None, None, rng=None, temperature=0.0)
    assert planner.calls == 1
    assert first.diagnostics["hge_replanned"] is True
    assert second.diagnostics["hge_replanned"] is False
    assert set(first.diagnostics) == set(second.diagnostics)


def test_h_goal_eur_controllers_are_public_cli_choices():
    assert run_baseline.parse_args(["--controller", "hge-synthetic"]).controller == (
        "hge-synthetic"
    )
    assert run_baseline.parse_args(["--controller", "hge-max-v"]).controller == (
        "hge-max-v"
    )


def test_h_goal_eur_launcher_delegates_to_shared_main(monkeypatch):
    received = []
    monkeypatch.setattr(run_baseline, "main", lambda argv=None: received.append(argv))
    run_h_goal_eur.main(
        ["--variant", "dataset-max-v", "--start-xy", "0", "0", "--goal-xy", "4", "4"]
    )
    assert received == [
        [
            "--controller",
            "hge-max-v",
            "--start-xy",
            "0",
            "0",
            "--goal-xy",
            "4",
            "4",
        ]
    ]
