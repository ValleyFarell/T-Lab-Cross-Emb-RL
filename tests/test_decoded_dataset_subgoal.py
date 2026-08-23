"""Проверки корректности компонента decoded dataset subgoal и его взаимодействия со стендом."""

import numpy as np
import pytest

from hypotheses.decoded_dataset_subgoal import DecodedDatasetSubgoalPlanner
from controllers.decoded_dataset_subgoal import DecodedDatasetSubgoalController
from scripts.run_decoded_dataset_subgoal import parse_args


class FakeDecoder:
    latent_dim = 3
    architecture = (3, 8, 2)

    def predict(self, intention):
        del intention
        return np.asarray([4.0, 4.0])


class FakeFB:
    latent_dim = 3

    def __init__(self):
        self.last_high_goal = None
        self.high_calls = 0

    def backward_repr(self, observations):
        observations = np.asarray(observations)
        return np.stack(
            [observations[:, 2], np.ones(len(observations)), np.zeros(len(observations))],
            axis=1,
        )

    def normalize_latent(self, latent):
        return np.asarray(latent)

    def forward_repr(self, observations, intentions):
        del observations
        intentions = np.asarray(intentions)
        return np.stack((intentions, intentions * np.asarray([0.9, 1.0, 1.0])))

    def baseline_high_intention(
        self, observation, task_latent, *, seed, temperature
    ):
        del observation, seed, temperature
        self.last_high_goal = np.asarray(task_latent)
        self.high_calls += 1
        return np.asarray(task_latent), np.asarray(task_latent)


def _dataset():
    observations = np.zeros((4, 29), dtype=np.float32)
    observations[0, :3] = [4.00, 4.00, 1.0]
    observations[1, :3] = [4.10, 4.00, 3.0]
    observations[2, :3] = [3.90, 4.05, 2.0]
    observations[3, :3] = [20.0, 20.0, 100.0]
    return {"observations": observations}


def test_max_v_projects_to_real_local_state():
    planner = DecodedDatasetSubgoalPlanner(
        FakeFB(),
        FakeDecoder(),
        _dataset(),
        goal_xy=np.asarray([4.0, 4.0]),
        candidate_radius=0.5,
        max_candidates=64,
        disagreement_penalty=0.5,
        selection_mode="max-v",
    )
    selection = planner.select(
        np.zeros(29),
        np.asarray([1.0, 0.0, 0.0]),
        np.asarray([1.0, 0.0, 0.0]),
    )
    assert int(selection.diagnostics["projected_dataset_index"]) == 1
    np.testing.assert_allclose(selection.diagnostics["decoded_subgoal_xy"], [4.0, 4.0])
    np.testing.assert_allclose(selection.diagnostics["projected_dataset_xy"], [4.1, 4.0])
    np.testing.assert_allclose(selection.intention, [3.0, 1.0, 0.0])


def test_nearest_xy_ablation_ignores_value():
    planner = DecodedDatasetSubgoalPlanner(
        FakeFB(),
        FakeDecoder(),
        _dataset(),
        goal_xy=np.asarray([4.0, 4.0]),
        candidate_radius=0.5,
        selection_mode="nearest-xy",
    )
    selection = planner.select(
        np.zeros(29),
        np.asarray([1.0, 0.0, 0.0]),
        np.asarray([1.0, 0.0, 0.0]),
    )
    assert int(selection.diagnostics["projected_dataset_index"]) == 0


def test_finish_is_reselected_by_vmax_for_the_true_xy_reward():
    planner = DecodedDatasetSubgoalPlanner(
        FakeFB(),
        FakeDecoder(),
        _dataset(),
        goal_xy=np.asarray([4.0, 4.0]),
        candidate_radius=0.5,
        max_candidates=64,
        disagreement_penalty=0.5,
        selection_mode="max-v",
    )
    finish = planner.select_finish(
        np.zeros(29),
        np.asarray([1.0, 0.0, 0.0]),
    )
    assert int(finish.diagnostics["finish_dataset_index"]) == 1
    np.testing.assert_allclose(finish.intention, [3.0, 1.0, 0.0])


def test_controller_reselects_vmax_finish_at_configured_interval():
    frozen_fb = FakeFB()
    planner = DecodedDatasetSubgoalPlanner(
        frozen_fb,
        FakeDecoder(),
        _dataset(),
        goal_xy=np.asarray([4.0, 4.0]),
        candidate_radius=0.5,
        selection_mode="max-v",
    )
    controller = DecodedDatasetSubgoalController(
        frozen_fb,
        planner,
        replan_interval=5,
        finish_mode="dynamic-v-max",
    )
    results = [
        controller.select_intention(
            np.zeros(29),
            np.asarray([1.0, 0.0, 0.0]),
            rng=step,
            temperature=0.0,
        )
        for step in range(6)
    ]
    np.testing.assert_allclose(frozen_fb.last_high_goal, [3.0, 1.0, 0.0])
    assert frozen_fb.high_calls == 2
    assert bool(results[0].diagnostics["projection_replanned"])
    assert bool(results[0].diagnostics["finish_replanned"])
    assert not bool(results[1].diagnostics["projection_replanned"])
    assert not bool(results[1].diagnostics["finish_replanned"])
    assert bool(results[5].diagnostics["projection_replanned"])
    assert bool(results[5].diagnostics["finish_replanned"])


@pytest.mark.parametrize(
    ("replan_interval", "expected_high_calls"),
    [(1, 6), (5, 2)],
)
def test_fixed_vmax_finish_is_selected_once_while_subgoal_replans(
    replan_interval,
    expected_high_calls,
):
    frozen_fb = FakeFB()
    planner = DecodedDatasetSubgoalPlanner(
        frozen_fb,
        FakeDecoder(),
        _dataset(),
        goal_xy=np.asarray([4.0, 4.0]),
        candidate_radius=0.5,
        selection_mode="max-v",
    )
    original_select_finish = planner.select_finish
    finish_calls = []

    def counted_select_finish(observation, task_latent):
        finish_calls.append(1)
        return original_select_finish(observation, task_latent)

    planner.select_finish = counted_select_finish
    controller = DecodedDatasetSubgoalController(
        frozen_fb,
        planner,
        replan_interval=replan_interval,
        finish_mode="fixed-v-max",
    )
    results = [
        controller.select_intention(
            np.zeros(29),
            np.asarray([1.0, 0.0, 0.0]),
            rng=step,
            temperature=0.0,
        )
        for step in range(6)
    ]

    assert len(finish_calls) == 1
    assert frozen_fb.high_calls == expected_high_calls
    assert sum(bool(result.diagnostics["finish_replanned"]) for result in results) == 1
    assert controller.experiment_config()["finish_selection"] == (
        "v-max_once_at_episode_start"
    )


@pytest.mark.parametrize(
    ("replan_interval", "expected_high_calls"),
    [(1, 6), (5, 2)],
)
def test_task_latent_finish_stays_exact_while_only_subgoal_replans(
    replan_interval,
    expected_high_calls,
):
    frozen_fb = FakeFB()
    planner = DecodedDatasetSubgoalPlanner(
        frozen_fb,
        FakeDecoder(),
        _dataset(),
        goal_xy=np.asarray([4.0, 4.0]),
        candidate_radius=0.5,
        selection_mode="max-v",
    )
    finish_calls = []

    def forbidden_select_finish(observation, task_latent):
        finish_calls.append((observation, task_latent))
        raise AssertionError("task-latent mode must not call select_finish")

    planner.select_finish = forbidden_select_finish
    controller = DecodedDatasetSubgoalController(
        frozen_fb,
        planner,
        replan_interval=replan_interval,
        finish_mode="task-latent",
    )
    task_latent = np.asarray([1.0, 0.0, 0.0])
    results = [
        controller.select_intention(
            np.zeros(29),
            task_latent,
            rng=step,
            temperature=0.0,
        )
        for step in range(6)
    ]

    assert finish_calls == []
    assert frozen_fb.high_calls == expected_high_calls
    np.testing.assert_allclose(frozen_fb.last_high_goal, task_latent)
    assert not any(bool(result.diagnostics["finish_replanned"]) for result in results)
    assert all(
        result.diagnostics["finish_source"] == "original_task_latent"
        for result in results
    )
    assert controller.experiment_config()["finish_selection"] == (
        "original_task_latent_fixed_for_episode"
    )


def test_cli_accepts_task_latent_finish_mode():
    args = parse_args(["--task-id", "1", "--finish-mode", "task-latent"])
    assert args.finish_mode == "task-latent"


def test_manual_coordinates_are_a_complete_alternative_to_task_id():
    args = parse_args(["--start-xy", "0", "0", "--goal-xy", "4", "4"])
    assert args.task_id is None
    assert args.start_xy == [0.0, 0.0]
    assert args.goal_xy == [4.0, 4.0]

    with pytest.raises(SystemExit):
        parse_args(["--start-xy", "0", "0"])
    with pytest.raises(SystemExit):
        parse_args(
            [
                "--task-id", "1",
                "--start-xy", "0", "0",
                "--goal-xy", "4", "4",
            ]
        )
