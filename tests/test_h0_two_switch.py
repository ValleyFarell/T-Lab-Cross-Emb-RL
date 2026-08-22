"""Contract tests for the H0 two-switch addon.

These tests intentionally cover boundaries between H0 and the existing stand:
the vectorized score, the first executable subgoal, CLI selection, and generic
diagnostic logging.  They do not load MuJoCo or a checkpoint.
"""

from __future__ import annotations

import ast
import inspect
import sys

import numpy as np
import pytest

from baseline.two_switch_planner import TwoSwitchPlanner
from controllers.base import HighLevelController, IntentionSelection
from controllers.two_switch import TwoSwitchController
from evaluation.runner import EpisodeRunner
from evaluation.scenarios import Scenario
from scripts import run_baseline


class FakeFrozenFB:
    """Small deterministic FB model with a two-member forward ensemble."""

    @staticmethod
    def normalize_latent(latent):
        latent = np.asarray(latent, dtype=np.float64)
        norm = np.linalg.norm(latent, axis=-1, keepdims=True)
        return latent / (norm + 1e-8) * np.sqrt(latent.shape[-1])

    @staticmethod
    def backward_repr(observations):
        observations = np.asarray(observations, dtype=np.float64)
        return observations[..., :2]

    @staticmethod
    def forward_repr(observations, intentions):
        observations = np.asarray(observations, dtype=np.float64)
        intentions = np.asarray(intentions, dtype=np.float64)
        first = observations[..., :2] + 0.25 * intentions + 0.5
        second = first + 0.2
        return np.stack([first, second], axis=0)


def _reference_score(fake_fb, observation, goal_latent, candidates):
    """Transparent scalar-loop version of the implemented H0 formula."""

    observation = np.asarray(observation, dtype=np.float64)
    goal_latent = np.asarray(goal_latent, dtype=np.float64)
    goal_policy_latent = fake_fb.normalize_latent(goal_latent)
    candidates = np.asarray(candidates, dtype=np.float64)
    latents = fake_fb.normalize_latent(fake_fb.backward_repr(candidates))

    def mean_forward(state, latent):
        return np.asarray(
            fake_fb.forward_repr(state[None], latent[None])
        ).mean(axis=0)[0]

    direct_value = np.dot(
        mean_forward(observation, goal_policy_latent),
        goal_latent,
    )
    scores = np.empty((len(candidates), len(candidates)))

    for i, (w1, z1) in enumerate(zip(candidates, latents)):
        fs_z1 = mean_forward(observation, z1)
        fw1_z1 = mean_forward(w1, z1)
        eta1 = np.dot(fs_z1, z1) / (np.dot(fw1_z1, z1) + 1e-8)

        for j, (w2, z2) in enumerate(zip(candidates, latents)):
            fw1_z2 = mean_forward(w1, z2)
            fw2_z2 = mean_forward(w2, z2)
            fw2_zg = mean_forward(w2, goal_policy_latent)
            eta2 = np.dot(fw1_z2, z2) / (np.dot(fw2_z2, z2) + 1e-8)

            scores[i, j] = (
                np.dot(fs_z1, goal_latent)
                + eta1
                * (
                    np.dot(fw1_z2, goal_latent)
                    - np.dot(fw1_z1, goal_latent)
                )
                + eta1
                * eta2
                * (
                    np.dot(fw2_zg, goal_latent)
                    - np.dot(fw2_z2, goal_latent)
                )
                - direct_value
            )
    return scores


def test_vectorized_pair_score_matches_transparent_reference():
    frozen_fb = FakeFrozenFB()
    candidates = np.array(
        [
            [1.0, 2.0],
            [2.0, 1.0],
            [1.5, 0.5],
        ]
    )
    observation = np.array([0.5, 1.5])
    goal_latent = np.array([0.8, 1.2])
    planner = TwoSwitchPlanner(frozen_fb, candidates)

    actual = np.asarray(planner.score_pairs(observation, goal_latent))
    expected = _reference_score(
        frozen_fb,
        observation,
        goal_latent,
        candidates,
    )

    assert actual.shape == (3, 3)
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)
    assert np.all(np.isfinite(actual))


def test_selected_pair_executes_w1_not_w2(monkeypatch):
    """A plan s -> w1 -> w2 -> g must first send w1 to low_actor."""

    candidates = np.array([[1.0, 2.0], [8.0, 9.0]])
    planner = TwoSwitchPlanner(FakeFrozenFB(), candidates)
    monkeypatch.setattr(
        planner,
        "score_pairs",
        lambda observation, goal_latent: np.array([[0.0, 10.0], [1.0, 2.0]]),
    )

    selection = planner.select(
        observation=np.array([0.0, 0.0]),
        goal_latent=np.array([1.0, 1.0]),
    )

    # argmax is pair (w1=0, w2=1), therefore the executable intention is B(w1).
    np.testing.assert_allclose(
        selection.intention,
        FakeFrozenFB.normalize_latent(candidates[0]),
    )
    assert selection.diagnostics["w1_index"] == 0
    assert selection.diagnostics["w2_index"] == 1


class SpyPlanner:
    def __init__(self):
        self.calls = []

    def select(self, observation, task_latent):
        self.calls.append((np.asarray(observation), np.asarray(task_latent)))
        return type(
            "Selection",
            (),
            {
                "intention": np.array([3.0, 4.0]),
                "diagnostics": {
                    "h0_score": 7.5,
                    "w1_index": 1,
                    "w2_index": 2,
                },
            },
        )()


def test_h0_controller_isolated_behind_common_interface():
    planner = SpyPlanner()
    controller = TwoSwitchController(planner)

    assert isinstance(controller, HighLevelController)
    assert controller.method_name == "fbpiswitch_h0_two_switch"

    result = controller.select_intention(
        observation=np.array([1.0, 2.0]),
        task_latent=np.array([5.0, 6.0]),
        rng=object(),
        temperature=0.0,
    )

    np.testing.assert_array_equal(result.intention, [3.0, 4.0])
    assert result.diagnostics == {
        "h0_score": 7.5,
        "w1_index": 1,
        "w2_index": 2,
    }
    assert len(planner.calls) == 1


@pytest.mark.parametrize("controller_name", ["baseline", "direct", "h0"])
def test_cli_accepts_every_public_controller(monkeypatch, controller_name):
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_baseline", "--controller", controller_name],
    )
    args = run_baseline.parse_args()
    assert args.controller == controller_name


def test_cli_names_and_runtime_dispatch_are_identical():
    """Prevent argparse choices from drifting away from main() branches."""

    tree = ast.parse(inspect.getsource(run_baseline.main))
    dispatched_names = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Compare) or len(node.comparators) != 1:
            continue
        left = node.left
        right = node.comparators[0]
        if (
            isinstance(left, ast.Attribute)
            and isinstance(left.value, ast.Name)
            and left.value.id == "args"
            and left.attr == "controller"
            and isinstance(right, ast.Constant)
            and isinstance(right.value, str)
        ):
            dispatched_names.add(right.value)

    assert {"baseline", "direct", "h0"}.issubset(dispatched_names)


class FakeActionSpace:
    def seed(self, seed):
        self.last_seed = seed


class OneStepEnv:
    def __init__(self):
        self.unwrapped = self
        self.action_space = FakeActionSpace()
        self.cur_goal_xy = np.array([0.0, 0.0])

    def reset(self, *, seed=None, options=None):
        return np.array([1.0, 1.0]), {}

    def step(self, action):
        return np.array([0.0, 0.0]), 0.0, True, False, {"success": True}


class FakeLowLevel:
    @staticmethod
    def low_action(observation, intention, *, seed, temperature):
        return np.array([0.0])


class DiagnosticController(HighLevelController):
    method_name = "diagnostic_h0"

    def select_intention(
        self,
        observation,
        task_latent,
        *,
        rng,
        temperature,
    ):
        return IntentionSelection(
            intention=np.array([1.0, 0.0]),
            diagnostics={
                "h0_score": 2.5,
                "w1_index": 3,
                "w2_index": 4,
            },
        )


def test_runner_preserves_h0_diagnostics():
    runner = EpisodeRunner(
        OneStepEnv(),
        FakeLowLevel(),
        DiagnosticController(),
        eval_temperature=0.0,
    )
    result = runner.run(
        Scenario(
            scenario_id="h0-test",
            task_id=1,
            environment_seed=5,
            controller_seed=7,
        ),
        task_latent=np.array([0.0, 1.0]),
    )

    np.testing.assert_array_equal(result.diagnostics["h0_score"], [2.5])
    np.testing.assert_array_equal(result.diagnostics["w1_index"], [3])
    np.testing.assert_array_equal(result.diagnostics["w2_index"], [4])


@pytest.mark.parametrize(
    ("candidates", "max_candidates"),
    [
        (np.empty((0, 2)), None),
        (np.ones((2, 2)), 0),
        (np.array([[1.0, np.nan]]), None),
    ],
)
def test_planner_rejects_invalid_candidate_sets(candidates, max_candidates):
    with pytest.raises(ValueError):
        TwoSwitchPlanner(
            FakeFrozenFB(),
            candidates,
            max_candidates=max_candidates,
        )
