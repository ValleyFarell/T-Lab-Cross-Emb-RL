from __future__ import annotations

import jax
import numpy as np

from controllers.base import HighLevelController, IntentionSelection
from evaluation.runner import EpisodeRunner
from evaluation.scenarios import Scenario


class FakeEnv:
    def __init__(self):
        self.unwrapped = self
        self.cur_goal_xy = np.array([2.0, 2.0])
        self._step = 0

    def reset(self, *, seed, options):
        rng = np.random.default_rng(seed)
        self._observation = np.concatenate(
            [rng.normal(size=2), np.zeros(2)]
        )
        self._step = 0
        return self._observation.copy(), {}

    def step(self, action):
        self._step += 1
        self._observation[:2] += np.asarray(action[:2])
        truncated = self._step == 2
        return (
            self._observation.copy(),
            0.0,
            False,
            truncated,
            {"success": False},
        )


class FakeController(HighLevelController):
    method_name = "fake"

    def select_intention(
        self,
        observation,
        task_latent,
        *,
        rng,
        temperature,
    ):
        intention = jax.random.normal(rng, (2,))
        return IntentionSelection(intention=intention)


class FakeFrozenFB:
    def low_action(
        self,
        observation,
        intention,
        *,
        seed,
        temperature,
    ):
        return np.asarray(intention) + np.asarray(
            jax.random.normal(seed, (2,))
        )


def run_scenario(environment_seed):
    runner = EpisodeRunner(
        FakeEnv(),
        FakeFrozenFB(),
        FakeController(),
        eval_temperature=1.0,
    )
    scenario = Scenario(
        scenario_id=f"test-{environment_seed}",
        task_id=1,
        environment_seed=environment_seed,
        controller_seed=7,
    )
    return runner.run(scenario, np.zeros(2))


def test_same_scenario_is_reproducible():
    first = run_scenario(3)
    second = run_scenario(3)

    np.testing.assert_array_equal(first.start_xy, second.start_xy)
    np.testing.assert_array_equal(first.actions, second.actions)
    np.testing.assert_array_equal(first.positions, second.positions)


def test_environment_seed_changes_initial_state_only():
    first = run_scenario(3)
    second = run_scenario(4)

    assert not np.array_equal(first.start_xy, second.start_xy)
    np.testing.assert_array_equal(first.actions, second.actions)
