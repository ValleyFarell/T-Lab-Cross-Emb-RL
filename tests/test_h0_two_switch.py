"""Contract tests for H0 and its boundaries with the shared stand."""

from __future__ import annotations

import sys

import numpy as np
import pytest

from baseline.two_switch_planner import TwoSwitchPlanner
from controllers import TwoSwitchController
from controllers.base import HighLevelController, IntentionSelection
from evaluation.runner import EpisodeRunner
from evaluation.save_episode import save_episode_result
from evaluation.scenarios import Scenario
from scripts import run_baseline, run_h0


class FakeFrozenFB:
    @staticmethod
    def normalize_latent(latent):
        latent = np.asarray(latent, dtype=np.float64)
        norm = np.linalg.norm(latent, axis=-1, keepdims=True)
        return latent / np.maximum(norm, 1e-8) * np.sqrt(latent.shape[-1])

    @staticmethod
    def backward_repr(observations):
        return np.asarray(observations, dtype=np.float64)[..., :2]

    @staticmethod
    def forward_repr(observations, intentions):
        observations = np.asarray(observations, dtype=np.float64)
        intentions = np.asarray(intentions, dtype=np.float64)
        first = observations[..., :2] + 0.25 * intentions + 0.5
        return np.stack([first, first + 0.2], axis=0)


def _safe_eta(numerator, denominator, epsilon=1e-6):
    if not np.isfinite(numerator) or not np.isfinite(denominator):
        return None
    if abs(denominator) < epsilon:
        return None
    return np.clip(numerator / denominator, 0.0, 1.0)


def _reference_score(fake_fb, observation, goal_latent, candidates):
    observation = np.asarray(observation, dtype=np.float64)
    goal_latent = np.asarray(goal_latent, dtype=np.float64)
    goal_policy = fake_fb.normalize_latent(goal_latent)
    candidates = np.asarray(candidates, dtype=np.float64)
    latents = fake_fb.normalize_latent(fake_fb.backward_repr(candidates))

    def mean_forward(state, latent):
        return np.asarray(fake_fb.forward_repr(state[None], latent[None])).mean(0)[0]

    direct = np.dot(mean_forward(observation, goal_policy), goal_latent)
    scores = np.full((len(candidates), len(candidates)), -np.inf)
    for i, (w1, z1) in enumerate(zip(candidates, latents)):
        fs_z1 = mean_forward(observation, z1)
        fw1_z1 = mean_forward(w1, z1)
        eta1 = _safe_eta(np.dot(fs_z1, z1), np.dot(fw1_z1, z1))
        if eta1 is None:
            continue
        for j, (w2, z2) in enumerate(zip(candidates, latents)):
            fw1_z2 = mean_forward(w1, z2)
            fw2_z2 = mean_forward(w2, z2)
            eta2 = _safe_eta(np.dot(fw1_z2, z2), np.dot(fw2_z2, z2))
            if eta2 is None:
                continue
            scores[i, j] = (
                np.dot(fs_z1, goal_latent)
                + eta1 * (np.dot(fw1_z2, goal_latent) - np.dot(fw1_z1, goal_latent))
                + eta1 * eta2
                * (
                    np.dot(mean_forward(w2, goal_policy), goal_latent)
                    - np.dot(fw2_z2, goal_latent)
                )
                - direct
            )
    return scores


def test_chunked_pair_score_matches_transparent_reference():
    frozen_fb = FakeFrozenFB()
    candidates = np.array([[1.0, 2.0], [2.0, 1.0], [1.5, 0.5]])
    planner = TwoSwitchPlanner(
        frozen_fb,
        candidates,
        max_candidates=3,
        pair_batch_size=2,
    )
    actual = np.asarray(planner.score_pairs([0.5, 1.5], [0.8, 1.2]))
    expected = _reference_score(
        frozen_fb,
        np.array([0.5, 1.5]),
        np.array([0.8, 1.2]),
        candidates,
    )
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)


def test_selected_pair_executes_w1_not_w2(monkeypatch):
    candidates = np.array([[1.0, 2.0], [8.0, 9.0]])
    planner = TwoSwitchPlanner(FakeFrozenFB(), candidates)
    monkeypatch.setattr(
        planner,
        "score_pairs",
        lambda observation, goal_latent: np.array([[0.0, 10.0], [1.0, 2.0]]),
    )
    selection = planner.select([0.0, 0.0], [1.0, 1.0])
    np.testing.assert_allclose(
        selection.intention,
        FakeFrozenFB.normalize_latent(candidates[0]),
    )
    assert selection.diagnostics["w1_index"] == 0
    assert selection.diagnostics["w2_index"] == 1


class ZeroMeasureFB(FakeFrozenFB):
    @staticmethod
    def forward_repr(observations, intentions):
        observations = np.asarray(observations)
        return np.zeros((2, len(observations), 2), dtype=np.float64)


def test_near_zero_eta_denominators_invalidate_pairs():
    planner = TwoSwitchPlanner(ZeroMeasureFB(), [[1.0, 2.0], [2.0, 1.0]])
    assert np.all(np.isneginf(np.asarray(planner.score_pairs([0.0, 0.0], [1.0, 1.0]))))
    with pytest.raises(RuntimeError, match="no valid candidate pair"):
        planner.select([0.0, 0.0], [1.0, 1.0])


@pytest.mark.parametrize(
    ("candidates", "kwargs"),
    [
        (np.empty((0, 2)), {}),
        (np.ones(2), {}),
        (np.array([["x", "y"]]), {}),
        (np.array([[1.0, np.nan]]), {}),
        (np.ones((2, 2)), {"max_candidates": 0}),
        (np.ones((2, 2)), {"pair_batch_size": 0}),
        (np.ones((2, 2)), {"eta_epsilon": 0.0}),
    ],
)
def test_invalid_candidate_configuration_is_rejected(candidates, kwargs):
    with pytest.raises(ValueError):
        TwoSwitchPlanner(FakeFrozenFB(), candidates, **kwargs)


def test_candidate_metadata_is_reproducible_and_complete():
    candidates = np.arange(20, dtype=np.float32).reshape(10, 2)
    first = TwoSwitchPlanner(FakeFrozenFB(), candidates, max_candidates=4)
    second = TwoSwitchPlanner(FakeFrozenFB(), candidates, max_candidates=4)
    config = first.experiment_config()
    assert config["candidate_count"] == 4
    assert config["pair_count"] == 16
    assert config["candidate_selection"] == "deterministic_linspace"
    assert config["candidate_checksum_sha256"] == second.candidate_checksum
    assert len(config["candidate_source_indices"]) == 4


class SpyPlanner:
    def __init__(self):
        self.calls = 0

    def select(self, observation, task_latent):
        self.calls += 1
        return type(
            "Selection",
            (),
            {
                "intention": np.array([3.0, 4.0]),
                "diagnostics": {"h0_score": 7.5, "w1_index": 1, "w2_index": 2},
            },
        )()

    def experiment_config(self):
        return {"candidate_count": 2}


def test_replan_interval_reuses_w1_with_stable_diagnostics():
    planner = SpyPlanner()
    controller = TwoSwitchController(planner, replan_interval=2)
    first = controller.select_intention([1.0, 2.0], [5.0, 6.0], rng=None, temperature=0)
    second = controller.select_intention([2.0, 3.0], [5.0, 6.0], rng=None, temperature=0)
    assert planner.calls == 1
    assert first.diagnostics["h0_replanned"] is True
    assert second.diagnostics["h0_replanned"] is False
    assert set(first.diagnostics) == set(second.diagnostics)
    assert controller.experiment_config()["execution_semantics"] == "execute_w1_then_replan"


@pytest.mark.parametrize("name", run_baseline.PUBLIC_CONTROLLERS)
def test_cli_accepts_every_public_controller(name):
    assert run_baseline.parse_args(["--controller", name]).controller == name


def test_h0_launcher_delegates_to_shared_main(monkeypatch):
    received = []
    monkeypatch.setattr(run_baseline, "main", lambda argv=None: received.append(argv))
    run_h0.main(["--task-id", "2"])
    assert received == [["--controller", "h0", "--task-id", "2"]]


class OneStepEnv:
    def __init__(self):
        self.unwrapped = self
        self.cur_goal_xy = np.array([0.0, 0.0])

    def reset(self, *, seed=None, options=None):
        return np.array([1.0, 1.0]), {}

    def step(self, action):
        return np.array([0.0, 0.0]), 0.0, True, False, {"success": True}


class FakeLowLevel:
    latent_dim = 2

    @staticmethod
    def low_action(observation, intention, *, seed, temperature):
        return np.array([0.0])


class DiagnosticController(HighLevelController):
    method_name = "diagnostic_h0"

    def select_intention(self, observation, task_latent, *, rng, temperature):
        return IntentionSelection(
            intention=np.array([1.0, 0.0]),
            diagnostics={"h0_score": 2.5, "w1_index": 3, "w2_index": 4},
        )


def test_runner_and_saver_preserve_arbitrary_h0_diagnostics(tmp_path, monkeypatch):
    runner = EpisodeRunner(OneStepEnv(), FakeLowLevel(), DiagnosticController())
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

    monkeypatch.setattr("evaluation.save_episode.plot_path", lambda *args, **kwargs: None)
    save_episode_result(result, tmp_path, OneStepEnv())
    with np.load(tmp_path / "trajectory.npz") as trajectory:
        np.testing.assert_array_equal(trajectory["diagnostic_h0_score"], [2.5])
        np.testing.assert_array_equal(trajectory["diagnostic_w1_index"], [3])
        np.testing.assert_array_equal(trajectory["diagnostic_w2_index"], [4])

