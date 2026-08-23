"""Проверки корректности компонента h0b adaptive и его взаимодействия со стендом."""

from __future__ import annotations

import numpy as np

from controllers import AdaptiveSwitchController
from hypotheses.h0b import AdaptiveSwitchPlanner
from scripts import run_baseline, run_h0b


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


def _planner():
    return AdaptiveSwitchPlanner(
        FakeFrozenFB(),
        np.array([[1.0, 2.0], [8.0, 9.0]]),
        max_candidates=2,
        pair_batch_size=2,
    )


def _install_scores(monkeypatch, planner, one_values, two_values):
    planner._last_details = {
        "direct_value": np.array(1.0),
        "eta1": np.array([0.5, 0.6]),
        "eta2": np.full((2, 2), 0.7),
        "eta1_valid": np.ones(2, dtype=bool),
        "eta2_valid": np.ones((2, 2), dtype=bool),
        "eta1_clipped": np.zeros(2, dtype=bool),
        "eta2_clipped": np.zeros((2, 2), dtype=bool),
    }

    def fake_score_depths(observation, goal_latent):
        return np.asarray(one_values), np.asarray(two_values)

    monkeypatch.setattr(planner, "score_depths", fake_score_depths)


def test_h0b_selects_depth_one_when_depth_two_has_no_strict_improvement(monkeypatch):
    planner = _planner()
    _install_scores(
        monkeypatch,
        planner,
        one_values=[10.0, 1.0],
        two_values=[[10.0, 3.0], [4.0, 1.0]],
    )
    selection = planner.select([0.0, 0.0], [1.0, 1.0])
    assert selection.diagnostics["h0b_selected_depth"] == 1
    assert selection.diagnostics["h0b_selected_value"] == 10.0
    assert selection.diagnostics["w1_index"] == 0
    assert selection.diagnostics["w2_index"] == -1
    np.testing.assert_allclose(selection.intention, planner.candidate_latents[0])


def test_h0b_selects_depth_two_and_executes_its_w1(monkeypatch):
    planner = _planner()
    _install_scores(
        monkeypatch,
        planner,
        one_values=[1.0, 2.0],
        two_values=[[1.0, 9.0], [3.0, 2.0]],
    )
    selection = planner.select([0.0, 0.0], [1.0, 1.0])
    assert selection.diagnostics["h0b_selected_depth"] == 2
    assert selection.diagnostics["h0b_selected_value"] == 9.0
    assert selection.diagnostics["w1_index"] == 0
    assert selection.diagnostics["w2_index"] == 1
    np.testing.assert_allclose(selection.intention, planner.candidate_latents[0])


def test_exact_value_tie_prefers_depth_one(monkeypatch):
    planner = _planner()
    _install_scores(
        monkeypatch,
        planner,
        one_values=[1.0, 9.0],
        two_values=[[1.0, 9.0], [3.0, 9.0]],
    )
    selection = planner.select([0.0, 0.0], [1.0, 1.0])
    assert selection.diagnostics["h0b_selected_depth"] == 1
    assert selection.diagnostics["w1_index"] == 1


def test_depth_scores_have_expected_shapes_and_common_finite_scale():
    planner = _planner()
    v1, v2 = planner.score_depths([0.5, 1.5], [0.8, 1.2])
    assert np.asarray(v1).shape == (2,)
    assert np.asarray(v2).shape == (2, 2)
    assert np.all(np.isfinite(np.asarray(v1)))
    assert np.all(np.isfinite(np.asarray(v2)))
    np.testing.assert_allclose(np.diag(np.asarray(v2)), np.asarray(v1))
    assert float(np.max(v2)) >= float(np.max(v1))


def test_h0b_controller_is_exported_and_identified():
    controller = AdaptiveSwitchController(_planner())
    assert controller.method_name == "fbpiswitch_h0b_adaptive_depth"
    config = controller.experiment_config()
    assert config["adaptive_depths"] == [1, 2]
    assert config["depth_tie_break"] == "prefer_depth_1"


def test_generic_cli_accepts_h0b():
    args = run_baseline.parse_args(["--controller", "h0b"])
    assert args.controller == "h0b"


def test_h0b_launcher_delegates_to_shared_main(monkeypatch):
    received = []
    monkeypatch.setattr(run_baseline, "main", lambda argv=None: received.append(argv))
    run_h0b.main(["--task-id", "2"])
    assert received == [["--controller", "h0b", "--task-id", "2"]]
