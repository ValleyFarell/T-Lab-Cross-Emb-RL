"""Проверки корректности компонента value geometry и его взаимодействия со стендом."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from hypotheses.value_geometry.analysis import (
    affine_probe,
    matched_goal_pose_diagnostics,
    regression_metrics,
)
from hypotheses.value_geometry.data import (
    MazeGeometry,
    PairSplit,
    build_state_pool,
    estimate_distance_edges,
    sample_pairs,
    select_goal_indices,
    split_state_indices,
)
from hypotheses.value_geometry.experiment import ExperimentConfig, run_experiment
from hypotheses.value_geometry.models import (
    TrainingConfig,
    _backward_model,
    _forward_model,
    _huber_loss,
    _init_mlp,
    fit_value_model,
)
from hypotheses.value_geometry.teacher import (
    OfflineFBTeacher,
    aggregate_ensemble,
    exact_binary_reward_raw_latent,
)


class _FakeFrozenFB:
    latent_dim = 4
    config = {"reward_temperature": 0.0, "normalize_latent": True}

    @staticmethod
    def normalize_latent(values):
        values = np.asarray(values, dtype=np.float32)
        return values / (np.linalg.norm(values, axis=-1, keepdims=True) + 1e-8) * 2.0

    @staticmethod
    def backward_repr(observations):
        observations = np.asarray(observations, dtype=np.float32)
        return np.column_stack(
            (
                observations[:, 0] + 2.0,
                observations[:, 1] + 1.0,
                observations[:, 2] + 0.5,
                np.ones(len(observations), dtype=np.float32),
            )
        ).astype(np.float32)

    @staticmethod
    def forward_repr(observations, intentions):
        observations = np.asarray(observations, dtype=np.float32)
        intentions = np.asarray(intentions, dtype=np.float32)
        base = intentions + 0.10 * observations[:, :4]
        return np.stack((base, base + 0.25), axis=0)


class _FakeMaze:
    maze_map = np.asarray(
        (
            (1, 1, 1, 1, 1),
            (1, 0, 1, 0, 1),
            (1, 0, 0, 0, 1),
            (1, 1, 1, 1, 1),
        ),
        dtype=np.int8,
    )

    @property
    def unwrapped(self):
        return self

    @staticmethod
    def ij_to_xy(ij):
        return np.asarray((ij[1], ij[0]), dtype=np.float32)


def _dataset(seed: int = 1, count: int = 600):
    rng = np.random.default_rng(seed)
    observations = rng.normal(size=(count, 29)).astype(np.float32)
    observations[:, :2] = rng.uniform(-2, 2, size=(count, 2))
    terminals = np.zeros(count, dtype=np.int8)
    terminals[19::20] = 1
    return observations, terminals


class DatasetTests(unittest.TestCase):
    def test_trajectory_split_has_no_overlap(self):
        observations, terminals = _dataset(count=600)
        splits, groups, strategy = split_state_indices(
            len(observations), terminals=terminals, max_states=300, seed=7
        )
        self.assertEqual(strategy, "trajectory")
        self.assertEqual(sum(map(len, splits.values())), 300)
        train_groups = set(groups[splits["train"]])
        validation_groups = set(groups[splits["validation"]])
        test_groups = set(groups[splits["test"]])
        self.assertTrue(train_groups.isdisjoint(validation_groups))
        self.assertTrue(train_groups.isdisjoint(test_groups))
        self.assertTrue(validation_groups.isdisjoint(test_groups))

    def test_maze_distance_respects_wall(self):
        positions = np.asarray(((1, 1), (3, 1)), dtype=np.float32)
        geometry = MazeGeometry.from_environment(_FakeMaze(), positions)
        distance = geometry.distance(np.asarray([0]), np.asarray([1]), positions)
        self.assertAlmostEqual(float(distance[0]), 4.0)
        self.assertAlmostEqual(float(np.linalg.norm(positions[0] - positions[1])), 2.0)

    def test_pairs_stay_inside_their_trajectory_split(self):
        observations, terminals = _dataset()
        pool = build_state_pool(observations, terminals=terminals, max_states=450, seed=2)
        goals = select_goal_indices(pool, "train", count=16, seed=3)
        edges = estimate_distance_edges(
            pool, pool.split_indices["train"], goals, geometry=None, seed=4
        )
        pairs = sample_pairs(
            pool,
            "train",
            goals,
            number_of_pairs=256,
            distance_edges=edges,
            geometry=None,
            seed=5,
        )
        allowed = pool.split_indices["train"]
        self.assertTrue(np.all(np.isin(pairs.start_indices, allowed)))
        self.assertTrue(np.all(np.isin(pairs.goal_indices, allowed)))
        self.assertGreaterEqual(len(np.unique(pairs.distance_bins)), 2)

    def test_matched_pose_groups_use_real_coordinate_neighborhoods(self):
        positions = np.asarray(
            ((0.0, 0.0), (1.99, 2.0), (2.01, 2.0), (2.40, 2.0)),
            dtype=np.float32,
        )
        pairs = PairSplit(
            start_indices=np.asarray((0, 0, 0), dtype=np.int64),
            goal_indices=np.asarray((1, 2, 3), dtype=np.int64),
            distances=np.asarray((2.8, 2.8, 3.1), dtype=np.float32),
            distance_bins=np.asarray((1, 1, 1), dtype=np.int64),
            values=np.asarray((0.4, 0.5, 0.9), dtype=np.float32),
        )
        result = matched_goal_pose_diagnostics(
            pairs,
            positions,
            np.asarray((0.42, 0.48, 0.8), dtype=np.float32),
            bucket_radius=0.10,
        )
        self.assertEqual(result["groups"], 1)
        self.assertAlmostEqual(result["teacher_mean_pose_spread"], 0.10, places=6)
        self.assertLess(result["max_goal_xy_span"], 0.03)


class TeacherTests(unittest.TestCase):
    def test_default_two_member_score_is_exact_minimum(self):
        rng = np.random.default_rng(3)
        forward = rng.normal(size=(2, 17, 4)).astype(np.float32)
        rewards = rng.normal(size=(17, 4)).astype(np.float32)
        score, members = aggregate_ensemble(
            forward, rewards, disagreement_penalty=0.5
        )
        np.testing.assert_allclose(score, members.min(axis=1), atol=3e-7)

    def test_binary_reward_latent_matches_original_softmax_then_mean(self):
        rng = np.random.default_rng(4)
        backward = rng.normal(size=(23, 4)).astype(np.float32)
        rewards = np.zeros(23, dtype=np.float32)
        rewards[[1, 5, 9, 12, 18]] = 1
        for temperature in (-1.5, 0.0, 2.0):
            weights = np.exp(temperature * rewards)
            weights /= weights.sum()
            expected = np.mean(weights[:, None] * rewards[:, None] * backward, axis=0)
            result = exact_binary_reward_raw_latent(
                backward[rewards > 0].sum(axis=0, dtype=np.float64),
                total_samples=len(rewards),
                positive_samples=int(rewards.sum()),
                reward_temperature=temperature,
            )
            np.testing.assert_allclose(result, expected, rtol=5e-6, atol=1e-8)

    def test_xy_goal_teacher_uses_supported_reference_states(self):
        observations, terminals = _dataset(count=240)
        pool = build_state_pool(observations, terminals=terminals, max_states=200, seed=4)
        split_goals = {
            name: select_goal_indices(pool, name, count=5, seed=index + 9)
            for index, name in enumerate(("train", "validation", "test"))
        }
        teacher = OfflineFBTeacher(
            _FakeFrozenFB(),
            pool,
            observations,
            observations[:, :2],
            goal_tolerance=0.6,
            target_mode="xy-goal",
            reference_samples=len(observations),
            batch_size=13,
        )
        banks = teacher.prepare_goal_banks(split_goals)
        self.assertTrue(np.all(banks["train"].support_sizes > 0))
        goals = banks["train"].goal_indices
        pairs = PairSplit(
            start_indices=np.asarray([pool.split_indices["train"][0]] * len(goals)),
            goal_indices=goals,
            distances=np.ones(len(goals), dtype=np.float32),
            distance_bins=np.zeros(len(goals), dtype=np.int16),
        )
        scored = teacher.score_pairs(pairs, banks["train"])
        self.assertEqual(scored.ensemble_values.shape, (len(goals), 2))
        np.testing.assert_allclose(
            scored.values,
            scored.ensemble_values.min(axis=1),
            atol=1e-6,
        )


class ModelTests(unittest.TestCase):
    def test_shared_encoder_gradient_matches_finite_difference(self):
        rng = np.random.default_rng(7)
        parameters = {}
        _init_mlp(parameters, "encoder", (4, 5, 2), rng)
        _init_mlp(parameters, "head", (4, 6, 1), rng)
        starts = rng.normal(size=(7, 4)).astype(np.float32)
        goals = rng.normal(size=(7, 4)).astype(np.float32)
        target = rng.normal(size=(7, 1)).astype(np.float32)
        prediction, caches = _forward_model(
            "latent2", parameters, starts, goals, cache=True
        )
        _, gradient = _huber_loss(prediction, target, delta=10.0)
        analytical = _backward_model("latent2", parameters, caches, gradient)
        epsilon = 2e-3
        name = "encoder.w0"
        index = (1, 0)
        original = float(parameters[name][index])
        parameters[name][index] = original + epsilon
        plus, _ = _forward_model("latent2", parameters, starts, goals, cache=False)
        loss_plus, _ = _huber_loss(plus, target, delta=10.0)
        parameters[name][index] = original - epsilon
        minus, _ = _forward_model("latent2", parameters, starts, goals, cache=False)
        loss_minus, _ = _huber_loss(minus, target, delta=10.0)
        parameters[name][index] = original
        numerical = (loss_plus - loss_minus) / (2 * epsilon)
        self.assertAlmostEqual(float(analytical[name][index]), numerical, delta=2e-3)

    def test_xy_model_learns_simple_directional_value(self):
        observations, terminals = _dataset(count=900)
        pool = build_state_pool(observations, terminals=terminals, max_states=800, seed=6)
        pair_splits = {}
        for index, name in enumerate(("train", "validation")):
            goals = select_goal_indices(pool, name, count=16, seed=15 + index)
            edges = estimate_distance_edges(
                pool, pool.split_indices[name], goals, geometry=None, seed=index
            )
            pairs = sample_pairs(
                pool,
                name,
                goals,
                number_of_pairs=1_200 if name == "train" else 320,
                distance_edges=edges,
                geometry=None,
                seed=18 + index,
            )
            start_xy = pool.positions[pairs.start_indices]
            goal_xy = pool.positions[pairs.goal_indices]
            values = 0.7 * start_xy[:, 0] - 0.4 * goal_xy[:, 1]
            pair_splits[name] = pairs.with_values(values)
        model = fit_value_model(
            "xy",
            pool.observations,
            pair_splits["train"],
            pair_splits["validation"],
            train_state_indices=pool.split_indices["train"],
            config=TrainingConfig(
                hidden_width=20,
                hidden_layers=1,
                max_epochs=45,
                patience=10,
                batch_size=96,
                learning_rate=5e-3,
            ),
            seed=11,
        )
        prediction = model.predict(
            pool.observations,
            pair_splits["validation"].start_indices,
            pair_splits["validation"].goal_indices,
        )
        metrics = regression_metrics(pair_splits["validation"].values, prediction)
        self.assertGreater(metrics["r2"], 0.97)

    def test_affine_probe_recovers_invertible_coordinates(self):
        rng = np.random.default_rng(10)
        xy = rng.normal(size=(100, 2))
        matrix = np.asarray(((2.0, 0.4), (-0.3, 1.5)))
        embedding = xy @ matrix + np.asarray((0.6, -0.2))
        result = affine_probe(embedding[:70], xy[:70], embedding[70:], xy[70:])
        self.assertGreater(result["metrics"]["r2"], 0.999999)


class EndToEndTests(unittest.TestCase):
    def test_synthetic_run_and_cache_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            training = TrainingConfig(
                hidden_width=16,
                hidden_layers=1,
                batch_size=64,
                max_epochs=5,
                patience=3,
                log_every=5,
            )
            first = ExperimentConfig(
                output_dir=directory,
                synthetic=True,
                synthetic_states=360,
                max_states=330,
                train_pairs=280,
                goal_count=12,
                models=("xy", "latent2"),
                training=training,
                plots=False,
            )
            summary = run_experiment(first)
            self.assertEqual(summary["dataset"]["source"], "synthetic")
            self.assertTrue((Path(directory) / "pair_cache.npz").is_file())
            self.assertTrue((Path(directory) / "model_comparison.csv").is_file())
            self.assertIn("latent2", summary["runs"]["0"]["models"])

            resumed = ExperimentConfig(
                output_dir=directory,
                resume=True,
                train_pairs=280,
                models=("full",),
                training=training,
                plots=False,
            )
            result = run_experiment(resumed)
            self.assertTrue(result["dataset"]["resumed_from_cached_teacher_labels"])
            self.assertIn("full", result["runs"]["0"]["models"])
            with (Path(directory) / "metrics.json").open(encoding="utf-8") as stream:
                saved = json.load(stream)
            self.assertEqual(saved["experiment"], "frozen_fb_value_geometry")


if __name__ == "__main__":
    unittest.main()
