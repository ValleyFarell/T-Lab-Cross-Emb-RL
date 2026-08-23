"""Проверки корректности компонента latent three dynamic и его взаимодействия со стендом."""

from __future__ import annotations

import tempfile
import unittest
import importlib.util
import contextlib
import io
from pathlib import Path
import sys
import types

import numpy as np

# Общий controllers/__init__.py заранее импортирует контроллеры с зависимостями JAX/Flax.
# Сам планировщик требует меньше зависимостей, поэтому разрешаем запуск его
# отдельных тестов в минимальном окружении без этих пакетов.
if importlib.util.find_spec("flax") is None and "controllers" not in sys.modules:
    controller_package = types.ModuleType("controllers")
    controller_package.__path__ = [str(Path(__file__).resolve().parents[1] / "controllers")]
    sys.modules["controllers"] = controller_package

from controllers.latent_three_dynamic import DynamicLatentThreeController
from hypotheses.latent_three_dynamic.geometry import (
    LatentGeometryModel,
    LatentIntentionDecoder,
    blocked_split,
    normalize_intentions,
)
from hypotheses.latent_three_dynamic.planner import DynamicThreeWaypointPlanner
from scripts.analyze_latent_three_dynamic import paired_report
from scripts.run_latent_three_dynamic import parse_args as parse_run_args
from scripts.train_latent_three_dynamic import main as train_main
from scripts.train_latent_three_dynamic import parse_args as parse_train_args


class FakeGeometry:
    embedding_dim = 4

    @staticmethod
    def encode(observations):
        array = np.asarray(observations, dtype=np.float32)
        return array[..., :4]

    @staticmethod
    def predict_value(starts, goals):
        starts = np.asarray(starts, dtype=np.float32)
        goals = np.asarray(goals, dtype=np.float32)
        single = starts.ndim == goals.ndim == 1
        if starts.ndim == 1:
            starts = starts[None, :]
        if goals.ndim == 1:
            goals = goals[None, :]
        if len(starts) == 1 and len(goals) > 1:
            starts = np.repeat(starts, len(goals), axis=0)
        if len(goals) == 1 and len(starts) > 1:
            goals = np.repeat(goals, len(starts), axis=0)
        distance = np.linalg.norm(starts - goals, axis=-1)
        output = 3.0 / (1.0 + distance)
        return float(output[0]) if single else output


class FakeDecoder:
    embedding_dim = 4
    latent_dim = 4

    @staticmethod
    def predict(embeddings):
        embeddings = np.asarray(embeddings, dtype=np.float32)
        raw = embeddings + np.asarray([1.3, 1.1, 1.4, 1.2], dtype=np.float32)
        return normalize_intentions(raw, 4)


class FakeFB:
    latent_dim = 4

    @staticmethod
    def normalize_latent(values):
        return normalize_intentions(values, 4)

    @staticmethod
    def backward_repr(observations):
        observations = np.asarray(observations, dtype=np.float32)
        return observations[..., :4] + np.asarray([1.3, 1.1, 1.4, 1.2], dtype=np.float32)

    @staticmethod
    def forward_repr(observations, intentions):
        observations = np.asarray(observations, dtype=np.float32)
        intentions = np.asarray(intentions, dtype=np.float32)
        base = 2.5 + 0.12 * observations[:, :4] + 0.08 * intentions
        return np.stack([base, base + 0.04], axis=0)

    @staticmethod
    def baseline_high_intention(observation, task_latent, *, seed, temperature):
        del observation, seed, temperature
        return normalize_intentions(task_latent, 4), np.asarray(task_latent)


class ZeroFB(FakeFB):
    @staticmethod
    def forward_repr(observations, intentions):
        del intentions
        return np.zeros((2, len(observations), 4), dtype=np.float32)


def make_dataset(size=72):
    rng = np.random.default_rng(7)
    observations = rng.uniform(0.1, 3.0, size=(size, 6)).astype(np.float32)
    return {"observations": observations}


def make_planner(**overrides):
    options = {
        "goal_xy": np.asarray([2.8, 2.8], dtype=np.float32),
        "max_states": 72,
        "max_candidates": 24,
        "grid_resolution": 3,
        "rerank_count": 8,
        "fb_batch_size": 16,
        "min_improvement": -1e6,
        "minimum_eta": 0.0,
    }
    options.update(overrides)
    frozen_fb = options.pop("frozen_fb", FakeFB())
    return DynamicThreeWaypointPlanner(
        frozen_fb,
        FakeGeometry(),
        FakeDecoder(),
        make_dataset(),
        **options,
    )


class GeometryTests(unittest.TestCase):
    def test_normalization_matches_frozen_actor_radius(self):
        vectors = normalize_intentions(np.asarray([[3.0, 4.0], [1.0, -2.0]]), 2)
        np.testing.assert_allclose(np.linalg.norm(vectors, axis=-1), np.sqrt(2), atol=1e-6)

    def test_zero_intention_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "non-zero"):
            normalize_intentions(np.zeros(4), 4)

    def test_blocked_split_keeps_entire_blocks_together(self):
        train, validation = blocked_split(96, seed=3, block_size=8)
        self.assertFalse(set(train.tolist()) & set(validation.tolist()))
        train_blocks = set((train // 8).tolist())
        validation_blocks = set((validation // 8).tolist())
        self.assertFalse(train_blocks & validation_blocks)

    def test_geometry_training_and_safe_roundtrip(self):
        rng = np.random.default_rng(4)
        states = rng.normal(size=(320, 6)).astype(np.float32)
        starts = rng.integers(0, len(states), size=1800)
        goals = rng.integers(0, len(states), size=1800)
        values = (
            4.0
            - 0.8 * np.square(states[starts, 0] - states[goals, 0])
            - 0.5 * np.square(states[starts, 1] - states[goals, 1])
        ).astype(np.float32)
        validation = np.zeros(len(starts), dtype=bool)
        validation[-300:] = True
        model, report = LatentGeometryModel.fit(
            states,
            starts,
            goals,
            values,
            validation_mask=validation,
            hidden_dim=24,
            value_hidden_dim=32,
            epochs=36,
            patience=12,
            batch_size=96,
            seed=2,
        )
        self.assertEqual(model.encode(states[:5]).shape, (5, 4))
        self.assertGreater(report.validation_r2, 0.45)
        with tempfile.TemporaryDirectory() as directory:
            model.save(directory)
            restored = LatentGeometryModel.load(directory)
            np.testing.assert_allclose(restored.encode(states[:6]), model.encode(states[:6]))
            np.testing.assert_allclose(
                restored.predict_value(model.encode(states[:6]), model.encode(states[6:12])),
                model.predict_value(model.encode(states[:6]), model.encode(states[6:12])),
            )

    def test_decoder_training_recovers_intention_direction(self):
        rng = np.random.default_rng(9)
        embeddings = rng.normal(size=(500, 4)).astype(np.float32)
        mapping = rng.normal(size=(4, 8)).astype(np.float32)
        intentions = normalize_intentions(embeddings @ mapping + 0.4, 8)
        train, validation = blocked_split(len(embeddings), seed=1, block_size=10)
        decoder, report = LatentIntentionDecoder.fit(
            embeddings,
            intentions,
            train_indices=train,
            validation_indices=validation,
            hidden_dim=32,
            epochs=55,
            patience=15,
            batch_size=64,
            seed=5,
        )
        prediction = decoder.predict(embeddings[validation])
        cosine = np.sum(prediction * intentions[validation], axis=-1) / 8
        self.assertGreater(float(cosine.mean()), 0.92)
        self.assertGreater(report.validation_r2, 0.92)
        np.testing.assert_allclose(np.linalg.norm(prediction, axis=-1), np.sqrt(8), atol=2e-6)
        with tempfile.TemporaryDirectory() as directory:
            decoder.save(directory)
            restored = LatentIntentionDecoder.load(directory)
            np.testing.assert_allclose(restored.predict(embeddings[:4]), decoder.predict(embeddings[:4]))


class PlannerTests(unittest.TestCase):
    def test_candidates_stay_inside_offline_support(self):
        planner = make_planner()
        self.assertLessEqual(len(planner.candidate_embeddings), 24)
        self.assertTrue(
            np.all(planner.candidate_support_distance <= planner.support_radius + 1e-6)
        )

    def test_exact_route_formula_matches_manual_three_switch_expression(self):
        planner = make_planner()
        observation = np.asarray([0.4, 0.6, 0.8, 0.7, 0.1, 0.2], dtype=np.float32)
        reward = np.asarray([1.0, 0.8, 0.9, 1.1], dtype=np.float32)
        route = (0, 1, 2)
        score, details = planner.score_routes(observation, reward, [route])
        goal_policy = normalize_intentions(reward, 4)
        reference = []
        for member in range(2):
            previous = observation
            discount = 1.0
            route_value = 0.0
            previous_self = None
            for position, identifier in enumerate(route):
                waypoint = planner.candidate_observations[identifier]
                intention = planner.candidate_intentions[identifier]
                forward_previous = FakeFB.forward_repr(previous[None, :], intention[None, :])[member, 0]
                forward_self = FakeFB.forward_repr(waypoint[None, :], intention[None, :])[member, 0]
                eta = np.clip(
                    np.dot(forward_previous, intention) / np.dot(forward_self, intention),
                    0,
                    1,
                )
                current_task = np.dot(forward_previous, reward)
                if position == 0:
                    route_value = float(current_task)
                else:
                    route_value += discount * (current_task - previous_self)
                discount *= float(eta)
                previous_self = float(np.dot(forward_self, reward))
                previous = waypoint
            terminal = FakeFB.forward_repr(previous[None, :], goal_policy[None, :])[member, 0]
            route_value += discount * (np.dot(terminal, reward) - previous_self)
            reference.append(route_value)
        reference = np.asarray(reference)
        expected = reference.mean() - 0.5 * (reference.max() - reference.min())
        np.testing.assert_allclose(score[0], expected, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(details["member_scores"][:, 0], reference, rtol=1e-6, atol=1e-6)

    def test_planner_builds_three_points_and_executes_first(self):
        planner = make_planner()
        observation = np.asarray([0.2, 0.3, 0.4, 0.5, 0, 0], dtype=np.float32)
        reward = np.asarray([1.0, 0.8, 1.1, 0.7], dtype=np.float32)
        selection = planner.select(observation, reward)
        self.assertFalse(selection.fallback)
        self.assertEqual(len(selection.route_indices), 3)
        np.testing.assert_allclose(
            selection.intention,
            planner.candidate_intentions[selection.route_indices[0]],
        )
        np.testing.assert_array_equal(
            selection.diagnostics["latent3_route_indices"],
            selection.route_indices,
        )

    def test_exact_b_ablation_uses_real_backward_representations(self):
        planner = make_planner(intention_mode="exact-b")
        expected = normalize_intentions(
            FakeFB.backward_repr(planner.candidate_observations),
            4,
        )
        np.testing.assert_allclose(planner.candidate_intentions, expected, atol=1e-6)

    def test_invalid_fb_denominators_fallback_safely(self):
        planner = make_planner(frozen_fb=ZeroFB())
        selection = planner.select(
            np.asarray([0.2, 0.3, 0.4, 0.5, 0, 0], dtype=np.float32),
            np.asarray([1.0, 0.8, 1.1, 0.7], dtype=np.float32),
        )
        self.assertTrue(selection.fallback)
        self.assertEqual(int(selection.diagnostics["latent3_fallback_reason"]), 2)

    def test_zero_level_rejects_routes_without_required_improvement(self):
        planner = make_planner(min_improvement=1e6)
        selection = planner.select(
            np.asarray([0.2, 0.3, 0.4, 0.5, 0, 0], dtype=np.float32),
            np.asarray([1.0, 0.8, 1.1, 0.7], dtype=np.float32),
        )
        self.assertTrue(selection.fallback)
        self.assertEqual(int(selection.diagnostics["latent3_fallback_reason"]), 3)


class ControllerTests(unittest.TestCase):
    def test_dynamic_controller_replans_and_keeps_diagnostic_keys_stable(self):
        planner = make_planner()
        controller = DynamicLatentThreeController(
            FakeFB(),
            planner,
            replan_interval=2,
            finish_radius=0.0,
            replan_on_arrival=False,
        )
        reward = np.asarray([1.0, 0.8, 1.1, 0.7], dtype=np.float32)
        states = [
            np.asarray([0.2, 0.3, 0.4, 0.5, 0, 0], dtype=np.float32),
            np.asarray([0.3, 0.4, 0.5, 0.6, 0, 0], dtype=np.float32),
            np.asarray([0.4, 0.5, 0.6, 0.7, 0, 0], dtype=np.float32),
        ]
        selections = [
            controller.select_intention(state, reward, rng=None, temperature=0)
            for state in states
        ]
        self.assertTrue(selections[0].diagnostics["latent3_replanned"])
        self.assertFalse(selections[1].diagnostics["latent3_replanned"])
        self.assertTrue(selections[2].diagnostics["latent3_replanned"])
        self.assertEqual(int(selections[2].diagnostics["latent3_replan_count"]), 2)
        self.assertEqual(set(selections[0].diagnostics), set(selections[1].diagnostics))
        self.assertEqual(set(selections[1].diagnostics), set(selections[2].diagnostics))

    def test_near_goal_uses_fixed_task_and_does_not_replan(self):
        planner = make_planner()
        controller = DynamicLatentThreeController(
            FakeFB(),
            planner,
            finish_radius=0.5,
            finish_mode="task-latent",
        )
        reward = np.asarray([1.0, 0.8, 1.1, 0.7], dtype=np.float32)
        state = np.asarray([2.8, 2.7, 0.4, 0.5, 0, 0], dtype=np.float32)
        selection = controller.select_intention(state, reward, rng=None, temperature=0)
        self.assertTrue(selection.diagnostics["latent3_finish"])
        self.assertFalse(selection.diagnostics["latent3_replanned"])
        np.testing.assert_allclose(selection.intention, normalize_intentions(reward, 4))


class CLITests(unittest.TestCase):
    def test_training_cli_defaults_are_cpu_safe(self):
        args = parse_train_args([])
        self.assertEqual(args.device, "cpu")
        self.assertGreater(args.teacher_batch_size, 0)

    def test_experiment_cli_supports_all_tasks_and_both_ablations(self):
        args = parse_run_args(
            [
                "--task-ids", "1", "2", "3", "4", "5",
                "--seeds", "0", "1", "2",
                "--intention-mode", "both",
                "--compare-baseline",
            ]
        )
        self.assertEqual(args.task_ids, [1, 2, 3, 4, 5])
        self.assertEqual(args.seeds, [0, 1, 2])
        self.assertTrue(args.compare_baseline)

    def test_paired_analysis_counts_discordant_episodes(self):
        results = {
            ("baseline", 4, 0): {"success": False, "steps": 1000},
            ("method", 4, 0): {"success": True, "steps": 200},
            ("baseline", 4, 1): {"success": True, "steps": 300},
            ("method", 4, 1): {"success": True, "steps": 250},
            ("baseline", 4, 2): {"success": True, "steps": 350},
            ("method", 4, 2): {"success": False, "steps": 1000},
        }
        report = paired_report(
            results,
            "baseline",
            "method",
            bootstrap_samples=100,
            seed=0,
        )
        self.assertEqual(report["paired_count"], 3)
        self.assertEqual(report["discordant_method_wins"], 1)
        self.assertEqual(report["discordant_baseline_wins"], 1)
        self.assertEqual(report["jointly_successful_median_step_delta"], -50.0)

    def test_full_training_command_reuses_teacher_cache_without_jax(self):
        rng = np.random.default_rng(12)
        states = rng.uniform(0.1, 2.0, size=(160, 6)).astype(np.float32)
        intentions = normalize_intentions(FakeFB.backward_repr(states), 4)
        train, validation = blocked_split(len(states), seed=2, block_size=8)
        starts = np.concatenate(
            [rng.choice(train, size=500), rng.choice(validation, size=100)]
        )
        goals = np.concatenate(
            [rng.choice(train, size=500), rng.choice(validation, size=100)]
        )
        forward = FakeFB.forward_repr(states[starts], intentions[goals])
        values = np.einsum("end,nd->en", forward, intentions[goals]).mean(axis=0)
        pair_validation = np.zeros(len(starts), dtype=bool)
        pair_validation[-100:] = True
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cache_path = root / "teacher_cache.npz"
            output_path = root / "trained"
            np.savez_compressed(
                cache_path,
                observations=states,
                source_indices=np.arange(len(states)),
                intentions=intentions,
                start_indices=starts,
                goal_indices=goals,
                teacher_values=values,
                validation_mask=pair_validation,
                train_state_indices=train,
                validation_state_indices=validation,
            )
            with contextlib.redirect_stdout(io.StringIO()):
                train_main(
                    [
                        "--cache", str(cache_path),
                        "--output-dir", str(output_path),
                        "--hidden-dim", "16",
                        "--decoder-hidden-dim", "16",
                        "--epochs", "6",
                        "--decoder-epochs", "8",
                        "--batch-size", "64",
                    ]
                )
            self.assertTrue((output_path / "geometry.npz").exists())
            self.assertTrue((output_path / "intention_decoder.npz").exists())
            self.assertTrue((output_path / "training_metrics.json").exists())
            geometry = LatentGeometryModel.load(output_path)
            decoder = LatentIntentionDecoder.load(output_path)
            planner = DynamicThreeWaypointPlanner(
                FakeFB(),
                geometry,
                decoder,
                {"observations": states},
                goal_xy=np.asarray([1.8, 1.8]),
                max_states=160,
                max_candidates=16,
                rerank_count=4,
                grid_resolution=2,
                minimum_eta=0,
                min_improvement=-1e6,
            )
            selection = planner.select(states[0], intentions[1])
            self.assertEqual(len(selection.route_indices), 3)


if __name__ == "__main__":
    unittest.main()
