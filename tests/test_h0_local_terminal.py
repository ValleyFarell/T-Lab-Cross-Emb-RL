"""Проверки корректности компонента h0 local terminal и его взаимодействия со стендом."""

from __future__ import annotations

import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock

import numpy as np

from hypotheses.h0_local_terminal import LocalTwoSwitchPlanner
from scripts import run_h0_local_terminal, summarize_h0_local_terminal


class FakeFrozenFB:
    latent_dim = 2

    def __init__(self):
        self.forward_batch_sizes = []
        self.high_calls = 0

    @staticmethod
    def normalize_latent(latent):
        latent = np.asarray(latent, dtype=np.float64)
        norm = np.linalg.norm(latent, axis=-1, keepdims=True)
        return latent / np.maximum(norm, 1e-8) * np.sqrt(latent.shape[-1])

    @staticmethod
    def backward_repr(observations):
        observations = np.asarray(observations, dtype=np.float64)
        return observations[..., :2] + np.asarray([2.0, 3.0])

    def forward_repr(self, observations, intentions):
        observations = np.asarray(observations, dtype=np.float64)
        intentions = np.asarray(intentions, dtype=np.float64)
        self.forward_batch_sizes.append(len(observations))
        first = observations[..., :2] + 0.25 * intentions + 0.5
        return np.stack([first, first + 0.2], axis=0)

    def baseline_high_intention(self, observation, task_latent, *, seed, temperature):
        del observation, task_latent, seed, temperature
        self.high_calls += 1
        intention = np.asarray([0.4, 1.3], dtype=np.float32)
        return intention, intention.copy()


def _dataset():
    observations = np.asarray(
        [
            [0.10, 0.20, 1.0],
            [-0.05, 0.04, 2.0],
            [0.70, -0.30, 3.0],
            [3.95, 0.05, 4.0],
            [4.50, 0.20, 5.0],
            [3.30, -0.10, 6.0],
            [8.10, 0.20, 7.0],
            [7.95, 0.05, 8.0],
            [8.40, -0.50, 9.0],
            [4.10, 4.10, 10.0],
            [3.95, 4.02, 11.0],
            [4.60, 3.80, 12.0],
        ],
        dtype=np.float32,
    )
    return {"observations": observations, "qpos": observations.copy()}


def _planner(frozen_fb=None, **kwargs):
    config = {
        "goal_xy": [8.0, 0.0],
        "candidates_per_cell": 2,
        "grid_cell_size": 4.0,
        "local_radius": 2.0,
        "max_local_candidates": 4,
        "finish_radius": 1.0,
        "pair_batch_size": 3,
    }
    config.update(kwargs)
    return LocalTwoSwitchPlanner(frozen_fb or FakeFrozenFB(), _dataset(), **config)


def _load_controller_class():
    """Загружает нужный контроллер без импорта необязательных соседних зависимостей."""

    root = Path(__file__).resolve().parents[1] / "controllers"
    package_name = "h0_local_terminal_isolated_controllers"
    if package_name not in sys.modules:
        package = types.ModuleType(package_name)
        package.__path__ = [str(root)]
        sys.modules[package_name] = package
    for name in ("base", "h0_local_terminal"):
        full_name = f"{package_name}.{name}"
        if full_name not in sys.modules:
            spec = importlib.util.spec_from_file_location(full_name, root / f"{name}.py")
            module = importlib.util.module_from_spec(spec)
            sys.modules[full_name] = module
            spec.loader.exec_module(module)
    return sys.modules[f"{package_name}.h0_local_terminal"].LocalTerminalController


class LocalTerminalPlannerTests(unittest.TestCase):
    def test_spatial_stratification_is_balanced_and_keeps_cell_centers(self):
        planner = _planner()
        self.assertEqual(planner.candidate_count, 8)
        np.testing.assert_array_equal(planner.per_cell_counts, [2, 2, 2, 2])
        for cell in planner.cell_coordinates:
            center = cell * planner.grid_cell_size
            expected = np.argmin(np.linalg.norm(_dataset()["qpos"][:, :2] - center, axis=1))
            self.assertIn(expected, planner.source_indices)

    def test_candidate_pool_is_deterministic(self):
        first = _planner()
        second = _planner()
        np.testing.assert_array_equal(first.source_indices, second.source_indices)
        self.assertEqual(first.candidate_checksum, second.candidate_checksum)

    def test_first_subgoal_is_local_but_second_can_use_all_cells(self):
        planner = _planner(enable_zero_level=False, finish_radius=0.0)
        selection = planner.select([0.0, 0.0, 1.0], [0.8, 1.2])
        self.assertEqual(selection.diagnostics["h0lt_selected_depth"], 2)
        self.assertLessEqual(
            selection.diagnostics["h0lt_selected_w1_distance"], planner.local_radius
        )
        self.assertGreater(
            selection.diagnostics["h0lt_global_candidate_count"],
            selection.diagnostics["h0lt_local_candidate_count"],
        )

    def test_two_switch_score_matches_the_original_h0_equation(self):
        frozen = FakeFrozenFB()
        planner = _planner(
            frozen,
            local_radius=100.0,
            max_local_candidates=20,
            finish_radius=0.0,
            direct_latent_mode="normalized",
            enable_zero_level=False,
        )
        state = np.asarray([0.3, 0.4, 1.0], dtype=np.float64)
        reward = np.asarray([0.8, 1.2], dtype=np.float64)
        goal_policy = frozen.normalize_latent(reward)

        def forward(observation, intention):
            value = frozen.forward_repr(
                np.asarray(observation)[None, :], np.asarray(intention)[None, :]
            )
            return np.asarray(value).mean(axis=0)[0]

        expected = np.full((planner.candidate_count, planner.candidate_count), -np.inf)
        for i, (first_state, first_latent) in enumerate(
            zip(planner.candidates, planner.candidate_latents)
        ):
            current_forward = forward(state, first_latent)
            first_self = forward(first_state, first_latent)
            eta1 = np.clip(
                (current_forward @ first_latent) / (first_self @ first_latent), 0, 1
            )
            for j, (second_state, second_latent) in enumerate(
                zip(planner.candidates, planner.candidate_latents)
            ):
                pair_forward = forward(first_state, second_latent)
                second_self = forward(second_state, second_latent)
                eta2 = np.clip(
                    (pair_forward @ second_latent) / (second_self @ second_latent), 0, 1
                )
                second_goal = forward(second_state, goal_policy)
                expected[i, j] = (
                    current_forward @ reward
                    + eta1 * (pair_forward @ reward - first_self @ reward)
                    + eta1 * eta2 * (second_goal @ reward - second_self @ reward)
                )
        selection = planner.select(state, reward)
        chosen_i = int(selection.diagnostics["w1_index"])
        chosen_j = int(selection.diagnostics["w2_index"])
        # Несколько тестовых кандидатов математически равны; формат float32 может
        # изменить порядок максимума без изменения самой целевой величины.
        self.assertAlmostEqual(expected[chosen_i, chosen_j], np.max(expected), places=5)
        self.assertAlmostEqual(
            selection.diagnostics["h0lt_best_two_value"],
            expected[chosen_i, chosen_j],
            places=5,
        )
    def test_zero_level_executes_raw_task_when_margin_dominates(self):
        planner = _planner(switch_margin=1_000_000.0)
        reward = np.asarray([0.8, 1.2], dtype=np.float32)
        selection = planner.select([0.0, 0.0, 1.0], reward)
        self.assertEqual(selection.diagnostics["h0lt_mode"], planner.MODE_DIRECT)
        self.assertEqual(selection.diagnostics["w1_index"], -1)
        np.testing.assert_array_equal(selection.intention, reward)

    def test_finish_direct_is_immediate_and_needs_no_pair_rows(self):
        planner = _planner()
        reward = np.asarray([0.8, 1.2], dtype=np.float32)
        selection = planner.select([7.8, 0.1, 1.0], reward)
        self.assertEqual(selection.diagnostics["h0lt_mode"], planner.MODE_TERMINAL_DIRECT)
        self.assertTrue(selection.diagnostics["h0lt_terminal_active"])
        self.assertEqual(selection.diagnostics["h0lt_cached_pair_rows"], 0)
        np.testing.assert_array_equal(selection.intention, reward)

    def test_normalized_direct_mode_is_available(self):
        planner = _planner(direct_latent_mode="normalized")
        selection = planner.select([7.8, 0.1, 1.0], [0.3, 0.4])
        self.assertAlmostEqual(np.linalg.norm(selection.intention), np.sqrt(2), places=5)

    def test_baseline_finish_requires_and_uses_actor_intention(self):
        planner = _planner(finish_mode="baseline")
        with self.assertRaisesRegex(ValueError, "baseline_intention"):
            planner.select([7.8, 0.1, 1.0], [0.8, 1.2])
        intention = np.asarray([0.4, 1.3], dtype=np.float32)
        selection = planner.select(
            [7.8, 0.1, 1.0], [0.8, 1.2], baseline_intention=intention
        )
        self.assertEqual(selection.diagnostics["h0lt_mode"], planner.MODE_TERMINAL_BASELINE)
        np.testing.assert_array_equal(selection.intention, intention)

    def test_pair_rows_are_cached_between_replans(self):
        frozen = FakeFrozenFB()
        planner = _planner(frozen, finish_radius=0.0)
        reward = [0.8, 1.2]
        planner.select([0.0, 0.0, 1.0], reward)
        first_cached = int(planner._row_cached.sum())
        calls_after_first = len(frozen.forward_batch_sizes)
        planner.select([0.0, 0.0, 1.0], reward)
        self.assertEqual(int(planner._row_cached.sum()), first_cached)
        # Повторно вычисляются только прямое намерение и локальные намерения текущего состояния.
        self.assertEqual(len(frozen.forward_batch_sizes) - calls_after_first, 2)

    def test_pair_cache_is_reset_when_reward_changes(self):
        planner = _planner(finish_radius=0.0)
        planner.select([0.0, 0.0, 1.0], [0.8, 1.2])
        old_key = planner._goal_key
        planner.select([0.0, 0.0, 1.0], [1.2, 0.8])
        self.assertNotEqual(planner._goal_key, old_key)

    def test_diagnostic_schema_is_identical_in_all_modes(self):
        planner = _planner(enable_zero_level=False)
        far = planner.select([0.0, 0.0, 1.0], [0.8, 1.2])
        near = planner.select([7.8, 0.1, 1.0], [0.8, 1.2])
        self.assertEqual(set(far.diagnostics), set(near.diagnostics))

    def test_empty_local_ball_uses_the_nearest_state(self):
        planner = _planner(local_radius=0.001, enable_zero_level=False, finish_radius=0.0)
        selection = planner.select([2.0, 2.0, 1.0], [0.8, 1.2])
        self.assertTrue(selection.diagnostics["h0lt_local_used_fallback"])
        self.assertEqual(selection.diagnostics["h0lt_local_candidate_count"], 1)

    def test_invalid_parameters_are_rejected(self):
        invalid = (
            {"candidates_per_cell": 0},
            {"grid_cell_size": 0.0},
            {"local_radius": -1.0},
            {"max_local_candidates": 0},
            {"finish_radius": -1.0},
            {"pair_batch_size": 0},
            {"eta_epsilon": 0.0},
            {"switch_margin": -1.0},
            {"finish_mode": "other"},
            {"direct_latent_mode": "other"},
        )
        for kwargs in invalid:
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    _planner(**kwargs)

    def test_stratification_can_use_observation_xy_without_qpos(self):
        dataset = {"observations": _dataset()["observations"]}
        planner = LocalTwoSwitchPlanner(
            FakeFrozenFB(), dataset, [8.0, 0.0], candidates_per_cell=2
        )
        self.assertIn("observations", planner.xy_source)


class LocalTerminalControllerTests(unittest.TestCase):
    def test_terminal_mode_interrupts_cached_plan_and_latches(self):
        frozen = FakeFrozenFB()
        planner = _planner(frozen, enable_zero_level=False)
        Controller = _load_controller_class()
        controller = Controller(frozen, planner, replan_interval=10, latch_finish=True)
        far = controller.select_intention(
            [0.0, 0.0, 1.0], [0.8, 1.2], rng=None, temperature=0.0
        )
        near = controller.select_intention(
            [7.8, 0.1, 1.0], [0.8, 1.2], rng=None, temperature=0.0
        )
        escaped = controller.select_intention(
            [6.0, 0.0, 1.0], [0.8, 1.2], rng=None, temperature=0.0
        )
        self.assertFalse(far.diagnostics["h0lt_terminal_active"])
        self.assertTrue(near.diagnostics["h0lt_terminal_active"])
        self.assertTrue(near.diagnostics["h0lt_replanned"])
        self.assertTrue(escaped.diagnostics["h0lt_finish_latched"])
        self.assertTrue(escaped.diagnostics["h0lt_terminal_active"])

    def test_baseline_finish_calls_frozen_high_actor(self):
        frozen = FakeFrozenFB()
        planner = _planner(frozen, finish_mode="baseline")
        Controller = _load_controller_class()
        controller = Controller(frozen, planner)
        selection = controller.select_intention(
            [7.9, 0.0, 1.0], [0.8, 1.2], rng=7, temperature=0.0
        )
        self.assertEqual(frozen.high_calls, 1)
        self.assertEqual(selection.diagnostics["h0lt_mode"], planner.MODE_TERMINAL_BASELINE)


class LauncherTests(unittest.TestCase):
    def test_extension_options_are_removed_before_shared_parser(self):
        extension, shared = run_h0_local_terminal.parse_extension_args(
            [
                "--task-id",
                "4",
                "--candidates-per-cell",
                "7",
                "--finish-radius",
                "3",
                "--pair-batch-size",
                "256",
            ]
        )
        self.assertEqual(extension.candidates_per_cell, 7)
        self.assertEqual(extension.finish_radius, 3.0)
        self.assertEqual(extension.pair_batch_size, 256)
        self.assertEqual(shared, ["--task-id", "4"])

    def test_explicit_controller_is_rejected(self):
        with mock.patch("sys.stderr", new_callable=io.StringIO):
            with self.assertRaises(SystemExit):
                run_h0_local_terminal.parse_extension_args(
                    ["--controller", "baseline"]
                )

    def test_launcher_restores_the_original_shared_dispatcher(self):
        fake_shared = types.ModuleType("scripts.run_baseline")

        def original_dispatch(*args):
            return args

        fake_shared.build_controller = original_dispatch
        fake_shared.main = mock.Mock(return_value="finished")
        import scripts

        with mock.patch.object(scripts, "run_baseline", fake_shared, create=True):
            result = run_h0_local_terminal.main(["--task-id", "4"])
        self.assertEqual(result, "finished")
        fake_shared.main.assert_called_once_with(["--controller", "h0", "--task-id", "4"])
        self.assertIs(fake_shared.build_controller, original_dispatch)

    def test_launcher_builds_the_addon_through_existing_h0_dispatch(self):
        fake_shared = types.ModuleType("scripts.run_baseline")
        original_dispatch = mock.Mock()
        fake_shared.build_controller = original_dispatch
        shared_args = types.SimpleNamespace(
            controller="h0",
            pair_batch_size=512,
            eta_epsilon=1e-5,
            h0_replan_interval=3,
        )
        fake_shared.main = lambda argv: fake_shared.build_controller(
            shared_args, "frozen", "dataset", [4.0, 4.0]
        )
        import scripts

        with mock.patch.object(scripts, "run_baseline", fake_shared, create=True):
            with mock.patch.object(
                run_h0_local_terminal,
                "make_h0_local_terminal_controller",
                return_value="new-controller",
            ) as factory:
                result = run_h0_local_terminal.main(
                    [
                        "--task-id",
                        "4",
                        "--candidates-per-cell",
                        "7",
                        "--pair-batch-size",
                        "256",
                    ]
                )
        self.assertEqual(result, ("new-controller", "h0_local_terminal"))
        self.assertEqual(factory.call_args.kwargs["candidates_per_cell"], 7)
        self.assertEqual(factory.call_args.kwargs["pair_batch_size"], 256)
        self.assertEqual(factory.call_args.kwargs["replan_interval"], 3)
        self.assertIs(fake_shared.build_controller, original_dispatch)


class DiagnosticSummaryTests(unittest.TestCase):
    def test_summary_reads_controller_modes_and_detects_falls(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            run = root / "task" / "runs" / "000001"
            run.mkdir(parents=True)
            with (run / "summary.json").open("w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "task_id": 4,
                        "environment_seed": 3,
                        "success": True,
                        "steps": 3,
                        "path_length": 4.5,
                        "final_distance": 0.2,
                    },
                    handle,
                )
            np.savez(
                run / "trajectory.npz",
                observations=np.asarray(
                    [[0, 0, 0.55], [1, 0, 0.29], [2, 0, 0.55]]
                ),
                diagnostic_h0lt_mode=np.asarray([1, 0, 2]),
                diagnostic_h0lt_goal_distance=np.asarray([4.0, 2.0, 1.0]),
                diagnostic_h0lt_local_candidate_count=np.asarray([5, 5, 0]),
                diagnostic_selected_eta1=np.asarray([1.0, np.nan, np.nan]),
            )
            rows = summarize_h0_local_terminal.collect(root)
            self.assertEqual(len(rows), 1)
            self.assertTrue(rows[0]["fell_below_0_3"])
            self.assertEqual(rows[0]["direct_zero_steps"], 1)
            self.assertEqual(rows[0]["terminal_direct_steps"], 1)
            with mock.patch("sys.stdout", new_callable=io.StringIO):
                result = summarize_h0_local_terminal.main(
                    ["--results-dir", str(root)]
                )
            self.assertEqual(result["overall"]["success_rate"], 1.0)
            self.assertTrue((root / "h0_local_terminal_diagnostics.json").exists())
            self.assertTrue((root / "h0_local_terminal_runs.csv").exists())


if __name__ == "__main__":
    unittest.main()
