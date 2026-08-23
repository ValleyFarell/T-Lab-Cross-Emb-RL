"""Проверки корректности компонента value state factors и его взаимодействия со стендом."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.run_value_state_factors import (
    FactorModel,
    PairData,
    load_cached_splits,
    main,
    regression_metrics,
    sigmoid,
    split_observation_indices,
    synthetic_splits,
)


class FactorExperimentTests(unittest.TestCase):
    def test_pair_shape_validation(self):
        with self.assertRaises(ValueError):
            PairData(np.zeros((3, 29)), np.zeros((4, 29)), np.zeros(3))

    def test_sigmoid_handles_extremes(self):
        values = sigmoid(np.asarray([-1e6, 0.0, 1e6]))
        self.assertTrue(np.isfinite(values).all())
        self.assertAlmostEqual(float(values[1]), 0.5)

    def test_factor_gates_do_not_receive_coordinates(self):
        data = synthetic_splits(0, 500)["train"]
        model = FactorModel("xy_both", data, seed=0, hidden=16, quality_hidden=8, gate_regularization=0.0)
        self.assertEqual(model.main.weights[0].shape[0], 4)
        self.assertEqual(model.start.weights[0].shape[0], 27)
        self.assertEqual(model.goal.weights[0].shape[0], 27)

    def test_bad_quality_reduces_shifted_prediction_even_when_value_is_negative(self):
        data = synthetic_splits(0, 500)["train"]
        model = FactorModel("xy_both", data, seed=0, hidden=16, quality_hidden=8, gate_regularization=0.0)
        model.start.weights[-1][:] = 0.0
        model.start.biases[-1][:] = -6.0
        bad, _ = model.predict(data.take(np.arange(8)))
        model.start.biases[-1][:] = 6.0
        good, _ = model.predict(data.take(np.arange(8)))
        self.assertTrue(np.all(good > bad))

    def test_trajectory_splits_do_not_overlap(self):
        terminals = np.zeros(100, dtype=bool)
        terminals[9::10] = True
        splits = split_observation_indices(100, terminals, np.random.default_rng(3))
        groups = {name: set((indices // 10).tolist()) for name, indices in splits.items()}
        self.assertFalse(groups["train"] & groups["val"])
        self.assertFalse(groups["train"] & groups["test"])
        self.assertFalse(groups["val"] & groups["test"])

    def test_cached_pairs_round_trip(self):
        splits = synthetic_splits(1, 100)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pairs.npz"
            arrays = {}
            for name, data in splits.items():
                arrays.update({f"{name}_starts": data.starts, f"{name}_goals": data.goals, f"{name}_values": data.values, f"{name}_start_ids": data.start_ids})
            np.savez(path, **arrays)
            loaded = load_cached_splits(Path(directory))
            for name in splits:
                np.testing.assert_allclose(loaded[name].values, splits[name].values)

    def test_regression_metrics(self):
        result = regression_metrics(np.asarray([1, 2, 3]), np.asarray([1, 2, 3]))
        self.assertAlmostEqual(result["r2"], 1.0)

    def test_end_to_end_factor_beats_xy_on_factorized_synthetic_data(self):
        with tempfile.TemporaryDirectory() as directory:
            result = main(["--synthetic", "--models", "xy", "xy_both", "--model-seeds", "0", "--train-pairs", "4000", "--epochs", "35", "--patience", "12", "--hidden-dim", "32", "--quality-hidden-dim", "24", "--output-dir", directory])
            self.assertGreater(result["aggregate"]["xy_both"]["test_r2_mean"], result["aggregate"]["xy"]["test_r2_mean"] + 0.15)
            self.assertTrue((Path(directory) / "metrics.json").exists())


if __name__ == "__main__":
    unittest.main()
