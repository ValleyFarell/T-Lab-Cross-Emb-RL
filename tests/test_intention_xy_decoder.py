"""Проверки корректности компонента intention xy decoder и его взаимодействия со стендом."""

import numpy as np

from probes.intention_xy import (
    IntentionXYDecoder,
    fit_decoder,
    regression_metrics,
    split_dataset_indices,
)


def test_decoder_has_requested_architecture_and_roundtrips(tmp_path):
    rng = np.random.default_rng(0)
    z = np.zeros((256, 128), dtype=np.float32)
    z[:, :4] = rng.normal(size=(256, 4)).astype(np.float32)
    xy = np.stack((z[:, 0] + 0.5 * z[:, 1], z[:, 2] - z[:, 3]), axis=1)
    decoder, summary = fit_decoder(
        z[:192],
        xy[:192],
        z[192:224],
        xy[192:224],
        seed=0,
        hidden_dims=(32, 16),
        batch_size=64,
        max_epochs=80,
        patience=15,
        warmup_epochs=2,
    )
    assert decoder.architecture == (128, 32, 16, 2)
    assert decoder.params["w1"].shape == (128, 32)
    assert decoder.params["w2"].shape == (32, 16)
    assert decoder.params["w3"].shape == (16, 2)
    assert decoder.params["ln_scale1"].shape == (32,)
    assert decoder.params["ln_bias1"].shape == (32,)
    assert summary["best_epoch"] > 0

    decoder.save(tmp_path, metadata={"test": True})
    restored = IntentionXYDecoder.load(tmp_path)
    np.testing.assert_allclose(
        np.asarray(restored.predict(z[224:])),
        np.asarray(decoder.predict(z[224:])),
        rtol=1e-6,
        atol=1e-6,
    )
    metrics = regression_metrics(restored, z[224:], xy[224:])
    assert metrics["rmse_xy"] < 1.0
    assert "rmse_euclidean" in metrics


def test_trajectory_split_has_no_group_leakage():
    terminals = np.zeros(30, dtype=np.float32)
    terminals[[4, 9, 14, 19, 24, 29]] = 1
    splits, strategy = split_dataset_indices(
        30,
        terminals=terminals,
        seed=7,
        max_samples=None,
    )
    assert strategy == "trajectory"
    trajectory_id = np.concatenate(
        [np.zeros(1, dtype=np.int64), np.cumsum(terminals[:-1] > 0)]
    )
    train_groups = set(trajectory_id[splits["train"]])
    validation_groups = set(trajectory_id[splits["validation"]])
    test_groups = set(trajectory_id[splits["test"]])
    assert train_groups.isdisjoint(validation_groups)
    assert train_groups.isdisjoint(test_groups)
    assert validation_groups.isdisjoint(test_groups)


def test_loader_remains_compatible_with_one_hidden_layer_decoder(tmp_path):
    legacy = IntentionXYDecoder(
        params={
            "w1": np.zeros((128, 20), dtype=np.float32),
            "b1": np.zeros(20, dtype=np.float32),
            "w2": np.zeros((20, 2), dtype=np.float32),
            "b2": np.zeros(2, dtype=np.float32),
        },
        input_mean=np.zeros(128, dtype=np.float32),
        input_scale=np.ones(128, dtype=np.float32),
        target_mean=np.zeros(2, dtype=np.float32),
        target_scale=np.ones(2, dtype=np.float32),
    )
    legacy.save(tmp_path, metadata={"architecture": [128, 20, 2]})
    restored = IntentionXYDecoder.load(tmp_path)
    assert restored.architecture == (128, 20, 2)
    np.testing.assert_array_equal(
        np.asarray(restored.predict(np.zeros(128, dtype=np.float32))),
        np.zeros(2, dtype=np.float32),
    )
