"""Проверки корректности компонента checkpoint и его взаимодействия со стендом."""

from __future__ import annotations

import importlib
import pickle
from pathlib import Path

import jax.numpy as jnp
import numpy as np

np.in1d = np.isin

from baseline.frozen_fb import EXPECTED_PARAM_MODULES, FrozenFB, load_checkpoint_config
from utils.datasets import Dataset
from utils.env_utils import make_env_and_datasets


CHECKPOINT = Path("checkpoints/antmaze-medium-navigate-v0")


def _build_frozen_fb():
    config, flags = load_checkpoint_config(CHECKPOINT)
    _, train_dataset, _ = make_env_and_datasets(
        flags["env_name"],
        frame_stack=config["frame_stack"],
        add_info=True,
    )
    train_dataset = Dataset.create(**train_dataset)
    dataset_module = importlib.import_module("utils.datasets")
    dataset_class = getattr(dataset_module, config["dataset_class"])
    wrapped = dataset_class(train_dataset, config)
    example_batch = wrapped.sample(1)
    return FrozenFB.from_checkpoint(CHECKPOINT, example_batch, config=config), example_batch


def test_checkpoint_contains_expected_modules():
    with (CHECKPOINT / "params.pkl").open("rb") as f:
        state = pickle.load(f)
    modules = set(state["agent"]["network"]["params"].keys())
    assert EXPECTED_PARAM_MODULES <= modules


def test_restored_network_shapes():
    frozen_fb, example_batch = _build_frozen_fb()
    observation = example_batch["observations"][0]
    latent = jnp.ones((frozen_fb.latent_dim,))
    latent = frozen_fb.normalize_latent(latent)
    shapes = frozen_fb.validate_shapes(observation, latent)

    assert shapes["B"] == (128,)
    assert shapes["F"] == (2, 128)
    assert shapes["high_intention"] == (128,)
    assert shapes["action"] == (8,)
