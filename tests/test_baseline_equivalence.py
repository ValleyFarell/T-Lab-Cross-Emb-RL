"""Проверки корректности компонента baseline equivalence и его взаимодействия со стендом."""

from __future__ import annotations

import importlib
from pathlib import Path

import jax
import numpy as np

np.in1d = np.isin

from baseline.frozen_fb import FrozenFB, load_checkpoint_config
from baseline.task_encoder import TaskEncoder
from controllers.baseline import BaselineController
from utils.datasets import Dataset
from utils.env_utils import make_env_and_datasets


CHECKPOINT = Path("checkpoints/antmaze-medium-navigate-v0")


def _setup():
    config, flags = load_checkpoint_config(CHECKPOINT)
    env, train_dataset, val_dataset = make_env_and_datasets(
        flags["env_name"],
        frame_stack=config["frame_stack"],
        add_info=True,
    )
    env.unwrapped._add_noise_to_goal = False

    train_dataset = Dataset.create(**train_dataset)
    val_dataset = Dataset.create(**val_dataset)
    dataset_module = importlib.import_module("utils.datasets")
    dataset_class = getattr(dataset_module, config["dataset_class"])
    wrapped_train = dataset_class(train_dataset, config)
    frozen_fb = FrozenFB.from_checkpoint(
        CHECKPOINT,
        wrapped_train.sample(1),
        config=config,
    )
    task = TaskEncoder(
        frozen_fb,
        env,
        val_dataset,
        env_name=flags["env_name"],
    ).encode_standard_task(1)
    return frozen_fb, BaselineController(frozen_fb), train_dataset, task.latent


def test_temperature_zero_matches_official_sample_actions():
    frozen_fb, controller, train_dataset, task_latent = _setup()
    # Берём равномерно распределённые состояния без случайности Dataset.sample.
    idxs = np.linspace(0, train_dataset.size - 1, num=16, dtype=np.int64)
    observations = np.asarray(train_dataset["observations"])[idxs]

    policy_rng = jax.random.PRNGKey(12345)
    for observation in observations:
        policy_rng, step_key = jax.random.split(policy_rng)

        reference = np.asarray(
            frozen_fb.reference_sample_actions(
                observation,
                task_latent,
                seed=step_key,
                temperature=0.0,
            )
        )

        high_key, low_key = jax.random.split(step_key)
        selection = controller.select_intention(
            observation,
            task_latent,
            rng=high_key,
            temperature=0.0,
        )
        candidate = np.asarray(
            frozen_fb.low_action(
                observation,
                selection.intention,
                seed=low_key,
                temperature=0.0,
            )
        )

        np.testing.assert_allclose(candidate, reference, rtol=1e-6, atol=1e-6)
