"""Frozen inference wrapper around the provided FB pi-Switch checkpoint.

This module intentionally delegates all mathematical operations to the original
``FBpiSwitchAgent`` implementation.  It only adapts the T-Lab checkpoint
packaging (``flags.json`` + ``params.pkl``) to the official Flax agent.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any, Mapping

import flax
import jax
import jax.numpy as jnp
import numpy as np

from agents.fbpiswitch import FBpiSwitchAgent, get_config


EXPECTED_PARAM_MODULES = frozenset(
    {
        "modules_actor",
        "modules_backward_repr",
        "modules_forward_repr",
        "modules_high_actor",
    }
)


def load_checkpoint_config(checkpoint_dir: str | Path):
    """Reconstruct the agent config from official defaults and saved flags.

    The provided T-Lab ``flags.json`` stores only a subset of the full
    ``fbpiswitch.get_config()``.  We therefore start from the official defaults
    and overwrite only values that were actually saved.
    """

    checkpoint_dir = Path(checkpoint_dir)
    flags_path = checkpoint_dir / "flags.json"
    with flags_path.open("r", encoding="utf-8") as f:
        saved_flags = json.load(f)

    if "agent" not in saved_flags:
        raise KeyError(f"Missing 'agent' section in {flags_path}")
    if saved_flags["agent"].get("agent_name") != "fbpiswitch":
        raise ValueError(
            "Expected an fbpiswitch checkpoint, got "
            f"{saved_flags['agent'].get('agent_name')!r}"
        )

    config = get_config()
    # ConfigDict is deliberately left unlocked by the official main.py config
    # flag.  Direct assignment preserves the official defaults for fields that
    # are absent from T-Lab's compact flags.json.
    for key, value in saved_flags["agent"].items():
        config[key] = value

    return config, saved_flags


def _load_checkpoint_state(checkpoint_dir: Path) -> Mapping[str, Any]:
    params_path = checkpoint_dir / "params.pkl"
    with params_path.open("rb") as f:
        state = pickle.load(f)

    if not isinstance(state, Mapping) or "agent" not in state:
        raise ValueError(f"Unexpected checkpoint structure in {params_path}")

    try:
        modules = state["agent"]["network"]["params"]
    except (KeyError, TypeError) as exc:
        raise ValueError(
            f"Checkpoint {params_path} does not contain agent/network/params"
        ) from exc

    module_names = set(modules.keys())
    missing = EXPECTED_PARAM_MODULES - module_names
    if missing:
        raise ValueError(
            "Checkpoint is missing expected parameter modules: "
            + ", ".join(sorted(missing))
        )

    return state


class FrozenFB:
    """Stable inference API over the original frozen FB pi-Switch agent."""

    def __init__(self, agent: FBpiSwitchAgent, checkpoint_dir: Path, saved_flags: Mapping[str, Any]):
        self._agent = agent
        self.checkpoint_dir = Path(checkpoint_dir)
        self.saved_flags = saved_flags

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_dir: str | Path,
        example_batch: Mapping[str, Any],
        *,
        config=None,
    ) -> "FrozenFB":
        """Create the official agent architecture and restore T-Lab ``params.pkl``.

        We intentionally do *not* call ``load_agent_from_frozen``: T-Lab's
        ``params.pkl`` already contains F, B, low actor, and high actor.
        """

        checkpoint_dir = Path(checkpoint_dir)
        if config is None:
            config, saved_flags = load_checkpoint_config(checkpoint_dir)
        else:
            _, saved_flags = load_checkpoint_config(checkpoint_dir)

        checkpoint_seed = int(saved_flags.get("seed", 0))
        agent = FBpiSwitchAgent.create(checkpoint_seed, example_batch, config)

        state = _load_checkpoint_state(checkpoint_dir)
        agent = flax.serialization.from_state_dict(agent, state["agent"])
        return cls(agent=agent, checkpoint_dir=checkpoint_dir, saved_flags=saved_flags)

    @property
    def agent(self) -> FBpiSwitchAgent:
        """Reference agent, exposed only for equivalence tests."""

        return self._agent

    @property
    def config(self):
        return self._agent.config

    @property
    def latent_dim(self) -> int:
        return int(self.config["latent_dim"])

    def normalize_latent(self, z):
        """Use the original ``FBpiSwitchAgent.normalize_z`` implementation."""

        return self._agent.normalize_z(z)

    def backward_repr(self, observations):
        """Return B(s) without adding any new normalization."""

        return self._agent.network.select("backward_repr")(observations)

    def forward_repr(self, observations, intentions):
        """Return the raw two-member forward ensemble F(s, z).

        No ensemble reduction is performed here because the official checkpoint
        stores two forward ensemble members and different algorithms may need to
        handle them explicitly.
        """

        return self._agent.network.select("forward_repr")(
            observations,
            intentions,
            goal_encoded=True,
        )

    def infer_task_latent(self, zero_shot_batch):
        """Reference task inference; delegates exactly to ``agent.infer_latent``."""

        return self._agent.infer_latent(zero_shot_batch)

    def baseline_high_intention(
        self,
        observation,
        task_latent,
        *,
        seed,
        temperature: float = 0.0,
    ):
        """Sample the original single-intention high actor.

        Returns both the raw actor sample and the exact normalized intention that
        is passed to the low-level actor in ``FBpiSwitchAgent.sample_actions``.
        """

        dist = self._agent.network.select("high_actor")(
            observation,
            task_latent,
            goal_encoded=True,
            temperature=temperature,
        )
        raw_intention = dist.sample(seed=seed)
        intention = self._agent.normalize_z(raw_intention)
        return intention, raw_intention

    def low_action(
        self,
        observation,
        intention,
        *,
        seed,
        temperature: float = 0.0,
    ):
        """Sample the original low-level actor and apply the official clipping."""

        dist = self._agent.network.select("actor")(
            observation,
            intention,
            goal_encoded=True,
            temperature=temperature,
        )
        action = dist.sample(seed=seed)
        return jnp.clip(action, -1.0, 1.0)

    def reference_sample_actions(
        self,
        observation,
        task_latent,
        *,
        seed,
        temperature: float = 0.0,
    ):
        """Call the untouched official ``agent.sample_actions`` path."""

        return self._agent.sample_actions(
            observation,
            task_latent,
            seed=seed,
            temperature=temperature,
        )

    def validate_shapes(self, observation, task_latent) -> dict[str, tuple[int, ...]]:
        """Run lightweight inference checks against the restored checkpoint."""

        observation = jnp.asarray(observation)
        task_latent = jnp.asarray(task_latent)
        if observation.shape[-1] != 29:
            raise ValueError(f"Expected observation dim 29, got {observation.shape}")
        if task_latent.shape[-1] != self.latent_dim:
            raise ValueError(
                f"Expected latent dim {self.latent_dim}, got {task_latent.shape}"
            )

        b = self.backward_repr(observation)
        f = self.forward_repr(observation, task_latent)
        high, _ = self.baseline_high_intention(
            observation,
            task_latent,
            seed=jax.random.PRNGKey(0),
            temperature=0.0,
        )
        action = self.low_action(
            observation,
            high,
            seed=jax.random.PRNGKey(1),
            temperature=0.0,
        )
        return {
            "B": tuple(np.asarray(b).shape),
            "F": tuple(np.asarray(f).shape),
            "high_intention": tuple(np.asarray(high).shape),
            "action": tuple(np.asarray(action).shape),
        }
