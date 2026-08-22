"""Deterministic episode runner with controller-agnostic diagnostics."""

from __future__ import annotations

import random
import time
from dataclasses import dataclass
from typing import Any, Mapping

import jax
import numpy as np

from baseline.frozen_fb import FrozenFB
from controllers.base import HighLevelController
from .scenarios import Scenario


@dataclass(frozen=True)
class EpisodeResult:
    scenario_id: str
    method: str
    task_id: int | None
    environment_seed: int | None
    controller_seed: int
    start_ij: tuple[int, int] | None
    goal_ij: tuple[int, int] | None
    success: bool
    steps: int
    duration: float
    final_distance: float
    path_length: float
    start_xy: np.ndarray
    goal_xy: np.ndarray
    observations: np.ndarray
    positions: np.ndarray
    actions: np.ndarray
    intentions: np.ndarray
    diagnostics: Mapping[str, np.ndarray]


class EpisodeRunner:
    def __init__(
        self,
        env,
        frozen_fb: FrozenFB,
        controller: HighLevelController,
        *,
        eval_temperature: float = 0.0,
    ):
        self.env = env
        self.frozen_fb = frozen_fb
        self.controller = controller
        self.eval_temperature = float(eval_temperature)

    def run(self, scenario: Scenario, task_latent) -> EpisodeResult:
        self.controller.reset(scenario)

        if scenario.environment_seed is not None:
            seed = int(scenario.environment_seed)
            random.seed(seed)
            np.random.seed(seed)
            action_space = getattr(self.env, "action_space", None)
            if action_space is not None and hasattr(action_space, "seed"):
                action_space.seed(seed)

        reset_kwargs: dict[str, Any] = {"options": scenario.reset_options()}
        if scenario.environment_seed is not None:
            reset_kwargs["seed"] = int(scenario.environment_seed)
        observation, info = self.env.reset(**reset_kwargs)

        goal_xy = np.asarray(
            self.env.unwrapped.cur_goal_xy,
            dtype=np.float64,
        ).copy()
        start_xy = np.asarray(observation[:2], dtype=np.float64).copy()
        policy_rng = jax.random.PRNGKey(int(scenario.controller_seed))

        observations = []
        positions = []
        actions = []
        intentions = []
        diagnostic_values: dict[str, list[np.ndarray]] = {}
        expected_diagnostic_keys: set[str] | None = None

        terminated = False
        truncated = False
        final_info = info
        started_at = time.perf_counter()

        while not (terminated or truncated):
            policy_rng, step_key = jax.random.split(policy_rng)
            high_key, low_key = jax.random.split(step_key)

            selection = self.controller.select_intention(
                observation,
                task_latent,
                rng=high_key,
                temperature=self.eval_temperature,
            )
            action = np.asarray(
                self.frozen_fb.low_action(
                    observation,
                    selection.intention,
                    seed=low_key,
                    temperature=self.eval_temperature,
                )
            )

            observations.append(np.asarray(observation).copy())
            positions.append(np.asarray(observation[:2], dtype=np.float64).copy())
            actions.append(action.copy())
            intentions.append(np.asarray(selection.intention).copy())

            step_diagnostics = dict(selection.diagnostics)
            keys = set(step_diagnostics)
            if expected_diagnostic_keys is None:
                expected_diagnostic_keys = keys
                diagnostic_values = {key: [] for key in sorted(keys)}
            elif keys != expected_diagnostic_keys:
                missing = sorted(expected_diagnostic_keys - keys)
                added = sorted(keys - expected_diagnostic_keys)
                raise ValueError(
                    "Controller diagnostic keys changed within an episode: "
                    f"missing={missing}, added={added}"
                )
            for key, value in step_diagnostics.items():
                diagnostic_values[key].append(np.asarray(value).copy())

            observation, _, terminated, truncated, final_info = self.env.step(action)

        duration = time.perf_counter() - started_at
        final_xy = np.asarray(observation[:2], dtype=np.float64)
        final_distance = float(np.linalg.norm(final_xy - goal_xy))

        if positions:
            path_points = np.concatenate(
                [np.asarray(positions, dtype=np.float64), final_xy[None, :]],
                axis=0,
            )
            path_length = float(
                np.linalg.norm(np.diff(path_points, axis=0), axis=-1).sum()
            )
        else:
            path_length = 0.0

        diagnostics = {
            "initial_observation_checksum": (
                np.asarray(
                    [
                        np.sum(observations[0]),
                        np.linalg.norm(observations[0]),
                    ]
                )
                if observations
                else np.zeros(2)
            )
        }
        diagnostics.update(
            {key: np.asarray(values) for key, values in diagnostic_values.items()}
        )

        return EpisodeResult(
            scenario_id=scenario.scenario_id,
            method=self.controller.method_name,
            task_id=scenario.task_id,
            environment_seed=scenario.environment_seed,
            controller_seed=scenario.controller_seed,
            start_ij=scenario.start_ij,
            goal_ij=scenario.goal_ij,
            success=bool(final_info.get("success", False)),
            steps=len(actions),
            duration=duration,
            final_distance=final_distance,
            path_length=path_length,
            start_xy=start_xy,
            goal_xy=goal_xy,
            observations=np.asarray(observations),
            positions=np.asarray(positions),
            actions=np.asarray(actions),
            intentions=np.asarray(intentions),
            diagnostics=diagnostics,
        )

