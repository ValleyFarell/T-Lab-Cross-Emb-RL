"""Controller-agnostic episode execution loop."""

from __future__ import annotations

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
    environment_seed: int | None
    controller_seed: int
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
    """Execute one episode without knowing how the controller plans."""

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

        reset_kwargs: dict[str, Any] = {"options": {"task_id": int(scenario.task_id)}}
        if scenario.environment_seed is not None:
            reset_kwargs["seed"] = int(scenario.environment_seed)
        observation, info = self.env.reset(**reset_kwargs)

        goal_xy = np.asarray(self.env.unwrapped.cur_goal_xy, dtype=np.float64).copy()
        start_xy = np.asarray(observation[:2], dtype=np.float64).copy()

        # Reproduce utils.evaluation.supply_rng: first split the persistent RNG
        # to obtain a per-step key, then reproduce FBpiSwitchAgent.sample_actions'
        # split into high- and low-level keys.
        policy_rng = jax.random.PRNGKey(int(scenario.controller_seed))

        observations = []
        positions = []
        actions = []
        intentions = []
        raw_high_outputs = []

        terminated = truncated = False
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
            action = self.frozen_fb.low_action(
                observation,
                selection.intention,
                seed=low_key,
                temperature=self.eval_temperature,
            )
            action = np.asarray(action)

            observations.append(np.asarray(observation).copy())
            positions.append(np.asarray(observation[:2], dtype=np.float64).copy())
            actions.append(action.copy())
            intentions.append(np.asarray(selection.intention).copy())
            if "raw_high_actor_output" in selection.diagnostics:
                raw_high_outputs.append(
                    np.asarray(selection.diagnostics["raw_high_actor_output"]).copy()
                )

            observation, _, terminated, truncated, final_info = self.env.step(action)

        duration = time.perf_counter() - started_at
        final_xy = np.asarray(observation[:2], dtype=np.float64)
        final_distance = float(np.linalg.norm(final_xy - goal_xy))
        if positions:
            path_points = np.concatenate(
                [np.asarray(positions, dtype=np.float64), final_xy[None, :]], axis=0
            )
            path_length = float(
                np.linalg.norm(np.diff(path_points, axis=0), axis=-1).sum()
            )
        else:
            path_length = 0.0
        success = bool(final_info.get("success", False))

        diagnostics: dict[str, np.ndarray] = {}
        if raw_high_outputs:
            diagnostics["raw_high_actor_output"] = np.asarray(raw_high_outputs)

        return EpisodeResult(
            scenario_id=scenario.scenario_id,
            method=self.controller.method_name,
            environment_seed=scenario.environment_seed,
            controller_seed=scenario.controller_seed,
            success=success,
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
