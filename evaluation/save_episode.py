"""Сохранение эпизода без зависимости от конкретного контроллера."""

from pathlib import Path

import numpy as np

from evaluation.logger import EpisodeLogger
from evaluation.visualization import plot_path


def _per_step_diagnostics(result):
    """Выбирает диагностические поля, согласованные с последовательностью действий."""

    step_count = len(result.actions)
    per_step = {}
    for key, value in result.diagnostics.items():
        if key == "initial_observation_checksum":
            continue
        array = np.asarray(value)
        if array.ndim == 0 or array.shape[0] != step_count:
            raise ValueError(
                f"Diagnostic {key!r} is not aligned with {step_count} actions: "
                f"shape={array.shape}"
            )
        per_step[key] = array
    return per_step


def save_episode_result(result, output_dir, env):
    output_dir = Path(output_dir)
    logger = EpisodeLogger(output_dir)
    per_step = _per_step_diagnostics(result)

    for i in range(len(result.actions)):
        logger.add_step(
            result.observations[i],
            result.positions[i],
            result.actions[i],
            result.intentions[i],
            {key: value[i] for key, value in per_step.items()},
        )

    logger.save(
        {
            "scenario_id": result.scenario_id,
            "method": result.method,
            "task_id": result.task_id,
            "environment_seed": result.environment_seed,
            "controller_seed": result.controller_seed,
            "start_ij": result.start_ij,
            "goal_ij": result.goal_ij,
            "start_xy": result.start_xy.tolist(),
            "goal_xy": result.goal_xy.tolist(),
            "success": result.success,
            "steps": result.steps,
            "path_length": result.path_length,
            "final_distance": result.final_distance,
        }
    )

    plot_path(
        result.positions,
        result.start_xy,
        result.goal_xy,
        result.success,
        output_dir / "path.png",
        env,
    )

