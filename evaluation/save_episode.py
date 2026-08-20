from pathlib import Path

from evaluation.logger import EpisodeLogger
from evaluation.visualization import plot_path


def save_episode_result(result, output_dir, env):
    output_dir = Path(output_dir)

    logger = EpisodeLogger(output_dir)

    raw = result.diagnostics.get(
        "raw_high_actor_output",
        None,
    )

    for i in range(len(result.actions)):
        diagnostics = {}

        if raw is not None:
            diagnostics["raw_high_actor_output"] = raw[i]

        logger.add_step(
            result.observations[i],
            result.positions[i],
            result.actions[i],
            result.intentions[i],
            diagnostics,
        )

    logger.save(
        {
            "scenario_id": result.scenario_id,
            "method": result.method,
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
