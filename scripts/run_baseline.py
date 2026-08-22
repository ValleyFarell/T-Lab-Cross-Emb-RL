"""Run one standard OGBench episode through the baseline runtime and save artifacts."""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path

import numpy as np

np.in1d = np.isin

from baseline.frozen_fb import FrozenFB, load_checkpoint_config
from baseline.task_encoder import TaskEncoder
from controllers.baseline import BaselineController
from controllers.direct_goal import DirectGoalController
from scripts.run_h0 import make_h0_controller
from evaluation.runner import EpisodeRunner
from evaluation.save_episode import save_episode_result
from evaluation.scenarios import Scenario, xy_to_free_grid_cell
from utils.datasets import Dataset
from utils.env_utils import make_env_and_datasets


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("checkpoints/antmaze-medium-navigate-v0"),
    )
    parser.add_argument(
        "--task-id",
        type=int,
        default=None,
        help="Official OGBench task id (default: 1 unless custom coordinates are used).",
    )
    parser.add_argument(
        "--start-xy",
        nargs=2,
        type=float,
        metavar=("X", "Y"),
        help="Nominal start at the center of a free maze cell.",
    )
    parser.add_argument(
        "--goal-xy",
        nargs=2,
        type=float,
        metavar=("X", "Y"),
        help="Goal at the center of a free maze cell.",
    )
    parser.add_argument(
        "--environment-seed",
        type=int,
        default=0,
    )
    parser.add_argument("--controller-seed", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--controller",
        choices=("baseline", "direct", "h0"),
        default="baseline",
        help="baseline uses high_actor; direct sends the task latent to low_actor.",
    )

    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results"),
    )

    args = parser.parse_args()

    has_start = args.start_xy is not None
    has_goal = args.goal_xy is not None
    if has_start != has_goal:
        parser.error("--start-xy and --goal-xy must be provided together.")
    if has_start and args.task_id is not None:
        parser.error("--task-id cannot be combined with custom coordinates.")
    if not has_start and args.task_id is None:
        args.task_id = 1

    return args


def format_coordinate(value: float) -> str:
    text = f"{value:g}"
    return text.replace("-", "m").replace(".", "p")


def create_run_dir(base_dir: Path):
    base_dir.mkdir(parents=True, exist_ok=True)

    existing_ids = [
        int(p.name)
        for p in base_dir.iterdir()
        if p.is_dir() and p.name.isdigit()
    ]

    next_id = max(existing_ids, default=0) + 1

    run_dir = base_dir / f"{next_id:06d}"
    run_dir.mkdir()

    return run_dir


def save_json(path: Path, data):
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def main():
    args = parse_args()

    config, saved_flags = load_checkpoint_config(args.checkpoint)
    env_name = saved_flags["env_name"]

    eval_env, train_dataset, val_dataset = make_env_and_datasets(
        env_name,
        frame_stack=config["frame_stack"],
        add_info=True,
    )

    latent_env, _, _ = make_env_and_datasets(
        env_name,
        frame_stack=config["frame_stack"],
        add_info=True,
    )

    eval_env.unwrapped._add_noise_to_goal = False
    latent_env.unwrapped._add_noise_to_goal = False

    train_dataset = Dataset.create(**train_dataset)

    if val_dataset is not None:
        val_dataset = Dataset.create(**val_dataset)

    zero_shot_dataset = (
        val_dataset
        if val_dataset is not None
        else train_dataset
    )

    dataset_module = importlib.import_module("utils.datasets")
    dataset_class = getattr(
        dataset_module,
        config["dataset_class"],
    )

    train_for_agent = dataset_class(
        train_dataset,
        config,
    )

    example_batch = train_for_agent.sample(1)

    frozen_fb = FrozenFB.from_checkpoint(
        args.checkpoint,
        example_batch,
        config=config,
    )

    task_encoder = TaskEncoder(
        frozen_fb,
        latent_env,
        zero_shot_dataset,
        env_name=env_name,
    )

    if args.start_xy is None:
        task = task_encoder.encode_standard_task(args.task_id)
        scenario = Scenario(
            scenario_id=f"ogbench-task-{args.task_id}",
            task_id=args.task_id,
            environment_seed=args.environment_seed,
            controller_seed=args.controller_seed,
        )
        scenario_name = f"task_{args.task_id}"
    else:
        start_ij = xy_to_free_grid_cell(
            eval_env,
            args.start_xy,
            name="start_xy",
        )
        goal_ij = xy_to_free_grid_cell(
            eval_env,
            args.goal_xy,
            name="goal_xy",
        )
        task = task_encoder.encode_custom_task(start_ij, goal_ij)

        requested_goal_xy = np.asarray(args.goal_xy, dtype=np.float64)
        if not np.allclose(task.goal_xy, requested_goal_xy, atol=1e-8, rtol=0.0):
            raise RuntimeError(
                "Custom task latent was encoded for a different goal: "
                f"requested={requested_goal_xy.tolist()}, "
                f"latent_goal={task.goal_xy.tolist()}."
            )

        start_label = "_".join(format_coordinate(v) for v in args.start_xy)
        goal_label = "_".join(format_coordinate(v) for v in args.goal_xy)
        scenario_id = f"custom-{start_label}-to-{goal_label}"
        scenario = Scenario(
            scenario_id=scenario_id,
            task_id=None,
            environment_seed=args.environment_seed,
            controller_seed=args.controller_seed,
            start_ij=start_ij,
            goal_ij=goal_ij,
        )
        scenario_name = scenario_id
    print(
        "latent checksum:",
        np.sum(task.latent),
        np.linalg.norm(task.latent)
    )
    if args.controller == "baseline":
        controller = BaselineController(frozen_fb)
        controller_slug = "baseline"

    elif args.controller == "direct":
        controller = DirectGoalController(frozen_fb)
        controller_slug = "direct_goal"

    elif args.controller == "h0":
        from scripts.run_h0 import make_h0_controller

        controller = make_h0_controller(
            frozen_fb,
            train_dataset,
            max_candidates=args.max_candidates,
        )
        controller_slug = "h0_two_switch"

    experiment_name = f"{controller_slug}_{scenario_name}"

    runner = EpisodeRunner(
        eval_env,
        frozen_fb,
        controller,
        eval_temperature=args.temperature,
    )

    result = runner.run(
        scenario,
        task.latent,
    )

    if not np.allclose(task.goal_xy, result.goal_xy, atol=1e-8, rtol=0.0):
        raise RuntimeError(
            "Task-latent goal and rollout goal differ: "
            f"latent_goal={task.goal_xy.tolist()}, "
            f"runner_goal={result.goal_xy.tolist()}."
        )

    experiment_dir = (
        args.results_dir
        / experiment_name
    )

    run_dir = create_run_dir(
        experiment_dir / "runs"
    )

    save_json(
        experiment_dir / "config.json",
        {
            "environment": env_name,
            "checkpoint": str(args.checkpoint),
            "controller": controller.method_name,
            "scenario_id": scenario.scenario_id,
            "task_id": scenario.task_id,
            "start_ij": scenario.start_ij,
            "goal_ij": scenario.goal_ij,
            "start_xy": args.start_xy,
            "goal_xy": np.asarray(task.goal_xy).tolist(),
            "temperature": args.temperature,
            "environment_seed": args.environment_seed,
            "controller_seed": args.controller_seed,
            "latent_dim": frozen_fb.latent_dim,
            "N_g": task.num_positive,
            "N_samples": task.num_samples,
            "latent_checksum": float(np.sum(task.latent)),
            "latent_norm": float(np.linalg.norm(task.latent)),
        },
    )

    save_json(
        run_dir / "scenario.json",
        {
            "scenario_id": scenario.scenario_id,
            "task_id": scenario.task_id,
            "start_ij": scenario.start_ij,
            "goal_ij": scenario.goal_ij,
            "nominal_start_xy": args.start_xy,
            "goal_xy": np.asarray(task.goal_xy).tolist(),
            "environment_seed": scenario.environment_seed,
            "controller_seed": scenario.controller_seed,
            "temperature": args.temperature,
        },
    )

    save_episode_result(
        result,
        run_dir,
        eval_env,
    )

    print(f"environment: {env_name}")
    print(f"controller: {controller.method_name}")
    print(f"scenario_id: {scenario.scenario_id}")
    print(f"task_id: {scenario.task_id if scenario.task_id is not None else 'custom'}")
    if scenario.is_custom:
        print(f"start_xy_nominal: {list(args.start_xy)}")
    print(f"N_g: {task.num_positive}/{task.num_samples}")
    print(f"latent_norm: {float(np.linalg.norm(task.latent)):.6f}")
    print(f"task_goal_xy: {np.asarray(task.goal_xy).tolist()}")
    print(f"runner_goal_xy: {np.asarray(result.goal_xy).tolist()}")
    print(f"actual_start_xy: {np.asarray(result.start_xy).tolist()}")
    print(f"success: {result.success}")
    print(f"steps: {result.steps}")
    print(f"path_length: {result.path_length:.6f}")
    print(f"final_distance: {result.final_distance:.6f}")
    print(f"duration_s: {result.duration:.3f}")
    print(f"saved_to: {run_dir}")


if __name__ == "__main__":
    main()
