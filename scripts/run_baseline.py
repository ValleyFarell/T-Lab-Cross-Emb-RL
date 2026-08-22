"""Run one OGBench episode with a selected high-level controller."""

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
from evaluation.runner import EpisodeRunner
from evaluation.save_episode import save_episode_result
from evaluation.scenarios import Scenario, xy_to_free_grid_cell
from utils.datasets import Dataset
from utils.env_utils import make_env_and_datasets


PUBLIC_CONTROLLERS = ("baseline", "direct", "h0", "h0b")


def parse_args(argv=None):
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
    parser.add_argument("--environment-seed", type=int, default=0)
    parser.add_argument("--controller-seed", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--controller",
        choices=PUBLIC_CONTROLLERS,
        default="baseline",
        help="High-level controller used for this episode.",
    )
    parser.add_argument(
        "--max-candidates",
        type=int,
        default=64,
        help="H0/H0-B: deterministic candidate subset size (default: 64).",
    )
    parser.add_argument(
        "--pair-batch-size",
        type=int,
        default=4096,
        help="H0/H0-B: number of candidate pairs per forward batch.",
    )
    parser.add_argument(
        "--eta-epsilon",
        type=float,
        default=1e-6,
        help="H0/H0-B: minimum absolute eta denominator.",
    )
    parser.add_argument(
        "--h0-replan-interval",
        type=int,
        default=1,
        help="H0/H0-B: execute the selected first subgoal before replanning.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results"),
    )

    args = parser.parse_args(argv)
    has_start = args.start_xy is not None
    has_goal = args.goal_xy is not None
    if has_start != has_goal:
        parser.error("--start-xy and --goal-xy must be provided together.")
    if has_start and args.task_id is not None:
        parser.error("--task-id cannot be combined with custom coordinates.")
    if not has_start and args.task_id is None:
        args.task_id = 1

    if args.controller in {"h0", "h0b"}:
        if args.max_candidates <= 0:
            parser.error("--max-candidates must be positive.")
        if args.pair_batch_size <= 0:
            parser.error("--pair-batch-size must be positive.")
        if not np.isfinite(args.eta_epsilon) or args.eta_epsilon <= 0:
            parser.error("--eta-epsilon must be a positive finite number.")
        if args.h0_replan_interval <= 0:
            parser.error("--h0-replan-interval must be positive.")
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
    run_dir = base_dir / f"{max(existing_ids, default=0) + 1:06d}"
    run_dir.mkdir()
    return run_dir


def save_json(path: Path, data):
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def build_controller(args, frozen_fb, train_dataset):
    """The only dispatch point needed when a new hypothesis is added."""

    if args.controller == "baseline":
        return BaselineController(frozen_fb), "baseline"
    if args.controller == "direct":
        return DirectGoalController(frozen_fb), "direct_goal"
    if args.controller == "h0":
        from scripts.run_h0 import make_h0_controller

        controller = make_h0_controller(
            frozen_fb,
            train_dataset,
            max_candidates=args.max_candidates,
            pair_batch_size=args.pair_batch_size,
            eta_epsilon=args.eta_epsilon,
            replan_interval=args.h0_replan_interval,
        )
        return controller, "h0_two_switch"
    if args.controller == "h0b":
        from scripts.run_h0b import make_h0b_controller

        controller = make_h0b_controller(
            frozen_fb,
            train_dataset,
            max_candidates=args.max_candidates,
            pair_batch_size=args.pair_batch_size,
            eta_epsilon=args.eta_epsilon,
            replan_interval=args.h0_replan_interval,
        )
        return controller, "h0b_adaptive_depth"
    raise ValueError(f"Unknown controller: {args.controller}")


def main(argv=None):
    args = parse_args(argv)
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
    zero_shot_dataset = val_dataset if val_dataset is not None else train_dataset

    dataset_module = importlib.import_module("utils.datasets")
    dataset_class = getattr(dataset_module, config["dataset_class"])
    train_for_agent = dataset_class(train_dataset, config)
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
        start_ij = xy_to_free_grid_cell(eval_env, args.start_xy, name="start_xy")
        goal_ij = xy_to_free_grid_cell(eval_env, args.goal_xy, name="goal_xy")
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

    print("latent checksum:", np.sum(task.latent), np.linalg.norm(task.latent))
    controller, controller_slug = build_controller(args, frozen_fb, train_dataset)
    experiment_name = f"{controller_slug}_{scenario_name}"

    runner = EpisodeRunner(
        eval_env,
        frozen_fb,
        controller,
        eval_temperature=args.temperature,
    )
    result = runner.run(scenario, task.latent)

    if not np.allclose(task.goal_xy, result.goal_xy, atol=1e-8, rtol=0.0):
        raise RuntimeError(
            "Task-latent goal and rollout goal differ: "
            f"latent_goal={task.goal_xy.tolist()}, "
            f"runner_goal={result.goal_xy.tolist()}."
        )

    experiment_dir = args.results_dir / experiment_name
    run_dir = create_run_dir(experiment_dir / "runs")
    run_config = {
        "environment": env_name,
        "checkpoint": str(args.checkpoint),
        "controller": controller.method_name,
        "method_config": dict(controller.experiment_config()),
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
    }
    # Keep the historical experiment-level file and also store an immutable
    # copy next to the run so later parameter changes cannot relabel old data.
    save_json(experiment_dir / "config.json", run_config)
    save_json(run_dir / "config.json", run_config)
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
    save_episode_result(result, run_dir, eval_env)

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
    return result


if __name__ == "__main__":
    main()
