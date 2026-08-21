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
from evaluation.runner import EpisodeRunner
from evaluation.save_episode import save_episode_result
from evaluation.scenarios import Scenario
from utils.datasets import Dataset
from utils.env_utils import make_env_and_datasets


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("checkpoints/antmaze-medium-navigate-v0"),
    )
    parser.add_argument("--task-id", type=int, default=1)
    parser.add_argument(
        "--environment-seed",
        type=int,
        default=0,
    )
    parser.add_argument("--controller-seed", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.0)

    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results"),
    )

    return parser.parse_args()


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

    task = task_encoder.encode_standard_task(
        args.task_id,
    )
    print(
        "latent checksum:",
        np.sum(task.latent),
        np.linalg.norm(task.latent)
    )
    controller = BaselineController(
        frozen_fb,
    )

    runner = EpisodeRunner(
        eval_env,
        frozen_fb,
        controller,
        eval_temperature=args.temperature,
    )

    scenario = Scenario(
        scenario_id=(
            f"ogbench-task-{args.task_id}"
            f"-env-{args.environment_seed}"
            f"-ctrl-{args.controller_seed}"
        ),
        task_id=args.task_id,
        environment_seed=args.environment_seed,
        controller_seed=args.controller_seed,
    )

    result = runner.run(
        scenario,
        task.latent,
    )

    experiment_dir = (
        args.results_dir
        / f"baseline_task_{args.task_id}"
    )

    run_dir = create_run_dir(
        experiment_dir / "runs"
    )

    save_json(
        experiment_dir / "config.json",
        {
            "environment": env_name,
            "checkpoint": str(args.checkpoint),
            "method": controller.method_name,
            "task_id": args.task_id,
            "temperature": args.temperature,
            "latent_dim": frozen_fb.latent_dim,
            "N_g": task.num_positive,
            "N_samples": task.num_samples,
        },
    )

    save_json(
        run_dir / "scenario.json",
        {
            "scenario_id": scenario.scenario_id,
            "task_id": scenario.task_id,
            "environment_seed": scenario.environment_seed,
            "controller_seed": scenario.controller_seed,
        },
    )

    save_episode_result(
        result,
        run_dir,
        eval_env,
    )

    print(f"environment: {env_name}")
    print(f"task_id: {args.task_id}")
    print(f"N_g: {task.num_positive}/{task.num_samples}")
    print(f"goal_xy: {np.asarray(task.goal_xy).tolist()}")
    print(f"success: {result.success}")
    print(f"steps: {result.steps}")
    print(f"path_length: {result.path_length:.6f}")
    print(f"final_distance: {result.final_distance:.6f}")
    print(f"duration_s: {result.duration:.3f}")
    print(f"saved_to: {run_dir}")


if __name__ == "__main__":
    main()
