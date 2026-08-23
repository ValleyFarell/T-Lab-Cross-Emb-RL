"""Запуск контроллера проекции декодированной подцели."""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path

import numpy as np

np.in1d = np.isin

from baseline.frozen_fb import FrozenFB, load_checkpoint_config
from baseline.task_encoder import TaskEncoder
from controllers.decoded_dataset_subgoal import DecodedDatasetSubgoalController
from evaluation.runner import EpisodeRunner
from evaluation.save_episode import save_episode_result
from evaluation.scenarios import Scenario
from hypotheses.decoded_dataset_subgoal import DecodedDatasetSubgoalPlanner
from probes.intention_xy import IntentionXYDecoder
from scripts.run_baseline import create_run_dir
from utils.datasets import Dataset
from utils.env_utils import make_env_and_datasets


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("checkpoints/antmaze-medium-navigate-v0"),
    help='Каталог замороженного агента с params.pkl и flags.json.',
    )
    parser.add_argument(
        "--decoder",
        type=Path,
        default=Path("artifacts/intention_xy_decoder_bmirror"),
    help='Каталог с обученным декодером намерения либо путь к decoder.npz.',
    )
    parser.add_argument("--task-id", type=int, default=None, help='Номер официальной задачи при --target-mode fixed-task.')
    parser.add_argument(
        "--start-xy",
        type=float,
        nargs=2,
        metavar=("X", "Y"),
        help='Координаты центра свободной начальной клетки; требуется --goal-xy.',
    )
    parser.add_argument(
        "--goal-xy",
        type=float,
        nargs=2,
        metavar=("X", "Y"),
        help='Координаты центра свободной целевой клетки.',
    )
    parser.add_argument("--environment-seed", type=int, default=0, help='Число, определяющее воспроизводимое начальное состояние среды.')
    parser.add_argument("--controller-seed", type=int, default=0, help='Число, определяющее воспроизводимую случайность контроллера.')
    parser.add_argument("--temperature", type=float, default=0.0, help='Случайность выбора действий; ноль означает детерминированный режим.')
    parser.add_argument("--candidate-radius", type=float, default=0.5, help='Радиус поиска реальных офлайн-состояний вокруг декодированной точки.')
    parser.add_argument("--max-candidates", type=int, default=64, help='Максимальное число промежуточных точек после построения набора кандидатов.')
    parser.add_argument("--disagreement-penalty", type=float, default=0.5, help='Штраф за расхождение участников FB-ансамбля.')
    parser.add_argument(
        "--selection-mode",
        choices=("max-v", "nearest-xy"),
        default="max-v",
    help='max-v выбирает по ценности; nearest-xy — по близости координат.',
    )
    parser.add_argument(
        "--finish-mode",
        choices=("task-latent", "fixed-v-max", "dynamic-v-max"),
        default="dynamic-v-max",
        help=(
            'Способ выбора конечного намерения или управления возле цели.'
        ),
    )
    parser.add_argument("--replan-interval", type=int, default=5, help='Число шагов между плановыми перестроениями маршрута.')
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results_decoded_dataset_subgoal"),
    help='Каталог сохранения эпизодов.',
    )
    args = parser.parse_args(argv)
    has_start = args.start_xy is not None
    has_goal = args.goal_xy is not None
    if has_start != has_goal:
        parser.error("--start-xy and --goal-xy must be specified together")
    if args.task_id is not None and has_start:
        parser.error("use either --task-id or --start-xy/--goal-xy, not both")
    if args.task_id is None and not has_start:
        args.task_id = 1
    if has_start:
        coordinates = np.asarray([args.start_xy, args.goal_xy], dtype=np.float64)
        if not np.all(np.isfinite(coordinates)):
            parser.error("manual coordinates must be finite")
    if not np.isfinite(args.candidate_radius) or args.candidate_radius <= 0:
        parser.error("--candidate-radius must be positive and finite")
    if args.max_candidates <= 0:
        parser.error("--max-candidates must be positive")
    if (
        not np.isfinite(args.disagreement_penalty)
        or args.disagreement_penalty < 0
    ):
        parser.error("--disagreement-penalty must be finite and non-negative")
    if args.replan_interval <= 0:
        parser.error("--replan-interval must be positive")
    return args


def _save_json(path: Path, data):
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def _coordinate_scenario_id(start_xy, goal_xy):
    values = (*start_xy, *goal_xy)
    sx, sy, gx, gy = (f"{float(value):g}" for value in values)
    return f"custom-{sx}_{sy}-to-{gx}_{gy}"


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
    frozen_fb = FrozenFB.from_checkpoint(
        args.checkpoint,
        train_for_agent.sample(1),
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
    else:
        try:
            from evaluation.scenarios import xy_to_free_grid_cell
        except ImportError as exc:
            raise RuntimeError(
                "Manual coordinates require the current evaluation/scenarios.py "
                "with xy_to_free_grid_cell()."
            ) from exc
        start_ij = xy_to_free_grid_cell(eval_env, args.start_xy, name="start_xy")
        goal_ij = xy_to_free_grid_cell(eval_env, args.goal_xy, name="goal_xy")
        task = task_encoder.encode_custom_task(start_ij, goal_ij)
        requested_goal_xy = np.asarray(args.goal_xy, dtype=np.float64)
        if not np.allclose(task.goal_xy, requested_goal_xy, atol=1e-6):
            raise RuntimeError(
                "The environment resolved a goal different from --goal-xy: "
                f"requested={requested_goal_xy.tolist()}, "
                f"resolved={np.asarray(task.goal_xy).tolist()}"
            )
        scenario = Scenario(
            scenario_id=_coordinate_scenario_id(args.start_xy, args.goal_xy),
            task_id=None,
            environment_seed=args.environment_seed,
            controller_seed=args.controller_seed,
            start_ij=start_ij,
            goal_ij=goal_ij,
        )

    decoder = IntentionXYDecoder.load(args.decoder)
    planner = DecodedDatasetSubgoalPlanner(
        frozen_fb,
        decoder,
        train_dataset,
        goal_xy=task.goal_xy,
        candidate_radius=args.candidate_radius,
        max_candidates=args.max_candidates,
        disagreement_penalty=args.disagreement_penalty,
        selection_mode=args.selection_mode,
    )
    controller = DecodedDatasetSubgoalController(
        frozen_fb,
        planner,
        replan_interval=args.replan_interval,
        finish_mode=args.finish_mode,
    )
    runner = EpisodeRunner(
        eval_env,
        frozen_fb,
        controller,
        eval_temperature=args.temperature,
    )
    result = runner.run(scenario, task.latent)

    task_slug = (
        f"task_{args.task_id}"
        if args.start_xy is None
        else scenario.scenario_id
    )
    experiment_name = f"{controller.method_name}_{args.selection_mode}_{task_slug}"
    experiment_dir = args.results_dir / experiment_name
    run_dir = create_run_dir(experiment_dir / "runs")
    experiment_config = {
        "environment": env_name,
        "checkpoint": str(args.checkpoint),
        "decoder": str(args.decoder),
        "task_id": args.task_id,
        "requested_start_xy": args.start_xy,
        "requested_goal_xy": args.goal_xy,
        "resolved_goal_xy": np.asarray(task.goal_xy).tolist(),
        "temperature": args.temperature,
        "environment_seed": args.environment_seed,
        "controller_seed": args.controller_seed,
        "latent_dim": frozen_fb.latent_dim,
        "N_g": task.num_positive,
        "N_samples": task.num_samples,
        "method": controller.method_name,
        "method_config": controller.experiment_config(),
    }
    _save_json(experiment_dir / "config.json", experiment_config)
    _save_json(
        run_dir / "scenario.json",
        {
            "scenario_id": scenario.scenario_id,
            "task_id": scenario.task_id,
            "environment_seed": scenario.environment_seed,
            "controller_seed": scenario.controller_seed,
            "start_ij": getattr(scenario, "start_ij", None),
            "goal_ij": getattr(scenario, "goal_ij", None),
        },
    )
    save_episode_result(result, run_dir, eval_env)

    print(f"environment: {env_name}")
    print(f"scenario: {scenario.scenario_id}")
    print(f"selection_mode: {args.selection_mode}")
    print(f"finish_mode: {args.finish_mode}")
    print(f"success: {result.success}")
    print(f"steps: {result.steps}")
    print(f"path_length: {result.path_length:.6f}")
    print(f"final_distance: {result.final_distance:.6f}")
    print(f"duration_s: {result.duration:.3f}")
    print(f"saved_to: {run_dir}")


if __name__ == "__main__":
    main()
