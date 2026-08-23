"""Запуск динамического 4D-планировщика на одной задаче или серии сценариев."""

from __future__ import annotations

import argparse
from datetime import datetime
import importlib
import json
from pathlib import Path

import numpy as np

from scripts.train_latent_three_dynamic import configure_device


def _log(message: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {message}", flush=True)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/antmaze-medium-navigate-v0"), help='Каталог замороженного агента с params.pkl и flags.json.')
    parser.add_argument("--model-dir", type=Path, default=Path("artifacts/latent_three_dynamic"), help='Каталог с четырёхмерной моделью геометрии и декодером намерений.')
    parser.add_argument("--results-dir", type=Path, default=Path("results_latent_three_dynamic"), help='Каталог сохранения эпизодов.')
    parser.add_argument("--device", choices=("cpu", "gpu", "auto"), default="cpu", help='Устройство вычислений: cpu, gpu или auto, если скрипт поддерживает эти варианты.')
    task_group = parser.add_mutually_exclusive_group()
    task_group.add_argument("--task-id", type=int, default=None, help='Номер официальной задачи при --target-mode fixed-task.')
    task_group.add_argument("--task-ids", type=int, nargs="+", default=None, help='Список номеров задач для одного группового запуска.')
    parser.add_argument("--environment-seed", type=int, default=0, help='Число, определяющее воспроизводимое начальное состояние среды.')
    # Для строгого парного сравнения задавайте то же значение, что использует исходный агент.
    parser.add_argument("--controller-seed", type=int, default=None, help='Число, определяющее воспроизводимую случайность контроллера.')
    parser.add_argument("--seeds", type=int, nargs="+", default=None, help='Список инициализаций среды для серии экспериментов.')
    parser.add_argument("--temperature", type=float, default=0.0, help='Случайность выбора действий; ноль означает детерминированный режим.')
    parser.add_argument("--compare-baseline", action="store_true", help='Дополнительно запускает исходный агент на тех же сценариях.')
    parser.add_argument("--intention-mode", choices=("decoded", "exact-b", "both"), default="decoded", help='decoded, exact-b или both: обученный декодер, прямое B(w) либо оба варианта.')
    parser.add_argument("--max-states", type=int, default=40_000, help='Верхняя граница числа используемых офлайн-состояний.')
    parser.add_argument("--max-candidates", type=int, default=256, help='Максимальное число промежуточных точек после построения набора кандидатов.')
    parser.add_argument("--grid-resolution", type=int, default=6, help='Число делений каждой оси вспомогательной четырёхмерной сетки.')
    parser.add_argument("--disable-grid", action="store_true", help='Оставляет только реальные офлайн-точки, отключая дополнительные точки сетки.')
    parser.add_argument("--rerank-count", type=int, default=16, help='Число лучших предварительных вариантов, перепроверяемых исходным FB-критиком.')
    parser.add_argument("--fb-batch-size", type=int, default=128, help='Максимальный размер блока обращений к замороженному FB-ансамблю.')
    parser.add_argument("--replan-interval", type=int, default=10, help='Число шагов между плановыми перестроениями маршрута.')
    parser.add_argument("--finish-radius", type=float, default=0.75, help='Расстояние до цели, на котором маршрут заменяется финишным управлением.')
    parser.add_argument("--finish-mode", choices=("baseline", "task-latent"), default="baseline", help='baseline использует исходный контроллер, task-latent — целевое намерение напрямую.')
    parser.add_argument("--fallback-mode", choices=("baseline", "task-latent"), default="baseline", help='Управление, которое применяется, когда подходящий маршрут не найден.')
    parser.add_argument("--eta-epsilon", type=float, default=1e-6, help='Защита от почти нулевого знаменателя при вычислении η.')
    parser.add_argument("--minimum-eta", type=float, default=0.01, help='Минимальная допустимая достижимость каждого перехода.')
    parser.add_argument("--min-improvement", type=float, default=0.0, help='Минимальное улучшение оценки маршрута относительно прямого варианта.')
    parser.add_argument("--disagreement-penalty", type=float, default=0.5, help='Штраф за расхождение участников FB-ансамбля.')
    parser.add_argument("--support-multiplier", type=float, default=2.5, help='Допустимое удаление искусственной точки от области существующих офлайн-данных.')
    parser.add_argument("--waypoint-replan-radius", type=float, default=0.15, help='Радиус, в котором текущая промежуточная точка считается достигнутой.')
    parser.add_argument("--disable-arrival-replan", action="store_true", help='Отключает внеплановое перепланирование при достижении первой точки.')
    args = parser.parse_args(argv)
    if args.task_id is None and args.task_ids is None:
        args.task_id = 1
    for name in (
        "max_states", "max_candidates", "grid_resolution", "rerank_count",
        "fb_batch_size", "replan_interval",
    ):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.max_candidates < 3:
        parser.error("--max-candidates must be at least three")
    return args


def _save_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2)


def _episode_configuration(args, *, env_name, task, scenario, controller):
    return {
        "environment": env_name,
        "checkpoint": str(args.checkpoint),
        "model_dir": str(args.model_dir),
        "method": controller.method_name,
        "method_config": dict(controller.experiment_config()),
        "scenario_id": scenario.scenario_id,
        "task_id": scenario.task_id,
        "environment_seed": scenario.environment_seed,
        "controller_seed": scenario.controller_seed,
        "temperature": args.temperature,
        "fixed_goal_xy": np.asarray(task.goal_xy).tolist(),
        "task_latent_norm": float(np.linalg.norm(task.latent)),
        "task_latent_checksum": float(np.asarray(task.latent).sum()),
        "num_positive_reward_states": int(task.num_positive),
    }


def main(argv=None):
    args = parse_args(argv)
    configure_device(args.device)
    np.in1d = np.isin

    from baseline.frozen_fb import FrozenFB, load_checkpoint_config
    from baseline.task_encoder import TaskEncoder
    from controllers.baseline import BaselineController
    from controllers.latent_three_dynamic import DynamicLatentThreeController
    from evaluation.runner import EpisodeRunner
    from evaluation.save_episode import save_episode_result
    from evaluation.scenarios import Scenario
    from hypotheses.latent_three_dynamic import (
        DynamicThreeWaypointPlanner,
        LatentGeometryModel,
        LatentIntentionDecoder,
    )
    from scripts.run_baseline import create_run_dir
    from utils.datasets import Dataset
    from utils.env_utils import make_env_and_datasets

    geometry = LatentGeometryModel.load(args.model_dir)
    decoder = LatentIntentionDecoder.load(args.model_dir)
    config, saved_flags = load_checkpoint_config(args.checkpoint)
    environment_name = saved_flags["env_name"]
    _log(f"Loading benchmark and frozen checkpoint: {environment_name}")
    eval_env, raw_train, raw_validation = make_env_and_datasets(
        environment_name,
        frame_stack=config["frame_stack"],
        add_info=True,
    )
    latent_env, _, _ = make_env_and_datasets(
        environment_name,
        frame_stack=config["frame_stack"],
        add_info=True,
    )
    eval_env.unwrapped._add_noise_to_goal = False
    latent_env.unwrapped._add_noise_to_goal = False
    train_dataset = Dataset.create(**raw_train)
    validation_dataset = Dataset.create(**raw_validation) if raw_validation is not None else None
    zero_shot_dataset = validation_dataset if validation_dataset is not None else train_dataset
    dataset_module = importlib.import_module("utils.datasets")
    dataset_class = getattr(dataset_module, config["dataset_class"])
    train_for_agent = dataset_class(train_dataset, config)
    frozen_fb = FrozenFB.from_checkpoint(args.checkpoint, train_for_agent.sample(1), config=config)
    if decoder.latent_dim != frozen_fb.latent_dim:
        raise ValueError(
            f"decoder dimension {decoder.latent_dim} does not match checkpoint "
            f"dimension {frozen_fb.latent_dim}"
        )
    task_encoder = TaskEncoder(
        frozen_fb,
        latent_env,
        zero_shot_dataset,
        env_name=environment_name,
    )

    task_ids = args.task_ids if args.task_ids is not None else [args.task_id]
    seeds = args.seeds if args.seeds is not None else [args.environment_seed]
    intention_modes = ("decoded", "exact-b") if args.intention_mode == "both" else (args.intention_mode,)
    methods_per_task = len(intention_modes) + int(args.compare_baseline)
    total_episodes = len(task_ids) * len(seeds) * methods_per_task
    completed = 0

    for task_id in task_ids:
        task = task_encoder.encode_standard_task(int(task_id))
        _log(f"Task {task_id}: fixed goal={np.asarray(task.goal_xy).tolist()}")
        controllers = []
        if args.compare_baseline:
            controllers.append(("baseline", BaselineController(frozen_fb)))

        for intention_mode in intention_modes:
            _log(
                f"Preparing 4D candidate index for task {task_id}, "
                f"intention_mode={intention_mode}"
            )
            planner = DynamicThreeWaypointPlanner(
                frozen_fb,
                geometry,
                decoder,
                train_dataset,
                goal_xy=task.goal_xy,
                max_states=args.max_states,
                max_candidates=args.max_candidates,
                grid_resolution=args.grid_resolution,
                rerank_count=args.rerank_count,
                fb_batch_size=args.fb_batch_size,
                eta_epsilon=args.eta_epsilon,
                minimum_eta=args.minimum_eta,
                disagreement_penalty=args.disagreement_penalty,
                support_multiplier=args.support_multiplier,
                min_improvement=args.min_improvement,
                intention_mode=intention_mode,
                include_grid=not args.disable_grid,
            )
            controller = DynamicLatentThreeController(
                frozen_fb,
                planner,
                replan_interval=args.replan_interval,
                finish_radius=args.finish_radius,
                finish_mode=args.finish_mode,
                fallback_mode=args.fallback_mode,
                replan_on_arrival=not args.disable_arrival_replan,
                waypoint_replan_radius=args.waypoint_replan_radius,
            )
            controller.method_name = f"latent_three_dynamic_{intention_mode.replace('-', '_')}"
            controllers.append((controller.method_name, controller))

        for method_name, controller in controllers:
            runner = EpisodeRunner(
                eval_env,
                frozen_fb,
                controller,
                eval_temperature=args.temperature,
            )
            for seed in seeds:
                environment_seed = int(seed)
                if args.seeds is not None or args.controller_seed is None:
                    controller_seed = environment_seed
                else:
                    controller_seed = int(args.controller_seed)
                scenario = Scenario(
                    scenario_id=f"ogbench-task-{task_id}",
                    task_id=int(task_id),
                    environment_seed=environment_seed,
                    controller_seed=controller_seed,
                )
                completed += 1
                _log(
                    f"[{completed}/{total_episodes}] "
                    f"method={method_name} task={task_id} seed={environment_seed}"
                )
                result = runner.run(scenario, task.latent)
                if not np.allclose(task.goal_xy, result.goal_xy, atol=1e-8, rtol=0.0):
                    raise RuntimeError("task latent and environment rollout use different final goals")
                experiment_dir = args.results_dir / f"{method_name}_task_{task_id}"
                run_dir = create_run_dir(experiment_dir / "runs")
                run_configuration = _episode_configuration(
                    args,
                    env_name=environment_name,
                    task=task,
                    scenario=scenario,
                    controller=controller,
                )
                _save_json(experiment_dir / "config.json", run_configuration)
                _save_json(run_dir / "config.json", run_configuration)
                _save_json(
                    run_dir / "scenario.json",
                    {
                        "scenario_id": scenario.scenario_id,
                        "task_id": scenario.task_id,
                        "environment_seed": scenario.environment_seed,
                        "controller_seed": scenario.controller_seed,
                        "goal_xy": np.asarray(task.goal_xy).tolist(),
                    },
                )
                save_episode_result(result, run_dir, eval_env)
                _log(
                    f"success={result.success} steps={result.steps} "
                    f"distance={result.final_distance:.4f} "
                    f"duration_s={result.duration:.2f} saved={run_dir}"
                )

    _log(f"Finished {completed} paired benchmark episodes")


if __name__ == "__main__":
    main()
