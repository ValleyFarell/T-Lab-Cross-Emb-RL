"""Запуск локального H0 с прямым резервом и режимом финиша."""

from __future__ import annotations

import argparse
import sys

import numpy as np


def _extension_parser():
    parser = argparse.ArgumentParser(
        prog="python -m scripts.run_h0_local_terminal",
        add_help=False,
        allow_abbrev=False,
        description=(
            'Запуск локального H0 с прямым резервом и режимом финиша.'
        ),
    )
    parser.add_argument("--candidates-per-cell", type=int, default=10, help='Сколько офлайн-состояний можно взять из одной клетки лабиринта.')
    parser.add_argument("--grid-cell-size", type=float, default=4.0, help='Размер клетки пространственной сетки в координатах карты.')
    parser.add_argument("--local-radius", type=float, default=5.0, help='Радиус вокруг робота, в котором разрешена первая исполняемая подцель.')
    parser.add_argument("--max-local-candidates", type=int, default=32, help='Верхняя граница числа локальных первых подцелей.')
    parser.add_argument("--finish-radius", type=float, default=2.0, help='Расстояние до цели, на котором маршрут заменяется финишным управлением.')
    parser.add_argument("--pair-batch-size", type=int, default=512, help='Максимальное количество пар в одном обращении к прямому представлению.')
    parser.add_argument(
        "--finish-mode", choices=("direct", "baseline"), default="direct"
    , help='baseline использует исходный контроллер, task-latent — целевое намерение напрямую.')
    parser.add_argument(
        "--direct-latent-mode", choices=("raw", "normalized"), default="raw"
    , help='raw оставляет целевой вектор без изменений, normalized нормализует его.')
    parser.add_argument("--switch-margin", type=float, default=0.0, help='Минимальное улучшение, необходимое для выбора маршрута с переключением.')
    parser.add_argument("--no-zero-level", action="store_true", help='Отключает сравнение с прямым вариантом без промежуточных целей.')
    parser.add_argument("--no-finish-latch", action="store_true", help='Отключает фиксацию финишного режима после первого входа в его область.')
    return parser


def parse_extension_args(argv):
    parser = _extension_parser()
    extension, shared = parser.parse_known_args(list(argv))
    if extension.candidates_per_cell <= 0:
        parser.error("--candidates-per-cell must be positive")
    if extension.max_local_candidates <= 0:
        parser.error("--max-local-candidates must be positive")
    if extension.pair_batch_size <= 0:
        parser.error("--pair-batch-size must be positive")
    for name in ("grid_cell_size", "local_radius"):
        value = getattr(extension, name)
        if not np.isfinite(value) or value <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive and finite")
    for name in ("finish_radius", "switch_margin"):
        value = getattr(extension, name)
        if not np.isfinite(value) or value < 0:
            parser.error(f"--{name.replace('_', '-')} must be non-negative and finite")
    if any(arg == "--controller" or arg.startswith("--controller=") for arg in shared):
        parser.error("the launcher selects its controller automatically")
    return extension, shared


def make_h0_local_terminal_controller(
    frozen_fb,
    offline_dataset,
    goal_xy,
    *,
    candidates_per_cell: int = 10,
    grid_cell_size: float = 4.0,
    local_radius: float = 5.0,
    max_local_candidates: int = 32,
    finish_radius: float = 2.0,
    finish_mode: str = "direct",
    direct_latent_mode: str = "raw",
    pair_batch_size: int = 1024,
    eta_epsilon: float = 1e-6,
    switch_margin: float = 0.0,
    enable_zero_level: bool = True,
    replan_interval: int = 1,
    latch_finish: bool = True,
):
    from controllers.h0_local_terminal import LocalTerminalController
    from hypotheses.h0_local_terminal import LocalTwoSwitchPlanner

    planner = LocalTwoSwitchPlanner(
        frozen_fb,
        offline_dataset,
        goal_xy,
        candidates_per_cell=candidates_per_cell,
        grid_cell_size=grid_cell_size,
        local_radius=local_radius,
        max_local_candidates=max_local_candidates,
        finish_radius=finish_radius,
        finish_mode=finish_mode,
        direct_latent_mode=direct_latent_mode,
        pair_batch_size=pair_batch_size,
        eta_epsilon=eta_epsilon,
        switch_margin=switch_margin,
        enable_zero_level=enable_zero_level,
    )
    return LocalTerminalController(
        frozen_fb,
        planner,
        replan_interval=replan_interval,
        latch_finish=latch_finish,
    )


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--help-local" in argv:
        _extension_parser().print_help()
        return None
    extension, shared_argv = parse_extension_args(argv)

    from scripts import run_baseline as shared_launcher

    original_build_controller = shared_launcher.build_controller

    def build_local_controller(args, frozen_fb, train_dataset, goal_xy):
        if args.controller != "h0":
            return original_build_controller(
                args, frozen_fb, train_dataset, goal_xy
            )
        controller = make_h0_local_terminal_controller(
            frozen_fb,
            train_dataset,
            goal_xy,
            candidates_per_cell=extension.candidates_per_cell,
            grid_cell_size=extension.grid_cell_size,
            local_radius=extension.local_radius,
            max_local_candidates=extension.max_local_candidates,
            finish_radius=extension.finish_radius,
            finish_mode=extension.finish_mode,
            direct_latent_mode=extension.direct_latent_mode,
            pair_batch_size=extension.pair_batch_size,
            eta_epsilon=getattr(args, "eta_epsilon", 1e-6),
            switch_margin=extension.switch_margin,
            enable_zero_level=not extension.no_zero_level,
            replan_interval=getattr(args, "h0_replan_interval", 1),
            latch_finish=not extension.no_finish_latch,
        )
        return controller, "h0_local_terminal"

    shared_launcher.build_controller = build_local_controller
    try:
        return shared_launcher.main(["--controller", "h0", *shared_argv])
    finally:
        shared_launcher.build_controller = original_build_controller


if __name__ == "__main__":
    main()
