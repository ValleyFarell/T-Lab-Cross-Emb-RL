"""Запуск способов выбора полного состояния около цели."""

from __future__ import annotations

import argparse
import sys

from controllers.h_goal_eur import GoalEurController
from hypotheses.h_goal_eur import (
    DatasetMaxValueGoalPlanner,
    SyntheticCurrentGoalPlanner,
)


CLI_VARIANTS = {
    "synthetic-current": "hge-synthetic",
    "dataset-max-v": "hge-max-v",
}


def make_h_goal_eur_controller(
    frozen_fb,
    offline_dataset,
    goal_xy,
    *,
    variant: str,
    replan_interval: int = 1,
    candidate_radius: float = 0.5,
    max_candidates: int = 64,
    disagreement_penalty: float = 0.5,
):
    if variant == "synthetic-current":
        planner = SyntheticCurrentGoalPlanner(frozen_fb, goal_xy)
    elif variant == "dataset-max-v":
        planner = DatasetMaxValueGoalPlanner(
            frozen_fb,
            offline_dataset,
            goal_xy,
            candidate_radius=candidate_radius,
            max_candidates=max_candidates,
            disagreement_penalty=disagreement_penalty,
        )
    else:
        raise ValueError(f"Unknown H_goal_eur variant: {variant!r}")
    return GoalEurController(planner, replan_interval=replan_interval)


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    if any(arg == "--controller" or arg.startswith("--controller=") for arg in args):
        raise SystemExit(
            "scripts.run_h_goal_eur selects the controller from --variant; "
            "remove the explicit --controller argument"
        )

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--variant", choices=tuple(CLI_VARIANTS), required=True, help='synthetic-current создаёт синтетическое целевое состояние; dataset-max-v выбирает реальное офлайн-состояние.')
    known, remaining = parser.parse_known_args(args)

    from scripts.run_baseline import main as run_experiment

    return run_experiment(
        ["--controller", CLI_VARIANTS[known.variant], *remaining]
    )


if __name__ == "__main__":
    main()
