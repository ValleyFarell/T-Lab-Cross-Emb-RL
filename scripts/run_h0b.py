"""Создание и запуск H0-B с выбором глубины планирования."""

from __future__ import annotations

import sys

import numpy as np

from controllers.adaptive_switch import AdaptiveSwitchController
from hypotheses.h0b import AdaptiveSwitchPlanner


def make_h0b_controller(
    frozen_fb,
    offline_dataset,
    *,
    max_candidates: int | None = 64,
    pair_batch_size: int = 4096,
    eta_epsilon: float = 1e-6,
    replan_interval: int = 1,
):
    candidates = np.asarray(offline_dataset["observations"])
    planner = AdaptiveSwitchPlanner(
        frozen_fb,
        candidates,
        max_candidates=max_candidates,
        pair_batch_size=pair_batch_size,
        eta_epsilon=eta_epsilon,
    )
    return AdaptiveSwitchController(
        planner,
        replan_interval=replan_interval,
    )


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    if any(arg == "--controller" or arg.startswith("--controller=") for arg in args):
        raise SystemExit(
            "scripts.run_h0b selects --controller h0b automatically; "
            "remove the explicit --controller argument"
        )

    from scripts.run_baseline import main as run_experiment

    return run_experiment(["--controller", "h0b", *args])


if __name__ == "__main__":
    main()

