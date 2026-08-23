"""Создание и запуск планировщика H0 с двумя подцелями."""

from __future__ import annotations

import sys

import numpy as np

from controllers.two_switch import TwoSwitchController
from hypotheses.h0 import TwoSwitchPlanner


def make_h0_controller(
    frozen_fb,
    offline_dataset,
    *,
    max_candidates: int | None = 64,
    pair_batch_size: int = 4096,
    eta_epsilon: float = 1e-6,
    replan_interval: int = 1,
):
    candidates = np.asarray(offline_dataset["observations"])
    planner = TwoSwitchPlanner(
        frozen_fb,
        candidates,
        max_candidates=max_candidates,
        pair_batch_size=pair_batch_size,
        eta_epsilon=eta_epsilon,
    )
    return TwoSwitchController(
        planner,
        replan_interval=replan_interval,
    )


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    if any(arg == "--controller" or arg.startswith("--controller=") for arg in args):
        raise SystemExit(
            "scripts.run_h0 selects --controller h0 automatically; "
            "remove the explicit --controller argument"
        )

    # Импортируем фабрику отложенно, чтобы избежать циклической зависимости:
    # run_baseline загружает H0 только при явном выборе этого метода.
    from scripts.run_baseline import main as run_experiment

    return run_experiment(["--controller", "h0", *args])


if __name__ == "__main__":
    main()

