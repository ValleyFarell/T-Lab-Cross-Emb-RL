"""
Minimal H0 launcher.

This script replaces only the controller.
The existing evaluation pipeline is reused.
"""

import numpy as np

from baseline.frozen_fb import FrozenFB
from baseline.two_switch_planner import TwoSwitchPlanner
from controllers.two_switch import TwoSwitchController


def make_h0_controller(
    frozen_fb,
    offline_dataset,
    max_candidates=512,
):
    candidates = np.asarray(
        offline_dataset["observations"]
    )

    planner = TwoSwitchPlanner(
        frozen_fb,
        candidates,
        max_candidates=max_candidates,
    )

    return TwoSwitchController(planner)
