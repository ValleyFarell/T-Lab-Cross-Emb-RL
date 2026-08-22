"""Backward-compatible import for the H0 planner.

New code should import from :mod:`hypotheses.h0`.  This shim prevents existing
experiments from breaking while keeping hypothesis code outside ``baseline``.
"""

from hypotheses.h0 import TwoSwitchPlanner, TwoSwitchSelection

__all__ = ["TwoSwitchPlanner", "TwoSwitchSelection"]

