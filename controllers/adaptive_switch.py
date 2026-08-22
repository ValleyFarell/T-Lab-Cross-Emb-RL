"""Controller adapter for H0-B adaptive switching depth."""

from .two_switch import TwoSwitchController


class AdaptiveSwitchController(TwoSwitchController):
    method_name = "fbpiswitch_h0b_adaptive_depth"
    replanned_diagnostic = "h0b_replanned"

