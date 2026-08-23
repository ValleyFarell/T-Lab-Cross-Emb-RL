"""Адаптер H0-B, выбирающего между одной и двумя подцелями."""

from .two_switch import TwoSwitchController


class AdaptiveSwitchController(TwoSwitchController):
    method_name = "fbpiswitch_h0b_adaptive_depth"
    replanned_diagnostic = "h0b_replanned"

