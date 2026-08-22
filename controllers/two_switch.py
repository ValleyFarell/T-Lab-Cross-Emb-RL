from .base import HighLevelController, IntentionSelection


class TwoSwitchController(HighLevelController):
    method_name = "fbpiswitch_h0_two_switch"

    def __init__(self, planner):
        self.planner = planner

    def select_intention(
        self,
        observation,
        task_latent,
        *,
        rng,
        temperature,
    ):
        del rng, temperature

        result = self.planner.select(
            observation,
            task_latent,
        )

        return IntentionSelection(
            intention=result.intention,
            diagnostics=result.diagnostics,
        )
