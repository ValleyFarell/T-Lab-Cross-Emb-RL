"""Проверки корректности компонента direct goal controller и его взаимодействия со стендом."""

import numpy as np

from controllers.direct_goal import DirectGoalController


class FakeFrozenFB:
    def normalize_latent(self, latent):
        raise AssertionError(
            "Raw direct controller must not normalize task_latent"
        )


def test_direct_goal_controller_passes_raw_latent_and_bypasses_high_actor():
    controller = DirectGoalController(FakeFrozenFB())
    task_latent = np.array([3.0, 4.0])

    selection = controller.select_intention(
        observation=np.array([100.0]),
        task_latent=task_latent,
        rng=object(),
        temperature=0.0,
    )

    np.testing.assert_array_equal(
        selection.intention,
        task_latent,
    )