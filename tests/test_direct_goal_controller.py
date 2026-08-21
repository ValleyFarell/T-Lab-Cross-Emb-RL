import numpy as np

from controllers.direct_goal import DirectGoalController


class FakeFrozenFB:
    @staticmethod
    def normalize_latent(latent):
        latent = np.asarray(latent)
        return latent / np.linalg.norm(latent) * np.sqrt(latent.size)


def test_direct_goal_controller_bypasses_high_actor():
    controller = DirectGoalController(FakeFrozenFB())
    task_latent = np.array([3.0, 4.0])

    selection = controller.select_intention(
        observation=np.array([100.0]),
        task_latent=task_latent,
        rng=object(),
        temperature=0.0,
    )

    expected = task_latent / 5.0 * np.sqrt(2.0)
    np.testing.assert_allclose(selection.intention, expected)
    assert selection.diagnostics == {}
