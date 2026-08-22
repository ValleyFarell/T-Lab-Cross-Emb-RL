from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import jax.numpy as jnp


@dataclass(frozen=True)
class TwoSwitchSelection:
    intention: object
    diagnostics: dict


class TwoSwitchPlanner:
    """
    H0 inference planner.

    Searches over supported offline states:
        (w1, w2) = argmax Score(s, w1, w2, g)

    It does not train any parameters.
    """

    def __init__(self, frozen_fb, candidate_observations, max_candidates=None):
        self.frozen_fb = frozen_fb
        candidates = np.asarray(candidate_observations)

        if max_candidates is not None and len(candidates) > max_candidates:
            idx = np.linspace(
                0,
                len(candidates) - 1,
                max_candidates,
                dtype=int,
            )
            candidates = candidates[idx]

        self.candidates = jnp.asarray(candidates)

    def _mean_forward(self, observations, intentions):
        value = self.frozen_fb.forward_repr(
            observations,
            intentions,
        )
        return jnp.mean(value, axis=0)

    def score_pairs(self, observation, goal_latent):
        s = jnp.asarray(observation)[None]
        zg = jnp.asarray(goal_latent)

        w = self.candidates
        zw = self.frozen_fb.normalize_latent(
            self.frozen_fb.backward_repr(w)
        )

        K = w.shape[0]

        # s -> w1
        fs_z1 = self._mean_forward(
            jnp.repeat(s, K, axis=0),
            zw,
        )

        # w1,w2 pair grid
        w1 = jnp.repeat(w[:, None], K, axis=1).reshape(K*K, -1)
        w2 = jnp.repeat(w[None, :], K, axis=0).reshape(K*K, -1)

        z1 = jnp.repeat(
            zw[:, None],
            K,
            axis=1,
        ).reshape(K*K, -1)

        z2 = jnp.repeat(
            zw[None, :],
            K,
            axis=0,
        ).reshape(K*K, -1)

        fw1_z1 = self._mean_forward(w1, z1)
        fw1_z2 = self._mean_forward(w1, z2)
        fw2_z2 = self._mean_forward(w2, z2)
        fw2_zg = self._mean_forward(
            w2,
            jnp.repeat(zg[None], K*K, axis=0),
        )

        fs_z1 = fs_z1.reshape(K, 1, -1)
        fw1_z1 = fw1_z1.reshape(K, K, -1)
        fw1_z2 = fw1_z2.reshape(K, K, -1)
        fw2_z2 = fw2_z2.reshape(K, K, -1)
        fw2_zg = fw2_zg.reshape(K, K, -1)

        eta1_num = jnp.sum(fs_z1 * zw[:, None, :], axis=-1)
        eta1_den = jnp.sum(fw1_z1 * zw[:, None, :], axis=-1)
        eta1 = eta1_num / (eta1_den + 1e-8)

        eta2_num = jnp.sum(fw1_z2 * zw[None, :, :], axis=-1)
        eta2_den = jnp.sum(fw2_z2 * zw[None, :, :], axis=-1)
        eta2 = eta2_num / (eta2_den + 1e-8)

        v_sg = jnp.sum(
            self._mean_forward(s, zg[None])[0] * zg
        )

        score = (
            jnp.sum(fs_z1 * zg, axis=-1)
            + eta1 * (
                jnp.sum(fw1_z2 * zg, axis=-1)
                -
                jnp.sum(fw1_z1 * zg, axis=-1)
            )
            + eta1 * eta2 * (
                jnp.sum(fw2_zg * zg, axis=-1)
                -
                jnp.sum(fw2_z2 * zg, axis=-1)
            )
            - v_sg
        )

        return score

    def select(self, observation, goal_latent):
        scores = self.score_pairs(
            observation,
            goal_latent,
        )

        index = np.unravel_index(
            int(jnp.argmax(scores)),
            scores.shape,
        )

        w2 = self.candidates[index[1]]

        intention = self.frozen_fb.normalize_latent(
            self.frozen_fb.backward_repr(w2[None])[0]
        )

        return TwoSwitchSelection(
            intention=intention,
            diagnostics={
                "h0_score": float(scores[index]),
                "w1_index": int(index[0]),
                "w2_index": int(index[1]),
            },
        )
