"""Adaptive-depth extension of H0.

The planner evaluates the best supported one-switch plan and the best
supported two-switch plan on the same value scale.  It executes the first
subgoal of the larger estimate.  Exact ties prefer depth one.
"""

from __future__ import annotations

import numpy as np
import jax.numpy as jnp

from hypotheses.h0.planner import TwoSwitchPlanner, TwoSwitchSelection


class AdaptiveSwitchPlanner(TwoSwitchPlanner):
    """Choose planning depth ``d in {1, 2}`` at every replanning step."""

    def experiment_config(self):
        config = super().experiment_config()
        config.update(
            {
                "hypothesis": "h0b_adaptive_switch_depth",
                "adaptive_depths": [1, 2],
                "depth_tie_break": "prefer_depth_1",
                "selection_objective": "max_estimated_switch_value",
            }
        )
        return config

    def score_depths(self, observation, goal_latent):
        """Return ``(V1[K], V2[K,K])`` on one common absolute-value scale."""

        _, details = self._evaluate_pairs(observation, goal_latent)
        one_values = details["one_switch_values"]
        two_values = details["two_switch_values"]

        # Embed every depth-1 plan as the diagonal depth-2 fallback.  This
        # makes the estimated plan classes nested exactly, instead of relying
        # on two separately evaluated neural-network expressions being equal
        # up to floating-point noise.
        diagonal = jnp.eye(self.candidate_count, dtype=bool)
        two_values = jnp.where(diagonal, one_values[:, None], two_values)
        details["two_switch_values"] = two_values
        self._last_details = details
        return one_values, two_values

    def select(self, observation, goal_latent) -> TwoSwitchSelection:
        one_values, two_values = self.score_depths(observation, goal_latent)
        one_np = np.asarray(one_values)
        two_np = np.asarray(two_values)

        one_finite = np.isfinite(one_np)
        two_finite = np.isfinite(two_np)
        if not np.any(one_finite) and not np.any(two_finite):
            raise RuntimeError(
                "H0-B found no valid depth-1 or depth-2 plan; "
                "inspect eta diagnostics or candidate set"
            )

        best_one_index = int(np.argmax(one_np)) if np.any(one_finite) else -1
        best_one_value = (
            float(one_np[best_one_index]) if best_one_index >= 0 else -np.inf
        )
        if np.any(two_finite):
            best_two_flat = int(np.argmax(two_np))
            best_two_i, best_two_j = np.unravel_index(
                best_two_flat,
                two_np.shape,
            )
            best_two_value = float(two_np[best_two_i, best_two_j])
        else:
            best_two_i, best_two_j = -1, -1
            best_two_value = -np.inf

        # Exact ties use the shallower plan.  No unreported depth bonus or
        # complexity penalty is introduced into the scientific objective.
        if best_one_value >= best_two_value:
            selected_depth = 1
            i, j = best_one_index, -1
            selected_value = best_one_value
        else:
            selected_depth = 2
            i, j = int(best_two_i), int(best_two_j)
            selected_value = best_two_value

        details = self._last_details
        eta1 = np.asarray(details["eta1"])
        eta2 = np.asarray(details["eta2"])
        eta1_valid = np.asarray(details["eta1_valid"])
        eta2_valid = np.asarray(details["eta2_valid"])
        eta1_clipped = np.asarray(details["eta1_clipped"])
        eta2_clipped = np.asarray(details["eta2_clipped"])
        direct_value = float(np.asarray(details["direct_value"]))

        return TwoSwitchSelection(
            intention=self.candidate_latents[i],
            diagnostics={
                "h0b_selected_depth": selected_depth,
                "h0b_selected_value": selected_value,
                "h0b_selected_advantage_over_direct": selected_value - direct_value,
                "h0b_best_v1": best_one_value,
                "h0b_best_v2": best_two_value,
                "h0b_v2_minus_v1": best_two_value - best_one_value,
                "h0b_direct_value": direct_value,
                "h0b_best_v1_index": best_one_index,
                "h0b_best_v2_w1_index": int(best_two_i),
                "h0b_best_v2_w2_index": int(best_two_j),
                "w1_index": i,
                "w2_index": j,
                "w1_source_index": int(self.source_indices[i]),
                "w2_source_index": (
                    int(self.source_indices[j]) if selected_depth == 2 else -1
                ),
                "selected_eta1": float(eta1[i]),
                "selected_eta2": (
                    float(eta2[i, j]) if selected_depth == 2 else np.nan
                ),
                "eta1_invalid_count": int((~eta1_valid).sum()),
                "eta2_invalid_count": int((~eta2_valid).sum()),
                "eta1_clipped_count": int(eta1_clipped.sum()),
                "eta2_clipped_count": int(eta2_clipped.sum()),
                "candidate_count": self.candidate_count,
                "selected_subgoals_equal": bool(selected_depth == 2 and i == j),
            },
        )
