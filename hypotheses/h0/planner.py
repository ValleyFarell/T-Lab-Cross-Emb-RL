"""H0 receding-horizon two-switch planner.

The planner is inference-only.  It searches pairs of supported offline states
``(w1, w2)``, executes the intention for ``w1``, and lets the controller
replan later.  Candidate-dependent representations are cached once; only the
pair-dependent ``F(w1, z2)`` values are evaluated at every replan.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from numbers import Integral
from typing import Any, Mapping

import jax.numpy as jnp
import numpy as np


@dataclass(frozen=True)
class TwoSwitchSelection:
    intention: Any
    diagnostics: Mapping[str, Any]


class TwoSwitchPlanner:
    """Search supported pairs without training new parameters."""

    def __init__(
        self,
        frozen_fb,
        candidate_observations,
        *,
        max_candidates: int | None = 64,
        pair_batch_size: int = 4096,
        eta_epsilon: float = 1e-6,
    ):
        self.frozen_fb = frozen_fb
        self.max_candidates = self._validate_optional_positive_int(
            "max_candidates", max_candidates
        )
        self.pair_batch_size = self._validate_positive_int(
            "pair_batch_size", pair_batch_size
        )
        self.eta_epsilon = self._validate_positive_float(
            "eta_epsilon", eta_epsilon
        )

        source = self._validate_candidates(candidate_observations)
        self.source_candidate_count = int(source.shape[0])

        if self.max_candidates is not None and len(source) > self.max_candidates:
            source_indices = np.linspace(
                0,
                len(source) - 1,
                self.max_candidates,
                dtype=np.int64,
            )
            selection_strategy = "deterministic_linspace"
        else:
            source_indices = np.arange(len(source), dtype=np.int64)
            selection_strategy = "all"

        selected = np.ascontiguousarray(source[source_indices])
        self.source_indices = source_indices
        self.selection_strategy = selection_strategy
        self.candidates = jnp.asarray(selected)
        self.candidate_count = int(selected.shape[0])

        digest = hashlib.sha256()
        digest.update(selected.tobytes())
        digest.update(source_indices.tobytes())
        self.candidate_checksum = digest.hexdigest()

        # These values depend only on the checkpoint and candidate set.
        backward = self.frozen_fb.backward_repr(self.candidates)
        candidate_latents = self.frozen_fb.normalize_latent(backward)
        candidate_latents_np = np.asarray(candidate_latents)
        if candidate_latents_np.ndim != 2 or candidate_latents_np.shape[0] != self.candidate_count:
            raise ValueError(
                "backward_repr must return one latent vector per candidate observation"
            )
        if not np.all(np.isfinite(candidate_latents_np)):
            raise ValueError("candidate latent representations must be finite")
        self.candidate_latents = jnp.asarray(candidate_latents)
        self.self_forward = self._mean_forward(
            self.candidates,
            self.candidate_latents,
        )
        self.self_measure = jnp.sum(
            self.self_forward * self.candidate_latents,
            axis=-1,
        )
        self._last_details: dict[str, Any] | None = None

    @staticmethod
    def _validate_candidates(candidate_observations) -> np.ndarray:
        candidates = np.asarray(candidate_observations)
        if candidates.ndim != 2:
            raise ValueError(
                "candidate_observations must be a two-dimensional numeric array"
            )
        if candidates.shape[0] == 0 or candidates.shape[1] == 0:
            raise ValueError("candidate_observations must not be empty")
        if candidates.dtype.kind not in "iuf":
            raise ValueError("candidate_observations must contain numeric values")
        candidates = candidates.astype(np.float32, copy=False)
        if not np.all(np.isfinite(candidates)):
            raise ValueError("candidate_observations must contain only finite values")
        return candidates

    @staticmethod
    def _validate_positive_int(name: str, value: int) -> int:
        if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
        return int(value)

    @classmethod
    def _validate_optional_positive_int(
        cls, name: str, value: int | None
    ) -> int | None:
        if value is None:
            return None
        return cls._validate_positive_int(name, value)

    @staticmethod
    def _validate_positive_float(name: str, value: float) -> float:
        try:
            parsed = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must be a positive finite number") from exc
        if not np.isfinite(parsed) or parsed <= 0.0:
            raise ValueError(f"{name} must be a positive finite number")
        return parsed

    def experiment_config(self) -> dict[str, Any]:
        """JSON-serializable candidate and numerical-stability metadata."""

        return {
            "hypothesis": "h0_two_switch",
            "source": "train_dataset.observations",
            "source_candidate_count": self.source_candidate_count,
            "max_candidates": self.max_candidates,
            "candidate_count": self.candidate_count,
            "candidate_selection": self.selection_strategy,
            "candidate_source_indices": self.source_indices.tolist(),
            "candidate_checksum_sha256": self.candidate_checksum,
            "pair_count": self.candidate_count**2,
            "pair_batch_size": self.pair_batch_size,
            "eta_epsilon": self.eta_epsilon,
            "eta_range": [0.0, 1.0],
        }

    def _mean_forward(self, observations, intentions):
        value = self.frozen_fb.forward_repr(observations, intentions)
        return jnp.mean(value, axis=0)

    def _safe_eta(self, numerator, denominator):
        """Return clipped eta plus masks for invalid and clipped ratios.

        Near-zero and non-finite denominators invalidate a candidate/pair.  A
        learned ratio outside the theoretical successor-measure range [0, 1]
        is clipped and reported instead of creating an unbounded score.
        """

        numerator = jnp.asarray(numerator)
        denominator = jnp.asarray(denominator)
        valid = (
            jnp.isfinite(numerator)
            & jnp.isfinite(denominator)
            & (jnp.abs(denominator) >= self.eta_epsilon)
        )
        safe_numerator = jnp.where(jnp.isfinite(numerator), numerator, 0.0)
        safe_denominator = jnp.where(valid, denominator, 1.0)
        raw = jnp.where(valid, safe_numerator / safe_denominator, 0.0)
        clipped = jnp.clip(raw, 0.0, 1.0)
        was_clipped = valid & (jnp.abs(raw - clipped) > 1e-7)
        return clipped, valid, was_clipped

    def _validate_query(self, observation, goal_latent):
        observation_np = np.asarray(observation)
        goal_np = np.asarray(goal_latent)
        if observation_np.ndim != 1 or observation_np.shape[0] != self.candidates.shape[1]:
            raise ValueError(
                "observation must be a finite vector with the candidate observation dimension"
            )
        if (
            goal_np.ndim != 1
            or goal_np.size == 0
            or goal_np.shape[0] != self.candidate_latents.shape[-1]
        ):
            raise ValueError(
                "goal_latent must be a finite vector with the candidate latent dimension"
            )
        if not np.all(np.isfinite(observation_np)):
            raise ValueError("observation must contain only finite values")
        if not np.all(np.isfinite(goal_np)):
            raise ValueError("goal_latent must contain only finite values")
        return jnp.asarray(observation_np), jnp.asarray(goal_np)

    def _evaluate_pairs(self, observation, goal_latent):
        observation, zg_reward = self._validate_query(observation, goal_latent)
        zg_policy = self.frozen_fb.normalize_latent(zg_reward)

        k = self.candidate_count
        z = self.candidate_latents
        w = self.candidates

        fs_z = self._mean_forward(
            jnp.repeat(observation[None], k, axis=0),
            z,
        )
        fw_zg = self._mean_forward(
            w,
            jnp.repeat(zg_policy[None], k, axis=0),
        )
        fs_zg = self._mean_forward(
            observation[None],
            zg_policy[None],
        )[0]
        direct_value = jnp.sum(fs_zg * zg_reward)

        eta1_num = jnp.sum(fs_z * z, axis=-1)
        eta1, eta1_valid, eta1_clipped = self._safe_eta(
            eta1_num,
            self.self_measure,
        )

        fs_goal = jnp.sum(fs_z * zg_reward, axis=-1)
        self_goal = jnp.sum(self.self_forward * zg_reward, axis=-1)
        fw_goal = jnp.sum(fw_zg * zg_reward, axis=-1)

        score_chunks = []
        eta2_chunks = []
        eta2_valid_chunks = []
        eta2_clipped_chunks = []

        pair_count = k * k
        for start in range(0, pair_count, self.pair_batch_size):
            stop = min(start + self.pair_batch_size, pair_count)
            flat = jnp.arange(start, stop)
            i = flat // k
            j = flat % k

            # This is the only representation that genuinely depends on both
            # members of the candidate pair.
            fw1_z2 = self._mean_forward(w[i], z[j])
            eta2_num = jnp.sum(fw1_z2 * z[j], axis=-1)
            eta2, eta2_valid, eta2_clipped = self._safe_eta(
                eta2_num,
                self.self_measure[j],
            )

            score = (
                fs_goal[i]
                + eta1[i]
                * (jnp.sum(fw1_z2 * zg_reward, axis=-1) - self_goal[i])
                + eta1[i]
                * eta2
                * (fw_goal[j] - self_goal[j])
                - direct_value
            )
            valid_pair = eta1_valid[i] & eta2_valid & jnp.isfinite(score)
            score_chunks.append(jnp.where(valid_pair, score, -jnp.inf))
            eta2_chunks.append(eta2)
            eta2_valid_chunks.append(eta2_valid)
            eta2_clipped_chunks.append(eta2_clipped)

        scores = jnp.concatenate(score_chunks).reshape(k, k)
        details = {
            "eta1": eta1,
            "eta1_valid": eta1_valid,
            "eta1_clipped": eta1_clipped,
            "eta2": jnp.concatenate(eta2_chunks).reshape(k, k),
            "eta2_valid": jnp.concatenate(eta2_valid_chunks).reshape(k, k),
            "eta2_clipped": jnp.concatenate(eta2_clipped_chunks).reshape(k, k),
        }
        return scores, details

    def score_pairs(self, observation, goal_latent):
        """Return a scalar ``K x K`` score matrix for inspection and tests."""

        scores, details = self._evaluate_pairs(observation, goal_latent)
        self._last_details = details
        return scores

    def select(self, observation, goal_latent) -> TwoSwitchSelection:
        scores = self.score_pairs(observation, goal_latent)
        scores_np = np.asarray(scores)
        finite = np.isfinite(scores_np)
        if not np.any(finite):
            raise RuntimeError(
                "H0 found no valid candidate pair; inspect eta diagnostics or candidate set"
            )

        i, j = np.unravel_index(int(np.argmax(scores_np)), scores_np.shape)
        # This cached value is exactly normalize(B(w1)); no extra network call
        # is needed after the pair has been selected.
        intention = self.candidate_latents[i]

        details = self._last_details or {}
        eta1 = np.asarray(details.get("eta1", np.full(self.candidate_count, np.nan)))
        eta2 = np.asarray(
            details.get(
                "eta2",
                np.full((self.candidate_count, self.candidate_count), np.nan),
            )
        )
        eta1_valid = np.asarray(
            details.get("eta1_valid", np.zeros(self.candidate_count, dtype=bool))
        )
        eta2_valid = np.asarray(
            details.get(
                "eta2_valid",
                np.zeros((self.candidate_count, self.candidate_count), dtype=bool),
            )
        )
        eta1_clipped = np.asarray(
            details.get("eta1_clipped", np.zeros(self.candidate_count, dtype=bool))
        )
        eta2_clipped = np.asarray(
            details.get(
                "eta2_clipped",
                np.zeros((self.candidate_count, self.candidate_count), dtype=bool),
            )
        )

        return TwoSwitchSelection(
            intention=intention,
            diagnostics={
                "h0_score": float(scores_np[i, j]),
                "w1_index": int(i),
                "w2_index": int(j),
                "w1_source_index": int(self.source_indices[i]),
                "w2_source_index": int(self.source_indices[j]),
                "selected_eta1": float(eta1[i]),
                "selected_eta2": float(eta2[i, j]),
                "eta1_invalid_count": int((~eta1_valid).sum()),
                "eta2_invalid_count": int((~eta2_valid).sum()),
                "eta1_clipped_count": int(eta1_clipped.sum()),
                "eta2_clipped_count": int(eta2_clipped.sum()),
                "candidate_count": self.candidate_count,
                "selected_subgoals_equal": bool(i == j),
            },
        )
