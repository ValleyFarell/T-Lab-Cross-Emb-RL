"""Поиск трёхточечного маршрута с повторной проверкой замороженным FB-критиком."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
import time
from typing import Any, Mapping

import numpy as np

from .geometry import EMBEDDING_DIM, normalize_intentions


@dataclass(frozen=True)
class RouteSelection:
    intention: np.ndarray
    diagnostics: Mapping[str, Any]
    route_indices: tuple[int, ...]
    fallback: bool = False


def _positive_int(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _finite_matrix(value: Any, *, name: str, width: int | None = None) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float32)
    if matrix.ndim != 2 or len(matrix) == 0:
        raise ValueError(f"{name} must be a non-empty matrix, got {matrix.shape}")
    if width is not None and matrix.shape[-1] != width:
        raise ValueError(f"{name} must have width {width}, got {matrix.shape}")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} contains non-finite values")
    return matrix


class _NearestIndex:
    """Ищет ближайшие точки через SciPy или ограниченный запасной алгоритм NumPy."""

    def __init__(self, points: np.ndarray):
        self.points = _finite_matrix(points, name="index_points", width=EMBEDDING_DIM)
        self.tree = None
        try:
            from scipy.spatial import cKDTree

            self.tree = cKDTree(self.points)
        except ImportError:
            self.tree = None

    def query(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        query = np.asarray(points, dtype=np.float32)
        single = query.ndim == 1
        if single:
            query = query[None, :]
        if self.tree is not None:
            distance, indices = self.tree.query(query, k=1)
            return np.asarray(distance, dtype=np.float32), np.asarray(indices, dtype=np.int64)
        distances = np.empty(len(query), dtype=np.float32)
        indices = np.empty(len(query), dtype=np.int64)
        for start in range(0, len(query), 64):
            block = query[start : start + 64]
            squared = np.sum((block[:, None, :] - self.points[None, :, :]) ** 2, axis=-1)
            nearest = np.argmin(squared, axis=1)
            distances[start : start + len(block)] = np.sqrt(
                squared[np.arange(len(block)), nearest]
            )
            indices[start : start + len(block)] = nearest
        return distances, indices


class DynamicThreeWaypointPlanner:
    """Оценивает три промежуточные точки и возвращает первое намерение."""

    waypoint_count = 3

    def __init__(
        self,
        frozen_fb,
        geometry,
        decoder,
        offline_dataset,
        *,
        goal_xy,
        max_states: int = 50_000,
        max_candidates: int = 512,
        grid_resolution: int = 6,
        rerank_count: int = 24,
        fb_batch_size: int = 128,
        eta_epsilon: float = 1e-6,
        minimum_eta: float = 0.01,
        disagreement_penalty: float = 0.5,
        support_multiplier: float = 2.5,
        min_improvement: float = 0.0,
        intention_mode: str = "decoded",
        include_grid: bool = True,
    ):
        self.frozen_fb = frozen_fb
        self.geometry = geometry
        self.decoder = decoder
        self.max_states = _positive_int("max_states", max_states)
        self.max_candidates = _positive_int("max_candidates", max_candidates)
        self.grid_resolution = _positive_int("grid_resolution", grid_resolution)
        self.rerank_count = _positive_int("rerank_count", rerank_count)
        self.fb_batch_size = _positive_int("fb_batch_size", fb_batch_size)
        self.eta_epsilon = float(eta_epsilon)
        self.minimum_eta = float(minimum_eta)
        self.disagreement_penalty = float(disagreement_penalty)
        self.support_multiplier = float(support_multiplier)
        self.min_improvement = float(min_improvement)
        self.intention_mode = str(intention_mode)
        self.include_grid = bool(include_grid)

        if not np.isfinite(self.eta_epsilon) or self.eta_epsilon <= 0:
            raise ValueError("eta_epsilon must be positive and finite")
        if not np.isfinite(self.minimum_eta) or not 0 <= self.minimum_eta < 1:
            raise ValueError("minimum_eta must belong to [0, 1)")
        if not np.isfinite(self.disagreement_penalty) or self.disagreement_penalty < 0:
            raise ValueError("disagreement_penalty must be non-negative and finite")
        if not np.isfinite(self.support_multiplier) or self.support_multiplier <= 0:
            raise ValueError("support_multiplier must be positive and finite")
        if not np.isfinite(self.min_improvement):
            raise ValueError("min_improvement must be finite")
        if self.intention_mode not in {"decoded", "exact-b"}:
            raise ValueError("intention_mode must be 'decoded' or 'exact-b'")
        if self.max_candidates < self.waypoint_count:
            raise ValueError("max_candidates must be at least three")

        observations = _finite_matrix(
            offline_dataset["observations"],
            name="offline observations",
        )
        if len(observations) > self.max_states:
            source_indices = np.linspace(0, len(observations) - 1, self.max_states, dtype=np.int64)
        else:
            source_indices = np.arange(len(observations), dtype=np.int64)
        self.source_indices = source_indices
        self.observations = observations[source_indices]
        self.goal_xy = np.asarray(goal_xy, dtype=np.float32)
        if self.goal_xy.shape != (2,) or not np.all(np.isfinite(self.goal_xy)):
            raise ValueError("goal_xy must be a finite two-dimensional vector")
        if int(getattr(geometry, "embedding_dim", EMBEDDING_DIM)) != EMBEDDING_DIM:
            raise ValueError("geometry must use a four-dimensional embedding")
        if int(getattr(decoder, "latent_dim")) != int(frozen_fb.latent_dim):
            raise ValueError("decoder intention dimension does not match the checkpoint")

        self.embeddings = _finite_matrix(
            geometry.encode(self.observations),
            name="offline embeddings",
            width=EMBEDDING_DIM,
        )
        self.embedding_mean = self.embeddings.mean(axis=0)
        self.embedding_scale = np.maximum(self.embeddings.std(axis=0), 1e-4)
        self.standard_embeddings = self._standardize(self.embeddings)
        self.index = _NearestIndex(self.standard_embeddings)
        self.support_radius = self._estimate_support_radius()

        goal_distance = np.linalg.norm(self.observations[:, :2] - self.goal_xy, axis=1)
        self.goal_anchor_index = int(np.argmin(goal_distance))
        self.goal_anchor_observation = self.observations[self.goal_anchor_index]
        self.goal_embedding = self.embeddings[self.goal_anchor_index]

        points, anchors, support_distance, grid_flags = self._build_candidates()
        self.candidate_embeddings = points
        self.candidate_anchor_indices = anchors
        self.candidate_support_distance = support_distance
        self.candidate_is_grid = grid_flags
        self.candidate_observations = self.observations[anchors]
        if self.intention_mode == "decoded":
            intentions = decoder.predict(points)
        else:
            intentions = self._backward_batches(self.candidate_observations)
        self.candidate_intentions = normalize_intentions(intentions, int(frozen_fb.latent_dim))
        self.candidate_self_forward = self._forward_batches(
            self.candidate_observations,
            self.candidate_intentions,
        )
        self.candidate_self_measure = np.einsum(
            "end,nd->en",
            self.candidate_self_forward,
            self.candidate_intentions,
        )
        self.candidate_surrogate_self = np.asarray(
            self.geometry.predict_value(points, points),
            dtype=np.float32,
        )
        self._last_details: dict[str, Any] = {}

    def _standardize(self, embedding: np.ndarray) -> np.ndarray:
        return ((np.asarray(embedding, dtype=np.float32) - self.embedding_mean) / self.embedding_scale).astype(np.float32)

    def _estimate_support_radius(self) -> float:
        if len(self.standard_embeddings) < 4:
            return 1.0
        if self.index.tree is not None:
            sample_indices = np.linspace(
                0,
                len(self.standard_embeddings) - 1,
                min(len(self.standard_embeddings), 2048),
                dtype=np.int64,
            )
            distances, _ = self.index.tree.query(self.standard_embeddings[sample_indices], k=2)
            reference = np.asarray(distances)[:, 1]
        else:
            consecutive = np.diff(self.standard_embeddings[: min(len(self.standard_embeddings), 2048)], axis=0)
            reference = np.linalg.norm(consecutive, axis=1)
        valid = reference[np.isfinite(reference) & (reference > 1e-7)]
        if not len(valid):
            return 0.25
        return max(float(np.quantile(valid, 0.80)) * self.support_multiplier, 1e-3)

    def _build_candidates(self):
        offline_count = min(len(self.embeddings), max(self.waypoint_count, self.max_candidates // 2))
        offline_indices = np.linspace(0, len(self.embeddings) - 1, offline_count, dtype=np.int64)
        candidate_points = [self.embeddings[offline_indices]]
        candidate_anchors = [offline_indices]
        candidate_distance = [np.zeros(len(offline_indices), dtype=np.float32)]
        candidate_grid = [np.zeros(len(offline_indices), dtype=bool)]

        # Искусственные точки сетки допускаются только рядом с опорой реальных офлайн-данных.
        if self.include_grid and self.grid_resolution >= 2:
            lower = np.quantile(self.standard_embeddings, 0.01, axis=0)
            upper = np.quantile(self.standard_embeddings, 0.99, axis=0)
            axes = [np.linspace(lower[i], upper[i], self.grid_resolution, dtype=np.float32) for i in range(EMBEDDING_DIM)]
            grid = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, EMBEDDING_DIM)
            distance, anchors = self.index.query(grid)
            valid = distance <= self.support_radius
            grid = grid[valid]
            anchors = anchors[valid]
            distance = distance[valid]
            remaining = max(0, self.max_candidates - len(offline_indices))
            if remaining and len(grid):
                if len(grid) > remaining:
                    selected = np.linspace(0, len(grid) - 1, remaining, dtype=np.int64)
                    grid, anchors, distance = grid[selected], anchors[selected], distance[selected]
                points = grid * self.embedding_scale + self.embedding_mean
                candidate_points.append(points.astype(np.float32))
                candidate_anchors.append(anchors.astype(np.int64))
                candidate_distance.append(distance.astype(np.float32))
                candidate_grid.append(np.ones(len(points), dtype=bool))

        points = np.concatenate(candidate_points, axis=0)
        anchors = np.concatenate(candidate_anchors, axis=0)
        distance = np.concatenate(candidate_distance, axis=0)
        flags = np.concatenate(candidate_grid, axis=0)
        if len(points) < self.waypoint_count:
            raise RuntimeError("fewer than three valid candidates remain after support filtering")
        return points, anchors, distance, flags

    def _backward_batches(self, observations: np.ndarray) -> np.ndarray:
        parts = []
        for start in range(0, len(observations), self.fb_batch_size):
            backward = np.asarray(
                self.frozen_fb.backward_repr(observations[start : start + self.fb_batch_size]),
                dtype=np.float32,
            )
            normalized = np.asarray(self.frozen_fb.normalize_latent(backward), dtype=np.float32)
            parts.append(normalized)
        return np.concatenate(parts, axis=0)

    def _forward_batches(self, observations: np.ndarray, intentions: np.ndarray) -> np.ndarray:
        observation_matrix = _finite_matrix(observations, name="forward observations")
        intention_matrix = _finite_matrix(
            intentions,
            name="forward intentions",
            width=int(self.frozen_fb.latent_dim),
        )
        if len(observation_matrix) != len(intention_matrix):
            raise ValueError("forward observations and intentions need matching batch lengths")
        parts = []
        for start in range(0, len(observation_matrix), self.fb_batch_size):
            stop = start + self.fb_batch_size
            forward = np.asarray(
                self.frozen_fb.forward_repr(
                    observation_matrix[start:stop],
                    intention_matrix[start:stop],
                ),
                dtype=np.float32,
            )
            if forward.ndim == 2:
                forward = forward[None, :, :]
            if forward.ndim != 3 or forward.shape[1:] != (
                len(observation_matrix[start:stop]),
                int(self.frozen_fb.latent_dim),
            ):
                raise RuntimeError(f"unexpected forward-ensemble shape: {forward.shape}")
            parts.append(forward)
        return np.concatenate(parts, axis=1)

    def _surrogate_edge_cost(self, starts: np.ndarray, goals: np.ndarray) -> np.ndarray:
        starts = np.asarray(starts, dtype=np.float32)
        goals = np.asarray(goals, dtype=np.float32)
        if starts.ndim == 1:
            starts = np.repeat(starts[None, :], len(goals), axis=0)
        if goals.ndim == 1:
            goals = np.repeat(goals[None, :], len(starts), axis=0)
        value = np.asarray(self.geometry.predict_value(starts, goals), dtype=np.float64)
        self_value = np.asarray(self.geometry.predict_value(goals, goals), dtype=np.float64)
        valid = np.isfinite(value) & np.isfinite(self_value) & (np.abs(self_value) >= self.eta_epsilon)
        ratio = np.divide(value, self_value, out=np.zeros_like(value), where=valid)
        eta = np.clip(ratio, 0.0, 1.0)
        return np.where(valid, -np.log(np.maximum(eta, self.eta_epsilon)), 1e6)

    def _shortlist_insertions(self, observation_embedding: np.ndarray, route: tuple[int, ...]):
        nodes = [observation_embedding]
        nodes.extend(self.candidate_embeddings[list(route)] if route else [])
        nodes.append(self.goal_embedding)
        candidates = self.candidate_embeddings
        all_scores = []
        all_routes = []
        used = set(route)
        for position in range(len(route) + 1):
            before = np.asarray(nodes[position], dtype=np.float32)
            after = np.asarray(nodes[position + 1], dtype=np.float32)
            current_cost = float(self._surrogate_edge_cost(before[None, :], after[None, :])[0])
            incoming = self._surrogate_edge_cost(before, candidates)
            outgoing = self._surrogate_edge_cost(candidates, after)
            scores = current_cost - incoming - outgoing
            scores -= 0.10 * self.candidate_support_distance
            if used:
                scores[list(used)] = -np.inf
            valid = np.isfinite(scores)
            if not np.any(valid):
                continue
            # Быстрая модель лишь отбирает предложения; окончательное решение принимает FB-критик.
            count = min(self.rerank_count, int(valid.sum()))
            selected = np.argpartition(scores, -count)[-count:]
            selected = selected[np.argsort(scores[selected])[::-1]]
            for candidate in selected:
                if not np.isfinite(scores[candidate]):
                    continue
                expanded = route[:position] + (int(candidate),) + route[position:]
                all_routes.append(expanded)
                all_scores.append(float(scores[candidate]))
        if not all_routes:
            return [], np.empty(0, dtype=np.float64)
        surrogate_scores = np.asarray(all_scores, dtype=np.float64)
        count = min(self.rerank_count, len(all_routes))
        selected = np.argpartition(surrogate_scores, -count)[-count:]
        selected = selected[np.argsort(surrogate_scores[selected])[::-1]]
        return [all_routes[index] for index in selected], surrogate_scores[selected]

    def score_routes(self, observation: Any, task_latent: Any, routes: list[tuple[int, ...]]):
        """Оценивает маршруты одинаковой длины по точной формуле переключения."""

        state = np.asarray(observation, dtype=np.float32)
        reward = np.asarray(task_latent, dtype=np.float32)
        if state.shape != (self.observations.shape[1],):
            raise ValueError(f"observation has wrong shape: {state.shape}")
        if reward.shape != (int(self.frozen_fb.latent_dim),):
            raise ValueError(f"task_latent has wrong shape: {reward.shape}")
        if not np.all(np.isfinite(state)) or not np.all(np.isfinite(reward)):
            raise ValueError("observation and task_latent must be finite")
        if not routes:
            return np.empty(0), {}
        lengths = {len(route) for route in routes}
        if len(lengths) != 1 or min(lengths) < 1 or max(lengths) > self.waypoint_count:
            raise ValueError("routes must have the same length between one and three")
        route_matrix = np.asarray(routes, dtype=np.int64)
        if np.any(route_matrix < 0) or np.any(route_matrix >= len(self.candidate_embeddings)):
            raise ValueError("route contains an invalid candidate index")

        count = len(route_matrix)
        terminal_policy = normalize_intentions(reward, int(self.frozen_fb.latent_dim))
        direct_forward = self._forward_batches(state[None, :], terminal_policy[None, :])[:, 0, :]
        direct_by_member = np.einsum("ed,d->e", direct_forward, reward)
        ensemble_size = len(direct_by_member)
        accumulated = np.zeros((ensemble_size, count), dtype=np.float64)
        discount = np.ones((ensemble_size, count), dtype=np.float64)
        valid = np.ones((ensemble_size, count), dtype=bool)
        previous_self_task_value = None
        etas = []
        clipped_counts = np.zeros(count, dtype=np.int64)

        # Последовательно разворачиваем формулу переключения для каждой промежуточной точки.
        for position in range(route_matrix.shape[1]):
            ids = route_matrix[:, position]
            intentions = self.candidate_intentions[ids]
            if position == 0:
                previous_states = np.repeat(state[None, :], count, axis=0)
            else:
                previous_states = self.candidate_observations[route_matrix[:, position - 1]]
            forward_previous = self._forward_batches(previous_states, intentions)
            forward_self = self.candidate_self_forward[:, ids, :]
            if len(forward_previous) != ensemble_size or len(forward_self) != ensemble_size:
                raise RuntimeError("forward ensemble size changed across evaluations")

            task_value_previous = np.einsum("end,d->en", forward_previous, reward)
            task_value_self = np.einsum("end,d->en", forward_self, reward)
            numerator = np.einsum("end,nd->en", forward_previous, intentions)
            denominator = self.candidate_self_measure[:, ids]
            step_valid = (
                np.isfinite(numerator)
                & np.isfinite(denominator)
                & (np.abs(denominator) >= self.eta_epsilon)
            )
            raw_eta = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=step_valid)
            eta = np.clip(raw_eta, 0.0, 1.0)
            step_valid &= eta >= self.minimum_eta
            clipped_counts += np.any(np.abs(raw_eta - eta) > 1e-7, axis=0).astype(np.int64)
            valid &= step_valid

            if position == 0:
                accumulated = task_value_previous.astype(np.float64)
            else:
                accumulated += discount * (task_value_previous - previous_self_task_value)
            discount *= eta
            previous_self_task_value = task_value_self
            etas.append(eta.mean(axis=0))

        last_states = self.candidate_observations[route_matrix[:, -1]]
        finish_intentions = np.repeat(terminal_policy[None, :], count, axis=0)
        terminal_forward = self._forward_batches(last_states, finish_intentions)
        terminal_task_value = np.einsum("end,d->en", terminal_forward, reward)
        # Последняя поправка переключает маршрут на неизменную конечную задачу.
        accumulated += discount * (terminal_task_value - previous_self_task_value)
        valid &= np.isfinite(accumulated)

        mean = accumulated.mean(axis=0)
        spread = accumulated.max(axis=0) - accumulated.min(axis=0)
        conservative = mean - self.disagreement_penalty * spread
        conservative = np.where(np.all(valid, axis=0), conservative, -np.inf)
        direct_mean = float(direct_by_member.mean())
        direct_spread = float(direct_by_member.max() - direct_by_member.min())
        direct_conservative = direct_mean - self.disagreement_penalty * direct_spread
        details = {
            "member_scores": accumulated,
            "mean_scores": mean,
            "score_spread": spread,
            "etas": np.stack(etas, axis=1),
            "valid": np.all(valid, axis=0),
            "clipped_counts": clipped_counts,
            "direct_score": direct_conservative,
            "direct_member_scores": direct_by_member,
        }
        return conservative.astype(np.float64), details

    def select(self, observation: Any, task_latent: Any) -> RouteSelection:
        started = time.perf_counter()
        observation = np.asarray(observation, dtype=np.float32)
        task_latent = np.asarray(task_latent, dtype=np.float32)
        start_embedding = np.asarray(self.geometry.encode(observation), dtype=np.float32)
        if start_embedding.shape != (EMBEDDING_DIM,):
            raise RuntimeError("geometry encoder returned an invalid current-state embedding")

        route: tuple[int, ...] = ()
        best_score = -np.inf
        direct_score = -np.inf
        best_details: dict[str, Any] = {}
        total_valid = 0
        considered = 0
        for _ in range(self.waypoint_count):
            expanded, surrogate = self._shortlist_insertions(start_embedding, route)
            if not expanded:
                return self._fallback(observation, task_latent, started, reason=1)
            exact, details = self.score_routes(observation, task_latent, expanded)
            considered += len(expanded)
            valid = np.isfinite(exact)
            total_valid += int(valid.sum())
            if not np.any(valid):
                return self._fallback(observation, task_latent, started, reason=2)
            selected = int(np.argmax(exact))
            route = expanded[selected]
            best_score = float(exact[selected])
            direct_score = float(details["direct_score"])
            best_details = {
                "etas": np.asarray(details["etas"])[selected],
                "score_spread": float(np.asarray(details["score_spread"])[selected]),
                "clipped_count": int(np.asarray(details["clipped_counts"])[selected]),
                "surrogate_score": float(surrogate[selected]),
            }

        if best_score < direct_score + self.min_improvement:
            return self._fallback(
                observation,
                task_latent,
                started,
                reason=3,
                plan_score=best_score,
                direct_score=direct_score,
                route=route,
            )

        diagnostics = self._diagnostics(
            route=route,
            plan_score=best_score,
            direct_score=direct_score,
            etas=best_details["etas"],
            score_spread=best_details["score_spread"],
            clipped_count=best_details["clipped_count"],
            surrogate_score=best_details["surrogate_score"],
            valid_count=total_valid,
            considered_count=considered,
            fallback_reason=0,
            elapsed=time.perf_counter() - started,
        )
        self._last_details = dict(diagnostics)
        return RouteSelection(
            intention=self.candidate_intentions[route[0]],
            diagnostics=diagnostics,
            route_indices=route,
            fallback=False,
        )

    def _fallback(
        self,
        observation: np.ndarray,
        task_latent: np.ndarray,
        started: float,
        *,
        reason: int,
        plan_score: float = np.nan,
        direct_score: float = np.nan,
        route: tuple[int, ...] = (),
    ) -> RouteSelection:
        del observation
        intention = normalize_intentions(task_latent, int(self.frozen_fb.latent_dim))
        diagnostics = self._diagnostics(
            route=route,
            plan_score=plan_score,
            direct_score=direct_score,
            etas=np.full(self.waypoint_count, np.nan),
            score_spread=np.nan,
            clipped_count=0,
            surrogate_score=np.nan,
            valid_count=0,
            considered_count=0,
            fallback_reason=reason,
            elapsed=time.perf_counter() - started,
        )
        self._last_details = dict(diagnostics)
        return RouteSelection(intention=intention, diagnostics=diagnostics, route_indices=route, fallback=True)

    def _diagnostics(
        self,
        *,
        route: tuple[int, ...],
        plan_score: float,
        direct_score: float,
        etas: np.ndarray,
        score_spread: float,
        clipped_count: int,
        surrogate_score: float,
        valid_count: int,
        considered_count: int,
        fallback_reason: int,
        elapsed: float,
    ) -> dict[str, Any]:
        route_ids = np.full(self.waypoint_count, -1, dtype=np.int64)
        source_ids = np.full(self.waypoint_count, -1, dtype=np.int64)
        waypoints = np.full((self.waypoint_count, EMBEDDING_DIM), np.nan, dtype=np.float32)
        waypoint_xy = np.full((self.waypoint_count, 2), np.nan, dtype=np.float32)
        supports = np.full(self.waypoint_count, np.nan, dtype=np.float32)
        grid_flags = np.zeros(self.waypoint_count, dtype=np.int64)
        for position, candidate in enumerate(route[: self.waypoint_count]):
            route_ids[position] = candidate
            anchor = self.candidate_anchor_indices[candidate]
            source_ids[position] = self.source_indices[anchor]
            waypoints[position] = self.candidate_embeddings[candidate]
            waypoint_xy[position] = self.candidate_observations[candidate, :2]
            supports[position] = self.candidate_support_distance[candidate]
            grid_flags[position] = int(self.candidate_is_grid[candidate])
        eta_values = np.full(self.waypoint_count, np.nan, dtype=np.float32)
        eta_values[: min(len(etas), self.waypoint_count)] = np.asarray(etas)[: self.waypoint_count]
        return {
            "latent3_route_indices": route_ids,
            "latent3_source_indices": source_ids,
            "latent3_waypoints": waypoints,
            "latent3_waypoint_xy": waypoint_xy,
            "latent3_support_distance": supports,
            "latent3_grid_flags": grid_flags,
            "latent3_etas": eta_values,
            "latent3_plan_score": np.float64(plan_score),
            "latent3_direct_score": np.float64(direct_score),
            "latent3_score_spread": np.float64(score_spread),
            "latent3_surrogate_score": np.float64(surrogate_score),
            "latent3_valid_routes": np.int64(valid_count),
            "latent3_considered_routes": np.int64(considered_count),
            "latent3_clipped_etas": np.int64(clipped_count),
            "latent3_fallback_reason": np.int64(fallback_reason),
            "latent3_planning_seconds": np.float64(elapsed),
        }

    def experiment_config(self) -> dict[str, Any]:
        return {
            "hypothesis": "dynamic_latent_three_waypoint",
            "planning_depth": self.waypoint_count,
            "execution_semantics": "execute_first_waypoint_then_replan",
            "embedding_dim": EMBEDDING_DIM,
            "intention_dim": int(self.frozen_fb.latent_dim),
            "intention_mode": self.intention_mode,
            "candidate_count": int(len(self.candidate_embeddings)),
            "grid_candidate_count": int(self.candidate_is_grid.sum()),
            "offline_candidate_count": int((~self.candidate_is_grid).sum()),
            "grid_resolution": self.grid_resolution,
            "support_radius": self.support_radius,
            "rerank_count": self.rerank_count,
            "fb_batch_size": self.fb_batch_size,
            "eta_epsilon": self.eta_epsilon,
            "minimum_eta": self.minimum_eta,
            "disagreement_penalty": self.disagreement_penalty,
            "min_improvement": self.min_improvement,
            "fixed_goal_xy": self.goal_xy.tolist(),
            "fixed_goal_anchor_source_index": int(self.source_indices[self.goal_anchor_index]),
            "terminal_policy": "normalize(original_fixed_task_latent)",
        }
