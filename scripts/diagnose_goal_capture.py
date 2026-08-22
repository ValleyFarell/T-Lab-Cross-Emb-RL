"""Branch saved near-goal Ant states under several low-level intentions.

This diagnostic compares, from exactly the same qpos/qvel state:

1. the raw inferred reward/task latent;
2. the same latent normalized with the checkpoint's official normalize_z;
3. normalized B(s_goal) latents of real offline states inside the goal region.

Example (PowerShell):

    python -m scripts.diagnose_goal_capture `
      --source-results results_raw `
      --goal-xy 4 4 `
      --goal-state-count 8 `
      --horizon 100 `
      --output-dir results_goal_capture
"""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax
import numpy as np

# Compatibility with dependency combinations where OGBench still refers to
# the NumPy alias removed in newer versions.
np.in1d = np.isin

from baseline.frozen_fb import FrozenFB, load_checkpoint_config
from evaluation.goal_capture import (
    EntryState,
    find_first_entry_state,
    summarize_branch,
    velocity_components,
)
from utils.datasets import Dataset
from utils.env_utils import make_env_and_datasets


@dataclass(frozen=True)
class SourceRun:
    run_id: str
    run_dir: Path
    scenario_id: str
    environment_seed: int
    controller_seed: int
    task_id: int
    entry: EntryState


def parse_args():
    parser = argparse.ArgumentParser(
        description="Test local goal capture from saved full Ant observations."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("checkpoints/antmaze-medium-navigate-v0"),
    )
    parser.add_argument(
        "--source-results",
        type=Path,
        required=True,
        help="Directory containing saved trajectory.npz files (searched recursively).",
    )
    parser.add_argument("--goal-xy", type=float, nargs=2, required=True)
    parser.add_argument(
        "--task-id",
        type=int,
        default=1,
        help="Only initializes OGBench before the saved qpos/qvel is restored.",
    )
    parser.add_argument("--entry-radius", type=float, default=1.0)
    parser.add_argument("--success-radius", type=float, default=0.5)
    parser.add_argument("--horizon", type=int, default=100)
    parser.add_argument("--goal-state-count", type=int, default=8)
    parser.add_argument("--goal-state-seed", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--max-source-runs",
        type=int,
        default=None,
        help="Optional cap after deterministic run-path sorting.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results_goal_capture"),
    )
    return parser.parse_args()


def _json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _first_present(*values, default=None):
    for value in values:
        if value is not None:
            return value
    return default


def _safe_run_id(path: Path, root: Path) -> str:
    relative = path.relative_to(root).as_posix()
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", relative).strip("_") or "run"


def load_source_runs(args) -> tuple[list[SourceRun], list[dict]]:
    """Load one first-entry state from every usable saved trajectory."""

    sources: list[SourceRun] = []
    skipped: list[dict] = []
    trajectory_paths = sorted(args.source_results.rglob("trajectory.npz"))
    if not trajectory_paths:
        raise FileNotFoundError(
            f"No trajectory.npz found below {args.source_results}"
        )

    for trajectory_path in trajectory_paths:
        run_dir = trajectory_path.parent
        scenario = _json(run_dir / "scenario.json")
        summary = _json(run_dir / "summary.json")
        parent_config = _json(run_dir.parent.parent / "config.json")
        run_id = _safe_run_id(run_dir, args.source_results)

        with np.load(trajectory_path, allow_pickle=False) as trajectory:
            if "observations" not in trajectory:
                skipped.append({"run_id": run_id, "reason": "missing observations"})
                continue
            observations = np.asarray(trajectory["observations"])

        entry = find_first_entry_state(
            observations,
            args.goal_xy,
            entry_radius=args.entry_radius,
            success_radius=args.success_radius,
        )
        if entry is None:
            skipped.append(
                {"run_id": run_id, "reason": "no saved state in capture annulus"}
            )
            continue

        environment_seed = int(
            _first_present(
                scenario.get("environment_seed"),
                summary.get("environment_seed"),
                parent_config.get("environment_seed"),
                default=0,
            )
        )
        controller_seed = int(
            _first_present(
                scenario.get("controller_seed"),
                summary.get("controller_seed"),
                parent_config.get("controller_seed"),
                default=0,
            )
        )
        task_id = int(
            _first_present(
                scenario.get("task_id"),
                parent_config.get("task_id"),
                default=args.task_id,
            )
        )
        sources.append(
            SourceRun(
                run_id=run_id,
                run_dir=run_dir,
                scenario_id=str(scenario.get("scenario_id", run_id)),
                environment_seed=environment_seed,
                controller_seed=controller_seed,
                task_id=task_id,
                entry=entry,
            )
        )

    if args.max_source_runs is not None:
        sources = sources[: args.max_source_runs]
    if not sources:
        raise RuntimeError(
            "None of the trajectories contains a saved state in the requested "
            f"annulus ({args.success_radius}, {args.entry_radius})."
        )
    return sources, skipped


def build_runtime(checkpoint: Path):
    config, saved_flags = load_checkpoint_config(checkpoint)
    if config["frame_stack"] is not None:
        raise ValueError(
            "This exact-state diagnostic currently requires frame_stack=None."
        )
    env_name = saved_flags["env_name"]

    env, train_dataset, val_dataset = make_env_and_datasets(
        env_name,
        frame_stack=config["frame_stack"],
        add_info=True,
    )
    env.unwrapped._add_noise_to_goal = False
    train_dataset = Dataset.create(**train_dataset)
    val_dataset = Dataset.create(**val_dataset) if val_dataset is not None else None
    zero_shot_dataset = val_dataset if val_dataset is not None else train_dataset

    dataset_module = importlib.import_module("utils.datasets")
    dataset_class = getattr(dataset_module, config["dataset_class"])
    train_for_agent = dataset_class(train_dataset, config)
    example_batch = train_for_agent.sample(1)
    frozen_fb = FrozenFB.from_checkpoint(
        checkpoint,
        example_batch,
        config=config,
    )
    return env, zero_shot_dataset, dataset_class, frozen_fb, config, env_name


def infer_custom_task_latent(
    frozen_fb,
    zero_shot_dataset,
    dataset_class,
    config,
    goal_xy,
    success_radius,
):
    """Reproduce zero-shot inference after relabeling for a custom XY goal."""

    goal_xy = np.asarray(goal_xy, dtype=np.float64)
    qpos = np.asarray(zero_shot_dataset["qpos"])
    distances = np.linalg.norm(qpos[:, :2] - goal_xy, axis=1)
    rewards = (distances <= success_radius).astype(np.float32)
    masks = 1.0 - rewards
    relabeled = zero_shot_dataset.copy(
        add_or_replace={"rewards": rewards, "masks": masks}
    )
    relabeled = dataset_class(Dataset.create(**relabeled), config)

    configured_n = config.get("num_zero_shot_samples")
    num_samples = int(configured_n if configured_n is not None else 100_000)
    if relabeled.size < num_samples:
        raise ValueError(
            f"Zero-shot dataset has {relabeled.size} states, but {num_samples} "
            "are requested by the checkpoint."
        )
    batch = relabeled.sample(
        num_samples,
        idxs=np.arange(num_samples),
        relabeling=False,
        augmentation=False,
    )
    num_positive = int(np.count_nonzero(np.asarray(batch["rewards"])))
    if num_positive == 0:
        raise ValueError(
            "No positive zero-shot states for this custom goal. Choose a free "
            "cell represented in the offline dataset or increase the radius."
        )
    latent = np.asarray(frozen_fb.infer_task_latent(batch))
    return latent, num_positive, num_samples, distances


def make_intentions(
    frozen_fb,
    zero_shot_dataset,
    goal_distances,
    task_latent,
    *,
    success_radius,
    count,
    seed,
):
    """Create named intention vectors and provenance records."""

    if count < 1:
        raise ValueError("goal-state-count must be at least one")
    candidates = np.flatnonzero(goal_distances <= success_radius)
    if candidates.size == 0:
        raise ValueError("No real dataset states inside the requested goal region")

    rng = np.random.default_rng(seed)
    chosen = np.sort(
        rng.choice(candidates, size=min(count, candidates.size), replace=False)
    )

    intentions: list[tuple[str, str, np.ndarray, dict]] = [
        (
            "task_raw",
            "task_raw",
            np.asarray(task_latent),
            {"source": "inferred reward latent", "normalized": False},
        ),
        (
            "task_normalized",
            "task_normalized",
            np.asarray(frozen_fb.normalize_latent(task_latent)),
            {"source": "inferred reward latent", "normalized": True},
        ),
    ]

    for number, dataset_index in enumerate(chosen):
        observation = np.asarray(zero_shot_dataset["observations"][dataset_index])
        backward = np.asarray(frozen_fb.backward_repr(observation))
        intention = np.asarray(frozen_fb.normalize_latent(backward))
        intentions.append(
            (
                f"goal_state_{number:02d}",
                "goal_state",
                intention,
                {
                    "source": "normalize(B(s_goal))",
                    "normalized": True,
                    "dataset_index": int(dataset_index),
                    "dataset_goal_xy": np.asarray(
                        zero_shot_dataset["qpos"][dataset_index, :2]
                    ).tolist(),
                    "distance_to_requested_goal": float(goal_distances[dataset_index]),
                    "backward_latent_norm_before_normalization": float(
                        np.linalg.norm(backward)
                    ),
                },
            )
        )

    result = []
    for name, family, intention, provenance in intentions:
        if intention.ndim > 1:
            intention = np.squeeze(intention)
        result.append(
            {
                "name": name,
                "family": family,
                "intention": intention,
                "intention_norm": float(np.linalg.norm(intention)),
                "provenance": provenance,
            }
        )
    return result


def _current_observation(env) -> np.ndarray:
    base = env.unwrapped
    if hasattr(base, "get_ob"):
        return np.asarray(base.get_ob()).copy()
    if hasattr(base, "_get_obs"):
        return np.asarray(base._get_obs()).copy()
    raise AttributeError("The unwrapped environment has neither get_ob nor _get_obs")


def _set_custom_goal(env, goal_xy) -> None:
    base = env.unwrapped
    goal_xy = np.asarray(goal_xy, dtype=np.float64).copy()
    # OGBench reads cur_goal_xy for success/termination.  Slice assignment is
    # preferable when the environment keeps other references to this array.
    try:
        base.cur_goal_xy[...] = goal_xy
    except (AttributeError, TypeError, ValueError):
        base.cur_goal_xy = goal_xy


def restore_saved_ant_state(env, source: SourceRun, goal_xy) -> np.ndarray:
    """Reset wrappers, then replace physics with the saved qpos/qvel exactly."""

    random.seed(source.environment_seed)
    np.random.seed(source.environment_seed)
    if hasattr(env, "action_space") and hasattr(env.action_space, "seed"):
        env.action_space.seed(source.environment_seed)

    env.reset(
        seed=source.environment_seed,
        options={"task_id": source.task_id},
    )
    _set_custom_goal(env, goal_xy)

    base = env.unwrapped
    observation = np.asarray(source.entry.observation)
    nq = int(base.model.nq)
    nv = int(base.model.nv)
    if observation.shape != (nq + nv,):
        raise ValueError(
            f"Saved observation {source.run_id!r} has shape {observation.shape}; "
            f"exact Ant restoration expects ({nq + nv},) = nq + nv."
        )
    qpos = observation[:nq]
    qvel = observation[nq : nq + nv]
    base.set_state(qpos, qvel)
    restored = _current_observation(env)
    if not np.allclose(restored, observation, rtol=1e-7, atol=1e-7):
        error = float(np.max(np.abs(restored - observation)))
        raise RuntimeError(
            f"Exact state restoration failed for {source.run_id!r}; "
            f"max observation error is {error:.3e}."
        )
    return restored


def run_branch(env, frozen_fb, source, condition, args):
    observation = restore_saved_ant_state(env, source, args.goal_xy)
    goal_xy = np.asarray(args.goal_xy, dtype=np.float64)
    intention = np.asarray(condition["intention"])
    policy_rng = jax.random.PRNGKey(source.controller_seed)

    observations = [observation.copy()]
    positions = [observation[:2].astype(np.float64).copy()]
    actions = []
    radial_velocities = []
    tangential_speeds = []
    torso_heights = []
    terminated = False
    truncated = False

    for _ in range(args.horizon):
        qvel_xy = np.asarray(env.unwrapped.data.qvel[:2], dtype=np.float64)
        radial, tangential = velocity_components(
            observation[:2], qvel_xy, goal_xy
        )
        radial_velocities.append(radial)
        tangential_speeds.append(tangential)
        torso_heights.append(float(env.unwrapped.data.qpos[2]))

        # Reproduce the low-key stream used by EpisodeRunner.  Every condition
        # receives the same stream for this source state.
        policy_rng, step_key = jax.random.split(policy_rng)
        _, low_key = jax.random.split(step_key)
        action = np.asarray(
            frozen_fb.low_action(
                observation,
                intention,
                seed=low_key,
                temperature=args.temperature,
            )
        )
        observation, _, terminated, truncated, info = env.step(action)
        observation = np.asarray(observation)

        actions.append(action.copy())
        observations.append(observation.copy())
        positions.append(observation[:2].astype(np.float64).copy())

        distance = float(np.linalg.norm(observation[:2] - goal_xy))
        if distance <= args.success_radius or terminated or truncated:
            break

    metrics = summarize_branch(
        positions,
        radial_velocities,
        tangential_speeds,
        torso_heights,
        goal_xy,
        success_radius=args.success_radius,
        entry_radius=args.entry_radius,
    )
    metrics.update(
        {
            "source_run_id": source.run_id,
            "scenario_id": source.scenario_id,
            "source_observation_index": source.entry.observation_index,
            "environment_seed": source.environment_seed,
            "controller_seed": source.controller_seed,
            "condition": condition["name"],
            "condition_family": condition["family"],
            "intention_norm": condition["intention_norm"],
            "actions_executed": len(actions),
            "environment_terminated": bool(terminated),
            "environment_truncated": bool(truncated),
        }
    )
    arrays = {
        # Includes the final next_observation, unlike the old episode logger.
        "observations": np.asarray(observations),
        "positions": np.asarray(positions),
        "actions": np.asarray(actions),
        "intention": intention,
        "radial_velocity": np.asarray(radial_velocities),
        "tangential_speed": np.asarray(tangential_speeds),
        "torso_height": np.asarray(torso_heights),
        "distances": np.linalg.norm(np.asarray(positions) - goal_xy, axis=1),
    }
    return metrics, arrays


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(value, file, indent=2, ensure_ascii=False)


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def aggregate(rows: list[dict]) -> list[dict]:
    """Aggregate variants separately and all B(s_goal) variants as a family."""

    groups: dict[tuple[str, str], list[dict]] = {}
    for row in rows:
        groups.setdefault(("condition", row["condition"]), []).append(row)
        if row["condition_family"] == "goal_state":
            groups.setdefault(("family", "goal_state"), []).append(row)

    output = []
    for (level, name), group in sorted(groups.items()):
        hits = np.asarray([row["hit_success_radius"] for row in group], dtype=float)
        minimum = np.asarray([row["minimum_distance"] for row in group], dtype=float)
        exits = np.asarray([row["radius_exits"] for row in group], dtype=float)
        radial = np.asarray([row["mean_radial_velocity"] for row in group], dtype=float)
        tangential = np.asarray(
            [row["mean_tangential_speed"] for row in group], dtype=float
        )
        hit_steps = [row["hit_step"] for row in group if row["hit_step"] is not None]
        output.append(
            {
                "level": level,
                "name": name,
                "branches": len(group),
                "distinct_source_states": len(
                    {row["source_run_id"] for row in group}
                ),
                "hit_rate": float(hits.mean()),
                "median_hit_step_among_hits": (
                    float(np.median(hit_steps)) if hit_steps else None
                ),
                "mean_minimum_distance": float(minimum.mean()),
                "median_minimum_distance": float(np.median(minimum)),
                "mean_radius_exits": float(exits.mean()),
                "mean_radial_velocity": float(radial.mean()),
                "mean_tangential_speed": float(tangential.mean()),
            }
        )
    return output


def main():
    args = parse_args()
    if args.horizon < 1:
        raise ValueError("horizon must be positive")
    if not 0 < args.success_radius < args.entry_radius:
        raise ValueError("expected 0 < success-radius < entry-radius")

    sources, skipped = load_source_runs(args)
    (
        env,
        zero_shot_dataset,
        dataset_class,
        frozen_fb,
        config,
        env_name,
    ) = build_runtime(args.checkpoint)

    task_latent, num_positive, num_samples, goal_distances = (
        infer_custom_task_latent(
            frozen_fb,
            zero_shot_dataset,
            dataset_class,
            config,
            args.goal_xy,
            args.success_radius,
        )
    )
    conditions = make_intentions(
        frozen_fb,
        zero_shot_dataset,
        goal_distances,
        task_latent,
        success_radius=args.success_radius,
        count=args.goal_state_count,
        seed=args.goal_state_seed,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        args.output_dir / "config.json",
        {
            "environment": env_name,
            "checkpoint": str(args.checkpoint),
            "source_results": str(args.source_results),
            "goal_xy": list(map(float, args.goal_xy)),
            "entry_radius": args.entry_radius,
            "success_radius": args.success_radius,
            "horizon": args.horizon,
            "temperature": args.temperature,
            "goal_state_count": args.goal_state_count,
            "goal_state_seed": args.goal_state_seed,
            "num_source_states": len(sources),
            "N_g": num_positive,
            "N_samples": num_samples,
            "task_latent_norm": float(np.linalg.norm(task_latent)),
            "conditions": [
                {
                    key: value
                    for key, value in condition.items()
                    if key != "intention"
                }
                for condition in conditions
            ],
            "skipped_sources": skipped,
            "statistical_note": (
                "goal_state variants from one source state are repeated "
                "interventions, not independent environment seeds"
            ),
        },
    )

    rows = []
    for source in sources:
        for condition in conditions:
            metrics, arrays = run_branch(env, frozen_fb, source, condition, args)
            branch_dir = (
                args.output_dir
                / "branches"
                / source.run_id
                / condition["name"]
            )
            branch_dir.mkdir(parents=True, exist_ok=True)
            write_json(branch_dir / "summary.json", metrics)
            np.savez_compressed(branch_dir / "trajectory.npz", **arrays)
            rows.append(metrics)
            print(
                f"{source.run_id} | {condition['name']} | "
                f"hit={metrics['hit_success_radius']} | "
                f"min_d={metrics['minimum_distance']:.4f} | "
                f"exits={metrics['radius_exits']}"
            )

    aggregate_rows = aggregate(rows)
    write_csv(args.output_dir / "branches.csv", rows)
    write_csv(args.output_dir / "aggregate.csv", aggregate_rows)
    write_json(args.output_dir / "aggregate.json", aggregate_rows)
    print(f"saved_to: {args.output_dir}")


if __name__ == "__main__":
    main()
