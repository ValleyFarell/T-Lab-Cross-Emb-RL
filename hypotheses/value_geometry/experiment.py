"""Полный эксперимент геометрии ценности и проверка на искусственных данных."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from datetime import datetime
import importlib
import json
from pathlib import Path
import time
from typing import Any, Mapping

import numpy as np

from .analysis import (
    compare_model_signals,
    distance_stratified_metrics,
    embedding_diagnostics,
    matched_goal_pose_diagnostics,
    ranking_metrics,
    regression_metrics,
)
from .data import (
    SPLIT_NAMES,
    MazeGeometry,
    PairSplit,
    StatePool,
    build_state_pool,
    estimate_distance_edges,
    sample_pairs,
    select_goal_indices,
    split_summary,
)
from .models import TrainingConfig, fit_value_model, latent_dimension
from .plots import generate_plots
from .teacher import OfflineFBTeacher


@dataclass(frozen=True)
class ExperimentConfig:
    checkpoint: str = "checkpoints/antmaze-medium-navigate-v0"
    output_dir: str = "artifacts/value_geometry"
    target_mode: str = "xy-goal"
    task_id: int = 4
    seed: int = 0
    model_seeds: tuple[int, ...] = (0,)
    max_states: int = 18_000
    train_pairs: int = 20_000
    goal_count: int = 64
    goal_variants: int = 2
    pose_radius: float = 0.20
    candidates_per_start: int = 8
    teacher_batch_size: int = 128
    reference_samples: int = 100_000
    disagreement_penalty: float = 0.5
    distance_mode: str = "auto"
    distance_bins: int = 4
    models: tuple[str, ...] = ("xy", "full", "latent2", "latent4")
    training: TrainingConfig = TrainingConfig()
    synthetic: bool = False
    synthetic_states: int = 3_000
    resume: bool = False
    plots: bool = True


def _progress(message: str) -> None:
    stamp = datetime.now().strftime("%H:%M:%S")
    memory = ""
    try:
        import psutil

        gigabytes = psutil.Process().memory_info().rss / 1024**3
        memory = f" | RAM {gigabytes:.2f} GB"
    except ImportError:
        pass
    print(f"[{stamp}] {message}{memory}", flush=True)


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"unsupported JSON value: {type(value)!r}")


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, ensure_ascii=False, default=_json_safe)


def _make_synthetic_source(count: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    if count < 90:
        raise ValueError("synthetic_states must be at least 90")
    rng = np.random.default_rng(seed)
    length = 30
    trajectories = int(np.ceil(count / length))
    observations = np.empty((trajectories * length, 29), dtype=np.float32)
    terminals = np.zeros(trajectories * length, dtype=np.int8)
    for trajectory in range(trajectories):
        first = trajectory * length
        # Независимые целые траектории покрывают ту же область координат,
        # поэтому проверка на непересекающихся траекториях остаётся осмысленной.
        center = rng.uniform(-4.5, 4.5, size=2)
        walk = rng.normal(0, 0.45, size=(length, 2)).cumsum(axis=0)
        xy = np.clip(center[None, :] + walk, -6.0, 6.0)
        pose = rng.normal(0, 1, size=(length, 27))
        pose[:, 0] = rng.uniform(0.40, 0.85, size=length)
        observations[first : first + length, :2] = xy
        observations[first : first + length, 2:] = pose
        terminals[first + length - 1] = 1
    return observations[:count], terminals[:count]


def _synthetic_teacher(pool: StatePool, pairs: PairSplit) -> PairSplit:
    starts = pool.observations[pairs.start_indices]
    goals = pool.observations[pairs.goal_indices]
    difference = goals[:, :2] - starts[:, :2]
    distance = np.linalg.norm(difference, axis=1)
    # Основную ценность задают координаты, а влияние позы специально сделано локальным.
    value = (
        np.exp(-distance / 5.0)
        + 0.12 * np.tanh(difference[:, 0] / 2.0)
        - 0.08 * np.tanh(difference[:, 1] / 2.5)
        + 0.06 * np.sin(starts[:, 0] / 2.0)
        + 0.10
        * np.exp(-distance / 1.2)
        * np.tanh(starts[:, 3] - goals[:, 3])
    ).astype(np.float32)
    disagreement = 0.015 * np.abs(np.tanh(starts[:, 4] - goals[:, 4]))
    members = np.column_stack((value, value + disagreement)).astype(np.float32)
    return pairs.with_values(value, members)


def _real_project_source(config: ExperimentConfig) -> dict[str, Any]:
    # Сохраняем совместимость исходного проекта с новыми версиями NumPy.
    np.in1d = np.isin
    _progress("Importing existing FB checkpoint and OGBench modules")
    try:
        from baseline.frozen_fb import FrozenFB, load_checkpoint_config
        from utils.datasets import Dataset
        from utils.env_utils import make_env_and_datasets
    except ImportError as exc:
        raise RuntimeError(
            "Cannot import the existing experiment project. Run from its root, "
            "using the same Python environment as scripts.run_baseline. "
            "Use --synthetic for a dependency-free smoke test."
        ) from exc

    checkpoint = Path(config.checkpoint)
    for required in (checkpoint / "flags.json", checkpoint / "params.pkl"):
        if not required.is_file():
            raise FileNotFoundError(f"Required checkpoint file not found: {required}")
    agent_config, saved_flags = load_checkpoint_config(checkpoint)
    environment_name = saved_flags["env_name"]
    _progress(f"Loading offline OGBench data for {environment_name}")
    environment, raw_train, raw_validation = make_env_and_datasets(
        environment_name,
        frame_stack=agent_config["frame_stack"],
        add_info=True,
    )
    train_dataset = Dataset.create(**raw_train)
    validation_dataset = (
        Dataset.create(**raw_validation)
        if raw_validation is not None
        else train_dataset
    )
    dataset_module = importlib.import_module("utils.datasets")
    dataset_class = getattr(dataset_module, agent_config["dataset_class"])
    np.random.seed(config.seed)
    example_batch = dataset_class(train_dataset, agent_config).sample(1)
    _progress("Restoring frozen F, B and policy parameters")
    frozen_fb = FrozenFB.from_checkpoint(
        checkpoint,
        example_batch,
        config=agent_config,
    )
    observations = np.asarray(train_dataset["observations"], dtype=np.float32)
    qpos = train_dataset.get("qpos")
    positions = (
        np.asarray(qpos[:, :2], dtype=np.float32)
        if qpos is not None
        else observations[:, :2]
    )
    reference_observations = np.asarray(
        validation_dataset["observations"], dtype=np.float32
    )
    reference_qpos = validation_dataset.get("qpos")
    reference_positions = (
        np.asarray(reference_qpos[:, :2], dtype=np.float32)
        if reference_qpos is not None
        else reference_observations[:, :2]
    )
    fixed_goal = None
    if config.target_mode == "fixed-task":
        environment.reset(seed=0, options={"task_id": int(config.task_id)})
        fixed_goal = np.asarray(environment.unwrapped.cur_goal_xy, dtype=np.float32)
    tolerance = float(getattr(environment.unwrapped, "_goal_tol", 0.5))
    return {
        "environment": environment,
        "environment_name": environment_name,
        "frozen_fb": frozen_fb,
        "observations": observations,
        "positions": positions,
        "terminals": train_dataset.get("terminals"),
        "reference_observations": reference_observations,
        "reference_positions": reference_positions,
        "goal_tolerance": tolerance,
        "fixed_goal_xy": fixed_goal,
    }


def _goal_counts(config: ExperimentConfig, pool: StatePool) -> dict[str, int]:
    return {
        "train": min(config.goal_count, len(pool.split_indices["train"])),
        "validation": min(
            max(8, config.goal_count // 3), len(pool.split_indices["validation"])
        ),
        "test": min(
            max(8, config.goal_count // 3), len(pool.split_indices["test"])
        ),
    }


def _pair_counts(config: ExperimentConfig) -> dict[str, int]:
    held_out = max(256, int(np.ceil(config.train_pairs * 0.20)))
    return {
        "train": config.train_pairs,
        "validation": held_out,
        "test": held_out,
    }


def _cache_arrays(
    pool: StatePool,
    pairs: Mapping[str, PairSplit],
    edges: np.ndarray,
) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {
        "observations": pool.observations,
        "positions": pool.positions,
        "original_indices": pool.original_indices,
        "trajectory_groups": pool.trajectory_groups,
        "distance_edges": np.asarray(edges, dtype=np.float32),
        "split_strategy": np.asarray(pool.split_strategy),
    }
    for name in SPLIT_NAMES:
        split = pairs[name]
        arrays[f"{name}_states"] = np.asarray(pool.split_indices[name])
        arrays[f"{name}_starts"] = np.asarray(split.start_indices)
        arrays[f"{name}_goals"] = np.asarray(split.goal_indices)
        arrays[f"{name}_distances"] = np.asarray(split.distances)
        arrays[f"{name}_bins"] = np.asarray(split.distance_bins)
        arrays[f"{name}_values"] = np.asarray(split.values)
        if split.ensemble_values is not None:
            arrays[f"{name}_ensemble"] = np.asarray(split.ensemble_values)
    return arrays


def _load_pair_cache(path: Path) -> tuple[StatePool, dict[str, PairSplit], np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        split_indices = {name: data[f"{name}_states"].copy() for name in SPLIT_NAMES}
        pool = StatePool(
            observations=data["observations"].copy(),
            positions=data["positions"].copy(),
            original_indices=data["original_indices"].copy(),
            trajectory_groups=data["trajectory_groups"].copy(),
            split_indices=split_indices,
            split_strategy=str(data["split_strategy"].item()),
        )
        result = {}
        for name in SPLIT_NAMES:
            ensemble_name = f"{name}_ensemble"
            result[name] = PairSplit(
                start_indices=data[f"{name}_starts"].copy(),
                goal_indices=data[f"{name}_goals"].copy(),
                distances=data[f"{name}_distances"].copy(),
                distance_bins=data[f"{name}_bins"].copy(),
                values=data[f"{name}_values"].copy(),
                ensemble_values=(
                    data[ensemble_name].copy() if ensemble_name in data.files else None
                ),
            )
        edges = data["distance_edges"].copy()
    return pool, result, edges


def _create_pairs(
    config: ExperimentConfig,
) -> tuple[StatePool, dict[str, PairSplit], np.ndarray, dict[str, Any]]:
    source = None
    if config.synthetic:
        observations, terminals = _make_synthetic_source(
            config.synthetic_states, config.seed
        )
        positions = observations[:, :2]
        metadata: dict[str, Any] = {
            "source": "synthetic",
            "environment": "synthetic-value-geometry",
            "target_formula": (
                "directional XY reachability plus pose dependence "
                "that decays with Euclidean distance"
            ),
        }
    else:
        source = _real_project_source(config)
        observations = source["observations"]
        positions = source["positions"]
        terminals = source["terminals"]
        metadata = {
            "source": "frozen_fb_checkpoint",
            "environment": source["environment_name"],
            "checkpoint": config.checkpoint,
        }

    _progress("Splitting complete offline trajectories before constructing pairs")
    pool = build_state_pool(
        observations,
        positions=positions,
        terminals=terminals,
        max_states=min(config.max_states, len(observations)),
        seed=config.seed,
    )
    geometry = None
    if source is not None and config.distance_mode != "euclidean":
        try:
            geometry = MazeGeometry.from_environment(source["environment"], pool.positions)
            metadata["distance_source"] = geometry.source
            _progress(
                f"Built diagnostic-only static maze graph with {len(geometry.centers)} free cells"
            )
        except (AttributeError, TypeError, ValueError, IndexError) as exc:
            if config.distance_mode == "maze":
                raise RuntimeError("Could not build the requested maze-distance graph") from exc
            _progress(f"Maze graph unavailable ({exc}); using Euclidean distance bins")
    if geometry is None:
        metadata["distance_source"] = "euclidean"

    counts = _goal_counts(config, pool)
    goal_indices = {
        name: select_goal_indices(
            pool,
            name,
            count=counts[name],
            seed=config.seed + 101 * (index + 1),
            pose_radius=config.pose_radius,
            variants_per_location=config.goal_variants,
        )
        for index, name in enumerate(SPLIT_NAMES)
    }

    teacher = None
    banks = None
    if source is not None:
        teacher = OfflineFBTeacher(
            source["frozen_fb"],
            pool,
            source["reference_observations"],
            source["reference_positions"],
            goal_tolerance=source["goal_tolerance"],
            target_mode=config.target_mode,
            fixed_goal_xy=source["fixed_goal_xy"],
            disagreement_penalty=config.disagreement_penalty,
            batch_size=config.teacher_batch_size,
            reference_samples=config.reference_samples,
            progress=_progress,
        )
        _progress("Preparing policy intentions and exact offline XY reward latents")
        banks = teacher.prepare_goal_banks(goal_indices)
        goal_indices = {name: bank.goal_indices for name, bank in banks.items()}
        metadata["teacher"] = teacher.description()
        metadata["goal_banks"] = {
            name: {
                "count": int(len(bank.goal_indices)),
                "support_min": int(bank.support_sizes.min()),
                "support_median": float(np.median(bank.support_sizes)),
                "support_max": int(bank.support_sizes.max()),
            }
            for name, bank in banks.items()
        }

    edges = estimate_distance_edges(
        pool,
        pool.split_indices["train"],
        goal_indices["train"],
        geometry=geometry,
        number_of_bins=config.distance_bins,
        seed=config.seed,
    )
    metadata["distance_edges"] = edges.tolist()
    counts = _pair_counts(config)
    pairs = {}
    for index, name in enumerate(SPLIT_NAMES):
        _progress(f"Sampling {counts[name]} {name} state-goal pairs")
        sampled = sample_pairs(
            pool,
            name,
            goal_indices[name],
            number_of_pairs=counts[name],
            distance_edges=edges,
            geometry=geometry,
            seed=config.seed + 307 * (index + 1),
            candidates_per_start=config.candidates_per_start,
            matched_goal_radius=max(config.pose_radius, 0.15),
        )
        if teacher is None:
            pairs[name] = _synthetic_teacher(pool, sampled)
        else:
            _progress(f"Evaluating frozen FB teacher for {name} pairs")
            pairs[name] = teacher.score_pairs(sampled, banks[name])

    if source is not None:
        try:
            source["environment"].close()
        except (AttributeError, RuntimeError):
            pass
    metadata["splits"] = split_summary(pool, pairs)
    return pool, pairs, edges, metadata


def _csv_summary(path: Path, runs: Mapping[str, Any]) -> None:
    columns = (
        "seed",
        "model",
        "test_r2",
        "test_rmse",
        "test_nrmse",
        "far_r2",
        "mean_spearman",
        "top1_agreement",
        "mean_relative_regret",
        "xy_readout_r2",
        "xy_readout_p90",
        "best_epoch",
    )
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for seed, run in runs.items():
            far_bin = str(run["signals"]["far_distance_bin"])
            for name, result in run["models"].items():
                readout = result.get("embedding", {}).get(
                    "coordinate_readout_polynomial", {}
                )
                writer.writerow(
                    {
                        "seed": seed,
                        "model": name,
                        "test_r2": result["overall"].get("r2"),
                        "test_rmse": result["overall"].get("rmse"),
                        "test_nrmse": result["overall"].get("nrmse"),
                        "far_r2": result["by_distance"].get(far_bin, {}).get("r2"),
                        "mean_spearman": result["ranking"].get("mean_spearman"),
                        "top1_agreement": result["ranking"].get("top1_agreement"),
                        "mean_relative_regret": result["ranking"].get(
                            "mean_relative_regret"
                        ),
                        "xy_readout_r2": readout.get("r2"),
                        "xy_readout_p90": readout.get("p90_euclidean_error"),
                        "best_epoch": result.get("best_epoch"),
                    }
                )


def run_experiment(config: ExperimentConfig) -> dict[str, Any]:
    config.training.validate()
    if config.train_pairs < 20:
        raise ValueError("train_pairs must be at least 20")
    if config.goal_count < 2:
        raise ValueError("goal_count must be at least 2")
    if not config.model_seeds:
        raise ValueError("model_seeds cannot be empty")
    for model_name in config.models:
        latent_dimension(model_name)

    started = time.perf_counter()
    output = Path(config.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    cache_path = output / "pair_cache.npz"
    metadata_path = output / "dataset_metadata.json"
    if config.resume:
        if not cache_path.is_file():
            raise FileNotFoundError(f"--resume requested but cache is missing: {cache_path}")
        _progress(f"Loading cached labels from {cache_path}")
        pool, pairs, edges = _load_pair_cache(cache_path)
        with metadata_path.open("r", encoding="utf-8") as stream:
            metadata = json.load(stream)
        metadata["resumed_from_cached_teacher_labels"] = True
    else:
        pool, pairs, edges, metadata = _create_pairs(config)
        _progress(f"Saving reusable teacher-label cache to {cache_path}")
        np.savez_compressed(cache_path, **_cache_arrays(pool, pairs, edges))
        _write_json(metadata_path, metadata)

    experiment_config = asdict(config)
    _write_json(output / "experiment_config.json", experiment_config)
    runs: dict[str, Any] = {}
    for model_seed in config.model_seeds:
        _progress(f"Starting model comparison for training seed {model_seed}")
        run_dir = output / f"seed_{model_seed}"
        run_dir.mkdir(parents=True, exist_ok=True)
        model_metrics: dict[str, Any] = {}
        predictions: dict[str, np.ndarray] = {}
        embedding_arrays: dict[str, dict[str, np.ndarray]] = {}
        histories: dict[str, list[dict[str, Any]]] = {}

        for model_index, name in enumerate(config.models):
            _progress(f"Training {name} using CPU NumPy")
            model = fit_value_model(
                name,
                pool.observations,
                pairs["train"],
                pairs["validation"],
                train_state_indices=pool.split_indices["train"],
                config=config.training,
                seed=int(model_seed) + 10_007 * (model_index + 1),
                progress=_progress,
            )
            predicted = model.predict(
                pool.observations,
                pairs["test"].start_indices,
                pairs["test"].goal_indices,
            )
            predictions[name] = predicted
            histories[name] = model.history
            result: dict[str, Any] = {
                "overall": regression_metrics(pairs["test"].values, predicted),
                "by_distance": distance_stratified_metrics(
                    pairs["test"].values,
                    predicted,
                    pairs["test"].distance_bins,
                ),
                "ranking": ranking_metrics(pairs["test"], predicted),
                "matched_goal_pose": matched_goal_pose_diagnostics(
                    pairs["test"],
                    pool.positions,
                    predicted,
                    bucket_radius=max(config.pose_radius, 0.15),
                ),
                "best_epoch": model.best_epoch,
                "epochs_ran": len(model.history),
            }
            if model.embedding_dimension is not None:
                _progress(f"Evaluating held-out coordinate probes for {name}")
                embedding, arrays = embedding_diagnostics(
                    model,
                    pool.observations,
                    pool.positions,
                    pool.split_indices,
                )
                result["embedding"] = embedding
                embedding_arrays[name] = arrays
                np.savez_compressed(run_dir / f"{name}_embeddings.npz", **arrays)
            model.save(run_dir / "models" / f"{name}.npz")
            model_metrics[name] = result
            _progress(
                f"{name}: test R2={result['overall']['r2']:.4f}, "
                f"RMSE={result['overall']['rmse']:.6f}, "
                f"top-1={result['ranking'].get('top1_agreement')}"
            )

        np.savez_compressed(
            run_dir / "test_predictions.npz",
            teacher=np.asarray(pairs["test"].values),
            distance_bins=np.asarray(pairs["test"].distance_bins),
            distances=np.asarray(pairs["test"].distances),
            **predictions,
        )
        far_bin = int(np.max(pairs["test"].distance_bins))
        signals = compare_model_signals(model_metrics, far_bin=far_bin)
        created_plots = []
        if config.plots:
            _progress("Generating held-out value and embedding-geometry plots")
            created_plots = generate_plots(
                run_dir / "plots",
                metrics=model_metrics,
                predictions=predictions,
                values=pairs["test"].values,
                distance_bins=pairs["test"].distance_bins,
                embedding_arrays=embedding_arrays,
                histories=histories,
                seed=model_seed,
            )
            if not created_plots:
                _progress("matplotlib is unavailable; metrics and arrays were still saved")
        run_summary = {
            "model_seed": int(model_seed),
            "models": model_metrics,
            "signals": signals,
            "plots": created_plots,
        }
        runs[str(model_seed)] = run_summary
        _write_json(run_dir / "metrics.json", run_summary)

    elapsed = time.perf_counter() - started
    summary = {
        "experiment": "frozen_fb_value_geometry",
        "elapsed_seconds": float(elapsed),
        "dataset": metadata,
        "training_config": asdict(config.training),
        "runs": runs,
        "limitations": [
            "The target is a frozen learned FB estimate, not a measured environment return.",
            "The maze map, when available, is used only for diagnostic distance bins.",
            "No environment transitions are collected and frozen checkpoint weights are unchanged.",
            "Post-hoc coordinate probes are not part of the encoder training objective.",
        ],
    }
    _write_json(output / "metrics.json", summary)
    _csv_summary(output / "model_comparison.csv", runs)
    _progress(f"Experiment finished in {elapsed:.1f}s; results: {output}")
    return summary
