"""Построение представления конечной задачи по правилам исходного агента."""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any

import numpy as np

from utils.datasets import Dataset
from utils.env_utils import relabel_dataset

from .frozen_fb import FrozenFB


class UnsupportedGoalError(RuntimeError):
    """Сообщает, что в офлайн-выборке нет состояний с положительной наградой."""


@dataclass(frozen=True)
class TaskEncoding:
    task_id: int | None
    latent: np.ndarray
    goal_xy: np.ndarray
    num_positive: int
    num_samples: int

    @property
    def supported(self) -> bool:
        return self.num_positive > 0


class TaskEncoder:
    """Строит представление конечной награды строго по правилам исходного агента."""

    def __init__(
        self,
        frozen_fb: FrozenFB,
        env,
        zero_shot_dataset: Dataset,
        *,
        env_name: str,
    ):
        self.frozen_fb = frozen_fb
        self.env = env
        self.zero_shot_dataset = zero_shot_dataset
        self.env_name = env_name

        dataset_module = importlib.import_module("utils.datasets")
        self.dataset_class = getattr(dataset_module, frozen_fb.config["dataset_class"])

    def encode_standard_task(
        self,
        task_id: int,
        *,
        require_support: bool = True,
    ) -> TaskEncoding:
        """Строит представление одной официальной задачи строго по исходному протоколу.

        Среда сначала сбрасывается на официальную цель, затем используются первые N офлайн-состояний и неизменённая функция infer_latent.
        """

        self._seed_action_space(0)
        # Сначала восстанавливаем официальную цель, затем размечаем тот же фиксированный офлайн-набор.
        self.env.reset(seed=0, options={"task_id": int(task_id)})
        return self._encode_current_goal(
            task_id=int(task_id),
            require_support=require_support,
        )

    def encode_custom_task(
        self,
        start_ij: tuple[int, int],
        goal_ij: tuple[int, int],
        *,
        require_support: bool = True,
    ) -> TaskEncoding:
        """Строит представление собственной пространственной цели лабиринта."""

        self._seed_action_space(0)
        self.env.reset(
            seed=0,
            options={
                "task_info": {
                    "init_ij": tuple(start_ij),
                    "goal_ij": tuple(goal_ij),
                }
            },
        )
        return self._encode_current_goal(
            task_id=None,
            require_support=require_support,
        )

    def _seed_action_space(self, seed: int) -> None:
        action_space = getattr(self.env, "action_space", None)
        if action_space is not None and hasattr(action_space, "seed"):
            action_space.seed(seed)

    def _encode_current_goal(
        self,
        *,
        task_id: int | None,
        require_support: bool,
    ) -> TaskEncoding:
        """Размечает фиксированную офлайн-выборку для текущей цели среды."""

        relabeled = relabel_dataset(
            self.env_name,
            self.env,
            self.zero_shot_dataset,
        )
        relabeled = self.dataset_class(Dataset.create(**relabeled), self.frozen_fb.config)

        configured_n = self.frozen_fb.config.get("num_zero_shot_samples")
        num_samples = int(configured_n if configured_n is not None else 100_000)
        if relabeled.size < num_samples:
            raise ValueError(
                f"Zero-shot dataset has {relabeled.size} states, "
                f"but reference inference requests {num_samples}."
            )

        # Берём первые N состояний: случайная подвыборка изменила бы представление задачи.
        batch = relabeled.sample(
            num_samples,
            idxs=np.arange(num_samples),
            relabeling=False,
            augmentation=False,
        )
        rewards = np.asarray(batch["rewards"])
        num_positive = int(np.count_nonzero(rewards))
        latent = np.asarray(self.frozen_fb.infer_task_latent(batch))
        goal_xy = np.asarray(self.env.unwrapped.cur_goal_xy, dtype=np.float64).copy()

        result = TaskEncoding(
            task_id=task_id,
            latent=latent,
            goal_xy=goal_xy,
            num_positive=num_positive,
            num_samples=num_samples,
        )
        # Нулевая поддержка цели дала бы бессодержательное намерение, поэтому останавливаем запуск явно.
        if require_support and not result.supported:
            task_name = f"Task {task_id}" if task_id is not None else "Custom goal"
            raise UnsupportedGoalError(
                f"{task_name} has N_g=0 in the exact {num_samples}-state "
                "zero-shot inference batch. The official code would silently "
                "produce a zero task latent; this runtime refuses to treat it "
                "as a normal supported goal."
            )
        return result
