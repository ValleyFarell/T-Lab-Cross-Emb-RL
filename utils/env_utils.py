"""Подготовка среды, офлайн-наборов и разметки награды."""

from collections import deque
import re
import time

import gymnasium
from gymnasium.spaces import Box
import numpy as np
import ogbench

from utils.datasets import Dataset
from utils.reward_configs import complex_rewards_maze

class EpisodeMonitor(gymnasium.Wrapper):
    """Собирает статистику эпизода поверх исходной среды."""

    def __init__(self, env, filter_regexes=None):
        super().__init__(env)
        self._reset_stats()
        self.total_timesteps = 0
        self.filter_regexes = filter_regexes if filter_regexes is not None else []

    def _reset_stats(self):
        self.reward_sum = 0.0
        self.episode_length = 0
        self.start_time = time.time()

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)

        # Удаляем поля, не нужные для сохранения журнала.
        for filter_regex in self.filter_regexes:
            for key in list(info.keys()):
                if re.match(filter_regex, key) is not None:
                    del info[key]

        self.reward_sum += reward
        self.episode_length += 1
        self.total_timesteps += 1
        info['total'] = {'timesteps': self.total_timesteps}

        if terminated or truncated:
            info['episode'] = {}
            info['episode']['final_reward'] = reward
            info['episode']['return'] = self.reward_sum
            info['episode']['length'] = self.episode_length
            info['episode']['duration'] = time.time() - self.start_time

            if hasattr(self.unwrapped, 'get_normalized_score'):
                info['episode']['normalized_return'] = (
                    self.unwrapped.get_normalized_score(info['episode']['return']) * 100.0
                )

        return observation, reward, terminated, truncated, info

    def reset(self, *args, **kwargs):
        self._reset_stats()
        return self.env.reset(*args, **kwargs)


class FrameStackWrapper(gymnasium.Wrapper):
    """Объединяет несколько последовательных наблюдений среды."""

    def __init__(self, env, num_stack):
        super().__init__(env)

        self.num_stack = num_stack
        self.frames = deque(maxlen=num_stack)

        low = np.concatenate([self.observation_space.low] * num_stack, axis=-1)
        high = np.concatenate([self.observation_space.high] * num_stack, axis=-1)
        self.observation_space = Box(low=low, high=high, dtype=self.observation_space.dtype)

    def get_observation(self):
        assert len(self.frames) == self.num_stack
        return np.concatenate(list(self.frames), axis=-1)

    def reset(self, **kwargs):
        ob, info = self.env.reset(**kwargs)
        for _ in range(self.num_stack):
            self.frames.append(ob)
        if 'goal' in info:
            info['goal'] = np.concatenate([info['goal']] * self.num_stack, axis=-1)
        return self.get_observation(), info

    def step(self, action):
        ob, reward, terminated, truncated, info = self.env.step(action)
        self.frames.append(ob)
        return self.get_observation(), reward, terminated, truncated, info


def make_env_and_datasets(dataset_name, frame_stack=None,
                          env_only=False, dataset_only=False, 
                          action_clip_eps=1e-5, 
                          **kwargs):
    """Создаёт среду и загружает существующие офлайн-наборы.

    Параметры:
        dataset_name: параметр исходного вычисления.
        frame_stack: параметр исходного вычисления.
        env_only: параметр исходного вычисления.
        dataset_only: параметр исходного вычисления.
        action_clip_eps: параметр исходного вычисления.
        **kwargs: дополнительные именованные аргументы.
    """
    # Используем компактный набор данных для экономии памяти.
    if 'ogbench' in dataset_name:
        dataset_name = '-'.join(dataset_name.split('-')[1:])
        env_and_datasets = ogbench.make_env_and_datasets(
            dataset_name, compact_dataset=False, 
            env_only=env_only, dataset_only=dataset_only, 
            **kwargs
        )
    else:
        raise NotImplementedError

    if env_only:
        env = env_and_datasets
        env = EpisodeMonitor(env, filter_regexes=['.*privileged.*', '.*proprio.*', '.*timestep*.'])
    elif dataset_only:
        train_dataset, val_dataset = env_and_datasets
    else:
        env, train_dataset, val_dataset = env_and_datasets
        env = EpisodeMonitor(env, filter_regexes=['.*privileged.*', '.*proprio.*', '.*timestep*.'])

    if not dataset_only and frame_stack is not None:
        env = FrameStackWrapper(env, frame_stack)

    if env_only:
        env.reset()
        return env
    
    train_dataset = Dataset.create(**train_dataset)
    val_dataset = Dataset.create(**val_dataset)

    if isinstance(env.action_space, gymnasium.spaces.Box):
        assert np.all(env.action_space.low == -1.0)
        assert np.all(env.action_space.high == 1.0)
        
        # Ограничиваем действия набора допустимым диапазоном.
        eps = action_clip_eps
        train_dataset = train_dataset.copy(
            add_or_replace=dict(actions=np.clip(train_dataset['actions'], -1 + eps, 1 - eps))
        )
        val_dataset = val_dataset.copy(add_or_replace=dict(actions=np.clip(val_dataset['actions'], -1 + eps, 1 - eps)))

    if dataset_only:
        return train_dataset, val_dataset
    else:
        env.reset()
        return env, train_dataset, val_dataset


def relabel_dataset(env_name, env, dataset, complex_task_name=None):
    """Пересчитывает награды существующего набора для текущей цели среды.

    Параметры:
        env_name: параметр исходного вычисления.
        env: экземпляр проверочной среды.
        dataset: параметр исходного вычисления.
        complex_task_name: параметр исходного вычисления.
    """

    # Обрабатываем среды физического передвижения.
    qpos_xy_start_idx = 0
    qvel_xy_start_idx = 0
    goal_xy = env.unwrapped.cur_goal_xy
    goal_tol = env.unwrapped._goal_tol

    # Определяем, какие состояния достигают заданной цели.
    dists = np.linalg.norm(dataset['qpos'][:, qpos_xy_start_idx : qpos_xy_start_idx + 2] - goal_xy, axis=-1)
    successes = (dists <= goal_tol).astype(np.float32)

    if complex_task_name is not None:
        observations = {   
            "xy_pos": dataset['qpos'][:, qpos_xy_start_idx : qpos_xy_start_idx + 2],
            "xy_vel": dataset['qvel'][:, qvel_xy_start_idx : qvel_xy_start_idx + 2],
            }
        rewards = complex_rewards_maze(env, observations, env.unwrapped.cur_task_id, complex_task_name)
        masks = np.ones_like(rewards)
    else:
        rewards = successes  # Награда равна 1.0 при достижении цели и 0.0 в остальных случаях.
        masks = 1.0 - successes

    new_dataset = dataset.copy(
        add_or_replace=dict(
            rewards=rewards.astype(np.float32),
            masks=masks.astype(np.float32),
        )
    )

    return new_dataset
