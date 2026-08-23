"""Офлайн-наборы, выбор переходов, целей и разбиение траекторий."""

import dataclasses
from functools import partial
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from flax.core.frozen_dict import FrozenDict


def get_size(data):
    """Возвращает число записей набора данных."""
    sizes = jax.tree_util.tree_map(lambda arr: len(arr), data)
    return max(jax.tree_util.tree_leaves(sizes))


@partial(jax.jit, static_argnames=('padding',))
def random_crop(img, crop_from, padding):
    """Вырезает случайную область изображения с заданным дополнением границ.

    Параметры:
        img: исходное изображение.
        crop_from: координаты начала вырезаемой области.
        padding: размер дополнения границ изображения.
    """
    padded_img = jnp.pad(img, ((padding, padding), (padding, padding), (0, 0)), mode='edge')
    return jax.lax.dynamic_slice(padded_img, crop_from, img.shape)


@partial(jax.jit, static_argnames=('padding',))
def batched_random_crop(imgs, crop_froms, padding):
    """Применяет случайное кадрирование сразу к блоку изображений."""
    return jax.vmap(random_crop, (0, 0, None))(imgs, crop_froms, padding)


class Dataset(FrozenDict):
    """Хранит офлайн-переходы и при необходимости восстанавливает соседние наблюдения.

    Поддерживаются полные переходы и компактные наборы наблюдений. Если следующее состояние не сохранено отдельно, оно восстанавливается по соседнему индексу с учётом границ траекторий.
    """

    @classmethod
    def create(cls, freeze=True, **fields):
        """Создаёт экземпляр агента или структуры данных с заданной конфигурацией.

        Параметры:
            freeze: признак защиты массивов от изменения.
            **fields: дополнительные именованные аргументы.
        """
        data = fields
        assert 'observations' in data
        if freeze:
            jax.tree_util.tree_map(lambda arr: arr.setflags(write=False), data)
        return cls(data)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.size = get_size(self._dict)
        if 'valids' in self._dict:
            (self.valid_idxs,) = np.nonzero(self['valids'] > 0)

    def get_random_idxs(self, num_idxs):
        """Выбирает заданное количество допустимых случайных индексов."""
        if 'valids' in self._dict:
            return self.valid_idxs[np.random.randint(len(self.valid_idxs), size=num_idxs)]
        else:
            return np.random.randint(self.size, size=num_idxs)

    def sample(self, batch_size, idxs=None):
        """Выбирает блок переходов или состояний из исходного набора данных."""
        if idxs is None:
            idxs = self.get_random_idxs(batch_size)
        return self.get_subset(idxs)

    def get_subset(self, idxs):
        """Выбирает подмножество переходов по указанным индексам."""
        result = jax.tree_util.tree_map(lambda arr: arr[idxs], self._dict)
        if 'next_observations' not in result:
            result['next_observations'] = self._dict['observations'][np.minimum(idxs + 1, self.size - 1)]
        if 'prev_actions' not in result:
            result['prev_actions'] = self._dict['actions'][np.minimum(np.maximum(idxs - 1, 0), self.size - 1)]
        if 'next_actions' not in result:
            result['next_actions'] = self._dict['actions'][np.minimum(idxs + 1, self.size - 1)]
        return result


class ReplayBuffer(Dataset):
    """Хранит переходы и позволяет добавлять новые записи."""

    @classmethod
    def create(cls, transition, size):
        """Создаёт экземпляр агента или структуры данных с заданной конфигурацией.

        Параметры:
            transition: пример одного перехода среды.
            size: максимальный размер создаваемой структуры.
        """

        def create_buffer(example):
            example = np.array(example)
            return np.zeros((size, *example.shape), dtype=example.dtype)

        buffer_dict = jax.tree_util.tree_map(create_buffer, transition)
        return cls(buffer_dict)

    @classmethod
    def create_from_initial_dataset(cls, init_dataset, size):
        """Создаёт хранилище переходов на основе существующего набора данных.

        Параметры:
            init_dataset: существующий набор переходов.
            size: максимальный размер создаваемой структуры.
        """

        def create_buffer(init_buffer):
            buffer = np.zeros((size, *init_buffer.shape[1:]), dtype=init_buffer.dtype)
            buffer[: len(init_buffer)] = init_buffer
            return buffer

        buffer_dict = jax.tree_util.tree_map(create_buffer, init_dataset)
        dataset = cls(buffer_dict)
        dataset.size = dataset.pointer = get_size(init_dataset)
        return dataset

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.max_size = get_size(self._dict)
        self.size = 0
        self.pointer = 0

    def add_transition(self, transition):
        """Добавляет один переход в хранилище опыта."""

        def set_idx(buffer, new_element):
            buffer[self.pointer] = new_element

        jax.tree_util.tree_map(set_idx, self._dict, transition)
        self.pointer = (self.pointer + 1) % self.max_size
        self.size = max(self.pointer, self.size)

    def clear(self):
        """Очищает содержимое хранилища переходов."""
        self.size = self.pointer = 0


@dataclasses.dataclass
class GCDataset:
    """Выбирает переходы и цели из текущего, будущих и случайных состояний.

    Цели выбираются из текущего состояния, будущих состояний той же траектории и случайных состояний. Вероятности и обработка кадров задаются сохранённой конфигурацией агента.
    """

    dataset: Dataset
    config: Any
    preprocess_frame_stack: bool = True

    def __post_init__(self):
        self.size = self.dataset.size

        # Заранее находим границы исходных траекторий.
        (self.terminal_locs,) = np.nonzero(self.dataset['terminals'] > 0)
        self.initial_locs = np.concatenate([[0], self.terminal_locs[:-1] + 1])
        assert self.terminal_locs[-1] == self.size - 1

        # Проверяем, что сумма вероятностей равна единице.
        assert np.isclose(
            self.config['value_p_curgoal'] + self.config['value_p_trajgoal'] + self.config['value_p_randomgoal'], 1.0
        )
        assert np.isclose(
            self.config['actor_p_curgoal'] + self.config['actor_p_trajgoal'] + self.config['actor_p_randomgoal'], 1.0
        )

        if self.config['frame_stack'] is not None:
            # Используем компактный набор, где явно хранятся только наблюдения.
            # Ранее использовавшийся вариант: assert 'next_observations' not in self.dataset
            if self.preprocess_frame_stack:
                stacked_observations = self.get_stacked_observations(np.arange(self.size))
                new_dict = dict(observations=stacked_observations)
                if 'next_observations' in self.dataset:
                    stacked_next_observations = self.get_stacked_observations(
                        np.arange(self.size), key='next_observations')
                    new_dict['next_observations'] = stacked_next_observations
                self.dataset = Dataset(self.dataset.copy(new_dict))

    def sample(self, batch_size, idxs=None, relabeling=True, augmentation=True):
        """Выбирает блок переходов или состояний из исходного набора данных.

        Параметры:
            batch_size: количество примеров в одном вычислительном блоке.
            idxs: индексы выбираемых элементов.
            relabeling: параметр исходного вычисления.
            augmentation: параметр исходного вычисления.
        """
        if idxs is None:
            idxs = self.dataset.get_random_idxs(batch_size)

        batch = self.dataset.sample(batch_size, idxs)
        if self.config['frame_stack'] is not None:
            batch['observations'] = self.get_observations(idxs)
            batch['next_observations'] = self.get_observations(np.minimum(idxs + 1, self.size - 1))
        if 'oracle_reps' in self.dataset:
            batch['goals'] = self.get_observations(idxs, key='oracle_reps')
            batch['next_goals'] =  self.get_observations(
                np.minimum(idxs + 1, self.size - 1), key='oracle_reps')
        else:
            batch['goals'] = self.get_observations(idxs)
            batch['next_goals'] =  self.get_observations(
                np.minimum(idxs + 1, self.size - 1))
        
        value_goal_idxs = self.sample_goals(
            idxs,
            self.config['value_p_curgoal'],
            self.config['value_p_trajgoal'],
            self.config['value_p_randomgoal'],
            self.config['value_geom_sample'],
        )
        actor_goal_idxs = self.sample_goals(
            idxs,
            self.config['actor_p_curgoal'],
            self.config['actor_p_trajgoal'],
            self.config['actor_p_randomgoal'],
            self.config['actor_geom_sample'],
        )

        batch['value_goal_observations'] = self.get_observations(value_goal_idxs)
        batch['actor_goal_observations'] = self.get_observations(actor_goal_idxs)
        if 'oracle_reps' in self.dataset:
            batch['value_goals'] = self.get_observations(value_goal_idxs, key='oracle_reps')
            batch['actor_goals'] = self.get_observations(actor_goal_idxs, key='oracle_reps')
        else:
            batch['value_goals'] = self.get_observations(value_goal_idxs)
            batch['actor_goals'] = self.get_observations(actor_goal_idxs)

        if self.config['relabeling'] and relabeling:
            successes = (idxs == value_goal_idxs).astype(float)
            batch['masks'] = 1.0 - successes
            batch['rewards'] = successes - (1.0 if self.config['gc_negative'] else 0.0)

        if self.config['p_aug'] is not None and augmentation:
            if np.random.rand() < self.config['p_aug']:
                self.augment(batch, ['observations', 'next_observations', 'value_goals', 'actor_goals'])

        return batch

    def sample_goals(
        self, idxs, p_curgoal, p_trajgoal, p_randomgoal, geom_sample
    ):
        """Выбирает подходящие целевые состояния для указанных переходов."""
        batch_size = len(idxs)

        # Выбираем случайные целевые состояния.
        random_goal_idxs = self.dataset.get_random_idxs(batch_size)

        # Выбираем будущие состояния той же траектории, кроме текущего до её завершения.
        final_state_idxs = self.terminal_locs[np.searchsorted(self.terminal_locs, idxs)]
        if geom_sample:
            # Выбираем шаг по геометрическому распределению.
            offsets = np.random.geometric(
                p=1 - self.config['discount'], size=batch_size
            ) - 1 # Допустимый диапазон: от нуля до бесконечности.
            traj_goal_idxs = np.minimum(idxs + offsets, final_state_idxs)
        else:
            # Выбираем состояния равномерно.
            distances = np.random.rand(batch_size)  # in [0, 1)
            traj_goal_idxs = np.round(
                (
                    np.minimum(idxs + 1, final_state_idxs) * distances
                    + final_state_idxs * (1 - distances)
                )
            ).astype(int)
        if p_curgoal == 1.0:
            goal_idxs = idxs
        else:
            goal_idxs = np.where(
                np.random.rand(batch_size) < p_trajgoal / (1.0 - p_curgoal),
                traj_goal_idxs,
                random_goal_idxs,
            )

            # Используем текущее состояние как цель.
            goal_idxs = np.where(
                np.random.rand(batch_size) < p_curgoal, idxs, goal_idxs
            )

        return goal_idxs

    def augment(self, batch, keys):
        """Применяет преобразования изображений к указанным полям набора данных."""
        padding = 3
        batch_size = len(batch[keys[0]])
        crop_froms = np.random.randint(0, 2 * padding + 1, (batch_size, 2))
        crop_froms = np.concatenate([crop_froms, np.zeros((batch_size, 1), dtype=np.int64)], axis=1)
        for key in keys:
            batch[key] = jax.tree_util.tree_map(
                lambda arr: np.array(batched_random_crop(arr, crop_froms, padding)) if len(arr.shape) == 4 else arr,
                batch[key],
            )

    def get_observations(self, idxs, key='observations'):
        """Возвращает наблюдения по заданным индексам."""
        if self.config['frame_stack'] is None or self.preprocess_frame_stack:
            return jax.tree_util.tree_map(lambda arr: arr[idxs], self.dataset[key])
        else:
            return self.get_stacked_observations(idxs)

    def get_stacked_observations(self, idxs, key='observations'):
        """Возвращает объединённые последовательные наблюдения."""
        initial_state_idxs = self.initial_locs[np.searchsorted(self.initial_locs, idxs, side='right') - 1]
        rets = []
        for i in reversed(range(self.config['frame_stack'])):
            cur_idxs = np.maximum(idxs - i, initial_state_idxs)
            rets.append(jax.tree_util.tree_map(lambda arr: arr[cur_idxs], self.dataset[key]))
        return jax.tree_util.tree_map(lambda *args: np.concatenate(args, axis=-1), *rets)


@dataclasses.dataclass
class HGCDataset(GCDataset):
    """Формирует переходы и цели для двух уровней иерархического агента."""
    def sample_sphere(self, batch_size, dim, dtype=np.float32, scale_sqrt_d=True):
        z = np.random.randn(batch_size, dim)
        z /= (np.linalg.norm(z, axis=-1, keepdims=True) + 1e-8)
        if scale_sqrt_d:
            z *= np.sqrt(dim)
        return z.astype(dtype)

    def sample_mask(self, batch_size, prob, dtype=bool):
        """Строит случайную логическую маску с заданной вероятностью."""
        return (np.random.rand(batch_size) < prob).astype(dtype)

    def sample(self, batch_size, idxs=None, relabeling=True, augmentation=True):
        """Выбирает блок переходов или состояний из исходного набора данных.

        Параметры:
            batch_size: количество примеров в одном вычислительном блоке.
            idxs: индексы выбираемых элементов.
            relabeling: параметр исходного вычисления.
            augmentation: параметр исходного вычисления.
        """
        if idxs is None:
            idxs = self.dataset.get_random_idxs(batch_size)

        batch = self.dataset.sample(batch_size, idxs)
        if self.config['frame_stack'] is not None:
            batch['observations'] = self.get_observations(idxs)
            batch['next_observations'] = self.get_observations(idxs + 1)


        # Выбираем цели для модели ценности.
        value_goal_idxs = self.sample_goals(idxs, self.config['value_p_curgoal'], self.config['value_p_trajgoal'],
            self.config['value_p_randomgoal'], self.config['value_geom_sample'])
        batch['value_goals'] = self.get_observations(value_goal_idxs)
                
        if self.config['relabeling'] and relabeling:
            successes = (idxs == value_goal_idxs).astype(float)
            batch['masks'] = 1.0 - successes
            batch['rewards'] = successes - (1.0 if self.config['gc_negative'] else 0.0)

        
        # Выбираем цели низкоуровневой политики.
        final_state_idxs = self.terminal_locs[np.searchsorted(self.terminal_locs, idxs)]
        if self.config['agent_name']=='hiql':
            low_goal_idxs = np.minimum(idxs + self.config['subgoal_steps'], final_state_idxs)
            mid_low_goal_idxs = np.minimum(idxs + 10, final_state_idxs)
        else:
            offsets = np.random.geometric(p=1-self.config['discount'], size=batch_size)  # Допустимый диапазон: от единицы до бесконечности.
            low_goal_idxs = np.minimum(idxs + offsets, final_state_idxs)
            mid_low_goal_idxs = np.minimum(idxs + offsets//2, final_state_idxs)

        batch['low_actor_goals'] = self.get_observations(low_goal_idxs)
        batch['mid_low_actor_goals'] = self.get_observations(mid_low_goal_idxs)

        
        # Выбираем цели высокоуровневой политики.
        if self.config['actor_geom_sample']:
            offsets = np.random.geometric(p=1-self.config['discount'], size=batch_size)  # Допустимый диапазон: от единицы до бесконечности.
            high_traj_goal_idxs = np.minimum(idxs + offsets, final_state_idxs)
        else:
            distances = np.random.rand(batch_size)  # in [0, 1)
            high_traj_goal_idxs = np.round((np.minimum(idxs + 1, final_state_idxs) * distances + final_state_idxs * (1 - distances))).astype(int)

        # Формируем случайные высокоуровневые цели.
        high_random_goal_idxs = self.dataset.get_random_idxs(batch_size)

        # Выбираем между будущими целями той же траектории и случайными целями.
        pick_random = np.random.rand(batch_size) < self.config['actor_p_randomgoal']
        high_goal_idxs = np.where(pick_random, high_random_goal_idxs, high_traj_goal_idxs)

        batch['high_actor_goals'] = self.get_observations(high_goal_idxs)
        
        if self.config['agent_name'] == 'hiql':
            high_traj_target_idxs = np.minimum(idxs + self.config['subgoal_steps'], high_traj_goal_idxs)
            high_random_target_idxs = np.minimum(idxs + self.config['subgoal_steps'], final_state_idxs)
            high_target_idxs = np.where(pick_random, high_random_target_idxs, high_traj_target_idxs)
            batch['high_actor_targets'] = self.get_observations(high_target_idxs)
        else:
            offsets = np.random.geometric(p=1-self.config['discount'], size=batch_size)  # Допустимый диапазон: от единицы до бесконечности.
            high_traj_target_idxs = np.minimum(idxs + offsets, final_state_idxs)
            high_random_target_idxs = np.minimum(idxs + offsets, final_state_idxs)
            high_target_idxs = np.where(pick_random, high_random_target_idxs, high_traj_target_idxs)
            batch['high_actor_targets'] = self.get_observations(high_target_idxs)


  
        return batch
