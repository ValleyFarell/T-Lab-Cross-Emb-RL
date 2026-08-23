"""Исходный агент обучения прямого и обратного представлений."""

import copy
from typing import Any

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import ml_collections
import optax

from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import GCActor, GCValue


class FBAgent(flax.struct.PyTreeNode):
    """Обучает прямое и обратное представления будущих состояний.

    Источник: https://arxiv.org/abs/2103.07945

    Источник: https://github.com/enjeeneer/zero-shot-rl/blob/main/agents/fb/agent.py
    """

    rng: Any
    network: Any
    config: Any = nonpytree_field()
    
    def normalize_z(self, z):
        return z / (jnp.linalg.norm(z, axis=-1, keepdims=True) + 1e-8) * jnp.sqrt(self.config['latent_dim'])

    def fb_repr_loss(self, batch, grad_params, rng):
        """Вычисляет потери обучения прямого и обратного представлений."""
        batch_size = batch['observations'].shape[0]
        observations = batch['observations']
        goals = batch['goals']
        actions = batch['actions']
        next_observations = batch['next_observations']
        latents = batch['latents']

        # Выбираем действия для следующих состояний.
        next_dist = self.network.select('actor')(
            next_observations, latents, goal_encoded=True)
        if self.config['const_std']:
            next_actions = jnp.clip(next_dist.mode(), -1, 1)
        else:
            next_actions = jnp.clip(next_dist.sample(seed=rng), -1, 1)

        # Вычисляем целевые меры будущих посещений.
        target_next_forward_reprs = self.network.select('target_forward_repr')(
            next_observations, latents, actions=next_actions, goal_encoded=True)
        target_backward_reprs = self.network.select('target_backward_repr')(goals)
        target_succ_measures = jnp.einsum(
            'eij,kj->eik',
            target_next_forward_reprs,
            target_backward_reprs,
        )
        if self.config['repr_agg'] == 'mean':
            target_succ_measures = jnp.mean(target_succ_measures, axis=0)
        else:
            target_succ_measures = jnp.min(target_succ_measures, axis=0)

        # Вычисляем текущие меры будущих посещений.
        forward_reprs = self.network.select('forward_repr')(
            observations, latents, actions=actions, goal_encoded=True, params=grad_params)
        backward_reprs = self.network.select('backward_repr')(
            goals, params=grad_params)
        succ_measures = jnp.einsum('eij,kj->eik', forward_reprs, backward_reprs)
        
        # Вычисляем потери временной разности для оценки мер посещений.
        I = jnp.eye(batch_size)
        repr_off_diag_loss = jax.vmap(
            lambda x: (x * (1 - I)) ** 2,
            0, 0
        )(succ_measures - self.config['discount'] * target_succ_measures[None])
        repr_off_diag_loss = 0.5 * jnp.sum(repr_off_diag_loss, axis=-1) / (batch_size - 1)
        repr_off_diag_loss = jnp.mean(repr_off_diag_loss)

        # Ранее использовавшийся вариант: repr_diag_loss = -(1 - self.config['discount']) * jax.vmap(jnp.diag, 0, 0)(succ_measures)
        repr_diag_loss = - jax.vmap(jnp.diag, 0, 0)(succ_measures)
        repr_diag_loss = jnp.mean(repr_diag_loss)

        repr_loss = repr_diag_loss + repr_off_diag_loss

        # Добавляем регуляризацию ортонормальности представлений.
        covariance = jnp.matmul(backward_reprs, backward_reprs.T)
        ortho_diag_loss = -jnp.diag(covariance).mean()
        ortho_off_diag_loss = 0.5 * jnp.sum((covariance * (1 - I)) ** 2, axis=-1) / (batch_size - 1)
        ortho_off_diag_loss = jnp.mean(ortho_off_diag_loss)
        ortho_loss = ortho_diag_loss + ortho_off_diag_loss

        fb_loss = repr_loss + self.config['orthonorm_coeff'] * ortho_loss 

        return fb_loss, {
            'fb_loss': fb_loss,
            'repr_loss': repr_loss,
            'repr_diag_loss': repr_diag_loss,
            'repr_off_diag_loss': repr_off_diag_loss,
            'ortho_loss': ortho_loss,
            'ortho_diag_loss': ortho_diag_loss,
            'ortho_off_diag_loss': ortho_off_diag_loss,
            'succ_measure_mean': succ_measures.mean(),
            'succ_measure_max': succ_measures.max(),
            'succ_measure_min': succ_measures.min(),
            'forward_norm': jnp.linalg.norm(forward_reprs, axis=-1).mean(),
            'backward_norm': jnp.linalg.norm(backward_reprs, axis=-1).mean(),
            'latent_norm': jnp.linalg.norm(latents, axis=-1).mean(),
            'succ_diag_mean': jnp.diag(succ_measures[0]).mean(),
        }

    def actor_loss(self, batch, grad_params, rng=None):
        """Вычисляет функцию потерь обучаемой политики действий."""
        observations = batch['observations']
        actions = batch['actions']
        latents = batch['latents']

        # Выбираем действия текущей политики.
        dist = self.network.select('actor')(
            observations, latents, goal_encoded=True, params=grad_params)
        if self.config['const_std']:
            q_actions = jnp.clip(dist.mode(), -1, 1)
        else:
            q_actions = jnp.clip(dist.sample(seed=rng), -1, 1)
        forward_reprs = self.network.select('forward_repr')(
            observations, latents, actions=q_actions, goal_encoded=True)
        qs = jnp.einsum('eik,ik->ei', forward_reprs, latents)
        if self.config['repr_agg'] == 'mean':
            q = jnp.mean(qs, axis=0)
        else:
            q = jnp.min(qs, axis=0)

        # Вычисляем потери воспроизведения действий офлайн-набора.
        log_prob = dist.log_prob(actions)
        bc_loss = -log_prob.mean()

        # Нормируем оценки Q, чтобы потери не зависели от общего масштаба.
        q_loss = -q.mean()
        if self.config['normalize_q_loss']:
            lam = jax.lax.stop_gradient(1 / jnp.abs(q).mean())
            q_loss = lam * q_loss

        actor_loss = q_loss + self.config['alpha'] * bc_loss

        if self.config['tanh_squash']:
            action_std = dist._distribution.stddev()
        else:
            action_std = dist.stddev().mean()

        return actor_loss, {
            'actor_loss': actor_loss,
            'q_loss': q_loss,
            'bc_loss': bc_loss,
            'q_mean': q.mean(),
            'q_abs_mean': jnp.abs(q).mean(),
            'bc_log_prob': log_prob.mean(),
            'mse': jnp.mean((dist.mode() - actions) ** 2),
            'std': action_std,
            'q_std': qs.std(),
        }

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        """Собирает общую функцию потерь всех обучаемых компонентов."""
        info = {}
        rng = rng if rng is not None else self.rng
        rng, latent_rng, fb_repr_rng, actor_rng = jax.random.split(rng, 4)

        # Выбираем намерения для текущего обучающего блока.
        batch['latents'] = self.sample_latents(batch, latent_rng)

        # Обновляем прямое и обратное представления.
        fb_repr_loss, fb_repr_info = self.fb_repr_loss(batch, grad_params, fb_repr_rng)
        for k, v in fb_repr_info.items():
            info[f'fb_repr/{k}'] = v

        # Обучаем политику увеличивать скалярную оценку ценности.
        actor_loss, actor_info = self.actor_loss(batch, grad_params, actor_rng)
        for k, v in actor_info.items():
            info[f'actor/{k}'] = v

        loss = fb_repr_loss + actor_loss

        return loss, info

    def target_update(self, network, module_name):
        """Плавно обновляет параметры целевой нейронной сети."""
        new_target_params = jax.tree_util.tree_map(
            lambda p, tp: p * self.config['tau'] + tp * (1 - self.config['tau']),
            self.network.params[f'modules_{module_name}'],
            self.network.params[f'modules_target_{module_name}'],
        )
        network.params[f'modules_target_{module_name}'] = new_target_params

    @jax.jit
    def update(self, batch):
        """Обновляет агента и возвращает новое состояние вместе с диагностикой."""
        new_rng, rng = jax.random.split(self.rng)

        def loss_fn(grad_params):
            return self.total_loss(batch, grad_params, rng=rng)

        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)
        self.target_update(new_network, 'forward_repr')
        self.target_update(new_network, 'backward_repr')

        return self.replace(network=new_network, rng=new_rng), info

    @jax.jit
    def infer_latent(self, batch):
        """Строит представление конечной задачи по наградам офлайн-состояний."""
        observations = batch['observations']
        rewards = batch['rewards']
        weights = jax.nn.softmax(self.config['reward_temperature'] * rewards, axis=0)
        
        backward_reprs = self.network.select('backward_repr')(observations)
        
        # Среднее представление состояний, взвешенное по награде.
        latent = jnp.mean((weights * rewards)[..., None] * backward_reprs, axis=0)
        if self.config['normalize_latent']:
            latent = self.normalize_z(latent)

        return latent

    @jax.jit
    def sample_latents(self, batch, rng):
        """Формирует намерения и связанные внутренние награды."""
        batch_size = batch['observations'].shape[0]
        observations = batch['observations']
        
        rng, latent_rng, perm_rng, mix_rng = jax.random.split(rng, 4)

        latents = jax.random.normal(latent_rng, shape=(batch_size, self.config['latent_dim']))
        if self.config['normalize_latent']:
            latents = self.normalize_z(latents)
        
        perm = jax.random.permutation(perm_rng, jnp.arange(batch_size))
        backward_reprs = self.network.select('backward_repr')(observations)
        latent_backward_reprs = backward_reprs[perm]
        if self.config['normalize_latent']:
            latent_backward_reprs = self.normalize_z(latent_backward_reprs)
        
        latents = jnp.where(
            jax.random.uniform(mix_rng, (batch_size, 1)) < self.config['latent_mix_prob'],
            latents,
            latent_backward_reprs,
        )

        return latents

    @jax.jit
    def sample_actions(self, observations, latents=None, seed=None, temperature=1.0):
        """Выбирает действие исходной политики по состоянию и намерению."""
        dist = self.network.select('actor')(observations, latents, goal_encoded=True, temperature=temperature)
        actions = dist.sample(seed=seed)
        return jnp.clip(actions, -1, 1)


    @classmethod
    def create(cls, seed, ex_batch, config):
        """Создаёт экземпляр агента или структуры данных с заданной конфигурацией.

        Параметры:
            seed: начальное значение генератора случайных чисел.
            ex_batch: пример блока переходов.
            config: конфигурация агента или вспомогательной модели.
        """
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng, 2)

        ex_observations = ex_batch['observations']
        ex_actions = ex_batch['actions']
        action_dim = ex_actions.shape[-1]
        ex_latents = jnp.ones((*ex_actions.shape[:-1], config['latent_dim']))

        # Создаём вычислительные сети агента.
        forward_repr_def = GCValue(
            hidden_dims=config['forward_repr_hidden_dims'],
            value_dim=config['latent_dim'],
            activations=getattr(nn, config['activation']),
            layer_norm=config['forward_repr_layer_norm'],
            num_ensembles=2,
        )
        
        backward_repr_def = GCValue(
            hidden_dims=config['backward_repr_hidden_dims'],
            value_dim=config['latent_dim'],
            activations=getattr(nn, config['activation']),
            layer_norm=config['backward_repr_layer_norm'],
            num_ensembles=1,
        )
        actor_def = GCActor(
            hidden_dims=config['actor_hidden_dims'],
            action_dim=action_dim,
            state_dependent_std=False,
            tanh_squash=config['tanh_squash'],
            activations=getattr(nn, config['activation']),
            layer_norm=config['actor_layer_norm'],
            const_std=config['const_std'],
        )

        network_info = dict(
            forward_repr=(forward_repr_def, (ex_observations, ex_latents, ex_actions, None, True)),
            backward_repr=(backward_repr_def, (ex_observations,)),
            target_forward_repr=(copy.deepcopy(forward_repr_def), (ex_observations, ex_latents, ex_actions, None, True)),
            target_backward_repr=(copy.deepcopy(backward_repr_def), (ex_observations, None, ex_actions)),
            actor=(actor_def, (ex_observations, ex_latents, True)),
        )
        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}

        network_def = ModuleDict(networks)
        network_tx = optax.adam(learning_rate=config['lr'])
        
        network_params = network_def.init(init_rng, **network_args)['params']
        network_params['modules_target_forward_repr'] = copy.deepcopy(network_params['modules_forward_repr'])
        network_params['modules_target_backward_repr'] = copy.deepcopy(network_params['modules_backward_repr'])
        network = TrainState.create(network_def, network_params, tx=network_tx)

        return cls(rng, network=network, config=flax.core.FrozenDict(**config))


def get_config():
    config = ml_collections.ConfigDict(
        dict(
            agent_name='fb',  # Название реализации агента.
            lr=1e-4,  # Шаг обновления обучаемых параметров.
            batch_size=1024,  # Количество примеров в одном обучающем блоке.
            actor_hidden_dims=(512, 512, 512, 512),  # Размеры скрытых слоёв сети политики.
            forward_repr_hidden_dims=(512, 512, 512, 512),  # Размеры скрытых слоёв прямого представления F.
            backward_repr_hidden_dims=(512, 512, 512, 512),  # Размеры скрытых слоёв обратного представления B.
            actor_layer_norm=False,  # Использовать ли нормализацию слоёв политики.
            forward_repr_layer_norm=False,  # Использовать ли нормализацию слоёв прямого представления.
            backward_repr_layer_norm=False,  # Использовать ли нормализацию слоёв обратного представления.
            activation='gelu',  # Функция активации нейронной сети.
            latent_dim=128,  # Размерность намерения: 128 для муравья и 32 для точечного робота.
            discount=0.99,  # Коэффициент дисконтирования будущих состояний.
            tau=0.005,  # Скорость плавного обновления целевой сети.
            normalize_latent=True,  # Нормализовать ли обратные представления состояний.
            reward_temperature=0.0,  # Температура весов, определяемых наградой.
            repr_agg='mean',  # Способ объединения оценок целевого FB-ансамбля.
            orthonorm_coeff=1.0,  # Коэффициент регуляризации ортонормальности.
            latent_mix_prob=0.5,  # Вероятность заменить случайное намерение реальным обратным представлением.
            alpha=0.03,  # Вес воспроизведения офлайн-действий при обучении политики.
            tanh_squash=True,  # Ограничивать ли действия политики функцией tanh.
            const_std=True,  # Использовать ли постоянный разброс действий политики.
            normalize_q_loss=True,  # Нормировать ли потери оценки Q.
            num_zero_shot_samples=100_000,  # Число состояний для построения представления новой задачи.
            
            # Настройки подготовки офлайн-набора данных.
            dataset_class='GCDataset',  # Имя класса набора данных: GCDataset, Dataset или другой вариант.
            relabeling=False,  # Пересчитывать ли награды для выбранной цели.
            value_p_curgoal=0.2,  # Не используется; сохранено для совместимости с GCDataset.
            value_p_trajgoal=0.5,  # Не используется; сохранено для совместимости с GCDataset.
            value_p_randomgoal=0.3,  # Не используется; сохранено для совместимости с GCDataset.
            value_geom_sample=True,  # Не используется; сохранено для совместимости с GCDataset.
            actor_p_curgoal=0.0,  # Не используется; сохранено для совместимости с GCDataset.
            actor_p_trajgoal=1.0,  # Не используется; сохранено для совместимости с GCDataset.
            actor_p_randomgoal=0.0,  # Не используется; сохранено для совместимости с GCDataset.
            actor_geom_sample=False,  # Не используется; сохранено для совместимости с GCDataset.
            gc_negative=False,  # Не используется; сохранено для совместимости с GCDataset.
            p_aug=0.0,  # Вероятность случайного преобразования изображения.
            frame_stack=ml_collections.config_dict.placeholder(int),  # Количество объединяемых последовательных кадров.
        )
    )
    return config
