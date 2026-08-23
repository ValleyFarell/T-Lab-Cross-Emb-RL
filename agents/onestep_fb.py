"""Одношаговое обучение прямого и обратного представлений."""

import copy
from typing import Any

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import ml_collections
import optax

from utils.encoders import GCEncoder, encoder_modules
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import GCActor, GCValue


class OneStepFBAgent(flax.struct.PyTreeNode):
    """Обучает FB-представление на одношаговых переходах.

    Источник: https://arxiv.org/abs/2602.11399
    """

    rng: Any
    network: Any
    config: Any = nonpytree_field()

    def fb_repr_loss(self, batch, grad_params):
        """Вычисляет потери обучения прямого и обратного представлений."""
        batch_size = batch['observations'].shape[0]
        observations = batch['observations']
        actions = batch['actions']
        next_observations = batch['next_observations']
        next_actions = batch['next_actions']

        # Вычисляем целевые меры будущих посещений.
        target_next_forward_reprs = self.network.select('target_forward_repr')(
            next_observations, actions=next_actions)
        target_backward_reprs = self.network.select('target_backward_repr')(
            observations, actions=actions)
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
            observations, actions=actions, params=grad_params)
        backward_reprs = self.network.select('backward_repr')(
            observations, actions=actions, params=grad_params)
        succ_measures = jnp.einsum('eij,kj->eik', forward_reprs, backward_reprs)

        # Вычисляем потери временной разности для оценки мер посещений.
        I = jnp.eye(batch_size)
        repr_off_diag_loss = jax.vmap(
            lambda x: (x * (1 - I)) ** 2,
            0, 0
        )(succ_measures - self.config['discount'] * target_succ_measures[None])
        repr_off_diag_loss = 0.5 * jnp.sum(repr_off_diag_loss, axis=-1) / (batch_size - 1)
        repr_off_diag_loss = jnp.mean(repr_off_diag_loss)

        repr_diag_loss = -(1 - self.config['discount']) * jax.vmap(jnp.diag, 0, 0)(succ_measures)
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
            observations, actions=q_actions)
        qs = jnp.einsum('eik,ik->ei', forward_reprs, latents)
        if self.config['q_agg'] == 'mean':
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
        }

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        """Собирает общую функцию потерь всех обучаемых компонентов."""
        info = {}
        rng = rng if rng is not None else self.rng
        rng, latent_rng, actor_rng = jax.random.split(rng, 3)

        # Выбираем намерения для текущего обучающего блока.
        batch['latents'] = self.sample_latents(batch, latent_rng)

        # Обновляем одношаговые прямое и обратное представления.
        fb_repr_loss, fb_repr_info = self.fb_repr_loss(batch, grad_params)
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
        actions = batch['actions']
        rewards = batch['rewards']
        weights = jax.nn.softmax(self.config['reward_temperature'] * rewards, axis=0)
        
        backward_reprs = self.network.select('backward_repr')(
            observations, actions=actions)

        # Среднее представление состояний, взвешенное по награде.
        latent = jnp.mean((weights * rewards)[..., None] * backward_reprs, axis=0)
        if self.config['normalize_latent']:
            latent = latent / jnp.linalg.norm(
                latent, axis=-1, keepdims=True) * jnp.sqrt(self.config['latent_dim'])

        return latent

    @jax.jit
    def sample_latents(self, batch, rng):
        """Формирует намерения и связанные внутренние награды."""
        batch_size = batch['observations'].shape[0]
        observations = batch['observations']
        actions = batch['actions']

        rng, latent_rng, perm_rng, mix_rng = jax.random.split(rng, 4)
        
        latents = jax.random.normal(latent_rng, shape=(batch_size, self.config['latent_dim']))
        if self.config['normalize_latent']:
            latents = latents / jnp.linalg.norm(
                latents, axis=-1, keepdims=True) * jnp.sqrt(self.config['latent_dim'])
        
        perm = jax.random.permutation(perm_rng, jnp.arange(batch_size))
        backward_reprs = self.network.select('backward_repr')(
            observations, actions=actions)
        latent_backward_reprs = backward_reprs[perm]
        if self.config['normalize_latent']:
            latent_backward_reprs = latent_backward_reprs / jnp.linalg.norm(
                latent_backward_reprs, axis=-1, keepdims=True) * jnp.sqrt(self.config['latent_dim'])
        
        latents = jnp.where(
            jax.random.uniform(mix_rng, (batch_size, 1)) < self.config['latent_mix_prob'],
            latents,
            latent_backward_reprs,
        )

        return latents

    @jax.jit
    def sample_actions(
        self,
        observations,
        latents=None,
        seed=None,
        temperature=1.0,
    ):
        """Выбирает действие исходной политики по состоянию и намерению."""
        dist = self.network.select('actor')(observations, latents,
                                            goal_encoded=True, temperature=temperature)
        actions = dist.sample(seed=seed)
        actions = jnp.clip(actions, -1, 1)
        return actions

    @classmethod
    def create(
        cls,
        seed,
        ex_batch,
        config,
    ):
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
        ex_latents = jnp.ones((*ex_actions.shape[:-1], config['latent_dim']))

        action_dim = ex_actions.shape[-1]

        # Создаём энкодеры исходных наблюдений.
        encoders = dict()
        if config['encoder'] is not None:
            encoder_module = encoder_modules[config['encoder']]
            encoders['forward_repr'] = GCEncoder(state_encoder=encoder_module())
            encoders['backward_repr'] = GCEncoder(state_encoder=encoder_module())
            encoders['actor'] = GCEncoder(state_encoder=encoder_module())

        # Создаём вычислительные сети агента.
        forward_repr_def = GCValue(
            hidden_dims=config['forward_repr_hidden_dims'],
            value_dim=config['latent_dim'],
            activations=getattr(nn, config['activation']),
            layer_norm=config['fackward_repr_layer_norm'],
            num_ensembles=2,
            gc_encoder=encoders.get('forward_repr'),
        )
        backward_repr_def = GCValue(
            hidden_dims=config['backward_repr_hidden_dims'],
            value_dim=config['latent_dim'],
            activations=getattr(nn, config['activation']),
            layer_norm=config['backward_repr_layer_norm'],
            num_ensembles=1,
            gc_encoder=encoders.get('backward_repr'),
        )
        actor_def = GCActor(
            hidden_dims=config['actor_hidden_dims'],
            action_dim=action_dim,
            activations=getattr(nn, config['activation']),
            state_dependent_std=False,
            tanh_squash=config['tanh_squash'],
            layer_norm=config['actor_layer_norm'],
            const_std=config['const_std'],
            final_fc_init_scale=config['actor_fc_scale'],
            gc_encoder=encoders.get('actor'),
        )

        network_info = dict(
            forward_repr=(forward_repr_def, (ex_observations, None, ex_actions)),
            backward_repr=(backward_repr_def, (ex_observations, None, ex_actions)),
            target_forward_repr=(copy.deepcopy(forward_repr_def), (ex_observations, None, ex_actions)),
            target_backward_repr=(copy.deepcopy(backward_repr_def), (ex_observations, None, ex_actions)),
            actor=(actor_def, (ex_observations, ex_latents, True))
        )
        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}

        network_def = ModuleDict(networks)
        network_tx = optax.adam(learning_rate=config['lr'])
        network_params = network_def.init(init_rng, **network_args)['params']
        network = TrainState.create(network_def, network_params, tx=network_tx)

        params = network.params
        params['modules_target_forward_repr'] = params['modules_forward_repr']
        params['modules_target_backward_repr'] = params['modules_backward_repr']
        
        return cls(rng, network=network, config=flax.core.FrozenDict(**config))


def get_config():
    config = ml_collections.ConfigDict(
        dict(
            agent_name='onestep_fb',  # Название реализации агента.
            lr=1e-4,  # Шаг обновления обучаемых параметров.
            batch_size=1024,  # Количество примеров в одном обучающем блоке.
            actor_hidden_dims=(512, 512, 512, 512),  # Размеры скрытых слоёв сети политики.
            forward_repr_hidden_dims=(512, 512, 512, 512),  # Размеры скрытых слоёв прямого представления F.
            backward_repr_hidden_dims=(512, 512, 512, 512),  # Размеры скрытых слоёв обратного представления B.
            actor_layer_norm=False,  # Использовать ли нормализацию слоёв политики.
            fackward_repr_layer_norm=True,  # Использовать ли нормализацию слоёв прямого представления.
            backward_repr_layer_norm=True,  # Использовать ли нормализацию слоёв обратного представления.
            activation='gelu',  # Функция активации нейронной сети.
            latent_dim=128,  # Размерность представления намерения.
            discount=0.99,  # Коэффициент дисконтирования будущих состояний.
            tau=0.005,  # Скорость плавного обновления целевой сети.
            normalize_latent=True,  # Нормализовать ли обратные представления состояний.
            reward_temperature=0.0,  # Температура весов, определяемых наградой.
            repr_agg='mean',  # Способ объединения оценок целевого FB-ансамбля.
            q_agg='min',  # Способ объединения скалярных оценок прямого представления.
            orthonorm_coeff=0.0,  # Коэффициент регуляризации ортонормальности.
            latent_mix_prob=0.5,  # Вероятность заменить случайное намерение реальным обратным представлением.
            alpha=0.3,  # Вес воспроизведения офлайн-действий при обучении градиентом политики.
            tanh_squash=True,  # Ограничивать ли действия политики функцией tanh.
            actor_fc_scale=0.01,  # Масштаб инициализации последнего слоя политики.
            const_std=True,  # Использовать ли постоянный разброс действий политики.
            normalize_q_loss=True,  # Нормировать ли потери оценки Q.
            num_zero_shot_samples=100_000,  # Число состояний для построения представления новой задачи.
            encoder=ml_collections.config_dict.placeholder(str),  # Имя энкодера либо отсутствие отдельного энкодера.
            
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
