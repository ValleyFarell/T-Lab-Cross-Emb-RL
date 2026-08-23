"""Агент с функцией ценности, зависящей от намерения."""

import copy
from typing import Any

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import ml_collections
import optax

from utils.encoders import GCEncoder, GCIntentionEncoder, encoder_modules
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import GCActor, GCValue, ICVFValue


class ICVFAgent(flax.struct.PyTreeNode):
    """Обучает функцию ценности, обусловленную намерением.

    Источник: https://arxiv.org/abs/2304.04782
    """

    rng: Any
    network: Any
    config: Any = nonpytree_field()

    @staticmethod
    def expectile_loss(adv, diff, expectile):
        """Вычисляет асимметричную функцию потерь оценки ценности."""
        weight = jnp.where(adv >= 0, expectile, (1 - expectile))
        return weight * (diff ** 2)

    def repr_loss(self, batch, grad_params):
        """Вычисляет функцию потерь обучаемого представления состояния."""
        observations = batch['observations']
        next_observations = batch['next_observations']
        rewards = batch['rewards']
        masks = batch['masks']
        value_goals = batch['value_goals']
        intention_goals = batch['intention_goals']
        # Ранее использовавшийся вариант: intention_rewards = batch['intention_rewards']
        # Ранее использовавшийся вариант: intention_masks = batch['intention_masks']
        intention_rewards = ((observations == intention_goals).all(axis=1)).astype(float)
        intention_masks = 1.0 - intention_rewards
        
        # Вычисляем преимущества выбранных действий.
        intention_vs, obs_phis, intention_psis, intention_transitions = self.network.select('repr')(
            observations, 
            goals=intention_goals, 
            intentions=intention_goals, 
            info=True,
            params=grad_params, 
        )
        next_intention_vs = self.network.select('repr')(
            next_observations, 
            goals=None, 
            intentions=None, 
            psis=intention_psis,
            transitions=intention_transitions,
            params=grad_params, 
        )
        intention_v = jnp.mean(intention_vs, axis=0)
        next_intention_v = jnp.min(next_intention_vs, axis=0)
        intention_q = intention_rewards + self.config['discount'] * intention_masks * next_intention_v
        adv = intention_q - intention_v
        
        # Вычисляем ошибки уравнения Беллмана.
        target_next_vs = self.network.select('target_repr')(
            next_observations, 
            goals=value_goals, 
            intentions=intention_goals, 
        )
        qs = rewards + self.config['discount'] * masks * target_next_vs
        
        vs = self.network.select('repr')(
            observations, 
            goals=value_goals, 
            intentions=None,
            phis=obs_phis, 
            transitions=intention_transitions,
            params=grad_params, 
        )

        # Вычисляем асимметричные потери оценки ценности.
        value_loss = self.expectile_loss(adv[None], qs - vs, self.config['expectile']).mean()
        
        # Собираем дополнительные показатели для журнала.
        v = jnp.mean(vs, axis=0)

        return value_loss, {
            'value_loss': value_loss,
            'v_mean': v.mean(),
            'v_max': v.max(),
            'v_min': v.min(),
        }

    def critic_loss(self, batch, grad_params, rng):
        """Вычисляет функцию потерь оценщика ценности."""
        observations = batch['observations']
        actions = batch['actions']
        next_observations = batch['next_observations']
        rewards = batch['latent_rewards']
        latents = batch['latents']
        
        # Выбираем действия для следующих состояний.
        rng, sample_rng = jax.random.split(rng)
        next_dist = self.network.select('actor')(next_observations, latents, goal_encoded=True)
        next_actions = next_dist.mode()
        noise = jnp.clip(
            (jax.random.normal(sample_rng, next_actions.shape) * self.config['actor_noise']),
            -self.config['actor_noise_clip'],
            self.config['actor_noise_clip'],
        )
        next_actions = jnp.clip(next_actions + noise, -1, 1)

        # Вычисляем целевую оценку Q.
        next_qs = self.network.select('target_critic')(
            next_observations, latents, actions=next_actions, goal_encoded=True)
        if self.config['q_agg'] == 'mean':
            next_q = jnp.mean(next_qs, axis=0)
        else:
            next_q = jnp.min(next_qs, axis=0)
        target_q = rewards + self.config['discount'] * next_q

        # Вычисляем потери временной разности.
        qs = self.network.select('critic')(
            observations, latents, actions=actions, goal_encoded=True, params=grad_params)
        critic_loss = jnp.mean((qs - target_q) ** 2)

        return critic_loss, {
            'critic_loss': critic_loss,
            'q_mean': qs.mean(),
            'q_max': qs.max(),
            'q_min': qs.min(),
        }
    
    def actor_loss(self, batch, grad_params):
        """Вычисляет функцию потерь обучаемой политики действий."""
        observations = batch['observations']
        actions = batch['actions']
        latents = batch['latents']
        
        # Выбираем действия текущей политики.
        dist = self.network.select('actor')(
            observations, latents, goal_encoded=True, params=grad_params)
        q_actions = jnp.clip(dist.mode(), -1, 1)
        
        # Вычисляем потери модели Q.
        qs = self.network.select('critic')(
            observations, latents, actions=q_actions, goal_encoded=True)
        if self.config['q_agg'] == 'mean':
            q = jnp.mean(qs, axis=0)
        else:
            q = jnp.min(qs, axis=0)
        
        # Вычисляем потери воспроизведения действий офлайн-набора.
        bc_loss = jnp.mean((q_actions - actions) ** 2)
        
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
            'std': action_std.mean(),
        }

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        """Собирает общую функцию потерь всех обучаемых компонентов."""
        info = {}
        rng = rng if rng is not None else self.rng

        rng, latent_rng, critic_rng = jax.random.split(rng, 3)

        # Выбираем намерения и соответствующие внутренние награды.
        batch['latents'], batch['latent_rewards'] = self.sample_latents(batch, latent_rng)

        # Обновляем представления модели ценности намерений.
        repr_loss, repr_info = self.repr_loss(batch, grad_params)
        for k, v in repr_info.items():
            info[f'repr/{k}'] = v

        # Обучаем критик по внутренним наградам модели намерений.
        critic_loss, critic_info = self.critic_loss(batch, grad_params, critic_rng)
        for k, v in critic_info.items():
            info[f'critic/{k}'] = v

        # Обучаем политику увеличивать оценку критика.
        actor_loss, actor_info = self.actor_loss(batch, grad_params)
        for k, v in actor_info.items():
            info[f'actor/{k}'] = v

        loss = repr_loss + critic_loss + actor_loss
        
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
        self.target_update(new_network, 'repr')
        self.target_update(new_network, 'critic')
        self.target_update(new_network, 'actor')

        return self.replace(network=new_network, rng=new_rng), info

    @jax.jit
    def infer_latent(self, batch):
        """Строит представление конечной задачи по наградам офлайн-состояний."""
        phis = self.network.select('repr')(batch['observations'], goals=None, intentions=None, info=True)[1]
        phis = phis[0]
        latent = jnp.linalg.lstsq(phis, batch['rewards'])[0]
        if self.config['normalize_latent']:
            latent = latent / jnp.linalg.norm(
                latent, axis=-1, keepdims=True) * jnp.sqrt(self.config['latent_dim'])

        return latent

    @jax.jit
    def sample_latents(self, batch, rng):
        """Формирует намерения и связанные внутренние награды."""
        batch_size = batch['observations'].shape[0]
        latents = jax.random.normal(rng, shape=(batch_size, self.config['latent_dim']),
                                    dtype=batch['actions'].dtype)
        if self.config['normalize_latent']:
            latents = latents / jnp.linalg.norm(
                latents, axis=-1, keepdims=True) * jnp.sqrt(self.config['latent_dim'])

        # Внутренние награды определяются представлением будущих посещений.
        phis = self.network.select('repr')(
            batch['observations'], goals=None, intentions=None, info=True)[1]
        phis = phis[0]
        rewards = (phis * latents).sum(axis=-1)

        return latents, rewards

    @jax.jit
    def sample_actions(
        self,
        observations,
        latents=None,
        seed=None,
        temperature=1.0,
    ):
        """Выбирает действие исходной политики по состоянию и намерению."""
        dist = self.network.select('actor')(
            observations, latents, goal_encoded=True, temperature=temperature)
        actions = dist.mode()
        noise = jnp.clip(
            (jax.random.normal(seed, actions.shape) * self.config['actor_noise'] * temperature),
            -self.config['actor_noise_clip'],
            self.config['actor_noise_clip'],
        )
        actions = jnp.clip(actions + noise, -1, 1)

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
        ex_goals = ex_batch['value_goals']
        ex_actions = ex_batch['actions']
        ex_latents = jnp.ones((ex_actions.shape[0], config['latent_dim']), dtype=ex_actions.dtype)

        action_dim = ex_actions.shape[-1]

        # Создаём энкодеры исходных наблюдений.
        encoders = dict()
        if config['encoder'] is not None:
            encoder_module = encoder_modules[config['encoder']]
            encoders['repr'] = GCIntentionEncoder(concat_encoder=encoder_module())
            encoders['critic'] = GCEncoder(state_encoder=encoder_module())
            encoders['actor'] = GCEncoder(state_encoder=encoder_module())

        # Создаём сети представления, ценности и политики.
        repr_def = ICVFValue(
            hidden_dims=config['repr_hidden_dims'],
            activations=getattr(nn, config['activation']),
            layer_norm=config['repr_layer_norm'],
            value_dim=config['latent_dim'],
            num_ensembles=2,
            icvf_encoder=encoders.get('repr'),
        )
        critic_def = GCValue(
            hidden_dims=config['value_hidden_dims'],
            activations=getattr(nn, config['activation']),
            layer_norm=config['value_layer_norm'],
            num_ensembles=2,
            gc_encoder=encoders.get('critic'),
        )
        actor_def = GCActor(
            hidden_dims=config['actor_hidden_dims'],
            action_dim=action_dim,
            activations=getattr(nn, config['activation']),
            state_dependent_std=False,
            tanh_squash=config['tanh_squash'],
            layer_norm=config['actor_layer_norm'],
            const_std=True,
            final_fc_init_scale=config['actor_fc_scale'],
            gc_encoder=encoders.get('actor'),
        )

        network_info = dict(
            repr=(repr_def, (ex_observations, ex_goals, ex_goals)),
            critic=(critic_def, (ex_observations, ex_latents, ex_actions, None, True)),
            actor=(actor_def, (ex_observations, ex_latents, True)),
            target_repr=(copy.deepcopy(repr_def), (ex_observations, ex_goals, ex_goals)),
            target_critic=(copy.deepcopy(critic_def), (ex_observations, ex_latents, ex_actions, None, True)),
            target_actor=(copy.deepcopy(actor_def), (ex_observations, ex_latents, True)),
        )

        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}

        network_def = ModuleDict(networks)
        network_tx = optax.adam(learning_rate=config['lr'])
        network_params = network_def.init(init_rng, **network_args)['params']
        network = TrainState.create(network_def, network_params, tx=network_tx)

        params = network.params
        params['modules_target_repr'] = params['modules_repr']
        params['modules_target_critic'] = params['modules_critic']
        params['modules_target_actor'] = params['modules_actor']

        return cls(rng, network=network, config=flax.core.FrozenDict(**config))


def get_config():
    config = ml_collections.ConfigDict(
        dict(
            # Настройки обучения и архитектуры агента.
            agent_name='icvf',  # Название реализации агента.
            lr=1e-4,  # Шаг обновления обучаемых параметров.
            batch_size=1024,  # Количество примеров в одном обучающем блоке.
            repr_hidden_dims=(512, 512, 512, 512),  # Размеры скрытых слоёв сети представления.
            value_hidden_dims=(512, 512, 512, 512),  # Размеры скрытых слоёв модели ценности.
            actor_hidden_dims=(512, 512, 512, 512),  # Размеры скрытых слоёв сети политики.
            repr_layer_norm=False,  # Использовать ли нормализацию слоёв представления.
            value_layer_norm=False,  # Использовать ли нормализацию слоёв критика.
            actor_layer_norm=False,  # Использовать ли нормализацию слоёв политики.
            activation='gelu',  # Функция активации нейронной сети.
            latent_dim=128,  # Размерность скрытого представления намерения.
            discount=0.99,  # Коэффициент дисконтирования будущих состояний.
            tau=0.005,  # Скорость плавного обновления целевой сети.
            expectile=0.5,  # Коэффициент асимметрии оценки ценности.
            normalize_latent=True,  # Нормализовать ли обратные представления состояний.
            q_agg='mean',  # Способ объединения оценок Q.
            alpha=0.3,  # Вес воспроизведения офлайн-действий при обучении политики.
            tanh_squash=True,  # Ограничивать ли действия политики функцией tanh.
            actor_fc_scale=0.01,  # Масштаб инициализации последнего слоя политики.
            actor_noise=0.2,  # Амплитуда случайного шума действий политики.
            actor_noise_clip=0.2,  # Предельная амплитуда шума действий.
            normalize_q_loss=True,  # Нормировать ли потери оценки Q.
            num_zero_shot_samples=100_000,  # Число состояний для вычисления представления конкретной задачи.
            encoder=ml_collections.config_dict.placeholder(str),  # Имя энкодера либо отсутствие отдельного энкодера.
            
            # Настройки подготовки офлайн-набора данных.
            dataset_class='GCDataset',  # Имя класса набора данных: GCDataset, Dataset или другой вариант.
            relabeling=True,  # Пересчитывать ли награды для выбранной цели.
            value_p_curgoal=0.0,  # Вероятность выбрать текущее состояние целью оценки ценности.
            value_p_trajgoal=0.625,  # Вероятность выбрать будущую точку той же траектории целью ценности.
            value_p_randomgoal=0.375,  # Вероятность выбрать случайное состояние целью ценности.
            value_geom_sample=True,  # Использовать ли геометрическое распределение для будущих целей ценности.
            actor_p_curgoal=0.0,  # Не используется; сохранено для совместимости с GCDataset.
            actor_p_trajgoal=1.0,  # Не используется; сохранено для совместимости с GCDataset.
            actor_p_randomgoal=0.0,  # Не используется; сохранено для совместимости с GCDataset.
            actor_geom_sample=False,  # Не используется; сохранено для совместимости с GCDataset.
            gc_negative=False,  # Выбор схемы награды: 0 при успехе и -1 иначе либо 1 при успехе и 0 иначе.
            p_aug=0.0,  # Вероятность случайного преобразования изображения.
            frame_stack=ml_collections.config_dict.placeholder(int),  # Количество объединяемых последовательных кадров.

        )
    )
    return config
