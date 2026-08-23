"""Иерархический агент с обучением ценности и двух уровней политики."""

from typing import Any
import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import ml_collections
import optax
from utils.encoders import GCEncoder
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import MLP, GCActor, GCValue, Identity, LengthNormalize


class HIQLAgent(flax.struct.PyTreeNode):
    """Обучает иерархическую политику и функцию ценности."""
    rng: Any
    network: Any
    config: Any = nonpytree_field()

    @staticmethod
    def expectile_loss(adv, diff, expectile):
        """Вычисляет асимметричную функцию потерь оценки ценности."""
        weight = jnp.where(adv >= 0, expectile, (1 - expectile))
        return weight * (diff**2)

    def value_loss(self, batch, grad_params):
        """Вычисляет функцию потерь модели ценности."""
        (next_v1_t, next_v2_t) = self.network.select('target_value')(batch['next_observations'], batch['value_goals'])
        next_v_t = jnp.minimum(next_v1_t, next_v2_t)
        q = batch['rewards'] + self.config['discount'] * next_v_t

        (v1_t, v2_t) = self.network.select('target_value')(batch['observations'], batch['value_goals'])
        v_t = (v1_t + v2_t) / 2
        adv = q - v_t

        q1 = batch['rewards'] + self.config['discount'] * next_v1_t
        q2 = batch['rewards'] + self.config['discount'] * next_v2_t
        (v1, v2) = self.network.select('value')(batch['observations'], batch['value_goals'], params=grad_params)
        v = (v1 + v2) / 2

        value_loss1 = self.expectile_loss(adv, q1 - v1, self.config['expectile']).mean()
        value_loss2 = self.expectile_loss(adv, q2 - v2, self.config['expectile']).mean()
        value_loss = value_loss1 + value_loss2

        return value_loss, {
            'value_loss': value_loss,
            'v_mean': v.mean(),
            'v_max': v.max(),
            'v_min': v.min(),
        }

    def low_actor_loss(self, batch, grad_params):
        """Вычисляет функцию потерь низкоуровневой политики."""
        v1, v2 = self.network.select('value')(batch['observations'], batch['low_actor_goals'])
        nv1, nv2 = self.network.select('value')(batch['next_observations'], batch['low_actor_goals'])
        v = (v1 + v2) / 2
        nv = (nv1 + nv2) / 2
        adv = jax.lax.stop_gradient(nv - v)
        # Ранее использовавшийся вариант: exp_a = jnp.exp(jnp.clip(adv * self.config['low_alpha'], a_max=5.0))
        exp_a = jnp.exp(jnp.clip(adv * self.config['low_alpha'], max=5.0))


        # Вычисляем представления выбранных промежуточных целей.
        goal_reps = self.network.select('goal_rep')(
            jnp.concatenate([batch['observations'], batch['low_actor_goals']], axis=-1),
            params=grad_params,
        )
        # Останавливаем распространение градиентов через представления целей.
        goal_reps = jax.lax.stop_gradient(goal_reps)
        
        dist = self.network.select('low_actor')(batch['observations'], goal_reps, goal_encoded=True, params=grad_params)
        log_prob = dist.log_prob(batch['actions'])
        actor_loss = -(exp_a * log_prob).mean()

        return actor_loss, {
            'actor_loss': actor_loss,
            'adv': adv.mean(),
            'bc_log_prob': log_prob.mean(),
            'mse': jnp.mean((dist.mode() - batch['actions']) ** 2),
            'std': jnp.mean(dist.scale_diag),
        }
    

    def high_actor_loss(self, batch, grad_params):
        """Вычисляет функцию потерь высокоуровневой политики намерений."""
        target = self.network.select('goal_rep')(jnp.concatenate([batch['observations'], batch['high_actor_targets']], axis=-1))

        v1, v2 = self.network.select('value')(batch['observations'], batch['high_actor_goals'])
        nv1, nv2 = self.network.select('value')(batch['high_actor_targets'], batch['high_actor_goals'])
        v = (v1 + v2) / 2
        nv = (nv1 + nv2) / 2
        adv = nv - v

        # Ранее использовавшийся вариант: exp_a = jnp.exp(jnp.clip(adv * self.config['high_alpha'], a_max=5.0))
        exp_a = jnp.exp(jnp.clip(adv * self.config['high_alpha'], max=5.0))

        dist = self.network.select('high_actor')(batch['observations'], batch['high_actor_goals'], params=grad_params)
        log_prob = dist.log_prob(target)
        actor_loss = -(exp_a * log_prob).mean()

        return actor_loss, {
            'actor_loss': actor_loss,
            'adv': adv.mean(),
            'bc_log_prob': log_prob.mean(),
            'mse': jnp.mean((dist.mode() - target) ** 2),
            'std': jnp.mean(dist.scale_diag),
        }

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        """Собирает общую функцию потерь всех обучаемых компонентов."""
        info = {}

        value_loss, value_info = self.value_loss(batch, grad_params)
        for k, v in value_info.items():
            info[f'value/{k}'] = v

        low_actor_loss, low_actor_info = self.low_actor_loss(batch, grad_params)
        for k, v in low_actor_info.items():
            info[f'low_actor/{k}'] = v

        high_actor_loss, high_actor_info = self.high_actor_loss(batch, grad_params)
        for k, v in high_actor_info.items():
            info[f'high_actor/{k}'] = v

        loss = value_loss + low_actor_loss + high_actor_loss
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
        self.target_update(new_network, 'value')

        return self.replace(network=new_network, rng=new_rng), info

    @jax.jit
    def sample_actions(self, observations, goals=None, seed=None, temperature=1.0):
        """Выбирает действие исходной политики по состоянию и намерению."""
        high_seed, low_seed = jax.random.split(seed)

        high_dist = self.network.select('high_actor')(observations, goals, temperature=temperature)
        goal_reps = high_dist.sample(seed=high_seed)
        goal_reps = goal_reps / jnp.linalg.norm(goal_reps, axis=-1, keepdims=True) * jnp.sqrt(goal_reps.shape[-1])

        low_dist = self.network.select('low_actor')(observations, goal_reps, goal_encoded=True, temperature=temperature)
        actions = low_dist.sample(seed=low_seed)

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
        ex_goals = ex_batch['value_goals']
        ex_latents = jnp.zeros([ex_observations.shape[0], config['rep_dim']])

        # Строим зависящее от состояния нормализованное представление подцели phi([s;g]).
        goal_rep_def = nn.Sequential([
            MLP(
                hidden_dims=(*config['value_hidden_dims'], config['rep_dim']),
                activate_final=False,
                layer_norm=config['layer_norm'],
            ), 
            LengthNormalize(),  
        ])

        # Оценка ценности: V(s, phi([s;g])).
        value_encoder_def = GCEncoder(state_encoder=Identity(), concat_encoder=goal_rep_def)
        target_value_encoder_def = GCEncoder(state_encoder=Identity(), concat_encoder=goal_rep_def)

        # Создаём сети ценности и политики.
        value_def = GCValue(
            hidden_dims=config['value_hidden_dims'],
            value_dim=1,
            layer_norm=config['layer_norm'],
            num_ensembles=2,
            gc_encoder=value_encoder_def,
        )
        target_value_def = GCValue(
            hidden_dims=config['value_hidden_dims'],
            value_dim=1,
            layer_norm=config['layer_norm'],
            num_ensembles=2,
            gc_encoder=target_value_encoder_def,
        )

        low_actor_def = GCActor(
            hidden_dims=config['actor_hidden_dims'],
            action_dim=action_dim,
            state_dependent_std=False,
            const_std=config['const_std'],
            gc_encoder=None,
        )

        high_actor_def = GCActor(
            hidden_dims=config['actor_hidden_dims'],
            action_dim=config['rep_dim'],
            state_dependent_std=False,
            const_std=config['const_std'],
            gc_encoder=None,
        )

        network_info = dict(
            goal_rep=(goal_rep_def, (jnp.concatenate([ex_observations, ex_goals], axis=-1))),
            value=(value_def, (ex_observations, ex_goals)),
            target_value=(target_value_def, (ex_observations, ex_goals)),
            low_actor=(low_actor_def, (ex_observations, ex_latents)),
            high_actor=(high_actor_def, (ex_observations, ex_goals)),
        )
        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}

        network_def = ModuleDict(networks)
        network_params = network_def.init(init_rng, **network_args)['params']
        network_tx = optax.adam(learning_rate=config['lr'])        
        network = TrainState.create(network_def, network_params, tx=network_tx)

        params = network.params
        params['modules_target_value'] = jax.tree_util.tree_map(lambda x: x, params['modules_value'])

        return cls(rng, network=network, config=flax.core.FrozenDict(**config))


def get_config():
    config = ml_collections.ConfigDict(
        dict(
            # Настройки обучения и архитектуры агента.
            agent_name='hiql',  # Название реализации агента.
            lr=3e-4,  # Шаг обновления обучаемых параметров.
            batch_size=1024,  # Количество примеров в одном обучающем блоке.
            actor_hidden_dims=(512, 512, 512),  # Размеры скрытых слоёв сети политики.
            value_hidden_dims=(512, 512, 512),  # Размеры скрытых слоёв модели ценности.
            layer_norm=True,  # Использовать ли нормализацию скрытых слоёв.
            discount=0.99,  # Коэффициент дисконтирования будущих состояний.
            tau=0.005,  # Скорость плавного обновления целевой сети.
            expectile=0.7,  # Коэффициент асимметрии оценки функции ценности.
            low_alpha=3.0,  # Температура взвешивания действий низкоуровневой политики.
            high_alpha=3.0,  # Температура взвешивания высокоуровневой политики.
            const_std=True,  # Использовать ли постоянный разброс действий обеих политик.
            
            # Настройки подготовки офлайн-набора данных.
            dataset_class='HGCDataset',  # Имя класса используемого набора данных.
            relabeling=True,  # Пересчитывать ли награды для выбранной цели.
            value_p_curgoal=0.2,  # Вероятность выбрать текущее состояние целью оценки ценности.
            value_p_trajgoal=0.5,  # Вероятность выбрать будущую точку той же траектории целью ценности.
            value_p_randomgoal=0.3,  # Вероятность выбрать случайное состояние целью ценности.
            value_geom_sample=True,  # Использовать ли геометрическое распределение для будущих целей ценности.
            actor_p_curgoal=0.0,  # Вероятность выбрать текущее состояние целью политики.
            actor_p_trajgoal=1.0,  # Вероятность выбрать будущую точку той же траектории целью политики.
            actor_p_randomgoal=0.0,  # Вероятность выбрать случайное состояние целью политики.
            actor_geom_sample=False,  # Использовать ли геометрическое распределение для будущих целей политики.
            gc_negative=False,  # Выбор схемы награды: 0 при успехе и -1 иначе либо 1 при успехе и 0 иначе.
            p_aug=0.0,  # Вероятность случайного преобразования изображения.
            frame_stack=ml_collections.config_dict.placeholder(int),  # Количество объединяемых последовательных кадров.
            
            # Специальные настройки иерархического обучения ценности.
            subgoal_steps=25,  # Число шагов до выбранной промежуточной цели.
            rep_dim=10,  # Размерность представления цели.
        )
    )
    return config
