"""Обучение FB-представлений и низкоуровневой политики без верхнего уровня."""

import copy
from typing import Any
import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import ml_collections
import optax
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import GCValue, GCActor

    
class FBpiSwitchNonHierarchicalAgent(flax.struct.PyTreeNode):
    """Обучает FB-представления и низкоуровневую политику без верхнего уровня."""
    
    rng: Any
    network: Any
    config: Any = nonpytree_field()

    @staticmethod
    def expectile_loss(adv, diff, expectile):
        """Вычисляет асимметричную функцию потерь оценки ценности."""
        weight = jnp.where(adv >= 0, expectile, (1 - expectile))
        return weight * (diff**2)
    
    def normalize_z(self, z):
        return z / (jnp.linalg.norm(z, axis=-1, keepdims=True) + 1e-8) * jnp.sqrt(self.config['latent_dim'])
    
    def successor_measure_extract(self, observations, z_goals, z_intents, module, grad_params=None):
        # Модуль является текущим или целевым прямым представлением.
        z_intents = self.normalize_z(z_intents)
        forward_reps = self.network.select(module)(observations, z_intents, goal_encoded=True, params=grad_params)
        return jnp.sum(forward_reps * z_goals[None, :, :], axis=-1)


    def value_loss(self, batch, rng, grad_params):
        """Вычисляет функцию потерь модели ценности."""
        obs = batch['observations']
        next_obs = batch['next_observations']
        dummy_goals = batch['value_goals']
        z_dummy_goals = self.network.select('backward_repr')(dummy_goals, params=grad_params)
        _, z = self.sample_latents(batch['intention_goals'], rng, self.config['critic_latent_mix_prob']) # Место возможной замены намерения z во вспомогательном варианте.

        # Вычисляем ковариационную матрицу представлений.
        Bs_ng = self.network.select('backward_repr')(obs)
        cov = (Bs_ng.T @ Bs_ng) / Bs_ng.shape[0]
        inv_cov = jax.scipy.linalg.solve(cov, jnp.eye(cov.shape[-1]), assume_a='pos')
    
        # Ошибка временной разности оценки.
        (next_v1, next_v2) = self.successor_measure_extract(next_obs, z_dummy_goals, z, 'target_forward_repr')
        q1 = batch['rewards'] + self.config['discount'] * next_v1 
        q2 = batch['rewards'] + self.config['discount'] * next_v2
        q1, q2 = jax.lax.stop_gradient(q1), jax.lax.stop_gradient(q2)
        (v1, v2) =  self.successor_measure_extract(obs, z_dummy_goals, z, 'forward_repr', grad_params=grad_params)
        v = (v1 + v2) / 2

        # Ошибка направления представления.
        reward_z = (jnp.einsum('bd,bd->b', Bs_ng @ inv_cov, z) / self.config['latent_dim'])[:, None]
        (next_v1_adv, next_v2_adv) = self.successor_measure_extract(next_obs, z, z, 'forward_repr', grad_params=grad_params)
        next_v_adv = jnp.minimum(next_v1_adv, next_v2_adv)
        q_adv = reward_z + self.config['discount'] * next_v_adv
        
        (v1_adv, v2_adv) = self.successor_measure_extract(obs, z, z, 'forward_repr', grad_params=grad_params)
        v_adv = (v1_adv + v2_adv) / 2
        adv = jax.lax.stop_gradient(q_adv - v_adv)

        value_loss1 = self.expectile_loss(adv, q1 - v1, self.config['expectile']).mean()
        value_loss2 = self.expectile_loss(adv, q2 - v2, self.config['expectile']).mean()
        value_loss = value_loss1 + value_loss2
    
        return value_loss, {
            'value_loss': value_loss,
            'v_mean': v.mean(),
            'v_max': v.max(),
            'v_min': v.min(),
        }


    def orthonormal_loss(self, batch, grad_params):
        Bs = self.network.select('backward_repr')(batch['observations'], params=grad_params)
    
        G = Bs @ Bs.T                   
        dot_sq = G**2
        dot_sq = dot_sq - jnp.diag(jnp.diag(dot_sq))
        norm_sq = jnp.sum(Bs**2, axis=1)
       
        loss = (jnp.mean(dot_sq) - 2*jnp.mean(norm_sq))

        return loss, {
            'ortho_loss': loss,
        }


    def actor_loss(self, batch, rng, grad_params):
        """Вычисляет функцию потерь обучаемой политики действий."""
        z_norm, z = self.sample_latents(batch['actor_goals'], rng, self.config['actor_latent_mix_prob'])
        
        v1, v2 = self.successor_measure_extract(batch['observations'], z, z, 'forward_repr')
        nv1, nv2 = self.successor_measure_extract(batch['next_observations'], z, z, 'forward_repr')
                
        v = (v1 + v2) / 2
        nv = (nv1 + nv2) / 2
        adv = jax.lax.stop_gradient(nv - v)
        exp_a = jnp.exp(jnp.clip(adv * self.config['alpha'], max=5.0))
            
        dist = self.network.select('actor')(batch['observations'], jax.lax.stop_gradient(z_norm), goal_encoded=True, params=grad_params)
        
        log_prob = dist.log_prob(batch['actions'])
        actor_loss = -(exp_a * log_prob).mean()

        return actor_loss, {
            'actor_loss': actor_loss,
            'adv': adv.mean(),
            'exp_a': exp_a.mean(),
            'adv_abs': jnp.abs(adv).mean(),
            'bc_log_prob': log_prob.mean(),
            'mse': jnp.mean((dist.mode() - batch['actions']) ** 2),
        }


    
    
    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        """Собирает общую функцию потерь всех обучаемых компонентов."""
        info = {}
        rng = rng if rng is not None else self.rng
        rng, value_rng, actor_rng = jax.random.split(rng, 3)
        
        value_loss, value_info = self.value_loss(batch, value_rng, grad_params)
        for k, v in value_info.items():
            info[f'value/{k}'] = v
            
        orthonormal_loss, orthonormal_info = self.orthonormal_loss(batch, grad_params)
        for k, v in orthonormal_info.items():
            info[f'orthonormal/{k}'] = v

        actor_loss, actor_info = self.actor_loss(batch, actor_rng, grad_params)
        for k, v in actor_info.items():
            info[f'actor/{k}'] = v

        loss = value_loss + self.config['orthonorm_coeff']*orthonormal_loss + actor_loss 

        return loss, info
    

    def target_update(self, network, module_name):
        """Плавно обновляет параметры целевой нейронной сети."""
        new_target_params = jax.tree_util.tree_map(
            lambda p, tp: p * self.config['tau'] + tp * (1 - self.config['tau']),
            network.params[f'modules_{module_name}'],
            network.params[f'modules_target_{module_name}'],
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
    def sample_latents(self, targets, rng, latent_mix_prob):
        """Формирует намерения и связанные внутренние награды."""
        batch_size = targets.shape[0]
        
        rng, latent_rng, mix_rng = jax.random.split(rng, 3)

        latents = jax.random.normal(latent_rng, shape=(batch_size, self.config['latent_dim']))
        norm_latents = self.normalize_z(latents)
        
        backward_reprs = self.network.select('backward_repr')(targets)
        norm_backward_reprs = self.normalize_z(backward_reprs)
        
        flags_latents = jax.random.uniform(mix_rng, (batch_size, 1)) < latent_mix_prob
        latents = jnp.where(flags_latents, latents, backward_reprs)
        norm_latents = jnp.where(flags_latents, norm_latents, norm_backward_reprs)

        return norm_latents, latents




    @jax.jit   
    def sample_actions(self, observations, latents=None, seed=None, temperature=1.0):
        """Выбирает действие исходной политики по состоянию и намерению."""
        dist = self.network.select('actor')(observations, latents, goal_encoded=True, temperature=temperature)
        actions = dist.sample(seed=seed)
        return jnp.clip(actions, -1, 1)



    @classmethod
    def create(cls, seed, ex_batch, config):
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
            # Ранее использовавшийся вариант: tanh_squash=config['tanh_squash'],
            # Ранее использовавшийся вариант: activations=getattr(nn, config['activation']),
            # Ранее использовавшийся вариант: layer_norm=config['actor_layer_norm'],
            const_std=config['const_std'],
        )
        
        network_info = dict(
            forward_repr=(forward_repr_def, (ex_observations, ex_latents, None, None, True)),
            backward_repr=(backward_repr_def, (ex_observations,)),
            target_forward_repr=(copy.deepcopy(forward_repr_def), (ex_observations, ex_latents, None, None, True)),
            target_backward_repr=(copy.deepcopy(backward_repr_def), (ex_observations,)),
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
            # Настройки обучения и архитектуры агента.
            agent_name='fbpiswitch_nonh',  # Название реализации агента.
            lr=3e-4,  # Шаг обновления обучаемых параметров.
            batch_size=1024,  # Количество примеров в одном обучающем блоке.
            actor_hidden_dims=(512, 512, 512),  # Размеры скрытых слоёв сети политики.
            forward_repr_hidden_dims=(512, 512, 512),  # Размеры скрытых слоёв прямого представления F.
            backward_repr_hidden_dims=(512, 512, 512),  # Размеры скрытых слоёв обратного представления B.
            actor_layer_norm=False,  # Использовать ли нормализацию слоёв политики.
            forward_repr_layer_norm=True,  # Использовать ли нормализацию слоёв прямого представления.
            backward_repr_layer_norm=True,  # Использовать ли нормализацию слоёв обратного представления.
            activation='gelu',  # Функция активации нейронной сети.
            discount=0.99,  # Коэффициент дисконтирования будущих состояний.
            tau=0.005,  # Скорость плавного обновления целевой сети.
            expectile=0.7,  # Коэффициент асимметрии оценки функции ценности.
            alpha=3.0,  # Температура взвешивания действий низкоуровневой политики.
            const_std=True,  # Использовать ли постоянный разброс действий обеих политик.
            
            # Настройки подготовки офлайн-набора данных.
            dataset_class='GCDataset',  # Имя класса используемого набора данных.
            value_p_curgoal=0.2,  # Вероятность выбрать текущее состояние целью оценки ценности.
            value_p_trajgoal=0.5,  # Вероятность выбрать будущую точку той же траектории целью ценности.
            value_p_randomgoal=0.3,  # Вероятность выбрать случайное состояние целью ценности.
            value_geom_sample=True,  # Использовать ли геометрическое распределение для будущих целей ценности.
            actor_p_curgoal=0.2,  # Вероятность выбрать текущее состояние целью политики.
            actor_p_trajgoal=0.5,  # Вероятность выбрать будущую точку той же траектории целью политики.
            actor_p_randomgoal=0.3,  # Вероятность выбрать случайное состояние целью политики.
            actor_geom_sample=False,  # Использовать ли геометрическое распределение для будущих целей политики.
            gc_negative=False,  # Выбор схемы награды: 0 при успехе и -1 иначе либо 1 при успехе и 0 иначе.
            p_aug=0.0,  # Вероятность случайного преобразования изображения.
            frame_stack=ml_collections.config_dict.placeholder(int),  # Количество объединяемых последовательных кадров.
            # Сохранённый вариант: tanh_squash=True, — Ограничивать ли действия политики функцией tanh.
            reward_temperature=0.0,  # Температура весов, определяемых наградой.
            num_zero_shot_samples = 100_000,


            # Специальные настройки данного алгоритма.
            relabeling=True,  # Пересчитывать ли награды для выбранной цели.
            latent_dim=128,  # Размерность скрытого FB-представления.
            normalize_latent=True,  # Нормализовать ли обратные представления состояний.
            orthonorm_coeff=1e-3,  # Коэффициент регуляризации ортонормальности.
            actor_latent_mix_prob=0.5,  # Вероятность заменить случайное намерение реальным обратным представлением.
            critic_latent_mix_prob=0.5,  # Вероятность заменить случайное намерение реальным обратным представлением.
        )
    )
    return config
