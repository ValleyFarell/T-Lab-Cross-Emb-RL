"""Высокоуровневая политика поверх заранее обученных FB-представлений."""

from typing import Any
import os
import json
import flax
import jax
import jax.numpy as jnp
import ml_collections
import optax

from utils.flax_utils import ModuleDict, TrainState, nonpytree_field, restore_agent

from utils.networks import GCActor, GCValue
from agents import FBAgent

class FBHierarchicalAgent(flax.struct.PyTreeNode):
    """Обучает верхний уровень управления поверх готового FB-агента."""
    
    rng: Any
    network: Any
    config: Any = nonpytree_field()
    
    
    def load_agent_from_frozen(self, FLAGS, config, example_batch):
        flags_path = os.path.join(config['frozen_path'], "flags.json")
        with open(flags_path, "r") as f:
            saved_flags = json.load(f)
            
        frozen_config = saved_flags['agent']
        FB_agent = FBAgent.create(FLAGS.seed, example_batch, frozen_config)
        FB_agent = restore_agent(FB_agent, config['frozen_path'], config['restore_epoch'])
        
        # Временно открываем параметры для изменения.
        agent_params = flax.core.unfreeze(self.network.params)
        fb_params = flax.core.unfreeze(FB_agent.network.params)
        
        for module_name in fb_params.keys():
            if module_name in ['modules_backward_repr', 'modules_forward_repr', 'modules_actor']:
                agent_params[module_name] = fb_params[module_name]
                
        # Снова запрещаем изменение параметров.
        agent = self.replace(network=self.network.replace(params=agent_params))
        return agent
    
    def normalize_z(self, z):
        return z / (jnp.linalg.norm(z, axis=-1, keepdims=True) + 1e-8) * jnp.sqrt(self.config['latent_dim'])
    
    
    def successor_measure_extract(self, observations, z_goals, z_intents):
        z_intents = self.normalize_z(z_intents)
        forward_reps = self.network.select('margin_forward_repr')(observations, z_intents, goal_encoded=True)
        return jnp.sum(forward_reps * z_goals, axis=1)
    
    
    def marginalize_forward_loss(self, batch, grad_params, rng): 
        observations = batch['observations']
        new_rng, rng = jax.random.split(self.rng)

        latents_norm, _ = self.sample_latents(batch['observations'], rng, self.config['margin_latent_mix_prob'], permute_bool=True)
        # Ранее использовавшийся вариант: latents = batch['latents']
        
        # Выбираем действия из распределения политики.
        dist = self.network.select('actor')(observations, latents_norm, goal_encoded=True)
            
        def sample_action(seed):
            return jnp.clip(dist.sample(seed=seed), -1, 1)
        
        seeds = jax.random.split(new_rng, self.config['num_actions_sampled'])
        actions = jax.vmap(sample_action)(seeds)
        
        target_forward_repr = jax.vmap(lambda a: self.network.select('forward_repr')(observations, latents_norm, actions=a, goal_encoded=True))(actions)
        target_forward_repr = jnp.mean(target_forward_repr, axis=0) # Объединяем значения по действиям.
        target_forward_repr = jnp.mean(target_forward_repr, axis=0) # Объединяем значения по участникам ансамбля.
        target_forward_repr = jax.lax.stop_gradient(target_forward_repr)
    
        # Вычисляем обученное усреднённое прямое представление.
        margin_forward_repr = self.network.select('margin_forward_repr')(observations, latents_norm, goal_encoded=True, params=grad_params)
        marginalization_loss = jnp.mean((target_forward_repr - margin_forward_repr) ** 2)
    
        return marginalization_loss, {
            'marginalization_loss': marginalization_loss,
            'target_forward_mean': target_forward_repr.mean(),
            'margin_forward_mean': margin_forward_repr.mean(),
        }



    def high_actor_loss(self, batch, grad_params, rng): 
        """Вычисляет функцию потерь высокоуровневой политики намерений."""
        obs = batch['observations']
        subgoals = batch['high_actor_targets']   
        goals = batch['high_actor_goals']
        new_rng, rng = jax.random.split(self.rng)


        latents_norm, latents = jax.lax.stop_gradient(self.sample_latents(subgoals, rng, 0.0, permute_bool=False))
        z_goals_norm, z_goals = jax.lax.stop_gradient(self.sample_latents(goals, new_rng, self.config['critic_latent_mix_prob'], permute_bool=False))

        Msww = self.successor_measure_extract(obs, latents, latents)
        Mwww = self.successor_measure_extract(subgoals, latents, latents)       
        Vwrr = self.successor_measure_extract(subgoals, z_goals, z_goals)
        Vrstar = self.successor_measure_extract(obs, z_goals, z_goals)
        Vswr = self.successor_measure_extract(obs, z_goals, latents)

        adv = jax.lax.stop_gradient(Vswr + Msww/Mwww*Vwrr - Vrstar)
        exp_a = jnp.exp(jnp.clip(adv * self.config['high_alpha'], max=5.0))

        dist = self.network.select('high_actor')(obs, z_goals_norm, params=grad_params)
        log_prob = dist.log_prob(latents_norm)

        actor_loss = -(exp_a * log_prob).mean()
        

        return actor_loss, {
            'actor_loss': actor_loss,
            'adv': adv.mean(),
            'adv_std': adv.std(),
            'adv_min': adv.min(),
            'adv_max': adv.max(),
            'bc_log_prob': log_prob.mean(),
            'mse': jnp.mean((dist.mode() - latents_norm) ** 2),
            'mse_z_latents': jnp.mean((z_goals_norm - latents_norm) ** 2),
            'Msww': Msww.mean(),
            'Vwrr': Vwrr.mean(),
            'Vrstar': Vrstar.mean(),
            'active_exps': (jnp.abs(exp_a) > 1e-3).mean(),
            'saturated': jnp.mean(adv * self.config['high_alpha'] >= 5.0),
        }
    
       
    
    
    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        """Собирает общую функцию потерь всех обучаемых компонентов."""
        info = {}
        rng = rng if rng is not None else self.rng
        rng, high_actor_rng, marginalize_rng = jax.random.split(rng, 3)

        high_actor_loss, high_actor_info = self.high_actor_loss(batch, grad_params, high_actor_rng)
        for k, v in high_actor_info.items():
            info[f'high_actor/{k}'] = v
            
        margin_loss, margin_info = self.marginalize_forward_loss(batch, grad_params, marginalize_rng)
        for k, v in margin_info.items():
            info[f'margin/{k}'] = v
            
        loss = high_actor_loss + margin_loss
        return loss, info


    @jax.jit
    def update(self, batch):
        """Обновляет агента и возвращает новое состояние вместе с диагностикой."""
        new_rng, rng = jax.random.split(self.rng)

        def loss_fn(grad_params):
            return self.total_loss(batch, grad_params, rng=rng)

        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)
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
        latent = self.normalize_z(latent)

        return latent

    @jax.jit
    def sample_latents(self, targets, rng, latent_mix_prob, permute_bool=False):
        """Формирует намерения и связанные внутренние награды."""
        batch_size = targets.shape[0]
        
        rng, latent_rng, perm_rng, mix_rng = jax.random.split(rng, 4)

        latents = jax.random.normal(latent_rng, shape=(batch_size, self.config['latent_dim']))
        # Ранее использовавшийся вариант: if self.config['normalize_latent']:
        norm_latents = self.normalize_z(latents)
        
        backward_reprs = self.network.select('backward_repr')(targets)
        
        latent_backward_reprs = jax.lax.cond(
            permute_bool,
            lambda _: backward_reprs[jax.random.permutation(perm_rng, jnp.arange(batch_size))],
            lambda _: backward_reprs,
            operand=None
        )
        
        # Ранее использовавшийся вариант: if self.config['normalize_latent']:
        norm_latent_backward_reprs = self.normalize_z(latent_backward_reprs)
        
        flags_latents = jax.random.uniform(mix_rng, (batch_size, 1)) < latent_mix_prob
        latents = jnp.where(flags_latents, latents, latent_backward_reprs)
        norm_latents = jnp.where(flags_latents, norm_latents, norm_latent_backward_reprs)

        return norm_latents, latents

    @jax.jit
    def sample_actions(self, observations, latents=None, seed=None, temperature=1.0):
        """Выбирает действие исходной политики по состоянию и намерению."""

        high_seed, low_seed = jax.random.split(seed)

        high_dist = self.network.select('high_actor')(observations, latents, goal_encoded=True, temperature=temperature)
        goal_reps = high_dist.sample(seed=high_seed)        
        goal_reps = self.normalize_z(goal_reps)
        
        low_dist = self.network.select('actor')(observations, goal_reps, goal_encoded=True, temperature=temperature)
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
        # Ранее использовавшийся вариант: state_dim = ex_observations.shape[-1]
        action_dim = ex_actions.shape[-1]       
        ex_latents = jnp.ones((*ex_actions.shape[:-1], config['latent_dim']))


        # Создаём вычислительные сети агента.
        forward_repr_def = GCValue(
            hidden_dims=config['forward_repr_hidden_dims'],
            value_dim=config['latent_dim'],
            layer_norm=config['forward_repr_layer_norm'],
            num_ensembles=2,
        )
                
        backward_repr_def = GCValue(
            hidden_dims=config['backward_repr_hidden_dims'],
            value_dim=config['latent_dim'],
            layer_norm=config['backward_repr_layer_norm'],
            num_ensembles=1,
        )
        
        actor_def = GCActor(
            hidden_dims=config['actor_hidden_dims'],
            action_dim=action_dim,
            state_dependent_std=False,
            tanh_squash=config['tanh_squash'],
            layer_norm=config['actor_layer_norm'],
            const_std=config['const_std'],
        )

        # Новые модули, параметры которых разрешено обучать.

        margin_forward_repr_def = GCValue(
            hidden_dims=config['forward_repr_hidden_dims'],
            layer_norm=config['forward_repr_layer_norm'],
            num_ensembles=1,
            value_dim=config['latent_dim'],
        )
        
        high_actor_def = GCActor(
            hidden_dims=config['actor_hidden_dims'],
            action_dim=config['latent_dim'],
            state_dependent_std=False,
            const_std=config['const_std'],
            layer_norm=config['actor_layer_norm'],
        )
        
        network_info = dict(
            forward_repr=(forward_repr_def, (ex_observations, ex_latents, ex_actions, None, True)),
            backward_repr=(backward_repr_def, (ex_observations,)),
            actor=(actor_def, (ex_observations, ex_latents, True)),
            
            # Новые модули, параметры которых разрешено обучать.
            margin_forward_repr=(margin_forward_repr_def, (ex_observations, ex_latents, None, None, True)),
            high_actor=(high_actor_def, (ex_observations, ex_latents, True)),
        )
        
        def mask_fn(params):
            flat = flax.traverse_util.flatten_dict(params)
            mask = {}
        
            for k in flat:
                module_name = k[0]  # Например: modules_forward_repr.
        
                if module_name in [
                    'modules_margin_forward_repr',
                    'modules_high_actor',
                ]:
                    mask[k] = 'train'
                else:
                    mask[k] = 'frozen'
        
            return flax.traverse_util.unflatten_dict(mask)


        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}

        network_def = ModuleDict(networks)
        network_params = network_def.init(init_rng, **network_args)['params']
        network_tx = optax.multi_transform(
            {
                'train': optax.adam(config['lr']),
                'frozen': optax.set_to_zero(),
            },
            mask_fn(network_params)
        )
        network = TrainState.create(network_def, network_params, tx=network_tx)    

        return cls(rng, network=network, config=flax.core.FrozenDict(**config))


def get_config():
    config = ml_collections.ConfigDict(
        dict(
            # Настройки обучения и архитектуры агента.
            agent_name='hfb',  # Название реализации агента.
            lr=1e-4,  # Шаг обновления обучаемых параметров.
            batch_size=1024,  # Количество примеров в одном обучающем блоке.
            actor_hidden_dims=(512, 512, 512, 512),  # Размеры скрытых слоёв сети политики.
            forward_repr_hidden_dims=(512, 512, 512, 512),  # Размеры скрытых слоёв прямого представления F.
            backward_repr_hidden_dims=(512, 512, 512, 512),  # Размеры скрытых слоёв обратного представления B.
            actor_layer_norm=False,  # Использовать ли нормализацию слоёв политики.
            forward_repr_layer_norm=False,  # Использовать ли нормализацию слоёв прямого представления.
            backward_repr_layer_norm=False,  # Использовать ли нормализацию слоёв обратного представления.
            activation='gelu',  # Функция активации нейронной сети.
            latent_dim=128,  # Размерность представления намерения.
            discount=0.99,  # Коэффициент дисконтирования будущих состояний.
            tau=0.005,  # Скорость плавного обновления целевой сети.
            normalize_latent=True,  # Нормализовать ли обратные представления состояний.
            reward_temperature=0.0,  # Температура весов, определяемых наградой.
            repr_agg='mean',  # Способ объединения оценок целевого FB-ансамбля.
            # Сохранённый вариант: orthonorm_coeff=0.0, — Коэффициент регуляризации ортонормальности.
            margin_latent_mix_prob=0.5,  # Вероятность заменить случайное намерение реальным обратным представлением.
            # Сохранённый вариант: alpha=0.3, — Вес воспроизведения офлайн-действий при обучении политики.
            tanh_squash=True,  # Ограничивать ли действия политики функцией tanh.
            const_std=True,  # Использовать ли постоянный разброс действий политики.
            # Сохранённый вариант: normalize_q_loss=True, — Нормировать ли потери оценки Q.
            num_zero_shot_samples=100_000,  
            
            # Настройки подготовки офлайн-набора данных.
            dataset_class='HGCDataset',  # Имя класса используемого набора данных.
            relabeling=False,  # Пересчитывать ли награды для выбранной цели.
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
            
            # Специальные настройки иерархического FB-агента.
            high_alpha=1e-3,
            frozen_path="",
            restore_epoch=1_000_000,
            num_actions_sampled = 8,
            critic_latent_mix_prob=0.5,  # Вероятность заменить случайное намерение реальным обратным представлением.
        )
    )
    return config