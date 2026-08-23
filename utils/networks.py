"""Исходные архитектуры политик, функций ценности и представлений."""

from typing import Any, Optional, Sequence
import distrax
import flax.linen as nn
import jax.numpy as jnp


def default_init(scale=1.0):
    """Возвращает стандартный способ начальной инициализации весов."""
    return nn.initializers.variance_scaling(scale, 'fan_avg', 'uniform')


def ensemblize(cls, num_qs, out_axes=0, **kwargs):
    """Создаёт несколько независимых экземпляров вычислительного модуля."""
    return nn.vmap(
        cls,
        variable_axes={'params': 0},
        split_rngs={'params': True},
        in_axes=None,
        out_axes=out_axes,
        axis_size=num_qs,
        **kwargs,
    )


class Identity(nn.Module):
    """Возвращает вход без преобразования."""

    def __call__(self, x):
        return x


class MLP(nn.Module):
    """Вычисляет многослойную полносвязную нейронную сеть."""

    hidden_dims: Sequence[int]
    activations: Any = nn.gelu
    initial_activation: Any = nn.tanh
    activate_final: bool = False
    kernel_init: Any = default_init()
    layer_norm: bool = False

    @nn.compact
    def __call__(self, x):
        # Архитектура адаптирована из исходной реализации HILP: https://github.com/seohongpark/HILP/blob/be2431bbb75e3b13cbdb1dec11776c42ef0f1593/hilp_zsrl/url_benchmark/agent/fb_modules.py#L148-L193.
        for i, size in enumerate(self.hidden_dims):
            x = nn.Dense(size, kernel_init=self.kernel_init)(x)
            if i == 0:
                if self.layer_norm:
                    x = nn.LayerNorm()(x)
                x = self.initial_activation(x)
            elif i + 1 < len(self.hidden_dims) or self.activate_final:
                 x = self.activations(x)
        return x

class LengthNormalize(nn.Module):
    """Приводит вектор к заданной норме."""

    @nn.compact
    def __call__(self, x):
        return x / jnp.linalg.norm(x, axis=-1, keepdims=True) * jnp.sqrt(x.shape[-1])
    

class TransformedWithMode(distrax.Transformed):
    """Дополняет преобразованное распределение вычислением наиболее вероятного значения."""

    def mode(self):
        return self.bijector.forward(self.distribution.mode())


class GCActor(nn.Module):
    """Выбирает действие с учётом текущего состояния и поставленной цели."""

    hidden_dims: Sequence[int]
    action_dim: int
    activations: Any = nn.gelu
    layer_norm: bool = False
    log_std_min: Optional[float] = -5
    log_std_max: Optional[float] = 2
    tanh_squash: bool = False
    state_dependent_std: bool = False
    const_std: bool = True
    final_fc_init_scale: float = 1e-2
    gc_encoder: nn.Module = None

    def setup(self):
        self.actor_net = MLP(
            self.hidden_dims,
            activations=self.activations,
            activate_final=True,
            layer_norm=self.layer_norm,
        )

        self.mean_net = nn.Dense(self.action_dim, kernel_init=default_init(self.final_fc_init_scale))
        if self.state_dependent_std:
            self.log_std_net = nn.Dense(self.action_dim, kernel_init=default_init(self.final_fc_init_scale))
        else:
            if not self.const_std:
                self.log_stds = self.param('log_stds', nn.initializers.zeros, (self.action_dim,))

    def __call__(
        self,
        observations,
        goals=None,
        goal_encoded=False,
        temperature=1.0,
    ):
        """Вычисляет результат модуля для переданных входных данных.

        Параметры:
            observations: блок полных состояний робота.
            goals: целевые наблюдения или их представления.
            goal_encoded: параметр исходного вычисления.
            temperature: степень случайности выбора действия.
        """
        if self.gc_encoder is not None:
            inputs = self.gc_encoder(observations, goals, goal_encoded=goal_encoded)
        else:
            inputs = [observations]
            if goals is not None:
                inputs.append(goals)
            inputs = jnp.concatenate(inputs, axis=-1)
        outputs = self.actor_net(inputs)

        means = self.mean_net(outputs)
        if self.state_dependent_std:
            log_stds = self.log_std_net(outputs)
        else:
            if self.const_std:
                log_stds = jnp.zeros_like(means)
            else:
                log_stds = self.log_stds

        log_stds = jnp.clip(log_stds, self.log_std_min, self.log_std_max)

        distribution = distrax.MultivariateNormalDiag(loc=means, scale_diag=jnp.exp(log_stds) * temperature)
        if self.tanh_squash:
            distribution = TransformedWithMode(distribution, distrax.Block(distrax.Tanh(), ndims=1))

        return distribution


class GCValue(nn.Module):
    """Оценивает ценность состояния при заданной цели."""

    hidden_dims: Sequence[int]
    value_dim: int = 1
    activations: Any = nn.gelu
    layer_norm: bool = True
    num_ensembles: int = 1
    gc_encoder: nn.Module = None

    def setup(self):
        mlp_class = MLP

        if self.num_ensembles > 1:
            mlp_class = ensemblize(mlp_class,  self.num_ensembles)

        self.value_net = mlp_class(
            (*self.hidden_dims, self.value_dim),
            activations=self.activations,
            activate_final=False,
            layer_norm=self.layer_norm
        )

    def __call__(self, observations, goals=None, actions=None, goal_actions=None, goal_encoded=False):
        """Вычисляет результат модуля для переданных входных данных.

        Параметры:
            observations: блок полных состояний робота.
            goals: целевые наблюдения или их представления.
            actions: параметр исходного вычисления.
            goal_actions: параметр исходного вычисления.
            goal_encoded: параметр исходного вычисления.
        """
        if self.gc_encoder is not None:
            inputs = [self.gc_encoder(observations, goals, goal_encoded=goal_encoded)]
        else:
            inputs = [observations]
            if goals is not None:
                inputs.append(goals)
        if actions is not None:
            inputs.append(actions)
        if goal_actions is not None:
            inputs.append(goal_actions)
        inputs = jnp.concatenate(inputs, axis=-1)

        if self.value_dim == 1:
            v = self.value_net(inputs).squeeze(-1)
        else:
            v = self.value_net(inputs)

        return v

class GCBilinearValue(nn.Module):
    """Оценивает ценность состояния и цели билинейной моделью."""

    hidden_dims: Sequence[int]
    latent_dim: int
    activations: Any = nn.gelu
    layer_norm: bool = True
    num_ensembles: int = 1
    state_encoder: nn.Module = None
    goal_encoder: nn.Module = None

    def setup(self):
        mlp_class = MLP
        if self.num_ensembles > 1:
            mlp_class = ensemblize(mlp_class,  self.num_ensembles)
        
        self.phi_net = mlp_class(
            (*self.hidden_dims, self.latent_dim),
            activations=self.activations,
            activate_final=False,
            layer_norm=self.layer_norm
        )
        
    def __call__(self, observations, goals, intents=None):
        """Вычисляет результат модуля для переданных входных данных.

        Параметры:
            observations: блок полных состояний робота.
            goals: целевые наблюдения или их представления.
            intents: параметр исходного вычисления.
        """
        if intents is None:
            intents = goals
    
        if self.state_encoder is not None:
            observations = self.state_encoder(observations)
        if self.goal_encoder is not None:
            goals = self.goal_encoder(goals)
            intents = self.goal_encoder(intents)

        phi_inputs = jnp.concatenate([observations, intents], axis=-1)
        phi = self.phi_net(phi_inputs)
        v = (phi * goals / jnp.sqrt(self.latent_dim)).sum(axis=-1)

        return v

        
        
    
    
class ICVFValue(nn.Module):
    """Вычисляет ценность при заданном состоянии, цели и намерении."""

    hidden_dims: Sequence[int]
    value_dim: int = 1
    activations: Any = nn.gelu
    layer_norm: bool = True
    num_ensembles: int = 1
    icvf_encoder: nn.Module = None

    def setup(self):
        mlp_class = MLP

        if self.num_ensembles > 1:
            mlp_class = ensemblize(mlp_class,  self.num_ensembles)

        self.phi_net = mlp_class(
            (*self.hidden_dims, self.value_dim),
            activations=self.activations,
            activate_final=False,
            layer_norm=self.layer_norm
        )
        self.transition_net = mlp_class(
            (*self.hidden_dims, self.value_dim * self.value_dim),
            activations=self.activations,
            activate_final=False,
            layer_norm=self.layer_norm
        )
        self.psi_net = mlp_class(
            (*self.hidden_dims, self.value_dim),
            activations=self.activations,
            activate_final=False,
            layer_norm=self.layer_norm
        )

    def __call__(self, observations, goals=None, intentions=None, actions=None, 
                 goal_actions=None, intention_actions=None, 
                 goal_encoded=False, intention_encoded=False,
                 phis=None, psis=None, transitions=None,
                 info=False):
        """Вычисляет результат модуля для переданных входных данных.

        Параметры:
            observations: блок полных состояний робота.
            goals: целевые наблюдения или их представления.
            intentions: одно или несколько намерений политики.
            actions: параметр исходного вычисления.
            goal_actions: параметр исходного вычисления.
            intention_actions: параметр исходного вычисления.
            goal_encoded: параметр исходного вычисления.
            intention_encoded: параметр исходного вычисления.
            phis: параметр исходного вычисления.
            psis: параметр исходного вычисления.
            transitions: параметр исходного вычисления.
            info: параметр исходного вычисления.
        """
        psi_inputs = []
        transition_inputs = []
        if self.icvf_encoder is not None:
            phi_inputs, psi_inputs, transition_inputs = self.icvf_encoder(
                observations, goals, intentions, 
                goal_encoded=goal_encoded,
                intention_encoded=intention_encoded
            )
        else:
            phi_inputs = [observations]
            if goals is not None:
                psi_inputs.append(goals)
            if intentions is not None:
                transition_inputs.append(intentions)
        if actions is not None:
            phi_inputs.append(actions)
        if goal_actions is not None:
            psi_inputs.append(goal_actions)
        if intention_actions is not None:
            transition_inputs.append(intention_actions)
        if phis is None:
            phi_inputs = jnp.concatenate(phi_inputs, axis=-1)
            phis = self.phi_net(phi_inputs)
        if psis is None:
            if len(psi_inputs) > 0:
                psi_inputs = jnp.concatenate(psi_inputs, axis=-1)
                psis = self.psi_net(psi_inputs)
            else:
                psis = None
        if transitions is None:
            if len(transition_inputs) > 0:
                transition_inputs = jnp.concatenate(transition_inputs, axis=-1)
                transitions = self.transition_net(transition_inputs)
                transitions = transitions.reshape(
                    *transitions.shape[:-1], 
                    self.value_dim, 
                    self.value_dim
                )
            else:
                transitions = None
        
        if phis is not None and psis is not None and transitions is not None:
            if self.num_ensembles > 1:
                inners = jnp.einsum('eij,eijk->eik', phis, transitions)
            else:
                inners = jnp.einsum('ij,ijk->ik', phis, transitions)
            vs = jnp.sum(inners * psis, axis=-1)
            
            if self.value_dim == 1:
                vs = vs.squeeze(-1)
        else:
            vs = None
        
        if info:
            return vs, phis, psis, transitions
        else:
            return vs
