import jax
import jax.numpy as jnp
import flax.core
import flax.linen as nn

from abc import abstractmethod
from collections.abc import Callable
from typing import Optional

from . import Actor
from .mlp import MLP
from ..utils import duplicate


class BaseCriticSingle(nn.Module):
    @staticmethod
    def get_param_batch_sizes(params: flax.core.scope.Collection) -> Optional[int]:
        shape = params['params']['MLP_0']['Dense_0']['kernel'].shape
        match len(shape):
            case 2:
                return None
            case 3:
                return shape[0]
            case _:
                raise ValueError

    @staticmethod
    def get_param_batch_shape(params: flax.core.scope.Collection) -> tuple[int, ...]:
        shape = params['params']['MLP_0']['Dense_0']['kernel'].shape[:-2]
        return shape


class BaseCritic(nn.Module):
    @abstractmethod
    def _critic1(
        self,
        state: jax.Array,
        action: jax.Array,
        *args,
        **kwargs,
    ) -> jax.Array:
        ...

    @abstractmethod
    def _critic2(
        self,
        state: jax.Array,
        action: jax.Array,
        *args,
        **kwargs,
    ) -> jax.Array:
        ...

    @staticmethod
    def get_param_batch_sizes(params: flax.core.scope.Collection) -> Optional[int]:
        shape = params['params']['critic1']['MLP_0']['Dense_0']['kernel'].shape
        match len(shape):
            case 2:
                return None
            case 3:
                return shape[0]
            case _:
                raise ValueError

    @staticmethod
    def get_param_batch_shape(params: flax.core.scope.Collection) -> tuple[int, ...]:
        shape = params['params']['critic1']['MLP_0']['Dense_0']['kernel'].shape[:-2]
        return shape

    @abstractmethod
    def apply_critic1(
        self,
        state: jax.Array,
        action: jax.Array,
        vars: flax.core.scope.VariableDict,
        *args,
        **kwargs,
    ) -> jax.Array:
        ...

    @staticmethod
    def extract_critic1_vars(
        vars: flax.core.scope.VariableDict,
    ) -> flax.core.scope.FrozenVariableDict:
        if isinstance(vars, flax.core.FrozenDict):
            vars = vars.unfreeze()

        assert vars.keys() == {'params'}
        assert vars['params'].keys() == {'critic1', 'critic2'}

        critic1_vars = flax.core.freeze({'params': {'critic1': vars['params']['critic1']}})

        return critic1_vars


class CriticSingle(BaseCriticSingle):

    hidden_layer_sizes: tuple[int, ...]
    n_values: Optional[int] = None
    activation: Callable[[jax.Array], jax.Array] = nn.leaky_relu
    final_activation: Optional[Callable[[jax.Array], jax.Array]] = None

    @nn.compact
    def __call__(
        self,
        state: jax.Array,
        action: jax.Array,
    ) -> jax.Array:
        value = MLP(
            layer_sizes=(*self.hidden_layer_sizes, self.n_values or 1),
            activation=self.activation,
            final_activation=self.final_activation,
        )(jnp.concatenate((state, action), axis=-1))
        assert isinstance(value, jax.Array)
        if self.n_values is None:
            assert value.shape[-1] == 1
            value = jnp.squeeze(value, axis=-1)

        return value


class Critic(BaseCritic):

    hidden_layer_sizes: tuple[int, ...]
    n_values: Optional[int] = None
    activation: Callable[[jax.Array], jax.Array] = nn.leaky_relu
    final_activation: Optional[Callable[[jax.Array], jax.Array]] = None

    def setup(self) -> None:
        params = (
            self.hidden_layer_sizes,
            self.n_values,
            self.activation,
            self.final_activation,
        )
        self.critic1 = CriticSingle(*params)
        self.critic2 = CriticSingle(*params)

    def _critic1(
        self,
        state: jax.Array,
        action: jax.Array,
    ) -> jax.Array:
        return self.critic1(state, action)

    def _critic2(
        self,
        state: jax.Array,
        action: jax.Array,
    ) -> jax.Array:
        return self.critic2(state, action)

    def __call__(
        self,
        state: jax.Array,
        action: jax.Array,
    ) -> jax.Array:
        q1 = self._critic1(state, action)
        q2 = self._critic2(state, action)
        return jnp.concatenate((q1[..., jnp.newaxis], q2[..., jnp.newaxis]), axis=-1)

    def apply_critic1(
        self,
        vars: flax.core.scope.VariableDict,
        state: jax.Array,
        action: jax.Array,
    ) -> jax.Array:
        critic1_vars = self.extract_critic1_vars(vars)
        q1 = self.apply(critic1_vars, state, action, method=self._critic1)
        assert isinstance(q1, jax.Array)
        return q1


class PolicyCriticSingle(BaseCriticSingle):

    encoder_layer_sizes: tuple[int, ...]
    hidden_layer_sizes: tuple[int, ...]
    n_values: Optional[int] = None
    encoder_activation: Callable[[jax.Array], jax.Array] = nn.leaky_relu
    encoder_final_activation: Optional[Callable[[jax.Array], jax.Array]] = None
    activation: Callable[[jax.Array], jax.Array] = nn.leaky_relu
    final_activation: Optional[Callable[[jax.Array], jax.Array]] = None

    @nn.compact
    def __call__(
        self,
        state: jax.Array,
        action: jax.Array,
        policy_var: flax.core.scope.VariableDict,
    ) -> jax.Array:
        assert policy_var.keys() == {'params'}
        assert policy_var['params'].keys() == {'decision_actor'}

        policy_param = Actor.make_params(policy_var)

        emb = MLP(
            layer_sizes=self.encoder_layer_sizes,
            activation=self.encoder_activation,
            final_activation=self.encoder_final_activation,
        )(policy_param)
        assert isinstance(emb, jax.Array)

        emb = emb.mean(axis=-2)

        assert len(emb.shape) == 1
        assert state.shape[:-1] == action.shape[:-1]
        assert len(action.shape) >= 1 and len(action.shape) <= 2
        if len(action.shape) == 2:
            emb = duplicate(emb, repeats=action.shape[0])

        value = MLP(
            layer_sizes=(*self.hidden_layer_sizes, self.n_values or 1),
            activation=self.activation,
            final_activation=self.final_activation,
        )(jnp.concatenate((state, action, emb), axis=-1))
        assert isinstance(value, jax.Array)
        if self.n_values is None:
            assert value.shape[-1] == 1
            value = jnp.squeeze(value, axis=-1)

        return value


class PolicyCritic(BaseCritic):

    encoder_layer_sizes: tuple[int, ...]
    hidden_layer_sizes: tuple[int, ...]
    n_values: Optional[int] = None
    encoder_activation: Callable[[jax.Array], jax.Array] = nn.leaky_relu
    encoder_final_activation: Optional[Callable[[jax.Array], jax.Array]] = None
    activation: Callable[[jax.Array], jax.Array] = nn.leaky_relu
    final_activation: Optional[Callable[[jax.Array], jax.Array]] = None

    def setup(self) -> None:
        params = (
            self.encoder_layer_sizes,
            self.hidden_layer_sizes,
            self.n_values,
            self.encoder_activation,
            self.encoder_final_activation,
            self.activation,
            self.final_activation,
        )
        self.critic1 = PolicyCriticSingle(*params)
        self.critic2 = PolicyCriticSingle(*params)

    def _critic1(
        self,
        state: jax.Array,
        action: jax.Array,
        policy_var: flax.core.scope.VariableDict
    ) -> jax.Array:
        return self.critic1(state, action, policy_var)

    def _critic2(
        self,
        state: jax.Array,
        action: jax.Array,
        policy_var: flax.core.scope.VariableDict
    ) -> jax.Array:
        return self.critic2(state, action, policy_var)

    def __call__(
        self,
        state: jax.Array,
        action: jax.Array,
        policy_var: flax.core.scope.VariableDict
    ) -> jax.Array:
        q1 = self._critic1(state, action, policy_var)
        q2 = self._critic2(state, action, policy_var)
        return jnp.concatenate((q1[..., jnp.newaxis], q2[..., jnp.newaxis]), axis=-1)

    def apply_critic1(
        self,
        vars: flax.core.scope.VariableDict,
        state: jax.Array,
        action: jax.Array,
        policy_var: flax.core.scope.VariableDict,
    ) -> jax.Array:
        critic1_vars = self.extract_critic1_vars(vars)
        q1 = self.apply(critic1_vars, state, action, policy_var, method=self._critic1)
        assert isinstance(q1, jax.Array)
        return q1
