import jax
import jax.numpy as jnp
import flax.core
import flax.linen as nn
import brax.jumpy as jp

import logging
import functools
import operator
from collections.abc import Sequence, Callable
from typing import Literal, Optional, Any

from .mlp import MLP
from .cnn import CNN
from ..utils import astype, tree_getitem, tree_repeat, tree_duplicate


_log = logging.getLogger(__name__)


class Actor(nn.Module):

    representation_hidden_layer_sizes: tuple[int, ...]
    decision_hidden_layer_sizes: tuple[int, ...]
    action_size: int
    activation: Callable[[jax.Array], jax.Array] = nn.relu
    final_activation: Optional[Callable[[jax.Array], jax.Array]] = None

    def setup(self) -> None:
        self.representation_actor = MLP(
            layer_sizes=self.representation_hidden_layer_sizes,
            activation=self.activation,
            final_activation=self.activation,
        )
        self.decision_actor = MLP(
            layer_sizes=(*self.decision_hidden_layer_sizes, self.action_size,),
            activation=self.activation,
            final_activation=self.final_activation,
        )

    def _representation_actor(self, data: jax.Array) -> jax.Array:
        return self.representation_actor(data)

    def _decision_actor(self, data: jax.Array) -> jax.Array:
        action = self.decision_actor(data)
        assert isinstance(action, jax.Array)
        return action

    @staticmethod
    def get_init_representation_indices(env_batch_size: int, n_representation: int) -> jax.Array:
        return jnp.repeat(
            jnp.arange(n_representation, dtype=jnp.int32),
            repeats=round(env_batch_size / n_representation),
            total_repeat_length=env_batch_size,
        )

    @staticmethod
    def get_param_batch_shape(params: flax.core.scope.Collection) -> tuple[int, ...]:
        if params.keys() == {'params'}:
            assert len(params['params']) == 1
            params, = params['params'].values()
        shape = params['Dense_0']['kernel'].shape[:-2]
        return shape

    @staticmethod
    def get_param_batch_sizes(params: flax.core.scope.Collection) -> Optional[int]:
        if params.keys() == {'params'}:
            assert len(params['params']) == 1
            params, = params['params'].values()
        shape = params['Dense_0']['kernel'].shape
        match len(shape):
            case 2:
                return None
            case 3:
                return shape[0]
            case _:
                raise ValueError

    @staticmethod
    def get_decision_param_batch_sizes(params: flax.core.scope.Collection) -> Optional[int]:
        shape = params['params']['decision_actor']['Dense_0']['kernel'].shape
        match len(shape):
            case 2:
                return None
            case 3:
                return shape[0]
            case _:
                raise ValueError

    @staticmethod
    def get_data_batch_sizes(data: jax.Array | jp.ndarray | Sequence[jax.Array]) -> Optional[int]:
        if isinstance(data, jax.Array):
            shape = data.shape
        else:
            shape = data[0].shape
        match len(shape):
            case 1:
                return None
            case 2:
                return shape[0]
            case _:
                raise ValueError

    def __call__(self, data: jax.Array) -> jax.Array:
        hidden = self._representation_actor(data)
        data = self._decision_actor(hidden)
        return data

    def init_separately(
        self,
        random_key: jax.random.KeyArray,
        obs_size: int,
        env_batch_size: int,
    ) -> tuple[flax.core.scope.FrozenVariableDict, flax.core.scope.FrozenVariableDict]:
        keys = jax.random.split(random_key, num=env_batch_size)
        fake_batch = jnp.zeros(shape=(env_batch_size, obs_size))
        init_vars = jax.vmap(self.init)(keys, fake_batch)
        return self.extract_vars(init_vars)

    def init_all(
        self,
        random_key: jax.random.KeyArray,
        obs_size: int,
        env_batch_size: int,
        n_representation: int,
    ) -> tuple[flax.core.scope.FrozenVariableDict, flax.core.scope.FrozenVariableDict]:
        subkey_representation, subkey_decision = jax.random.split(random_key)
        subkeys_decision = jax.random.split(subkey_decision, num=env_batch_size)

        subkeys_representation = jax.random.split(subkey_representation, num=n_representation)
        fake_state = jnp.zeros(shape=(n_representation, obs_size))
        init_representation_actor_fn = functools.partial(
            self.init_with_output, method=self._representation_actor
        )
        fake_hidden, init_representation_vars = jax.vmap(init_representation_actor_fn)(
            subkeys_representation, fake_state
        )
        fake_hidden_batch, init_representation_vars = tree_repeat(
            (fake_hidden, init_representation_vars),
            repeats=round(env_batch_size / n_representation),
            total_repeat_length=env_batch_size,
        )

        init_decision_actor_fn = functools.partial(self.init, method=self._decision_actor)
        init_decision_vars = jax.vmap(init_decision_actor_fn)(
            subkeys_decision, fake_hidden_batch
        )

        return init_representation_vars, init_decision_vars

    def apply_all(
        self,
        vars: flax.core.scope.VariableDict,
        data: jax.Array | jp.ndarray,
    ) -> jax.Array:
        assert vars.keys() == {'params'}
        assert vars['params'].keys() == {'representation_actor', 'decision_actor'}

        representation_vars, decision_vars = self.extract_vars(vars)

        data_batch_size = self.get_data_batch_sizes(data)
        representation_batch_size = self.get_param_batch_sizes(
            vars['params']['representation_actor']
        )

        if representation_batch_size is None:
            hidden_ = self.apply(representation_vars, data, method=self._representation_actor)
        elif data_batch_size == representation_batch_size:
            apply_representation_actor_fn = functools.partial(
                self.apply, method=self._representation_actor
            )
            hidden_ = jax.vmap(apply_representation_actor_fn)(representation_vars, data)
        elif data_batch_size is None:
            apply_representation_actor_fn = functools.partial(
                self.apply, method=self._representation_actor
            )
            hidden_ = jax.vmap(apply_representation_actor_fn, in_axes=(0, None))(
                representation_vars, data
            )
        else:
            raise ValueError
        hidden = astype(hidden_, jax.Array)
        del hidden_

        hidden_batch_size = self.get_data_batch_sizes(hidden)
        decision_batch_size = self.get_param_batch_sizes(vars['params']['decision_actor'])

        if decision_batch_size is None:
            action = self.apply(decision_vars, hidden, method=self._decision_actor)
        else:
            if hidden_batch_size is None:
                hidden = tree_duplicate(hidden, repeats=decision_batch_size)
            elif hidden_batch_size != decision_batch_size:
                raise ValueError
            apply_decision_actor_fn = functools.partial(self.apply, method=self._decision_actor)
            action = jax.vmap(apply_decision_actor_fn)(decision_vars, hidden)
        assert isinstance(action, jax.Array)

        return action

    def apply_partial(
        self,
        main_vars: flax.core.scope.VariableDict,
        data: jax.Array | jp.ndarray,
        *other_vars: flax.core.scope.VariableDict,
    ) -> jax.Array:
        return self.apply_all(self.make_vars(main_vars, *other_vars), data)

    @staticmethod
    def extract_vars(
        vars: flax.core.scope.VariableDict,
    ) -> tuple[flax.core.scope.FrozenVariableDict, flax.core.scope.FrozenVariableDict]:
        if isinstance(vars, flax.core.FrozenDict):
            vars = vars.unfreeze()

        assert vars.keys() == {'params'}
        assert vars['params'].keys() == {'representation_actor', 'decision_actor'}

        representation_vars = flax.core.freeze({
            'params': {'representation_actor': vars['params']['representation_actor']}}
        )
        decision_vars = flax.core.freeze(
            {'params': {'decision_actor': vars['params']['decision_actor']}}
        )

        return representation_vars, decision_vars

    @classmethod
    def make_vars(
        cls,
        *vars_tuple: flax.core.scope.VariableDict,
        representation_indices: Optional[jax.Array] = None,
    ) -> flax.core.scope.FrozenVariableDict:
        for vars in vars_tuple:
            assert isinstance(vars, flax.core.FrozenDict) or isinstance(vars, dict)
            if isinstance(vars, flax.core.FrozenDict):
                vars = vars.unfreeze()
            assert vars.keys() == {'params'}

        vars_ = {'params': functools.reduce(
            operator.ior,
            map(lambda vars: vars['params'], vars_tuple),
            {}
        )}

        if representation_indices is not None:
            vars_['params']['representation_actor'] = tree_getitem(
                vars_['params']['representation_actor'], representation_indices
            )
            assert (
                cls.get_param_batch_sizes(vars_['params']['representation_actor'])
                == cls.get_param_batch_sizes(vars_['params']['decision_actor'])
            )

        return flax.core.freeze(vars_)

    @classmethod
    def make_genotypes(
        cls,
        vars: flax.core.scope.VariableDict,
    ) -> flax.core.scope.FrozenVariableDict:
        if not isinstance(vars, flax.core.FrozenDict):
            vars = flax.core.freeze(vars)

        assert vars.keys() == {'params'}
        assert vars['params'].keys() == {'representation_actor', 'decision_actor'}

        representation_batch_size = cls.get_param_batch_sizes(
            vars['params']['representation_actor']
        )
        decision_batch_size = cls.get_param_batch_sizes(vars['params']['decision_actor'])

        if decision_batch_size is not None:
            if representation_batch_size is None:
                vars_ = astype(vars.unfreeze(), dict[str, dict[str, Any]])
                vars_['params']['representation_actor'] = tree_duplicate(
                    vars['params']['representation_actor'],
                    repeats=decision_batch_size,
                )
                vars = flax.core.freeze(vars_)
        else:
            assert representation_batch_size is None

        return vars

    @staticmethod
    def make_params(vars: flax.core.scope.VariableDict) -> jax.Array:
        n_layers = len(vars['params']['decision_actor'])
        policy_param = jnp.concatenate((
            vars['params']['decision_actor'][f'Dense_{n_layers - 1}']['kernel'].swapaxes(-1, -2),
            jnp.expand_dims(
                vars['params']['decision_actor'][f'Dense_{n_layers - 1}']['bias'], axis=-1
            ),
        ), axis=-1)
        return policy_param


class CNNActor(Actor):

    representation_conv_features: Sequence[int] = ()
    decision_conv_features: Sequence[int] = ()

    representation_kernel_sizes: Sequence[Sequence[int]] = ()
    decision_kernel_sizes: Sequence[Sequence[int]] = ()

    representation_cnn_input_shape: Sequence[int] = (-1,)
    decision_cnn_input_shape: Sequence[int] = (-1,)

    conv_activation: Callable[[jax.Array], jax.Array] = nn.relu
    representation_conv_strides: Optional[Sequence[int | Sequence[int]]] = None
    decision_conv_strides: Optional[Sequence[int | Sequence[int]]] = None
    conv_padding: Literal['SAME', 'VALID'] = 'VALID'

    def setup(self) -> None:
        self.representation_actor = CNN(
            conv_features=self.representation_conv_features,
            conv_kernel_sizes=self.representation_kernel_sizes,
            conv_activation=self.conv_activation,
            conv_strides=self.representation_conv_strides,
            conv_padding=self.conv_padding,
            mlp_layer_sizes=self.representation_hidden_layer_sizes,
            mlp_activation=self.activation,
            mlp_final_activation=self.activation,
            cnn_input_shape=self.representation_cnn_input_shape,
        )
        self.decision_actor = CNN(
            conv_features=self.decision_conv_features,
            conv_kernel_sizes=self.decision_kernel_sizes,
            conv_activation=self.conv_activation,
            conv_strides=self.decision_conv_strides,
            conv_padding=self.conv_padding,
            mlp_layer_sizes=(*self.decision_hidden_layer_sizes, self.action_size,),
            mlp_activation=self.activation,
            mlp_final_activation=self.final_activation,
            cnn_input_shape=self.decision_cnn_input_shape,
        )

    @staticmethod
    def get_param_batch_shape(params: flax.core.scope.Collection) -> tuple[int, ...]:
        if params.keys() == {'params'}:
            assert len(params['params']) == 1
            params, = params['params'].values()
        shape = params['MLP_0']['Dense_0']['kernel'].shape[:-2]
        return shape

    @staticmethod
    def get_param_batch_sizes(params: flax.core.scope.Collection) -> Optional[int]:
        if params.keys() == {'params'}:
            assert len(params['params']) == 1
            params, = params['params'].values()
        shape = params['MLP_0']['Dense_0']['kernel'].shape
        match len(shape):
            case 2:
                return None
            case 3:
                return shape[0]
            case _:
                raise ValueError

    @staticmethod
    def get_decision_param_batch_sizes(params: flax.core.scope.Collection) -> Optional[int]:
        shape = params['params']['decision_actor']['MLP_0']['Dense_0']['kernel'].shape
        match len(shape):
            case 2:
                return None
            case 3:
                return shape[0]
            case _:
                raise ValueError

    @staticmethod
    def make_params(vars: flax.core.scope.VariableDict) -> jax.Array:
        n_layers = len(vars['params']['decision_actor'])
        _log.debug('n_layers = %s', n_layers)
        policy_param = jnp.concatenate((
            vars['params']['decision_actor']['MLP_0'][f'Dense_{n_layers - 1}']['kernel'].swapaxes(
                -1, -2
            ),
            jnp.expand_dims(
                vars['params']['decision_actor']['MLP_0'][f'Dense_{n_layers - 1}']['bias'], axis=-1
            ),
        ), axis=-1)
        _log.debug('policy_param.shape = %s', policy_param.shape)
        return policy_param
