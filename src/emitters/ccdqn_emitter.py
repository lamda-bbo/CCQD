import jax
import jax.numpy as jnp
import optax
import flax.core.scope
import qdax.core.containers.repertoire
import qdax.core.emitters.emitter
import qdax.core.emitters.standard_emitters
import qdax.core.emitters.multi_emitter
from qdax.core.neuroevolution.buffers import buffer
from qdax.types import Params, Fitness, Descriptor, ExtraScores

from dataclasses import dataclass
import functools
from collections.abc import Callable
from typing import Optional, Any, TYPE_CHECKING

from ..neuroevolution import make_dqn_loss_fn
from ..utils import (
    astype, astype_wrap, tree_copy, tree_getitem, tree_concatenate, jax_jit, jax_value_and_grad
)

if TYPE_CHECKING:
    from ..map_elites import ExtendedMapElitesRepertoire
    from ..neuroevolution import Actor
    from ..env import Env


@dataclass
class CCDQNEmitterConfig:
    env_batch_size: int = 20
    num_dqn_training_steps: int = 300
    num_mutation_steps: int = 100
    replay_buffer_size: int = 200000

    representation_learning_rate: float = 3e-4
    greedy_learning_rate: float = 3e-4
    learning_rate: float = 1e-3

    discount: float = 0.99
    reward_scaling: float = 1.0
    batch_size: int = 32
    target_policy_update_interval: int = 10
    num_decision_updating_representation: int = 20
    decision_factor: float = 1.0
    using_greedy: bool = True

    n_representation: int = 5


class CCDQNEmitterState(qdax.core.emitters.emitter.EmitterState):
    representation_vars: flax.core.scope.VariableDict
    target_representation_vars: flax.core.scope.VariableDict
    representation_optimizer_state: optax.OptState
    actor_params: flax.core.scope.VariableDict
    target_actor_params: flax.core.scope.VariableDict
    greedy_optimizer_state: optax.OptState
    replay_buffer: buffer.ReplayBuffer
    random_key: jax.random.KeyArray
    step: jax.Array


class CCDQNEmitter(qdax.core.emitters.emitter.Emitter):

    def __init__(self, config: CCDQNEmitterConfig, policy_network: 'Actor', env: 'Env'):
        self._config = config
        self._env = env
        self._policy_network = policy_network

        def _loss_fn(
            representation_vars: flax.core.scope.VariableDict,
            decision_vars: flax.core.scope.VariableDict,
            target_representation_vars: flax.core.scope.VariableDict,
            target_decision_vars: flax.core.scope.VariableDict,
            transition: buffer.QDTransition,
            representation_indices: Optional[jax.Array] = None,
        ):
            fn = make_dqn_loss_fn(
                astype_wrap(policy_network.apply, jax.Array),
                reward_scaling=self._config.reward_scaling,
                discount=self._config.discount,
            )
            vars = policy_network.make_genotypes(policy_network.make_vars(
                representation_vars,
                decision_vars,
                representation_indices=representation_indices,
            ))
            target_vars = policy_network.make_genotypes(policy_network.make_vars(
                target_representation_vars,
                target_decision_vars,
                representation_indices=representation_indices,
            ))
            batch_sizes = policy_network.get_decision_param_batch_sizes(vars)
            if batch_sizes is None:
                return fn(vars, target_vars, transition)
            else:
                loss = jax.vmap(fn, in_axes=(0, 0, None))(vars, target_vars, transition)
                return jnp.sum(loss, axis=-1)

        self._loss_fn = _loss_fn

        self._representation_optimizer = optax.adam(
            learning_rate=self._config.representation_learning_rate
        )
        self._greedy_optimizer = optax.adam(learning_rate=self._config.greedy_learning_rate)
        self._optimizer = optax.adam(learning_rate=self._config.learning_rate)

    @property
    def batch_size(self) -> int:
        return self._config.env_batch_size

    @property
    def use_all_data(self) -> bool:
        """Whether to use all data or not when used along other emitters.

        CCDQNEmitter uses the transitions from the genotypes that were generated
        by other emitters.
        """
        return True

    def init(
        self, init_genotypes: flax.core.scope.VariableDict, random_key: jax.random.KeyArray
    ) -> tuple[CCDQNEmitterState, jax.random.KeyArray]:
        representation_vars, decision_vars = self._policy_network.extract_vars(init_genotypes)
        representation_batch_sizes = self._policy_network.get_param_batch_sizes(representation_vars)
        decision_batch_sizes = self._policy_network.get_param_batch_sizes(decision_vars)
        assert decision_batch_sizes is not None

        assert representation_batch_sizes is not None
        assert representation_batch_sizes == decision_batch_sizes
        init_representation_indices = self._policy_network.get_init_representation_indices(
            self._config.env_batch_size, self._config.n_representation
        )
        greedy_policy_network_indices: jax.Array = jnp.searchsorted(
            init_representation_indices, jnp.arange(self._config.n_representation)
        )
        greedy_policy_networks = tree_getitem(init_genotypes, greedy_policy_network_indices)
        representation_vars, greedy_decision_vars = self._policy_network.extract_vars(
            greedy_policy_networks
        )
        assert (
            self._policy_network.get_param_batch_sizes(representation_vars)
            == self._config.n_representation
        )
        assert (
            self._policy_network.get_param_batch_sizes(greedy_decision_vars)
            == self._config.n_representation
        )

        observation_size = self._env.obs_size
        action_size = self._env.action_size
        descriptor_size = self._env.state_descriptor_len

        target_representation_vars = tree_copy(representation_vars)
        representation_optimizer_state = self._representation_optimizer.init(representation_vars)

        target_greedy_decision_vars = tree_copy(greedy_decision_vars)
        greedy_decision_optimizer_state = self._greedy_optimizer.init(
            greedy_decision_vars
        )

        dummy_transition = buffer.QDTransition.init_dummy(
            observation_dim=observation_size,
            action_dim=action_size,
            descriptor_dim=descriptor_size,
        )

        replay_buffer = buffer.ReplayBuffer.init(
            buffer_size=self._config.replay_buffer_size, transition=dummy_transition
        )

        random_key, subkey = jax.random.split(random_key)
        emitter_state = CCDQNEmitterState(
            representation_vars=representation_vars,
            target_representation_vars=target_representation_vars,
            representation_optimizer_state=representation_optimizer_state,
            actor_params=greedy_decision_vars,
            target_actor_params=target_greedy_decision_vars,
            greedy_optimizer_state=greedy_decision_optimizer_state,
            replay_buffer=replay_buffer,
            random_key=subkey,
            step=jnp.zeros((), dtype=jnp.int32),
        )

        return emitter_state, random_key

    @functools.partial(jax_jit, static_argnames=('self',))
    def emit(
        self,
        repertoire: qdax.core.containers.repertoire.Repertoire,
        emitter_state: CCDQNEmitterState,
        random_key: jax.random.KeyArray,
    ) -> tuple[flax.core.scope.VariableDict, jax.random.KeyArray]:
        batch_size = (
            self._config.env_batch_size
            - self._config.using_greedy * self._config.n_representation
        )
        parents, random_key = astype(
            repertoire.sample(random_key, batch_size),
            tuple[flax.core.scope.VariableDict, jax.random.KeyArray],
        )
        _, parents = self._policy_network.extract_vars(parents)
        random_key, subkey = jax.random.split(random_key)
        indices = jax.random.randint(
            subkey,
            (batch_size,),
            minval=0,
            maxval=self._config.n_representation,
        )
        representation = tree_getitem(emitter_state.representation_vars, indices)
        random_key, subkey = jax.random.split(random_key)
        keys = jax.random.split(subkey, batch_size)
        offsprings = jax.vmap(
            self._mutation_function,
            in_axes=(0, 0, None, 0),
        )(parents, representation, emitter_state.replay_buffer, keys)
        offsprings = self._policy_network.make_genotypes(
            self._policy_network.make_vars(representation, offsprings)
        )

        if self._config.using_greedy:
            actor_params = self._policy_network.make_vars(
                emitter_state.representation_vars, emitter_state.actor_params
            )
            offsprings = tree_concatenate(
                offsprings, actor_params
            )

        return offsprings, random_key

    @functools.partial(jax_jit, static_argnames=('self',))
    def state_update(
        self,
        emitter_state: CCDQNEmitterState,
        repertoire: 'ExtendedMapElitesRepertoire',
        genotypes: Optional[flax.core.scope.VariableDict],
        fitnesses: Optional[Fitness],
        descriptors: Optional[Descriptor],
        extra_scores: ExtraScores,
    ) -> CCDQNEmitterState:
        assert "transitions" in extra_scores.keys(), "Missing transitions or wrong key"
        transitions = extra_scores["transitions"]
        assert isinstance(transitions, buffer.Transition)

        replay_buffer = emitter_state.replay_buffer.insert(transitions)
        emitter_state = emitter_state.replace(replay_buffer=replay_buffer)

        if self._config.using_greedy:
            def scan_train(
                emitter_state: CCDQNEmitterState, _: Any
            ) -> tuple[CCDQNEmitterState, None]:
                emitter_state = self._train(emitter_state, repertoire)
                return emitter_state, None

            emitter_state, _ = jax.lax.scan(
                scan_train,
                emitter_state,
                None,
                length=self._config.num_dqn_training_steps,
            )

        return emitter_state

    @functools.partial(jax_jit, static_argnames=('self',))
    def _train(
        self, emitter_state: CCDQNEmitterState, repertoire: 'ExtendedMapElitesRepertoire'
    ) -> CCDQNEmitterState:
        random_key = emitter_state.random_key
        replay_buffer = emitter_state.replay_buffer
        transitions, random_key = replay_buffer.sample(
            random_key, sample_size=self._config.batch_size
        )
        representation_vars = emitter_state.representation_vars
        target_representation_vars = emitter_state.target_representation_vars
        representation_optimizer_state = emitter_state.representation_optimizer_state
        policy_params = emitter_state.actor_params
        target_policy_params = emitter_state.target_actor_params
        optimizer_state = emitter_state.greedy_optimizer_state
        step = emitter_state.step

        decision_vars, random_key = repertoire.sample(
            random_key, self._config.num_decision_updating_representation
        )
        random_key, subkey = jax.random.split(random_key)
        representation_indices = jax.random.randint(
            subkey,
            (self._config.num_decision_updating_representation,),
            minval=0,
            maxval=self._config.n_representation,
            dtype=jnp.int32,
        )

        (
            _greedy_loss, (greedy_representation_gradient, greedy_decision_gradient)
        ) = jax_value_and_grad(self._loss_fn, argnums=(0, 1))(
            representation_vars,
            policy_params,
            target_representation_vars,
            target_policy_params,
            transitions,
        )
        policy_updates, optimizer_state = self._greedy_optimizer.update(
            greedy_decision_gradient, optimizer_state
        )

        policy_params_ = optax.apply_updates(policy_params, policy_updates)
        policy_params = astype(policy_params_, flax.core.scope.VariableDict)
        del policy_params_

        (
            _representation_loss, representation_gradient
        ) = jax_value_and_grad(self._loss_fn)(
            representation_vars,
            decision_vars,
            target_representation_vars,
            decision_vars,
            transitions,
            representation_indices,
        )
        representation_gradient = jax.tree_map(
            lambda x1, x2: self._config.decision_factor * x1 + x2,
            representation_gradient,
            greedy_representation_gradient,
        )

        (
            representation_updates,
            representation_optimizer_state,
        ) = self._representation_optimizer.update(
            representation_gradient, representation_optimizer_state
        )
        representation_vars_ = optax.apply_updates(representation_vars, representation_updates)
        representation_vars = astype(representation_vars_, flax.core.scope.VariableDict)
        del representation_vars_

        (target_representation_vars, target_policy_params) = jax.lax.cond(
            step % self._config.target_policy_update_interval == 0,
            lambda: (representation_vars, policy_params),
            lambda: (target_representation_vars, target_policy_params),
        )

        emitter_state = emitter_state.replace(
            random_key=random_key,
            representation_vars=representation_vars,
            target_representation_vars=target_representation_vars,
            representation_optimizer_state=representation_optimizer_state,
            actor_params=policy_params,
            target_actor_params=target_policy_params,
            greedy_optimizer_state=optimizer_state,
            step=step + 1,
        )

        return emitter_state

    @functools.partial(jax_jit, static_argnames=('self',))
    def _mutation_function(
        self,
        policy_params: flax.core.scope.VariableDict,
        representation_vars: flax.core.scope.VariableDict,
        replay_buffer: buffer.ReplayBuffer,
        random_key: jax.random.KeyArray,
    ) -> flax.core.scope.VariableDict:
        target_policy_params = tree_copy(policy_params)
        optimizer_state = self._optimizer.init(policy_params)

        def scan_train_policy(
            carry: tuple[
                buffer.ReplayBuffer,
                flax.core.scope.VariableDict,
                flax.core.scope.VariableDict,
                optax.OptState,
            ],
            x: tuple[jax.random.KeyArray, jax.Array],
        ) -> tuple[
            tuple[
                buffer.ReplayBuffer,
                flax.core.scope.VariableDict,
                flax.core.scope.VariableDict,
                optax.OptState,
            ],
            None,
        ]:
            replay_buffer, policy_params, target_policy_params, optimizer_state = carry
            random_key, update_target_policy = x
            (
                policy_params, target_policy_params, optimizer_state
            ) = self._train_policy(
                replay_buffer,
                policy_params,
                target_policy_params,
                optimizer_state,
                representation_vars,
                random_key,
                update_target_policy,
            )
            return (
                replay_buffer, policy_params, target_policy_params, optimizer_state
            ), None

        keys = jax.random.split(random_key, self._config.num_mutation_steps)
        (replay_buffer, policy_params, target_policy_params, optimizer_state,), _ = jax.lax.scan(
            scan_train_policy,
            (replay_buffer, policy_params, target_policy_params, optimizer_state,),
            (
                keys,
                jnp.arange(
                    1, self._config.num_mutation_steps + 1
                ) % self._config.target_policy_update_interval == 0,
            ),
            length=self._config.num_mutation_steps,
        )

        return policy_params

    @functools.partial(jax_jit, static_argnames=('self',))
    def _train_policy(
        self,
        replay_buffer: buffer.ReplayBuffer,
        policy_params: flax.core.scope.VariableDict,
        target_policy_params: flax.core.scope.VariableDict,
        optimizer_state: optax.OptState,
        representation_vars: flax.core.scope.VariableDict,
        random_key: jax.random.KeyArray,
        update_target_policy: jax.Array,
    ) -> tuple[
        flax.core.scope.VariableDict,
        flax.core.scope.VariableDict,
        optax.OptState,
    ]:
        transitions, _ = replay_buffer.sample(
            random_key, sample_size=self._config.batch_size
        )

        _loss, gradient = jax_value_and_grad(self._loss_fn, argnums=1)(
            representation_vars,
            policy_params,
            representation_vars,
            target_policy_params,
            transitions,
        )
        policy_updates, optimizer_state = self._optimizer.update(gradient, optimizer_state)

        policy_params_ = optax.apply_updates(policy_params, policy_updates)
        policy_params = astype(policy_params_, flax.core.scope.VariableDict)
        del policy_params_

        target_policy_params = jax.lax.cond(
            update_target_policy,
            lambda: policy_params,
            lambda: target_policy_params,
        )

        return policy_params, target_policy_params, optimizer_state


@dataclass
class CCDQNMEEmitterConfig(CCDQNEmitterConfig):
    proportion_mutation_ga: float = 0.5


class CCDQNMEEmitter(qdax.core.emitters.multi_emitter.MultiEmitter):
    def __init__(
        self,
        config: CCDQNMEEmitterConfig,
        policy_network: 'Actor',
        env: 'Env',
        variation_fn: Callable[
            [Params, Params, jax.random.KeyArray], tuple[Params, jax.random.KeyArray]
        ],
    ):
        self._config = config
        self._policy_network = policy_network
        self._env = env
        self._variation_fn = variation_fn

        ga_batch_size = int(self._config.proportion_mutation_ga * config.env_batch_size)
        dqn_batch_size = config.env_batch_size - ga_batch_size

        dqn_config = CCDQNEmitterConfig(
            env_batch_size=dqn_batch_size,
            num_dqn_training_steps=self._config.num_dqn_training_steps,
            num_mutation_steps=self._config.num_mutation_steps,
            replay_buffer_size=self._config.replay_buffer_size,
            representation_learning_rate=self._config.representation_learning_rate,
            greedy_learning_rate=self._config.greedy_learning_rate,
            learning_rate=self._config.learning_rate,
            discount=self._config.discount,
            reward_scaling=self._config.reward_scaling,
            batch_size=self._config.batch_size,
            target_policy_update_interval=self._config.target_policy_update_interval,
            num_decision_updating_representation=self._config.num_decision_updating_representation,
            decision_factor=self._config.decision_factor,
            using_greedy=self._config.using_greedy,
            n_representation=self._config.n_representation,
        )

        dqn_emitter = CCDQNEmitter(config=dqn_config, policy_network=policy_network, env=env)

        ga_emitter = qdax.core.emitters.standard_emitters.MixingEmitter(
            mutation_fn=lambda x, r: (x, r),
            variation_fn=variation_fn,
            variation_percentage=1.0,
            batch_size=ga_batch_size,
        )

        super().__init__(emitters=(dqn_emitter, ga_emitter))
