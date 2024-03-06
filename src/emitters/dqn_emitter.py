import jax
import jax.numpy as jnp
import optax
import flax.linen as nn
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
    from ..env import Env


@dataclass
class DQNEmitterConfig:
    env_batch_size: int = 20
    num_dqn_training_steps: int = 300
    num_mutation_steps: int = 100
    replay_buffer_size: int = 200000
    greedy_learning_rate: float = 3e-4
    learning_rate: float = 1e-3
    discount: float = 0.99
    reward_scaling: float = 1.0
    batch_size: int = 32
    target_policy_update_interval: int = 10
    using_greedy: bool = True


class DQNEmitterState(qdax.core.emitters.emitter.EmitterState):
    actor_params: flax.core.scope.VariableDict
    target_actor_params: flax.core.scope.VariableDict
    greedy_optimizer_state: optax.OptState
    replay_buffer: buffer.ReplayBuffer
    random_key: jax.random.KeyArray
    step: jax.Array


class DQNEmitter(qdax.core.emitters.emitter.Emitter):

    def __init__(self, config: DQNEmitterConfig, policy_network: nn.Module, env: 'Env'):
        self._config = config
        self._env = env
        self._policy_network = policy_network

        self._loss_fn = make_dqn_loss_fn(
            astype_wrap(policy_network.apply, jax.Array),
            reward_scaling=self._config.reward_scaling,
            discount=self._config.discount,
        )

        self._greedy_optimizer = optax.adam(learning_rate=self._config.greedy_learning_rate)
        self._optimizer = optax.adam(learning_rate=self._config.learning_rate)

    @property
    def batch_size(self) -> int:
        return self._config.env_batch_size

    @property
    def use_all_data(self) -> bool:
        """Whether to use all data or not when used along other emitters.

        DQNEmitter uses the transitions from the genotypes that were generated
        by other emitters.
        """
        return True

    def init(
        self, init_genotypes: flax.core.scope.VariableDict, random_key: jax.random.KeyArray
    ) -> tuple[DQNEmitterState, jax.random.KeyArray]:
        observation_size = self._env.obs_size
        action_size = self._env.action_size
        descriptor_size = self._env.state_descriptor_len

        actor_params = tree_getitem(init_genotypes, 0)
        target_actor_params = tree_copy(actor_params)

        greedy_optimizer_state = self._greedy_optimizer.init(actor_params)

        dummy_transition = buffer.QDTransition.init_dummy(
            observation_dim=observation_size,
            action_dim=action_size,
            descriptor_dim=descriptor_size,
        )

        replay_buffer = buffer.ReplayBuffer.init(
            buffer_size=self._config.replay_buffer_size, transition=dummy_transition
        )

        random_key, subkey = jax.random.split(random_key)
        emitter_state = DQNEmitterState(
            actor_params=actor_params,
            target_actor_params=target_actor_params,
            greedy_optimizer_state=greedy_optimizer_state,
            replay_buffer=replay_buffer,
            random_key=subkey,
            step=jnp.zeros((), dtype=jnp.int32),
        )

        return emitter_state, random_key

    @functools.partial(jax_jit, static_argnames=('self',))
    def emit(
        self,
        repertoire: qdax.core.containers.repertoire.Repertoire,
        emitter_state: DQNEmitterState,
        random_key: jax.random.KeyArray,
    ) -> tuple[flax.core.scope.VariableDict, jax.random.KeyArray]:
        batch_size = self._config.env_batch_size - self._config.using_greedy
        parents, random_key = astype(
            repertoire.sample(random_key, batch_size),
            tuple[flax.core.scope.VariableDict, jax.random.KeyArray],
        )
        random_key, subkey = jax.random.split(random_key)
        keys = jax.random.split(subkey, batch_size)
        offsprings = jax.vmap(
            self._mutation_function,
            in_axes=(0, None, 0),
        )(parents, emitter_state.replay_buffer, keys)

        if self._config.using_greedy:
            offsprings = tree_concatenate(
                offsprings, tree_getitem(emitter_state.actor_params, None)
            )

        return offsprings, random_key

    @functools.partial(jax_jit, static_argnames=('self',))
    def state_update(
        self,
        emitter_state: DQNEmitterState,
        repertoire: Optional[qdax.core.containers.repertoire.Repertoire],
        genotypes: Optional[flax.core.scope.VariableDict],
        fitnesses: Optional[Fitness],
        descriptors: Optional[Descriptor],
        extra_scores: ExtraScores,
    ) -> DQNEmitterState:
        assert "transitions" in extra_scores.keys(), "Missing transitions or wrong key"
        transitions = extra_scores["transitions"]
        assert isinstance(transitions, buffer.Transition)

        replay_buffer = emitter_state.replay_buffer.insert(transitions)
        emitter_state = emitter_state.replace(replay_buffer=replay_buffer)

        if self._config.using_greedy:
            def scan_train(
                emitter_state: DQNEmitterState, _: Any
            ) -> tuple[DQNEmitterState, None]:
                emitter_state = self._train(emitter_state)
                return emitter_state, None

            emitter_state, _ = jax.lax.scan(
                scan_train,
                emitter_state,
                None,
                length=self._config.num_dqn_training_steps,
            )

        return emitter_state

    @functools.partial(jax_jit, static_argnames=('self',))
    def _train(self, emitter_state: DQNEmitterState) -> DQNEmitterState:
        random_key = emitter_state.random_key
        replay_buffer = emitter_state.replay_buffer
        transitions, random_key = replay_buffer.sample(
            random_key, sample_size=self._config.batch_size
        )
        policy_params = emitter_state.actor_params
        target_policy_params = emitter_state.target_actor_params
        optimizer_state = emitter_state.greedy_optimizer_state
        step = emitter_state.step

        _loss, gradient = jax_value_and_grad(self._loss_fn)(
            policy_params,
            target_policy_params,
            transitions,
        )
        policy_updates, optimizer_state = self._greedy_optimizer.update(gradient, optimizer_state)

        policy_params_ = optax.apply_updates(policy_params, policy_updates)
        policy_params = astype(policy_params_, flax.core.scope.VariableDict)
        del policy_params_

        target_policy_params = jax.lax.cond(
            step % self._config.target_policy_update_interval == 0,
            lambda: policy_params,
            lambda: target_policy_params,
        )

        emitter_state = emitter_state.replace(
            random_key=random_key,
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

        _loss, gradient = jax_value_and_grad(self._loss_fn)(
            policy_params,
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
class DQNMEEmitterConfig(DQNEmitterConfig):
    proportion_mutation_ga: float = 0.5


class DQNMEEmitter(qdax.core.emitters.multi_emitter.MultiEmitter):
    def __init__(
        self,
        config: DQNMEEmitterConfig,
        policy_network: nn.Module,
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

        dqn_config = DQNEmitterConfig(
            env_batch_size=dqn_batch_size,
            num_dqn_training_steps=self._config.num_dqn_training_steps,
            num_mutation_steps=self._config.num_mutation_steps,
            replay_buffer_size=self._config.replay_buffer_size,
            greedy_learning_rate=self._config.greedy_learning_rate,
            learning_rate=self._config.learning_rate,
            discount=self._config.discount,
            reward_scaling=self._config.reward_scaling,
            batch_size=self._config.batch_size,
            target_policy_update_interval=self._config.target_policy_update_interval,
            using_greedy=self._config.using_greedy,
        )

        dqn_emitter = DQNEmitter(config=dqn_config, policy_network=policy_network, env=env)

        ga_emitter = qdax.core.emitters.standard_emitters.MixingEmitter(
            mutation_fn=lambda x, r: (x, r),
            variation_fn=variation_fn,
            variation_percentage=1.0,
            batch_size=ga_batch_size,
        )

        super().__init__(emitters=(dqn_emitter, ga_emitter))
