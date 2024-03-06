import jax
import jax.numpy as jnp
import optax
import flax.linen as nn
import qdax.core.containers.repertoire
import qdax.core.emitters.multi_emitter
import qdax.core.emitters.pga_me_emitter
import qdax.core.emitters.qpg_emitter
import qdax.core.emitters.standard_emitters
from qdax.core.neuroevolution.buffers import buffer
import qdax.environments.base_wrappers
from qdax.types import Genotype, Params, RNGKey

from dataclasses import dataclass
import functools
from collections.abc import Callable
from typing import Any, TypeVar
from overrides import override

from .operators import OPTIMIZERS
from ..utils import jax_jit


@dataclass
class QualityPGEmitterConfig(qdax.core.emitters.qpg_emitter.QualityPGConfig):
    actor_optimizer: str = 'Adam'


_TQualityPGEmitterState = TypeVar(
    '_TQualityPGEmitterState', bound=qdax.core.emitters.qpg_emitter.QualityPGEmitterState
)


class QualityPGEmitter(qdax.core.emitters.qpg_emitter.QualityPGEmitter):

    def __init__(
        self,
        config: QualityPGEmitterConfig,
        policy_network: nn.Module,
        env: qdax.environments.base_wrappers.QDEnv,
    ) -> None:
        super().__init__(config, policy_network, env)
        self._config = config

        self._actor_optimizer = OPTIMIZERS[self._config.actor_optimizer](
            learning_rate=self._config.actor_learning_rate
        )
        self._policies_optimizer = OPTIMIZERS[self._config.actor_optimizer](
            learning_rate=self._config.policy_learning_rate
        )

    @functools.partial(
        jax_jit,
        static_argnames=("self",),
    )
    @override
    def emit(
        self,
        repertoire: qdax.core.containers.repertoire.Repertoire,
        emitter_state: qdax.core.emitters.qpg_emitter.QualityPGEmitterState,
        random_key: RNGKey,
    ) -> tuple[Genotype, RNGKey]:
        batch_size = self._config.env_batch_size

        # sample parents
        mutation_pg_batch_size = int(batch_size - 1)
        parents, random_key = repertoire.sample(random_key, mutation_pg_batch_size)

        # apply the pg mutation
        random_key, subkey = jax.random.split(random_key)
        offsprings_pg = self.emit_pg(emitter_state, parents, subkey)

        # get the actor (greedy actor)
        offspring_actor = self.emit_actor(emitter_state)

        # add dimension for concatenation
        offspring_actor = jax.tree_util.tree_map(
            lambda x: jnp.expand_dims(x, axis=0), offspring_actor
        )

        # gather offspring
        genotypes = jax.tree_util.tree_map(
            lambda x, y: jnp.concatenate([x, y], axis=0),
            offsprings_pg,
            offspring_actor,
        )

        return genotypes, random_key

    @functools.partial(
        jax_jit,
        static_argnames=("self",),
    )
    def emit_pg(
        self,
        emitter_state: qdax.core.emitters.qpg_emitter.QualityPGEmitterState,
        parents: Genotype,
        random_key: RNGKey,
    ) -> Genotype:
        keys = jax.random.split(random_key, self._config.env_batch_size - 1)
        offsprings = jax.vmap(
            self._mutation_function_pg,
            in_axes=(0, None, 0),
        )(parents, emitter_state, keys)

        return offsprings

    @functools.partial(jax_jit, static_argnames=("self",))
    def _train_critics(
        self, emitter_state: _TQualityPGEmitterState
    ) -> _TQualityPGEmitterState:
        # Sample a batch of transitions in the buffer
        random_key = emitter_state.random_key
        replay_buffer = emitter_state.replay_buffer
        transitions, random_key = replay_buffer.sample(
            random_key, sample_size=self._config.batch_size
        )
        transitions = self._transform_transitions(emitter_state, transitions)

        # Update Critic
        (
            critic_optimizer_state,
            critic_params,
            target_critic_params,
            random_key,
        ) = self._update_critic(
            critic_params=emitter_state.critic_params,
            target_critic_params=emitter_state.target_critic_params,
            target_actor_params=emitter_state.target_actor_params,
            critic_optimizer_state=emitter_state.critic_optimizer_state,
            transitions=transitions,
            random_key=random_key,
        )

        # Update greedy actor
        (actor_optimizer_state, actor_params, target_actor_params,) = jax.lax.cond(
            emitter_state.steps % self._config.policy_delay == 0,
            lambda x: self._update_actor(*x),
            lambda _: (
                emitter_state.actor_opt_state,
                emitter_state.actor_params,
                emitter_state.target_actor_params,
            ),
            operand=(
                emitter_state.actor_params,
                emitter_state.actor_opt_state,
                emitter_state.target_actor_params,
                emitter_state.critic_params,
                transitions,
            ),
        )

        # Create new training state
        new_emitter_state = emitter_state.replace(
            critic_params=critic_params,
            critic_optimizer_state=critic_optimizer_state,
            actor_params=actor_params,
            actor_opt_state=actor_optimizer_state,
            target_critic_params=target_critic_params,
            target_actor_params=target_actor_params,
            random_key=random_key,
            steps=emitter_state.steps + 1,
            replay_buffer=replay_buffer,
        )

        return new_emitter_state  # type: ignore

    @functools.partial(jax_jit, static_argnames=("self",))
    def _mutation_function_pg(
        self,
        policy_params: Genotype,
        emitter_state: qdax.core.emitters.qpg_emitter.QualityPGEmitterState,
        random_key: RNGKey,
    ) -> Genotype:
        # Define new policy optimizer state
        policy_optimizer_state = self._policies_optimizer.init(policy_params)

        def scan_train_policy(
            carry: tuple[
                qdax.core.emitters.qpg_emitter.QualityPGEmitterState, Genotype, optax.OptState
            ],
            random_key: RNGKey,
        ) -> tuple[
            tuple[qdax.core.emitters.qpg_emitter.QualityPGEmitterState, Genotype, optax.OptState],
            Any,
        ]:
            emitter_state, policy_params, policy_optimizer_state = carry
            (
                new_emitter_state,
                new_policy_params,
                new_policy_optimizer_state,
            ) = self._train_policy(
                emitter_state,
                policy_params,
                policy_optimizer_state,
                random_key,
            )
            return (
                new_emitter_state,
                new_policy_params,
                new_policy_optimizer_state,
            ), ()

        keys = jax.random.split(random_key, self._config.num_pg_training_steps)
        (emitter_state, policy_params, policy_optimizer_state,), _ = jax.lax.scan(
            scan_train_policy,
            (emitter_state, policy_params, policy_optimizer_state),
            keys,
            length=self._config.num_pg_training_steps,
        )

        return policy_params

    @functools.partial(jax_jit, static_argnames=("self",))
    def _train_policy(
        self,
        emitter_state: _TQualityPGEmitterState,
        policy_params: Params,
        policy_optimizer_state: optax.OptState,
        random_key: RNGKey,
    ) -> tuple[_TQualityPGEmitterState, Params, optax.OptState]:
        # Sample a batch of transitions in the buffer
        replay_buffer = emitter_state.replay_buffer
        transitions, _ = replay_buffer.sample(
            random_key, sample_size=self._config.batch_size
        )
        transitions = self._transform_transitions(emitter_state, transitions)

        # update policy
        policy_optimizer_state, policy_params = self._update_policy(
            critic_params=emitter_state.critic_params,
            policy_optimizer_state=policy_optimizer_state,
            policy_params=policy_params,
            transitions=transitions,
        )

        return emitter_state, policy_params, policy_optimizer_state

    def update_replay_buffer(
        self,
        emitter_state: _TQualityPGEmitterState,
        transitions: buffer.QDTransition,
    ) -> _TQualityPGEmitterState:
        replay_buffer = emitter_state.replay_buffer.insert(transitions)
        emitter_state = emitter_state.replace(replay_buffer=replay_buffer)
        return emitter_state

    @functools.partial(jax_jit, static_argnames=('self', 'batch_size'))
    def calc_policy_gradient(
        self,
        emitter_state: qdax.core.emitters.qpg_emitter.QualityPGEmitterState,
        policy_params: Params,
        random_key: RNGKey,
        batch_size: int,
    ) -> tuple[Params, RNGKey]:
        transitions, random_key = emitter_state.replay_buffer.sample(
            random_key, sample_size=batch_size,
        )
        transitions = self._transform_transitions(emitter_state, transitions)
        _policy_loss, policy_gradient = jax.value_and_grad(self._policy_loss_fn)(
            policy_params,
            emitter_state.critic_params,
            transitions,
        )
        return policy_gradient, random_key

    def _transform_transitions(
        self,
        emitter_state: qdax.core.emitters.qpg_emitter.QualityPGEmitterState,
        transitions: buffer.QDTransition,
    ) -> buffer.QDTransition:
        return transitions


@dataclass
class PGAMEEmitterConfig(qdax.core.emitters.pga_me_emitter.PGAMEConfig):
    pass


class PGAMEEmitter(qdax.core.emitters.multi_emitter.MultiEmitter):
    def __init__(
        self,
        config: PGAMEEmitterConfig,
        policy_network: nn.Module,
        env: qdax.environments.base_wrappers.QDEnv,
        variation_fn: Callable[[Params, Params, RNGKey], tuple[Params, RNGKey]],
    ):

        self._config = config
        self._policy_network = policy_network
        self._env = env
        self._variation_fn = variation_fn

        ga_batch_size = int(self._config.proportion_mutation_ga * config.env_batch_size)
        qpg_batch_size = config.env_batch_size - ga_batch_size

        qpg_config = QualityPGEmitterConfig(
            env_batch_size=qpg_batch_size,
            num_critic_training_steps=config.num_critic_training_steps,
            num_pg_training_steps=config.num_pg_training_steps,
            replay_buffer_size=config.replay_buffer_size,
            critic_hidden_layer_size=config.critic_hidden_layer_size,
            critic_learning_rate=config.critic_learning_rate,
            actor_learning_rate=config.greedy_learning_rate,
            policy_learning_rate=config.policy_learning_rate,
            noise_clip=config.noise_clip,
            policy_noise=config.policy_noise,
            discount=config.discount,
            reward_scaling=config.reward_scaling,
            batch_size=config.batch_size,
            soft_tau_update=config.soft_tau_update,
            policy_delay=config.policy_delay,
        )

        # define the quality emitter
        q_emitter = QualityPGEmitter(
            config=qpg_config, policy_network=policy_network, env=env
        )

        # define the GA emitter
        ga_emitter = qdax.core.emitters.standard_emitters.MixingEmitter(
            mutation_fn=lambda x, r: (x, r),
            variation_fn=variation_fn,
            variation_percentage=1.0,
            batch_size=ga_batch_size,
        )

        super().__init__(emitters=(q_emitter, ga_emitter))
