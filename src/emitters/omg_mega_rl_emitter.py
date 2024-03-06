import jax
import jax.numpy as jnp
import optax
import flax.core
import flax.linen as nn
import qdax.core.emitters.qpg_emitter
import qdax.core.emitters.multi_emitter
import qdax.core.emitters.standard_emitters
import qdax.environments.base_wrappers
from qdax.core.neuroevolution.buffers import buffer
from qdax.types import Descriptor, ExtraScores, Fitness, Genotype, Params, RNGKey

from dataclasses import dataclass
import functools
from collections.abc import Callable
from typing import Optional, Any, TYPE_CHECKING

from .pga_me_emitter import QualityPGEmitterConfig, QualityPGEmitter, PGAMEEmitterConfig
from ..neuroevolution import CriticSingle, make_td3_loss_fn, ExtendedQDTransition
from ..utils import (
    astype, jax_jit, jax_jacrev, jax_value_and_grad,
    filter_wrap, astype_wrap, tree_copy, tree_getitem, tree_concatenate
)

if TYPE_CHECKING:
    from ..map_elites import ExtendedMapElitesRepertoire


@dataclass
class _OMGMEGARLEmitterConfig(QualityPGEmitterConfig):
    desc_critic_hidden_layer_size: tuple[int, ...] = (256, 256)

    desc_critic_learning_rate: float = 3e-4

    desc_reward_scaling: float = 1.0
    desc_discount: float = 1.0


class _OMGMEGARLEmitterState(qdax.core.emitters.qpg_emitter.QualityPGEmitterState):
    critic_params: tuple[Params, flax.core.scope.VariableDict]
    critic_optimizer_state: tuple[optax.OptState, optax.OptState]
    actor_params: flax.core.scope.VariableDict
    target_critic_params: tuple[Params, flax.core.scope.VariableDict]
    target_actor_params: flax.core.scope.VariableDict


class _OMGMEGARLEmitter(QualityPGEmitter):
    def __init__(
        self,
        config: _OMGMEGARLEmitterConfig,
        policy_network: nn.Module,
        env: qdax.environments.base_wrappers.QDEnv,
    ):
        super().__init__(config, policy_network, env)
        self._config = config

        self._desc_critic_network = CriticSingle(
            self._config.desc_critic_hidden_layer_size,
            n_values=self._env.behavior_descriptor_length,
            activation=nn.relu,
        )

        self._policy_desc_loss_fn, self._desc_critic_loss_fn = make_td3_loss_fn(
            policy_fn=astype_wrap(policy_network.apply, jax.Array),
            critic_fn=filter_wrap(
                astype_wrap(self._desc_critic_network.apply, jax.Array),
                functools.partial(jnp.expand_dims, axis=-1),
            ),
            reward_scaling=self._config.desc_reward_scaling,
            discount=self._config.desc_discount,
            noise_clip=self._config.noise_clip,
            policy_noise=self._config.policy_noise,
        )

        self._desc_critic_optimizer = optax.adam(
            learning_rate=self._config.desc_critic_learning_rate
        )

    def init(
        self, init_genotypes: Genotype, random_key: RNGKey
    ) -> tuple[_OMGMEGARLEmitterState, RNGKey]:
        base_emitter_state, random_key = super().init(init_genotypes, random_key)

        random_key, subkey = jax.random.split(random_key)
        fake_obs = jnp.zeros(shape=(self._env.observation_size,))
        action = jnp.zeros(shape=(self._env.action_size,))
        desc_critic_params = self._desc_critic_network.init(subkey, fake_obs, action)
        target_desc_critic_params = tree_copy(desc_critic_params)
        desc_critic_optimizer_state = self._desc_critic_optimizer.init(desc_critic_params)

        dummy_transition = ExtendedQDTransition.init_dummy(
            observation_dim=self._env.observation_size,
            action_dim=self._env.action_size,
            descriptor_dim=self._env.behavior_descriptor_length,
        )
        replay_buffer = buffer.ReplayBuffer.init(
            buffer_size=self._config.replay_buffer_size,
            transition=dummy_transition,
        )

        emitter_state = _OMGMEGARLEmitterState(
            critic_params=(
                base_emitter_state.critic_params, desc_critic_params
            ),
            critic_optimizer_state=(
                base_emitter_state.critic_optimizer_state, desc_critic_optimizer_state
            ),
            actor_params=astype(
                base_emitter_state.actor_params, flax.core.scope.VariableDict
            ),
            actor_opt_state=base_emitter_state.actor_opt_state,
            target_critic_params=(
                base_emitter_state.target_critic_params, target_desc_critic_params
            ),
            target_actor_params=astype(
                base_emitter_state.target_actor_params, flax.core.scope.VariableDict
            ),
            replay_buffer=replay_buffer,
            random_key=base_emitter_state.random_key,
            steps=base_emitter_state.steps,
        )

        return emitter_state, random_key

    @functools.partial(
        jax_jit,
        static_argnames=('self',),
    )
    def state_update(
        self,
        emitter_state: _OMGMEGARLEmitterState,
        repertoire: 'ExtendedMapElitesRepertoire',
        genotypes: Optional[flax.core.scope.VariableDict],
        fitnesses: Optional[Fitness],
        descriptors: Optional[Descriptor],
        extra_scores: ExtraScores,
    ) -> _OMGMEGARLEmitterState:
        assert 'transitions' in extra_scores.keys()
        assert isinstance(extra_scores['transitions'], ExtendedQDTransition)

        emitter_state = super().state_update(
            emitter_state,
            repertoire,
            genotypes,
            fitnesses,
            descriptors,
            extra_scores,
        )
        return emitter_state

    @functools.partial(jax_jit, static_argnames=("self",))
    def _train_critics(
        self, emitter_state: _OMGMEGARLEmitterState
    ) -> _OMGMEGARLEmitterState:
        # Sample a batch of transitions in the buffer
        random_key = emitter_state.random_key
        replay_buffer = emitter_state.replay_buffer
        transitions, random_key = replay_buffer.sample(
            random_key, sample_size=self._config.batch_size
        )

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
        (actor_optimizer_state, actor_params, target_actor_params, random_key) = jax.lax.cond(
            emitter_state.steps % self._config.policy_delay == 0,
            lambda x: self._update_actor(*x),
            lambda _: (
                emitter_state.actor_opt_state,
                emitter_state.actor_params,
                emitter_state.target_actor_params,
                random_key,
            ),
            operand=(
                emitter_state,
                emitter_state.actor_params,
                emitter_state.actor_opt_state,
                emitter_state.target_actor_params,
                emitter_state.critic_params,
                transitions,
                random_key,
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

        return new_emitter_state

    @functools.partial(jax_jit, static_argnames=("self",))
    def _update_critic(
        self,
        critic_params: tuple[Params, flax.core.scope.VariableDict],
        target_critic_params: tuple[Params, flax.core.scope.VariableDict],
        target_actor_params: flax.core.scope.VariableDict,
        critic_optimizer_state: tuple[optax.OptState, optax.OptState],
        transitions: ExtendedQDTransition,
        random_key: RNGKey,
    ) -> tuple[
        tuple[optax.OptState, optax.OptState],
        tuple[Params, flax.core.scope.VariableDict],
        tuple[Params, flax.core.scope.VariableDict],
        RNGKey,
    ]:

        # compute loss and gradients
        random_key, subkey = jax.random.split(random_key)
        critic_loss_f, critic_gradient_f = jax_value_and_grad(self._critic_loss_fn)(
            critic_params[0],
            target_actor_params,
            target_critic_params[0],
            transitions,
            subkey,
        )
        critic_updates_f, critic_optimizer_state_f = self._critic_optimizer.update(
            critic_gradient_f, critic_optimizer_state[0]
        )

        # update critic
        critic_params_f = optax.apply_updates(critic_params[0], critic_updates_f)

        desc_transitions = transitions.replace(rewards=transitions.desc_rewards)

        # compute loss and gradients
        random_key, subkey = jax.random.split(random_key)
        critic_loss_d, critic_gradient_d = jax_value_and_grad(self._desc_critic_loss_fn)(
            critic_params[1],
            target_actor_params,
            target_critic_params[1],
            desc_transitions,
            subkey,
        )
        critic_updates_d, critic_optimizer_state_d = self._desc_critic_optimizer.update(
            critic_gradient_d, critic_optimizer_state[1]
        )

        # update critic
        critic_params_d = astype(
            optax.apply_updates(critic_params[1], critic_updates_d), flax.core.scope.VariableDict
        )

        critic_params = (critic_params_f, critic_params_d)
        critic_optimizer_state = (critic_optimizer_state_f, critic_optimizer_state_d)

        # Soft update of target critic network
        target_critic_params = jax.tree_map(
            lambda x1, x2: (1.0 - self._config.soft_tau_update) * x1
            + self._config.soft_tau_update * x2,
            target_critic_params,
            critic_params,
        )

        return critic_optimizer_state, critic_params, target_critic_params, random_key

    @functools.partial(jax_jit, static_argnames=("self",))
    def _update_actor(
        self,
        emitter_state: _OMGMEGARLEmitterState,
        actor_params: flax.core.scope.VariableDict,
        actor_opt_state: optax.OptState,
        target_actor_params: flax.core.scope.VariableDict,
        critic_params: tuple[Params, flax.core.scope.VariableDict],
        transitions: ExtendedQDTransition,
        random_key: RNGKey,
    ) -> tuple[optax.OptState, flax.core.scope.VariableDict, flax.core.scope.VariableDict, RNGKey]:

        desc_transitions = transitions.replace(rewards=transitions.desc_rewards)

        # Update greedy actor
        policy_loss_f, policy_gradient_f = jax_value_and_grad(self._policy_loss_fn)(
            actor_params,
            critic_params[0],
            transitions,
        )
        policy_gradient_d: flax.core.scope.VariableDict = jax_jacrev(self._policy_desc_loss_fn)(
            actor_params,
            critic_params[1],
            desc_transitions,
        )
        policy_gradients = tree_concatenate(
            tree_getitem(policy_gradient_f, jnp.newaxis),
            policy_gradient_d,
        )

        random_key, subkey = jax.random.split(random_key)
        grad_dir = self._get_grad_dir(emitter_state, subkey)
        policy_gradient = self._reduce_gradients(policy_gradients, grad_dir)

        (
            policy_updates,
            actor_optimizer_state,
        ) = self._actor_optimizer.update(policy_gradient, actor_opt_state)
        actor_params = astype(
            optax.apply_updates(actor_params, policy_updates), flax.core.scope.VariableDict
        )

        # Soft update of target greedy actor
        target_actor_params = jax.tree_map(
            lambda x1, x2: (1.0 - self._config.soft_tau_update) * x1
            + self._config.soft_tau_update * x2,
            target_actor_params,
            actor_params,
        )

        return (
            actor_optimizer_state,
            actor_params,
            target_actor_params,
            random_key,
        )

    @functools.partial(
        jax_jit,
        static_argnames=("self",),
    )
    def emit_pg(
        self,
        emitter_state: _OMGMEGARLEmitterState,
        parents: flax.core.scope.VariableDict,
        random_key: RNGKey,
    ) -> Genotype:
        random_key, subkey = jax.random.split(random_key)
        keys = jax.random.split(subkey, self._config.env_batch_size - 1)
        grad_dir = jax.vmap(self._get_grad_dir, in_axes=(None, 0))(emitter_state, keys)

        keys = jax.random.split(random_key, self._config.env_batch_size - 1)
        offsprings = jax.vmap(
            self._mutation_function_pg,
            in_axes=(0, 0, None, 0),
        )(parents, grad_dir, emitter_state, keys)

        return offsprings

    @functools.partial(jax_jit, static_argnames=("self",))
    def _mutation_function_pg(
        self,
        policy_params: flax.core.scope.VariableDict,
        grad_dir: jax.Array,
        emitter_state: _OMGMEGARLEmitterState,
        random_key: RNGKey,
    ) -> Genotype:
        # Define new policy optimizer state
        policy_optimizer_state = self._policies_optimizer.init(policy_params)

        def scan_train_policy(
            carry: tuple[
                _OMGMEGARLEmitterState,
                flax.core.scope.VariableDict,
                optax.OptState,
                jax.Array,
            ],
            random_key: RNGKey,
        ) -> tuple[
            tuple[
                _OMGMEGARLEmitterState,
                flax.core.scope.VariableDict,
                optax.OptState,
                jax.Array,
            ],
            Any,
        ]:
            emitter_state, policy_params, policy_optimizer_state, grad_dir = carry
            (
                new_emitter_state,
                new_policy_params,
                new_policy_optimizer_state,
            ) = self._train_policy(
                emitter_state,
                policy_params,
                grad_dir,
                policy_optimizer_state,
                random_key,
            )
            return (
                new_emitter_state,
                new_policy_params,
                new_policy_optimizer_state,
                grad_dir,
            ), ()

        keys = jax.random.split(random_key, self._config.num_pg_training_steps)
        (emitter_state, policy_params, policy_optimizer_state, _), _ = jax.lax.scan(
            scan_train_policy,
            (emitter_state, policy_params, policy_optimizer_state, grad_dir),
            keys,
            length=self._config.num_pg_training_steps,
        )

        return policy_params

    @functools.partial(jax_jit, static_argnames=("self",))
    def _train_policy(
        self,
        emitter_state: _OMGMEGARLEmitterState,
        policy_params: flax.core.scope.VariableDict,
        grad_dir: jax.Array,
        policy_optimizer_state: optax.OptState,
        random_key: RNGKey,
    ) -> tuple[_OMGMEGARLEmitterState, flax.core.scope.VariableDict, optax.OptState]:
        # Sample a batch of transitions in the buffer
        replay_buffer = emitter_state.replay_buffer
        transitions, random_key = replay_buffer.sample(
            random_key, sample_size=self._config.batch_size
        )

        # update policy
        policy_optimizer_state, policy_params = self._update_policy(
            critic_params=emitter_state.critic_params,
            policy_optimizer_state=policy_optimizer_state,
            policy_params=policy_params,
            grad_dir=grad_dir,
            transitions=transitions,
        )

        return emitter_state, policy_params, policy_optimizer_state

    @functools.partial(jax_jit, static_argnames=("self",))
    def _update_policy(
        self,
        critic_params: tuple[Params, flax.core.scope.VariableDict],
        policy_optimizer_state: optax.OptState,
        policy_params: flax.core.scope.VariableDict,
        grad_dir: jax.Array,
        transitions: ExtendedQDTransition,
    ) -> tuple[optax.OptState, flax.core.scope.VariableDict]:

        desc_transitions = transitions.replace(rewards=transitions.desc_rewards)

        # compute loss
        _policy_loss_f, policy_gradient_f = jax_value_and_grad(self._policy_loss_fn)(
            policy_params,
            critic_params[0],
            transitions,
        )
        policy_gradient_d = jax_jacrev(self._policy_desc_loss_fn)(
            policy_params,
            critic_params[1],
            desc_transitions,
        )
        policy_gradients = tree_concatenate(
            tree_getitem(policy_gradient_f, jnp.newaxis),
            policy_gradient_d,
        )
        policy_gradient = self._reduce_gradients(policy_gradients, grad_dir)
        # Compute gradient and update policies
        (
            policy_updates,
            policy_optimizer_state,
        ) = self._policies_optimizer.update(policy_gradient, policy_optimizer_state)
        policy_params = astype(
            optax.apply_updates(policy_params, policy_updates),
            flax.core.scope.VariableDict,
        )

        return policy_optimizer_state, policy_params

    def _reduce_gradients(
        self,
        gradients: flax.core.scope.VariableDict,
        grad_dir: jax.Array,
    ) -> flax.core.scope.VariableDict:
        sum_square_gradients = jnp.sum(jnp.concatenate(tuple(
            jnp.sum(jnp.square(leaf), axis=tuple(range(1, len(leaf.shape))))
            for leaf in jax.tree_util.tree_leaves(gradients)
        )), axis=0)
        cnt_gradients = jnp.sum(jnp.concatenate(tuple(
            jnp.sum(jnp.ones_like(leaf), axis=tuple(range(1, len(leaf.shape))))
            for leaf in jax.tree_util.tree_leaves(gradients)
        )), axis=0)
        assert sum_square_gradients.shape == cnt_gradients.shape
        norm_gradients = jnp.sqrt(sum_square_gradients / cnt_gradients)

        gradient: flax.core.scope.VariableDict = jax.tree_map(
            lambda grads: jnp.sum(
                jnp.expand_dims(
                    grad_dir / norm_gradients,
                    axis=tuple(range(-(len(grads.shape) - 1), 0)),
                ) * grads,
                axis=(0,),
            ),
            gradients,
        )

        return gradient

    def _get_grad_dir(
        self,
        emitter_state: _OMGMEGARLEmitterState,
        random_key: RNGKey,
    ) -> jax.Array:
        random_key, subkey = jax.random.split(random_key)
        grad_dir = jax.random.normal(
            subkey,
            shape=(1 + self._env.behavior_descriptor_length,),
        )
        grad_dir = grad_dir.at[0].set(jnp.abs(grad_dir[0]))
        grad_dir = astype(
            grad_dir / jnp.linalg.norm(grad_dir, axis=-1, keepdims=True), jax.Array
        )
        return grad_dir


@dataclass
class OMGMEGARLEmitterConfig(PGAMEEmitterConfig):
    desc_critic_hidden_layer_size: tuple[int, ...] = (256, 256)

    desc_critic_learning_rate: float = 3e-4

    desc_reward_scaling: float = 1.0
    desc_discount: float = 1.0


class OMGMEGARLEmitter(qdax.core.emitters.multi_emitter.MultiEmitter):
    def __init__(
        self,
        config: OMGMEGARLEmitterConfig,
        policy_network: nn.Module,
        env: qdax.environments.base_wrappers.QDEnv,
        variation_fn: Callable[[Params, Params, RNGKey], tuple[Params, RNGKey]],
    ):

        self._config = config
        self._policy_network = policy_network
        self._env = env
        self._variation_fn = variation_fn

        ga_batch_size = int(self._config.proportion_mutation_ga * config.env_batch_size)
        omg_batch_size = config.env_batch_size - ga_batch_size

        omg_config = _OMGMEGARLEmitterConfig(
            env_batch_size=omg_batch_size,
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
            desc_critic_hidden_layer_size=config.desc_critic_hidden_layer_size,
            desc_critic_learning_rate=config.desc_critic_learning_rate,
            desc_reward_scaling=config.desc_reward_scaling,
            desc_discount=config.desc_discount,
        )

        # define the quality emitter
        omg_emitter = _OMGMEGARLEmitter(
            config=omg_config, policy_network=policy_network, env=env
        )

        # define the GA emitter
        ga_emitter = qdax.core.emitters.standard_emitters.MixingEmitter(
            mutation_fn=lambda x, r: (x, r),
            variation_fn=variation_fn,
            variation_percentage=1.0,
            batch_size=ga_batch_size,
        )

        super().__init__(emitters=(omg_emitter, ga_emitter))
