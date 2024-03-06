import jax
import jax.numpy as jnp
import flax.core
from qdax.core.neuroevolution.buffers import buffer
from qdax.types import Action, Observation, RNGKey

from collections.abc import Callable
from typing import Optional, TYPE_CHECKING

from ..utils import tree_getitem, jax_jit

if TYPE_CHECKING:
    from ..neuroevolution import Actor, CriticSingle, Critic, PolicyCriticSingle, PolicyCritic


def make_td3_loss_fn(
    policy_fn: Callable[[flax.core.scope.VariableDict, Observation], jax.Array],
    critic_fn: Callable[[flax.core.scope.VariableDict, Observation, Action], jax.Array],
    reward_scaling: float,
    discount: float,
    noise_clip: float,
    policy_noise: float,
) -> tuple[
    Callable[[
        flax.core.scope.VariableDict,
        flax.core.scope.VariableDict,
        buffer.Transition,
    ], jax.Array],
    Callable[[
        flax.core.scope.VariableDict,
        flax.core.scope.VariableDict,
        flax.core.scope.VariableDict,
        buffer.Transition,
        RNGKey
    ], jax.Array],
]:

    @jax_jit
    def _policy_loss_fn(
        policy_params: flax.core.scope.VariableDict,
        critic_params: flax.core.scope.VariableDict,
        transitions: buffer.Transition,
    ) -> jax.Array:
        """Policy loss function for TD3 agent"""

        action = policy_fn(policy_params, transitions.obs)
        q_value = critic_fn(
            critic_params, transitions.obs, action
        )
        q1_action = jnp.take(q_value, jnp.asarray(0), axis=-1)
        policy_loss = -jnp.mean(q1_action, axis=0)
        return policy_loss

    @jax.jit
    def _critic_loss_fn(
        critic_params: flax.core.scope.VariableDict,
        target_policy_params: flax.core.scope.VariableDict,
        target_critic_params: flax.core.scope.VariableDict,
        transitions: buffer.Transition,
        random_key: RNGKey,
    ) -> jax.Array:
        """Critics loss function for TD3 agent"""
        noise = (
            jax.random.normal(random_key, shape=transitions.actions.shape)
            * policy_noise
        ).clip(-noise_clip, noise_clip)

        next_action = (
            policy_fn(target_policy_params, transitions.next_obs) + noise
        ).clip(-1.0, 1.0)
        next_q = critic_fn(
            target_critic_params, transitions.next_obs, next_action
        )
        next_v = jnp.min(next_q, axis=-1)
        target_q = jax.lax.stop_gradient(
            transitions.rewards * reward_scaling
            + (1.0 - transitions.dones[..., jnp.newaxis]) * discount * next_v
        )
        q_old_action = critic_fn(
            critic_params,
            transitions.obs,
            transitions.actions,
        )
        q_error = q_old_action - jnp.expand_dims(target_q, -1)

        # Better bootstrapping for truncated episodes.
        mask = jnp.expand_dims(1.0 - transitions.truncations, -1)
        mask = jnp.expand_dims(mask, -1)
        q_error = q_error * mask

        # compute the loss
        q_losses = jnp.mean(jnp.square(q_error), axis=-3)
        q_loss = jnp.sum(q_losses, axis=-1)
        q_loss = jnp.sum(q_loss, axis=-1)

        return q_loss

    return _policy_loss_fn, _critic_loss_fn


def make_cc_td3_loss_fn(
    actor: 'Actor',
    critic: 'CriticSingle | Critic | PolicyCriticSingle | PolicyCritic',
    actor_fn: Callable[[
        flax.core.scope.VariableDict,
        jax.Array,
        flax.core.scope.VariableDict,
    ], jax.Array],
    critic_fn: Callable[[
        flax.core.scope.VariableDict,
        jax.Array,
        jax.Array,
    ], jax.Array],
    critic1_fn: Callable[[
        flax.core.scope.VariableDict,
        jax.Array,
        jax.Array,
    ], jax.Array],
    policy_critic_fn: Callable[[
        flax.core.scope.VariableDict,
        jax.Array,
        jax.Array,
        flax.core.scope.VariableDict,
    ], jax.Array],
    policy_critic1_fn: Callable[[
        flax.core.scope.VariableDict,
        jax.Array,
        jax.Array,
        flax.core.scope.VariableDict,
    ], jax.Array],
    reward_scaling: float,
    discount: float,
    greedy_noise_clip: float,
    noise_clip: float,
    greedy_policy_noise: float,
    policy_noise: float,
) -> tuple[
    Callable[[
        flax.core.scope.VariableDict,
        flax.core.scope.VariableDict,
        flax.core.scope.VariableDict,
        buffer.QDTransition,
        Optional[jax.Array],
    ], jax.Array],
    Callable[[
        flax.core.scope.VariableDict,
        flax.core.scope.VariableDict,
        flax.core.scope.VariableDict,
        buffer.QDTransition,
    ], jax.Array],
    Callable[[
        flax.core.scope.VariableDict,
        flax.core.scope.VariableDict,
        flax.core.scope.VariableDict,
        flax.core.scope.VariableDict,
        buffer.QDTransition,
        RNGKey,
    ], jax.Array],
    Callable[[
        flax.core.scope.VariableDict,
        flax.core.scope.VariableDict,
        flax.core.scope.VariableDict,
        flax.core.scope.VariableDict,
        buffer.QDTransition,
        RNGKey,
        Optional[jax.Array],
    ], jax.Array],
]:

    @jax_jit
    def _actor_loss_fn(
        representation_vars: flax.core.scope.VariableDict,
        decision_vars: flax.core.scope.VariableDict,
        policy_critic_vars: flax.core.scope.VariableDict,
        transitions: buffer.QDTransition,
        representation_indices: Optional[jax.Array] = None,
    ) -> jax.Array:
        representation_batch_size = actor.get_param_batch_sizes(representation_vars)
        decision_batch_size = actor.get_param_batch_sizes(decision_vars)
        policy_critic_batch_size = critic.get_param_batch_sizes(policy_critic_vars)
        if representation_batch_size is not None:
            assert representation_indices is not None
            representation_vars = tree_getitem(representation_vars, representation_indices)
            representation_batch_size = actor.get_param_batch_sizes(representation_vars)
            assert representation_batch_size == decision_batch_size
        if policy_critic_batch_size is not None:
            assert representation_indices is not None
            policy_critic_vars = tree_getitem(policy_critic_vars, representation_indices)
            policy_critic_batch_size = critic.get_param_batch_sizes(policy_critic_vars)
            assert policy_critic_batch_size == decision_batch_size
        assert representation_batch_size == policy_critic_batch_size
        vmapped_actor_fn = jax.vmap(
            actor_fn,
            in_axes=(0 if representation_batch_size is not None else None, None, 0),
        ) if decision_batch_size is not None else actor_fn
        action = vmapped_actor_fn(representation_vars, transitions.obs, decision_vars)

        vmapped_policy_critic1_fn = jax.vmap(
            policy_critic1_fn,
            in_axes=(0 if policy_critic_batch_size is not None else None, None, 0, 0),
        ) if decision_batch_size is not None else policy_critic1_fn
        q1 = vmapped_policy_critic1_fn(policy_critic_vars, transitions.obs, action, decision_vars)

        actor_loss = -jnp.mean(q1, axis=-1)
        if decision_batch_size is not None:
            actor_loss = jnp.sum(actor_loss, axis=0)
        return actor_loss

    @jax_jit
    def _greedy_actor_loss_fn(
        greedy_decision_vars: flax.core.scope.VariableDict,
        representation_vars: flax.core.scope.VariableDict,
        critic_vars: flax.core.scope.VariableDict,
        transitions: buffer.QDTransition,
    ) -> jax.Array:
        action = actor_fn(
            greedy_decision_vars,
            transitions.obs,
            representation_vars,
        )

        q1 = critic1_fn(
            critic_vars, transitions.obs, action,
        )

        greedy_actor_loss = -jnp.mean(q1, axis=-1)
        return greedy_actor_loss

    @jax_jit
    def _critic_loss_fn(
        critic_vars: flax.core.scope.VariableDict,
        target_representation_vars: flax.core.scope.VariableDict,
        target_greedy_decision_vars: flax.core.scope.VariableDict,
        target_critic_vars: flax.core.scope.VariableDict,
        transitions: buffer.QDTransition,
        random_key: RNGKey,
    ) -> jax.Array:
        noise = (
            jax.random.normal(random_key, shape=transitions.actions.shape)
            * greedy_policy_noise
        ).clip(-greedy_noise_clip, greedy_noise_clip)

        next_action = (
            actor_fn(
                target_representation_vars,
                transitions.next_obs,
                target_greedy_decision_vars,
            ) + noise
        ).clip(-1.0, 1.0)
        next_q = critic_fn(
            target_critic_vars, transitions.next_obs, next_action,
        ).min(axis=-1)
        target_q = jax.lax.stop_gradient(
            transitions.rewards * reward_scaling
            + (1.0 - transitions.dones) * discount * next_q
        )
        q = critic_fn(
            critic_vars,
            transitions.obs,
            transitions.actions,
        )
        q_error = q - jnp.expand_dims(target_q, -1)
        mask = jnp.expand_dims(1.0 - transitions.truncations, -1)
        q_error = q_error * mask

        q_losses = jnp.mean(jnp.square(q_error), axis=-2)
        q_loss = q_losses.sum(axis=-1)

        return q_loss

    @jax_jit
    def _policy_critic_loss_fn(
        policy_critic_vars: flax.core.scope.VariableDict,
        used_representation_vars: flax.core.scope.VariableDict,
        decision_vars: flax.core.scope.VariableDict,
        target_policy_critic_vars: flax.core.scope.VariableDict,
        transitions: buffer.QDTransition,
        random_key: RNGKey,
        representation_indices: Optional[jax.Array] = None,
    ) -> jax.Array:
        used_representation_batch_size = actor.get_param_batch_sizes(used_representation_vars)
        decision_batch_size = actor.get_param_batch_sizes(decision_vars)
        policy_critic_batch_size = critic.get_param_batch_sizes(policy_critic_vars)
        target_policy_critic_batch_size = critic.get_param_batch_sizes(target_policy_critic_vars)
        assert decision_batch_size is not None
        if used_representation_batch_size is not None:
            assert representation_indices is not None
            used_representation_vars = tree_getitem(
                used_representation_vars, representation_indices
            )
            used_representation_batch_size = actor.get_param_batch_sizes(used_representation_vars)
            assert used_representation_batch_size == decision_batch_size
        if policy_critic_batch_size is not None:
            assert representation_indices is not None
            policy_critic_vars = tree_getitem(policy_critic_vars, representation_indices)
            policy_critic_batch_size = critic.get_param_batch_sizes(
                policy_critic_vars
            )
            assert policy_critic_batch_size == decision_batch_size
        if target_policy_critic_batch_size is not None:
            assert representation_indices is not None
            target_policy_critic_vars = tree_getitem(
                target_policy_critic_vars, representation_indices
            )
            target_policy_critic_batch_size = critic.get_param_batch_sizes(
                target_policy_critic_vars
            )
            assert target_policy_critic_batch_size == decision_batch_size
        assert policy_critic_batch_size == target_policy_critic_batch_size
        assert used_representation_batch_size == policy_critic_batch_size
        noise = (
            jax.random.normal(
                random_key, shape=(decision_batch_size, *transitions.actions.shape)
            ) * policy_noise
        ).clip(-noise_clip, noise_clip)

        next_action = (
            jax.vmap(
                actor_fn,
                in_axes=(0 if used_representation_batch_size is not None else None, None, 0),
            )(used_representation_vars, transitions.next_obs, decision_vars) + noise
        ).clip(-1.0, 1.0)
        next_q = jax.vmap(
            policy_critic_fn,
            in_axes=(0 if target_policy_critic_batch_size is not None else None, None, 0, 0),
        )(
            target_policy_critic_vars, transitions.next_obs, next_action, decision_vars
        ).min(axis=-1)
        target_q = jax.lax.stop_gradient(
            transitions.rewards * reward_scaling
            + (1.0 - transitions.dones) * discount * next_q
        )

        q = jax.vmap(
            policy_critic_fn,
            in_axes=(0 if policy_critic_batch_size is not None else None, None, None, 0),
        )(policy_critic_vars, transitions.obs, transitions.actions, decision_vars)
        q_error = q - jnp.expand_dims(target_q, -1)
        mask = jnp.expand_dims(1.0 - transitions.truncations, -1)
        q_error = q_error * mask

        q_losses = jnp.mean(jnp.square(q_error), axis=-2)
        q_loss = q_losses.sum(axis=-1)
        q_loss = q_loss.sum(axis=-1)

        return q_loss

    return _actor_loss_fn, _greedy_actor_loss_fn, _critic_loss_fn, _policy_critic_loss_fn


def make_dqn_loss_fn(
    fn: Callable[[flax.core.scope.VariableDict, jax.Array], jax.Array],
    reward_scaling: float,
    discount: float,
) -> Callable[[
    flax.core.scope.VariableDict,
    flax.core.scope.VariableDict,
    buffer.Transition,
], jax.Array]:

    @jax_jit
    def _loss_fn(
        policy_params: flax.core.scope.VariableDict,
        target_policy_params: flax.core.scope.VariableDict,
        transitions: buffer.Transition,
    ) -> jax.Array:
        next_action = jnp.argmax(fn(policy_params, transitions.next_obs), axis=-1, keepdims=True)
        next_v = jnp.take_along_axis(
            fn(target_policy_params, transitions.next_obs), next_action, axis=-1
        ).squeeze(axis=-1)
        assert isinstance(next_v, jax.Array)
        target_q = jax.lax.stop_gradient(
            transitions.rewards * reward_scaling
            + (1.0 - transitions.dones) * discount * next_v
        )
        q = jnp.take_along_axis(
            fn(policy_params, transitions.obs),
            (transitions.actions + 0.5 * jnp.sign(transitions.actions)).astype(jnp.int32),
            axis=-1,
        ).squeeze(axis=-1)
        assert isinstance(q, jax.Array)
        q_error = q - target_q

        # Better bootstrapping for truncated episodes.
        q_error = q_error * (1.0 - transitions.truncations)

        q_loss = jnp.mean(jnp.square(q_error), axis=0)

        return q_loss

    return _loss_fn
