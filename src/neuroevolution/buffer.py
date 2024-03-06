import jax
import jax.numpy as jnp
import qdax.core.neuroevolution.buffers.buffer
from qdax.types import Reward

from typing import Self
from overrides import override


class ExtendedQDTransition(qdax.core.neuroevolution.buffers.buffer.QDTransition):

    desc_rewards: Reward

    @property
    @override
    def flatten_dim(self) -> int:
        return super().flatten_dim + self.state_descriptor_dim

    def flatten(self) -> jax.Array:
        flatten_transition = jnp.concatenate(
            [
                self.obs,
                self.next_obs,
                jnp.expand_dims(self.rewards, axis=-1),
                jnp.expand_dims(self.dones, axis=-1),
                jnp.expand_dims(self.truncations, axis=-1),
                self.actions,
                self.state_desc,
                self.next_state_desc,
                self.desc_rewards,
            ],
            axis=-1,
        )
        return flatten_transition

    @classmethod
    def from_flatten(
        cls,
        flattened_transition: jax.Array,
        transition: Self,
    ) -> Self:
        obs_dim = transition.observation_dim
        action_dim = transition.action_dim
        desc_dim = transition.state_descriptor_dim

        obs = flattened_transition[:, :obs_dim]
        next_obs = flattened_transition[:, obs_dim:(2 * obs_dim)]
        rewards = jnp.ravel(flattened_transition[:, (2 * obs_dim):(2 * obs_dim + 1)])
        dones = jnp.ravel(
            flattened_transition[:, (2 * obs_dim + 1):(2 * obs_dim + 2)]
        )
        truncations = jnp.ravel(
            flattened_transition[:, (2 * obs_dim + 2):(2 * obs_dim + 3)]
        )
        actions = flattened_transition[
            :, (2 * obs_dim + 3):(2 * obs_dim + 3 + action_dim)
        ]
        state_desc = flattened_transition[
            :,
            (2 * obs_dim + 3 + action_dim):(2 * obs_dim + 3 + action_dim + desc_dim),
        ]
        next_state_desc = flattened_transition[
            :,
            (2 * obs_dim + 3 + action_dim + desc_dim):(
                2 * obs_dim + 3 + action_dim + 2 * desc_dim
            ),
        ]
        desc_rewards = flattened_transition[
            :,
            (2 * obs_dim + 3 + action_dim + 2 * desc_dim):(
                2 * obs_dim + 3 + action_dim + 3 * desc_dim
            )
        ]
        return cls(
            obs=obs,
            next_obs=next_obs,
            rewards=rewards,
            dones=dones,
            truncations=truncations,
            actions=actions,
            state_desc=state_desc,
            next_state_desc=next_state_desc,
            desc_rewards=desc_rewards,
        )

    @classmethod
    def init_dummy(
        cls, observation_dim: int, action_dim: int, descriptor_dim: int
    ) -> Self:
        dummy_transition = cls(
            obs=jnp.zeros(shape=(1, observation_dim)),
            next_obs=jnp.zeros(shape=(1, observation_dim)),
            rewards=jnp.zeros(shape=(1,)),
            dones=jnp.zeros(shape=(1,)),
            truncations=jnp.zeros(shape=(1,)),
            actions=jnp.zeros(shape=(1, action_dim)),
            state_desc=jnp.zeros(shape=(1, descriptor_dim)),
            next_state_desc=jnp.zeros(shape=(1, descriptor_dim)),
            desc_rewards=jnp.zeros(shape=(1, descriptor_dim)),
        )
        return dummy_transition
