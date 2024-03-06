import jax
import jax.numpy as jnp
import dm_pix
import flax.core.scope
from brax.envs import State
import brax.jumpy as jp
from qdax.core.neuroevolution.buffers import buffer
import qdax.core.neuroevolution.mdp_utils
import qdax.environments
from qdax.types import StateDescriptor, EnvState, Params, RNGKey, Fitness, Descriptor, ExtraScores
from chex import ArrayTree

import numpy as np
import gym.vector
import gymnasium.vector
import gym.spaces
import gymnasium.spaces
from chex import ArrayNumpyTree

import logging
import functools
from abc import abstractmethod
from collections.abc import Callable
from typing import Any, TypeVar, TYPE_CHECKING
from overrides import override

from .neuroevolution import Actor, ExtendedQDTransition
from .config import GymEnvConfig
from .utils import (
    astype,
    duplicate,
    tree_asarray,
    tree_shape,
    tree_shape_dtype,
    tree_getitem,
    tree_reduplicate,
    tree_indentical_duplicates,
    jax_jit,
    jax_pure_callback,
)

if TYPE_CHECKING:
    from .config import Config


_log = logging.getLogger(__name__)


_GymOnpState = tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, ArrayNumpyTree]]
GymJnpState = tuple[jax.Array, jax.Array, jax.Array, jax.Array, dict[str, ArrayTree]]


REWARD_OFFSET = {
    'ALE/Pong-v5': 22.0,
}


BEHAVIOR_DESCRIPTOR_RANGE = {
    'ALE/Pong-v5': ([0.0], [1.0]),
}


def get_empty_onp_info(
    obs: np.ndarray,
    reward: np.ndarray,
    terminated: np.ndarray,
    truncated: np.ndarray,
    info: dict[str, Any],
) -> dict[str, ArrayNumpyTree]:
    return {}


ONP_INFO_EXTRACTOR = {
    'ALE/Pong-v5': get_empty_onp_info,
}


def wrap_atari_obs_space(obs_space: gym.Space | gymnasium.Space) -> gym.Space | gymnasium.Space:
    assert isinstance(obs_space, (gym.spaces.Box, gymnasium.spaces.Box))
    return type(obs_space)(
        low=np.zeros((84, 84, 1), dtype=np.float32),
        high=np.ones((84, 84, 1), dtype=np.float32),
    )


OBS_SPACE_WRAPPER = {
    'ALE/Pong-v5': wrap_atari_obs_space,
}


def wrap_atari_obs(obs: jax.Array, obs_space: gym.Space | gymnasium.Space) -> jax.Array:
    assert isinstance(obs_space, (gym.spaces.Box, gymnasium.spaces.Box))
    obs = (obs - obs_space.low) / (obs_space.high - obs_space.low)
    obs = jnp.squeeze(dm_pix.rgb_to_grayscale(obs), axis=-1)
    resize_fn: Callable[[jax.Array], jax.Array] = (
        lambda obs: jax.image.resize(obs, (84, 84), jax.image.ResizeMethod.LINEAR)
    )
    for _ in range(len(obs.shape[:-2])):
        resize_fn = jax.vmap(resize_fn)
    obs = jnp.expand_dims(resize_fn(obs), axis=-1)
    return obs


OBS_WRAPPER = {
    'ALE/Pong-v5': wrap_atari_obs,
}


def get_pong_state_descriptor(state: GymJnpState, action: jax.Array) -> StateDescriptor:
    return jnp.expand_dims((action >= 2).astype(jnp.float32), axis=-1)


STATE_DESCRIPTOR_LEN = {
    'ALE/Pong-v5': 1,
}


STATE_DESCRIPTOR_EXTRACTOR = {
    'ALE/Pong-v5': get_pong_state_descriptor,
}


def get_pong_descriptor(data: buffer.QDTransition, mask: jax.Array) -> Descriptor:
    # reshape mask for bd extraction
    mask = jnp.expand_dims(mask, axis=-1)

    # Get behavior descriptor
    descriptors = jnp.sum(data.state_desc * (1.0 - mask), axis=1)
    descriptors = descriptors / jnp.sum(1.0 - mask, axis=1)

    return descriptors


BEHAVIOR_DESCRIPTOR_EXTRACTOR = {
    'ALE/Pong-v5': get_pong_descriptor,
}


@functools.partial(
    jax_jit,
    static_argnames=(
        "episode_length",
        "play_step_fn",
        "behavior_descriptor_extractor",
    ),
)
def scoring_function_brax_envs(
    policies_params: Params,
    random_key: RNGKey,
    init_states: EnvState,
    episode_length: int,
    play_step_fn: Callable[
        [EnvState, Params, RNGKey],
        tuple[EnvState, Params, RNGKey, buffer.QDTransition],
    ],
    behavior_descriptor_extractor: Callable[[buffer.QDTransition, jax.Array], Descriptor],
) -> tuple[Fitness, Descriptor, ExtraScores, RNGKey, jax.Array]:
    """Evaluates policies contained in policies_params in parallel in
    deterministic or pseudo-deterministic environments.

    This rollout is only deterministic when all the init states are the same.
    If the init states are fixed but different, as a policy is not necessarily
    evaluated with the same environment everytime, this won't be determinist.
    When the init states are different, this is not purely stochastic.

    Args:
        policies_params: The parameters of closed-loop controllers/policies to evaluate.
        random_key: A jax random key
        episode_length: The maximal rollout length.
        play_step_fn: The function to play a step of the environment.
        behavior_descriptor_extractor: The function to extract the behavior descriptor.

    Returns:
        fitness: Array of fitnesses of all evaluated policies
        descriptor: Behavioural descriptors of all evaluated policies
        extra_scores: Additional information resulting from evaluation
        random_key: The updated random key.
    """

    # Perform rollouts with each policy
    random_key, subkey = jax.random.split(random_key)
    unroll_fn = functools.partial(
        qdax.core.neuroevolution.mdp_utils.generate_unroll,
        episode_length=episode_length,
        play_step_fn=play_step_fn,
        random_key=subkey,
    )

    if hasattr(init_states, 'obs'):
        env_batch_size = init_states.obs.shape[0]
    else:
        raise NotImplementedError('Cannot check the batch size of init_states.')

    try:
        param_batch_size = Actor.get_decision_param_batch_sizes(
            astype(policies_params, flax.core.scope.VariableDict)
        )
        if param_batch_size is None:
            _log.warning(
                'The batch size of policies_params is None.'
            )
        elif env_batch_size != param_batch_size:
            _log.warning(
                'The batch sizes of init_states and policies_params are not the same. '
                'Reduplicating...'
            )
            init_states = tree_reduplicate(init_states, repeats=param_batch_size)
    except (KeyError, TypeError):
        _log.info('The check of batch size of policies_params is skipped.')

    _final_state, data = unroll_fn(init_states, policies_params)
    data: buffer.QDTransition = jax.tree_map(
        lambda x: jnp.swapaxes(x, 0, 1), data
    )

    # create a mask to extract data properly
    is_done = jnp.clip(jnp.cumsum(data.dones, axis=1), 0, 1)
    mask = jnp.roll(is_done, 1, axis=1)
    mask = mask.at[:, 0].set(0)

    # scores
    fitnesses = jnp.sum(data.rewards * (1.0 - mask), axis=1)
    descriptors = behavior_descriptor_extractor(data, mask)
    n_step = jnp.sum(1.0 - mask, axis=1).astype(jnp.int32)

    if isinstance(data, ExtendedQDTransition):
        desc_rewards = astype(data, ExtendedQDTransition).desc_rewards
        last_index = jnp.int32(jnp.sum(1.0 - mask, axis=1)) - 1
        desc_rewards = jax.vmap(lambda x, y, z: x.at[y].set(z))(
            desc_rewards, last_index, descriptors
        )
        data = astype(data.replace(desc_rewards=desc_rewards), ExtendedQDTransition)

    return (
        fitnesses,
        descriptors,
        {
            "transitions": data,
        },
        random_key,
        n_step,
    )


@functools.partial(
    jax_jit,
    static_argnames=(
        'reset_fn',
        'episode_length',
        'play_step_fn',
        'behavior_descriptor_extractor',
    ),
)
def scoring_function_gym_envs(
    policies_params: Params,
    random_key: jax.random.KeyArray,
    reset_fn: Callable[..., GymJnpState],
    episode_length: int,
    play_step_fn: Callable[
        [GymJnpState, Params, jax.random.KeyArray],
        tuple[GymJnpState, Params, jax.random.KeyArray, buffer.QDTransition],
    ],
    behavior_descriptor_extractor: Callable[[buffer.QDTransition, jax.Array], Descriptor],
) -> tuple[Fitness, Descriptor, ExtraScores, RNGKey, jax.Array]:
    '''Evaluates policies contained in policies_params in parallel in
    deterministic or pseudo-deterministic environments.

    Args:
        policies_params: The parameters of closed-loop controllers/policies to evaluate.
        random_key: A jax random key
        episode_length: The maximal rollout length.
        play_step_fn: The function to play a step of the environment.
        behavior_descriptor_extractor: The function to extract the behavior descriptor.

    Returns:
        fitness: Array of fitnesses of all evaluated policies
        descriptor: Behavioural descriptors of all evaluated policies
        extra_scores: Additional information resulting from evaluation
        random_key: The updated random key.
    '''

    init_states = reset_fn(random_key)

    # Perform rollouts with each policy
    random_key, subkey = jax.random.split(random_key)
    unroll_fn = functools.partial(
        qdax.core.neuroevolution.mdp_utils.generate_unroll,
        episode_length=episode_length,
        play_step_fn=play_step_fn,
        random_key=subkey,
    )

    _final_state, data = unroll_fn(init_states, policies_params)
    data: buffer.QDTransition = jax.tree_map(
        lambda x: jnp.swapaxes(x, 0, 1), data
    )
    _log.debug('shape(data) = %s', tree_shape(data))

    # create a mask to extract data properly
    is_done = jnp.clip(jnp.cumsum(data.dones, axis=1), 0, 1)
    mask = jnp.roll(is_done, 1, axis=1)
    mask = mask.at[:, 0].set(0)

    # scores
    fitnesses = jnp.sum(data.rewards * (1.0 - mask), axis=1)
    descriptors = behavior_descriptor_extractor(data, mask)
    n_step = jnp.sum(1.0 - mask, axis=1).astype(jnp.int32)

    if isinstance(data, ExtendedQDTransition):
        desc_rewards = astype(data, ExtendedQDTransition).desc_rewards
        last_index = jnp.int32(jnp.sum(1.0 - mask, axis=1)) - 1
        desc_rewards = jax.vmap(lambda x, y, z: x.at[y].set(z))(
            desc_rewards, last_index, descriptors
        )
        data = astype(data.replace(desc_rewards=desc_rewards), ExtendedQDTransition)

    return (
        fitnesses,
        descriptors,
        {
            'transitions': data,
        },
        random_key,
        n_step,
    )


_TState = TypeVar('_TState', bound=State | GymJnpState)


class Env:
    @abstractmethod
    def __init__(self, random_key: jax.random.KeyArray, cfg: 'Config'):
        self.cfg = cfg
        self.env: Any
        self.init_states = self.get_init_states(random_key)

    def get_init_states(self, random_key: jax.random.KeyArray) -> EnvState:
        keys = duplicate(random_key, repeats=self.cfg.algo.env_batch_size)
        reset_fn = jax_jit(jax.vmap(self.env.reset))
        init_states = reset_fn(keys)
        assert tree_indentical_duplicates(init_states)
        return init_states

    @property
    @abstractmethod
    def obs_space(self) -> gym.Space | gymnasium.Space:
        ...

    @property
    @abstractmethod
    def action_space(self) -> gym.Space | gymnasium.Space:
        ...

    @property
    def obs_size(self) -> int:
        shape = self.obs_space.shape
        assert shape is not None
        return int(np.prod(shape))

    @property
    def action_size(self) -> int:
        shape = self.action_space.shape
        assert shape is not None
        return int(np.prod(shape))

    def _flatten_obs(self, obs: jax.Array) -> jax.Array:
        shape = self.obs_space.shape
        assert shape is not None
        return obs.reshape(*obs.shape[:-len(shape)], self.obs_size)

    def _unflatten_action(self, action: jax.Array) -> jax.Array:
        shape = self.action_space.shape
        assert shape is not None
        return action.reshape(*action.shape[:-1], *shape)

    @property
    @abstractmethod
    def bd_len(self) -> int:
        ...

    @property
    @abstractmethod
    def bd_range(self) -> tuple[list[float], list[float]]:
        ...

    @property
    @abstractmethod
    def state_descriptor_len(self) -> int:
        ...

    @abstractmethod
    def step(self, state: _TState, action: jax.Array) -> _TState:
        ...

    @abstractmethod
    def get_scoring_fn(
        self,
        select_action_fn: Callable[[Params, jax.Array | jp.ndarray], jax.Array],
        TransitionType: type[buffer.Transition],
    ) -> Callable[[Params, RNGKey], tuple[Fitness, Descriptor, ExtraScores, RNGKey, jax.Array]]:
        ...


class QDEnv(Env):
    @override
    def __init__(self, random_key: jax.random.KeyArray, cfg: 'Config', eval_metrics: bool = False):
        self.cfg = cfg
        self.env: qdax.environments.QDEnv = qdax.environments.create(
            self.cfg.env.name,
            episode_length=self.cfg.env.episode_len,
            eval_metrics=eval_metrics,
        )
        self.init_states = self.get_init_states(random_key)

    @property
    @override
    def obs_space(self) -> gymnasium.Space:
        return gymnasium.spaces.Box(-np.inf, np.inf, shape=(self.env.observation_size,))

    @property
    @override
    def action_space(self) -> gymnasium.Space:
        return gymnasium.spaces.Box(-np.inf, np.inf, shape=(self.env.action_size,))

    @property
    @override
    def bd_len(self) -> int:
        return self.env.behavior_descriptor_length

    @property
    @override
    def bd_range(self) -> tuple[list[float], list[float]]:
        return self.env.behavior_descriptor_limits

    @property
    @override
    def state_descriptor_len(self) -> int:
        return self.env.behavior_descriptor_length

    @override
    def step(self, state: State, action: jax.Array) -> State:
        return jax.vmap(self.env.step)(state, action)

    @override
    def get_scoring_fn(
        self,
        select_action_fn: Callable[
            [Params, jax.Array | jp.ndarray], jax.Array
        ],
        TransitionType: type[buffer.Transition],
    ) -> Callable[[Params, RNGKey], tuple[Fitness, Descriptor, ExtraScores, RNGKey, jax.Array]]:
        assert issubclass(TransitionType, buffer.QDTransition)

        def play_step_fn(
            state: State,
            policy_params: Params,
            random_key: jax.random.KeyArray,
        ):
            actions: jax.Array = select_action_fn(policy_params, state.obs)

            state_desc: StateDescriptor = state.info['state_descriptor']
            next_state = self.step(state, actions)

            kwargs: dict[str, jax.Array] = {}
            if issubclass(TransitionType, ExtendedQDTransition):
                kwargs['desc_rewards'] = jnp.zeros(
                    (*next_state.reward.shape, self.bd_len), dtype=jnp.float32
                )

            transition = TransitionType(
                obs=state.obs,
                next_obs=next_state.obs,
                rewards=next_state.reward,
                dones=next_state.done,
                actions=actions,
                truncations=next_state.info['truncation'],
                state_desc=state_desc,
                next_state_desc=next_state.info['state_descriptor'],
                **kwargs,
            )

            return next_state, policy_params, random_key, transition

        qd_extraction_fn = qdax.environments.behavior_descriptor_extractor[self.cfg.env.name]
        return functools.partial(
            scoring_function_brax_envs,
            init_states=self.init_states,
            episode_length=self.cfg.env.episode_len,
            play_step_fn=play_step_fn,
            behavior_descriptor_extractor=qd_extraction_fn,
        )


class GymEnv(Env):
    @override
    def __init__(self, random_key: jax.random.KeyArray, cfg: 'Config', eval_metrics: bool = False):
        self.cfg = cfg
        env_cfg = astype(self.cfg.env, GymEnvConfig)
        match env_cfg.package:
            case 'gym':
                make_fn = gym.vector.make
            case 'gymnasium':
                make_fn = gymnasium.vector.make
            case _:
                raise NotImplementedError
        self.env: gym.vector.VectorEnv | gymnasium.vector.VectorEnv = make_fn(
            self.env_name,
            num_envs=self.cfg.algo.env_batch_size,
            max_episode_steps=env_cfg.episode_len,
        )
        self._seed: int = jax.random.randint(random_key, (), 0, 0x7FFFFFFF).item()
        state: GymJnpState = tree_asarray(self._onp_reset())
        self._state_shape_dtype: GymJnpState = tree_shape_dtype(tree_getitem(state, 0))
        self._batched_state_shape_dtype: GymJnpState = tree_shape_dtype(state)
        self._zero_action: jax.Array = jnp.zeros_like(self.env.action_space.sample())

    @property
    def env_name(self) -> str:
        return self.cfg.env.name.replace('--', '/')

    @property
    @override
    def obs_space(self) -> gym.Space | gymnasium.Space:
        return OBS_SPACE_WRAPPER[self.env_name](self.env.single_observation_space)

    @property
    @override
    def action_space(self) -> gym.Space | gymnasium.Space:
        return self.env.single_action_space

    @property
    @override
    def bd_len(self) -> int:
        return len(BEHAVIOR_DESCRIPTOR_RANGE[self.env_name][0])

    @property
    @override
    def bd_range(self) -> tuple[list[float], list[float]]:
        return BEHAVIOR_DESCRIPTOR_RANGE[self.env_name]

    @property
    @override
    def state_descriptor_len(self) -> int:
        return STATE_DESCRIPTOR_LEN[self.env_name]

    def _build_onp_state(
        self,
        obs: np.ndarray,
        reward: np.ndarray,
        terminated: np.ndarray,
        truncated: np.ndarray,
        info: dict[str, Any],
    ) -> _GymOnpState:
        done = np.logical_or(terminated, truncated)
        extracted_info = ONP_INFO_EXTRACTOR[self.env_name](
            obs=obs,
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            info=info,
        )
        return obs, reward, done, truncated, extracted_info

    def _onp_reset(self, *_: Any) -> _GymOnpState:
        _log.info('_onp_reset')
        batch_size = self.env.num_envs
        obs, info = self.env.reset(seed=[self._seed] * batch_size)
        onp_state = self._build_onp_state(
            obs=obs,
            reward=np.zeros(batch_size, dtype=np.float32),
            terminated=np.zeros(batch_size, dtype=np.bool_),
            truncated=np.zeros(batch_size, dtype=np.bool_),
            info=info,
        )
        return onp_state

    def reset(self, *unused: Any) -> GymJnpState:
        state = jax_pure_callback(
            self._onp_reset, self._batched_state_shape_dtype, *unused, vectorized=True
        )
        obs, reward, done, truncated, info = state
        obs = OBS_WRAPPER[self.env_name](obs, self.env.single_observation_space)
        info['state_descriptor'] = STATE_DESCRIPTOR_EXTRACTOR[self.env_name](
            state, self._zero_action
        )
        return obs, reward, done, truncated, info

    def _onp_step(self, action: np.ndarray) -> _GymOnpState:
        results = self.env.step(action)
        assert results is not None
        obs, reward, terminated, truncated, info = results
        onp_state = self._build_onp_state(
            obs=obs,
            reward=np.asarray(reward, dtype=np.float32),
            terminated=terminated,
            truncated=truncated,
            info=info,
        )
        return onp_state

    def _step(self, state: GymJnpState, action: jax.Array) -> GymJnpState:
        next_state = jax_pure_callback(
            self._onp_step, self._state_shape_dtype, action, vectorized=True
        )
        next_obs, reward, done, truncated, info = next_state
        next_obs = OBS_WRAPPER[self.env_name](next_obs, self.env.single_observation_space)
        info['state_descriptor'] = STATE_DESCRIPTOR_EXTRACTOR[self.env_name](next_state, action)
        return next_obs, reward, done, truncated, info

    @override
    def step(self, state: GymJnpState, action: jax.Array) -> GymJnpState:
        return jax.vmap(self._step)(state, action)

    @override
    def get_scoring_fn(
        self,
        select_action_fn: Callable[
            [Params, jax.Array | jp.ndarray], jax.Array
        ],
        TransitionType: type[buffer.Transition],
    ) -> Callable[[Params, RNGKey], tuple[Fitness, Descriptor, ExtraScores, RNGKey, jax.Array]]:

        def play_step_fn(
            state: GymJnpState,
            policy_params: Params,
            random_key: jax.random.KeyArray,
        ):
            obs = state[0]
            _log.debug('obs.shape = %s', obs.shape)
            obs = self._flatten_obs(obs)
            _log.debug('obs.shape = %s', obs.shape)
            actions = select_action_fn(policy_params, obs)
            _log.debug('actions.shape = %s', actions.shape)
            unflattened_actions = self._unflatten_action(actions)
            _log.debug('unflattened_actions.shape = %s', unflattened_actions.shape)

            state_desc = state[-1]['state_descriptor']
            assert isinstance(state_desc, StateDescriptor)

            next_state = self.step(state, unflattened_actions)
            next_obs, rewards, dones, truncations, info = next_state

            _log.debug('next_obs.shape = %s', next_obs.shape)
            next_obs = self._flatten_obs(next_obs)
            _log.debug('next_obs.shape = %s', next_obs.shape)
            next_state_desc = info['state_descriptor']
            assert isinstance(next_state_desc, StateDescriptor)

            transition = TransitionType(
                obs=obs,
                next_obs=next_obs,
                rewards=rewards,
                dones=dones,
                actions=actions,
                truncations=truncations,
                state_desc=state_desc,
                next_state_desc=next_state_desc,
            )

            return next_state, policy_params, random_key, transition

        qd_extraction_fn = BEHAVIOR_DESCRIPTOR_EXTRACTOR[self.env_name]

        return functools.partial(
            scoring_function_gym_envs,
            reset_fn=self.reset,
            episode_length=self.cfg.env.episode_len,
            play_step_fn=play_step_fn,
            behavior_descriptor_extractor=qd_extraction_fn,
        )
