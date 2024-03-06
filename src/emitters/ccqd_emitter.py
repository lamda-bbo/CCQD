import jax
import jax.numpy as jnp
import optax
import flax.core
import flax.struct
import qdax.core.containers.mapelites_repertoire
import qdax.core.emitters.emitter
from qdax.core.neuroevolution.buffers import buffer
from qdax.types import Descriptor, ExtraScores, Fitness, RNGKey

from dataclasses import dataclass
import logging
import functools
from collections.abc import Sequence, Callable
from typing import Optional, Any, TypeVar, TYPE_CHECKING

from ..neuroevolution import Critic, PolicyCritic, make_cc_td3_loss_fn
from .operators import OPTIMIZERS, OPERATORS
from .selection import SELECTORS
from ..utils import (
    activation, astype, jax_jit, jax_value_and_grad, dataclass_field, filter_wrap, astype_wrap,
    tree_copy, tree_getitem, tree_reversed_broadcasted_where_y, tree_concatenate,
)

if TYPE_CHECKING:
    from ..map_elites import ExtendedMapElitesRepertoire
    from ..neuroevolution import Actor
    from ..env import Env


_log = logging.getLogger(__name__)


@dataclass
class CCQDEmitterConfig:
    env_batch_size: int = 100
    proportion_pg: float = 0.5

    using_greedy_decision: bool = True
    using_critic: bool = True
    using_policy_critic: bool = True

    updating_policy_critic_with_target_representation: bool = True
    updating_actors_with_target_critics: bool = False
    updating_actors_with_new_critics: bool = False

    critic_hidden_layer_size: tuple[int, ...] = (256, 256)
    critic_activation: str = 'leaky_relu'
    critic_final_activation: Optional[str] = None

    policy_critic_encoder_layer_size: tuple[int, ...] = (64, 64, 64)
    policy_critic_hidden_layer_size: tuple[int, ...] = (256, 256)
    policy_critic_encoder_activation: str = 'leaky_relu'
    policy_critic_encoder_final_activation: Optional[str] = None
    policy_critic_activation: str = 'leaky_relu'
    policy_critic_final_activation: Optional[str] = None

    num_critic_training_steps: int = 300

    training_batch_size: int = 256
    training_decision_batch_size: int = 256

    num_pg_training_steps: int = 100

    actor_optimizer: str = 'Adam'
    representation_learning_rate: float = 3e-4
    decision_learning_rate: float = 0.001
    greedy_decision_learning_rate: float = 3e-4
    critic_learning_rate: float = 3e-4
    policy_critic_learning_rate: float = 3e-4

    reward_scaling: float = 1.0
    discount: float = 0.99
    greedy_noise_clip: float = 0.5
    noise_clip: float = 0.5
    greedy_policy_noise: float = 0.2
    policy_noise: float = 0.2

    soft_tau_update: float = 0.005
    policy_delay: int = 2

    num_decision_updating_policy_critic: int = 100
    num_decision_updating_representation: int = 100
    decision_factor: float = 1.0

    batch_updating_representation: bool = True

    replay_buffer_size: int = 1000000

    pg_selection: str = 'Random'
    ea_selection: tuple[str, ...] = ('Random', 'Random')
    crossover: Optional[str] = 'Iso-LineDD'
    mutation: Optional[str] = None
    crossover_params: dict[str, Any] = dataclass_field({
        'iso_sigma': 0.005,
        'line_sigma': 0.05,
    })
    mutation_params: dict[str, Any] = dataclass_field({})

    decision_sampling_updating_policy_critic: str = 'Random'
    decision_sampling_updating_representation: str = 'Random'

    n_centroids: int = 1024

    n_representation: int = 20
    p_concat_crossover: float = 1.0
    replacement_threshold: int = 0
    replaced_by: str = 'most'
    representation_indices_reassignment: str = 'Random'
    representation_indices_reassignment_params: dict[str, Any] = dataclass_field({})


class CCQDEmitterState(qdax.core.emitters.emitter.EmitterState):
    random_key: RNGKey

    representation_vars: flax.core.scope.VariableDict
    target_representation_vars: flax.core.scope.VariableDict
    representation_optimizer_state: optax.OptState

    greedy_decision_vars: flax.core.scope.VariableDict
    target_greedy_decision_vars: flax.core.scope.VariableDict
    greedy_decision_optimizer_state: optax.OptState

    critic_vars: flax.core.scope.VariableDict
    target_critic_vars: flax.core.scope.VariableDict
    critic_optimizer_state: optax.OptState

    policy_critic_vars: flax.core.scope.VariableDict
    target_policy_critic_vars: flax.core.scope.VariableDict
    policy_critic_optimizer_state: optax.OptState

    replay_buffer: buffer.ReplayBuffer

    offspring: flax.core.scope.VariableDict

    steps: jax.Array

    repertoire_representation_indices: Optional[jax.Array]
    offspring_representation_indices: Optional[jax.Array]


_TCCQDEmitterState = TypeVar('_TCCQDEmitterState', bound=CCQDEmitterState)


class CCQDDecisionInfo(flax.struct.PyTreeNode):
    representation_indices: Optional[jax.Array]


class CCQDEmitter(qdax.core.emitters.emitter.Emitter):

    def __init__(
        self,
        config: CCQDEmitterConfig,
        actor: 'Actor',
        env: 'Env',
        offspring_slice: slice = slice(None),
        StateType: type[_TCCQDEmitterState] = CCQDEmitterState
    ):
        self._config = config
        self._env = env
        self._actor = actor
        self._offspring_slice = offspring_slice
        self._StateType = StateType

        assert self._config.using_greedy_decision or not self._config.using_critic
        assert self._config.using_critic or self._config.using_policy_critic

        if not (
            self._config.using_greedy_decision
            and self._config.using_critic
        ):
            raise NotImplementedError

        self._critic = Critic(
            self._config.critic_hidden_layer_size,
            activation=activation(self._config.critic_activation),
            final_activation=activation(self._config.critic_final_activation),
        )

        if self._config.using_policy_critic:
            self._policy_critic = PolicyCritic(
                self._config.policy_critic_encoder_layer_size,
                self._config.policy_critic_hidden_layer_size,
                encoder_activation=activation(
                    self._config.policy_critic_encoder_activation
                ),
                encoder_final_activation=activation(
                    self._config.policy_critic_encoder_final_activation
                ),
                activation=activation(self._config.policy_critic_activation),
                final_activation=activation(self._config.policy_critic_final_activation),
            )
            policy_critic_fn = self._policy_critic.apply  # type: ignore
            policy_critic1_fn = self._policy_critic.apply_critic1
        else:
            self._policy_critic = None
            policy_critic_fn: Callable[[
                flax.core.scope.VariableDict,
                jax.Array,
                jax.Array,
                flax.core.scope.VariableDict,
            ], jax.Array] = lambda x, y, z, _: self._critic.apply(x, y, z)  # type: ignore
            policy_critic1_fn: Callable[[
                flax.core.scope.VariableDict,
                jax.Array,
                jax.Array,
                flax.core.scope.VariableDict,
            ], jax.Array] = lambda x, y, z, _: self._critic.apply_critic1(x, y, z)

        (
            self._actor_loss_fn,
            greedy_actor_loss_fn,
            self._critic_loss_fn,
            self._policy_critic_loss_fn,
        ) = make_cc_td3_loss_fn(
            actor=self._actor,
            critic=self._critic,
            actor_fn=self._actor.apply_partial,
            critic_fn=astype_wrap(self._critic.apply, jax.Array),
            critic1_fn=self._critic.apply_critic1,
            policy_critic_fn=policy_critic_fn,
            policy_critic1_fn=policy_critic1_fn,
            reward_scaling=self._config.reward_scaling,
            discount=self._config.discount,
            greedy_noise_clip=self._config.greedy_noise_clip,
            noise_clip=self._config.noise_clip,
            greedy_policy_noise=self._config.greedy_policy_noise,
            policy_noise=self._config.policy_noise,
        )

        if self._reduce_decision_batch:
            greedy_actor_loss_fn = jax.vmap(
                greedy_actor_loss_fn,
                in_axes=(0, 0, 0, None),
            )
            greedy_actor_loss_fn = filter_wrap(
                greedy_actor_loss_fn, functools.partial(jnp.sum, axis=0)
            )
            self._greedy_actor_loss_and_grad_fn = astype_wrap(
                jax_value_and_grad(greedy_actor_loss_fn, argnums=(0, 1)),
                tuple[
                    jax.Array, tuple[flax.core.scope.VariableDict, flax.core.scope.VariableDict]
                ],
            )
        else:
            self._greedy_actor_loss_and_grad_fn = astype_wrap(
                jax_value_and_grad(greedy_actor_loss_fn, argnums=(0, 1)),
                tuple[
                    jax.Array, tuple[flax.core.scope.VariableDict, flax.core.scope.VariableDict]
                ],
            )
            self._greedy_actor_loss_and_grad_fn = jax.vmap(
                self._greedy_actor_loss_and_grad_fn,
                in_axes=(0, 0, 0, None),
            )

        critic_loss_fn = jax.vmap(
            self._critic_loss_fn,
            in_axes=(0, 0, 0, 0, None, 0),
        )
        self._critic_loss_fn = filter_wrap(
            critic_loss_fn, functools.partial(jnp.sum, axis=0)
        )

        self._representation_optimizer = OPTIMIZERS[self._config.actor_optimizer](
            learning_rate=self._config.representation_learning_rate
        )
        self._decision_optimizer = OPTIMIZERS[self._config.actor_optimizer](
            learning_rate=self._config.decision_learning_rate
        )
        self._greedy_decision_optimizer = OPTIMIZERS[self._config.actor_optimizer](
            learning_rate=self._config.greedy_decision_learning_rate
        )
        self._critic_optimizer = optax.adam(
            learning_rate=self._config.critic_learning_rate
        )
        self._policy_critic_optimizer = optax.adam(
            learning_rate=self._config.policy_critic_learning_rate
        )

    @property
    def batch_size(self) -> int:
        return self._config.env_batch_size

    @property
    def use_all_data(self) -> bool:
        return True

    @classmethod
    @property
    def _reduce_decision_batch(cls) -> bool:
        return True

    def init(
        self, init_genotypes: flax.core.scope.VariableDict, random_key: RNGKey
    ) -> tuple[CCQDEmitterState, RNGKey]:
        representation_vars, decision_vars = self._actor.extract_vars(init_genotypes)
        representation_batch_sizes = self._actor.get_param_batch_sizes(representation_vars)
        decision_batch_sizes = self._actor.get_param_batch_sizes(decision_vars)
        assert decision_batch_sizes is not None
        if decision_batch_sizes != self._config.env_batch_size:
            _log.warning('decision_batch_sizes and env_batch_size are not the same.')

        assert representation_batch_sizes is not None
        assert representation_batch_sizes == decision_batch_sizes
        init_representation_indices = self._actor.get_init_representation_indices(
            self._config.env_batch_size, self._config.n_representation
        )
        greedy_actor_indices: jax.Array = jnp.searchsorted(
            init_representation_indices, jnp.arange(self._config.n_representation)
        )
        greedy_actors = tree_getitem(init_genotypes, greedy_actor_indices)
        representation_vars, greedy_decision_vars = self._actor.extract_vars(greedy_actors)
        assert (
            self._actor.get_param_batch_sizes(representation_vars)
            == self._config.n_representation
        )
        assert (
            self._actor.get_param_batch_sizes(greedy_decision_vars)
            == self._config.n_representation
        )

        target_representation_vars = tree_copy(representation_vars)
        representation_optimizer_state = self._representation_optimizer.init(representation_vars)

        target_greedy_decision_vars = tree_copy(greedy_decision_vars)
        greedy_decision_optimizer_state = self._greedy_decision_optimizer.init(
            greedy_decision_vars
        )

        random_key, subkey = jax.random.split(random_key)
        keys = jax.random.split(subkey, num=self._config.n_representation)
        fake_obs = jnp.zeros(shape=(self._config.n_representation, self._env.obs_size))
        fake_action = jnp.zeros(shape=(self._config.n_representation, self._env.action_size))
        critic_vars = jax.vmap(self._critic.init)(keys, fake_obs, fake_action)
        if self._policy_critic is not None:
            policy_critic_vars = jax.vmap(self._policy_critic.init)(
                keys, fake_obs, fake_action, greedy_decision_vars
            )
        else:
            policy_critic_vars = critic_vars

        target_critic_vars = tree_copy(critic_vars)
        critic_optimizer_state = self._critic_optimizer.init(critic_vars)
        target_policy_critic_vars = tree_copy(policy_critic_vars)
        policy_critic_optimizer_state = self._policy_critic_optimizer.init(policy_critic_vars)

        dummy_transition = buffer.QDTransition.init_dummy(
            observation_dim=self._env.obs_size,
            action_dim=self._env.action_size,
            descriptor_dim=self._env.bd_len,
        )
        replay_buffer = buffer.ReplayBuffer.init(
            buffer_size=self._config.replay_buffer_size,
            transition=dummy_transition,
        )

        repertoire_representation_indices = jnp.full(
            (self._config.n_centroids,), self._config.n_representation, dtype=jnp.int32
        )

        random_key, subkey = jax.random.split(random_key)
        emitter_state = self._StateType(
            random_key=subkey,

            representation_vars=representation_vars,
            target_representation_vars=target_representation_vars,
            representation_optimizer_state=representation_optimizer_state,

            greedy_decision_vars=greedy_decision_vars,
            target_greedy_decision_vars=target_greedy_decision_vars,
            greedy_decision_optimizer_state=greedy_decision_optimizer_state,

            critic_vars=critic_vars,
            target_critic_vars=target_critic_vars,
            critic_optimizer_state=critic_optimizer_state,

            policy_critic_vars=policy_critic_vars,
            target_policy_critic_vars=target_policy_critic_vars,
            policy_critic_optimizer_state=policy_critic_optimizer_state,

            offspring=init_genotypes,

            replay_buffer=replay_buffer,

            steps=jnp.array(0, dtype=jnp.int32),

            repertoire_representation_indices=repertoire_representation_indices,
            offspring_representation_indices=init_representation_indices,
        )
        return emitter_state, random_key

    @functools.partial(
        jax_jit,
        static_argnames=("self",),
    )
    def emit(
        self,
        repertoire: 'ExtendedMapElitesRepertoire',
        emitter_state: CCQDEmitterState,
        random_key: RNGKey,
    ) -> tuple[flax.core.scope.VariableDict, RNGKey]:
        return emitter_state.offspring, random_key

    @functools.partial(
        jax_jit,
        static_argnames=('self',),
    )
    def state_update(
        self,
        emitter_state: _TCCQDEmitterState,
        repertoire: 'ExtendedMapElitesRepertoire',
        genotypes: Optional[flax.core.scope.VariableDict],
        fitnesses: Optional[Fitness],
        descriptors: Optional[Descriptor],
        extra_scores: ExtraScores,
    ) -> _TCCQDEmitterState:
        assert hasattr(repertoire, 'latest_batch_of_indices')

        repertoire_representation_indices = emitter_state.repertoire_representation_indices
        offspring_representation_indices = emitter_state.offspring_representation_indices
        assert repertoire_representation_indices is not None
        assert offspring_representation_indices is not None

        repertoire_representation_indices = repertoire_representation_indices.at[
            repertoire.latest_batch_of_indices.squeeze(axis=-1)[self._offspring_slice]
        ].set(offspring_representation_indices)

        if self._config.replacement_threshold >= 0:
            n_decision = jnp.sum(
                repertoire_representation_indices == jnp.expand_dims(
                    jnp.arange(self._config.n_representation), axis=-1
                ),
                axis=-1,
            )
            replaced = n_decision <= self._config.replacement_threshold
            match self._config.replaced_by:
                case 'most':
                    replaced_by = jnp.argmax(n_decision)
                case _:
                    raise NotImplementedError
            assert len(replaced_by.shape) in (0, 1)
            new_tuple = (
                emitter_state.representation_vars,
                emitter_state.target_representation_vars,
                emitter_state.greedy_decision_vars,
                emitter_state.target_greedy_decision_vars,
                emitter_state.critic_vars,
                emitter_state.target_critic_vars,
                emitter_state.policy_critic_vars,
                emitter_state.target_policy_critic_vars,
            )
            new_tuple = tree_reversed_broadcasted_where_y(
                replaced,
                tree_getitem(new_tuple, replaced_by),
                new_tuple,
            )
            (
                representation_vars,
                target_representation_vars,
                greedy_decision_vars,
                target_greedy_decision_vars,
                critic_vars,
                target_critic_vars,
                policy_critic_vars,
                target_policy_critic_vars,
            ) = new_tuple
            del new_tuple

            random_key = emitter_state.random_key
            match self._config.representation_indices_reassignment:
                case 'Random':
                    if len(replaced_by.shape) == 0:
                        p = replaced.at[replaced_by].set(True)
                        p = p / p.sum()
                        random_key, subkey = jax.random.split(random_key)
                        repertoire_representation_indices = jnp.where(
                            repertoire_representation_indices == replaced_by,
                            jax.random.choice(
                                subkey,
                                self._config.n_representation,
                                shape=repertoire_representation_indices.shape,
                                p=p,
                            ),
                            repertoire_representation_indices,
                        )
                    else:
                        raise NotImplementedError
                case _:
                    raise NotImplementedError

            emitter_state = emitter_state.replace(
                random_key=random_key,
                representation_vars=representation_vars,
                target_representation_vars=target_representation_vars,
                greedy_decision_vars=greedy_decision_vars,
                target_greedy_decision_vars=target_greedy_decision_vars,
                critic_vars=critic_vars,
                target_critic_vars=target_critic_vars,
                policy_critic_vars=policy_critic_vars,
                target_policy_critic_vars=target_policy_critic_vars,
            )

        emitter_state = emitter_state.replace(
            repertoire_representation_indices=repertoire_representation_indices,
        )

        assert 'transitions' in extra_scores.keys()
        transitions = extra_scores['transitions']
        assert isinstance(transitions, buffer.QDTransition)

        replay_buffer = emitter_state.replay_buffer.insert(transitions)

        emitter_state = emitter_state.replace(
            replay_buffer=replay_buffer,
        )

        def scan_train(
            carry: _TCCQDEmitterState, unused: Any
        ) -> tuple[_TCCQDEmitterState, None]:
            emitter_state = carry
            emitter_state = self._train(
                emitter_state,
                repertoire,
            )
            return emitter_state, None

        emitter_state, _ = jax.lax.scan(
            scan_train,
            emitter_state,
            None,
            length=self._config.num_critic_training_steps,
            unroll=self._config.policy_delay,
        )

        emitter_state = self._select_and_emit(emitter_state, repertoire)

        return emitter_state

    def _get_decision_info(
        self,
        emitter_state: CCQDEmitterState,
        repertoire: 'ExtendedMapElitesRepertoire',
        random_key: Optional[jax.random.KeyArray],
        *indices_tuple: Optional[jax.Array],
    ) -> CCQDDecisionInfo:
        representation_indices = self._get_representation_indices(
            emitter_state, random_key, *indices_tuple
        )
        decision_info = CCQDDecisionInfo(representation_indices=representation_indices)
        return decision_info

    def _get_representation_indices(
        self,
        emitter_state: CCQDEmitterState,
        random_key: Optional[jax.random.KeyArray],
        *indices_tuple: Optional[jax.Array],
    ) -> Optional[jax.Array]:
        assert emitter_state.repertoire_representation_indices is not None
        representation_indices_list: list[jax.Array] = []
        for indices in indices_tuple:
            if indices is not None:
                if random_key is not None:
                    random_key, cond_key, sampling_key = jax.random.split(random_key, 3)
                    representation_indices_list.append(jnp.where(
                        jax.random.bernoulli(
                            cond_key,
                            p=self._config.p_concat_crossover,
                            shape=(indices.shape[0],),
                        ),
                        jax.random.randint(
                            sampling_key,
                            shape=(indices.shape[0],),
                            minval=0,
                            maxval=self._config.n_representation,
                            dtype=jnp.int32,
                        ),
                        emitter_state.repertoire_representation_indices[indices],
                    ))
                else:
                    representation_indices_list.append(
                        emitter_state.repertoire_representation_indices[indices]
                    )
            else:
                representation_indices_list.append(
                    jnp.arange(self._config.n_representation, dtype=jnp.int32)
                )
        representation_indices = jnp.concatenate(representation_indices_list)
        return representation_indices

    @functools.partial(jax_jit, static_argnames=('self',))
    def _train(
        self,
        emitter_state: _TCCQDEmitterState,
        repertoire: 'ExtendedMapElitesRepertoire',
    ) -> _TCCQDEmitterState:
        random_key = emitter_state.random_key
        transitions, random_key = emitter_state.replay_buffer.sample(
            random_key, sample_size=self._config.training_batch_size,
        )
        transitions = self._transform_transitions(emitter_state, transitions)

        (
            policy_critic_optimizer_state,
            policy_critic_vars,
            target_policy_critic_vars,
        ) = (
            emitter_state.policy_critic_optimizer_state,
            emitter_state.policy_critic_vars,
            emitter_state.target_policy_critic_vars,
        )

        if self._policy_critic is not None:
            selected_indices, random_key = SELECTORS[
                self._config.decision_sampling_updating_policy_critic
            ](
                repertoire, random_key, self._config.num_decision_updating_policy_critic
            )
            vars = tree_getitem(repertoire.genotypes, selected_indices)
            vars = astype(vars, flax.core.scope.VariableDict)
            _, decision_vars = self._actor.extract_vars(vars)

            random_key, subkey = jax.random.split(random_key)
            (
                policy_critic_optimizer_state,
                policy_critic_vars,
                target_policy_critic_vars,
                random_key,
            ) = self._update_policy_critic(
                policy_critic_vars,
                target_policy_critic_vars,
                (
                    emitter_state.target_representation_vars
                    if self._config.updating_policy_critic_with_target_representation
                    else emitter_state.representation_vars
                ),
                decision_vars,
                self._get_decision_info(emitter_state, repertoire, subkey, selected_indices),
                policy_critic_optimizer_state,
                transitions,
                random_key,
            )

        (
            critic_optimizer_state,
            critic_vars,
            target_critic_vars,
            random_key,
        ) = self._update_critic(
            critic_vars=emitter_state.critic_vars,
            target_critic_vars=emitter_state.target_critic_vars,
            target_representation_vars=emitter_state.target_representation_vars,
            target_greedy_decision_vars=emitter_state.target_greedy_decision_vars,
            critic_optimizer_state=emitter_state.critic_optimizer_state,
            transitions=transitions,
            random_key=random_key,
        )

        if self._policy_critic is None:
            (
                policy_critic_optimizer_state,
                policy_critic_vars,
                target_policy_critic_vars,
            ) = (
                critic_optimizer_state,
                critic_vars,
                target_critic_vars,
            )

        def cond_update_actors(
            emitter_state: _TCCQDEmitterState,
            critic_vars: flax.core.scope.VariableDict,
            policy_critic_vars: flax.core.scope.VariableDict,
            transitions: buffer.QDTransition,
            random_key: RNGKey,
            repertoire: 'ExtendedMapElitesRepertoire',
        ) -> tuple[
            optax.OptState,
            optax.OptState,
            flax.core.scope.VariableDict,
            flax.core.scope.VariableDict,
            flax.core.scope.VariableDict,
            flax.core.scope.VariableDict,
            RNGKey,
        ]:
            representation_vars = emitter_state.representation_vars
            representation_optimizer_state = emitter_state.representation_optimizer_state

            selected_indices, random_key = SELECTORS[
                self._config.decision_sampling_updating_representation
            ](
                repertoire, random_key, self._config.num_decision_updating_representation
            )
            vars = tree_getitem(repertoire.genotypes, selected_indices)
            vars = astype(vars, flax.core.scope.VariableDict)
            _, decision_vars = self._actor.extract_vars(vars)
            random_key, subkey = jax.random.split(random_key)
            decision_info = self._get_decision_info(
                emitter_state, repertoire, subkey, selected_indices
            )

            if not self._config.batch_updating_representation:
                def scan_update_representation(
                    carry: tuple[optax.OptState, flax.core.scope.VariableDict, RNGKey],
                    decision_with_info: tuple[
                        flax.core.scope.VariableDict, CCQDDecisionInfo
                    ]
                ) -> tuple[tuple[optax.OptState, flax.core.scope.VariableDict, RNGKey], None]:
                    representation_optimizer_state, representation_vars, random_key = carry
                    decision_vars, decision_info = tree_getitem(
                        decision_with_info, jnp.newaxis
                    )
                    carry = self._update_representation(
                        representation_vars,
                        decision_vars,
                        decision_info,
                        representation_optimizer_state,
                        policy_critic_vars,
                        transitions,
                        random_key,
                    )
                    return carry, None
                # end of 'def scan_update_representation'

                (representation_optimizer_state, representation_vars, random_key), _ = jax.lax.scan(
                    scan_update_representation,
                    (representation_optimizer_state, representation_vars, random_key),
                    (decision_vars, decision_info),
                )

                decision_vars = None
            # end of 'if not self._config.batch_updating_representation'

            (
                representation_optimizer_state,
                greedy_decision_optimizer_state,
                representation_vars,
                greedy_decision_vars,
                target_representation_vars,
                target_greedy_decision_vars,
                random_key,
            ) = self._update_actors(
                representation_vars,
                decision_vars,
                decision_info,
                emitter_state.greedy_decision_vars,
                self._get_decision_info(emitter_state, repertoire, None, None),
                representation_optimizer_state,
                emitter_state.greedy_decision_optimizer_state,
                emitter_state.target_representation_vars,
                emitter_state.target_greedy_decision_vars,
                critic_vars,
                policy_critic_vars,
                transitions,
                random_key,
            )

            return (
                representation_optimizer_state,
                greedy_decision_optimizer_state,
                representation_vars,
                greedy_decision_vars,
                target_representation_vars,
                target_greedy_decision_vars,
                random_key,
            )
        # end of 'def cond_update_actors'

        (
            representation_optimizer_state,
            greedy_decision_optimizer_state,
            representation_vars,
            greedy_decision_vars,
            target_representation_vars,
            target_greedy_decision_vars,
            random_key,
        ) = jax.lax.cond(
            emitter_state.steps % self._config.policy_delay == 0,
            lambda x: cond_update_actors(*x),
            lambda _: (
                emitter_state.representation_optimizer_state,
                emitter_state.greedy_decision_optimizer_state,
                emitter_state.representation_vars,
                emitter_state.greedy_decision_vars,
                emitter_state.target_representation_vars,
                emitter_state.target_greedy_decision_vars,
                random_key,
            ),
            operand=(
                emitter_state,
                (
                    (
                        target_critic_vars
                        if self._config.updating_actors_with_new_critics
                        else emitter_state.target_critic_vars
                    ) if self._config.updating_actors_with_target_critics else (
                        critic_vars
                        if self._config.updating_actors_with_new_critics
                        else emitter_state.critic_vars
                    )
                ),
                (
                    policy_critic_vars
                    if self._config.updating_actors_with_new_critics
                    else emitter_state.policy_critic_vars
                ),
                transitions,
                random_key,
                repertoire,
            ),
        )

        emitter_state = emitter_state.replace(
            random_key=random_key,

            representation_vars=representation_vars,
            target_representation_vars=target_representation_vars,
            representation_optimizer_state=representation_optimizer_state,

            greedy_decision_vars=greedy_decision_vars,
            target_greedy_decision_vars=target_greedy_decision_vars,
            greedy_decision_optimizer_state=greedy_decision_optimizer_state,

            critic_vars=critic_vars,
            target_critic_vars=target_critic_vars,
            critic_optimizer_state=critic_optimizer_state,

            policy_critic_vars=policy_critic_vars,
            target_policy_critic_vars=target_policy_critic_vars,
            policy_critic_optimizer_state=policy_critic_optimizer_state,
        )

        return emitter_state

    @functools.partial(jax_jit, static_argnames=('self',))
    def _update_policy_critic(
        self,
        policy_critic_vars: flax.core.scope.VariableDict,
        target_policy_critic_vars: flax.core.scope.VariableDict,
        used_representation_vars: flax.core.scope.VariableDict,
        decision_vars: flax.core.scope.VariableDict,
        decision_info: CCQDDecisionInfo,
        policy_critic_optimizer_state: optax.OptState,
        transitions: buffer.QDTransition,
        random_key: RNGKey,
    ) -> tuple[
        optax.OptState,
        flax.core.scope.VariableDict,
        flax.core.scope.VariableDict,
        RNGKey,
    ]:
        random_key, subkey = jax.random.split(random_key)
        (
            policy_critic_loss, policy_critic_gradient
        ) = jax_value_and_grad(self._policy_critic_loss_fn)(
            policy_critic_vars,
            used_representation_vars,
            decision_vars,
            target_policy_critic_vars,
            transitions,
            subkey,
            decision_info.representation_indices,
        )
        policy_critic_updates, policy_critic_optimizer_state = self._policy_critic_optimizer.update(
            policy_critic_gradient, policy_critic_optimizer_state
        )

        policy_critic_vars_ = optax.apply_updates(policy_critic_vars, policy_critic_updates)
        policy_critic_vars = astype(policy_critic_vars_, flax.core.scope.VariableDict)
        del policy_critic_vars_

        target_policy_critic_vars = jax.tree_map(
            lambda x1, x2: (1.0 - self._config.soft_tau_update) * x1
            + self._config.soft_tau_update * x2,
            target_policy_critic_vars,
            policy_critic_vars,
        )

        return (
            policy_critic_optimizer_state, policy_critic_vars, target_policy_critic_vars, random_key
        )

    @functools.partial(jax_jit, static_argnames=('self',))
    def _update_critic(
        self,
        critic_vars: flax.core.scope.VariableDict,
        target_critic_vars: flax.core.scope.VariableDict,
        target_representation_vars: flax.core.scope.VariableDict,
        target_greedy_decision_vars: flax.core.scope.VariableDict,
        critic_optimizer_state: optax.OptState,
        transitions: buffer.QDTransition,
        random_key: RNGKey,
    ) -> tuple[
        optax.OptState,
        flax.core.scope.VariableDict,
        flax.core.scope.VariableDict,
        RNGKey,
    ]:
        random_key, subkey = jax.random.split(random_key)
        target_representation_batch_size = self._actor.get_param_batch_sizes(
            target_representation_vars
        )
        if target_representation_batch_size is not None:
            subkey = jax.random.split(subkey, target_representation_batch_size)
        critic_loss, critic_gradient = jax_value_and_grad(self._critic_loss_fn)(
            critic_vars,
            target_representation_vars,
            target_greedy_decision_vars,
            target_critic_vars,
            transitions,
            subkey,
        )
        critic_updates, critic_optimizer_state = self._critic_optimizer.update(
            critic_gradient, critic_optimizer_state
        )

        critic_vars_ = optax.apply_updates(critic_vars, critic_updates)
        critic_vars = astype(critic_vars_, flax.core.scope.VariableDict)
        del critic_vars_

        target_critic_vars = jax.tree_map(
            lambda x1, x2: (1.0 - self._config.soft_tau_update) * x1
            + self._config.soft_tau_update * x2,
            target_critic_vars,
            critic_vars,
        )

        return critic_optimizer_state, critic_vars, target_critic_vars, random_key

    @functools.partial(jax_jit, static_argnames=('self',))
    def _update_actors(
        self,
        representation_vars: flax.core.scope.VariableDict,
        decision_vars: Optional[flax.core.scope.VariableDict],
        decision_info: CCQDDecisionInfo,
        greedy_decision_vars: flax.core.scope.VariableDict,
        greedy_decision_info: CCQDDecisionInfo,
        representation_optimizer_state: optax.OptState,
        greedy_decision_optimizer_state: optax.OptState,
        target_representation_vars: flax.core.scope.VariableDict,
        target_greedy_decision_vars: flax.core.scope.VariableDict,
        critic_vars: flax.core.scope.VariableDict,
        policy_critic_vars: flax.core.scope.VariableDict,
        transitions: buffer.QDTransition,
        random_key: RNGKey,
    ) -> tuple[
        optax.OptState,
        optax.OptState,
        flax.core.scope.VariableDict,
        flax.core.scope.VariableDict,
        flax.core.scope.VariableDict,
        flax.core.scope.VariableDict,
        RNGKey,
    ]:
        (
            greedy_actor_loss, (greedy_decision_gradient, greedy_representation_gradient)
        ) = self._greedy_actor_loss_and_grad_fn(
            greedy_decision_vars,
            representation_vars,
            critic_vars,
            transitions,
        )

        (
            greedy_decision_updates,
            greedy_decision_optimizer_state,
        ) = self._greedy_decision_optimizer.update(
            greedy_decision_gradient,
            greedy_decision_optimizer_state,
        )
        greedy_decision_vars_ = optax.apply_updates(
            greedy_decision_vars, greedy_decision_updates
        )
        greedy_decision_vars = astype(greedy_decision_vars_, flax.core.scope.VariableDict)
        del greedy_decision_vars_

        if decision_vars is not None:
            representation_loss, representation_gradient = jax_value_and_grad(self._actor_loss_fn)(
                representation_vars,
                decision_vars,
                policy_critic_vars,
                transitions,
                decision_info.representation_indices,
            )
            representation_gradient = jax.tree_map(
                lambda x1, x2: self._config.decision_factor * x1 + x2,
                representation_gradient,
                greedy_representation_gradient,
            )
        else:
            representation_gradient = greedy_representation_gradient

        (
            representation_updates,
            representation_optimizer_state,
        ) = self._representation_optimizer.update(
            representation_gradient, representation_optimizer_state
        )
        representation_vars_ = optax.apply_updates(representation_vars, representation_updates)
        representation_vars = astype(representation_vars_, flax.core.scope.VariableDict)
        del representation_vars_

        target_representation_vars = jax.tree_map(
            lambda x1, x2: (1.0 - self._config.soft_tau_update) * x1
            + self._config.soft_tau_update * x2,
            target_representation_vars,
            representation_vars,
        )

        target_greedy_decision_vars = jax.tree_map(
            lambda x1, x2: (1.0 - self._config.soft_tau_update) * x1
            + self._config.soft_tau_update * x2,
            target_greedy_decision_vars,
            greedy_decision_vars,
        )

        return (
            representation_optimizer_state,
            greedy_decision_optimizer_state,
            representation_vars,
            greedy_decision_vars,
            target_representation_vars,
            target_greedy_decision_vars,
            random_key,
        )

    @functools.partial(jax_jit, static_argnames=('self',))
    def _update_representation(
        self,
        representation_vars: flax.core.scope.VariableDict,
        decision_vars: flax.core.scope.VariableDict,
        decision_info: CCQDDecisionInfo,
        representation_optimizer_state: optax.OptState,
        policy_critic_vars: flax.core.scope.VariableDict,
        transitions: buffer.QDTransition,
        random_key: RNGKey,
    ) -> tuple[optax.OptState, flax.core.scope.VariableDict, RNGKey]:
        representation_loss, representation_gradient = jax_value_and_grad(self._actor_loss_fn)(
            representation_vars,
            decision_vars,
            policy_critic_vars,
            transitions,
            decision_info.representation_indices,
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

        return representation_optimizer_state, representation_vars, random_key

    def _select_and_emit(
        self,
        emitter_state: _TCCQDEmitterState,
        repertoire: 'ExtendedMapElitesRepertoire',
    ) -> _TCCQDEmitterState:
        random_key = emitter_state.random_key
        (
            ea_vars_list,
            decision_vars_pg,
            selected_indices_ea,
            selected_indices_pg,
            random_key,
        ) = self._select(emitter_state, repertoire, random_key)
        emitter_state = emitter_state.replace(random_key=random_key)

        emitter_state = self._emit(
            emitter_state,
            repertoire,
            ea_vars_list,
            selected_indices_ea,
            selected_indices_pg,
            decision_vars_pg,
        )

        return emitter_state

    def _select(
        self,
        emitter_state: CCQDEmitterState,
        repertoire: 'ExtendedMapElitesRepertoire',
        random_key: RNGKey,
    ) -> tuple[
        list[flax.core.scope.VariableDict],
        flax.core.scope.VariableDict,
        jax.Array,
        jax.Array,
        RNGKey,
    ]:
        env_batch_size_pg_all = round(self._config.env_batch_size * self._config.proportion_pg)
        representation_batch_size = self._actor.get_param_batch_sizes(
            emitter_state.representation_vars
        )
        if representation_batch_size is not None:
            env_batch_size_pg = env_batch_size_pg_all - representation_batch_size
        else:
            env_batch_size_pg = env_batch_size_pg_all - 1
        env_batch_size_ea = self._config.env_batch_size - env_batch_size_pg_all

        selected_indices_pg, random_key = SELECTORS[self._config.pg_selection](
            repertoire, random_key, env_batch_size_pg,
        )
        vars_ = tree_getitem(repertoire.genotypes, selected_indices_pg)
        vars = astype(vars_, flax.core.scope.VariableDict)
        del vars_
        _, decision_vars_pg = self._actor.extract_vars(vars)

        selected_indices_ea = None
        ea_vars_list = []
        for selection in self._config.ea_selection:
            selected_indices, random_key = SELECTORS[selection](
                repertoire, random_key, env_batch_size_ea
            )
            if selected_indices_ea is None:
                selected_indices_ea = selected_indices
            vars_ = tree_getitem(repertoire.genotypes, selected_indices)
            vars = astype(vars_, flax.core.scope.VariableDict)
            del vars_
            ea_vars_list.append(vars)

        assert all(
            self._actor.get_decision_param_batch_sizes(ea_vars) == env_batch_size_ea
            for ea_vars in ea_vars_list
        )
        assert self._actor.get_param_batch_sizes(decision_vars_pg) == env_batch_size_pg

        assert selected_indices_ea is not None
        assert selected_indices_pg is not None
        return (
            ea_vars_list, decision_vars_pg, selected_indices_ea, selected_indices_pg, random_key
        )

    @functools.partial(
        jax_jit,
        static_argnames=("self",),
    )
    def _emit(
        self,
        emitter_state: _TCCQDEmitterState,
        repertoire: 'ExtendedMapElitesRepertoire',
        ea_vars_list: Sequence[flax.core.scope.VariableDict],
        selected_indices_ea: jax.Array,
        selected_indices_pg: jax.Array,
        decision_vars_pg: flax.core.scope.VariableDict,
    ) -> _TCCQDEmitterState:
        random_key = emitter_state.random_key

        env_batch_size_pg_all = round(self._config.env_batch_size * self._config.proportion_pg)
        representation_batch_size = self._actor.get_param_batch_sizes(
            emitter_state.representation_vars
        )
        if representation_batch_size is not None:
            env_batch_size_pg = env_batch_size_pg_all - representation_batch_size
        else:
            env_batch_size_pg = env_batch_size_pg_all - 1
        env_batch_size_ea = self._config.env_batch_size - env_batch_size_pg_all

        random_key, emit_pg_key, representation_indices_key = jax.random.split(random_key, 3)
        decision_info_pg = self._get_decision_info(
            emitter_state,
            repertoire,
            representation_indices_key,
            selected_indices_pg,
        )
        decision_vars_pg = self._emit_decision_pg(
            emitter_state,
            emitter_state.representation_vars,
            decision_vars_pg,
            decision_info_pg,
            emitter_state.policy_critic_vars,
            emitter_state.replay_buffer,
            emit_pg_key,
        )
        assert self._actor.get_param_batch_sizes(decision_vars_pg) == env_batch_size_pg

        greedy_decision_vars = self._emit_greedy_decision(emitter_state.greedy_decision_vars)

        random_key, subkey = jax.random.split(random_key)
        ea_vars = self._emit_decision_ea(
            ea_vars_list, subkey,
        )
        assert self._actor.get_decision_param_batch_sizes(ea_vars) == env_batch_size_ea

        decision_vars: flax.core.scope.VariableDict = jax.tree_map(
            lambda x, y: jnp.concatenate((x, y), axis=0),
            decision_vars_pg,
            greedy_decision_vars,
        )
        random_key, subkey = jax.random.split(random_key)
        decision_info = self._get_decision_info(
            emitter_state,
            repertoire,
            subkey,
            None,
        )
        representation_indices = tree_concatenate(
            decision_info_pg.representation_indices, decision_info.representation_indices
        )
        vars = self._actor.make_genotypes(
            self._actor.make_vars(
                emitter_state.representation_vars,
                decision_vars,
                representation_indices=representation_indices,
            )
        )
        vars = jax.tree_map(
            lambda x, y: jnp.concatenate((x, y), axis=0),
            vars,
            ea_vars,
        )
        assert self._actor.get_decision_param_batch_sizes(vars) == self._config.env_batch_size
        decision_info_ea = self._get_decision_info(
            emitter_state,
            repertoire,
            None,
            selected_indices_ea,
        )
        representation_indices = tree_concatenate(
            representation_indices, decision_info_ea.representation_indices
        )

        if representation_indices is not None:
            assert len(representation_indices) == self._config.env_batch_size
        emitter_state = emitter_state.replace(
            random_key=random_key,
            offspring=vars,
            offspring_representation_indices=representation_indices,
        )

        return emitter_state

    def _emit_decision_ea(
        self,
        ea_vars_list: Sequence[flax.core.scope.VariableDict],
        random_key: RNGKey,
    ) -> flax.core.scope.VariableDict:
        if self._config.crossover is not None:
            ea_vars_, random_key = OPERATORS[self._config.crossover](
                *ea_vars_list,
                random_key=random_key,
                **self._config.crossover_params,
            )
            ea_vars = astype(ea_vars_, flax.core.scope.VariableDict)
            del ea_vars_
            ea_vars_list = [ea_vars]

        if self._config.mutation is not None:
            ea_vars_, random_key = OPERATORS[self._config.mutation](
                *ea_vars_list,
                random_key=random_key,
                **self._config.mutation_params,
            )
            ea_vars = astype(ea_vars_, flax.core.scope.VariableDict)
            del ea_vars_
            ea_vars_list = [ea_vars]

        assert len(ea_vars_list) == 1
        ea_vars = ea_vars_list[0]
        return ea_vars

    def _emit_greedy_decision(
        self,
        greedy_decision_vars: flax.core.scope.VariableDict,
    ) -> flax.core.scope.VariableDict:
        greedy_decision_batch_size = self._actor.get_param_batch_sizes(greedy_decision_vars)
        if greedy_decision_batch_size is None:
            greedy_decision_vars = tree_getitem(greedy_decision_vars, jnp.newaxis)
            greedy_decision_batch_size = self._actor.get_param_batch_sizes(greedy_decision_vars)
        assert greedy_decision_batch_size is not None
        return greedy_decision_vars

    @functools.partial(jax_jit, static_argnames=('self',))
    def _emit_decision_pg(
        self,
        emitter_state: CCQDEmitterState,
        representation_vars: flax.core.scope.VariableDict,
        decision_vars: flax.core.scope.VariableDict,
        decision_info: CCQDDecisionInfo,
        policy_critic_vars: flax.core.scope.VariableDict,
        replay_buffer: buffer.ReplayBuffer,
        random_key: RNGKey,
    ) -> flax.core.scope.VariableDict:
        decision_optimizer_state = self._decision_optimizer.init(decision_vars)

        def scan_train_decision(
            carry: tuple[
                flax.core.scope.VariableDict,
                flax.core.scope.VariableDict,
                CCQDDecisionInfo,
                optax.OptState,
                flax.core.scope.VariableDict,
                buffer.ReplayBuffer,
            ],
            random_key: RNGKey,
        ) -> tuple[tuple[
            flax.core.scope.VariableDict,
            flax.core.scope.VariableDict,
            CCQDDecisionInfo,
            optax.OptState,
            flax.core.scope.VariableDict,
            buffer.ReplayBuffer,
        ], None]:
            (
                representation_vars,
                decision_vars,
                decision_info,
                decision_optimizer_state,
                policy_critic_vars,
                replay_buffer,
            ) = carry

            (
                decision_vars,
                decision_optimizer_state,
            ) = self._train_decision(
                emitter_state,
                representation_vars,
                decision_vars,
                decision_info,
                decision_optimizer_state,
                policy_critic_vars,
                replay_buffer,
                random_key,
            )

            return (
                representation_vars,
                decision_vars,
                decision_info,
                decision_optimizer_state,
                policy_critic_vars,
                replay_buffer,
            ), None

        keys = jax.random.split(random_key, self._config.num_pg_training_steps)

        (_, decision_vars, _, _, _, _), _ = jax.lax.scan(
            scan_train_decision,
            (
                representation_vars,
                decision_vars,
                decision_info,
                decision_optimizer_state,
                policy_critic_vars,
                replay_buffer,
            ),
            keys,
            length=self._config.num_pg_training_steps,
        )

        return decision_vars

    @functools.partial(jax_jit, static_argnames=('self',))
    def _train_decision(
        self,
        emitter_state: CCQDEmitterState,
        representation_vars: flax.core.scope.VariableDict,
        decision_vars: flax.core.scope.VariableDict,
        decision_info: CCQDDecisionInfo,
        decision_optimizer_state: optax.OptState,
        policy_critic_vars: flax.core.scope.VariableDict,
        replay_buffer: buffer.ReplayBuffer,
        random_key: RNGKey,
    ) -> tuple[flax.core.scope.VariableDict, optax.OptState]:
        transitions, random_key = replay_buffer.sample(
            random_key, sample_size=self._config.training_decision_batch_size,
        )
        transitions = self._transform_transitions(emitter_state, transitions)

        decision_optimizer_state, decision_vars = self._update_decision(
            policy_critic_vars=policy_critic_vars,
            decision_optimizer_state=decision_optimizer_state,
            representation_vars=representation_vars,
            decision_vars=decision_vars,
            decision_info=decision_info,
            transitions=transitions,
        )

        return decision_vars, decision_optimizer_state

    @functools.partial(jax_jit, static_argnames=('self',))
    def _update_decision(
        self,
        policy_critic_vars: flax.core.scope.VariableDict,
        decision_optimizer_state: optax.OptState,
        representation_vars: flax.core.scope.VariableDict,
        decision_vars: flax.core.scope.VariableDict,
        decision_info: CCQDDecisionInfo,
        transitions: buffer.QDTransition,
    ) -> tuple[optax.OptState, flax.core.scope.VariableDict]:
        actor_loss, decision_gradient = jax_value_and_grad(
            self._actor_loss_fn,
            argnums=1,
        )(
            representation_vars,
            decision_vars,
            policy_critic_vars,
            transitions,
            decision_info.representation_indices,
        )

        (
            decision_updates,
            decision_optimizer_state,
        ) = self._decision_optimizer.update(decision_gradient, decision_optimizer_state)
        decision_vars_ = optax.apply_updates(decision_vars, decision_updates)
        decision_vars = astype(decision_vars_, flax.core.scope.VariableDict)
        del decision_vars_

        return decision_optimizer_state, decision_vars

    def _transform_transitions(
        self,
        emitter_state: CCQDEmitterState,
        transitions: buffer.QDTransition,
    ) -> buffer.QDTransition:
        return transitions
