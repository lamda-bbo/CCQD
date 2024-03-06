import jax
import jax.numpy as jnp
import flax.core
import brax.jumpy as jp
import qdax.baselines.sac
import qdax.baselines.sac_pbt
import qdax.core.containers.archive
import qdax.core.emitters.emitter
import qdax.core.emitters.standard_emitters
import qdax.core.emitters.pbt_me_emitter
import qdax.core.emitters.pga_me_emitter
import qdax.core.emitters.multi_emitter
import qdax.core.emitters.mutation_operators
import qdax.core.emitters.pbt_variation_operators
import qdax.environments
from qdax.core.neuroevolution.buffers import buffer
import qdax.utils.metrics
import qdax.utils.plotting
from qdax.types import Metrics, Params

import gym.spaces
import gymnasium.spaces

import hydra.utils
from omegaconf import OmegaConf
import logging
import wandb
import matplotlib
import matplotlib.figure
import matplotlib.pyplot as plt
import pickle
import cloudpickle
import time
import os
import lzma
import lz4.frame
import tqdm
import tqdm.contrib.logging
import functools
import itertools
from collections.abc import Iterable, Callable
from typing import Optional, Any, Final, TYPE_CHECKING

from .map_elites import (
    ExtendedMAPElites,
    ExtendedMapElitesRepertoire,
    compute_cvt_centroids,
)
from .emitters import (
    PBTEmitter,
    PGAMEEmitter,
    QualityPGEmitterConfig,
    DiversityPGEmitterConfig,
    QDPGEmitterConfig,
    QDPGEmitter,
    OMGMEGARLEmitter,
    CCQDEmitterState,
    CCQDEmitter,
    DQNMEEmitter,
    CCDQNMEEmitter,
)
from .neuroevolution import Actor, CNNActor, ExtendedQDTransition
from .env import REWARD_OFFSET, QDEnv, GymEnv
from .metrics import CSVLogger, qd_metrics, get_logged_metrics
from .utils import (
    DERIVED_INT,
    DERIVED_TUPLE_INT,
    activation,
    astype,
    class_partial,
    uninterrupted,
    tree_repeat,
    jax_jit
)
from .config import (
    PBTMESacConfig,
    MEConfig,
    PGAMEConfig,
    QDPGConfig,
    OMGMEGARLConfig,
    CCQDConfig,
    CNNAlgoConfig,
    DQNMEConfig,
    CCDQNMEConfig,
)

if TYPE_CHECKING:
    from .config import Config


_log = logging.getLogger(__name__)


class Main:

    def __init__(self, cfg: 'Config'):
        self.cfg: Final['Config'] = cfg

        if wandb.run is not None:
            self.resumed: bool = wandb.run.resumed
        else:
            self.resumed = os.path.exists(self.cfg.config_filename)

        assert not os.path.exists(self.cfg.checkpoint_filename)
        if self.resumed:
            _log.info('Resuming from a previous run...')
            self.check_config()
        if not self.resumed:
            _log.info('Setting up a fresh run...')
            assert not os.path.exists(self.cfg.config_filename)
            assert not os.path.exists(self.cfg.metrics_filename)
            assert not os.path.exists(self.cfg.compressed_checkpoint_filename)
            self.save_config()

        self.random_key = jax.random.PRNGKey(self.cfg.seed)

        self.random_key, subkey = jax.random.split(self.random_key)
        if hasattr(self.cfg.env, 'package'):
            self.env = GymEnv(subkey, self.cfg)
            qd_offset = REWARD_OFFSET[self.cfg.env.name.replace('--', '/')]
        else:
            self.env = QDEnv(subkey, self.cfg)
            qd_offset = (
                qdax.environments.reward_offset[self.cfg.env.name] * self.cfg.env.episode_len
            )
        _log.info('obs_size = %s, action_size = %s', self.env.obs_size, self.env.action_size)

        metrics_fn = functools.partial(
            qd_metrics,
            qd_offset=qd_offset,
        )

        self.random_key, subkey = jax.random.split(self.random_key)
        if self.cfg.algo.init is not None:

            obs_space = self.env.obs_space
            match obs_space:
                case gym.spaces.Box() | gymnasium.spaces.Box():
                    cnn_input_shape = obs_space.shape
                case _:
                    raise NotImplementedError

            action_space = self.env.action_space
            match action_space:
                case gym.spaces.Box() | gymnasium.spaces.Box():
                    output_size = self.env.action_size
                case gym.spaces.Discrete() | gymnasium.spaces.Discrete():
                    output_size = int(action_space.n)
                case _:
                    raise NotImplementedError

            if hasattr(self.cfg.algo, 'conv_activation'):
                algo_cfg = astype(self.cfg.algo, CNNAlgoConfig)

                assert algo_cfg.conv_padding == 'SAME' or algo_cfg.conv_padding == 'VALID'

                actor = CNNActor(
                    representation_cnn_input_shape=cnn_input_shape,
                    representation_conv_features=algo_cfg.representation_conv_features,
                    representation_kernel_sizes=algo_cfg.representation_kernel_sizes,
                    representation_hidden_layer_sizes=algo_cfg.representation_hidden_layer_sizes,
                    representation_conv_strides=algo_cfg.representation_conv_strides,
                    decision_cnn_input_shape=algo_cfg.decision_cnn_input_shape,
                    decision_conv_features=algo_cfg.decision_conv_features,
                    decision_kernel_sizes=algo_cfg.decision_kernel_sizes,
                    decision_hidden_layer_sizes=algo_cfg.decision_hidden_layer_sizes,
                    decision_conv_strides=algo_cfg.decision_conv_strides,
                    action_size=output_size,
                    conv_activation=activation(algo_cfg.conv_activation),
                    conv_padding=algo_cfg.conv_padding,
                    activation=activation(algo_cfg.activation),
                    final_activation=activation(algo_cfg.final_activation),
                )
            else:
                actor = Actor(
                    representation_hidden_layer_sizes=self.cfg.algo.representation_hidden_layer_sizes,  # noqa: E501
                    decision_hidden_layer_sizes=self.cfg.algo.decision_hidden_layer_sizes,
                    action_size=output_size,
                    activation=activation(self.cfg.algo.activation),
                    final_activation=activation(self.cfg.algo.final_activation),
                )
            print(actor.tabulate(
                jax.random.PRNGKey(0), jnp.zeros((self.cfg.algo.n_init_vars, self.env.obs_size))
            ))
            match self.cfg.algo.init:
                case 'classic':
                    init_fn = actor.init_separately
                case 'representation':
                    algo_cfg = astype(self.cfg.algo, CCQDConfig)
                    init_fn = functools.partial(
                        actor.init_all,
                        n_representation=algo_cfg.n_representation,
                    )
                case _:
                    raise NotImplementedError
            init_representation_vars, init_decision_vars = init_fn(
                subkey, self.env.obs_size, self.cfg.algo.n_init_vars
            )
            init_vars = actor.make_vars(init_representation_vars, init_decision_vars)
            init_vars = actor.make_genotypes(init_vars)

            match (obs_space, action_space):
                case (
                    gym.spaces.Box() | gymnasium.spaces.Box(),
                    gym.spaces.Box() | gymnasium.spaces.Box(),
                ):
                    def _select_action_fn(
                        genotypes: Params, obs: jax.Array | jp.ndarray
                    ) -> jax.Array:
                        genotypes_ = astype(genotypes, flax.core.scope.VariableDict)
                        del genotypes
                        action = actor.apply_all(genotypes_, obs)
                        assert isinstance(action, jax.Array)
                        return action
                case (
                    gym.spaces.Box() | gymnasium.spaces.Box(),
                    gym.spaces.Discrete() | gymnasium.spaces.Discrete(),
                ):
                    def _select_action_fn(  # noqa: F811
                        genotypes: Params, obs: jax.Array | jp.ndarray
                    ) -> jax.Array:
                        genotypes_ = astype(genotypes, flax.core.scope.VariableDict)
                        del genotypes
                        action = actor.apply_all(genotypes_, obs)
                        assert isinstance(action, jax.Array)
                        action = jnp.argmax(action, axis=-1, keepdims=True)
                        return action
                case _:
                    raise NotImplementedError

            scoring_fn = self.env.get_scoring_fn(
                _select_action_fn,
                buffer.QDTransition
                if self.cfg.algo.name not in ['OMG-MEGA-RL']
                else ExtendedQDTransition,
            )
            make_genotypes_fn = actor.make_genotypes

            match self.cfg.algo.name:
                case 'ME':
                    algo_cfg = astype(self.cfg.algo, MEConfig)
                    self.map_elites = ExtendedMAPElites(
                        scoring_function=scoring_fn,
                        emitter=qdax.core.emitters.standard_emitters.MixingEmitter(
                            mutation_fn=NotImplemented,
                            variation_fn=functools.partial(
                                qdax.core.emitters.mutation_operators.isoline_variation,
                                iso_sigma=algo_cfg.iso_sigma,
                                line_sigma=algo_cfg.line_sigma,
                            ),
                            variation_percentage=1.0,
                            batch_size=self.cfg.algo.env_batch_size,
                        ),
                        metrics_function=metrics_fn,
                    )
                case 'PGA-ME':
                    algo_cfg = astype(self.cfg.algo, PGAMEConfig)
                    assert isinstance(self.env, QDEnv)
                    self.map_elites = ExtendedMAPElites(
                        scoring_function=scoring_fn,
                        emitter=PGAMEEmitter(
                            algo_cfg,
                            actor,
                            self.env.env,
                            functools.partial(
                                qdax.core.emitters.mutation_operators.isoline_variation,
                                iso_sigma=algo_cfg.iso_sigma,
                                line_sigma=algo_cfg.line_sigma,
                            ),
                        ),
                        metrics_function=metrics_fn,
                    )
                case 'QD-PG':
                    algo_cfg = astype(self.cfg.algo, QDPGConfig)
                    assert isinstance(self.env, QDEnv)
                    qpg_env_batch_size = round(algo_cfg.env_batch_size * algo_cfg.proportion_qpg)
                    dpg_env_batch_size = round(algo_cfg.env_batch_size * algo_cfg.proportion_dpg)
                    ga_env_batch_size = (
                        algo_cfg.env_batch_size - qpg_env_batch_size - dpg_env_batch_size
                    )
                    qpg_emitter_config = QualityPGEmitterConfig(
                        env_batch_size=qpg_env_batch_size,
                        num_critic_training_steps=algo_cfg.num_critic_training_steps,
                        num_pg_training_steps=algo_cfg.num_pg_training_steps,
                        replay_buffer_size=algo_cfg.replay_buffer_size,
                        critic_hidden_layer_size=algo_cfg.critic_hidden_layer_size,
                        critic_learning_rate=algo_cfg.critic_learning_rate,
                        actor_learning_rate=algo_cfg.greedy_learning_rate,
                        policy_learning_rate=algo_cfg.policy_learning_rate,
                        noise_clip=algo_cfg.noise_clip,
                        policy_noise=algo_cfg.policy_noise,
                        discount=algo_cfg.discount,
                        reward_scaling=algo_cfg.reward_scaling,
                        batch_size=algo_cfg.batch_size,
                        soft_tau_update=algo_cfg.soft_tau_update,
                        policy_delay=algo_cfg.policy_delay,
                        actor_optimizer=algo_cfg.actor_optimizer,
                    )
                    dpg_emitter_config = DiversityPGEmitterConfig(
                        env_batch_size=dpg_env_batch_size,
                        num_critic_training_steps=algo_cfg.num_critic_training_steps,
                        num_pg_training_steps=algo_cfg.num_pg_training_steps,
                        replay_buffer_size=algo_cfg.replay_buffer_size,
                        critic_hidden_layer_size=algo_cfg.critic_hidden_layer_size,
                        critic_learning_rate=algo_cfg.critic_learning_rate,
                        actor_learning_rate=algo_cfg.greedy_learning_rate,
                        policy_learning_rate=algo_cfg.policy_learning_rate,
                        noise_clip=algo_cfg.noise_clip,
                        policy_noise=algo_cfg.policy_noise,
                        discount=algo_cfg.discount,
                        reward_scaling=algo_cfg.reward_scaling,
                        batch_size=algo_cfg.batch_size,
                        soft_tau_update=algo_cfg.soft_tau_update,
                        policy_delay=algo_cfg.policy_delay,
                        actor_optimizer=algo_cfg.actor_optimizer,
                        archive_acceptance_threshold=algo_cfg.archive_acceptance_threshold,
                        archive_max_size=algo_cfg.archive_max_size,
                    )
                    qdpg_emitter_config = QDPGEmitterConfig(
                        qpg_config=qpg_emitter_config,
                        dpg_config=dpg_emitter_config,
                        iso_sigma=algo_cfg.iso_sigma,
                        line_sigma=algo_cfg.line_sigma,
                        ga_batch_size=ga_env_batch_size,
                    )
                    score_novelty = jax_jit(
                        functools.partial(
                            qdax.core.containers.archive.score_euclidean_novelty,
                            num_nearest_neighb=algo_cfg.num_nearest_neighb,
                            scaling_ratio=algo_cfg.novelty_scaling_ratio,
                        )
                    )
                    self.map_elites = ExtendedMAPElites(
                        scoring_function=scoring_fn,
                        emitter=QDPGEmitter(
                            qdpg_emitter_config,
                            actor,
                            self.env.env,
                            score_novelty,
                        ),
                        metrics_function=metrics_fn,
                    )
                case 'OMG-MEGA-RL':
                    algo_cfg = astype(self.cfg.algo, OMGMEGARLConfig)
                    assert isinstance(self.env, QDEnv)
                    self.map_elites = ExtendedMAPElites(
                        scoring_function=scoring_fn,
                        emitter=OMGMEGARLEmitter(
                            algo_cfg,
                            actor,
                            self.env.env,
                            functools.partial(
                                qdax.core.emitters.mutation_operators.isoline_variation,
                                iso_sigma=algo_cfg.iso_sigma,
                                line_sigma=algo_cfg.line_sigma,
                            ),
                        ),
                        metrics_function=metrics_fn,
                    )
                case 'CCQD':
                    algo_cfg = astype(self.cfg.algo, CCQDConfig)
                    self.map_elites = ExtendedMAPElites(
                        scoring_function=scoring_fn,
                        emitter=CCQDEmitter(
                            algo_cfg,
                            actor,
                            self.env,
                        ),
                        metrics_function=metrics_fn,
                    )
                case 'DQN-ME':
                    algo_cfg = astype(self.cfg.algo, DQNMEConfig)
                    self.map_elites = ExtendedMAPElites(
                        scoring_function=scoring_fn,
                        emitter=DQNMEEmitter(
                            algo_cfg,
                            actor,
                            self.env,
                            functools.partial(
                                qdax.core.emitters.mutation_operators.isoline_variation,
                                iso_sigma=algo_cfg.iso_sigma,
                                line_sigma=algo_cfg.line_sigma,
                            ),
                        ),
                        metrics_function=metrics_fn,
                    )
                case 'CCDQN-ME':
                    algo_cfg = astype(self.cfg.algo, CCDQNMEConfig)
                    self.map_elites = ExtendedMAPElites(
                        scoring_function=scoring_fn,
                        emitter=CCDQNMEEmitter(
                            algo_cfg,
                            actor,
                            self.env,
                            functools.partial(
                                qdax.core.emitters.mutation_operators.isoline_variation,
                                iso_sigma=algo_cfg.iso_sigma,
                                line_sigma=algo_cfg.line_sigma,
                            ),
                        ),
                        metrics_function=metrics_fn,
                    )
                case _:
                    raise NotImplementedError

        else:
            if self.cfg.algo.name == 'PBT-ME-SAC':
                algo_cfg = astype(self.cfg.algo, PBTMESacConfig)
                algo_cfg.num_training_iterations //= algo_cfg.training_env_batch_size
                assert algo_cfg.episode_length is DERIVED_INT
                algo_cfg.episode_length = self.cfg.env.episode_len
                assert algo_cfg.hidden_layer_sizes == DERIVED_TUPLE_INT
                algo_cfg.hidden_layer_sizes = (
                    *self.cfg.algo.representation_hidden_layer_sizes,
                    *self.cfg.algo.decision_hidden_layer_sizes,
                )

                env = qdax.environments.create(
                    env_name=self.cfg.env.name,
                    batch_size=(
                        algo_cfg.training_env_batch_size * algo_cfg.pg_population_size_per_device
                    ),
                    episode_length=self.cfg.env.episode_len,
                )
                agent = qdax.baselines.sac_pbt.PBTSAC(algo_cfg, self.env.action_size)

                @jax_jit
                def init_fn_pbt(random_key: jax.random.KeyArray):
                    random_key, init_vars, _ = agent.get_init_fn(
                        algo_cfg.env_batch_size,
                        self.env.action_size,
                        self.env.obs_size,
                        algo_cfg.buffer_size,
                    )(random_key)
                    return random_key, init_vars

                self.random_key, init_vars = init_fn_pbt(self.random_key)
                init_vars = jax.vmap(init_vars.__class__.empty_optimizers_states)(init_vars)

                def select_action_fn(
                    training_state: qdax.baselines.sac.SacTrainingState, obs: jp.ndarray
                ) -> jax.Array:
                    _select_action_fn = functools.partial(
                        agent.select_action,
                        random_key=None,
                        deterministic=True,
                    )
                    return jax.vmap(_select_action_fn)(obs, training_state.policy_params)[0]

                scoring_fn = self.env.get_scoring_fn(
                    astype(select_action_fn, Callable[[Params, jp.ndarray], jax.Array]),
                    buffer.QDTransition,
                )
                make_genotypes_fn: Callable[
                    [flax.core.scope.VariableDict], flax.core.scope.VariableDict
                ] = lambda x: x

                pbt_emitter_config = qdax.core.emitters.pbt_me_emitter.PBTEmitterConfig(
                    buffer_size=algo_cfg.buffer_size,
                    num_training_iterations=algo_cfg.num_training_iterations,
                    env_batch_size=algo_cfg.training_env_batch_size,
                    grad_updates_per_step=astype(algo_cfg.grad_updates_per_step, int),
                    pg_population_size_per_device=algo_cfg.pg_population_size_per_device,
                    ga_population_size_per_device=algo_cfg.ga_population_size_per_device,
                    num_devices=algo_cfg.num_devices,
                    fraction_best_to_replace_from=algo_cfg.fraction_best_to_replace_from,
                    fraction_to_replace_from_best=algo_cfg.fraction_to_replace_from_best,
                    fraction_to_replace_from_samples=algo_cfg.fraction_to_replace_from_samples,
                    fraction_sort_exchange=algo_cfg.fraction_sort_exchange,
                )

                def variation_fn(
                    training_state1: Params,
                    training_state2: Params,
                    random_key: jax.random.KeyArray,
                ) -> tuple[Params, jax.random.KeyArray]:
                    assert isinstance(training_state1, qdax.baselines.sac_pbt.PBTSacTrainingState)
                    assert isinstance(training_state2, qdax.baselines.sac_pbt.PBTSacTrainingState)
                    return astype(qdax.core.emitters.pbt_variation_operators.sac_pbt_variation_fn(
                        training_state1, training_state2, random_key,
                        algo_cfg.iso_sigma, algo_cfg.line_sigma,
                    ), tuple[Params, jax.random.KeyArray])

                self.map_elites = ExtendedMAPElites(
                    scoring_function=scoring_fn,
                    emitter=PBTEmitter(
                        agent,
                        pbt_emitter_config,
                        env,
                        variation_fn,
                    ),
                    metrics_function=metrics_fn,
                )
            else:
                raise NotImplementedError

        minval, maxval = self.env.bd_range
        centroids, self.random_key = compute_cvt_centroids(
            num_descriptors=self.env.bd_len,
            num_init_cvt_samples=self.cfg.algo.n_init_cvt_samples,
            num_centroids=self.cfg.algo.n_centroids,
            minval=minval,
            maxval=maxval,
            random_key=self.random_key,
        )

        self.repertoire, self.emitter_state, self.random_key = self.map_elites.init(
            init_vars, centroids, self.random_key, self.cfg.algo.env_batch_size,
            class_partial(ExtendedMapElitesRepertoire, make_genotypes=make_genotypes_fn),
        )

        self.csv_header = [
            'n_loop',
            'n_iter',
            'n_eval',
            'n_real_step',
            'qd_score',
            'max_fitness',
            'coverage',
            'min_fitness',
            'mean_fitness',
            'max_n_step',
            'min_n_step',
            'mean_n_step',
        ]

        self.loop_completed = 0
        self.n_real_step = 0
        self.history = {
            'fitnesses': jnp.empty((0, self.cfg.algo.n_centroids), dtype=jnp.float32),
            'descriptors': jnp.empty((0, *centroids.shape), dtype=jnp.float32),
        }

        log_period_ = self.cfg.env.log_period / self.cfg.algo.env_batch_size
        # assert log_period_.is_integer()
        self.log_period = int(log_period_)
        plot_interval = self.cfg.env.total_steps / self.cfg.total_plots
        assert plot_interval.is_integer()
        self.plot_interval = int(plot_interval)

        self.saved = False
        self.compressed = False

        if self.resumed:
            self.load()

        self.csv_logger = CSVLogger(self.cfg.metrics_filename, header=self.csv_header)

    @staticmethod
    def restore_from_wandb(filename: str, warning: str):
        try:
            _f = wandb.restore(filename, replace=True, root='.')
            assert _f is not None
        except Exception as e:
            _log.warning(warning, e)
        else:
            _f.close()

    def save_config(self):
        with open(self.cfg.config_filename, 'w') as f:
            f.write(OmegaConf.to_yaml(self.cfg))
        if wandb.run is not None:
            wandb.run.save(self.cfg.config_filename, policy='now')

    def check_config_locally(self):
        with open(self.cfg.config_filename, 'r') as f:
            old_config_yaml = f.read()
        new_config_yaml = OmegaConf.to_yaml(self.cfg)
        if new_config_yaml != old_config_yaml:
            msg = (
                'The new config is different from the old config.\n'
                'The old config:\n\n{old}\nThe new config: \n\n{new}\n'
                .format(old=old_config_yaml, new=new_config_yaml)
            )
            if 'FORCE_RESUME' in os.environ:
                _log.warning(msg)
            else:
                raise ValueError(
                    msg +
                    'Delete the local files and set WANDB_FORCE_OVERWRITE '
                    'in environment variables to overwrite the previous run.\n'
                    'Set FORCE_RESUME in environment variables to resume from the previous run.'
                )

    def check_config(self):
        _log.info('Checking the config...')

        try:
            self.check_config_locally()
        except FileNotFoundError:
            pass

        if wandb.run is not None:
            self.restore_from_wandb(
                self.cfg.config_filename,
                '%s Using the local config instead.',
            )
        else:
            _log.warning('Wandb is disabled. Using the local checkpoint instead.')

        try:
            self.check_config_locally()
        except FileNotFoundError as e:
            _log.warning('%s. Restarting the run instead.', e)
            self.resumed = False

    def save(self):
        if self.saved:
            return
        saved_keys = (
            'repertoire',
            'emitter_state',
            'random_key',
            'loop_completed',
            'n_real_step',
            'history',
        )
        saved_dict = {key: getattr(self, key) for key in saved_keys}
        with lz4.frame.open(self.cfg.checkpoint_filename + self.cfg.tmpfile_postfix, 'wb') as f:
            cloudpickle.dump(saved_dict, f)
        self.compressed = False
        os.replace(
            self.cfg.checkpoint_filename + self.cfg.tmpfile_postfix,
            self.cfg.checkpoint_filename,
        )
        self.saved = True

    def compress(self):
        if self.compressed:
            return
        with lz4.frame.open(self.cfg.checkpoint_filename, 'rb') as f:
            data = f.read()
            assert isinstance(data, bytes)
        with lzma.open(self.cfg.compressed_checkpoint_filename, 'wb', preset=0) as f:
            f.write(data)
        os.remove(self.cfg.checkpoint_filename)
        self.compressed = True

    def load(self):
        _log.info('Restoring the metrics and the checkpoint...')
        if wandb.run is not None:
            self.restore_from_wandb(
                self.cfg.metrics_filename,
                '%s Using the local metrics file instead.',
            )
            self.restore_from_wandb(
                self.cfg.compressed_checkpoint_filename,
                '%s Using the local checkpoint instead.',
            )
        else:
            _log.warning('Wandb is disabled. Using the local checkpoint instead.')

        if os.path.exists(self.cfg.metrics_filename):
            with open(self.cfg.metrics_filename, 'r') as f:
                n_lines = len(f.readlines())
            if n_lines > 1:
                assert os.path.exists(self.cfg.compressed_checkpoint_filename)
        else:
            assert not os.path.exists(self.cfg.compressed_checkpoint_filename)

        _log.info('Loading the checkpoint...')
        try:
            with lzma.open(self.cfg.compressed_checkpoint_filename, 'rb') as f:
                saved_dict: dict[str, Any] = pickle.load(f)
            for key, value in saved_dict.items():
                setattr(self, key, value)
        except FileNotFoundError as e:
            _log.warning('%s. Restarting the run instead.', e)

    def plot_repertoire(self) -> matplotlib.figure.Figure:
        minval, maxval = self.env.bd_range
        minval = jnp.array(minval)
        maxval = jnp.array(maxval)
        match self.env.bd_len:
            case 1:
                raise NotImplementedError
            case 2:
                fig, _ = qdax.utils.plotting.plot_2d_map_elites_repertoire(
                    self.repertoire.centroids,
                    self.repertoire.fitnesses,
                    minval,
                    maxval,
                    self.repertoire.descriptors,
                )
            case _:
                raise NotImplementedError
        assert fig is not None
        return fig

    def run(self):
        @functools.partial(jax_jit, donate_argnums=(0, 1, 2))
        def scan(
            repertoire_: ExtendedMapElitesRepertoire,
            emitter_state_: Optional[qdax.core.emitters.emitter.EmitterState],
            key_: jax.random.KeyArray,
        ) -> tuple[
            tuple[
                ExtendedMapElitesRepertoire,
                Optional[qdax.core.emitters.emitter.EmitterState],
                jax.random.KeyArray,
            ],
            tuple[Metrics, jax.Array, jax.Array],
        ]:
            return jax.lax.scan(
                self.map_elites.scan_update,
                (repertoire_, emitter_state_, key_,),
                (),
                length=self.log_period,
            )

        interrupted = False
        _log.info('metrics: %s', ' '.join(self.csv_header))
        trange = tqdm.tqdm(
            initial=int(self.n_real_step),
            total=self.cfg.env.total_steps,
            dynamic_ncols=True,
        )
        try:
            with tqdm.contrib.logging.logging_redirect_tqdm():
                while self.n_real_step < self.cfg.env.total_steps:
                    with uninterrupted():
                        self.saved = False
                        (
                            (self.repertoire, self.emitter_state, self.random_key),
                            (main_metrics_, fitnesses, descriptors),
                        ) = scan(self.repertoire, self.emitter_state, self.random_key)
                        self.loop_completed += 1
                        self.history['fitnesses'] = jnp.concatenate(
                            (self.history['fitnesses'], fitnesses)
                        )
                        self.history['descriptors'] = jnp.concatenate(
                            (self.history['descriptors'], descriptors)
                        )

                        main_metrics = astype(
                            main_metrics_,
                            dict[str, Iterable[int | float | jax.Array]],
                        )
                        sum_n_step = astype(main_metrics.pop('sum_n_step'), jax.Array)

                        stop_n_iter = self.loop_completed * self.log_period + 1
                        start_n_iter = stop_n_iter - self.log_period

                        if not self.cfg.algo.count_eval_steps:
                            sum_n_step = jnp.zeros_like(sum_n_step)
                        if self.cfg.algo.name in ['PBT-ME-SAC']:
                            algo_cfg = astype(self.cfg.algo, PBTMESacConfig)
                            sum_n_step += (
                                algo_cfg.num_training_iterations
                                * algo_cfg.training_env_batch_size
                                * algo_cfg.pg_population_size_per_device
                            )
                        else:
                            assert self.cfg.algo.init is not None
                        n_real_step = jnp.array(list(
                            itertools.accumulate(sum_n_step, initial=self.n_real_step)
                        )[1:])
                        other_metrics: dict[str, Iterable[int | float | jax.Array]] = {
                            'n_loop': [self.loop_completed] * self.log_period,
                            'n_iter': range(start_n_iter, stop_n_iter),
                            'n_eval': range(
                                start_n_iter * self.cfg.algo.env_batch_size,
                                stop_n_iter * self.cfg.algo.env_batch_size,
                                self.cfg.algo.env_batch_size,
                            ),
                            'n_real_step': n_real_step,
                        }

                        log_metrics, csv_metrics, wandb_metrics = get_logged_metrics(
                            main_metrics, other_metrics, self.csv_header
                        )

                        new_n_real_step = int(csv_metrics[-1]['n_real_step'])
                        finish = new_n_real_step >= self.cfg.env.total_steps

                        self.csv_logger.log(csv_metrics)
                        if wandb.run is not None:
                            if (
                                self.env.bd_len == 2
                                and (finish or (
                                    new_n_real_step // self.plot_interval
                                    > self.n_real_step // self.plot_interval
                                ))
                            ):
                                fig = self.plot_repertoire()
                                wandb_metrics['plots/repertoire'] = wandb.Image(fig)
                                plt.close(fig)
                            wandb.run.log(wandb_metrics)
                            wandb.run.summary.update(csv_metrics[-1])
                        _log.info('metrics: %s', log_metrics)

                        self.n_real_step = new_n_real_step

                        if finish or self.loop_completed % self.cfg.checkpoint_saving_interval == 0:
                            _log.info('Saving...')
                            self.save()

                        if wandb.run is not None and (
                            finish or self.loop_completed % self.cfg.metrics_uploading_interval == 0
                        ):
                            _log.info('Uploading metrics...')
                            wandb.run.save(self.cfg.metrics_filename, policy='now')

                        if (
                            isinstance(self.emitter_state, CCQDEmitterState)
                            and self.emitter_state.repertoire_representation_indices is not None
                        ):
                            algo_cfg = astype(self.cfg.algo, CCQDConfig)
                            _log.info(
                                (
                                    self.emitter_state.repertoire_representation_indices
                                    == jnp.expand_dims(
                                        jnp.arange(algo_cfg.n_representation),
                                        axis=-1,
                                    )
                                ).sum(axis=-1)
                            )
                        elif isinstance(
                            self.emitter_state,
                            qdax.core.emitters.multi_emitter.MultiEmitterState,
                        ):
                            for emitter_state in self.emitter_state.emitter_states:
                                if (
                                    isinstance(emitter_state, CCQDEmitterState)
                                    and emitter_state.repertoire_representation_indices is not None
                                ):
                                    algo_cfg = astype(self.cfg.algo, CCQDConfig)
                                    _log.info(
                                        (
                                            emitter_state.repertoire_representation_indices
                                            == jnp.expand_dims(
                                                jnp.arange(algo_cfg.n_representation),
                                                axis=-1,
                                            )
                                        ).sum(axis=-1)
                                    )

                        trange.total = max(trange.total, self.n_real_step)
                        trange.update(self.n_real_step - trange.n)

                        if finish:
                            break

                        time_interrupted = time.time()
                    time_interrupted = time.time() - time_interrupted
                    trange.start_t += time_interrupted
                    trange.last_print_t += time_interrupted

        except KeyboardInterrupt:
            interrupted = True
            _log.info('Saving...')
            self.save()
            raise
        finally:
            trange.close()
            with uninterrupted():
                _log.info('Compressing...')
                try:
                    self.compress()
                except FileNotFoundError as e:
                    _log.warning('%s. Failed to compress the checkpoint.', e)

                if wandb.run is not None:
                    _log.info('Uploading metrics...')
                    wandb.run.save(self.cfg.metrics_filename, policy='end')

            if wandb.run is not None:
                if interrupted:
                    wandb.run.save(self.cfg.compressed_checkpoint_filename, policy='end')

        if self.cfg.representation_diversity_verification:
            with uninterrupted():
                if self.cfg.algo.name in ['CCQD']:
                    _log.info('Verifying representation diversity...')
                    algo_cfg = astype(self.cfg.algo, CCQDConfig)
                    emitter_state = astype(self.emitter_state, CCQDEmitterState)
                    assert algo_cfg.n_representation is not None
                    assert (algo_cfg.env_batch_size / algo_cfg.n_representation).is_integer()
                    representation_vars = tree_repeat(
                        emitter_state.representation_vars,
                        repeats=algo_cfg.env_batch_size // algo_cfg.n_representation,
                    )
                    scoring_fn = self.map_elites._scoring_function

                    def verify_representation_diversity(
                        decision_vars: flax.core.scope.VariableDict,
                        random_key: jax.random.KeyArray,
                    ):
                        vars = Actor.make_vars(representation_vars, decision_vars)
                        fitnesses, descriptors, _, _, _ = scoring_fn(vars, random_key)
                        return fitnesses, descriptors

                    vars = astype(self.repertoire.genotypes, flax.core.scope.VariableDict)
                    _, decision_vars = Actor.extract_vars(vars)
                    self.random_key, subkey = jax.random.split(self.random_key)
                    keys = jax.random.split(subkey, algo_cfg.n_centroids)
                    fitnesses, descriptors = jax_jit(jax.vmap(verify_representation_diversity))(
                        decision_vars, keys
                    )
                    _log.info('Saving...')
                    with lzma.open(self.cfg.representation_diversity_filename, 'wb', preset=0) as f:
                        cloudpickle.dump({
                            'centroids': self.repertoire.centroids,
                            'fitnesses': self.repertoire.fitnesses,
                            'descriptors': self.repertoire.descriptors,
                            'repertoire_representation_indices':
                                emitter_state.repertoire_representation_indices,
                            'new_fitnesses': fitnesses,
                            'new_descriptors': descriptors,
                        }, f)
                    if wandb.run is not None:
                        _log.info('Uploading representation diversity verification results...')
                        wandb.run.save(self.cfg.representation_diversity_filename, policy='end')
                else:
                    _log.info('Saving...')
                    with lzma.open(self.cfg.representation_diversity_filename, 'wb', preset=0) as f:
                        cloudpickle.dump({
                            'centroids': self.repertoire.centroids,
                            'fitnesses': self.repertoire.fitnesses,
                            'descriptors': self.repertoire.descriptors,
                        }, f)
                    if wandb.run is not None:
                        _log.info('Uploading...')
                        wandb.run.save(self.cfg.representation_diversity_filename, policy='end')


def main(cfg: 'Config'):
    _log.info('Config:\n\n%s', OmegaConf.to_yaml(cfg))
    _log.info('Logdir: %s', os.getcwd())
    original_dir = os.path.realpath(hydra.utils.get_original_cwd())

    if cfg.wandb:
        if 'WANDB_FORCE_OVERWRITE' in os.environ:
            _log.warning('WANDB_FORCE_OVERWRITE is set. The previous run may be overwritten.')
            resume = None
        elif 'WANDB_ALLOW_RESUME' in os.environ or 'FORCE_RESUME' in os.environ:
            resume = 'allow'
        else:
            resume = 'never'

        cfg_dict = OmegaConf.to_container(cfg)
        assert isinstance(cfg_dict, dict)
        wandb.init(
            project='CCQD',
            id='{algo}.{env}.{code}.{run}.{seed}'.format(
                algo=cfg.algo.name, env=cfg.env.name, code=cfg.code, run=cfg.run, seed=cfg.seed
            ),
            resume=resume,
            name='{algo} {env} {code} {run} {seed}'.format(
                algo=cfg.algo.name, env=cfg.env.name, code=cfg.code, run=cfg.run, seed=cfg.seed
            ),
            config=cfg_dict,
            tags=[cfg.env.name, cfg.algo.name],
            job_type=cfg.type,
            dir=original_dir,
        )
        assert wandb.run is not None
        if os.path.islink('wandb-run-dir'):
            os.remove('wandb-run-dir')
        os.symlink(os.path.relpath(wandb.run.dir), 'wandb-run-dir')

    main = Main(cfg)
    main.run()
