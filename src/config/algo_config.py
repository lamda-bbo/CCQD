import qdax.baselines.sac_pbt

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING
from dataclasses import dataclass
from typing import Optional, Any, TypeVar

from ..emitters import (
    PGAMEEmitterConfig,
    OMGMEGARLEmitterConfig,
    CCQDEmitterConfig,
    DQNMEEmitterConfig,
    CCDQNMEEmitterConfig,
)
from ..utils import DERIVED_INT, DERIVED_TUPLE_INT, dataclass_field


@dataclass
class AlgoConfig:
    name: str = MISSING
    env_batch_size: int = 100
    n_init_vars: int = 100
    init: Optional[str] = MISSING
    count_eval_steps: bool = True
    representation_hidden_layer_sizes: tuple[int, ...] = (256,)
    decision_hidden_layer_sizes: tuple[int, ...] = (256,)
    activation: str = 'tanh'
    final_activation: Optional[str] = 'tanh'
    n_init_cvt_samples: int = 50000
    n_centroids: int = 1024


_algo_configs: dict[str, type[AlgoConfig]] = {}


_TAlgoConfig = TypeVar('_TAlgoConfig', bound=AlgoConfig)


def _register(cls: type[_TAlgoConfig]) -> type[_TAlgoConfig]:
    assert cls.name not in _algo_configs
    _algo_configs[cls.name] = cls
    return cls


@_register
@dataclass
class MEConfig(AlgoConfig):
    name: str = 'ME'
    init: Optional[str] = 'classic'
    iso_sigma: float = 0.005
    line_sigma: float = 0.05


@_register
@dataclass
class PBTMESacConfig(AlgoConfig, qdax.baselines.sac_pbt.PBTSacConfig):
    name: str = 'PBT-ME-SAC'
    env_batch_size: int = 320
    init: Optional[str] = None
    batch_size: int = 256
    training_env_batch_size: int = 250
    episode_length: int = DERIVED_INT
    tau: float = 0.005
    normalize_observations: bool = False
    alpha_init: float = 1.0
    hidden_layer_sizes: tuple[int, ...] = DERIVED_TUPLE_INT
    fix_alpha: bool = False
    buffer_size: int = 100000
    num_training_iterations: int = 5000
    grad_updates_per_step: float = 1.0
    pg_population_size_per_device: int = 80
    ga_population_size_per_device: int = 240
    num_devices: int = 1
    fraction_best_to_replace_from: float = 0.1
    fraction_to_replace_from_best: float = 0.2
    fraction_to_replace_from_samples: float = 0.4
    fraction_sort_exchange: float = 0.1
    iso_sigma: float = 0.005
    line_sigma: float = 0.05


@_register
@dataclass
class PGAMEConfig(AlgoConfig, PGAMEEmitterConfig):
    name: str = 'PGA-ME'
    init: Optional[str] = 'classic'
    proportion_mutation_ga: float = 0.5
    num_critic_training_steps: int = 300
    num_pg_training_steps: int = 100
    replay_buffer_size: int = 1000000
    critic_hidden_layer_size: tuple[int, ...] = (256, 256)
    critic_learning_rate: float = 3e-4
    greedy_learning_rate: float = 3e-4
    policy_learning_rate: float = 0.001
    noise_clip: float = 0.5
    policy_noise: float = 0.2
    discount: float = 0.99
    reward_scaling: float = 1.0
    batch_size: int = 256
    soft_tau_update: float = 0.005
    policy_delay: int = 2
    actor_optimizer: str = 'Adam'
    iso_sigma: float = 0.005
    line_sigma: float = 0.05


@_register
@dataclass
class QDPGConfig(AlgoConfig):
    name: str = 'QD-PG'
    init: Optional[str] = 'classic'

    num_critic_training_steps: int = 300
    num_pg_training_steps: int = 100
    replay_buffer_size: int = 1000000
    critic_hidden_layer_size: tuple[int, ...] = (256, 256)
    critic_learning_rate: float = 3e-4
    greedy_learning_rate: float = 3e-4
    policy_learning_rate: float = 0.001
    noise_clip: float = 0.5
    policy_noise: float = 0.2
    discount: float = 0.99
    reward_scaling: float = 1.0
    batch_size: int = 256
    soft_tau_update: float = 0.005
    policy_delay: int = 2
    actor_optimizer: str = 'Adam'

    archive_acceptance_threshold: float = 0.1
    archive_max_size: int = 10000

    iso_sigma: float = 0.005
    line_sigma: float = 0.05
    proportion_qpg: float = 0.34
    proportion_dpg: float = 0.33

    num_nearest_neighb: int = 5
    novelty_scaling_ratio: float = 1.0


@_register
@dataclass
class OMGMEGARLConfig(AlgoConfig, OMGMEGARLEmitterConfig):
    name: str = 'OMG-MEGA-RL'
    init: Optional[str] = 'classic'
    proportion_mutation_ga: float = 0.5
    num_critic_training_steps: int = 300
    num_pg_training_steps: int = 100
    replay_buffer_size: int = 1000000
    critic_hidden_layer_size: tuple[int, ...] = (256, 256)
    critic_learning_rate: float = 3e-4
    greedy_learning_rate: float = 3e-4
    policy_learning_rate: float = 0.001
    noise_clip: float = 0.5
    policy_noise: float = 0.2
    discount: float = 0.99
    reward_scaling: float = 1.0
    batch_size: int = 256
    soft_tau_update: float = 0.005
    policy_delay: int = 2
    actor_optimizer: str = 'Adam'
    iso_sigma: float = 0.005
    line_sigma: float = 0.05

    desc_critic_hidden_layer_size: tuple[int, ...] = (256, 256)
    desc_critic_learning_rate: float = 3e-4
    desc_reward_scaling: float = 1.0
    desc_discount: float = 1.0


@_register
@dataclass
class CCQDConfig(AlgoConfig, CCQDEmitterConfig):
    name: str = 'CCQD'
    init: Optional[str] = 'representation'

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

    n_representation: int = 20
    single_critic: bool = False
    p_concat_crossover: float = 1.0
    replacement_threshold: int = 0
    replaced_by: str = 'most'
    representation_indices_reassignment: str = 'Random'
    representation_indices_reassignment_params: dict[str, Any] = dataclass_field({})


@dataclass
class CNNAlgoConfig(AlgoConfig):
    env_batch_size: int = 20
    n_init_vars: int = 20
    n_centroids: int = 100

    representation_conv_features: tuple[int, ...] = (32, 64, 64)
    representation_kernel_sizes: tuple[Any, ...] = ((8, 8), (4, 4), (3, 3))
    representation_conv_strides: tuple[int, ...] = (4, 2, 1)
    representation_hidden_layer_sizes: tuple[int, ...] = ()

    decision_cnn_input_shape: tuple[int, ...] = (-1,)
    decision_conv_features: tuple[int, ...] = ()
    decision_kernel_sizes: tuple[Any, ...] = ()
    decision_conv_strides: tuple[int, ...] = ()
    decision_hidden_layer_sizes: tuple[int, ...] = (512,)

    conv_activation: str = 'leaky_relu'
    conv_padding: str = 'VALID'

    activation: str = 'leaky_relu'
    final_activation: Optional[str] = None


@_register
@dataclass
class DQNMEConfig(CNNAlgoConfig, DQNMEEmitterConfig):
    name: str = 'DQN-ME'
    init: Optional[str] = 'classic'
    proportion_mutation_ga: float = 0.5
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
    iso_sigma: float = 0.005
    line_sigma: float = 0.05


@_register
@dataclass
class CCDQNMEConfig(CNNAlgoConfig, CCDQNMEEmitterConfig):
    name: str = 'CCDQN-ME'
    init: Optional[str] = 'representation'
    proportion_mutation_ga: float = 0.5
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
    iso_sigma: float = 0.005
    line_sigma: float = 0.05


def register_algo_configs():
    cs = ConfigStore.instance()
    for name, cls in _algo_configs.items():
        cs.store(
            group='algo',
            name=name,
            node=cls,
        )
