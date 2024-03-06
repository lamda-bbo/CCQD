import jax
import flax.linen as nn
import qdax.core.containers.archive
import qdax.core.containers.repertoire
import qdax.core.emitters.dpg_emitter
import qdax.core.emitters.multi_emitter
import qdax.core.emitters.mutation_operators
import qdax.core.emitters.standard_emitters
from qdax.core.neuroevolution.buffers import buffer
import qdax.environments.base_wrappers
from qdax.types import (
    Descriptor,
    ExtraScores,
    Fitness,
    Genotype,
    Reward,
    RNGKey,
    StateDescriptor,
)

from dataclasses import dataclass
import functools
from collections.abc import Callable
from typing import Optional

from .pga_me_emitter import QualityPGEmitterConfig, QualityPGEmitter
from ..utils import astype


@dataclass
class DiversityPGEmitterConfig(QualityPGEmitterConfig):
    """Configuration for DiversityPG Emitter"""

    # inherits fields from QualityPGEmitterConfig

    # Archive params
    archive_acceptance_threshold: float = 0.1
    archive_max_size: int = 10000


class DiversityPGEmitter(QualityPGEmitter):
    """
    A diversity policy gradient emitter used to implement QDPG algorithm.
    """

    def __init__(
        self,
        config: DiversityPGEmitterConfig,
        policy_network: nn.Module,
        env: qdax.environments.base_wrappers.QDEnv,
        score_novelty: Callable[[qdax.core.containers.archive.Archive, StateDescriptor], Reward],
    ):
        # usual init operations from PGAME
        super().__init__(config, policy_network, env)

        self._config: DiversityPGEmitterConfig = config

        # define scoring function
        self._score_novelty = score_novelty

    def init(
        self, init_genotypes: Genotype, random_key: RNGKey
    ) -> tuple[qdax.core.emitters.dpg_emitter.DiversityPGEmitterState, RNGKey]:
        # init elements of diversity emitter state with QualityEmitterState.init()
        diversity_emitter_state, random_key = super().init(init_genotypes, random_key)

        # store elements in a dictionary
        attributes_dict = vars(diversity_emitter_state)

        # init archive
        archive = qdax.core.containers.archive.Archive.create(
            acceptance_threshold=self._config.archive_acceptance_threshold,
            state_descriptor_size=self._env.state_descriptor_length,
            max_size=self._config.archive_max_size,
        )

        # init emitter state
        emitter_state = qdax.core.emitters.dpg_emitter.DiversityPGEmitterState(
            # retrieve all attributes from the QualityPGEmitterState
            **attributes_dict,
            # add the last element: archive
            archive=archive,
        )

        return emitter_state, random_key

    @functools.partial(jax.jit, static_argnames=("self",))
    def state_update(
        self,
        emitter_state: qdax.core.emitters.dpg_emitter.DiversityPGEmitterState,
        repertoire: Optional[qdax.core.containers.repertoire.Repertoire],
        genotypes: Optional[Genotype],
        fitnesses: Optional[Fitness],
        descriptors: Optional[Descriptor],
        extra_scores: ExtraScores,
    ) -> qdax.core.emitters.dpg_emitter.DiversityPGEmitterState:
        # get the transitions out of the dictionary
        assert "transitions" in extra_scores.keys(), "Missing transitions or wrong key"
        transitions = astype(extra_scores["transitions"], buffer.QDTransition)

        archive = emitter_state.archive.insert(transitions.state_desc)
        emitter_state = emitter_state.replace(archive=archive)

        return super().state_update(
            emitter_state,
            repertoire,
            genotypes,
            fitnesses,
            descriptors,
            extra_scores,
        )

    def _transform_transitions(
        self,
        emitter_state: qdax.core.emitters.dpg_emitter.DiversityPGEmitterState,
        transitions: buffer.QDTransition,
    ) -> buffer.QDTransition:
        return self._get_diversity_transitions(emitter_state.archive, transitions)

    def _get_diversity_transitions(
        self, archive: qdax.core.containers.archive.Archive, transitions: buffer.QDTransition
    ) -> buffer.QDTransition:
        state_descriptors = transitions.state_desc
        diversity_rewards = self._score_novelty(archive, state_descriptors)
        transitions = transitions.replace(rewards=diversity_rewards)
        return transitions


@dataclass
class QDPGEmitterConfig:
    qpg_config: QualityPGEmitterConfig
    dpg_config: DiversityPGEmitterConfig
    iso_sigma: float
    line_sigma: float
    ga_batch_size: int


class QDPGEmitter(qdax.core.emitters.multi_emitter.MultiEmitter):
    def __init__(
        self,
        config: QDPGEmitterConfig,
        policy_network: nn.Module,
        env: qdax.environments.base_wrappers.QDEnv,
        score_novelty: Callable[[qdax.core.containers.archive.Archive, StateDescriptor], Reward],
    ):
        self._config = config
        self._policy_network = policy_network
        self._env = env

        # define the quality emitter
        q_emitter = QualityPGEmitter(
            config=config.qpg_config, policy_network=policy_network, env=env
        )
        # define the diversity emitter
        d_emitter = DiversityPGEmitter(
            config=config.dpg_config,
            policy_network=policy_network,
            env=env,
            score_novelty=score_novelty,
        )

        # define the GA emitter
        variation_fn = functools.partial(
            qdax.core.emitters.mutation_operators.isoline_variation,
            iso_sigma=config.iso_sigma,
            line_sigma=config.line_sigma,
        )
        ga_emitter = qdax.core.emitters.standard_emitters.MixingEmitter(
            mutation_fn=lambda x, r: (x, r),
            variation_fn=variation_fn,
            variation_percentage=1.0,
            batch_size=config.ga_batch_size,
        )

        super().__init__(emitters=(q_emitter, d_emitter, ga_emitter))
