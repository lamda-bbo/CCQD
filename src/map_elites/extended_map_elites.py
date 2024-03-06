import jax
import jax.numpy as jnp
import qdax.core.map_elites
import qdax.core.emitters.emitter
from qdax.types import Centroid, Descriptor, ExtraScores, Fitness, Genotype, Metrics, RNGKey

import logging
import functools
from collections.abc import Callable
from typing import Optional, Any, TypeVar

from .extended_repertoire import ExtendedMapElitesRepertoire
from ..utils import jax_jit


_log = logging.getLogger(__name__)


_TGenotypeWrappedMapElitesRepertoire = TypeVar(
    '_TGenotypeWrappedMapElitesRepertoire',
    bound=ExtendedMapElitesRepertoire,
)


class ExtendedMAPElites(qdax.core.map_elites.MAPElites):

    def __init__(
        self,
        scoring_function: Callable[
            [Genotype, RNGKey], tuple[Fitness, Descriptor, ExtraScores, RNGKey, jax.Array]
        ],
        emitter: qdax.core.emitters.emitter.Emitter,
        metrics_function: Callable[[ExtendedMapElitesRepertoire], Metrics],
    ):
        self._scoring_function = scoring_function
        self._emitter = emitter
        self._metrics_function = metrics_function

    @functools.partial(jax_jit, static_argnames=('self', 'env_batch_size', 'RepertoreType'))
    def init(
        self,
        init_genotypes: Genotype,
        centroids: Centroid,
        random_key: RNGKey,
        env_batch_size: int,
        RepertoreType: type[
            _TGenotypeWrappedMapElitesRepertoire
        ] = ExtendedMapElitesRepertoire,
    ) -> tuple[
        _TGenotypeWrappedMapElitesRepertoire,
        Optional[qdax.core.emitters.emitter.EmitterState],
        RNGKey,
    ]:
        """
        Initialize a Map-Elites repertoire with an initial population of genotypes.
        Requires the definition of centroids that can be computed with any method
        such as CVT or Euclidean mapping.

        Args:
            init_genotypes: initial genotypes, pytree in which leaves
                have shape (batch_size, num_features)
            centroids: tesselation centroids of shape (batch_size, num_descriptors)
            random_key: a random key used for stochastic operations.

        Returns:
            An initialized MAP-Elite repertoire with the initial state of the emitter,
            and a random key.
        """
        # score initial genotypes
        fitnesses, descriptors, extra_scores, random_key, n_step = self._scoring_function(
            init_genotypes, random_key
        )

        # init the repertoire
        repertoire_ = RepertoreType.init(
            genotypes=init_genotypes,
            fitnesses=fitnesses,
            descriptors=descriptors,
            centroids=centroids,
            extra_scores=extra_scores,
        )
        repertoire: _TGenotypeWrappedMapElitesRepertoire = repertoire_  # type: ignore
        del repertoire_

        # get initial state of the emitter
        emitter_state, random_key = self._emitter.init(
            init_genotypes=init_genotypes, random_key=random_key
        )

        # update emitter state
        emitter_state = self._emitter.state_update(
            emitter_state=emitter_state,
            repertoire=repertoire,
            genotypes=init_genotypes,
            fitnesses=fitnesses,
            descriptors=descriptors,
            extra_scores=extra_scores,
        )

        assert repertoire.latest_batch_of_indices.shape[0] >= env_batch_size
        if repertoire.latest_batch_of_indices.shape[0] > env_batch_size:
            _log.warning('latest_batch_of_indices batch size > env_batch_size.')
            repertoire = repertoire.replace(
                latest_batch_of_indices=repertoire.latest_batch_of_indices[:env_batch_size]
            )

        assert isinstance(repertoire, ExtendedMapElitesRepertoire)

        return repertoire, emitter_state, random_key

    @functools.partial(jax_jit, static_argnames=("self",))
    def update(
        self,
        repertoire: _TGenotypeWrappedMapElitesRepertoire,
        emitter_state: Optional[qdax.core.emitters.emitter.EmitterState],
        random_key: RNGKey,
    ) -> tuple[
        _TGenotypeWrappedMapElitesRepertoire,
        Optional[qdax.core.emitters.emitter.EmitterState],
        Metrics,
        RNGKey,
    ]:
        """
        Performs one iteration of the MAP-Elites algorithm.
        1. A batch of genotypes is sampled in the repertoire and the genotypes
            are copied.
        2. The copies are mutated and crossed-over
        3. The obtained offsprings are scored and then added to the repertoire.


        Args:
            repertoire: the MAP-Elites repertoire
            emitter_state: state of the emitter
            random_key: a jax PRNG random key

        Returns:
            the updated MAP-Elites repertoire
            the updated (if needed) emitter state
            metrics about the updated repertoire
            a new jax PRNG key
        """
        # generate offsprings with the emitter
        genotypes, random_key = self._emitter.emit(
            repertoire, emitter_state, random_key
        )
        # scores the offsprings
        fitnesses, descriptors, extra_scores, random_key, n_step = self._scoring_function(
            genotypes, random_key
        )

        # add genotypes in the repertoire
        repertoire = repertoire.add(genotypes, descriptors, fitnesses, extra_scores)

        # update emitter state after scoring is made
        emitter_state = self._emitter.state_update(
            emitter_state=emitter_state,
            repertoire=repertoire,
            genotypes=genotypes,
            fitnesses=fitnesses,
            descriptors=descriptors,
            extra_scores=extra_scores,
        )

        # update the metrics
        metrics = self._metrics_function(repertoire)
        metrics['max_n_step'] = jnp.max(n_step)
        metrics['min_n_step'] = jnp.min(n_step)
        metrics['mean_n_step'] = jnp.mean(n_step)
        metrics['sum_n_step'] = jnp.sum(n_step)

        return repertoire, emitter_state, metrics, random_key

    @functools.partial(jax_jit, static_argnames=("self",))
    def scan_update(
        self,
        carry: tuple[
            _TGenotypeWrappedMapElitesRepertoire,
            Optional[qdax.core.emitters.emitter.EmitterState],
            RNGKey,
        ],
        unused: Any,
    ) -> tuple[
        tuple[
            _TGenotypeWrappedMapElitesRepertoire,
            Optional[qdax.core.emitters.emitter.EmitterState],
            RNGKey,
        ],
        tuple[Metrics, Fitness, Descriptor]
    ]:
        (repertoire, emitter_state, random_key), metrics = super().scan_update(carry, unused)
        return (
            (repertoire, emitter_state, random_key),
            (metrics, repertoire.fitnesses, repertoire.descriptors),
        )
