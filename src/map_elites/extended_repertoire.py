import jax
import jax.numpy as jnp
import qdax.core.containers.mapelites_repertoire
from qdax.types import Centroid, Descriptor, ExtraScores, Fitness, Genotype, RNGKey

import logging
from typing import Self
from overrides import override

from ..cluster import KMeans
from ..utils import jax_jit


_log = logging.getLogger(__name__)


def compute_cvt_centroids(
    num_descriptors: int,
    num_init_cvt_samples: int,
    num_centroids: int,
    minval: float | list[float],
    maxval: float | list[float],
    random_key: RNGKey,
) -> tuple[jax.Array, RNGKey]:

    minval_ = jnp.array(minval)
    maxval_ = jnp.array(maxval)

    # assume here all values are in [0, 1] and rescale later
    random_key, subkey = jax.random.split(random_key)
    x = jax.random.uniform(key=subkey, shape=(num_init_cvt_samples, num_descriptors))

    # compute k means
    random_key, subkey = jax.random.split(random_key)
    k_means = KMeans(
        k=num_centroids,
        init='k-means++',
        n_init=1,
    )
    k_means_state = k_means.fit(subkey, x)
    centroids = k_means_state.cluster_centers_
    assert not jnp.isnan(jnp.sum(centroids))
    # rescale now
    return jnp.asarray(centroids) * (maxval_ - minval_) + minval_, random_key


class ExtendedMapElitesRepertoire(
    qdax.core.containers.mapelites_repertoire.MapElitesRepertoire
):

    extra_scores: ExtraScores
    latest_batch_of_indices: jax.Array

    @staticmethod
    def make_genotypes(batch_of_genotypes: Genotype) -> Genotype:
        raise NotImplementedError

    @staticmethod
    def filter_extra_scores(extra_scores: ExtraScores) -> ExtraScores:
        return {}

    @jax_jit
    @override
    def add(
        self,
        batch_of_genotypes: Genotype,
        batch_of_descriptors: Descriptor,
        batch_of_fitnesses: Fitness,
        batch_of_extra_scores: ExtraScores,
    ) -> Self:

        batch_of_genotypes = self.__class__.make_genotypes(batch_of_genotypes)
        batch_of_extra_scores = self.filter_extra_scores(batch_of_extra_scores)

        batch_of_indices = qdax.core.containers.mapelites_repertoire.get_cells_indices(
            batch_of_descriptors, self.centroids
        )
        batch_of_indices = jnp.expand_dims(batch_of_indices, axis=-1)
        batch_of_fitnesses = jnp.expand_dims(batch_of_fitnesses, axis=-1)

        num_centroids = self.centroids.shape[0]

        # get fitness segment max
        best_fitnesses = jax.ops.segment_max(
            batch_of_fitnesses,
            batch_of_indices.astype(jnp.int32).squeeze(axis=-1),
            num_segments=num_centroids,
        )

        cond_values = jnp.take_along_axis(best_fitnesses, batch_of_indices, 0)

        # put dominated fitness to -jnp.inf
        batch_of_fitnesses = jnp.where(
            batch_of_fitnesses == cond_values, x=batch_of_fitnesses, y=-jnp.inf
        )

        # get addition condition
        repertoire_fitnesses = jnp.expand_dims(self.fitnesses, axis=-1)
        current_fitnesses = jnp.take_along_axis(
            repertoire_fitnesses, batch_of_indices, 0
        )
        addition_condition = batch_of_fitnesses > current_fitnesses

        # assign fake position when relevant : num_centroids is out of bound
        batch_of_indices = jnp.where(
            addition_condition, x=batch_of_indices, y=num_centroids
        )

        # create new repertoire
        new_repertoire_genotypes = jax.tree_util.tree_map(
            lambda repertoire_genotypes, new_genotypes: repertoire_genotypes.at[
                batch_of_indices.squeeze(axis=-1)
            ].set(new_genotypes),
            self.genotypes,
            batch_of_genotypes,
        )

        # compute new fitness and descriptors
        new_fitnesses = self.fitnesses.at[batch_of_indices.squeeze(axis=-1)].set(
            batch_of_fitnesses.squeeze(axis=-1)
        )
        new_descriptors = self.descriptors.at[batch_of_indices.squeeze(axis=-1)].set(
            batch_of_descriptors
        )

        # create new extra_scores
        new_extra_scores: ExtraScores = jax.tree_util.tree_map(
            lambda extra_scores, new_genotypes: extra_scores.at[
                batch_of_indices.squeeze(axis=-1)
            ].set(new_genotypes),
            self.extra_scores,
            batch_of_extra_scores,
        )

        return self.replace(
            genotypes=new_repertoire_genotypes,
            fitnesses=new_fitnesses,
            descriptors=new_descriptors,
            centroids=self.centroids,
            extra_scores=new_extra_scores,
            latest_batch_of_indices=batch_of_indices,
        )

    @classmethod
    @override
    def init(
        cls,
        genotypes: Genotype,
        fitnesses: Fitness,
        descriptors: Descriptor,
        centroids: Centroid,
        extra_scores: ExtraScores,
    ) -> Self:

        genotypes_ = cls.make_genotypes(genotypes)
        extra_scores_ = cls.filter_extra_scores(extra_scores)

        # retrieve one genotype from the population
        first_genotype = jax.tree_util.tree_map(lambda x: x[0], genotypes_)
        first_extra_scores = jax.tree_util.tree_map(lambda x: x[0], extra_scores_)

        # create a repertoire with default values
        repertoire = cls.init_default(
            genotype=first_genotype,
            centroids=centroids,
            extra_scores=first_extra_scores,
        )

        # add initial population to the repertoire
        new_repertoire = repertoire.add(genotypes, descriptors, fitnesses, extra_scores)

        return new_repertoire

    @classmethod
    def init_default(
        cls,
        genotype: Genotype,
        centroids: Centroid,
        extra_scores: ExtraScores,
    ) -> Self:

        # get number of centroids
        num_centroids = centroids.shape[0]

        # default fitness is -inf
        default_fitnesses = -jnp.inf * jnp.ones(shape=num_centroids)

        # default genotypes is all 0
        default_genotypes = jax.tree_util.tree_map(
            lambda x: jnp.zeros(shape=(num_centroids,) + x.shape, dtype=x.dtype),
            genotype,
        )

        # default descriptor is all zeros
        default_descriptors = jnp.zeros_like(centroids)

        # default extra_scores is all 0
        default_extra_scores = jax.tree_util.tree_map(
            lambda x: jnp.zeros(shape=(num_centroids,) + x.shape, dtype=x.dtype),
            extra_scores,
        )

        # default latest_batch_of_indices is num_centroids
        default_latest_batch_of_indices = num_centroids * jnp.ones(shape=(), dtype=jnp.int32)

        return cls(
            genotypes=default_genotypes,
            fitnesses=default_fitnesses,
            descriptors=default_descriptors,
            centroids=centroids,
            extra_scores=default_extra_scores,
            latest_batch_of_indices=default_latest_batch_of_indices,
        )
