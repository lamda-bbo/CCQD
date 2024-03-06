import jax
import jax.numpy as jnp
import qdax.core.containers.repertoire
import qdax.core.emitters.pbt_me_emitter
from qdax.types import Descriptor, ExtraScores, Fitness, Genotype

import functools
from typing import Optional

from ..utils import jax_jit


class PBTEmitter(qdax.core.emitters.pbt_me_emitter.PBTEmitter):
    @functools.partial(jax_jit, static_argnames=("self",))
    def state_update(
        self,
        emitter_state: qdax.core.emitters.pbt_me_emitter.PBTEmitterState,
        repertoire: qdax.core.containers.repertoire.Repertoire,
        genotypes: Optional[Genotype],
        fitnesses: Fitness,
        descriptors: Optional[Descriptor],
        extra_scores: ExtraScores,
    ) -> qdax.core.emitters.pbt_me_emitter.PBTEmitterState:
        # Look only at the fitness corresponding to emitter state individuals
        fitnesses = fitnesses[self._config.ga_population_size_per_device:]
        fitnesses = jnp.ravel(fitnesses)
        training_states = emitter_state.training_states
        replay_buffers = emitter_state.replay_buffers
        genotypes = (training_states, replay_buffers)

        # Incremental algorithm to gather top best among the population on each device
        # First exchange
        indices_to_share = jnp.arange(self._config.pg_population_size_per_device)
        num_best_local = int(
            self._config.pg_population_size_per_device
            * self._config.fraction_best_to_replace_from
        )
        indices_to_share = indices_to_share[:num_best_local]
        genotypes_to_share, fitnesses_to_share = jax.tree_util.tree_map(
            lambda x: x[indices_to_share], (genotypes, fitnesses)
        )
        gathered_genotypes, gathered_fitnesses = jax.tree_util.tree_map(
            lambda x: jnp.concatenate(jnp.expand_dims(x, axis=0), axis=0),
            (genotypes_to_share, fitnesses_to_share),
        )

        genotypes_stacked, fitnesses_stacked = gathered_genotypes, gathered_fitnesses
        best_indices_stacked = jnp.argsort(-fitnesses_stacked)
        best_indices_stacked = best_indices_stacked[: self._num_best_to_replace_from]
        best_genotypes_local, best_fitnesses_local = jax.tree_util.tree_map(
            lambda x: x[best_indices_stacked], (genotypes_stacked, fitnesses_stacked)
        )

        # Define loop fn for the other exchanges
        def _loop_fn(i, val):  # type: ignore
            best_genotypes_local, best_fitnesses_local = val
            indices_to_share = jax.lax.dynamic_slice(
                jnp.arange(self._config.pg_population_size_per_device),
                [i * self._num_to_exchange],
                [self._num_to_exchange],
            )
            genotypes_to_share, fitnesses_to_share = jax.tree_util.tree_map(
                lambda x: x[indices_to_share], (genotypes, fitnesses)
            )
            gathered_genotypes, gathered_fitnesses = jax.tree_util.tree_map(
                lambda x: jnp.concatenate(jnp.expand_dims(x, axis=0), axis=0),
                (genotypes_to_share, fitnesses_to_share),
            )

            genotypes_stacked, fitnesses_stacked = jax.tree_util.tree_map(
                lambda x, y: jnp.concatenate([x, y], axis=0),
                (gathered_genotypes, gathered_fitnesses),
                (best_genotypes_local, best_fitnesses_local),
            )

            best_indices_stacked = jnp.argsort(-fitnesses_stacked)
            best_indices_stacked = best_indices_stacked[
                : self._num_best_to_replace_from
            ]
            best_genotypes_local, best_fitnesses_local = jax.tree_util.tree_map(
                lambda x: x[best_indices_stacked],
                (genotypes_stacked, fitnesses_stacked),
            )
            return (best_genotypes_local, best_fitnesses_local)  # type: ignore

        # Incrementally get the top fraction_best_to_replace_from best individuals
        # on each device
        (best_genotypes_local, best_fitnesses_local) = jax.lax.fori_loop(
            lower=1,
            upper=int(1.0 // self._config.fraction_sort_exchange) + 1,
            body_fun=_loop_fn,
            init_val=(best_genotypes_local, best_fitnesses_local),
        )

        # Gather fitnesses from all devices to rank locally against it
        all_fitnesses = jax.tree_util.tree_map(
            lambda x: jnp.concatenate(jnp.expand_dims(x, axis=0), axis=0),
            fitnesses,
        )
        all_fitnesses = jnp.ravel(all_fitnesses)
        all_fitnesses = -jnp.sort(-all_fitnesses)
        random_key = emitter_state.random_key
        random_key, sub_key = jax.random.split(random_key)
        best_genotypes = jax.tree_util.tree_map(
            lambda x: jax.random.choice(
                sub_key, x, shape=(len(fitnesses),), replace=True
            ),
            best_genotypes_local,
        )
        best_training_states, best_replay_buffers = best_genotypes

        # Resample hyper-params
        best_training_states = jax.vmap(
            best_training_states.__class__.resample_hyperparams
        )(best_training_states)

        # Replace by individuals from the best
        lower_bound = all_fitnesses[-self._num_to_replace_from_best]
        cond = fitnesses <= lower_bound

        training_states = jax.tree_util.tree_map(
            lambda x, y: jnp.where(
                jnp.expand_dims(
                    cond, axis=tuple([-(i + 1) for i in range(x.ndim - 1)])
                ),
                x,
                y,
            ),
            best_training_states,
            training_states,
        )
        replay_buffers = jax.tree_util.tree_map(
            lambda x, y: jnp.where(
                jnp.expand_dims(
                    cond, axis=tuple([-(i + 1) for i in range(x.ndim - 1)])
                ),
                x,
                y,
            ),
            best_replay_buffers,
            replay_buffers,
        )

        # Replacing with samples from the ME repertoire
        if self._num_to_replace_from_samples > 0:
            me_samples, random_key = repertoire.sample(
                random_key, self._config.pg_population_size_per_device
            )
            # Resample hyper-params
            me_samples = jax.vmap(me_samples.__class__.resample_hyperparams)(me_samples)
            upper_bound = all_fitnesses[
                -self._num_to_replace_from_best - self._num_to_replace_from_samples
            ]
            cond = jnp.logical_and(fitnesses <= upper_bound, fitnesses >= lower_bound)
            training_states = jax.tree_util.tree_map(
                lambda x, y: jnp.where(
                    jnp.expand_dims(
                        cond, axis=tuple([-(i + 1) for i in range(x.ndim - 1)])
                    ),
                    x,
                    y,
                ),
                me_samples,
                training_states,
            )

        # Train the agents
        env_states = emitter_state.env_states
        # Init optimizers state before training the population
        training_states = jax.vmap(training_states.__class__.init_optimizers_states)(
            training_states
        )
        (training_states, env_states, replay_buffers), metrics = self._train_fn(
            training_states, env_states, replay_buffers
        )
        # Empty optimizers states to avoid storing the info in the RAM
        # and having too heavy repertoires
        training_states = jax.vmap(training_states.__class__.empty_optimizers_states)(
            training_states
        )

        # Update emitter state
        emitter_state = emitter_state.replace(
            training_states=training_states,
            replay_buffers=replay_buffers,
            env_states=env_states,
            random_key=random_key,
        )
        return emitter_state  # type: ignore
