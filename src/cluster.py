import jax
import jax.numpy as jnp
import flax.struct
import numpy as np

from dataclasses import dataclass
import logging
import functools
from typing import Literal, Optional, TypeVar

from .utils import tree_getitem, calc_dist, jax_jit


_log = logging.getLogger(__name__)


class KMeansState(flax.struct.PyTreeNode):

    cluster_centers_: jax.Array
    labels_: jax.Array
    diff_: jax.Array
    n_iter_: jax.Array
    inertia_: jax.Array
    random_key_: jax.random.KeyArray


_TKMeansState = TypeVar('_TKMeansState', bound=KMeansState)


@dataclass
class KMeans:

    k: int
    init: str = 'k-means++'
    n_init: int | Literal['auto'] = 10
    max_iter: int = 300
    tol: float = 0.0001
    n_local_trials: Optional[int] = None
    StateType: type[KMeansState] = KMeansState

    @staticmethod
    @functools.partial(jax_jit, donate_argnums=0)
    def _step(state: _TKMeansState, x: jax.Array, mask: jax.Array) -> _TKMeansState:

        dist = calc_dist(x, state.cluster_centers_)
        labels_ = jnp.argmin(dist, axis=1)

        def calc_cluster_centers(
            cluster_idx: jax.Array, random_key: jax.random.KeyArray
        ) -> jax.Array:
            cluster_mask = jnp.logical_and(labels_ == cluster_idx, mask)
            cluster_mask_sum = jnp.sum(cluster_mask)
            cluster_center = jnp.where(
                cluster_mask_sum > 0,
                jnp.sum(x * cluster_mask[:, jnp.newaxis], axis=0) / cluster_mask_sum,
                jax.random.choice(random_key, x),
            )
            return cluster_center

        random_key_, subkey = jax.random.split(state.random_key_)
        keys = jax.random.split(subkey, state.cluster_centers_.shape[0])
        calc_cluster_centers_fn = jax.vmap(calc_cluster_centers)
        cluster_centers_ = calc_cluster_centers_fn(
            jnp.arange(state.cluster_centers_.shape[0]), keys
        )

        return state.replace(
            cluster_centers_=cluster_centers_,
            labels_=labels_,
            diff_=jnp.sum(jnp.square(cluster_centers_ - state.cluster_centers_)),
            n_iter_=state.n_iter_ + 1,
            random_key_=random_key_,
        )

    def _kmeans_single(self, x: jax.Array, mask: jax.Array, state: _TKMeansState) -> _TKMeansState:

        @jax_jit
        def cond_fn(state: _TKMeansState) -> jax.Array:
            return jnp.logical_and(state.n_iter_ < self.max_iter, state.diff_ > self.tol)

        @functools.partial(jax_jit, donate_argnums=0)
        def body_fn(state: _TKMeansState) -> _TKMeansState:
            return self._step(state, x, mask)

        state = jax.lax.while_loop(
            cond_fn,
            body_fn,
            state,
        )

        state.replace(
            inertia_=jnp.dot(
                jnp.sum(jnp.square(x - state.cluster_centers_[state.labels_]), axis=-1),
                mask,
            )
        )

        return state

    def _kmeans_random_single(self, random_key: jax.random.KeyArray, x: jax.Array, mask: jax.Array):
        p = mask / jnp.sum(mask)
        random_key, subkey = jax.random.split(random_key)
        idx = jax.random.choice(subkey, x.shape[0], (self.k,), replace=False, p=p)
        cluster_centers_ = x[idx]
        return self._kmeans_single(x, mask, self.StateType(
            cluster_centers_=cluster_centers_,
            labels_=jnp.zeros(x.shape[0], dtype=jnp.int32),
            diff_=jnp.array(jnp.inf, dtype=jnp.float32),
            n_iter_=jnp.array(0, dtype=jnp.int32),
            inertia_=jnp.array(0, dtype=jnp.float32),
            random_key_=random_key,
        ))

    @staticmethod
    @functools.partial(jax_jit, static_argnames='n_local_trials', donate_argnums=(0, 1, 2))
    def _kmeans_pp_step(
        closest_dist_sq: jax.Array,
        current_pot: jax.Array,
        x: jax.Array,
        n_local_trials: int,
        random_key: jax.random.KeyArray,
    ) -> tuple[tuple[jax.Array, jax.Array], jax.Array]:
        candidate_ids = jax.random.choice(
            random_key, x.shape[0], (n_local_trials,), p=closest_dist_sq / current_pot
        )

        distance_to_candidates: jax.Array = calc_dist(x[candidate_ids], x)

        distance_to_candidates = jnp.minimum(closest_dist_sq, distance_to_candidates)
        candidates_pot = distance_to_candidates.sum(axis=1)

        best_candidate = candidates_pot.argmin()
        current_pot = candidates_pot[best_candidate]
        closest_dist_sq = distance_to_candidates[best_candidate]
        best_candidate = candidate_ids[best_candidate]

        return (closest_dist_sq, current_pot), x[best_candidate]

    def _kmeans_pp_single(
        self,
        random_key: jax.random.KeyArray,
        x: jax.Array,
        mask: jax.Array,
    ) -> KMeansState:
        random_key, first_key, others_key = jax.random.split(random_key, 3)
        keys = jax.random.split(others_key, self.k - 1)
        p = mask / jnp.sum(mask)
        idx = jax.random.choice(first_key, x.shape[0], p=p)
        closest_dist_sq: jax.Array = calc_dist(x, x[idx]) * mask

        @functools.partial(jax_jit, donate_argnums=0)
        def body_fn(
            carry: tuple[jax.Array, jax.Array],
            random_key: jax.random.KeyArray,
        ) -> tuple[tuple[jax.Array, jax.Array], jax.Array]:
            assert self.n_local_trials is not None
            return self._kmeans_pp_step(*carry, x, self.n_local_trials, random_key)

        _, cluster_centers_ = jax.lax.scan(
            body_fn,
            (closest_dist_sq, closest_dist_sq.sum()),
            keys,
        )

        cluster_centers_ = jnp.concatenate((x[0][jnp.newaxis], cluster_centers_))

        return self._kmeans_single(x, mask, self.StateType(
            cluster_centers_=cluster_centers_,
            labels_=jnp.zeros(x.shape[0], dtype=jnp.int32),
            diff_=jnp.array(jnp.inf, dtype=jnp.float32),
            n_iter_=jnp.array(0, dtype=jnp.int32),
            inertia_=jnp.array(0, dtype=jnp.float32),
            random_key_=random_key,
        ))

    def fit(
        self,
        random_key: jax.random.KeyArray,
        x: jax.Array,
        mask: Optional[jax.Array] = None,
    ) -> KMeansState:
        assert x.shape[0] >= self.k

        n_init = self.n_init
        match self.init:
            case 'random':
                if n_init == 'auto':
                    n_init = 10
                kmeans_fn = self._kmeans_random_single
            case 'k-means++':
                if n_init == 'auto':
                    n_init = 1
                if self.n_local_trials is None:
                    self.n_local_trials = 2 + int(np.log(self.k))
                kmeans_fn = self._kmeans_pp_single
            case _:
                raise NotImplementedError(f"No such method of initialization: '{self.init}'")
        assert isinstance(n_init, int)

        if mask is None:
            mask = jnp.ones(x.shape[0])

        kmeans_fn = functools.partial(kmeans_fn, x=x, mask=mask)
        keys = jax.random.split(random_key, n_init)

        states = jax.vmap(kmeans_fn)(keys)
        idx = jnp.argmin(states.inertia_)
        state = tree_getitem(states, idx)
        return state
