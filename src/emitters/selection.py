import jax
import jax.numpy as jnp
import qdax.core.containers.mapelites_repertoire
from qdax.types import RNGKey

import functools
from collections.abc import Callable
from typing import Final

from ..utils import jax_jit


Selector = Callable[
    [
        qdax.core.containers.mapelites_repertoire.MapElitesRepertoire,
        RNGKey,
        int,
    ],
    tuple[jax.Array, RNGKey]
]


@functools.partial(jax_jit, static_argnames=("num_samples",))
def random_selection(
    repertoire: qdax.core.containers.mapelites_repertoire.MapElitesRepertoire,
    random_key: RNGKey,
    num_samples: int,
) -> tuple[jax.Array, RNGKey]:
    repertoire_empty = repertoire.fitnesses == -jnp.inf
    p = (1.0 - repertoire_empty) / jnp.sum(1.0 - repertoire_empty)

    random_key, subkey = jax.random.split(random_key)
    selected_indices = jax.random.choice(
        subkey, repertoire.fitnesses.shape[0], (num_samples,), p=p
    )
    return selected_indices, random_key


SELECTORS: Final[dict[str, Selector]] = {
    'Random': random_selection,
}
