import optax
import flax.core
import qdax.core.emitters.mutation_operators
from qdax.types import RNGKey

from collections.abc import Callable
from typing import Final


OPTIMIZERS: Final[dict[str, Callable[..., optax.GradientTransformation]]] = {
    'SGD': optax.sgd,
    'Adam': optax.adam,
}


OPERATORS: Final[dict[str, Callable[..., tuple[flax.core.scope.VariableDict, RNGKey]]]] = {
    'Iso-LineDD': qdax.core.emitters.mutation_operators.isoline_variation,  # type: ignore
}
