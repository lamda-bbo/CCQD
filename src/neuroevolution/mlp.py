import jax
import flax.linen as nn

import logging
from collections.abc import Callable
from typing import Optional, Any


_log = logging.getLogger(__name__)


class MLP(nn.Module):
    """MLP module."""

    layer_sizes: tuple[int, ...]
    activation: Callable[[jax.Array], jax.Array] = nn.relu
    kernel_init: Callable[..., Any] = jax.nn.initializers.lecun_uniform()
    final_activation: Optional[Callable[[jax.Array], jax.Array]] = None
    bias: bool = True
    kernel_init_final: Optional[Callable[..., Any]] = None

    @nn.compact
    def __call__(self, data: jax.Array) -> jax.Array:

        if len(self.layer_sizes) == 0:
            _ = nn.Dense(2)(data)
            return data

        hidden = data if isinstance(data, jax.Array) else data[0]

        for i, hidden_size in enumerate(self.layer_sizes):

            if i != len(self.layer_sizes) - 1:
                hidden = nn.Dense(
                    hidden_size,
                    # name=f"hidden_{i}", with this version of flax, changing the name
                    # changes the initialization
                    kernel_init=self.kernel_init,
                    use_bias=self.bias,
                )(hidden)
                hidden = self.activation(hidden)

            else:
                if self.kernel_init_final is not None:
                    kernel_init = self.kernel_init_final
                else:
                    kernel_init = self.kernel_init

                hidden = nn.Dense(
                    hidden_size,
                    # name=f"hidden_{i}",
                    kernel_init=kernel_init,
                    use_bias=self.bias,
                )(hidden)

                if self.final_activation is not None:
                    hidden = self.final_activation(hidden)

        return hidden
