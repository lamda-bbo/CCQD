import jax
from flax import linen as nn

from collections.abc import Sequence, Callable
from typing import Literal, Optional

from .mlp import MLP


class CNN(nn.Module):

    conv_features: Sequence[int]
    conv_kernel_sizes: Sequence[Sequence[int]]

    mlp_layer_sizes: tuple[int, ...]

    cnn_input_shape: Sequence[int]

    conv_activation: Callable[[jax.Array], jax.Array] = nn.relu
    conv_strides: Optional[Sequence[int | Sequence[int]]] = None
    conv_padding: Literal['SAME', 'VALID'] = 'VALID'

    mlp_activation: Callable[[jax.Array], jax.Array] = nn.relu
    mlp_final_activation: Optional[Callable[[jax.Array], jax.Array]] = None

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array | Sequence[jax.Array]:
        n_conv = len(self.conv_features)
        assert len(self.conv_kernel_sizes) == n_conv
        conv_strides = self.conv_strides
        if conv_strides is None or conv_strides == ():
            conv_strides = (1,) * n_conv
        assert len(conv_strides) == n_conv

        batch_shape = x.shape[:-1]
        if n_conv > 0:
            x = x.reshape(*batch_shape, *self.cnn_input_shape)

        for i in range(n_conv):
            x = nn.Conv(
                self.conv_features[i],
                self.conv_kernel_sizes[i],
                strides=conv_strides[i],
                padding=self.conv_padding,
            )(x)

        if n_conv > 0:
            x = x.reshape(*batch_shape, -1)

        y = MLP(
            layer_sizes=self.mlp_layer_sizes,
            activation=self.mlp_activation,
            final_activation=self.mlp_final_activation,
        )(x)

        return y
