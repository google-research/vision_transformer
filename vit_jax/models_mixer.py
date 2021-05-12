# Copyright 2021 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Any

import flax.linen as nn
import jax.numpy as jnp
from jax import lax

init = nn.initializers.lecun_normal()


def conv_dimension_numbers(_):
    return lax.ConvDimensionNumbers((0, 1, 2), (2, 1, 0), (0, 1, 2))


nn.linear._conv_dimension_numbers = conv_dimension_numbers


class ResMlpBlock(nn.Module):
    """
    MLP with input norm.
    Dense layers are either across spatial or feature dimension.
    Residual is added within the block
    """
    mlp_dim: int
    spatial: bool

    @nn.compact
    def __call__(self, x):
        dense = nn.Conv if self.spatial else nn.Dense
        y = nn.LayerNorm()(x)
        y = dense(self.mlp_dim, 1)(y)
        y = nn.gelu(y)
        y = dense(x.shape[1 if self.spatial else -1], 1)(y)
        return x + y


class MlpMixer(nn.Module):
    """Mixer architecture."""
    patches: Any
    num_classes: int
    num_blocks: int
    hidden_dim: int
    tokens_mlp_dim: int
    channels_mlp_dim: int

    @nn.compact
    def __call__(self, inputs, *, train):
        del train
        psize = self.patches.size
        n, h, w, c = inputs.shape
        x = jnp.reshape(inputs, (n, h // psize, psize, w // psize, psize, c))
        x = jnp.transpose(x, (0, 1, 3, 2, 4, 5))
        x = jnp.reshape(x, (n, h * w // psize ** 2, c * psize ** 2))
        x = nn.Dense(self.hidden_dim, name='stem')(x)
        for _ in range(self.num_blocks):
            x = ResMlpBlock(self.tokens_mlp_dim, True)(x)
            x = ResMlpBlock(self.channels_mlp_dim, False)(x)
        x = nn.LayerNorm(name='pre_head_layer_norm')(x)
        x = jnp.mean(x, axis=1)
        return nn.Dense(self.num_classes, kernel_init=nn.initializers.zeros,
                        name='head')(x)
