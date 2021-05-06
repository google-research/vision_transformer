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
import jax.nn.initializers
import jax.numpy as jnp
import jax.lax as lax

init = nn.initializers.lecun_normal()


class SpatialDense(nn.Module):
    mlp_dim: int

    @nn.compact
    def __call__(self, x):
        batch, space, *_ = x.shape
        weight = self.param(f'kernel', init, (space, self.mlp_dim))
        weight = lax.broadcast(weight, (batch,))
        out = lax.dot_general(weight, x, (((1,), (1,)), ((0,), (0,))))
        out += self.param(f'bias', jax.nn.initializers.zeros,
                          (1, self.mlp_dim, 1))
        return out


class MlpBlock(nn.Module):
    mlp_dim: int
    dense: nn.Module = nn.Dense

    @nn.compact
    def __call__(self, x):
        y = nn.LayerNorm()(x)
        y = self.dense(self.mlp_dim)(y)
        y = nn.gelu(y)
        return self.dense(x.shape[-1])(y)


class MixerBlock(nn.Module):
    """Mixer block layer."""
    tokens_mlp_dim: int
    channels_mlp_dim: int

    @nn.compact
    def __call__(self, x):
        x += MlpBlock(self.tokens_mlp_dim, SpatialDense)(x)
        x += MlpBlock(self.channels_mlp_dim, nn.Dense)(x)
        return x


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
        x = nn.Conv(self.hidden_dim, self.patches.size,
                    strides=self.patches.size, name='stem')(inputs)
        n, h, w, c = x.shape
        x = jnp.reshape(x, (n, h * w, c))
        for _ in range(self.num_blocks):
            x = MixerBlock(self.tokens_mlp_dim, self.channels_mlp_dim)(x)
        x = nn.LayerNorm(name='pre_head_layer_norm')(x)
        x = jnp.mean(x, axis=1)
        return nn.Dense(self.num_classes, kernel_init=nn.initializers.zeros,
                        name='head')(x)
