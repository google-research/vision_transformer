# Copyright 2024 Google LLC.
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

from typing import Callable, Sequence, TypeVar

from flax import linen as nn
import jax.numpy as jnp

T = TypeVar('T')


def weight_standardize(w, axis, eps):
  """Subtracts mean and divides by standard deviation."""
  w = w - jnp.mean(w, axis=axis)
  w = w / (jnp.std(w, axis=axis) + eps)
  return w


class StdConv(nn.Conv):
  """Convolution with weight standardization."""

  def param(self,
            name: str,
            init_fn: Callable[..., T],
            *init_args) -> T:
    param = super().param(name, init_fn, *init_args)
    if name == 'kernel':
      param = weight_standardize(param, axis=[0, 1, 2], eps=1e-5)
    return param


class ResidualUnit(nn.Module):
  """Bottleneck ResNet block."""

  features: int
  strides: Sequence[int] = (1, 1)

  @nn.compact
  def __call__(self, x):
    needs_projection = (
        x.shape[-1] != self.features * 4 or self.strides != (1, 1))

    residual = x
    if needs_projection:
      residual = StdConv(
          features=self.features * 4,
          kernel_size=(1, 1),
          strides=self.strides,
          use_bias=False,
          name='conv_proj')(
              residual)
      residual = nn.GroupNorm(name='gn_proj')(residual)

    y = StdConv(
        features=self.features,
        kernel_size=(1, 1),
        use_bias=False,
        name='conv1')(
            x)
    y = nn.GroupNorm(name='gn1')(y)
    y = nn.relu(y)
    y = StdConv(
        features=self.features,
        kernel_size=(3, 3),
        strides=self.strides,
        use_bias=False,
        name='conv2')(
            y)
    y = nn.GroupNorm(name='gn2')(y)
    y = nn.relu(y)
    y = StdConv(
        features=self.features * 4,
        kernel_size=(1, 1),
        use_bias=False,
        name='conv3')(
            y)

    y = nn.GroupNorm(name='gn3', scale_init=nn.initializers.zeros)(y)
    y = nn.relu(residual + y)
    return y


class ResNetStage(nn.Module):
  """A ResNet stage."""

  block_size: Sequence[int]
  nout: int
  first_stride: Sequence[int]

  @nn.compact
  def __call__(self, x):
    x = ResidualUnit(self.nout, strides=self.first_stride, name='unit1')(x)
    for i in range(1, self.block_size):
      x = ResidualUnit(self.nout, strides=(1, 1), name=f'unit{i + 1}')(x)
    return x
