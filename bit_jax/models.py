# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import jax
import jax.numpy as jnp

import flax.nn as nn


def fixed_padding(x, kernel_size):
  pad_total = kernel_size - 1
  pad_beg = pad_total // 2
  pad_end = pad_total - pad_beg

  x = jax.lax.pad(x, 0.0,
                  ((0, 0, 0),
                   (pad_beg, pad_end, 0), (pad_beg, pad_end, 0),
                   (0, 0, 0)))
  return x


def standardize(x, axis, eps):
  x = x - jnp.mean(x, axis=axis, keepdims=True)
  x = x / jnp.sqrt(jnp.mean(jnp.square(x), axis=axis, keepdims=True) + eps)
  return x


class GroupNorm(nn.Module):
  """Group normalization (arxiv.org/abs/1803.08494)."""

  def apply(self, x, num_groups=32):

    input_shape = x.shape
    group_shape = x.shape[:-1] + (num_groups, x.shape[-1] // num_groups)

    x = x.reshape(group_shape)

    # Standardize along spatial and group dimensions
    x = standardize(x, axis=[1, 2, 4], eps=1e-5)
    x = x.reshape(input_shape)

    bias_scale_shape = tuple([1, 1, 1] + [input_shape[-1]])
    x = x * self.param('scale', bias_scale_shape, nn.initializers.ones)
    x = x + self.param('bias', bias_scale_shape, nn.initializers.zeros)
    return x


class StdConv(nn.Conv):

  def param(self, name, shape, initializer):
    param = super().param(name, shape, initializer)
    if name == 'kernel':
      param = standardize(param, axis=[0, 1, 2], eps=1e-10)
    return param


class RootBlock(nn.Module):

  def apply(self, x, width):
    x = fixed_padding(x, 7)
    x = StdConv(x, width, (7, 7), (2, 2),
                padding="VALID",
                bias=False,
                name="conv_root")

    x = fixed_padding(x, 3)
    x = nn.max_pool(x, (3, 3), strides=(2, 2), padding="VALID")

    return x


class ResidualUnit(nn.Module):
  """Bottleneck ResNet block."""

  def apply(self, x, nout, strides=(1, 1)):
    x_shortcut = x
    needs_projection = x.shape[-1] != nout * 4 or strides != (1, 1)

    group_norm = GroupNorm
    conv = StdConv.partial(bias=False)

    x = group_norm(x, name="gn1")
    x = nn.relu(x)
    if needs_projection:
      x_shortcut = conv(x, nout * 4, (1, 1), strides, name="conv_proj")
    x = conv(x, nout, (1, 1), name="conv1")

    x = group_norm(x, name="gn2")
    x = nn.relu(x)
    x = fixed_padding(x, 3)
    x = conv(x, nout, (3, 3), strides, name="conv2", padding='VALID')

    x = group_norm(x, name="gn3")
    x = nn.relu(x)
    x = conv(x, nout * 4, (1, 1), name="conv3")

    return x + x_shortcut


class ResidualBlock(nn.Module):

  def apply(self, x, block_size, nout, first_stride):
    x = ResidualUnit(
        x, nout, strides=first_stride,
        name="unit01")
    for i in range(1, block_size):
      x = ResidualUnit(
          x, nout, strides=(1, 1),
          name=f"unit{i+1:02d}")
    return x


class ResNet(nn.Module):
  """ResNetV2."""

  def apply(self, x, num_classes=1000,
            width_factor=1, num_layers=50):
    block_sizes = _block_sizes[num_layers]

    width = 64 * width_factor

    root_block = RootBlock.partial(width=width)
    x = root_block(x, name='root_block')

    # Blocks
    for i, block_size in enumerate(block_sizes):
      x = ResidualBlock(x, block_size, width * 2 ** i,
                        first_stride=(1, 1) if i == 0 else (2, 2),
                        name=f"block{i + 1}")

    # Pre-head
    x = GroupNorm(x, name='norm-pre-head')
    x = nn.relu(x)
    x = jnp.mean(x, axis=(1, 2))

    # Head
    x = nn.Dense(x, num_classes, name="conv_head",
                 kernel_init=nn.initializers.zeros)

    return x.astype(jnp.float32)


_block_sizes = {
      50: [3, 4, 6, 3],
      101: [3, 4, 23, 3],
      152: [3, 8, 36, 3],
  }


KNOWN_MODELS = dict(
  [(bit + f'-R{l}x{w}', ResNet.partial(num_layers=l, width_factor=w))
   for bit in ['BiT-S', 'BiT-M']
   for l, w in [(50, 1), (50, 3), (101, 1), (152, 2), (101, 3), (152, 4)]]
)
