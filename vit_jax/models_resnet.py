import flax.nn as nn
import jax.numpy as jnp


def weight_standardize(w, axis, eps):
  """Subtracts mean and divides by standard deviation."""
  w = w - jnp.mean(w, axis=axis)
  w = w / (jnp.std(w, axis=axis) + eps)
  return w


class StdConv(nn.Conv):
  """Convolution with weight standardization."""

  def param(self, name, shape, initializer):
    param = super().param(name, shape, initializer)
    if name == 'kernel':
      param = weight_standardize(param, axis=[0, 1, 2], eps=1e-5)
    return param


class ResidualUnit(nn.Module):
  """Bottleneck ResNet block."""

  def apply(self, x, nout, strides=(1, 1)):
    needs_projection = x.shape[-1] != nout * 4 or strides != (1, 1)

    residual = x
    if needs_projection:
      residual = StdConv(residual, nout * 4, (1, 1), strides, bias=False,
                         name='conv_proj')
      residual = nn.GroupNorm(residual, name='gn_proj')

    y = StdConv(x, nout, (1, 1), bias=False, name='conv1')
    y = nn.GroupNorm(y, name='gn1')
    y = nn.relu(y)
    y = StdConv(y, nout, (3, 3), strides, bias=False, name='conv2')
    y = nn.GroupNorm(y, name='gn2')
    y = nn.relu(y)
    y = StdConv(y, nout * 4, (1, 1), bias=False, name='conv3')

    y = nn.GroupNorm(y, name='gn3', scale_init=nn.initializers.zeros)
    y = nn.relu(residual + y)
    return y


class ResNetStage(nn.Module):
  """A ResNet stage."""

  def apply(self, x, block_size, nout, first_stride):
    x = ResidualUnit(x, nout, strides=first_stride, name='unit1')
    for i in range(1, block_size):
      x = ResidualUnit(x, nout, strides=(1, 1), name=f'unit{i + 1}')
    return x
