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

from vit_jax import configs
from vit_jax import models_resnet


class IdentityLayer(nn.Module):
  """Identity layer, convenient for giving a name to an array."""

  def apply(self, x):
    return x


class AddPositionEmbs(nn.Module):
  """Adds (optionally learned) positional embeddings to the inputs."""

  def apply(self, inputs, inputs_positions=None, posemb_init=None):
    """Applies AddPositionEmbs module.

    By default this layer uses a fixed sinusoidal embedding table. If a
    learned position embedding is desired, pass an initializer to
    posemb_init.

    Args:
      inputs: input data.
      inputs_positions: input position indices for packed sequences.
      posemb_init: positional embedding initializer.

    Returns:
      output: `(bs, timesteps, in_dim)`
    """
    # inputs.shape is (batch_size, seq_len, emb_dim).
    assert inputs.ndim == 3, ('Number of dimensions should be 3,'
                              ' but it is: %d' % inputs.ndim)
    pos_emb_shape = (1, inputs.shape[1], inputs.shape[2])
    pe = self.param('pos_embedding', pos_emb_shape, posemb_init)
    if inputs_positions is None:
      # Normal unpacked case:
      return inputs + pe
    else:
      # For packed data we need to use known position indices:
      return inputs + jnp.take(pe[0], inputs_positions, axis=0)


class MlpBlock(nn.Module):
  """Transformer MLP / feed-forward block."""

  def apply(self,
            inputs,
            mlp_dim,
            dtype=jnp.float32,
            out_dim=None,
            dropout_rate=0.1,
            deterministic=True,
            kernel_init=nn.initializers.xavier_uniform(),
            bias_init=nn.initializers.normal(stddev=1e-6)):
    """Applies Transformer MlpBlock module."""
    actual_out_dim = inputs.shape[-1] if out_dim is None else out_dim
    x = nn.Dense(
        inputs,
        mlp_dim,
        dtype=dtype,
        kernel_init=kernel_init,
        bias_init=bias_init)
    x = nn.gelu(x)
    x = nn.dropout(x, rate=dropout_rate, deterministic=deterministic)
    output = nn.Dense(
        x,
        actual_out_dim,
        dtype=dtype,
        kernel_init=kernel_init,
        bias_init=bias_init)
    output = nn.dropout(output, rate=dropout_rate, deterministic=deterministic)
    return output


class Encoder1DBlock(nn.Module):
  """Transformer encoder layer."""

  def apply(self,
            inputs,
            mlp_dim,
            dtype=jnp.float32,
            dropout_rate=0.1,
            attention_dropout_rate=0.1,
            deterministic=True,
            **attention_kwargs):
    """Applies Encoder1DBlock module.

    Args:
      inputs: input data.
      mlp_dim: dimension of the mlp on top of attention block.
      dtype: the dtype of the computation (default: float32).
      dropout_rate: dropout rate.
      attention_dropout_rate: dropout for attention heads.
      deterministic: bool, deterministic or not (to apply dropout).
      **attention_kwargs: kwargs passed to nn.SelfAttention

    Returns:
      output after transformer encoder block.
    """

    # Attention block.
    assert inputs.ndim == 3
    x = nn.LayerNorm(inputs, dtype=dtype)
    x = nn.SelfAttention(
        x,
        dtype=dtype,
        inputs_kv=x,
        attention_axis=(1,),
        causal_mask=False,
        kernel_init=nn.initializers.xavier_uniform(),
        broadcast_dropout=False,
        deterministic=deterministic,
        dropout_rate=attention_dropout_rate,
        **attention_kwargs)
    x = nn.dropout(x, rate=dropout_rate, deterministic=deterministic)
    x = x + inputs

    # MLP block.
    y = nn.LayerNorm(x, dtype=dtype)
    y = MlpBlock(
        y,
        mlp_dim=mlp_dim,
        dtype=dtype,
        dropout_rate=dropout_rate,
        deterministic=deterministic)

    return x + y


class Encoder(nn.Module):
  """Transformer Model Encoder for sequence to sequence translation."""

  def apply(self,
            inputs,
            num_layers,
            mlp_dim,
            inputs_positions=None,
            dropout_rate=0.1,
            train=False,
            **attention_kwargs):
    """Applies Transformer model on the inputs.

    Args:
      inputs: input data
      num_layers: number of layers
      mlp_dim: dimension of the mlp on top of attention block
      inputs_positions: input subsequence positions for packed examples.
      dropout_rate: dropout rate
      train: if it is training,
      **attention_kwargs: kwargs passed to nn.SelfAttention

    Returns:
      output of a transformer encoder.
    """
    assert inputs.ndim == 3  # (batch, len, emb)

    x = AddPositionEmbs(
        inputs,
        inputs_positions=inputs_positions,
        posemb_init=nn.initializers.normal(stddev=0.02),  # from BERT.
        name='posembed_input')
    x = nn.dropout(x, rate=dropout_rate, deterministic=not train)

    # Input Encoder
    for lyr in range(num_layers):
      x = Encoder1DBlock(
          x,
          mlp_dim=mlp_dim,
          dropout_rate=dropout_rate,
          deterministic=not train,
          name=f'encoderblock_{lyr}',
          **attention_kwargs)
    encoded = nn.LayerNorm(x, name='encoder_norm')

    return encoded


class VisionTransformer(nn.Module):
  """VisionTransformer."""

  def apply(self,
            x,
            num_classes=1000,
            train=False,
            resnet=None,
            patches=None,
            hidden_size=None,
            transformer=None,
            representation_size=None,
            classifier='gap'):

    # (Possibly partial) ResNet root.
    if resnet is not None:
      width = int(64 * resnet.width_factor)

      # Root block.
      x = models_resnet.StdConv(
          x, width, (7, 7), (2, 2), bias=False, name='conv_root')
      x = nn.GroupNorm(x, name='gn_root')
      x = nn.relu(x)
      x = nn.max_pool(x, (3, 3), strides=(2, 2), padding='SAME')

      # ResNet stages.
      x = models_resnet.ResNetStage(
          x, resnet.num_layers[0], width, first_stride=(1, 1), name='block1')
      for i, block_size in enumerate(resnet.num_layers[1:], 1):
        x = models_resnet.ResNetStage(
            x,
            block_size,
            width * 2**i,
            first_stride=(2, 2),
            name=f'block{i + 1}')

    n, h, w, c = x.shape

    # We can merge s2d+emb into a single conv; it's the same.
    x = nn.Conv(
        x,
        hidden_size, patches.size,
        strides=patches.size,
        padding='VALID',
        name='embedding')

    # Here, x is a grid of embeddings.

    # (Possibly partial) Transformer.
    if transformer is not None:
      n, h, w, c = x.shape
      x = jnp.reshape(x, [n, h * w, c])

      # If we want to add a class token, add it here.
      if classifier == 'token':
        cls = self.param('cls', (1, 1, c), nn.initializers.zeros)
        cls = jnp.tile(cls, [n, 1, 1])
        x = jnp.concatenate([cls, x], axis=1)

      x = Encoder(x, train=train, name='Transformer', **transformer)

    if classifier == 'token':
      x = x[:, 0]
    elif classifier == 'gap':
      x = jnp.mean(x, axis=list(range(1, x.ndim - 1)))  # (1,) or (1,2)

    if representation_size is not None:
      x = nn.Dense(x, representation_size, name='pre_logits')
      x = nn.tanh(x)
    else:
      x = IdentityLayer(x, name='pre_logits')

    x = nn.Dense(x, num_classes, name='head', kernel_init=nn.initializers.zeros)
    return x


CONFIGS = {
    'ViT-B_16': configs.get_b16_config(),
    'R50+ViT-B_16': configs.get_r50_b16_config(),
    'ViT-B_32': configs.get_b32_config(),
    'ViT-L_16': configs.get_l16_config(),
    'ViT-L_32': configs.get_l32_config(),
    'ViT-H_14': configs.get_h14_config(),
    'testing': configs.get_testing(),
}
KNOWN_MODELS = {
    name: VisionTransformer.partial(**config)
    for name, config in CONFIGS.items()
}
