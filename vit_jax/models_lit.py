# Copyright 2022 Google LLC.
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

"""Models from Locked-image text Tuning.

See paper https://arxiv.org/abs/2111.07991
"""

import dataclasses
import os
from typing import Optional, Tuple

import flax.linen as nn
import jax.numpy as jnp
import ml_collections
from vit_jax import checkpoint
from vit_jax import models_vit
from vit_jax import preprocess

from flaxformer.architectures.bert import bert
from flaxformer.architectures.bert import configs


BASE_PATH = 'gs://vit_models/lit'


class BertModel(nn.Module):
  """BERT encoder with linear projection on last layer CLS token."""

  config: str
  num_classes: Optional[int] = None

  @nn.compact
  def __call__(self, tokens):
    out = {}

    batch_size, max_len = tokens.shape
    bert_model = bert.BertEncoder(**dataclasses.asdict({
        'base': configs.BertBaseConfig(),
        'large': configs.BertLargeConfig(),
    }[self.config]))
    x = out['transformed'] = bert_model(
        token_ids=tokens,
        position_ids=jnp.tile(
            jnp.arange(0, max_len, dtype=jnp.int32), [batch_size, 1]),
        segment_ids=jnp.zeros([batch_size, max_len], dtype=jnp.int32),
        input_mask=tokens.astype(jnp.bool_).astype(jnp.int32),
        enable_dropout=False,
    )

    x = out['pre_logits'] = x[:, 0]  # CLS token

    if self.num_classes:
      x = out['logits'] = nn.Dense(self.num_classes, name='head')(x)

    return x, out


class LitModel(nn.Module):
  """Locked-image text Tuning model.

  See paper https://arxiv.org/abs/2111.07991

  For examples, refer to Colab

  https://colab.research.google.com/github/google-research/vision_transformer/blob/main/lit.ipynb

  Attributes:
    image: Configuration for ViT image tower.
    text: Configuration for text tower.
    pp: Preprocessing configuration.
    out_dim: Size of optional image/text heads that are added to the towers.
    model_name: Refers to the key in `model_configs.MODEL_CONFIGS`.
  """

  image: ml_collections.ConfigDict
  text: ml_collections.ConfigDict
  pp: ml_collections.ConfigDict
  out_dim: Tuple[Optional[int], Optional[int]]
  model_name: str

  def load_variables(self, path=None, cache=True):
    """Loads variables.

    Args:
      path: Path to load params from. If not specified, then the parms will be
        loaded from the default public Cloud storage path, unless they exist in
        the current working directory.
      cache: If set to `True` and `path` is not specified (the default), then
        the files will be copied from Cloud and stored in the current working
        directory.

    Returns:
      The module variables, to be used with `model.apply()`
    """
    if path is None:
      local_path = f'{self.model_name}.npz'
      if not os.path.exists(local_path):
        path = f'{BASE_PATH}/{self.model_name}.npz'
        print('Loading params from cloud:', path)
        if cache:
          checkpoint.copy(path, local_path)
      if os.path.exists(local_path):
        print('\nReusing local copy:', local_path)
        path = local_path
    return {'params': checkpoint.load(path)}

  def get_pp(self, crop=False):
    """Returns a preprocessing function suitable for `tf.data.Dataset.map()`."""
    return preprocess.get_pp(
        vocab_path=f'{BASE_PATH}/{self.model_name}.txt',
        max_len=self.pp.max_len,
        size=self.pp.size,
        crop=crop)

  def get_tokenizer(self):
    """Returns a tokenizer."""
    return preprocess.BertTokenizer(
        vocab_path=f'{BASE_PATH}/{self.model_name}.txt',
        max_len=self.pp.max_len)

  def get_image_preprocessing(self, crop=False):
    """Returns a function to pre-process images (resize, value range)."""
    return preprocess.PreprocessImages(size=self.pp.size, crop=crop)

  @nn.compact
  def __call__(self, *, images=None, tokens=None):
    """Embeds images and/or tokens.

    Args:
      images: Batch of images, prepared with the function returned by
        `get_image_preprocessing()` or `get_pp()`.
      tokens: Batch of tokens, prepared with the function returned by
        `get_tokenizer()` or `get_pp()`.

    Returns:
      A tuple of `(zimg, ztxt, out)`, where `zimg` is a batch of embeddings for
      the images (or `None`, if images were not specified), `ztxt` is a batch
      of embeddings for the tokens (or `None`, if tokens were not specified),
      and `out` is a dictionary of additional values, such as `out['t']` that
      is the temperature multiplied with the vector dot products before the
      softmax is applied.
    """

    # Support calling without text or without images, for example for few-shot.
    ztxt, zimg = None, None
    out = {}
    out_dims = self.out_dim
    if isinstance(out_dims, int):
      out_dims = (out_dims, out_dims)

    if tokens is not None:
      # Embed the text:
      text_model = BertModel(
          **{
              'num_classes': out_dims[1],
              **(self.text or {})
          }, name='txt')

      ztxt, out_txt = text_model(tokens)
      for k, v in out_txt.items():
        out[f'txt/{k}'] = v

      # Normalize the embeddings the models give us.
      out['txt/norm'] = jnp.linalg.norm(ztxt, axis=1, keepdims=True)
      out['txt/normalized'] = ztxt = ztxt / (out['txt/norm'] + 1e-8)

    if images is not None:
      image_model = models_vit.VisionTransformer(
          **{
              **self.image,
              'num_classes': out_dims[0],
          }, name='img')  # pylint: disable=not-a-mapping
      zimg = image_model(images, train=False)

      # Normalize the embeddings the models give us.
      out['img/norm'] = jnp.linalg.norm(zimg, axis=1, keepdims=True)
      out['img/normalized'] = zimg = zimg / (out['img/norm'] + 1e-8)

    t = self.param('t', nn.initializers.zeros, (1,), jnp.float32)
    out['t'] = jnp.exp(t)

    return zimg, ztxt, out
