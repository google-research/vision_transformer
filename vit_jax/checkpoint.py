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

import collections
from collections import abc
import re

from absl import logging
import flax
from flax.training import checkpoints
import jax.numpy as jnp
import numpy as np
from packaging import version
import pandas as pd
import scipy.ndimage
from tensorflow.io import gfile  # pylint: disable=import-error
import tqdm


def _flatten_dict(d, parent_key='', sep='/'):
  """Flattens a dictionary, keeping empty leaves."""
  items = []
  for k, v in d.items():
    path = parent_key + sep + k if parent_key else k
    if isinstance(v, abc.Mapping):
      items.extend(_flatten_dict(v, path, sep=sep).items())
    else:
      items.append((path, v))

  # Keeps the empty dict if it was set explicitly.
  if parent_key and not d:
    items.append((parent_key, {}))

  return dict(items)


def inspect_params(*,
                   params,
                   expected,
                   fail_if_extra=True,
                   fail_if_missing=True):
  """Inspects whether the params are consistent with the expected keys."""
  params_flat = _flatten_dict(params)
  expected_flat = _flatten_dict(expected)
  missing_keys = expected_flat.keys() - params_flat.keys()
  extra_keys = params_flat.keys() - expected_flat.keys()

  # Adds back empty dict explicitly, to support layers without weights.
  # Context: FLAX ignores empty dict during serialization.
  empty_keys = set()
  for k in missing_keys:
    if isinstance(expected_flat[k], dict) and not expected_flat[k]:
      params[k] = {}
      empty_keys.add(k)
  missing_keys -= empty_keys

  if empty_keys:
    logging.warning('Inspect recovered empty keys:\n%s', empty_keys)
  if missing_keys:
    logging.info('Inspect missing keys:\n%s', missing_keys)
  if extra_keys:
    logging.info('Inspect extra keys:\n%s', extra_keys)

  if (missing_keys and fail_if_missing) or (extra_keys and fail_if_extra):
    raise ValueError(f'Missing params from checkpoint: {missing_keys}.\n'
                     f'Extra params in checkpoint: {extra_keys}.\n'
                     f'Restored params from checkpoint: {params_flat.keys()}.\n'
                     f'Expected params from code: {expected_flat.keys()}.')
  return params


def recover_tree(keys, values):
  """Recovers a tree as a nested dict from flat names and values.

  This function is useful to analyze checkpoints that are without need to access
  the exact source code of the experiment. In particular, it can be used to
  extract an reuse various subtrees of the scheckpoint, e.g. subtree of
  parameters.

  Args:
    keys: a list of keys, where '/' is used as separator between nodes.
    values: a list of leaf values.

  Returns:
    A nested tree-like dict.
  """
  tree = {}
  sub_trees = collections.defaultdict(list)
  for k, v in zip(keys, values):
    if '/' not in k:
      tree[k] = v
    else:
      k_left, k_right = k.split('/', 1)
      sub_trees[k_left].append((k_right, v))
  for k, kv_pairs in sub_trees.items():
    k_subtree, v_subtree = zip(*kv_pairs)
    tree[k] = recover_tree(k_subtree, v_subtree)
  return tree


def copy(src, dst, progress=True, block_size=1024 * 1024 * 10):
  """Copies a file with progress bar.

  Args:
    src: Source file. Path must be readable by `tf.io.gfile`.
    dst: Destination file. Path must be readable by `tf.io.gfile`.
    progress: Whether to show a progres bar.
    block_size: Size of individual blocks to be read/written.
  """
  stats = gfile.stat(src)
  n = int(np.ceil(stats.length / block_size))
  range_or_trange = tqdm.trange if progress else range
  with gfile.GFile(src, 'rb') as fin:
    with gfile.GFile(dst, 'wb') as fout:
      for _ in range_or_trange(n):
        fout.write(fin.read(block_size))


def load(path):
  """Loads params from a checkpoint previously stored with `save()`."""
  with gfile.GFile(path, 'rb') as f:
    ckpt_dict = np.load(f, allow_pickle=False)
    keys, values = zip(*list(ckpt_dict.items()))
  params = checkpoints.convert_pre_linen(recover_tree(keys, values))
  if isinstance(params, flax.core.FrozenDict):
    params = params.unfreeze()
  if version.parse(flax.__version__) >= version.parse('0.3.6'):
    params = _fix_groupnorm(params)
  return params


def _fix_groupnorm(params):
  # See https://github.com/google/flax/issues/1721
  regex = re.compile(r'gn(\d+|_root|_proj)$')

  def fix_gn(args):
    path, array = args
    if len(path) > 1 and regex.match(
        path[-2]) and path[-1] in ('bias', 'scale'):
      array = array.squeeze()
    return (path, array)

  return flax.traverse_util.unflatten_dict(
      dict(map(fix_gn,
               flax.traverse_util.flatten_dict(params).items())))


def load_pretrained(*, pretrained_path, init_params, model_config):
  """Loads/converts a pretrained checkpoint for fine tuning.

  Args:
    pretrained_path: File pointing to pretrained checkpoint.
    init_params: Parameters from model. Will be used for the head of the model
      and to verify that the model is compatible with the stored checkpoint.
    model_config: Configuration of the model. Will be used to configure the head
      and rescale the position embeddings.

  Returns:
    Parameters like `init_params`, but loaded with pretrained weights from
    `pretrained_path` and adapted accordingly.
  """

  restored_params = inspect_params(
      params=load(pretrained_path),
      expected=init_params,
      fail_if_extra=False,
      fail_if_missing=False)

  # The following allows implementing fine-tuning head variants depending on the
  # value of `representation_size` in the fine-tuning job:
  # - `None` : drop the whole head and attach a nn.Linear.
  # - same number as in pre-training means : keep the head but reset the last
  #    layer (logits) for the new task.
  if model_config.get('representation_size') is None:
    if 'pre_logits' in restored_params:
      logging.info('load_pretrained: drop-head variant')
      restored_params['pre_logits'] = {}
  restored_params['head']['kernel'] = init_params['head']['kernel']
  restored_params['head']['bias'] = init_params['head']['bias']

  if 'posembed_input' in restored_params.get('Transformer', {}):
    # Rescale the grid of position embeddings. Param shape is (1,N,1024)
    posemb = restored_params['Transformer']['posembed_input']['pos_embedding']
    posemb_new = init_params['Transformer']['posembed_input']['pos_embedding']
    if posemb.shape != posemb_new.shape:
      logging.info('load_pretrained: resized variant: %s to %s', posemb.shape,
                   posemb_new.shape)
      posemb = interpolate_posembed(
          posemb, posemb_new.shape[1], model_config.classifier == 'token')
      restored_params['Transformer']['posembed_input']['pos_embedding'] = posemb

  if version.parse(flax.__version__) >= version.parse('0.3.6'):
    restored_params = _fix_groupnorm(restored_params)

  return flax.core.freeze(restored_params)


def interpolate_posembed(posemb, num_tokens: int, has_class_token: bool):
  """Interpolate given positional embedding parameters into a new shape.

  Args:
    posemb: positional embedding parameters.
    num_tokens: desired number of tokens.
    has_class_token: True if the positional embedding parameters contain a
      class token.

  Returns:
    Positional embedding parameters interpolated into the new shape.
  """
  assert posemb.shape[0] == 1
  if has_class_token:
    posemb_tok, posemb_grid = posemb[:, :1], posemb[0, 1:]
    num_tokens -= 1
  else:
    posemb_tok, posemb_grid = posemb[:, :0], posemb[0, 0:]

  gs_old = int(np.sqrt(len(posemb_grid)))
  gs_new = int(np.sqrt(num_tokens))
  logging.info('interpolate_posembed: grid-size from %s to %s', gs_old, gs_new)
  assert gs_old ** 2 == len(posemb_grid), f'{gs_old ** 2} != {len(posemb_grid)}'
  assert gs_new ** 2 == num_tokens, f'{gs_new ** 2} != {num_tokens}'
  posemb_grid = posemb_grid.reshape(gs_old, gs_old, -1)

  zoom = (gs_new / gs_old, gs_new / gs_old, 1)
  posemb_grid = scipy.ndimage.zoom(posemb_grid, zoom, order=1)
  posemb_grid = posemb_grid.reshape(1, gs_new * gs_new, -1)
  return jnp.array(np.concatenate([posemb_tok, posemb_grid], axis=1))


def get_augreg_df(directory='gs://vit_models/augreg'):
  """Reads DataFrame describing AugReg models from GCS bucket.

  This function returns a dataframe that describes the models that were
  published as part of the paper "How to train your ViT? Data, Augmentation, and
  Regularization in Vision Transformers" (https://arxiv.org/abs/TODO).

  Note that every row in the dataset corresponds to a pre-training checkpoint
  (column "filename"), and a fine-tuning checkpoint (column "adapt_filename").
  Every pre-trained checkpoint is fine-tuned many times.

  Args:
    directory: Pathname of directory containing "index.csv"

  Returns:
    Dataframe with the following columns:
      - name: Name of the model, as used in descriptions in paper (e.g. "B/16",
        or "R26+S/32").
      - ds: Dataset used for pre-training: "i1k" (300 epochs), "i21k" (300
        epochs), and "i21k_30" (30 epochs).
      - lr: Learning rate used for pre-training.
      - aug: Data augmentation used for pre-training. Refer to paper for
        details.
      - wd: Weight decay used for pre-training.
      - do: Dropout used for pre-training.
      - sd: Stochastic depth used for pre-training.
      - best_val: Best accuracy on validation set that was reached during the
        pre-training. Note that "validation set" can refer to minival (meaning
        split from training set, as for example for "imagenet2012" dataset).
      - final_val: Final validation set accuracy.
      - final_test: Final testset accuracy (in cases where there is no official
        testset, like for "imagenet2012", this refers to the validation set).
      - adapt_ds: What dataset was used for fine-tuning.
      - adapt_lr: Learning rate used for fine-tuning.
      - adapt_steps: Number of steps used for fine-tuning (with a fixed batch
        size of 512).
      - adapt_resolution: Resolution that was used for fine-tuning.
      - adapt_final_val: Final validation accuracy after fine-tuning.
      - adapt_final_test: Final test accuracy after fine-tuning.
      - params: Number of parameters.
      - infer_samples_per_sec: Numbers of sample per seconds during inference on
        a V100 GPU (measured with `timm` implementation).
      - filename: Name of the pre-training checkpoint. Can be found at
        "gs://vit_models/augreg/{filename}.npz".
      - adapt_filename: Name of the fine-tuning checkpoint.
  """
  with gfile.GFile(f'{directory}/index.csv') as f:
    return pd.read_csv(f)
