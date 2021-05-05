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

import collections
import io
import os
import re

from absl import logging
import dataclasses
import flax
import flax.training.checkpoints
import jax
import jax.numpy as jnp
import numpy as np
import scipy
from tensorflow.io import gfile  # pylint: disable=import-error

_MODULE_NUM_RE = re.compile(r'(.*)_\d+$')


def _flatten_dict(d, parent_key='', sep='/'):
  """Flattens a dictionary, keeping empty leaves."""
  items = []
  for k, v in d.items():
    path = parent_key + sep + k if parent_key else k
    if isinstance(v, collections.Mapping):
      items.extend(_flatten_dict(v, path, sep=sep).items())
    else:
      items.append((path, v))

  # Keeps the empty dict if it was set explicitly.
  if parent_key and not d:
    items.append((parent_key, {}))

  return dict(items)


def convert_pre_linen_pytree(params):
  """Converts a pre-Linen parameter pytree.

  In pre-Linen API submodules were numbered incrementially, independent of the
  submodule class. With Linen this behavior has changed to keep separate
  submodule counts per module class.

  Consider the following module:

    class Model(nn.Module):
      @nn.compact
      def __call__(self, x):
        x = nn.Conv(1, 1)(x)
        x = nn.Dense(1)(x)
        return x

  In pre-Linen the resulting params would have had the structure:
    {'Conv_0': { ... }, 'Dense_1': { ... } }

  With Linen the resulting params would instead have had the structure:
    {'Conv_0': { ... }, 'Dense_0': { ... } }

  Args:
    params: Parameter pytree in pre-Linen format. If the pytree is already in
      Linen format, then the returned pytree has an identical structure.

  Returns:
    Parameter pytree with Linen submodule naming.
  """
  if isinstance(params, flax.core.FrozenDict):
    params = flax.core.unfreeze(params)
  if not isinstance(params, dict):
    return params
  params_renamed = {}
  counts = {}
  names = flax.training.checkpoints.natural_sort(params.keys())
  for name in names:
    value = params[name]
    match = _MODULE_NUM_RE.match(name)
    if match:
      module = match.group(1)
      num = counts.get(module, 0)
      name = f'{module}_{num}'
      counts[module] = num + 1
    params_renamed[name] = convert_pre_linen_pytree(value)
  return params_renamed


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


def _traverse_with_names(tree):
  """Traverses nested dicts/dataclasses and emits (leaf_name, leaf_val)."""
  if dataclasses.is_dataclass(tree):
    tree = flax.serialization.to_state_dict(tree)
  if isinstance(tree, dict) or isinstance(tree, flax.core.FrozenDict):
    keys = sorted(tree.keys())
    for key in keys:
      for path, v in _traverse_with_names(tree[key]):
        yield (key + '/' + path).rstrip('/'), v
  else:
    yield '', tree


def tree_flatten_with_names(tree):
  """Populates tree_flatten with leaf names.

  This function populates output of tree_flatten with leaf names, using a
  custom traversal that produces names is provided. The custom traversal does
  NOT have to traverse tree in the same order as jax, as we take care of
  automatically aligning jax' and custom traversals.

  Args:
    tree: python tree.

  Returns:
    A list of values with names: [(name, value), ...]
  """
  vals, tree_def = jax.tree_flatten(tree)

  # "Fake" token tree that is use to track jax internal tree traversal and
  # adjust our custom tree traversal to be compatible with it.
  tokens = range(len(vals))
  token_tree = tree_def.unflatten(tokens)
  val_names, perm = zip(*_traverse_with_names(token_tree))
  inv_perm = np.argsort(perm)

  # Custom traversal should visit the same number of leaves.
  assert len(val_names) == len(vals)

  return [(val_names[i], v) for i, v in zip(inv_perm, vals)], tree_def


def save(data, path):
  """Util for checkpointing: saves jax pytree objects to the disk.

  These checkpoints can later be recovered with `load()`.

  Args:
    data: arbitrary jax pytree to be saved.
    path: a path to save the data.
  """
  names_and_vals, _ = tree_flatten_with_names(data)
  io_buffer = io.BytesIO()

  # savez uses `seek()` API call, which is not supported by cns. Thus, we first
  # write the checkpoint to the temp buffer and then write it to the disk.
  np.savez(io_buffer, **{k: v for k, v in names_and_vals})

  # In order to be robust to interruptions we first save checkpoint to the
  # temporal file and then move to actual path name.
  path_tmp = path + '-TEMPORARY'
  gfile.makedirs(os.path.dirname(path))
  with gfile.GFile(path_tmp, 'wb') as f:
    f.write(io_buffer.getvalue())
  gfile.rename(path_tmp, path, overwrite=True)


def load(path):
  """Loads params from a checkpoint previously stored with `save()`."""
  with gfile.GFile(path, 'rb') as f:
    ckpt_dict = np.load(f, allow_pickle=False)
    keys, values = zip(*list(ckpt_dict.items()))
  return convert_pre_linen_pytree(recover_tree(keys, values))


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
      ntok_new = posemb_new.shape[1]

      if model_config.classifier == 'token':
        posemb_tok, posemb_grid = posemb[:, :1], posemb[0, 1:]
        ntok_new -= 1
      else:
        posemb_tok, posemb_grid = posemb[:, :0], posemb[0]

      gs_old = int(np.sqrt(len(posemb_grid)))
      gs_new = int(np.sqrt(ntok_new))
      logging.info('load_pretrained: grid-size from %s to %s', gs_old, gs_new)
      posemb_grid = posemb_grid.reshape(gs_old, gs_old, -1)

      zoom = (gs_new / gs_old, gs_new / gs_old, 1)
      posemb_grid = scipy.ndimage.zoom(posemb_grid, zoom, order=1)
      posemb_grid = posemb_grid.reshape(1, gs_new * gs_new, -1)
      posemb = jnp.array(np.concatenate([posemb_tok, posemb_grid], axis=1))
      restored_params['Transformer']['posembed_input']['pos_embedding'] = posemb

  return flax.core.freeze(restored_params)
