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

"""Utility code to create fake pre-trained checkpoints."""

import os

import dataclasses
import flax
import jax
import jax.numpy as jnp
import numpy as np

from vit_jax import models


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


def _tree_flatten_with_names(tree):
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
  vals, tree_def = jax.tree.flatten(tree)

  # "Fake" token tree that is use to track jax internal tree traversal and
  # adjust our custom tree traversal to be compatible with it.
  tokens = range(len(vals))
  token_tree = tree_def.unflatten(tokens)
  val_names, perm = zip(*_traverse_with_names(token_tree))
  inv_perm = np.argsort(perm)

  # Custom traversal should visit the same number of leaves.
  assert len(val_names) == len(vals)

  return [(val_names[i], v) for i, v in zip(inv_perm, vals)], tree_def


def _save(data, path):
  """Util for checkpointing: saves jax pytree objects to the disk."""
  names_and_vals, _ = _tree_flatten_with_names(data)
  os.makedirs(os.path.dirname(path), exist_ok=True)
  with open(path, 'wb') as f:
    np.savez(f, **{k: v for k, v in names_and_vals})


def create_checkpoint(model_config, path):
  """Initializes model and stores weights in specified path."""
  model = models.VisionTransformer(num_classes=1, **model_config)
  variables = model.init(
      jax.random.PRNGKey(0),
      jnp.ones([1, 16, 16, 3], jnp.float32),
      train=False,
  )
  _save(variables['params'], path)
