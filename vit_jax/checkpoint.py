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


# --- MODIFIED load_pretrained FUNCTION ---
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

    # Head modification logic (remains the same)
    if model_config.get('representation_size') is None:
        if 'pre_logits' in restored_params:
            logging.info('load_pretrained: drop-head variant')
            restored_params['pre_logits'] = {}
    # Always reset the final classification layer
    if 'head' in restored_params and 'head' in init_params:
        logging.info('load_pretrained: resetting final classifier layer')
        restored_params['head']['kernel'] = init_params['head']['kernel']
        restored_params['head']['bias'] = init_params['head']['bias']
    else:
         logging.warning('Could not find head parameters in restored or init params.')


    # --- Positional Embedding Interpolation ---
    # Check if positional embeddings exist in both the restored and initial parameters
    if 'Transformer' in restored_params and \
       'Transformer' in init_params and \
       'posembed_input' in restored_params.get('Transformer', {}) and \
       'posembed_input' in init_params.get('Transformer', {}):

        posemb_restored = restored_params['Transformer']['posembed_input']['pos_embedding']
        posemb_init = init_params['Transformer']['posembed_input']['pos_embedding']

        # Check if the shapes are different (indicating different resolutions/patch counts)
        if posemb_restored.shape != posemb_init.shape:
            logging.info('load_pretrained: resizing positional embeddings from %s to %s',
                         posemb_restored.shape, posemb_init.shape)

            # Determine if the model uses a class token based on shape difference
            # Assumes num_tokens = H * W (+ 1 if class token exists)
            has_class_token_restored = posemb_restored.shape[1] % int(np.sqrt(posemb_restored.shape[1])) != 0
            if not has_class_token_restored and posemb_restored.shape[1]-1 > 0 and (posemb_restored.shape[1]-1) > 0 and int(np.sqrt(posemb_restored.shape[1]-1)) > 0 and (posemb_restored.shape[1]-1) % int(np.sqrt(posemb_restored.shape[1]-1)) == 0:
                 # Alternative check if sqrt logic fails for non-square grids - check if removing 1 makes it match expected grid
                 has_class_token_restored = True # Heuristic guess

            # Interpolate the restored positional embedding to match the shape of the initial model
            posemb_new = interpolate_posembed(
                posemb_restored,
                posemb_init.shape[1],  # Target number of tokens
                has_class_token=has_class_token_restored # Pass whether the *restored* embedding had a class token
            )

            # Check if the new shape matches the target shape
            if posemb_new.shape != posemb_init.shape:
                raise ValueError(f'Interpolated positional embedding shape mismatch: '
                                 f'Expected {posemb_init.shape}, Got {posemb_new.shape}. '
                                 f'Check class token logic.')


            # Update the restored parameters with the new positional embedding
            restored_params['Transformer']['posembed_input']['pos_embedding'] = posemb_new
        else:
            logging.info('load_pretrained: positional embeddings shapes match, no resize needed.')
    # --- End of Positional Embedding Interpolation ---


    # GroupNorm fix (remains the same)
    if version.parse(flax.__version__) >= version.parse('0.3.6'):
        restored_params = _fix_groupnorm(restored_params)

    return flax.core.freeze(restored_params)


# --- MODIFIED interpolate_posembed FUNCTION ---
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
        num_tokens -= 1 # Adjust target token count if class token exists
    else:
        posemb_tok, posemb_grid = posemb[:, :0], posemb[0, 0:]

    # Check if posemb_grid is empty
    if posemb_grid.shape[0] == 0:
        if num_tokens == 0: # Only class token exists
             if not has_class_token:
                  raise ValueError("Cannot interpolate empty grid without class token")
             return posemb_tok # Return only the class token
        else:
             raise ValueError("Cannot interpolate empty grid to non-empty target")


    # Calculate grid size, handle potential floating point issues with sqrt
    grid_len = len(posemb_grid)
    gs_old_float = np.sqrt(grid_len)
    gs_old = int(gs_old_float)
    if gs_old_float % 1 != 0: # Check if it wasn't a perfect square
        raise ValueError(f"Original positional embedding grid size ({grid_len}) is not a perfect square.")

    gs_new_float = np.sqrt(num_tokens)
    gs_new = int(gs_new_float)
    logging.info('interpolate_posembed: grid-size from %s to %s', gs_old, gs_new)

    # Ensure gs_new calculation is valid
    if gs_new_float % 1 != 0:
        raise ValueError(f"Target number of grid tokens ({num_tokens}) is not a perfect square.")
    if gs_new == 0 and num_tokens > 0 :
        raise ValueError(f"Calculated gs_new is 0 but num_tokens is {num_tokens}. Check input.")

    posemb_grid = posemb_grid.reshape(gs_old, gs_old, -1)

    zoom = (gs_new / gs_old, gs_new / gs_old, 1)
    posemb_grid = scipy.ndimage.zoom(posemb_grid, zoom, order=1)
    posemb_grid = posemb_grid.reshape(1, gs_new * gs_new, -1)

    # Concatenate the class token (if it exists) back with the resized grid
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