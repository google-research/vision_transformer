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

import functools
import os
import time

from absl import logging
from clu import metric_writers
import flax
import flax.jax_utils as flax_utils
import jax
import jax.numpy as jnp
import ml_collections
import numpy as np
import tensorflow as tf

from vit_jax import checkpoint
from vit_jax import models
from vit_jax.configs import models as config_lib


def inference_time(config: ml_collections.ConfigDict, workdir: str):
  """Runs a number of steps and measures inference time."""

  assert config.batch, f'Expected --config.batch={config.batch} > 0'
  assert config.num_classes, (
      f'Expected --config.num_classes={config.num_classes} > 0')
  assert config.image_size, (
      f'Expected --config.image_size={config.image_size} > 0')

  # Build VisionTransformer architecture
  model_config = config_lib.MODEL_CONFIGS[config.model_name]
  model = models.VisionTransformer(
      num_classes=config.num_classes, **model_config)

  # Make sure initial model parameters (before replication) are on CPU only.
  @functools.partial(jax.jit, backend='cpu')
  def init(rng):
    return model.init(
        rng,
        # Discard the "num_local_devices" dimension for initialization.
        inputs=jnp.ones([1, config.image_size, config.image_size, 3],
                        jnp.float32),
        train=False)

  variables = init(jax.random.PRNGKey(0))

  params_repl = flax_utils.replicate(variables['params'])

  # pmap replicates the models over all TPUs/GPUs
  vit_fn_repl = jax.pmap(functools.partial(model.apply, train=False))
  images = jnp.ones([
      jax.local_device_count(), config.batch // jax.local_device_count(),
      config.image_size, config.image_size, 3
  ], jnp.float32)

  writer = metric_writers.create_default_writer(workdir, asynchronous=False)
  writer.write_hparams(config.to_dict())

  logging.info('Starting training loop; initial compile can take a while...')
  logits = vit_fn_repl(flax.core.FrozenDict(params=params_repl), images)
  logits.block_until_ready()
  logging.info('Done.')

  logging.info('Going to run %d inferences WITHOUT measuring...',
               config.initial_steps)
  for _ in range(config.initial_steps):
    logits = vit_fn_repl(flax.core.FrozenDict(params=params_repl), images)
    logits.block_until_ready()

  logging.info('Going to run %d inferences measuring...', config.steps)
  times = []
  for _ in range(config.initial_steps):
    t0 = time.time()
    logits = vit_fn_repl(flax.core.FrozenDict(params=params_repl), images)
    logits.block_until_ready()
    times.append(time.time() - t0)
  logging.info('times=%s', times)
  imgs_sec_core = config.batch / jax.local_device_count() / np.array(times)
  logging.info('imgs_sec_core_min=%f', imgs_sec_core.min())
  logging.info('imgs_sec_core_max=%f', imgs_sec_core.max())
  logging.info('imgs_sec_core_mean=%f', imgs_sec_core.mean())
  logging.info('imgs_sec_core_std=%f', imgs_sec_core.std())
  writer.write_scalars(
      0,
      dict(
          imgs_sec_core_min=imgs_sec_core.min(),
          imgs_sec_core_max=imgs_sec_core.max(),
          imgs_sec_core_mean=imgs_sec_core.mean(),
          imgs_sec_core_std=imgs_sec_core.std(),
      ))
