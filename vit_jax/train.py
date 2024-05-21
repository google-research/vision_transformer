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
from clu import periodic_actions
import flax
from flax.training import checkpoints as flax_checkpoints
import jax
import jax.numpy as jnp
import ml_collections
import numpy as np
import optax
import tensorflow as tf

from vit_jax import checkpoint
from vit_jax import input_pipeline
from vit_jax import models
from vit_jax import utils


def make_update_fn(*, apply_fn, accum_steps, tx):
  """Returns update step for data parallel training."""

  def update_fn(params, opt_state, batch, rng):

    _, new_rng = jax.random.split(rng)
    # Bind the rng key to the device id (which is unique across hosts)
    # Note: This is only used for multi-host training (i.e. multiple computers
    # each with multiple accelerators).
    dropout_rng = jax.random.fold_in(rng, jax.lax.axis_index('batch'))

    def cross_entropy_loss(*, logits, labels):
      logp = jax.nn.log_softmax(logits)
      return -jnp.mean(jnp.sum(logp * labels, axis=1))

    def loss_fn(params, images, labels):
      logits = apply_fn(
          dict(params=params),
          rngs=dict(dropout=dropout_rng),
          inputs=images,
          train=True)
      return cross_entropy_loss(logits=logits, labels=labels)

    l, g = utils.accumulate_gradient(
        jax.value_and_grad(loss_fn), params, batch['image'], batch['label'],
        accum_steps)
    g = jax.tree.map(lambda x: jax.lax.pmean(x, axis_name='batch'), g)
    updates, opt_state = tx.update(g, opt_state)
    params = optax.apply_updates(params, updates)
    l = jax.lax.pmean(l, axis_name='batch')

    return params, opt_state, l, new_rng

  return jax.pmap(update_fn, axis_name='batch', donate_argnums=(0,))


def train_and_evaluate(config: ml_collections.ConfigDict, workdir: str):
  """Runs training interleaved with evaluation."""

  # Setup input pipeline
  dataset_info = input_pipeline.get_dataset_info(config.dataset, 'train')

  ds_train, ds_test = input_pipeline.get_datasets(config)
  batch = next(iter(ds_train))
  logging.info(ds_train)
  logging.info(ds_test)

  # Build VisionTransformer architecture
  model_cls = {'ViT': models.VisionTransformer,
               'Mixer': models.MlpMixer}[config.get('model_type', 'ViT')]
  model = model_cls(num_classes=dataset_info['num_classes'], **config.model)

  def init_model():
    return model.init(
        jax.random.PRNGKey(0),
        # Discard the "num_local_devices" dimension for initialization.
        jnp.ones(batch['image'].shape[1:], batch['image'].dtype.name),
        train=False)

  # Use JIT to make sure params reside in CPU memory.
  variables = jax.jit(init_model, backend='cpu')()

  model_or_filename = config.get('model_or_filename')
  if model_or_filename:
    # Loading model from repo published with  "How to train your ViT? Data,
    # Augmentation, and Regularization in Vision Transformers" paper.
    # https://arxiv.org/abs/2106.10270
    if '-' in model_or_filename:
      filename = model_or_filename
    else:
      # Select best checkpoint from i21k pretraining by final upstream
      # validation accuracy.
      df = checkpoint.get_augreg_df(directory=config.pretrained_dir)
      sel = df.filename.apply(
          lambda filename: filename.split('-')[0] == model_or_filename)
      best = df.loc[sel].query('ds=="i21k"').sort_values('final_val').iloc[-1]
      filename = best.filename
      logging.info('Selected fillename="%s" for "%s" with final_val=%.3f',
                   filename, model_or_filename, best.final_val)
    pretrained_path = os.path.join(config.pretrained_dir,
                                   f'{config.model.model_name}.npz')
  else:
    # ViT / Mixer papers
    filename = config.model.model_name

  pretrained_path = os.path.join(config.pretrained_dir, f'{filename}.npz')
  if not tf.io.gfile.exists(pretrained_path):
    raise ValueError(
        f'Could not find "{pretrained_path}" - you can download models from '
        '"gs://vit_models/imagenet21k" or directly set '
        '--config.pretrained_dir="gs://vit_models/imagenet21k".')
  params = checkpoint.load_pretrained(
      pretrained_path=pretrained_path,
      init_params=variables['params'],
      model_config=config.model)

  total_steps = config.total_steps

  lr_fn = utils.create_learning_rate_schedule(total_steps, config.base_lr,
                                              config.decay_type,
                                              config.warmup_steps)
  tx = optax.chain(
      optax.clip_by_global_norm(config.grad_norm_clip),
      optax.sgd(
          learning_rate=lr_fn,
          momentum=0.9,
          accumulator_dtype='bfloat16',
      ),
  )

  update_fn_repl = make_update_fn(
      apply_fn=model.apply, accum_steps=config.accum_steps, tx=tx)
  infer_fn_repl = jax.pmap(functools.partial(model.apply, train=False))

  initial_step = 1
  opt_state = tx.init(params)
  params, opt_state, initial_step = flax_checkpoints.restore_checkpoint(
      workdir, (params, opt_state, initial_step))
  logging.info('Will start/continue training at initial_step=%d', initial_step)

  params_repl, opt_state_repl = flax.jax_utils.replicate((params, opt_state))

  # Delete references to the objects that are not needed anymore
  del opt_state
  del params

  # Prepare the learning-rate and pre-fetch it to device to avoid delays.
  update_rng_repl = flax.jax_utils.replicate(jax.random.PRNGKey(0))

  # Setup metric writer & hooks.
  writer = metric_writers.create_default_writer(workdir, asynchronous=False)
  writer.write_hparams(config.to_dict())
  hooks = [
      periodic_actions.Profile(logdir=workdir),
      periodic_actions.ReportProgress(
          num_train_steps=total_steps, writer=writer),
  ]

  # Run training loop
  logging.info('Starting training loop; initial compile can take a while...')
  t0 = lt0 = time.time()
  lstep = initial_step
  for step, batch in zip(
      range(initial_step, total_steps + 1),
      input_pipeline.prefetch(ds_train, config.prefetch)):

    with jax.profiler.StepTraceAnnotation('train', step_num=step):
      params_repl, opt_state_repl, loss_repl, update_rng_repl = update_fn_repl(
          params_repl, opt_state_repl, batch, update_rng_repl)

    for hook in hooks:
      hook(step)

    if step == initial_step:
      logging.info('First step took %.1f seconds.', time.time() - t0)
      t0 = time.time()
      lt0, lstep = time.time(), step

    # Report training metrics
    if config.progress_every and step % config.progress_every == 0:
      img_sec_core_train = (config.batch * (step - lstep) /
                            (time.time() - lt0)) / jax.device_count()
      lt0, lstep = time.time(), step
      writer.write_scalars(
          step,
          dict(
              train_loss=float(flax.jax_utils.unreplicate(loss_repl)),
              img_sec_core_train=img_sec_core_train))
      done = step / total_steps
      logging.info(f'Step: {step}/{total_steps} {100*done:.1f}%, '  # pylint: disable=logging-fstring-interpolation
                   f'img/sec/core: {img_sec_core_train:.1f}, '
                   f'ETA: {(time.time()-t0)/done*(1-done)/3600:.2f}h')

    # Run evaluation
    if ((config.eval_every and step % config.eval_every == 0) or
        (step == total_steps)):

      accuracies = []
      lt0 = time.time()
      for test_batch in input_pipeline.prefetch(ds_test, config.prefetch):
        logits = infer_fn_repl(
            dict(params=params_repl), test_batch['image'])
        accuracies.append(
            (np.argmax(logits,
                       axis=-1) == np.argmax(test_batch['label'],
                                             axis=-1)).mean())
      accuracy_test = np.mean(accuracies)
      img_sec_core_test = (
          config.batch_eval * ds_test.cardinality().numpy() /
          (time.time() - lt0) / jax.device_count())
      lt0 = time.time()

      lr = float(lr_fn(step))
      logging.info(f'Step: {step} '  # pylint: disable=logging-fstring-interpolation
                   f'Learning rate: {lr:.7f}, '
                   f'Test accuracy: {accuracy_test:0.5f}, '
                   f'img/sec/core: {img_sec_core_test:.1f}')
      writer.write_scalars(
          step,
          dict(
              accuracy_test=accuracy_test,
              lr=lr,
              img_sec_core_test=img_sec_core_test))

    # Store checkpoint.
    if ((config.checkpoint_every and step % config.eval_every == 0) or
        step == total_steps):
      checkpoint_path = flax_checkpoints.save_checkpoint(
          workdir, (flax.jax_utils.unreplicate(params_repl),
                    flax.jax_utils.unreplicate(opt_state_repl), step), step)
      logging.info('Stored checkpoint at step %d to "%s"', step,
                   checkpoint_path)

  return flax.jax_utils.unreplicate(params_repl)
