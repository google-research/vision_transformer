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

import functools
import os
import time

from absl import logging
from clu import metric_writers
import flax
import jax
import jax.numpy as jnp
import ml_collections
import numpy as np
import tensorflow as tf

from vit_jax import checkpoint
from vit_jax import input_pipeline
from vit_jax import models
from vit_jax import momentum_clip
from vit_jax import utils


def make_update_fn(*, apply_fn, accum_steps, lr_fn):
    """Returns update step for data parallel training."""

    def update_fn(opt, step, batch, rng):
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
            jax.value_and_grad(loss_fn), opt.target, batch['image'], batch['label'],
            accum_steps)
        g = jax.tree_map(lambda x: jax.lax.pmean(x, axis_name='batch'), g)
        l = jax.lax.pmean(l, axis_name='batch')

        opt = opt.apply_gradient(g, learning_rate=lr_fn(step))
        return opt, l, new_rng

    return jax.pmap(update_fn, axis_name='batch', donate_argnums=(0,))


def train_and_evaluate(config: ml_collections.ConfigDict, workdir: str):
    """Runs training interleaved with evaluation."""

    # Setup input pipeline
    dataset_info = input_pipeline.get_dataset_info(config.dataset, 'train')

    ds_train = input_pipeline.get_data(
        dataset=config.dataset,
        mode='train',
        repeats=None,
        mixup_alpha=config.mixup_alpha,
        batch_size=config.batch,
        pp_config=config.pp,
        shuffle_buffer=config.shuffle_buffer,
        tfds_data_dir=config.tfds_data_dir,
        tfds_manual_dir=config.tfds_manual_dir)
    batch = next(iter(ds_train))
    logging.info(ds_train)
    ds_test = input_pipeline.get_data(
        dataset=config.dataset,
        mode='test',
        repeats=1,
        batch_size=config.batch_eval,
        pp_config=config.pp,
        tfds_data_dir=config.tfds_data_dir,
        tfds_manual_dir=config.tfds_manual_dir)
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

    # pretrained_path = os.path.join(config.pretrained_dir,
    #                               f'{config.model.name}.npz')
    # if not tf.io.gfile.exists(pretrained_path):
    #  raise ValueError(
    #      f'Could not find "{pretrained_path}" - you can download models from '
    #      '"gs://vit_models/imagenet21k" or directly set '
    #      '--config.pretrained_dir="gs://vit_models/imagenet21k".')
    # params = checkpoint.load_pretrained(
    #    pretrained_path=pretrained_path,
    #    init_params=variables['params'],
    #    model_config=config.model)
    params = variables['params']
    total_steps = config.total_steps
    lr_fn = utils.create_learning_rate_schedule(total_steps, config.base_lr,
                                                config.decay_type,
                                                config.warmup_steps)

    update_fn_repl = make_update_fn(
        apply_fn=model.apply, accum_steps=config.accum_steps, lr_fn=lr_fn)
    infer_fn_repl = jax.pmap(functools.partial(model.apply, train=False))

    # Create optimizer and replicate it over all TPUs/GPUs
    opt = momentum_clip.Optimizer(
        dtype=config.optim_dtype,
        grad_norm_clip=config.grad_norm_clip).create(params)
    opt_repl = flax.jax_utils.replicate(opt)

    # Delete references to the objects that are not needed anymore
    del opt
    del params

    # Prepare the learning-rate and pre-fetch it to device to avoid delays.
    update_rng_repl = flax.jax_utils.replicate(jax.random.PRNGKey(0))

    # Run training loop
    writer = metric_writers.create_default_writer(workdir, asynchronous=False)
    writer.write_hparams(config.to_dict())
    logging.info('Starting training loop; initial compile can take a while...')
    t0 = lt0 = time.time()

    for step, batch in zip(
            range(1, total_steps + 1),
            input_pipeline.prefetch(ds_train, config.prefetch)):

        opt_repl, loss_repl, update_rng_repl = update_fn_repl(
            opt_repl, flax.jax_utils.replicate(step), batch, update_rng_repl)

        if step == 1:
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
            logging.info(
                f'Step: {step}/{total_steps} {100 * done:.1f}%, '  # pylint: disable=logging-format-interpolation
                f'img/sec/core: {img_sec_core_train:.1f}, '
                f'ETA: {(time.time() - t0) / done * (1 - done) / 3600:.2f}h')

        # Run evaluation
        if ((config.eval_every and step % config.eval_every == 0) or
                (step == total_steps)):

            accuracies = []
            lt0 = time.time()
            for test_batch in input_pipeline.prefetch(ds_test, config.prefetch):
                logits = infer_fn_repl(
                    dict(params=opt_repl.target), test_batch['image'])
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
            logging.info(f'Step: {step} '  # pylint: disable=logging-format-interpolation
                         f'Learning rate: {lr:.7f}, '
                         f'Test accuracy: {accuracy_test:0.5f}, '
                         f'img/sec/core: {img_sec_core_test:.1f}')
            writer.write_scalars(
                step,
                dict(
                    accuracy_test=accuracy_test,
                    lr=lr,
                    img_sec_core_test=img_sec_core_test))

    opt = flax.jax_utils.unreplicate(opt_repl)
    del opt_repl
    checkpoint.save(opt.target, f'{workdir}/model.npz')
    logging.info('Stored fine tuned checkpoint to %s', workdir)

    return opt
