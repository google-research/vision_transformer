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

import logging as python_logging
import os
import threading

from absl import logging
import jax
import jax.numpy as jnp
import tensorflow as tf


class GFileHandler(python_logging.StreamHandler):
  """Writes log messages to file using tf.io.gfile."""

  def __init__(self, filename, mode, flush_secs=1.0):
    super().__init__()
    tf.io.gfile.makedirs(os.path.dirname(filename))
    if mode == 'a' and not tf.io.gfile.exists(filename):
      mode = 'w'
    self.filehandle = tf.io.gfile.GFile(filename, mode)
    self.flush_secs = flush_secs
    self.flush_timer = None

  def flush(self):
    self.filehandle.flush()

  def emit(self, record):
    msg = self.format(record)
    self.filehandle.write(f'{msg}\n')
    if self.flush_timer is not None:
      self.flush_timer.cancel()
    self.flush_timer = threading.Timer(self.flush_secs, self.flush)
    self.flush_timer.start()


def add_gfile_logger(workdir, *, basename='train', level=python_logging.INFO):
  """Adds GFile file logger to Python logging handlers."""
  fh = GFileHandler(f'{workdir}/{basename}.log', 'a')
  fh.setLevel(level)
  fh.setFormatter(logging.PythonFormatter())
  python_logging.getLogger('').addHandler(fh)


def create_learning_rate_schedule(total_steps,
                                  base,
                                  decay_type,
                                  warmup_steps,
                                  linear_end=1e-5):
  """Creates learning rate schedule.

  Currently only warmup + {linear,cosine} but will be a proper mini-language
  like preprocessing one in the future.

  Args:
    total_steps: The total number of steps to run.
    base: The starting learning-rate (without warmup).
    decay_type: 'linear' or 'cosine'.
    warmup_steps: how many steps to warm up for.
    linear_end: Minimum learning rate.

  Returns:
    A function learning_rate(step): float -> {"learning_rate": float}.
  """

  def step_fn(step):
    """Step to learning rate function."""
    lr = base

    progress = (step - warmup_steps) / float(total_steps - warmup_steps)
    progress = jnp.clip(progress, 0.0, 1.0)
    if decay_type == 'linear':
      lr = linear_end + (lr - linear_end) * (1.0 - progress)
    elif decay_type == 'cosine':
      lr = lr * 0.5 * (1. + jnp.cos(jnp.pi * progress))
    else:
      raise ValueError(f'Unknown lr type {decay_type}')

    if warmup_steps:
      lr = lr * jnp.minimum(1., step / warmup_steps)

    return jnp.asarray(lr, dtype=jnp.float32)

  return step_fn


def accumulate_gradient(loss_and_grad_fn, params, images, labels, accum_steps):
  """Accumulate gradient over multiple steps to save on memory."""
  if accum_steps and accum_steps > 1:
    assert images.shape[0] % accum_steps == 0, (
        f'Bad accum_steps {accum_steps} for batch size {images.shape[0]}')
    step_size = images.shape[0] // accum_steps
    l, g = loss_and_grad_fn(params, images[:step_size], labels[:step_size])

    def acc_grad_and_loss(i, l_and_g):
      imgs = jax.lax.dynamic_slice(images, (i * step_size, 0, 0, 0),
                                   (step_size,) + images.shape[1:])
      lbls = jax.lax.dynamic_slice(labels, (i * step_size, 0),
                                   (step_size, labels.shape[1]))
      li, gi = loss_and_grad_fn(params, imgs, lbls)
      l, g = l_and_g
      return (l + li, jax.tree.map(lambda x, y: x + y, g, gi))

    l, g = jax.lax.fori_loop(1, accum_steps, acc_grad_and_loss, (l, g))
    return jax.tree.map(lambda x: x / accum_steps, (l, g))
  else:
    return loss_and_grad_fn(params, images, labels)
