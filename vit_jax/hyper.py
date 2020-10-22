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

import flax
import jax
import numpy as np


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
    progress = np.clip(progress, 0.0, 1.0)
    if decay_type == 'linear':
      lr = linear_end + (lr - linear_end) * (1.0 - progress)
    elif decay_type == 'cosine':
      lr = lr * 0.5 * (1. + np.cos(np.pi * progress))
    else:
      raise ValueError(f'Unknown lr type {decay_type}')

    if warmup_steps:
      lr = lr * np.minimum(1., step / warmup_steps)

    return np.asarray(lr, dtype=np.float32)

  return step_fn


def lr_prefetch_iter(lr_fn,
                     first_step,
                     total_steps,
                     prefetch_to_device=2,
                     devices=None):
  local_device_count = (
      jax.local_device_count() if devices is None else len(devices))
  lr_iter = (
      np.ones([local_device_count]) * lr_fn(i)
      for i in range(first_step, total_steps))
  # Prefetching learning rate eliminates significant TPU transfer overhead.
  return flax.jax_utils.prefetch_to_device(
      lr_iter, prefetch_to_device, devices=devices)


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
      return (l + li, jax.tree_multimap(lambda x, y: x + y, g, gi))

    l, g = jax.lax.fori_loop(1, accum_steps, acc_grad_and_loss, (l, g))
    return jax.tree_map(lambda x: x / accum_steps, (l, g))
  else:
    return loss_and_grad_fn(params, images, labels)
