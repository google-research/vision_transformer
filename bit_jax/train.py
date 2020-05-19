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

import os

from functools import partial

import numpy as np

import jax
import jax.numpy as jnp

import flax.optim as optim
import flax.jax_utils as flax_utils

import input_pipeline_tf2_or_jax as input_pipeline
import bit_jax.models as models
import bit_jax.tf2jax as tf2jax

import bit_common
import bit_hyperrule


def main(args):
  logger = bit_common.setup_logger(args)

  logger.info(f'Available devices: {jax.devices()}')

  model = models.KNOWN_MODELS[args.model]

  # Load weigths of a BiT model
  bit_model_file = os.path.join(args.bit_pretrained_dir, f'{args.model}.npz')
  if not os.path.exists(bit_model_file):
    raise FileNotFoundError(
      f'Model file is not found in "{args.bit_pretrained_dir}" directory.')
  with open(bit_model_file, 'rb') as f:
    params_tf = np.load(f)
    params_tf = dict(zip(params_tf.keys(), params_tf.values()))

  resize_size, crop_size = bit_hyperrule.get_resolution_from_dataset(
    args.dataset)

  # Setup input pipeline
  dataset_info = input_pipeline.get_dataset_info(
    args.dataset, 'train', args.examples_per_class)

  data_train = input_pipeline.get_data(
    dataset=args.dataset,
    mode='train',
    repeats=None, batch_size=args.batch,
    resize_size=resize_size, crop_size=crop_size,
    examples_per_class=args.examples_per_class,
    examples_per_class_seed=args.examples_per_class_seed,
    mixup_alpha=bit_hyperrule.get_mixup(dataset_info['num_examples']),
    num_devices=jax.local_device_count(),
    tfds_manual_dir=args.tfds_manual_dir)
  logger.info(data_train)
  data_test = input_pipeline.get_data(
    dataset=args.dataset,
    mode='test',
    repeats=1, batch_size=args.batch_eval,
    resize_size=resize_size, crop_size=crop_size,
    examples_per_class=None, examples_per_class_seed=0,
    mixup_alpha=None,
    num_devices=jax.local_device_count(),
    tfds_manual_dir=args.tfds_manual_dir)
  logger.info(data_test)

  # Build ResNet architecture
  ResNet = model.partial(num_classes=dataset_info['num_classes'])
  _, params = ResNet.init_by_shape(
    jax.random.PRNGKey(0),
    [([1, crop_size, crop_size, 3], jnp.float32)])
  resnet_fn = ResNet.call

  # pmap replicates the models over all GPUs
  resnet_fn_repl = jax.pmap(ResNet.call)

  def cross_entropy_loss(*, logits, labels):
    logp = jax.nn.log_softmax(logits)
    return -jnp.mean(jnp.sum(logp * labels, axis=1))

  def loss_fn(params, images, labels):
    logits = resnet_fn(params, images)
    return cross_entropy_loss(logits=logits, labels=labels)

  # Update step, replicated over all GPUs
  @partial(jax.pmap, axis_name='batch')
  def update_fn(opt, lr, batch):
    l, g = jax.value_and_grad(loss_fn)(opt.target,
                                       batch['image'],
                                       batch['label'])
    g = jax.tree_map(lambda x: jax.lax.pmean(x, axis_name='batch'), g)
    opt = opt.apply_gradient(g, learning_rate=lr)
    return opt

  # In-place update of randomly initialized weights by BiT weigths
  tf2jax.transform_params(params, params_tf,
                          num_classes=dataset_info['num_classes'])

  # Create optimizer and replicate it over all GPUs
  opt = optim.Momentum(beta=0.9).create(params)
  opt_repl = flax_utils.replicate(opt)

  # Delete referenes to the objects that are not needed anymore
  del opt
  del params

  total_steps = bit_hyperrule.get_schedule(dataset_info['num_examples'])[-1]

  # Run training loop
  for step, batch in zip(range(1, total_steps + 1),
                         data_train.as_numpy_iterator()):
    lr = bit_hyperrule.get_lr(step - 1,
                              dataset_info['num_examples'],
                              args.base_lr)
    opt_repl = update_fn(opt_repl, flax_utils.replicate(lr), batch)

    # Run eval step
    if ((args.eval_every and step % args.eval_every == 0)
         or (step == total_steps)):

      accuracy_test = np.mean([
        c
        for batch in data_test.as_numpy_iterator()
        for c in (
          np.argmax(resnet_fn_repl(opt_repl.target, batch['image']), axis=2) ==
          np.argmax(batch['label'], axis=2)).ravel()])

      logger.info(
              f'Step: {step}, '
              f'learning rate: {lr:.07f}, '
              f'Test accuracy: {accuracy_test:0.3f}')


if __name__ == "__main__":
  parser = bit_common.argparser(models.KNOWN_MODELS.keys())
  parser.add_argument("--tfds_manual_dir", default=None,
                      help="Path to maually downloaded dataset.")
  parser.add_argument("--batch_eval", default=32, type=int,
                      help="Eval batch size.")
  main(parser.parse_args())
